#!/usr/bin/env python
"""
Empirical validation of Proposition 2 (volume preservation via temporal sparsity).

For each model M in {ref, iadd, full}, we run the standard DDIM reverse loop
(T steps, CFG-guided) and at every step t estimate the divergence of the
guided noise predictor with Hutchinson's trace estimator:

    div(eps_tilde_t)(x_t, c, t)  =  tr( d eps_tilde / d x_t )
                                  ~= 1/K * sum_k  v_k^T J v_k,   v_k ~ Rademacher

By Liouville, the local DDIM-step log-determinant is proportional to delta_t
times this divergence, and the global log|det J_{T->0}| is the sum:

    log|det J_{T->0}|  ~  sum_t  delta_t * div(eps_tilde_t)

Outputs one row per (model, prompt_id, seed, step_idx). Cumulative log-det
is computed downstream from the parquet.

Run (single GPU, inference only):

    python scripts/diagnostics/volume_preservation.py \
        --base_model CompVis/stable-diffusion-v1-4 \
        --prompt_file configs/prompt/template1_test.json \
        --output_dir rebuttal/artifacts/volume_preservation \
        --num_inference_steps 20 \
        --guidance_scale 5.0 \
        --num_probes 8 \
        --num_prompts 32 \
        --seeds_per_prompt 2

Active set S for the "iadd" model is the set of timestep indices in
np.linspace(0, T-1, |S|).round() (matches train_pipeline.py incremental rule).
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Project root on path
_here = os.path.abspath(__file__)
_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
sys.path.insert(0, _root)


# --------------------------------------------------------------------------
# Model loading (mirrors scripts/diversity_eval/measure_diversity_spread.py)
# --------------------------------------------------------------------------

def load_pipeline_base(base_model: str, device, dtype=torch.float32):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.safety_checker = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.to(device, dtype=dtype)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.unet.to(device, dtype=dtype)
    pipe.unet.eval()
    return pipe


def load_lora_attnprocs(pipe, checkpoint_path: str, base_model: str):
    """Load LoRA weights and fuse them into the base unet's Linear modules.

    Fusion is required because diffusers' LoRAAttnProcessor implementation
    is not forward-mode-AD friendly (segfaults under torch.func.jvp). After
    fuse_lora() the forward path uses plain nn.Linear with the LoRA delta
    baked in, which is JVP-clean."""
    weights_file = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(weights_file):
        raise FileNotFoundError(weights_file)
    # diffusers expects either a dir or a single-file path
    pipe.load_lora_weights(checkpoint_path)
    pipe.fuse_lora()
    # After fusing we can unload the (now-redundant) LoRA modules to keep
    # the forward path identical to a plain unet.
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        print(f"[warn] unload_lora_weights failed (non-fatal): {e}")
    pipe.unet.eval()
    return pipe


def load_model(model_name: str, lora_path: str | None, base_model: str, device):
    pipe = load_pipeline_base(base_model, device)
    if lora_path is None:
        print(f"[{model_name}] pretrained, no LoRA")
    else:
        load_lora_attnprocs(pipe, lora_path, base_model)
        print(f"[{model_name}] LoRA loaded from {lora_path}")
    return pipe


# --------------------------------------------------------------------------
# Hutchinson divergence estimator
# --------------------------------------------------------------------------

@torch.no_grad()
def sample_rademacher(shape, device, dtype):
    v = torch.randint(0, 2, shape, device=device, dtype=torch.int8).to(dtype)
    return v.mul_(2).sub_(1)  # in {-1, +1}


def make_guided_eps_fn(unet, combined_embed, t, guidance_scale: float):
    """Returns f(x) -> guided eps_tilde at (x, t, c). combined_embed is
    [uncond_embed; cond_embed] of shape (2, seq, d)."""
    def f(x):
        inp = torch.cat([x, x], dim=0)
        out = unet(inp, t, encoder_hidden_states=combined_embed, return_dict=False)[0]
        uncond, cond = out.chunk(2)
        return uncond + guidance_scale * (cond - uncond)
    return f


def hutchinson_logdet_step(f, x, a_t: float, b_t: float, num_probes: int,
                            num_terms: int = 3):
    """Estimate log|det J_t| for a single DDIM reverse step.

    The DDIM step Jacobian is
        J_t  =  a_t * I  +  b_t * M           with  M = d eps_tilde / d x_t.
    Therefore
        log|det J_t|  =  D * log|a_t|  +  log|det( I + c * M )|,   c = b_t/a_t.

    The second term is computed via the Taylor series of log(I + cM):
        log|det(I + cM)|  =  sum_{k>=1}  (-1)^(k+1) / k  *  c^k * tr(M^k)
    truncated at `num_terms`. Each tr(M^k) is Hutchinson-estimated with
    repeated JVPs (M v, M^2 v, ..., M^k v computed by iterating M·).

    Returns dict with the cumulative log|det J_t| (per-sample tensor) and the
    individual Taylor terms for diagnostics.
    """
    from torch.func import jvp
    from torch.nn.attention import sdpa_kernel, SDPBackend

    D = int(np.prod(x.shape[1:]))  # per-sample dim (CHW)
    c = float(b_t) / float(a_t)

    # term_k will hold per-probe estimates of  c^k * tr(M^k)
    term_estimates = [[] for _ in range(num_terms)]

    with sdpa_kernel([SDPBackend.MATH]):
        for _ in range(num_probes):
            v = sample_rademacher(x.shape, x.device, x.dtype)
            w = v
            for k in range(1, num_terms + 1):
                # w  <-  M w   (one JVP)
                _, w = jvp(f, (x,), (w,))
                # Hutchinson trace of M^k:   tr(M^k) ~  v^T (M^k v)
                est = (v.float() * w.float()).flatten(1).sum(dim=1)  # (B,)
                term_estimates[k - 1].append((c ** k) * est)

    # Average over probes; combine with alternating-sign Taylor coefficients
    log_det_minus_const = torch.zeros_like(term_estimates[0][0])
    term_means = []
    term_stds = []
    for k in range(1, num_terms + 1):
        stack = torch.stack(term_estimates[k - 1], dim=0)   # (K, B)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0) if num_probes > 1 else torch.zeros_like(mean)
        sign = (-1) ** (k + 1)
        log_det_minus_const = log_det_minus_const + sign * mean / k
        term_means.append(mean)
        term_stds.append(std)

    log_det_step = D * float(np.log(abs(a_t))) + log_det_minus_const
    return {
        "log_det": log_det_step,        # (B,) tensor
        "term_means": term_means,       # list of (B,) tensors, each c^k * tr(M^k)
        "term_stds": term_stds,
        "a_t": float(a_t),
        "b_t": float(b_t),
        "c": c,
        "D": D,
    }


def get_ab_for_step(scheduler, t_now, t_prev):
    """Return DDIM step coefficients (a_t, b_t) for eta=0 deterministic step:
        x_{t-1} = a_t * x_t + b_t * eps_tilde
    where
        a_t = sqrt(alpha_bar_{t-1} / alpha_bar_t)
        b_t = sqrt(1 - alpha_bar_{t-1}) - sqrt(alpha_bar_{t-1}/alpha_bar_t * (1 - alpha_bar_t))

    For the final step, t_prev is None and we use alpha_bar_{-1} = 1
    (the convention diffusers uses)."""
    ab = scheduler.alphas_cumprod
    a_t_cum = float(ab[int(t_now)].item())
    if t_prev is None or int(t_prev) < 0:
        a_prev_cum = 1.0       # diffusers convention for final step
    else:
        a_prev_cum = float(ab[int(t_prev)].item())
    a_t = (a_prev_cum / a_t_cum) ** 0.5
    b_t = (1.0 - a_prev_cum) ** 0.5 - (a_prev_cum / a_t_cum * (1.0 - a_t_cum)) ** 0.5
    return a_t, b_t


# --------------------------------------------------------------------------
# Per-model trajectory + divergence
# --------------------------------------------------------------------------

@dataclass
class Row:
    model: str
    prompt_id: int
    prompt: str
    seed: int
    step_idx: int
    t: int
    t_prev: int
    a_t: float
    b_t: float
    c_t: float
    D: int
    # Per-step log|det J_t| (single number per (prompt, seed, step) -- Hutchinson + Taylor)
    log_det_step: float
    # Taylor terms (signed, per definition above) for diagnostics
    term1: float                 # +c * tr(M)
    term2: float                 # -c^2/2 * tr(M^2)
    term3: float                 # +c^3/3 * tr(M^3)
    term1_se: float
    term2_se: float
    term3_se: float


def encode_prompts(pipe, prompts, device):
    """Return per-prompt cond embedding (B=1) and a shared uncond embedding."""
    tok = pipe.tokenizer
    enc = pipe.text_encoder
    max_len = tok.model_max_length
    neg_ids = tok([""], return_tensors="pt", padding="max_length", truncation=True,
                  max_length=max_len).input_ids.to(device)
    with torch.no_grad():
        uncond = enc(neg_ids)[0]
    cond_list = []
    for p in prompts:
        ids = tok([p], return_tensors="pt", padding="max_length", truncation=True,
                  max_length=max_len).input_ids.to(device)
        with torch.no_grad():
            cond_list.append(enc(ids)[0])
    return uncond, cond_list


def get_delta_t_seq(scheduler, timesteps):
    """delta_t per step in the DDIM reverse pass. Equal in indices: 1 step each
    in the discrete index space. Using normalized 1/T so the cumulative sum
    has a comparable scale across configurations."""
    T = len(timesteps)
    return [1.0 / T] * T


def run_model(model_name: str,
              pipe,
              prompts,
              num_prompts: int,
              seeds_per_prompt: int,
              num_inference_steps: int,
              guidance_scale: float,
              num_probes: int,
              num_terms: int,
              base_seed: int,
              device) -> list[Row]:
    """Deterministic DDIM (eta=0) reverse loop per (prompt, seed). At each step
    we compute log|det J_t| using Hutchinson + Taylor of log(I + c*M)."""
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ts = pipe.scheduler.timesteps                # tensor of length T
    uncond_embed, cond_embeds = encode_prompts(pipe, prompts[:num_prompts], device)

    rows: list[Row] = []
    pbar = tqdm(total=num_prompts * seeds_per_prompt * num_inference_steps,
                desc=f"{model_name}")
    for prompt_idx in range(num_prompts):
        cond_embed = cond_embeds[prompt_idx]                         # (1, seq, d)
        combined = torch.cat([uncond_embed, cond_embed], dim=0)      # (2, seq, d)

        for kseed in range(seeds_per_prompt):
            seed = base_seed + prompt_idx * 1000 + kseed
            torch.manual_seed(seed)
            latent = pipe.prepare_latents(
                1, pipe.unet.config.in_channels,
                pipe.unet.config.sample_size * pipe.vae_scale_factor,
                pipe.unet.config.sample_size * pipe.vae_scale_factor,
                cond_embed.dtype, device, None,
            )

            for step_idx, t in enumerate(ts):
                t_int = int(t.item()) if torch.is_tensor(t) else int(t)
                # next timestep (smaller t); diffusers convention: index -1
                # means the final step; we use t_prev = -1 to flag it.
                t_prev = int(ts[step_idx + 1].item()) if step_idx + 1 < len(ts) else -1

                a_t, b_t = get_ab_for_step(pipe.scheduler, t_int,
                                           t_prev if t_prev >= 0 else None)

                f = make_guided_eps_fn(pipe.unet, combined, t, guidance_scale)
                res = hutchinson_logdet_step(f, latent, a_t, b_t,
                                              num_probes=num_probes,
                                              num_terms=num_terms)
                # log_det_step is shape (B=1,) -> .item()
                ld = float(res["log_det"].item())
                tm = [float(x.item()) for x in res["term_means"]]
                ts_ = [float(x.item()) for x in res["term_stds"]]
                # pad to 3 terms for the dataclass
                while len(tm) < 3:
                    tm.append(0.0); ts_.append(0.0)
                # apply Taylor signs/divisions for the saved 'term_i' values
                # (so reading them back gives the actual contribution to log|det|)
                signed_terms = [((-1) ** (k + 1)) * tm[k] / (k + 1) for k in range(3)]
                signed_ses   = [ts_[k] / (k + 1) for k in range(3)]
                rows.append(Row(
                    model=model_name,
                    prompt_id=prompt_idx,
                    prompt=prompts[prompt_idx],
                    seed=seed,
                    step_idx=step_idx,
                    t=t_int,
                    t_prev=t_prev,
                    a_t=a_t,
                    b_t=b_t,
                    c_t=res["c"],
                    D=res["D"],
                    log_det_step=ld,
                    term1=signed_terms[0],
                    term2=signed_terms[1],
                    term3=signed_terms[2],
                    term1_se=signed_ses[0],
                    term2_se=signed_ses[1],
                    term3_se=signed_ses[2],
                ))

                # Take the actual DDIM step (with the most recent guided eps,
                # recomputed cheaply with no_grad to keep latent advancement
                # exactly aligned with the scheduler's expectations).
                with torch.no_grad():
                    eps_tilde = f(latent)
                    latent = pipe.scheduler.step(eps_tilde, t, latent,
                                                 return_dict=False)[0]
                pbar.update(1)
    pbar.close()
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_active_set_S(num_inference_steps: int, size: int) -> list[int]:
    """Reproduce train_pipeline.py incremental rule for first fill."""
    candidate = np.arange(0, num_inference_steps)
    idx = np.linspace(0, len(candidate) - 1, size).round().astype(int)
    return sorted(candidate[idx].tolist())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--iadd_ckpt", default="model/lora/only_10_ddpo/stage27/checkpoints/checkpoint_1")
    p.add_argument("--full_ckpt", default="model/lora/vanilla_ddpo/stage27/checkpoints/checkpoint_1")
    p.add_argument("--prompt_file", default="configs/prompt/template1_test.json")
    p.add_argument("--output_dir", default="rebuttal/artifacts/volume_preservation")
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_probes", type=int, default=4)
    p.add_argument("--num_terms", type=int, default=3,
                   help="Taylor expansion terms for log|det(I + cM)|")
    p.add_argument("--num_prompts", type=int, default=6)
    p.add_argument("--seeds_per_prompt", type=int, default=4)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--iadd_S_size", type=int, default=10,
                   help="|S| for the iadd model (used to annotate rows; "
                        "must match training-time only_train_steps)")
    p.add_argument("--models", default="ref,iadd,full",
                   help="comma-list subset of {ref,iadd,full}")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    with open(args.prompt_file) as f:
        prompts = json.load(f)
    if len(prompts) < args.num_prompts:
        print(f"[warn] prompt file has {len(prompts)} prompts < {args.num_prompts}; "
              f"using all of them")
        args.num_prompts = len(prompts)

    active_S = build_active_set_S(args.num_inference_steps, args.iadd_S_size)
    print(f"[info] T = {args.num_inference_steps}, |S| = {args.iadd_S_size}, "
          f"S (timestep indices) = {active_S}")

    cfg = vars(args).copy()
    cfg["active_S"] = active_S
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    model_paths = {
        "ref":  None,
        "iadd": args.iadd_ckpt,
        "full": args.full_ckpt,
    }

    want = [m.strip() for m in args.models.split(",") if m.strip()]
    all_rows: list[Row] = []
    for name in want:
        if name not in model_paths:
            raise ValueError(f"unknown model {name}")
        t0 = time.time()
        pipe = load_model(name, model_paths[name], args.base_model, device)
        rows = run_model(
            model_name=name, pipe=pipe, prompts=prompts,
            num_prompts=args.num_prompts,
            seeds_per_prompt=args.seeds_per_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_probes=args.num_probes,
            num_terms=args.num_terms,
            base_seed=args.base_seed,
            device=device,
        )
        all_rows.extend(rows)
        # Save incrementally so partial runs are useful
        df = pd.DataFrame([r.__dict__ for r in rows])
        df["in_active_S"] = df["step_idx"].isin(active_S)
        out = os.path.join(args.output_dir, f"divergence_{name}.csv")
        df.to_csv(out, index=False)
        print(f"[{name}] wrote {len(df)} rows -> {out}   ({time.time()-t0:.1f}s)")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    if len(all_rows) == 0:
        return
    full_df = pd.DataFrame([r.__dict__ for r in all_rows])
    full_df["in_active_S"] = full_df["step_idx"].isin(active_S)
    full_df.to_csv(os.path.join(args.output_dir, "divergence.csv"), index=False)
    print(f"[done] combined csv: divergence.csv  ({len(full_df)} rows)")


if __name__ == "__main__":
    main()
