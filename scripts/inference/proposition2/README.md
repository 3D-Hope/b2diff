# Volume-preservation diagnostic (Prop 2)

Empirical validation of Proposition 2: at matched training budget the sparse
update strategy contracts the volume of the noise prior less than the dense
strategy. Produces the headline `log|det J_{T→0}|` numbers and the cumulative
plot from the rebuttal.

This directory is **self-contained** — re-runs only need these 3 files plus
the LoRA checkpoints. No code outside `scripts/diagnostics/` is required.

## Files

| File | Purpose |
|------|---------|
| `volume_preservation.py` | Main script. Loads `{ref, iadd, full}`, runs deterministic DDIM (η=0), at every step estimates `log|det J_t|` via Hutchinson + 3-term Taylor expansion of `log det(I + c_t M_t)`. Writes `divergence_{model}.csv`. |
| `plot_cumulative.py` | Reads the CSVs and produces the two-panel figure: (a) cumulative `log|det J_{T→t}|` vs step, (b) paired cumulative gap vs `ref`. Also prints the headline numbers + SEMs. |
| `run_volume_preservation_msp3.sh` | sbatch wrapper for **msp3** (H200 + neurips-2026 QoS). Installs/activates the `b2` conda env, then runs the python script. |
| `run_volume_preservation.sh` | sbatch wrapper for **hala/sof1** (a6000 / H200). Uses the pre-built env at `/work/pramish/tools/miniconda3/envs/b2`. |

## Method (one paragraph)

For each (prompt, seed), run a deterministic DDIM reverse pass `x_T → x_0`. The
DDIM step is `x_{t-1} = a_t · x_t + b_t · ε̃_θ(x_t, c, t)`, so the step
Jacobian is `J_t = a_t · I + b_t · M` with `M = ∂ε̃/∂x`. We split:

```
log|det J_t|  =  D · log|a_t|  +  log|det(I + c_t · M)|,   c_t = b_t / a_t
```

The first term is exact and analytic. For the second term we use the Taylor
series

```
log|det(I + cM)|  =  c·tr(M)  -  c²·tr(M²)/2  +  c³·tr(M³)/3  -  ...
```

truncated at 3 terms. Each `tr(M^k)` is Hutchinson-estimated with K Rademacher
probes; `M^k v` is computed by iterating `torch.func.jvp` k times (no graph
materialization, no explicit Jacobian). Sum across t to get
`log|det J_{T→0}|`. Compare across models.

Both finetuned models are compared at **stage 27** (matched optimizer steps).

## Reproducing the rebuttal numbers

Default config: 6 prompts × 4 seeds × T=20 × K=4 probes × 3 Taylor terms.
Wall time on a single H200: ~18 min/model. 3 models in parallel: ~25 min total.

### On msp3 (recommended for speed)

```bash
# from hala login
ssh msp3
cd /home/pramish_paudel/codes/b2diff

# launch 3 parallel jobs, one per model
for m in ref iadd full; do
  sbatch --job-name=volprop2_${m} \
    scripts/diagnostics/run_volume_preservation_msp3.sh \
    --models $m --num_prompts 6 --seeds_per_prompt 4 \
    --num_probes 4 --num_terms 3 \
    --iadd_ckpt model/lora/only_10_ddpo/stage27/checkpoints/checkpoint_1 \
    --full_ckpt model/lora/vanilla_ddpo/stage27/checkpoints/checkpoint_1 \
    --output_dir rebuttal/artifacts/volume_preservation/run2_stage27
done

# wait for completion, then plot
python scripts/diagnostics/plot_cumulative.py \
    --input_dir rebuttal/artifacts/volume_preservation/run2_stage27 \
    --output rebuttal/figures/volume_preservation_cumulative.pdf
```

### On hala/sof1 (uses `/work` env)

```bash
for m in ref iadd full; do
  sbatch --partition=batch --gpus=a6000:1 --time=04:00:00 \
    --job-name=volprop2_${m} \
    scripts/diagnostics/run_volume_preservation.sh \
    --models $m --num_prompts 6 --seeds_per_prompt 4 \
    --num_probes 4 --num_terms 3 \
    --iadd_ckpt model/lora/only_10_ddpo/stage27/checkpoints/checkpoint_1 \
    --full_ckpt model/lora/vanilla_ddpo/stage27/checkpoints/checkpoint_1 \
    --output_dir rebuttal/artifacts/volume_preservation/run2_stage27
done
```

## Output schema

`rebuttal/artifacts/volume_preservation/run2_stage27/divergence_{model}.csv`
columns:
- `model, prompt_id, prompt, seed, step_idx, t, t_prev`
- `a_t, b_t, c_t, D` — DDIM step constants and latent dimension
- `log_det_step` — per-step `log|det J_t|` (single scalar per row)
- `term1, term2, term3` — signed Taylor contributions (diagnostics)
- `term1_se, term2_se, term3_se` — per-probe std-err of each term

To get the headline number per trajectory: `groupby(model, prompt_id, seed) →
sum log_det_step` ⇒ that's `log|det J_{T→0}|` for that (image, model).
Then `mean ± SEM` across the 24 trajectories.

## Rebuttal numbers (sanity check)

Final cumulative `log|det J_{T→0}|`, mean ± SEM over 6 prompts × 4 seeds:

| Model | Value | SEM |
|---|---|---|
| sparse (iadd, stage 27) | **−25,963** | 584 |
| ref (SD v1.4)            | −26,996  | 727 |
| dense (full DDPO, stage 27) | −27,568 | 658 |

Paired sparse − dense: **+1,605 ± 437** (n=24, ~3.7σ).

## Caveats

- η=0 (deterministic) at measurement time only. Training/sampling stays η=1.
- Hutchinson is unbiased but variance grows at low t (large `c_t` means series
  converges more slowly). 3 terms is empirically enough — `|term3| / |term1|`
  stays under ~3% on average.
- LoRA weights are loaded via `pipeline.load_lora_weights` + `fuse_lora()` so
  the JVP runs through plain `nn.Linear` modules. The legacy
  `LoRAAttnProcessor` path is not forward-mode-AD compatible and segfaults.
- Math SDPA kernel is forced inside the JVP region because flash/mem-eff
  kernels don't implement forward AD.
- All compute is fp32 (forward AD with fp16 layernorm gives dtype mismatches).
