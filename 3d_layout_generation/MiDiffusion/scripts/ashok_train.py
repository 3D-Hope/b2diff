# 
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
# 

"""Script used to train a diffusion models."""
import argparse
import os
import sys
import shutil
import time
import uuid
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from utils import PROJ_DIR, update_data_file_paths, id_generator, save_experiment_params, \
    load_config, get_time_str, load_checkpoints, save_checkpoints
from midiffusion.datasets.threed_front_encoding import get_encoded_dataset
import wandb
from midiffusion.stats_logger import StatsLogger, WandB
from midiffusion.ashok_midiffusion import SceneDiffuserMiDiffusion


# ---------------------------------------------------------------------------
# DDPM diffusion helpers
# ---------------------------------------------------------------------------

def make_ddpm_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """Pre-compute the linear beta schedule and the derived alpha / alpha-bar
    tensors that are needed for the DDPM forward (noising) process.

    Returns a dict with all schedule tensors on *device*.
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)              # ᾱ_t
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)           # √ᾱ_t
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # √(1-ᾱ_t)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def q_sample(x0, t, schedule):
    """DDPM forward process: add noise to *x0* at timestep *t*.

    x_t = √ᾱ_t · x_0  +  √(1–ᾱ_t) · ε,   ε ~ N(0, I)

    Args:
        x0  : (B, N, C) clean scene tensor
        t   : (B,) integer timestep indices
        schedule : dict of beta-schedule tensors (all on same device as x0)

    Returns:
        x_t   : (B, N, C) noisy scene
        noise : (B, N, C) the Gaussian noise that was added
    """
    noise = torch.randn_like(x0)
    sqrt_ab  = schedule["sqrt_alphas_cumprod"][t]              # (B,)
    sqrt_1ab = schedule["sqrt_one_minus_alphas_cumprod"][t]    # (B,)
    # broadcast over (N, C) dims
    sqrt_ab  = sqrt_ab[:, None, None]
    sqrt_1ab = sqrt_1ab[:, None, None]
    x_t = sqrt_ab * x0 + sqrt_1ab * noise
    return x_t, noise


def build_scene_tensor(sample):
    """Concatenate per-object attributes from the dataset batch dict into a
    single (B, N, C) scene tensor.

    The encoded dataset stores object attributes as separate keys:
        translations : (B, N, 3)
        sizes        : (B, N, 3)
        angles       : (B, N, 2)   (cosine/sine encoding)
        class_labels : (B, N, 22)
    Together they form the 30-D object vector used by MIDiffusionContinuous.
    """
    parts = []
    for key in ("translations", "sizes", "angles", "class_labels"):
        if key in sample:
            parts.append(sample[key])
    if not parts:
        raise KeyError(
            "Expected at least one of 'translations', 'sizes', 'angles', "
            "'class_labels' in the batch dict. Got: " + str(list(sample.keys()))
        )
    return torch.cat(parts, dim=-1)   # (B, N, C)


# ---------------------------------------------------------------------------
# train_on_batch  /  val_on_batch
# ---------------------------------------------------------------------------

def train_on_batch(network, optimizer, sample, schedule, max_grad_norm=None):
    """Run one training step and return the scalar loss value.

    Steps
    -----
    1. Build the clean scene tensor x0 from the batch dict.
    2. Sample random diffusion timesteps t ~ Uniform{1 … T}.
    3. Apply the DDPM forward process to get (x_t, ε).
    4. Forward the denoising network to predict ε̂.
    5. Compute MSE( ε̂ , ε ) loss, back-prop, clip grads, and step the optimizer.

    Args:
        network      : SceneDiffuserMiDiffusion – must already be in train() mode
                       and have its parameters on the correct device.
        optimizer    : torch.optim.Optimizer
        sample       : dict – one batch from the DataLoader (already on device)
        schedule     : dict of DDPM schedule tensors produced by make_ddpm_schedule
        max_grad_norm: float or None – gradient clipping threshold

    Returns:
        float – scalar loss value for logging
    """
    # --- 1. build clean scene tensor ------------------------------------------
    x0 = build_scene_tensor(sample)            # (B, N, C)

    # --- 2. sample timesteps --------------------------------------------------
    B = x0.shape[0]
    num_timesteps = schedule["betas"].shape[0]
    t = torch.randint(0, num_timesteps, (B,), device=x0.device)  # (B,)

    # --- 3. add noise ---------------------------------------------------------
    x_t, noise = q_sample(x0, t, schedule)    # (B, N, C), (B, N, C)

    # --- 4. predict noise -----------------------------------------------------
    # floor plan boundary points normals: key used by PointNet_Point encoder
    fpbpn = sample["fpbpn"]
    predicted_noise = network.predict_noise(x_t, t, fpbpn)  # (B, N, C)

    # --- 5. loss + backward ---------------------------------------------------
    class_loss      = F.mse_loss(predicted_noise[..., 8:],  noise[..., 8:])
    translation_loss = F.mse_loss(predicted_noise[..., :3],  noise[..., :3])
    size_loss       = F.mse_loss(predicted_noise[..., 3:6], noise[..., 3:6])
    angle_loss      = F.mse_loss(predicted_noise[..., 6:8], noise[..., 6:8])
    loss = class_loss + translation_loss + size_loss + angle_loss
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
    optimizer.step()
    return (
        loss.item(),
        class_loss.item(),
        translation_loss.item(),
        size_loss.item(),
        angle_loss.item(),
    )


@torch.no_grad()
def val_on_batch(network, sample, schedule):
    """Compute the validation loss for a single batch without gradients.

    Args:
        network  : SceneDiffuserMiDiffusion – must already be in eval() mode.
        sample   : dict – one batch from the DataLoader (already on device)
        schedule : dict of DDPM schedule tensors produced by make_ddpm_schedule

    Returns:
        float – scalar loss value for logging
    """
    # --- 1. build clean scene tensor ------------------------------------------
    x0 = build_scene_tensor(sample)           # (B, N, C)

    # --- 2. sample timesteps --------------------------------------------------
    B = x0.shape[0]
    num_timesteps = schedule["betas"].shape[0]
    t = torch.randint(0, num_timesteps, (B,), device=x0.device)

    # --- 3. add noise ---------------------------------------------------------
    x_t, noise = q_sample(x0, t, schedule)

    # --- 4. predict noise and compute loss ------------------------------------
    fpbpn = sample["fpbpn"]
    predicted_noise = network.predict_noise(x_t, t, fpbpn)
    class_loss      = F.mse_loss(predicted_noise[..., 8:],  noise[..., 8:])
    translation_loss = F.mse_loss(predicted_noise[..., :3],  noise[..., :3])
    size_loss       = F.mse_loss(predicted_noise[..., 3:6], noise[..., 3:6])
    angle_loss      = F.mse_loss(predicted_noise[..., 6:8], noise[..., 6:8])
    loss = class_loss + translation_loss + size_loss + angle_loss
    return (
        loss.item(),
        class_loss.item(),
        translation_loss.item(),
        size_loss.item(),
        angle_loss.item(),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR+"/output/log",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default="test",
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help=(
            "Overfit sanity check: take the first training sample, replicate "
            "it to fill the batch, and use that same batch every iteration."
        )
    )

    args = parser.parse_args(argv)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))
    
    if args.gpu < torch.cuda.device_count():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    experiment_tag = args.experiment_tag
    experiment_directory = os.path.join(args.output_directory, experiment_tag)
    os.makedirs(experiment_directory, exist_ok=True)
    # output files
    path_to_config = os.path.join(experiment_directory, "config.yaml")
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    path_to_params = os.path.join(experiment_directory, "params.json")
    path_to_stats = os.path.join(experiment_directory, "stats.txt")
    path_to_best_model = os.path.join(experiment_directory, "best_model.pt")

    # Parse the config file
    config = load_config(args.config_file)
    shutil.copyfile(args.config_file, path_to_config)

    train_dataset = get_encoded_dataset(
        update_data_file_paths(config["data"]),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=["train", "val"],
        max_length=config["network"]["sample_num_points"],
        include_room_mask=(config["network"]["room_mask_condition"] and \
                           config["feature_extractor"]["name"]=="resnet18")
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"],
        #add objfeats
        objfeats=train_dataset.bounds["objfeats"],
    )

    validation_dataset = get_encoded_dataset(
        update_data_file_paths(config["data"]),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=(config["network"]["room_mask_condition"] and \
                           config["feature_extractor"]["name"]=="resnet18")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # ---------------------------------------------------------------------- #
    #  Overfit-test: build a fixed batch from the very first training sample  #
    # ---------------------------------------------------------------------- #
    overfit_batch = None
    if args.overfit_test:
        print("[overfit_test] building fixed single-sample batch …")
        batch_size = config["training"].get("batch_size", 128)
        # grab the first raw batch produced by the loader (shape B×…)
        _first = next(iter(train_loader))
        # keep only the first sample (index 0) and tile it to batch_size
        overfit_batch = {}
        for k, v in _first.items():
            if isinstance(v, torch.Tensor):
                # take first element → (1, …), repeat along batch dim
                overfit_batch[k] = v[:1].expand(batch_size, *v.shape[1:]).clone()
            else:
                overfit_batch[k] = v      # lists / non-tensor metadata
        print(f"[overfit_test] fixed batch keys: {list(overfit_batch.keys())}")
        print(f"[overfit_test] scene shape: {overfit_batch['translations'].shape}")
    print("Saved dataset bounds to {}".format(path_to_bounds))
    print("  Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    print("  Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )

    # Build the network architecture to be used for training
    network = SceneDiffuserMiDiffusion()
    network = network.to(device)
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in \
        filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}: "
          f"{n_trainable_params} / {n_all_params}")
    config["network"]["n_params"] = n_trainable_params

    # Optionally load a pre-trained / checkpoint weight file
    if args.weight_file is not None:
        print(f"Loading weights from {args.weight_file}")
        state = torch.load(args.weight_file, map_location=device)
        network.load_state_dict(state)

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(network.parameters(), lr=config["training"]["lr"])

    epochs = config["training"].get("epochs", 65000)

    if args.continue_from_epoch == epochs:
        return

    # lr scheduler: piecewise-linear lambda decay (epoch-based)
    _lr_start_epoch = config["training"].get("lr_start_epoch", 1000)
    _lr_end_epoch   = config["training"].get("lr_end_epoch",   epochs)
    _lr_start       = config["training"]["lr"]
    _lr_end         = config["training"].get("lr_end", _lr_start)

    def _lambda_lr(epoch):
        """Hold start_lr up to start_epoch, then linearly decay to end_lr."""
        if epoch <= _lr_start_epoch:
            return 1.0
        elif epoch <= _lr_end_epoch:
            total = _lr_end_epoch - _lr_start_epoch
            frac  = (epoch - _lr_start_epoch) / total
            return (1 - frac) * 1.0 + frac * (_lr_end / _lr_start)
        else:
            return _lr_end / _lr_start

    # last_epoch=-1 so PyTorch's internal init-step sets last_epoch=0;
    # we then call scheduler.step() once per epoch so it tracks loop var i.
    scheduler = LambdaLR(optimizer, lr_lambda=_lambda_lr,
                         last_epoch=args.continue_from_epoch - 1)

    # Pre-compute the DDPM noise schedule (kept on device for efficiency)
    num_timesteps = config["network"].get("time_num", 1000)
    diff_geo = config["network"].get("diffusion_geometric_kwargs", {})
    ddpm_schedule = make_ddpm_schedule(
        num_timesteps=num_timesteps,
        beta_start=diff_geo.get("beta_start", 1e-4),
        beta_end=diff_geo.get("beta_end", 0.02),
        device=device,
    )

    # Initialize the logger
    wandb_id = str(uuid.uuid4())
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get("project", "MiDiffusion"),
            name=experiment_tag,
            watch=False,
            log_frequency=10,
            id=wandb_id,
        )
        args.with_wandb_logger = WandB.instance().id

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(path_to_stats, "w"))

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, path_to_params)
    print("Save experiment statistics in {}".format(experiment_directory))

    # Do the training
    max_grad_norm = config["training"].get("max_grad_norm", None)
    save_every    = config["training"].get("save_frequency", 1000)
    val_every     = config["validation"].get("frequency", 1000)

    min_val_loss       = float("inf")
    min_val_loss_epoch = 0
    global_step        = 0          # counts every optimizer step across all epochs
    tic = time.perf_counter()

    for i in range(args.continue_from_epoch + 1, epochs + 1):

        # ------------------------------------------------------------------ #
        #  Training epoch                                                      #
        # ------------------------------------------------------------------ #
        network.train()
        _train_iter = (
            [(0, overfit_batch)]          # single fixed batch every epoch
            if args.overfit_test
            else enumerate(train_loader)
        )
        for b, sample in _train_iter:
            # Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)

            optimizer.zero_grad()
            loss, cls, trans, sz, ang = train_on_batch(
                network, optimizer, sample, ddpm_schedule, max_grad_norm
            )
            global_step += 1
            StatsLogger.instance().print_progress(i, b + 1, loss)

            # -- log every step to wandb ---------------------------------------
            if args.with_wandb_logger:
                wandb.log({
                    "train/loss":        loss,
                    "train/class":       cls,
                    "train/translation": trans,
                    "train/size":        sz,
                    "train/angle":       ang,
                    "train/lr":          optimizer.param_groups[0]["lr"],
                }, step=global_step)

        scheduler.step()              # step LR once per epoch
        if (i % save_every) == 0:
            save_checkpoints(i, network, optimizer, experiment_directory)
        StatsLogger.instance().clear()

        # ------------------------------------------------------------------ #
        #  Validation epoch                                                    #
        # ------------------------------------------------------------------ #
        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            val_totals = {"loss": 0., "class": 0., "translation": 0., "size": 0., "angle": 0.}
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if not isinstance(v, list):
                        sample[k] = v.to(device)
                loss, cls, trans, sz, ang = val_on_batch(network, sample, ddpm_schedule)
                val_totals["loss"]        += loss
                val_totals["class"]       += cls
                val_totals["translation"] += trans
                val_totals["size"]        += sz
                val_totals["angle"]       += ang
                StatsLogger.instance().print_progress(-1, b + 1, loss)
            StatsLogger.instance().clear()

            # -- log val losses ------------------------------------------------
            n_val_batches = b + 1
            val_log = {f"val/{k}": v / n_val_batches for k, v in val_totals.items()}
            print(
                f"[Epoch {i}] val   | "
                + "  ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in val_log.items())
            )
            if args.with_wandb_logger:
                wandb.log(val_log, step=i)

            toc = time.perf_counter()
            elapsed_time = toc - tic
            estimated_total_time = (
                elapsed_time / (i - args.continue_from_epoch)
            ) * (epochs - args.continue_from_epoch)
            print("====> [Elapsed time: {}] / [Estimated total: {}] ====>" .format(
                get_time_str(elapsed_time), get_time_str(estimated_total_time)
            ))

            val_loss_total = val_totals["loss"]
            if val_loss_total < min_val_loss:
                # Overwrite best_model.pt
                min_val_loss       = val_loss_total
                min_val_loss_epoch = i
                torch.save(network.state_dict(), path_to_best_model)

    print(
        "Best model saved at epoch {} with validation loss = {}".format(
            min_val_loss_epoch, min_val_loss
        ),
        file=open(path_to_stats, "a"),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
