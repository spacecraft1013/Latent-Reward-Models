# train_lrm.py (Revised and complete with Logging/Plotting)
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
import csv  # <--- ADDED
import matplotlib.pyplot as plt  # <--- ADDED

# Assuming 'rewards.latent_aesthetic' is available
from rewards.latent_aesthetic import LatentAestheticViaLAION


# --------------------------- Dataset Class ---------------------------
class LatentDataset(Dataset):
    """
    Loads latents from .pt files and maps them to scores saved in scores.json.
    """

    def __init__(
        self, latents_dir: str, id_to_score: Dict[str, float], allowed_ids: List[str]
    ):
        self.latents_dir = Path(latents_dir)
        self.id_to_score = id_to_score
        self.ids = allowed_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        path = self.latents_dir / f"{uid}.pt"
        target_score = self.id_to_score[uid]

        try:
            latent = torch.load(path, map_location="cpu")
            # Latent format correction: WanVAE latents are often (C, 1, H, W)
            if latent.dim() == 4 and latent.shape[1] == 1:
                latent = latent.squeeze(1)
            elif latent.dim() == 5:
                latent = latent.squeeze(0).squeeze(1)

            return uid, latent, target_score
        except Exception as e:
            # Raise an exception to let the DataLoader skip this sample
            raise RuntimeError(f"Failed to load latent for {uid}")


# Custom collate function to handle the RuntimeError in __getitem__
def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if not batch:
        return None, None, None

    uids, latents, targets = zip(*batch)

    # Stack latents and targets
    latents = torch.stack(latents, dim=0)
    targets = torch.tensor(targets, dtype=torch.float32)

    return uids, latents, targets


# --------------------------- Logging & Plotting (NEW) ---------------------------
# --------------------------- Logging & Plotting (CORRECTED) ---------------------------


def log_metrics(
    save_dir: Path, epoch: int, metrics: Dict[str, float], init: bool = False
) -> None:
    """Logs epoch metrics to a CSV file."""
    log_path = save_dir / "training_log.csv"
    # Ensure 'epoch' is the first field
    fieldnames = [
        "epoch",
        "train_loss",
        "val_mse",
        "val_mae",
        "train_pearson",
        "val_pearson",
        "val_spearman",
        "val_r2",
        "lr",
    ]

    # Use 'w' mode to write the header only on initialization
    write_mode = "w" if init else "a"

    # Check if header needs to be written (only if file is new or we are initializing)
    write_header = init or not log_path.exists()

    with open(log_path, write_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()
            if init:
                return  # Only write header on init call

        # Prepare row data, ensuring we only write the fields we defined
        row_data = {k: metrics.get(k, 0.0) for k in fieldnames if k != "epoch"}
        row_data["epoch"] = epoch
        writer.writerow(row_data)


def plot_metrics(save_dir: Path) -> None:
    """Reads the CSV log and generates two plot files (loss and correlation)."""
    log_path = save_dir / "training_log.csv"
    if not log_path.exists():
        return

    data = []
    with open(log_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            processed_row = {}
            for k, v in row.items():
                if k == "epoch":
                    # Ensure 'epoch' is converted to integer
                    try:
                        processed_row[k] = int(v)
                    except ValueError:
                        continue  # Skip if epoch is not a valid number
                else:
                    # Convert all metric values to float
                    try:
                        processed_row[k] = float(v)
                    except ValueError:
                        pass  # Ignore if a metric value is empty or malformed

            # Only append rows that actually contain data (i.e., epoch)
            if "epoch" in processed_row:
                data.append(processed_row)

    if not data:
        return

    # THE FIX: 'epoch' is now guaranteed to be in the dictionaries in 'data'
    epochs = [row["epoch"] for row in data]

    exp_name = save_dir.name  # Use the experiment folder name for the title

    # --- Plot 1: Loss Metrics (Train Loss, Validation MSE) ---
    plt.figure(figsize=(10, 6))

    train_losses = [row["train_loss"] for row in data]
    val_mses = [row["val_mse"] for row in data]

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss (Criterion)",
        marker="o",
        linestyle="--",
        markersize=3,
    )
    plt.plot(epochs, val_mses, label="Validation MSE", marker="o", markersize=3)

    plt.title(f"Loss Metrics vs. Epoch ({exp_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_metrics.png")
    plt.close()

    # --- Plot 2: Correlation Metrics (Train/Val Pearson, Val Spearman) ---
    plt.figure(figsize=(10, 6))

    train_pearsons = [row["train_pearson"] for row in data]
    val_pearsons = [row["val_pearson"] for row in data]
    val_spearmans = [row["val_spearman"] for row in data]

    plt.plot(
        epochs,
        train_pearsons,
        label="Train Pearson",
        marker="o",
        linestyle="--",
        markersize=3,
    )
    plt.plot(epochs, val_pearsons, label="Validation Pearson", marker="o", markersize=3)
    plt.plot(
        epochs, val_spearmans, label="Validation Spearman", marker="x", markersize=3
    )

    plt.title(f"Correlation Metrics vs. Epoch ({exp_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_metrics.png")
    plt.close()


# --------------------------- Utilities (Unchanged) ---------------------------
def split_ids(
    all_ids: Iterable[str], val_size: int = 100, seed: int = 42
) -> Tuple[List[str], List[str]]:
    all_ids = list(all_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    val_ids = all_ids[:val_size]
    train_ids = all_ids[val_size:]
    return train_ids, val_ids


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    use_autocast: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    mse = nn.MSELoss(reduction="mean")
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    preds_all, targets_all = [], []

    for _, latents, targets in dl:
        if latents is None:
            continue

        latents = latents.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                pred = model(latents)
        else:
            pred = model(latents)

        pred = pred.float()
        targets = targets.float()

        mse_sum += mse(pred, targets).item() * targets.numel()
        mae_sum += torch.mean(torch.abs(pred - targets)).item() * targets.numel()
        n += targets.numel()

        preds_all.append(pred.detach().cpu())
        targets_all.append(targets.detach().cpu())

    if n == 0:
        return {
            "val_mse": 0.0,
            "val_mae": 0.0,
            "val_pearson": 0.0,
            "val_spearman": 0.0,
            "val_r2": 0.0,
        }

    preds_all = torch.cat(preds_all).numpy()
    targets_all = torch.cat(targets_all).numpy()

    # Correlations & R^2
    pearson = (
        float(np.corrcoef(preds_all, targets_all)[0, 1])
        if np.std(targets_all) > 1e-6
        else 0.0
    )
    try:
        spearman = float(spearmanr(preds_all, targets_all).correlation)
    except Exception:
        spearman = 0.0

    ss_res = float(np.sum((targets_all - preds_all) ** 2))
    ss_tot = float(np.sum((targets_all - np.mean(targets_all)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    return {
        "val_mse": mse_sum / n,
        "val_mae": mae_sum / n,
        "val_pearson": pearson,
        "val_spearman": spearman,
        "val_r2": r2,
    }


# --------------------------- Augmentation (Unchanged) ---------------------------
class LatentAugmentation:
    def __init__(self, noise_std: float = 0.0, dropout_prob: float = 0.0):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            noise = torch.randn_like(latents) * self.noise_std
            latents = latents + noise
        if self.dropout_prob > 0:
            mask = torch.bernoulli(torch.ones_like(latents) * (1 - self.dropout_prob))
            latents = latents * mask / (1 - self.dropout_prob + 1e-6)  # Scale up
        return latents


# --------------------------- Training (Modified) ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--latents_dir", type=str, default="data/latents")
    parser.add_argument("--scores_json", type=str, default="data/scores.json")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Seed for train/val split"
    )

    # Model arguments
    parser.add_argument(
        "--in_channels",
        type=int,
        default=16,
        help="Channels in latent (WanVAE is usually 16)",
    )
    parser.add_argument("--clip_model", type=str, default="vit_l_14")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in the LatentToCLIPEmbedding module",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # Loss/Scheduler/Augmentation arguments (for ablation)
    parser.add_argument(
        "--loss_fn", type=str, default="mse", choices=["mse", "huber", "smooth_l1"]
    )
    parser.add_argument(
        "--huber_delta", type=float, default=1.0, help="Delta parameter for Huber loss"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "plateau"]
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise added to latents",
    )
    parser.add_argument(
        "--latent_dropout",
        type=float,
        default=0.0,
        help="Latent dropout (channel/feature mask) applied to latents",
    )

    # Checkpoint/Stopping arguments
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_pearson",
        choices=["val_mse", "val_mae", "val_pearson", "val_spearman", "val_r2"],
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Number of epochs to wait for improvement before stopping. 0 to disable.",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # Set Torch Seed for reproducibility
    torch.manual_seed(args.split_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.split_seed)

    # Load scores
    print(f"Loading scores from {args.scores_json}...")
    with open(args.scores_json, "r") as f:
        id_to_score: Dict[str, float] = json.load(f)

    all_ids = list(id_to_score.keys())
    train_ids, val_ids = split_ids(
        all_ids, val_size=args.val_size, seed=args.split_seed
    )

    print(
        f"Total samples found: {len(all_ids)}. Training on {len(train_ids)}, Val on {len(val_ids)}"
    )

    # Datasets and DataLoaders
    ds_train = LatentDataset(args.latents_dir, id_to_score, train_ids)
    ds_val = LatentDataset(args.latents_dir, id_to_score, val_ids)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    model = LatentAestheticViaLAION(
        in_channels=args.in_channels, clip_model=args.clip_model, dropout=args.dropout
    ).to(device)

    # Optimizer
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Loss Function selection
    if args.loss_fn == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta)
        print(f"Using Huber Loss (delta={args.huber_delta})")
    elif args.loss_fn == "smooth_l1":
        criterion = nn.SmoothL1Loss()
        print("Using Smooth L1 Loss (default delta=1.0)")
    else:  # Default is mse
        criterion = nn.MSELoss()
        print("Using MSE Loss")

    # Scheduler selection
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs * len(dl_train)
        )
        print("Using Cosine Annealing LR Scheduler")
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max"
            if "pearson" in args.best_metric or "r2" in args.best_metric
            else "min",
            patience=10,
            factor=0.5,
        )
        print("Using ReduceLROnPlateau Scheduler")

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    use_autocast = args.mixed_precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    # Augmentation
    augmenter = LatentAugmentation(
        noise_std=args.noise_std, dropout_prob=args.latent_dropout
    )

    # Setup directories and checkpoint tracking
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_metric = None
    patience_counter = 0

    # Initialize logging file <--- NEW
    log_metrics(save_dir, 0, {}, init=True)

    print("Starting training...")

    for epoch in range(args.epochs):
        # --------------------- TRAINING LOOP ---------------------
        model.train()
        running_loss = 0.0
        train_preds_all, train_targets_all = [], []  # <--- COLLECT TRAIN METRICS

        for batch_idx, (_, latents, targets) in enumerate(dl_train):
            if latents is None:
                continue  # Skip if the whole batch failed

            latents = latents.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Apply latent augmentation
            if args.noise_std > 0 or args.latent_dropout > 0:
                latents = augmenter(latents)

            optim.zero_grad()

            with torch.set_grad_enabled(True):
                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        pred = model(latents)
                        loss = criterion(pred, targets)

                    if args.mixed_precision == "fp16":
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                    else:
                        loss.backward()
                        optim.step()
                else:
                    pred = model(latents)
                    loss = criterion(pred, targets)
                    loss.backward()
                    optim.step()

            # Cosine scheduler step happens every batch
            if args.scheduler == "cosine":
                scheduler.step()

            running_loss += loss.item()
            train_preds_all.append(pred.detach().cpu())  # <--- COLLECT PRED
            train_targets_all.append(targets.detach().cpu())  # <--- COLLECT TARGET

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(dl_train)}] Loss: {loss.item():.4f}, LR: {optim.param_groups[0]['lr']:.2e}"
                )

        # --------------------- EPOCH METRICS CALCULATION (NEW) ---------------------

        # Calculate final train loss and train Pearson
        avg_train_loss = running_loss / len(dl_train)

        train_preds_all_np = torch.cat(train_preds_all).numpy()
        train_targets_all_np = torch.cat(train_targets_all).numpy()

        train_pearson = (
            float(np.corrcoef(train_preds_all_np, train_targets_all_np)[0, 1])
            if np.std(train_targets_all_np) > 1e-6
            else 0.0
        )

        # --------------------- VALIDATION LOOP ---------------------
        val_metrics = evaluate(model, dl_val, device, use_autocast, amp_dtype)

        # Plateau scheduler step happens every epoch (after validation)
        if args.scheduler == "plateau":
            metric_to_monitor = val_metrics[args.best_metric]
            scheduler.step(metric_to_monitor)

        # Combine all metrics for logging
        current_lr = optim.param_groups[0]["lr"]
        all_metrics = {
            "train_loss": avg_train_loss,
            "train_pearson": train_pearson,
            "lr": current_lr,
            **val_metrics,  # unpacks val_mse, val_pearson, etc.
        }

        # --------------------- LOGGING AND PLOTTING (NEW) ---------------------
        log_metrics(save_dir, epoch, all_metrics)
        plot_metrics(save_dir)

        print(
            f"Epoch {epoch} Done. Train Loss: {all_metrics['train_loss']:.4f} | Train Pearson: {all_metrics['train_pearson']:.4f} | "
            f"Val MSE: {all_metrics['val_mse']:.4f} | Val Pearson: {all_metrics['val_pearson']:.4f}"
        )

        # --------------------- CHECKPOINT & EARLY STOPPING (Unchanged) ---------------------

        # 2. Best checkpoint tracking
        metric_value = all_metrics[args.best_metric]
        is_best = False

        if best_ckpt_metric is None:
            best_ckpt_metric = metric_value
            is_best = True
        else:
            # Determine if this epoch is better (higher for Pearson/R2, lower for MSE/MAE)
            is_improving = "pearson" in args.best_metric or "r2" in args.best_metric

            if is_improving:
                is_better = metric_value > best_ckpt_metric
            else:
                is_better = metric_value < best_ckpt_metric

            if is_better:
                best_ckpt_metric = metric_value
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1

        if is_best:
            # Save best checkpoint separately
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  ✓ New best {args.best_metric}: {best_ckpt_metric:.4f}")

        # Early stopping
        if (
            args.early_stop_patience > 0
            and patience_counter >= args.early_stop_patience
        ):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            print(f"Best {args.best_metric}: {best_ckpt_metric:.4f}")
            break

    print("\n" + "=" * 60)
    print("Training complete!")

if __name__ == "__main__":
    main()