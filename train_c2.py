# ============================================================
# train_challenge2.py
# ============================================================
# EEG Challenge 2: Subject-Invariant Externalizing Factor Regression
# - Uses adversarial subject loss during training
# - Evaluates only regression performance on val/test
# ============================================================

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import EEGChallengeDataLoader
from models import EEGFeatureExtractor, EEG_Externalizing_Model
from config import DEVICE, LEARNING_RATE, LOG_INTERVAL, BATCH_SIZE

# ============================================================
# CONFIG
# ============================================================
N_EPOCHS = 50
ADV_WEIGHT = 0.1
GRL_MAX = 0.3
DISC_LR_RATIO = 0.5
LOSS_FN = "MSELoss"
OPTIMIZER = "AdamW"
CSV_FILE = "results_challenge2.csv"

os.makedirs("checkpoints", exist_ok=True)
from config import RELEASES
# ============================================================
# LOSS HELPERS
# ============================================================
def get_regression_loss(name="Hybrid"):
    if name == "Hybrid":
        return lambda pred, y: 0.7 * F.mse_loss(pred, y) + 0.3 * F.l1_loss(pred, y)
    else:
        return getattr(nn, name)()

def dann_lambda(epoch, batch_idx, total_batches, total_epochs, max_l=GRL_MAX):
    """Schedule for lambda used in GRL"""
    p = (epoch - 1 + batch_idx / float(total_batches)) / float(total_epochs)
    val = float(2.0 / (1.0 + torch.exp(-10.0 * torch.tensor(p))) - 1.0)
    return min(max_l, val)

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, dataloader, loss_fn, y_mean, y_std):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for X, y, *_ in dataloader:
            X = X.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)

            y_pred, _ = model(X, lambd=0.0)  # disable GRL for eval
            y_pred, y = y_pred.view(-1), y.view(-1)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * X.size(0)

            preds.append(y_pred.cpu())
            targets.append(y.cpu())

    preds, targets = torch.cat(preds), torch.cat(targets)

    # Denormalize for computing RMSE and NRMSE
    preds_denorm = preds * y_std + y_mean
    targets_denorm = targets * y_std + y_mean

    rmse = torch.sqrt(F.mse_loss(preds_denorm, targets_denorm))
    nrmse = rmse / y_std
    corr = torch.corrcoef(torch.stack([preds_denorm, targets_denorm]))[0, 1].item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, rmse.item(), float(nrmse), float(corr)

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train_model(full_dataset, n_subjects, n_epochs=50):
    writer = SummaryWriter(log_dir="runs/challenge2")
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_task_loss", "train_adv_loss", "val_rmse", "val_nrmse", "val_corr", "lr"])

    # Split data (80/20)
    val_len = int(0.2 * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Label normalization based on train set
    all_targets = []
    for _, y, *_ in train_loader:
        all_targets.append(y.view(-1))
    y_all = torch.cat(all_targets).to(DEVICE, dtype=torch.float32)
    y_mean, y_std = y_all.mean(), y_all.std()

    print(f"📏 Label normalization: mean={y_mean.item():.4f}, std={y_std.item():.4f}")

    # Initialize model and optimizers
    backbone = EEGFeatureExtractor(channels=129, time_steps=200)
    model = EEG_Externalizing_Model(backbone, num_subjects=n_subjects).to(DEVICE)
    print(model)

    loss_task = get_regression_loss(LOSS_FN)
    loss_adv = nn.CrossEntropyLoss()

    optimizer_main = getattr(optim, OPTIMIZER)(list(model.backbone.parameters()) + list(model.reg_head.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    optimizer_disc = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE * DISC_LR_RATIO, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer_main, T_max=n_epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_corr = -1.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_task_loss, running_adv_loss = 0.0, 0.0

        for batch_idx, (X, y, *_, info) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")):
            X = X.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)
            subj_ids = torch.tensor(info["subj_id"], dtype=torch.long, device=DEVICE)

            # --- Update Discriminator ---
            with torch.no_grad():
                z = model.backbone(X)
            logits_disc = model.discriminator(z.detach())
            loss_d = loss_adv(logits_disc, subj_ids)

            optimizer_disc.zero_grad()
            loss_d.backward()
            optimizer_disc.step()

            # --- Update Main Network (with GRL) ---
            lambd = float(dann_lambda(epoch, batch_idx, len(train_loader), n_epochs))
            optimizer_main.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                y_pred, logits_subj = model(X, lambd=lambd)
                y_pred = y_pred.view(-1)
                y_true = y.view(-1)
                L_task = loss_task(y_pred, y_true)
                L_adv = loss_adv(logits_subj, subj_ids)
                total_loss = L_task + ADV_WEIGHT * L_adv

            scaler.scale(total_loss).backward()
            scaler.step(optimizer_main)
            scaler.update()

            running_task_loss += L_task.item()
            running_adv_loss += L_adv.item()

        scheduler.step()
        current_lr = optimizer_main.param_groups[0]["lr"]

        # Validation
        val_loss, val_rmse, val_nrmse, val_corr = evaluate(model, val_loader, loss_task, y_mean, y_std)

        # Log to TensorBoard
        writer.add_scalar("Loss/train_task", running_task_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/train_adv", running_adv_loss / len(train_loader), epoch)
        writer.add_scalar("Val/RMSE", val_rmse, epoch)
        writer.add_scalar("Val/NRMSE", val_nrmse, epoch)
        writer.add_scalar("Val/Correlation", val_corr, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        # Save all metrics to CSV
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                running_task_loss / len(train_loader),
                running_adv_loss / len(train_loader),
                val_rmse,
                val_nrmse,
                val_corr,
                current_lr])

        print(f"Epoch {epoch} | LR={current_lr:.6e} | ValCorr={val_corr:.3f}")

        # Save best model by correlation
        if val_corr > best_corr:
            best_corr = val_corr
            checkpoint_path = f"checkpoints/best_challenge2_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved best model: {checkpoint_path} (Corr={best_corr:.3f})")

    writer.close()

# ============================================================
# MAIN
# ============================================================
def main():
    print(" Starting Challenge 2 Training...")
    loader = EEGChallengeDataLoader()
    train_loader, _, _ = loader.load_challenge2(releases=RELEASES,task_fraction=0.1)
    full_dataset = train_loader.dataset
    n_subjects = len(getattr(loader, "subject_to_idx", {}))
    print(f" Total subjects: {n_subjects}")
    print(f" Total training samples: {len(full_dataset)}")
    train_model(full_dataset, n_subjects, n_epochs=N_EPOCHS)

if __name__ == "__main__":
    main()
