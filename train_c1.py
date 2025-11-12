# ============================================================
# train_challenge1_final.py 
# ============================================================
# EEG Challenge 1: Reaction-Time Regression
# ============================================================

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloader import EEGChallengeDataLoader
from models import EEGFeatureExtractor, EEG_CCD_Model
from config import DEVICE, LEARNING_RATE, LOG_INTERVAL,RELEASES

# ============================================================
# Loss Functions
# ============================================================
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Prevent sqrt(0)

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

def get_loss_fn(name="Hybrid"):
    if name == "Hybrid":
        def hybrid_loss(pred, y):
            pred = pred.view(-1)
            y = y.view(-1)
            return 0.7 * F.mse_loss(pred, y) + 0.3 * F.l1_loss(pred, y)
        return hybrid_loss
    elif name == "RMSE":
        return RMSELoss()
    else:
        base_loss = getattr(nn, name)()
        return lambda pred, y: base_loss(pred.view(-1), y.view(-1))

# ============================================================
# Evaluation
# ============================================================
def evaluate(model, dataloader, loss_fn, y_mean, y_std):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for X, y, *_ in dataloader:
            X = X.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32).view(-1)
            y_norm = (y - y_mean) / y_std

            y_pred = model(X).view(-1)
            loss = loss_fn(y_pred, y_norm)
            total_loss += loss.item() * X.size(0)

            preds.append(y_pred.detach().cpu())
            targets.append(y_norm.detach().cpu())

    preds = torch.cat(preds).view(-1)
    targets = torch.cat(targets).view(-1)

    y_mean_cpu, y_std_cpu = y_mean.cpu(), y_std.cpu()
    preds_denorm = preds * y_std_cpu + y_mean_cpu
    targets_denorm = targets * y_std_cpu + y_mean_cpu

    avg_loss = total_loss / len(dataloader.dataset)
    rmse = torch.sqrt(F.mse_loss(preds_denorm, targets_denorm))
    nrmse = rmse / y_std_cpu
    corr = torch.corrcoef(torch.stack([preds_denorm, targets_denorm]))[0, 1].item()

    return avg_loss, rmse.item(), nrmse.item(), corr

# ============================================================
# Training Loop (Single Run)
# ============================================================
def train_model(data, n_epochs=50, loss_type="RMSE"):
    writer = SummaryWriter(log_dir=f"runs/challenge1")
    csv_path = f"results_challenge1.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_rmse", "val_nrmse", "val_corr","val_loss","LR"])

    # Split train/val (80/20)
    full_len = len(data)
    val_len = int(0.2 * full_len)
    train_len = full_len - val_len
    train_ds, val_ds = random_split(data, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    # Compute normalization stats
    all_targets = []
    for _, y, *_ in train_loader:
        all_targets.append(y.view(-1))
    y_all = torch.cat(all_targets).to(DEVICE, dtype=torch.float32)
    y_mean, y_std = y_all.mean(), y_all.std()
    print(f"Data normalization: mean={y_mean.item():.4f}, std={y_std.item():.4f}")

    backbone = EEGFeatureExtractor(channels=129, time_steps=200)
    model = EEG_CCD_Model(backbone).to(DEVICE)
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler = GradScaler()
    loss_fn = get_loss_fn("RMSE")

    best_corr = -1.0  # Save based on highest correlation score

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (X, y, *_) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")):
            X = X.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32).view(-1)
            y_norm = (y - y_mean) / y_std

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                y_pred = model(X).view(-1)
                loss = loss_fn(y_pred, y_norm)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Step [{batch_idx+1}/{len(train_loader)}] | Loss={loss.item():.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr'] 
        train_loss = running_loss / len(train_loader)
        val_loss, val_rmse, val_nrmse, val_corr = evaluate(model, val_loader, loss_fn, y_mean, y_std)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Val/RMSE", val_rmse, epoch)
        writer.add_scalar("Val/NRMSE", val_nrmse, epoch)
        writer.add_scalar("Val/Correlation", val_corr, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_rmse, val_nrmse, val_corr,val_loss,current_lr])

        print(f"Epoch {epoch} | TrainLoss={train_loss:.4f} | ValNRMSE={val_nrmse:.4f} | Corr={val_corr:.3f}")

        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), f"checkpoints/best_challenge1_epoch{epoch:03d}.pt")
            print(f" Saved best model: Epoch {epoch} (Corr={val_corr:.3f})")

    writer.close()


# ============================================================
# Main
# ============================================================
def main():
    print("Challenge 1 Training")
    os.makedirs("checkpoints", exist_ok=True)
    loader = EEGChallengeDataLoader()
    full_ds, _, _ = loader.load_challenge1(releases=RELEASES)
    
    train_model(full_ds.dataset, n_epochs=50, loss_type="MSELoss")
    print("Training complete and ready for research plotting!")

if __name__ == "__main__":
    main()
