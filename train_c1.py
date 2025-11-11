# ============================================================
# train_challenge1.py
# ============================================================
# EEG Challenge 1: Reaction-Time Regression with TensorBoard
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import EEGChallengeDataLoader
from models import EEGFeatureExtractor, EEG_CCD_Model
from config import (
    DEVICE, LEARNING_RATE, N_EPOCHS, LOSS_FN, OPTIMIZER,
    MODEL_SAVE_PATH, LOG_INTERVAL
)
import os

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss, preds, targets = 0, [], []

    with torch.no_grad():
        for X, y, *_ in dataloader:
            X, y = X.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
            y_pred = model(X).squeeze()
            y = y.squeeze()
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * X.size(0)
            preds.append(y_pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    avg_loss = total_loss / len(dataloader.dataset)
    corr = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()

    return avg_loss, corr


def main():
    print("🚀 Starting Challenge 1: Reaction-Time Regression")

    # TensorBoard setup
    writer = SummaryWriter(log_dir="runs/challenge1")

    # 1️⃣ Load Data
    loader = EEGChallengeDataLoader()
    train_loader, val_loader, test_loader = loader.load_challenge1()

    # 2️⃣ Model
    backbone = EEGFeatureExtractor(channels=129, time_steps=200)
    model = EEG_CCD_Model(backbone).to(DEVICE)

    # 3️⃣ Optimizer + Loss
    loss_fn = getattr(nn, LOSS_FN)()
    optimizer = getattr(optim, OPTIMIZER)(model.parameters(), lr=LEARNING_RATE)

    best_corr = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    # 4️⃣ Training Loop
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0

        for batch_idx, (X, y, *_ ) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")):
            X, y = X.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(X).squeeze()
            y = y.squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        val_loss, val_corr = evaluate(model, val_loader, loss_fn)

        # Log to TensorBoard
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Correlation/val", val_corr, epoch)

        print(f"✅ Epoch {epoch}: TrainLoss={running_loss/len(train_loader):.4f} | "
              f"ValLoss={val_loss:.4f} | Corr={val_corr:.3f}")

        # Save checkpoint if improved
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), f"checkpoints/best_challenge1.pt")
            print(f"💾 Saved new best model (corr={best_corr:.3f})")

    # 5️⃣ Final Test Evaluation
    test_loss, test_corr = evaluate(model, test_loader, loss_fn)
    print(f"\n🎯 Test Loss={test_loss:.4f} | Test Corr={test_corr:.3f}")
    writer.add_hparams(
        {"lr": LEARNING_RATE, "epochs": N_EPOCHS, "optimizer": OPTIMIZER},
        {"hparam/test_loss": test_loss, "hparam/test_corr": test_corr}
    )

    writer.close()


if __name__ == "__main__":
    main()
