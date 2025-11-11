# ============================================================
# train_challenge2.py
# ============================================================
# EEG Challenge 2: Subject-Invariant Externalizing Factor Regression
# - Uses adversarial subject loss during training
# - Evaluates only regression performance on val/test
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import EEGChallengeDataLoader
from models import EEGFeatureExtractor, EEG_Externalizing_Model
from config import DEVICE, LEARNING_RATE, N_EPOCHS, LOSS_FN, OPTIMIZER, LOG_INTERVAL
import os


# ============================================================
# Evaluation helper
# ============================================================
def evaluate(model, dataloader, loss_fn):
    """Evaluate regression performance only (no adversarial loss)."""
    model.eval()
    total_loss, preds, targets = 0, [], []

    with torch.no_grad():
        for X, y, *_ in dataloader:
            X, y = X.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
            y_pred, _ = model(X, lambd=0.0)  # no GRL during eval
            y_pred, y = y_pred.view(-1), y.view(-1)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * X.size(0)
            preds.append(y_pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).flatten()
    targets = torch.cat(targets).flatten()
    avg_loss = total_loss / len(dataloader.dataset)
    corr = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()

    return avg_loss, corr


# ============================================================
# Training loop
# ============================================================
def main():
    print("🚀 Starting Challenge 2: Externalizing Factor Regression (Adversarial)")

    # Setup
    writer = SummaryWriter(log_dir="runs/challenge2")
    os.makedirs("checkpoints", exist_ok=True)

    # 1️⃣ Load data
    loader = EEGChallengeDataLoader()
    dataloader = loader.load_challenge2()
    n_subjects = len(loader.subject_to_idx)
    print(f"🧠 Detected {n_subjects} unique subjects.")

    # 2️⃣ Model
    backbone = EEGFeatureExtractor(channels=129, time_steps=200)
    model = EEG_Externalizing_Model(backbone, num_subjects=n_subjects).to(DEVICE)

    # 3️⃣ Losses & Optimizers
    loss_task = getattr(nn, LOSS_FN)()
    loss_adv = nn.CrossEntropyLoss()

    optimizer_main = getattr(optim, OPTIMIZER)(
        list(model.backbone.parameters()) + list(model.reg_head.parameters()),
        lr=LEARNING_RATE
    )
    optimizer_disc = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE * 0.5)

    best_corr = -1.0

    # 4️⃣ Training loop
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_task_loss, total_adv_loss = 0, 0

        for batch_idx, (X, y, _, infos) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{N_EPOCHS}")):
            X, y = X.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
            subj_ids = torch.tensor(infos["subj_id"], dtype=torch.long, device=DEVICE)

            # Forward + backward
            optimizer_main.zero_grad()
            optimizer_disc.zero_grad()

            y_pred, subj_logits = model(X, lambd=0.5)
            y_pred, y = y_pred.view(-1), y.view(-1)
            L_task = loss_task(y_pred, y)
            L_adv = loss_adv(subj_logits, subj_ids)

            (L_task + L_adv).backward()
            optimizer_main.step()
            optimizer_disc.step()

            total_task_loss += L_task.item()
            total_adv_loss += L_adv.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Step [{batch_idx+1}/{len(dataloader)}] | "
                      f"L_task: {L_task.item():.4f} | L_adv: {L_adv.item():.4f}")

        # ✅ Validation (no adversarial loss, no subject IDs)
        val_loss, val_corr = evaluate(model, dataloader, loss_task)

        writer.add_scalar("Loss/train_task", total_task_loss / len(dataloader), epoch)
        writer.add_scalar("Loss/train_adv", total_adv_loss / len(dataloader), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Correlation/val", val_corr, epoch)

        print(f"✅ Epoch {epoch:03d} | TaskLoss={total_task_loss/len(dataloader):.4f} "
              f"| AdvLoss={total_adv_loss/len(dataloader):.4f} | Corr={val_corr:.3f}")

        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), "checkpoints/best_challenge2.pt")
            print(f"💾 Saved new best model (corr={best_corr:.3f})")

    # 5️⃣ Final Evaluation
    test_loss, test_corr = evaluate(model, dataloader, loss_task)
    print(f"\n🎯 Final Test Loss={test_loss:.4f} | Test Corr={test_corr:.3f}")

    writer.add_hparams(
        {"lr": LEARNING_RATE, "epochs": N_EPOCHS, "optimizer": OPTIMIZER},
        {"hparam/test_loss": test_loss, "hparam/test_corr": test_corr}
    )
    writer.close()


if __name__ == "__main__":
    main()
