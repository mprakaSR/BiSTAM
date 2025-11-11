"""
verify_dataloader.py

Check that both Challenge 1 and Challenge 2 dataloaders work
and that EEG batches can be loaded from local data.
"""

from dataloader import EEGChallengeDataLoader
import torch

# Initialize unified loader
loader = EEGChallengeDataLoader()

# ================================================================
# ✅ Verify Challenge 1: Contrast Change Detection
# ================================================================
print("\n=== Verifying Challenge 1 (Contrast Change Detection) ===")
try:
    train_loader, val_loader, test_loader = loader.load_challenge1()

    # Take one batch from training loader
    X, y, crop_inds = next(iter(train_loader))

    print(f"✅ Challenge 1 batch loaded successfully!")
    print(f"X shape: {X.shape},  y shape: {y.shape}")
    print(f"Crop indices sample: {crop_inds[0]}")
    #print(f"Metadata example: {infos['subject'][0] if isinstance(infos, dict) else infos}")
    print(f"Data type: {X.dtype}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

except Exception as e:
    print("❌ Challenge 1 dataloader failed!")
    print(e)


# ================================================================
# ✅ Verify Challenge 2: Externalizing Factor
# ================================================================
print("\n=== Verifying Challenge 2 (Externalizing Factor) ===")
try:
    dataloader = loader.load_challenge2()

    # Take one batch
    X, y, crop_inds, infos = next(iter(dataloader))

    print(f"✅ Challenge 2 batch loaded successfully!")
    print(f"X shape: {X.shape},  y shape: {y.shape}")
    print(f"Crop indices sample: {crop_inds[0]}")
    print(f"Metadata example: {infos}")
    print(f"Data type: {X.dtype}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

except Exception as e:
    print("❌ Challenge 2 dataloader failed!")
    print(e)


# ================================================================
# ✅ Verify Local Data Availability
# ================================================================
print("\n=== Checking Local Dataset Directory ===")
from config import DATA_DIR
from pathlib import Path

data_dir = Path(DATA_DIR)
if data_dir.exists():
    print(f"📂 Data directory found: {data_dir.resolve()}")
    contents = [p.name for p in data_dir.iterdir()]
    print(f"Available contents: {contents}")
    
else:
    print(f"❌ Data directory {data_dir} does not exist. Please verify path in config.py.")
