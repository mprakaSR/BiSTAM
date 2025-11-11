"""
config.py

Central configuration file for EEG Challenge training pipeline.
Contains all tunable parameters, paths, and environment settings.
"""

from pathlib import Path
import torch

# =====================
# DATA CONFIGURATION
# =====================

# Root data directory
DATA_DIR = Path(r"D:\Work_data\EEG_challenge")

# Dataset release(s) to include
RELEASES = ["R7","R8","R9","R10","R11"]# "R2","R3","R4","R6","R7","R8","R9","R10","R11"]# "R6","R7","R8","R9"

# EEG tasks to include
TASKS = ["contrastChangeDetection","RestingState","surroundSupp","ThePresent","seqLearning6target","symbolSearch"]
 #, "RestingState","surroundSupp","ThePresent","seqLearning6target","symbolSearch"]

# Excluded subjects (incomplete or corrupted data)
EXCLUDE_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
    "NDARBA381JGH",
]

# Sampling frequency
SFREQ = 100  # Hz

# Windowing parameters
WINDOW_SIZE_SECONDS = 2        # seconds
WINDOW_STRIDE_SAMPLES = 50     # overlap stride (in samples)
DROP_LAST_WINDOW = True

# =====================
# MODEL CONFIGURATION
# =====================

N_CHANNELS = 129
N_OUTPUTS = 1
N_TIMES = SFREQ * WINDOW_SIZE_SECONDS  # samples per window

# =====================
# TRAINING CONFIGURATION
# =====================

BATCH_SIZE = 1024
NUM_WORKERS = 0
LEARNING_RATE = 0.01 


# =====================
# HARDWARE SETTINGS
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    print("CUDA-enabled GPU detected. Using GPU for training.")
else:
    print("No GPU found. Using CPU (training may be slower).")

# =====================
# SAVE & LOGGING CONFIG
# =====================

MODEL_SAVE_PATH = Path("model_weights_challenge_2.pt")
LOG_INTERVAL = 50  # print loss every N steps
