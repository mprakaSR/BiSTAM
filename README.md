# BiSTAM: Bidirectional Spatio-Temporal Attention Model for EEG Feature Decoding

BiSTAM (Bidirectional Spatio-Temporal Attention Model) is a transformer-based EEG foundation model designed for cross-task transfer learning and subject-invariant EEG decoding.
Developed for the [NeurIPS 2025 EEG Foundation Challenge](https://eeg2025.github.io/), BiSTAM learns generalizable spatio-temporal representations from raw EEG signals to predict both behavioral response time (RT) and psychopathology scores (Externalizing factor).

## Overview

BiSTAM combines dual-stream spatial and temporal Transformers with a bidirectional cross-attention fusion layer to jointly capture inter-channel and intra-temporal dependencies in EEG.
An adversarial discriminator with a Gradient Reversal Layer (GRL) enables subject-invariant representation learning, enhancing cross-subject generalization.

<p align="center"> <img src="RAW EEG.png" alt="BiSTAM Overview" width="800"> </p>

## Model Architecture

### Key Components:
 **Preprocessing**: FIR band-pass filter (0.5–40 Hz), Common Average Referencing (CAR), and channel-wise z-score normalization.
 
**EEG Spatial Transformer:** Models spatial (inter-channel) dependencies via multi-head self-attention.

**EEG Temporal Transformer**: Captures temporal (intra-signal) dependencies across time steps.

**Cross-Attention Fusion:** Merges spatial and temporal embeddings into a shared latent vector.

**Adversarial Subject Discriminator**: Uses a GRL to enforce subject-invariant EEG representations.

The total training loss combines regression and adversarial components:
L_total = L_reg + α × L_adv

# Challenge Tasks
| Challenge       | Description                          | Target              | Loss Function                     |
| --------------- | ------------------------------------ | ------------------- | --------------------------------- |
| **Challenge 1** | Reaction Time Regression             | Behavioral RT       | Mean Squared Error (MSE)          |
| **Challenge 2** | Subject-Invariant EEG Representation | Externalizing Score | MSE + Cross-Entropy (Adversarial) |



# Results
| Challenge             | Dataset            | Metric | Avg. RMSE (R1–R6) | Normalized RMSE |
| --------------------- | ------------------ | ------ | ----------------- | --------------- |
| **Challenge 1 (RT)**  | HBN-EEG CCD        | RMSE   | **0.4067**        | 1.0000          |
| **Challenge 2 (EXT)** | HBN-EEG Multi-task | RMSE   | **0.7588**        | 1.0014          |


## Training Configuration

- Optimizer: AdamW
- Learning Rate: 5 × 10⁻³ → 1 × 10⁻⁶ (Cosine Annealing, 50 epochs)
- Batch Size: 1024
- Hardware: NVIDIA RTX A4000
- Training Data: Releases R7–R11
- Testing Data: Releases R1–R6



This work builds upon:

HBN-EEG Dataset — Shirazi et al., 2024

EEGformer — Wan et al., 2023

EEGPT — Wang et al., 2024

Gradient Reversal Layer (GRL) — Ganin et al., 2016