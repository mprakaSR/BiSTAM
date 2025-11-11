# ============================================================
# model.py
# ============================================================
# Unified EEG foundation model and task-specific heads
# for:
#   - Challenge 1: Cross-Task Transfer Learning (Response Time Regression)
#   - Challenge 2: Subject-Invariant Externalizing Factor Prediction
# ============================================================
"""| Component                 | Description                                            | Used In          |
| ------------------------- | ------------------------------------------------------ | ---------------- |
| `EEGFeatureExtractor`     | Foundation model (spatio-temporal EEG encoder)         | Both challenges  |
| `EEG_CCD_Model`           | Behavioral RT regression                               | Challenge 1      |
| `EEG_Externalizing_Model` | Psychopathology regression + subject adversary         | Challenge 2      |
| `SubjectDiscriminator`    | Learns subject-invariant EEG representation            | Challenge 2 only |
| `RegressionHead`          | Predicts continuous score (RT or externalizing factor) | Both             |
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. EEG Preprocessing Block
# ============================================================
class EEGPreprocessing(nn.Module):
    """Band-pass filter (FIR), Common Average Reference, and Z-score normalization."""
    def __init__(self, channels=129, fs=100, low_hz=0.50, high_hz=40.0, filter_order=129):
        super().__init__()
        self.channels = channels
        self.filter_order = filter_order

        t = torch.arange(-(filter_order // 2), (filter_order // 2) + 1)
        low = 2 * low_hz / fs
        high = 2 * high_hz / fs
        band = torch.sinc(high * t) - torch.sinc(low * t)
        window = torch.hamming_window(len(t))
        fir = band * window
        fir = fir / (fir.sum() + 1e-12)

        kernel = torch.zeros(channels, 1, len(fir))
        for ch in range(channels):
            kernel[ch, 0, :] = fir
        self.register_buffer("fir_kernel", kernel)

    def forward(self, x):
        # x: (B, C, T)
        pad = self.filter_order // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        x = F.conv1d(x, self.fir_kernel, groups=self.channels)
        x = x - x.mean(dim=1, keepdim=True)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        return (x - mean) / std



# ============================================================
# 2. Transformer Blocks (Spatial, Temporal)
# ============================================================
class EEGSpatialTransformer(nn.Module):
    """Transformer encoder over channels (spatial relationships)."""
    def __init__(self, channels, time_steps, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(time_steps, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, channels, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        # x: (B, C, T)
        x = self.proj(x) + self.pos_embed[:, :x.size(1), :]
        out = self.encoder(x)
        return out.mean(dim=1)


class EEGTemporalTransformer(nn.Module):
    """Transformer encoder over time (temporal dynamics)."""
    def __init__(self, channels, time_steps, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, time_steps, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.proj(x) + self.pos_embed[:, :x.size(1), :]
        out = self.encoder(x)
        return out.mean(dim=1)


# ============================================================
# 3. Cross-Attention Fusion Layer
# ============================================================
class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion between spatial and temporal features."""
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.attn_s2t = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.attn_t2s = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, s, t):
        s, t = s.unsqueeze(1), t.unsqueeze(1)
        s2t, _ = self.attn_s2t(s, t, t)
        t2s, _ = self.attn_t2s(t, s, s)
        fused = torch.cat([s2t.squeeze(1), t2s.squeeze(1)], dim=1)
        return self.fuse(fused)


# ============================================================
# 4. Shared EEGFeatureExtractor Backbone
# ============================================================
class EEGFeatureExtractor(nn.Module):
    """Foundation EEG model capturing spatial, temporal, and cross-modal dependencies."""
    def __init__(self, channels=129, time_steps=200, d_model=256, nhead=8, num_layers=3, fs=100):
        super().__init__()
        self.pre = EEGPreprocessing(channels, fs)
        self.spatial = EEGSpatialTransformer(channels, time_steps, d_model, nhead, num_layers)
        self.temporal = EEGTemporalTransformer(channels, time_steps, d_model, nhead, num_layers)
        self.fusion = CrossAttentionFusion(d_model, nhead // 2 if nhead >= 4 else 2)
        self.d_model = d_model

    def forward(self, x):
        x = self.pre(x)
        s = self.spatial(x)
        t = self.temporal(x)
        z = self.fusion(s, t)
        return z  # (B, d_model)


# ============================================================
# 5. Gradient Reversal + Subject Discriminator
# ============================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class SubjectDiscriminator(nn.Module):
    """  for domain (subject) invariance."""
    def __init__(self, in_dim=256, n_subjects=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_subjects)
        )
    def forward(self, z, lambd=1.0):
        return self.net(grad_reverse(z, lambd))


# ============================================================
# 6. Task Heads
# ============================================================
class RegressionHead(nn.Module):
    """Simple 2-layer regression head."""
    def __init__(self, in_dim=256, hidden=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z): return self.net(z)


# ============================================================
# 7. Challenge Models
# ============================================================
class EEG_CCD_Model(nn.Module):
    """Challenge 1: Cross-Task Transfer Learning (Response Time Regression)."""
    def __init__(self, backbone: EEGFeatureExtractor):
        super().__init__()
        self.backbone = backbone
        self.reg_head = RegressionHead(in_dim=backbone.d_model, out_dim=1)

    def forward(self, x):
        z = self.backbone(x)
        y_pred = self.reg_head(z)
        return y_pred


class EEG_Externalizing_Model(nn.Module):
    """Challenge 2: Subject-Invariant Externalizing Factor Prediction."""
    def __init__(self, backbone: EEGFeatureExtractor, num_subjects=50):
        super().__init__()
        self.backbone = backbone
        self.reg_head = RegressionHead(in_dim=backbone.d_model, out_dim=1)
        self.discriminator = SubjectDiscriminator(in_dim=backbone.d_model, n_subjects=num_subjects)

    def forward(self, x, lambd=0):
        z = self.backbone(x)
        y_pred = self.reg_head(z)
        subj_logits = self.discriminator(z, lambd)
        return y_pred, subj_logits
