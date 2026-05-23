import os
import math
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dataset
# ============================================================

class HCDDataset(Dataset):
    def __init__(self, length=100, image_size=128):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        optical = torch.randn(3, self.image_size, self.image_size)
        sar = torch.randn(1, self.image_size, self.image_size)
        mask = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        return optical, sar, mask

# ============================================================
# Basic CNN Block
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

# ============================================================
# Transformer Encoder Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads,
                                          dropout=dropout,
                                          batch_first=True)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x

# ============================================================
# Hybrid CNN–Transformer Encoder (HCTE)
# ============================================================

class HCTE(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.patch_embed = PatchEmbedding(128, embed_dim)

        self.transformer1 = TransformerBlock(embed_dim)
        self.transformer2 = TransformerBlock(embed_dim)

    def forward(self, x):
        f1 = self.conv1(x)
        x = self.pool1(f1)

        f2 = self.conv2(x)
        x = self.pool2(f2)

        tokens, H, W = self.patch_embed(x)

        tokens = self.transformer1(tokens)
        tokens = self.transformer2(tokens)

        return f1, f2, tokens, H, W

# ============================================================
# Cross-Attention Fusion (CAF)
# ============================================================

class CAF(nn.Module):
    def __init__(self, dim=128, heads=8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        out1, _ = self.cross_attn(x1, x2, x2)
        out2, _ = self.cross_attn(x2, x1, x1)

        fused = (out1 + out2) / 2
        fused = self.norm(fused)

        return fused

# ============================================================
# Multi-Scale Graph Aggregation (MSGA)
# ============================================================

class MSGA(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.fc = nn.Linear(dim, dim)

    def normalize_adj(self, adj):
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        return torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def forward(self, x):
        similarity = torch.bmm(x, x.transpose(1, 2))
        adj = F.softmax(similarity, dim=-1)

        adj = self.normalize_adj(adj)

        out = torch.bmm(adj, x)
        out = self.fc(out)

        return out

# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(embed_dim, 128, 2, stride=2)
        self.conv1 = ConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = ConvBlock(64, 64)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.up1(x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.conv2(x)

        return x

# ============================================================
# Change Token Detector (CTD)
# ============================================================

class CTD(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        out = self.head(diff)
        return torch.sigmoid(out)

# ============================================================
# Domain Consistency Regularizer (DCR)
# ============================================================

class DCRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        # KL divergence
        p1 = F.log_softmax(z1, dim=-1)
        p2 = F.softmax(z2, dim=-1)

        kl = F.kl_div(p1, p2, reduction='batchmean')

        # Cosine similarity
        cos = 1 - F.cosine_similarity(z1, z2, dim=-1).mean()

        # Structural consistency (L1 proxy)
        ssim_proxy = F.l1_loss(z1, z2)

        # Semantic contrastive approximation
        contrast = ((z1 - z2) ** 2).mean()

        total = 0.5 * kl + 0.5 * cos + 0.5 * ssim_proxy + 0.5 * contrast

        return total

# ============================================================
# Full HCFFormer Model
# ============================================================

class HCFFormer(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.opt_encoder = HCTE(3, embed_dim)
        self.sar_encoder = HCTE(1, embed_dim)

        self.caf = CAF(embed_dim)
        self.msga = MSGA(embed_dim)

        self.decoder_opt = Decoder(embed_dim)
        self.decoder_sar = Decoder(embed_dim)

        self.ctd = CTD(64)

    def forward(self, optical, sar):
        _, _, opt_tokens, H1, W1 = self.opt_encoder(optical)
        _, _, sar_tokens, H2, W2 = self.sar_encoder(sar)

        fused = self.caf(opt_tokens, sar_tokens)
        fused = self.msga(fused)

        dec_opt = self.decoder_opt(fused, H1, W1)
        dec_sar = self.decoder_sar(fused, H2, W2)

        change_map = self.ctd(dec_opt, dec_sar)

        return change_map, fused, opt_tokens, sar_tokens

# ============================================================
# Training Loop
# ============================================================


def train_model():
    dataset = HCDDataset(length=200)

    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0)

    model = HCFFormer().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4,
                                  weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    criterion = nn.BCELoss()
    dcr_loss_fn = DCRLoss()

    epochs = 5

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for optical, sar, mask in loader:
            optical = optical.to(DEVICE)
            sar = sar.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            pred, fused, opt_tokens, sar_tokens = model(optical, sar)

            pred = F.interpolate(pred,
                                 size=mask.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False)

            bce = criterion(pred, mask)
            dcr = dcr_loss_fn(opt_tokens, sar_tokens)

            loss = bce + 0.5 * dcr

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

    os.makedirs("weights", exist_ok=True)

    torch.save(model.state_dict(),
               "weights/hcf_former.pth")

    print("Training complete.")

# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def inference(model, optical, sar):
    model.eval()

    optical = optical.to(DEVICE)
    sar = sar.to(DEVICE)

    pred, _, _, _ = model(optical, sar)

    pred = (pred > 0.5).float()

    return pred

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()
