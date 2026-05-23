"""HCFFormer model implementation.

This module contains a compact, executable PyTorch implementation of the
Hybrid CNN-Transformer Framework for heterogeneous remote sensing image change
detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, height, width


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HCTE(nn.Module):
    """Hybrid CNN-Transformer Encoder."""

    def __init__(self, in_channels: int, embed_dim: int = 128, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.patch_embed = PatchEmbedding(128, embed_dim, patch_size=4)
        self.transformer1 = TransformerBlock(embed_dim, heads=heads, dropout=dropout)
        self.transformer2 = TransformerBlock(embed_dim, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor):
        f1 = self.conv1(x)
        x = self.pool1(f1)
        f2 = self.conv2(x)
        x = self.pool2(f2)
        tokens, height, width = self.patch_embed(x)
        tokens = self.transformer1(tokens)
        tokens = self.transformer2(tokens)
        return f1, f2, tokens, height, width


class CAF(nn.Module):
    """Cross-Attention Fusion module."""

    def __init__(self, dim: int = 128, heads: int = 8) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1, _ = self.cross_attn(x1, x2, x2)
        out2, _ = self.cross_attn(x2, x1, x1)
        fused = (out1 + out2) / 2
        return self.norm(fused)


class MSGA(nn.Module):
    """Multi-Scale Graph Aggregation module."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        return torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        similarity = torch.bmm(x, x.transpose(1, 2))
        adj = F.softmax(similarity, dim=-1)
        adj = self.normalize_adj(adj)
        out = torch.bmm(adj, x)
        return self.fc(out)


class Decoder(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 64)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return x


class CTD(nn.Module):
    """Change Token Detector."""

    def __init__(self, in_channels: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x1 - x2)
        out = self.head(diff)
        return torch.sigmoid(out)


class DCRLoss(nn.Module):
    """Domain consistency regularizer.

    This implementation combines KL divergence, cosine consistency, an L1
    structural proxy, and a simple contrastive approximation.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        p1 = F.log_softmax(z1, dim=-1)
        p2 = F.softmax(z2, dim=-1)
        kl = F.kl_div(p1, p2, reduction="batchmean")
        cos = 1 - F.cosine_similarity(z1, z2, dim=-1).mean()
        ssim_proxy = F.l1_loss(z1, z2)
        contrast = ((z1 - z2) ** 2).mean()
        return 0.5 * kl + 0.5 * cos + 0.5 * ssim_proxy + 0.5 * contrast


class HCFFormer(nn.Module):
    def __init__(self, embed_dim: int = 128, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.opt_encoder = HCTE(3, embed_dim, heads=heads, dropout=dropout)
        self.sar_encoder = HCTE(1, embed_dim, heads=heads, dropout=dropout)
        self.caf = CAF(embed_dim, heads=heads)
        self.msga = MSGA(embed_dim)
        self.decoder_opt = Decoder(embed_dim)
        self.decoder_sar = Decoder(embed_dim)
        self.ctd = CTD(64)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        _, _, opt_tokens, h1, w1 = self.opt_encoder(optical)
        _, _, sar_tokens, h2, w2 = self.sar_encoder(sar)
        fused = self.caf(opt_tokens, sar_tokens)
        fused = self.msga(fused)
        dec_opt = self.decoder_opt(fused, h1, w1)
        dec_sar = self.decoder_sar(fused, h2, w2)
        change_map = self.ctd(dec_opt, dec_sar)
        return change_map, fused, opt_tokens, sar_tokens
