"""Training script for HCFFormer.

The default dataset class below uses synthetic tensors so the repository can be
executed immediately by reviewers. For final reproducibility, replace
SyntheticHCDDataset with the exact dataset loader and split files used in the
paper experiments.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

from models import HCFFormer, DCRLoss
from utils import set_seed


class SyntheticHCDDataset(Dataset):
    """Small synthetic dataset used only as an executable smoke test."""

    def __init__(self, length: int = 100, image_size: int = 128) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        optical = torch.randn(3, self.image_size, self.image_size)
        sar = torch.randn(1, self.image_size, self.image_size)
        mask = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        return optical, sar, mask


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(args):
    config = load_config(args.config)
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = args.batch_size or int(config["training"]["batch_size"])
    dataset_name = args.dataset or config["data"]["dataset"]
    epochs_config = config["training"]["epochs"]
    epochs = args.epochs or int(epochs_config.get(dataset_name, 200))

    # Replace this with the real remote sensing dataset loader.
    image_size = args.image_size or int(config["data"]["patch_size"])
    dataset = SyntheticHCDDataset(length=args.synthetic_length, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = HCFFormer(
        embed_dim=int(config["model"]["embed_dim"]),
        heads=int(config["model"]["attention_heads"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()
    dcr_loss_fn = DCRLoss()
    grad_clip = float(config["training"]["gradient_clipping"])

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for optical, sar, mask in loop:
            optical = optical.to(device)
            sar = sar.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred, _, opt_tokens, sar_tokens = model(optical, sar)
            pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)

            bce = criterion(pred, mask)
            dcr = dcr_loss_fn(opt_tokens, sar_tokens)
            loss = bce + 0.5 * dcr

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss:.4f}")

    weights_dir = Path(config["paths"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "hcf_former.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete. Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HCFFormer")
    parser.add_argument("--config", default="configs/hcfformer.yaml", help="Path to YAML configuration file")
    parser.add_argument("--dataset", default=None, choices=["Texas", "California", "Shuguang"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--synthetic_length", type=int, default=100, help="Synthetic sample count for smoke testing")
    parser.add_argument("--image_size", type=int, default=None, help="Override synthetic image size for quick smoke tests")
    train(parser.parse_args())
