"""Evaluation script for HCFFormer.

This script runs an executable evaluation smoke test using synthetic data. Replace
SyntheticHCDDataset with the real test-set loader for final experiments.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import HCFFormer
from train import SyntheticHCDDataset
from utils import compute_binary_metrics, set_seed


@torch.no_grad()
def evaluate(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCFFormer().to(device)

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"Warning: checkpoint not found: {checkpoint}")
        print("Running evaluation with randomly initialized weights.")

    dataset = SyntheticHCDDataset(length=args.synthetic_length, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_true = []
    all_pred = []

    for optical, sar, mask in loader:
        optical = optical.to(device)
        sar = sar.to(device)
        pred, _, _, _ = model(optical, sar)
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_bin = (pred.cpu() > args.threshold).int().numpy()
        all_pred.append(pred_bin)
        all_true.append(mask.int().numpy())

    import numpy as np

    metrics = compute_binary_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HCFFormer")
    parser.add_argument("--checkpoint", default="weights/hcf_former.pth", help="Path to model checkpoint")
    parser.add_argument("--dataset", default="Texas", choices=["Texas", "California", "Shuguang"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--synthetic_length", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    evaluate(parser.parse_args())
