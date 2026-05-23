"""Inference script for HCFFormer.

The included inference example performs a smoke test with synthetic tensors. For
real data, replace the input-loading section with image loading from the dataset
folders.
"""

import argparse
from pathlib import Path

import torch

from models import HCFFormer
from utils import set_seed


@torch.no_grad()
def run_inference(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCFFormer().to(device)

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"Warning: checkpoint not found: {checkpoint}")
        print("Running inference with randomly initialized weights.")

    model.eval()

    # Smoke-test input. Replace this with real image loading for publication runs.
    optical = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    sar = torch.randn(1, 1, args.image_size, args.image_size, device=device)

    pred, _, _, _ = model(optical, sar)
    pred_bin = (pred > args.threshold).float()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prediction_tensor.pt"
    torch.save(pred_bin.cpu(), output_path)
    print(f"Prediction tensor saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HCFFormer inference")
    parser.add_argument("--checkpoint", default="weights/hcf_former.pth", help="Path to model checkpoint")
    parser.add_argument("--input", default="datasets/Texas", help="Input dataset folder")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    run_inference(parser.parse_args())
