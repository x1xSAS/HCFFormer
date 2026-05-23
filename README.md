# HCFFormer

Official PyTorch implementation and reproducibility materials for **HCFFormer**, a hybrid CNN-Transformer framework for heterogeneous remote sensing image change detection.

## Repository Contents

This repository is intentionally kept very simple. It contains only five files:

```text
HCFFormer/
├── README.md
├── Installation_Instructions.md
├── Implementation_Details.md
├── Reproducibility_Guidelines.md
└── Code.py
```

## Files

- `README.md`: General project overview.
- `Installation_Instructions.md`: Environment setup, dependency installation, dataset organization, and execution instructions.
- `Implementation_Details.md`: Model implementation details, training configuration, loss functions, and evaluation metrics.
- `Reproducibility_Guidelines.md`: Settings required to reproduce the reported experiments.
- `Code.py`: Fully executable PyTorch implementation of HCFFormer.

## Dataset Notice

Dataset files are not included in this repository due to licensing and distribution restrictions. Users should obtain the datasets from their original sources as described in the associated paper and organize them according to the directory structure provided in `Installation_Instructions.md`.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/x1xsas/HCFFormer.git
cd HCFFormer
```

Create and activate a Python environment:

```bash
conda create -n hcfformer python=3.10
conda activate hcfformer
```

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy opencv-python matplotlib scikit-image scikit-learn tqdm einops networkx albumentations tensorboard pyyaml
```

Run the implementation:

```bash
python Code.py
```

## Authors

- Salman Alsalman
- Amina Salhi

## Citation

If you use this repository, please cite:

```bibtex
@article{HCFFormer2026,
  title={HCFFormer: A Hybrid CNN--Transformer Framework for Heterogeneous Remote Sensing Image Change Detection},
  author={Alsalman, Salman and Salhi, Amina},
  journal={International Journal of Remote Sensing},
  year={2026}
}
```

## Code Availability

The source code, installation instructions, implementation details, and reproducibility guidelines for HCFFormer are publicly available at:

```text
https://github.com/x1xsas/HCFFormer
```
