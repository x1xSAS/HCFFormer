# HCFFormer

**HCFFormer** is a Hybrid CNN-Transformer Framework for Heterogeneous Remote Sensing Image Change Detection.

This repository provides a clean and simple public implementation package associated with the manuscript:

> **HCFFormer: A Hybrid CNN--Transformer Framework for Heterogeneous Remote Sensing Image Change Detection**

The repository is organized to support public access, reproducibility, and journal review requirements. It includes installation instructions, implementation details, reproducibility guidelines, model code, configuration files, and example training / inference commands.

---

## Repository Structure

```text
HCFFormer/
|
|-- configs/
|   |-- hcfformer.yaml
|
|-- datasets/
|   |-- README.md
|
|-- models/
|   |-- __init__.py
|   |-- hcfformer.py
|
|-- utils/
|   |-- __init__.py
|   |-- metrics.py
|   |-- seed.py
|
|-- docs/
|   |-- INSTALLATION.md
|   |-- IMPLEMENTATION_DETAILS.md
|   |-- REPRODUCIBILITY.md
|
|-- weights/
|   |-- README.md
|
|-- train.py
|-- test.py
|-- infer.py
|-- requirements.txt
|-- environment.yml
|-- CITATION.cff
|-- LICENSE
|-- README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/x1xsas/HCFFormer.git
cd HCFFormer
```

Create a Conda environment:

```bash
conda env create -f environment.yml
conda activate hcfformer
```

Or create a Python virtual environment:

```bash
python -m venv hcfformer_env
source hcfformer_env/bin/activate   # Linux/Mac
hcfformer_env\Scripts\activate      # Windows
pip install -r requirements.txt
```

For CUDA 11.7, install PyTorch using:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

For CPU-only execution:

```bash
pip install torch torchvision torchaudio
```

---

## Dataset Preparation

The benchmark datasets should be organized as follows:

```text
datasets/
|
|-- Texas/
|   |-- A/
|   |-- B/
|   |-- label/
|
|-- California/
|   |-- A/
|   |-- B/
|   |-- label/
|
|-- Shuguang/
    |-- A/
    |-- B/
    |-- label/
```

Where:

- `A/` contains images from the first modality or first time point.
- `B/` contains images from the second modality or second time point.
- `label/` contains binary change masks.

Dataset files are **not included** in this repository unless redistribution permission is available.

---

## Quick Start

Train with the default configuration:

```bash
python train.py --config configs/hcfformer.yaml
```

Train on a specific dataset:

```bash
python train.py --dataset Texas --batch_size 8 --epochs 200
```

Run evaluation:

```bash
python test.py --checkpoint weights/hcf_former.pth --dataset Texas
```

Run inference:

```bash
python infer.py --checkpoint weights/hcf_former.pth --input datasets/Texas
```

---

## Reproducibility Settings

The default reproducibility setup uses:

- Python 3.10
- PyTorch 2.0
- CUDA 11.7
- Fixed random seed: `42`
- Deterministic CuDNN enabled
- 70% / 10% / 20% train-validation-test split
- Patch size: `128 x 128`
- AdamW optimizer
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Batch size: `8`
- Dropout: `0.1`
- Gradient clipping: `1.0`
- Cosine annealing learning-rate scheduler

More details are provided in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Model Overview

HCFFormer consists of the following main components:

1. **Hybrid CNN-Transformer Encoder (HCTE)**
2. **Cross-Attention Fusion (CAF)**
3. **Multi-Scale Graph Aggregation (MSGA)**
4. **Change Token Detector (CTD)**
5. **Domain Consistency Regularizer (DCR)**

The model is designed for heterogeneous remote sensing change detection, where input images may come from different sensors or modalities.

---

## Evaluation Metrics

The following evaluation metrics are used:

- Overall Accuracy (OA)
- Precision
- Recall
- F1-score
- Kappa coefficient

---

## Important Note

The included code provides a clean, executable PyTorch implementation and repository scaffold. Before final publication, replace any placeholder or synthetic dataset logic with the exact dataset-loading, preprocessing, split files, and trained weights used in the manuscript experiments.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{HCFFormer2025,
  title={HCFFormer: A Hybrid CNN--Transformer Framework for Heterogeneous Remote Sensing Image Change Detection},
  author={Alsalman, Salman and Salhi, Amina},
  journal={International Journal of Remote Sensing},
  year={2026}
}
```

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
