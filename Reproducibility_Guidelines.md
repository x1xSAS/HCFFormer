# Reproducibility Guidelines

The following guidelines are provided to ensure full reproducibility of the experimental results reported for the proposed HCFFormer framework. All implementation details, training settings, and evaluation procedures have been standardized to facilitate fair comparison and independent verification by future researchers.

## 1. Software Environment

The proposed framework was implemented using the following software configuration:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0 |
| CUDA | 11.7 |
| cuDNN | 8.x |
| NumPy | 1.24+ |
| OpenCV | 4.7+ |
| SciPy | 1.10+ |

To avoid dependency conflicts, users are encouraged to create a dedicated virtual environment using one of the following:

- Conda
- venv
- Docker

## 2. Hardware Configuration

All experiments were conducted on the following hardware platform:

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GB VRAM |
| CPU | Intel Xeon Silver 4314 |
| RAM | 128 GB |

Minimum recommended GPU memory:

```text
12 GB VRAM
```

## 3. Deterministic Training Configuration

To ensure deterministic reproducibility, all random generators must be fixed before training.

The following settings were used:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Fixed random seed:

```text
42
```

## 4. Dataset Preparation

The following benchmark datasets were used:

- Texas
- California
- Shuguang

All datasets should be organized using the following structure:

```text
datasets/
│
├── Texas/
│   ├── A/
│   ├── B/
│   └── label/
│
├── California/
│   ├── A/
│   ├── B/
│   └── label/
│
└── Shuguang/
    ├── A/
    ├── B/
    └── label/
```

Dataset files are not included in this repository due to licensing and distribution restrictions. Users should obtain the datasets from their original sources as described in the associated paper.

## 5. Dataset Split Strategy

All datasets were divided using the same strategy:

| Split | Ratio |
|---|---|
| Training | 70% |
| Validation | 10% |
| Testing | 20% |

The same split files should be reused for all experiments to guarantee fair comparison.

## 6. Data Preprocessing

The following preprocessing operations were applied:

- Min-max normalization to `[0, 1]`
- Patch extraction using sliding windows
- Patch size: `128 x 128`

## 7. Data Augmentation

Online augmentation techniques include:

- Random horizontal flip
- Random vertical flip
- Random rotation
- Gaussian noise perturbation
- Random cropping

These augmentations should remain identical across all experiments.

## 8. Training Configuration

The following hyperparameters were used during training:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1 x 10^-4 |
| Weight Decay | 1 x 10^-5 |
| Batch Size | 8 |
| Dropout | 0.1 |
| Gradient Clipping | 1.0 |
| Scheduler | Cosine Annealing |

Training epochs:

- Texas: 200
- California: 250
- Shuguang: 300

## 9. Transformer Configuration

The transformer encoder uses:

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| Attention Heads | 8 |
| MLP Expansion Ratio | 4 |
| Patch Size | 4 x 4 |

## 10. Graph Aggregation Configuration

The Multi-Scale Graph Aggregation (MSGA) module uses:

- Cosine similarity adjacency construction
- Symmetric graph normalization

This normalization is essential for stable graph optimization.

## 11. Loss Function Configuration

The total optimization objective is:

```text
L_total = L_change + lambda_1 L_KL + lambda_2 L_SSIM + lambda_3 L_cos + lambda_4 L_sem
```

Weighting coefficients:

```text
lambda_1 = lambda_2 = lambda_3 = lambda_4 = 0.5
```

## 12. Mixed Precision Training

To reduce GPU memory consumption and improve efficiency, mixed precision training is recommended:

```python
torch.cuda.amp.autocast()
```

## 13. Checkpoint Saving

Model checkpoints should be saved after each epoch:

```python
torch.save(model.state_dict(), path)
```

Recommended checkpoint filename:

```text
hcf_former.pth
```

## 14. Evaluation Metrics

The following metrics were used:

- Overall Accuracy (OA)
- Precision
- Recall
- F1-score
- Kappa coefficient

Evaluation should be performed only on the test set.

## 15. Inference Procedure

During inference:

- Predicted maps are resized to original resolution
- Adaptive thresholding is applied
- Binary masks are generated

## 16. FLOPs and Runtime Measurement

Computational complexity was evaluated using:

- `256 x 256` input patches
- Identical hardware settings
- Single forward-pass measurement

Metrics include:

- FLOPs
- Trainable parameters
- GPU memory
- Inference time

## 17. Recommended Repository Structure

This simplified repository uses only five files:

```text
HCFFormer/
├── README.md
├── Installation_Instructions.md
├── Implementation_Details.md
├── Reproducibility_Guidelines.md
└── Code.py
```

## 18. Pretrained Models

Pretrained weights are not included directly in this simple repository. If available, they should be provided separately through an official release link or supplementary material.

## 19. Logging and Visualization

TensorBoard logging is recommended if training logs are enabled:

```bash
tensorboard --logdir runs/
```
