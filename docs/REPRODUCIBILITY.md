# Reproducibility Guidelines

This document summarizes the configuration needed to reproduce the HCFFormer experiments.

## 1. Software Environment

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0 |
| CUDA | 11.7 |
| cuDNN | 8.x |
| NumPy | 1.24+ |
| OpenCV | 4.7+ |
| SciPy | 1.10+ |

Use a dedicated Conda environment, Python virtual environment, or Docker image to avoid dependency conflicts.

## 2. Hardware Configuration

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GB VRAM |
| CPU | Intel Xeon Silver 4314 |
| RAM | 128 GB |

Minimum recommended GPU memory: 12 GB VRAM.

## 3. Deterministic Training

Use the following seed configuration before training:

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

Fixed random seed: `42`.

## 4. Dataset Structure

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

## 5. Dataset Split Strategy

| Split | Ratio |
|---|---:|
| Training | 70% |
| Validation | 10% |
| Testing | 20% |

The same split files should be reused across experiments.

## 6. Preprocessing

- Min-max normalization to `[0, 1]`
- Sliding-window patch extraction
- Patch size: `128 x 128`

## 7. Training Hyperparameters

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 8 |
| Dropout | 0.1 |
| Gradient clipping | 1.0 |
| Scheduler | Cosine annealing |

Training epochs:

| Dataset | Epochs |
|---|---:|
| Texas | 200 |
| California | 250 |
| Shuguang | 300 |

## 8. Transformer Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Attention heads | 8 |
| MLP expansion ratio | 4 |
| Patch size | 4 x 4 |

## 9. Graph Aggregation

The MSGA module uses:

- Cosine similarity adjacency construction
- Symmetric graph normalization

## 10. Loss Function

```text
L_total = L_change + lambda_1 L_KL + lambda_2 L_SSIM + lambda_3 L_cos + lambda_4 L_sem
```

Weighting coefficients:

```text
lambda_1 = lambda_2 = lambda_3 = lambda_4 = 0.5
```

## 11. Checkpoints

Recommended checkpoint directory:

```text
weights/
```

Save model checkpoints after training:

```python
torch.save(model.state_dict(), path)
```

## 12. Evaluation Metrics

- Overall Accuracy
- Precision
- Recall
- F1-score
- Kappa coefficient

Evaluation should be performed only on the test set.

## 13. FLOPs and Runtime Measurement

Computational complexity should be measured using:

- `256 x 256` input patches
- Same hardware configuration
- Single forward-pass measurement

Report:

- FLOPs
- Trainable parameters
- GPU memory
- Inference time
