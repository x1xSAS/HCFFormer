# Implementation Details

HCFFormer was implemented using PyTorch 2.0 and trained on a workstation with the following hardware:

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GB VRAM |
| CPU | Intel Xeon Silver 4314 |
| RAM | 128 GB |
| CUDA | 11.7 |

All experiments were configured with deterministic training enabled.

## Dataset Preparation and Splitting

The framework uses three benchmark heterogeneous remote sensing datasets:

- Texas
- California
- Shuguang

Each dataset is divided using a fixed split strategy:

| Split | Ratio |
|---|---:|
| Training | 70% |
| Validation | 10% |
| Testing | 20% |

Images are normalized to `[0, 1]` using min-max normalization. Patches of size `128 x 128` are extracted using a sliding-window strategy with overlap.

## Data Augmentation

The following online data augmentation techniques are used during training:

- Random horizontal flipping
- Random vertical flipping
- Random rotation: 90, 180, and 270 degrees
- Gaussian noise perturbation
- Random brightness variation
- Random cropping

## Training Configuration

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
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

## Transformer and Graph Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Attention heads | 8 |
| MLP expansion ratio | 4 |
| Patch size | 4 x 4 |

The Multi-Scale Graph Aggregation module constructs adjacency matrices using cosine feature similarity between latent tokens and applies symmetric graph normalization to stabilize message passing.

## Loss Function

The total optimization objective is:

```text
L_total = L_change + lambda_1 L_KL + lambda_2 L_SSIM + lambda_3 L_cos + lambda_4 L_sem
```

Where:

- `L_change`: binary cross-entropy loss
- `L_KL`: KL divergence
- `L_SSIM`: structural similarity regularization
- `L_cos`: cosine similarity loss
- `L_sem`: semantic contrastive alignment

The weighting coefficients are:

```text
lambda_1 = lambda_2 = lambda_3 = lambda_4 = 0.5
```

## Inference and Evaluation

During inference:

1. Prediction probability maps are resized to the original image resolution using bilinear interpolation.
2. Binary change masks are generated using adaptive thresholding.
3. Evaluation is performed on the test set only.

Evaluation metrics:

- Overall Accuracy
- Precision
- Recall
- F1-score
- Kappa coefficient
