# Implementation Details and Reproducibility Settings

The proposed HCFFormer framework was implemented using PyTorch 2.0 and trained on a workstation equipped with an NVIDIA RTX A6000 GPU with 48 GB VRAM, an Intel Xeon Silver 4314 CPU, and 128 GB RAM. All experiments were conducted under CUDA 11.7 with deterministic training enabled to ensure reproducibility.

The complete framework was trained in an end-to-end manner, allowing all modules to be jointly optimized during backpropagation, including:

- Hybrid CNN-Transformer Encoder (HCTE)
- Cross-Attention Fusion (CAF)
- Multi-Scale Graph Aggregation (MSGA)
- Change Token Detector (CTD)
- Domain Consistency Regularizer (DCR)

## Dataset Preparation and Splitting

Three benchmark heterogeneous remote sensing datasets were used for evaluation:

- Texas
- California
- Shuguang

Each dataset was divided into training, validation, and testing subsets using a 70%-10%-20% split strategy.

During preprocessing, all images were normalized to the range `[0, 1]` using min-max normalization. To improve training efficiency and preserve local contextual information, image patches of size `128 x 128` were extracted using a sliding-window strategy with overlap.

## Data Augmentation

To improve robustness and reduce overfitting, several data augmentation techniques were applied online during training, including:

- Random horizontal and vertical flipping
- Random rotation at 90, 180, and 270 degrees
- Gaussian noise perturbation
- Random brightness variation
- Random cropping

These augmentation strategies improve the generalization capability of the proposed model under varying sensor conditions and geometric distortions.

## Training Configuration

The network was optimized using the AdamW optimizer with an initial learning rate of:

```text
1 x 10^-4
```

A cosine annealing learning-rate scheduler was employed to gradually reduce the learning rate during optimization. The weight decay coefficient was set to:

```text
1 x 10^-5
```

The dropout rate within transformer blocks was fixed at:

```text
0.1
```

The mini-batch size was set to:

```text
8
```

The maximum number of training epochs was set to:

- Texas: 200 epochs
- California: 250 epochs
- Shuguang: 300 epochs

To improve optimization stability, gradient clipping with a maximum norm of `1.0` was applied during backpropagation. All experiments were initialized using a fixed random seed of `42` to ensure deterministic reproducibility.

## Transformer and Graph Configuration

Within the Hybrid CNN-Transformer Encoder (HCTE), the transformer embedding dimension was set to:

```text
128
```

The transformer encoder used:

- 8 attention heads
- 4x MLP expansion ratio
- GELU activation functions

The patch embedding layer employed a patch size of:

```text
4 x 4
```

For the proposed Multi-Scale Graph Aggregation (MSGA) module, graph adjacency matrices were dynamically constructed using cosine feature similarity between latent tokens. Symmetric graph normalization was applied before graph aggregation to stabilize message passing and prevent gradient explosion during training.

## Loss Functions and Optimization Objective

The overall optimization objective consists of:

- Binary cross-entropy loss for change prediction
- Feature-level domain consistency regularization
- Structural consistency constraints
- Semantic contrastive alignment

The total loss is formulated as:

```text
L_total = L_change + lambda_1 L_KL + lambda_2 L_SSIM + lambda_3 L_cos + lambda_4 L_sem
```

where:

- `L_change` denotes binary cross-entropy loss
- `L_KL` represents KL divergence
- `L_SSIM` corresponds to structural similarity regularization
- `L_cos` denotes cosine similarity loss
- `L_sem` indicates semantic contrastive alignment

The weighting coefficients were empirically selected through validation experiments:

```text
lambda_1 = lambda_2 = lambda_3 = lambda_4 = 0.5
```

## Inference and Evaluation

During inference, the predicted probability maps were resized to the original image resolution using bilinear interpolation. Binary change masks were generated using adaptive thresholding.

Evaluation was performed using:

- Overall Accuracy (OA)
- Precision
- Recall
- F1-score
- Kappa coefficient

## Code Availability and Reproducibility

To ensure transparency and facilitate future research, the PyTorch implementation of HCFFormer is publicly released through a GitHub repository. This repository includes installation instructions, implementation details, reproducibility guidelines, and executable code to support replication of the experimental results reported in the associated paper.
