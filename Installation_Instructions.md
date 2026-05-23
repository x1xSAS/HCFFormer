# Installation Instructions

The proposed HCFFormer framework was implemented using Python 3.10, PyTorch 2.0, and CUDA 11.7. The following instructions describe the recommended environment setup and installation procedure for reproducing the experimental results.

## 1. Clone the Repository

```bash
git clone https://github.com/x1xsas/HCFFormer.git
cd HCFFormer
```

## 2. Create a Virtual Environment

It is recommended to use either Conda or Python virtual environments to avoid dependency conflicts.

### Using Conda

```bash
conda create -n hcfformer python=3.10
conda activate hcfformer
```

### Using venv

```bash
python -m venv hcfformer_env
source hcfformer_env/bin/activate      # Linux/Mac
hcfformer_env\Scripts\activate       # Windows
```

## 3. Install PyTorch

Install PyTorch with CUDA support compatible with your GPU configuration.

### CUDA 11.7

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### CPU-only Version

```bash
pip install torch torchvision torchaudio
```

## 4. Install Required Dependencies

Install the required Python packages using:

```bash
pip install numpy scipy opencv-python matplotlib scikit-image scikit-learn tqdm einops networkx albumentations tensorboard pyyaml
```

The main dependencies include:

- numpy
- scipy
- opencv-python
- matplotlib
- scikit-image
- scikit-learn
- tqdm
- einops
- networkx
- albumentations
- tensorboard
- pyyaml

## 5. Verify CUDA Availability

Run the following command to verify GPU availability:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output on the experimental workstation:

```text
True
NVIDIA RTX A6000
```

## 6. Dataset Preparation

Download the benchmark datasets:

- Texas
- California
- Shuguang

Dataset files are not included in this repository due to licensing and distribution restrictions. Users should obtain the datasets from their original sources as described in the associated paper and organize them according to the following directory structure:

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

## 7. Running the Code

Run the HCFFormer implementation using:

```bash
python Code.py
```

The provided code file contains a fully executable PyTorch implementation and uses a simple synthetic dataset class for demonstration and execution testing. For full experimental reproduction, replace the dataset loader with the prepared benchmark datasets described above.

## 8. TensorBoard Visualization

Training logs can be visualized using TensorBoard if logging is enabled:

```bash
tensorboard --logdir runs/
```

Open:

```text
http://localhost:6006
```

## 9. Reproducibility Settings

To ensure deterministic reproducibility:

- Random seed fixed to: 42
- Deterministic CuDNN enabled
- Fixed dataset split strategy
- Fixed hyperparameter configuration

## 10. Hardware Configuration

Recommended hardware:

- NVIDIA RTX A6000 GPU with 48 GB VRAM
- Intel Xeon CPU
- 128 GB RAM

Minimum recommended GPU memory:

```text
12 GB VRAM
```

## 11. Pretrained Models

Pretrained weights are not included directly in this simple repository. If available, they should be provided separately through an official release link or supplementary material.

## 12. Citation

If you use this repository, please cite:

```bibtex
@article{HCFFormer2026,
  title={HCFFormer: A Hybrid CNN--Transformer Framework for Heterogeneous Remote Sensing Image Change Detection},
  author={Alsalman, Salman and Salhi, Amina},
  journal={International Journal of Remote Sensing},
  year={2026}
}
```
