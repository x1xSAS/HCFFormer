# Installation Instructions

The HCFFormer framework was implemented using Python 3.10, PyTorch 2.0, and CUDA 11.7.

## 1. Clone the Repository

```bash
git clone https://github.com/x1xSAS/HCFFormer.git
cd HCFFormer
```

## 2. Create a Virtual Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate hcfformer
```

Using `venv`:

```bash
python -m venv hcfformer_env
source hcfformer_env/bin/activate   # Linux/Mac
hcfformer_env\Scripts\activate      # Windows
pip install -r requirements.txt
```

## 3. Install PyTorch

CUDA 11.7:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

CPU-only version:

```bash
pip install torch torchvision torchaudio
```

## 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

## 5. Verify CUDA Availability

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected GPU output on the original experimental workstation:

```text
True
NVIDIA RTX A6000
```

## 6. Dataset Preparation

Organize datasets as:

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

## 7. Training

```bash
python train.py --config configs/hcfformer.yaml
```

Example:

```bash
python train.py --dataset Texas --batch_size 8 --epochs 200
```

## 8. Testing and Inference

```bash
python test.py --checkpoint weights/hcf_former.pth --dataset Texas
python infer.py --checkpoint weights/hcf_former.pth --input datasets/Texas
```

## 9. TensorBoard

```bash
tensorboard --logdir runs/
```

Then open:

```text
http://localhost:6006
```
