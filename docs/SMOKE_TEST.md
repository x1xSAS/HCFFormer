# Smoke Test

The main configuration follows the manuscript setting. For a quick CPU smoke test, use a small synthetic input size and limit PyTorch CPU threads:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --epochs 1 --batch_size 1 --synthetic_length 1 --image_size 32
```

This command verifies that the model, optimizer, loss function, and checkpoint saving pipeline execute correctly. It is not intended to reproduce the reported paper results.

For manuscript reproducibility, use the real datasets, fixed split files, and the default configuration in:

```text
configs/hcfformer.yaml
```
