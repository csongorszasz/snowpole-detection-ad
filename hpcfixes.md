# Disk quota exceeded

## Redirect Matplotlib cache
`export MPLCONFIGDIR=/work/csongors/.config/matplotlib`

## Redirect Ultralytics config/downloads
`export YOLO_CONFIG_DIR=/work/csongors/.config/ultralytics`

## Redirect PyTorch cache
`export PYTORCH_KERNEL_CACHE_PATH=/work/csongors/.cache/torch/kernels`


# CUDA driver incompatible with PyTorch

## Downgrade torch
```
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
