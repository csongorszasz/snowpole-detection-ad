# Setup Python venv without conda

## Base
```
python -m venv .venv
source .venv/bin/activate
pip install ultralytics codecarbon
```

## CUDA driver incompatible with PyTorch ("RuntimeError: CUDA driver error: invalid argument") -> Downgrade torch
```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## matplotlib patch for corrupt font
```
sed -i 's/font_manager.fontManager.addfont(str(f))/\
try:\n    font_manager.fontManager.addfont(str(f))\nexcept Exception:\n    pass/g' \
.venv/lib/python3.12/site-packages/ultralytics/utils/__init__.py
```

# Setup dataset
```
cp -r /datasets/tdt4265/Poles2025/ dataset/
python split_dataset.py
```

# Disk quota exceeded

## Redirect Matplotlib cache
`export MPLCONFIGDIR=/work/csongors/.config/matplotlib`

## Redirect Ultralytics config/downloads
`export YOLO_CONFIG_DIR=/work/csongors/.config/ultralytics`

## Redirect PyTorch cache
`export PYTORCH_KERNEL_CACHE_PATH=/work/csongors/.cache/torch/kernels`

