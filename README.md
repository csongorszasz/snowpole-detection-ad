# Snow Pole Detection for Autonomous Driving

### Demo

TODO

### Problem statement

There has been major advancements in Autonomous driving in recent years, attributed to computer
vision and Deep Learning (CV & DL). However, such models are mostly focused on driving in ideal
conditions, and thus struggle with challenging conditions like snowy roads. One way to localize the
road in winter time is by relying on the location of snow poles, which are typically erected on either side
of the road in areas prone to snow in the winter. The task here is to perform real time object detection
of snow poles, in order to further develop AD capabilities in winter conditions.

### Dataset

Data captured in Norway, Trøndelag region.

### Reproduce

1. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate snowpole-detection
```

2. Make sure your dataset is available under `dataset/Poles2025/...` and that paths in the selected config files are valid for your machine.

3. Train a model (example):

```bash
python train.py -c configs/train/001.yaml
```

4. Run prediction with a trained checkpoint (example):

```bash
python predict.py -c configs/predict/roadpoles_v1_013.yaml
```

5. Check outputs:

- Training runs are saved in `runs/detect/train*/`
- Prediction outputs are saved in `runs/detect/predict*/`
- Label txt files are zipped to `runs/detect/predict*/labels.zip`

