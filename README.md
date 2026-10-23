# Pothole Detection

Classifies road photos as pothole or plain road. Three models are trained and compared on the same test set:

- **Baseline CNN**: three convolutional blocks and a dense head, trained from scratch.
- **CNN with augmentation**: the same convolutional blocks with random flips, rotations, and zoom, a global average pooling head, class weighting, and early stopping.
- **MobileNetV2**: a frozen ImageNet-pretrained base with a small trainable head.

All images are resized to 128x128 and split 80/20 into train and test sets, stratified by class.

## Data

The dataset combines two public Kaggle datasets:

- [Pothole and Plain Road Images](https://www.kaggle.com/datasets/virenbr11/pothole-and-plain-rode-images): pothole and plain road images.
- [Pothole Image Data Set](https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset): pothole images only.

`prepare_data.py` merges them into one folder per class and removes exact duplicates by MD5 hash. The merged dataset has 919 pothole images and 375 plain road images.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download both datasets and extract them to:

```
data\raw\pothole-and-plain-road-images\
data\raw\pothole-image-dataset\
```

## Usage

```powershell
python prepare_data.py
python train.py
```

`train.py` trains all three models and saves training curves, confusion matrices, a model comparison chart, and `metrics.json` to `figures/`. The MobileNetV2 weights download automatically on the first run.
