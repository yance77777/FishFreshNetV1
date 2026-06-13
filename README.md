# FishFreshNetV1

Lightweight and explainable fish freshness assessment from fish-eye images.

FishFreshNetV1 is a PyTorch implementation of the framework described in **FishFreshNetV1: A Lightweight and Explainable Framework Based on Attention Mechanism for Fish Freshness Assessment**. It uses an ImageNet-pretrained EfficientNet-B0 backbone, a Convolutional Block Attention Module (CBAM), and a compact classifier head to classify fish-eye images into three freshness stages: **Highly Fresh**, **Fresh**, and **Not Fresh**.

> The paper manuscript is not included in this public repository. This repository only releases the implementation, reproducible training workflow, method overview, and non-manuscript figures.

<p align="center">
  <img src="figure/model_architecture.png" alt="FishFreshNetV1 architecture" width="92%">
</p>

## Highlights

| Item | Description |
| --- | --- |
| Task | Fine-grained fish freshness classification from fish-eye images |
| Backbone | ImageNet-pretrained EfficientNet-B0 |
| Attention | CBAM after the final EfficientNet-B0 feature extraction layer |
| Classifier | Adaptive average pooling, dropout, and a linear layer |
| Input size | 224 x 224 RGB images |
| Classes | Highly Fresh, Fresh, Not Fresh |
| Split | 70% train, 20% validation, 10% test |
| Repeated runs | 5 independent runs with seeds starting from 42 |
| Reported MFED accuracy | 99.23% |
| Reported complexity | 4.22M parameters, 0.41G FLOPs |

## Dataset

The experiments use the **Multistage Fish Eyes Dataset (MFED)**, a fish-eye image dataset constructed for three-stage freshness assessment.

| Item | MFED details |
| --- | --- |
| Total images | 4,800 |
| Classes | Highly Fresh, Fresh, Not Fresh |
| Class balance | 1,600 images per class |
| Fish species | Rice flower fish and crucian carp |
| Storage condition | 4 degrees C over six days |
| Acquisition settings | Four lighting/background settings, left and right eyes, five shooting angles |
| Dataset page | https://data.mendeley.com/datasets/67nmx3mhwh/2 |
| DOI | `10.17632/67nmx3mhwh.2` |
| License | CC BY 4.0 |

<p align="center">
  <img src="figure/MFED_dataset.png" alt="MFED image acquisition process" width="92%">
</p>

Expected dataset layout:

```text
MFED/
  Highly Fresh/
  Fresh/
  Not Fresh/
```

## Method

FishFreshNetV1 removes the original EfficientNet-B0 classification head and adds a custom classification block for the three freshness categories. CBAM is placed after the last EfficientNet-B0 feature extraction layer with 1280 input channels, helping the network emphasize biologically meaningful regions such as the pupil and cornea while suppressing background interference from lighting, scales, and acquisition surfaces.

Training uses 224 x 224 resized images, random horizontal flipping, random rotation within 15 degrees, ImageNet normalization, Adam optimization, an initial learning rate of `1e-4`, batch size `64`, and a ReduceLROnPlateau scheduler.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train FishFreshNetV1:

```bash
python FishFreshNetV1.py --data-dir /path/to/MFED
```

Useful options:

```bash
python FishFreshNetV1.py \
  --data-dir /path/to/MFED \
  --output-dir runs/fishfreshnet_v1 \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --runs 5
```

The training script saves per-run checkpoints, learning curves, confusion matrices, and CSV metrics under `runs/fishfreshnet_v1/`. Model weights and run outputs are intentionally ignored by git.

## Results Reported In The Paper

| Model | Params / M | FLOPs / G | MFED Acc / % | MFED Pr / % | MFED Re / % | MFED F1 / % | FFE Acc / % | FFE Pr / % | FFE Re / % | FFE F1 / % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGG16 | 134.27 | 15.47 | 98.08 | 98.03 | 98.13 | 98.07 | 77.40 | 76.72 | 76.51 | 76.52 |
| ResNet18 | 11.18 | 1.82 | 98.67 | 98.66 | 98.67 | 98.66 | 79.36 | 78.64 | 78.74 | 78.66 |
| MobileNetV2 | 2.23 | 0.33 | 98.54 | 98.54 | 98.54 | 98.53 | 79.59 | 79.02 | 78.59 | 78.65 |
| EfficientNet-B0 | 4.01 | 0.41 | 98.96 | 99.00 | 98.91 | 98.95 | 81.64 | 81.00 | 80.88 | 80.90 |
| **FishFreshNetV1** | **4.22** | **0.41** | **99.23** | **99.20** | **99.22** | **99.21** | **81.78** | **81.45** | **81.25** | **81.20** |

## Explainability

Grad-CAM is used to visualize the decision-making process of FishFreshNetV1. In successful cases, the strongest activation usually appears around the pupil and cornea, indicating that the model uses biologically relevant freshness cues. In failed cases, high-activation regions may shift toward specular highlights or bright scales near the eye, suggesting that strong reflections can interfere with model attention.

<p align="center">
  <img src="figure/Grad-CAM_example.png" alt="Grad-CAM correct and failure cases" width="92%">
</p>

## Repository Layout

```text
FishFreshNetV1.py          # Training entry point
fishfreshnet/
  data.py                  # MFED transforms, class order, split and dataloaders
  models.py                # EfficientNet-B0 + CBAM model definition
  train.py                 # Training, evaluation, metrics and plots
figure/
  model_architecture.png   # Model architecture figure
  MFED_dataset.png         # Dataset acquisition figure
  Grad-CAM_example.png     # Summary Grad-CAM figure with correct and failure cases
requirements.txt
```

## Public Release Notes

The following files are intentionally not included in the public repository:

- Paper manuscript files such as `FishFreshNetV1.docx`
- Trained weights such as `*.pth`, `*.pt`, and `*.onnx`
- Training outputs under `runs/`
- Local datasets and generated experiment results
