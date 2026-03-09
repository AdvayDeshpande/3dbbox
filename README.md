# 3dbbox
3D Bounding Box 

# 3D Bounding Box Prediction — Multi-Modal Fusion (RGB + Point Cloud + Mask)

This repository contains a **multi-modal deep learning pipeline** designed to predict **normalized 3D bounding box coordinates** (8 corners, shape `(8, 3)`) from a combination of 2D visual data and 3D geometric data. By fusing **RGB images**, **dense point cloud tensors**, and **object segmentation masks**, the model learns to localize objects in 3D space with high precision.

---

## Project Overview

3D object localization benefits from multiple data sources. This project implements a custom fusion architecture:

- **Visual Context**: A **Vision Transformer (ViT)** extracts high-level features from RGB images.
- **Geometric Structure**: A **ResNet-18** backbone processes dense point cloud tensors to capture spatial depth information.
- **Object Guidance**: A dedicated **CNN Mask Encoder** processes segmentation masks, allowing the model to focus on specific instances within a scene.
- **Coordinate Regression**: Features from all three encoders are fused through an MLP to regress the **normalized 3D coordinates** of the bounding box corners.

---

## Key Technical Features

- **Multi-Modal Fusion**: Late-fusion architecture that concatenates embeddings from a Transformer (ViT) and CNNs (ResNet-18 + Mask CNN).
- **Normalized Coordinate Regression**: The model predicts coordinates in a normalized range, ensuring stable training and scale-invariance across scenes.
- **Robust Loss Function**: Uses **Smooth L1 (Huber) Loss** to handle outliers in 3D coordinate predictions.
- **End-to-End Pipeline**: Custom data loading, modular training with checkpointing, and a testing suite with 2D/3D visualization.

---

## Model Architecture

Implemented in `model_new.py`:

| Component | Backbone | Output Dim |
|---|---|---|
| Image Encoder | ViT (`google/vit-base-patch16-224-in21k`) | 256-d |
| Point Cloud Encoder | ResNet-18 (ImageNet weights) | 256-d |
| Mask Encoder | Lightweight CNN + pooling | 128-d |
| Fusion Head | MLP → bbox regression | `(B, 8, 3)` |

---

## Dataset Format

The dataloader (`dataloader_new.py`) expects a directory structured as:

```
root_dir/
  sample_000/
    rgb.jpg
    pc.npy
    mask.npy
    bbox3d.npy
  sample_001/
    ...
```

- `mask.npy` — multiple instance masks (one per object)
- `bbox3d.npy` — corresponding normalized 3D bounding boxes `(N, 8, 3)`
- Each `(mask[i], bbox[i])` pair becomes a separate training sample

> **Note**: The dataset used in this project is private and not included in this repository.

---

## Training

```bash
python train_new.py
```

- **Optimizer**: Adam
- **Loss**: SmoothL1Loss (Huber)
- **Checkpoints**: saved each epoch under `checkpoints/`
- **Logs**: written to `training_log.txt`

---

## Testing & Visualization

```bash
python test_new.py
```

- Evaluates model with SmoothL1 loss on the test split
- Projects predicted and GT 3D corners to 2D image space (GT in **green**, prediction in **red**)
- Optional 3D visualization with **Open3D**
- Outputs saved to `test_results/` and `test_results.log`

---

## Skills Demonstrated

### Deep Learning & Multi-Modal AI
- Hybrid architecture design combining **Vision Transformers** and **CNNs**
- Sensor fusion across heterogeneous modalities (RGB, point cloud, mask)
- 3D coordinate regression with multi-dimensional output `(B, 8, 3)`
- Transfer learning with pretrained ViT and ResNet-18 backbones

### Computer Vision & 3D Geometry
- 8-corner 3D bounding box representation and normalization
- Point cloud tensor processing and depth normalization
- 3D-to-2D projection for visualization and debugging
- Instance-level supervision using segmentation masks

### Frameworks & Libraries
- **PyTorch** — custom `Dataset`, `DataLoader`, training/eval loops
- **HuggingFace Transformers** — ViT integration
- **OpenCV** — image I/O and 2D visualization
- **Open3D** — 3D point cloud and bounding box visualization
- **NumPy** — array manipulation and preprocessing

---

## Repository Structure

```
├── model_new.py          # Multi-modal model architecture
├── dataloader_new.py     # Dataset class and dataloaders
├── train_new.py          # Training loop + checkpointing
├── test_new.py           # Evaluation + visualization
├── training_log.txt      # Training logs
└── test_results.log      # Test logs
```

---

## Future Work

- Add **3D IoU** as a primary evaluation metric
- Explore lighter backbones (e.g., MobileNetV3) for edge deployment
- Extend to video sequences for temporal 3D tracking

---

## License

MIT License
