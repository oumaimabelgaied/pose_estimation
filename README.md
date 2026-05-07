# pose_estimation

---

```markdown
# Sitting Posture Detection using YOLO Pose

A vision-based sitting posture detection system that combines YOLO Pose 
models for keypoint extraction with two classification strategies to 
detect and classify seated posture in real time using only a standard camera.

## Overview

Prolonged poor sitting posture is a leading cause of musculoskeletal 
disorders in office environments. This project builds an automated 
posture monitoring system that classifies sitting posture into three 
categories:

- ✅ **Good Posture**
- ⚠️ **Forward Lean**
- ⚠️ **Backward Lean**

Two YOLO models and two classification approaches are systematically 
compared to identify the best-performing pipeline.

---

## Results Summary

| Approach | Model | Classifier | Test Accuracy |
|----------|-------|-----------|--------------|
| A — Geometric Features | YOLOv8-Pose | Random Forest | 94.68% |
| A — Geometric Features | YOLOv8-Pose | Gradient Boosting | 92.55% |
| A — Geometric Features | YOLOv8-Pose | SVM | 88.30% |
| A — Geometric Features | YOLOv11-Pose | Random Forest | 92.55% |
| A — Geometric Features | YOLOv11-Pose | Gradient Boosting | 93.62% |
| A — Geometric Features | YOLOv11-Pose | SVM | 90.43% |
| **B — Fine-tuning** | **YOLOv8-cls** | **YOLO-classify** | **96.81%** |
| **B — Fine-tuning** | **YOLOv11-cls** | **YOLO-classify** | **97.87%** |

🏆 **Best model: YOLOv11n-cls — 97.87% test accuracy**

---

## Approaches

### Approach A — Geometric Features + Classical ML
- Extract 17 COCO keypoints using YOLOv8n-pose / YOLOv11n-pose
- Compute 7 geometric features from relevant keypoints:
  - `trunk_angle`, `neck_angle`, `head_angle`, `head_tilt`
  - `ear_shoulder_ratio`, `shoulder_hip_offset_x`, `hip_angle`
- Classify using Random Forest, Gradient Boosting, or SVM
- **Advantage**: fully interpretable, every decision traceable to 
  anatomically meaningful measurements

### Approach B — YOLO Classification Fine-tuning
- Fine-tune YOLOv8n-cls / YOLOv11n-cls directly on posture images
- End-to-end learning from raw pixels
- Transfer learning from ImageNet pretrained weights
- **Advantage**: highest accuracy, fastest inference

---

## Dataset

- **Source**: [Roboflow Universe — Sitting Posture v2](https://universe.roboflow.com/dataset-sqm0h/sitting-posture-ezkda)
- **License**: CC BY 4.0
- **Size**: 1,595 images (1,314 train / 187 valid / 94 test)
- **Format**: Multilabel classification
- **Classes**: goodposture, forwardbadposture, backwardbadposture

---

## Real-Time Demo

The best model (YOLOv11n-cls) is deployed in a live webcam pipeline:
- Colour-coded posture banner (green / red / orange)
- Confidence score bar
- Anatomical keypoint overlay with skeleton lines
- Session statistics (timeline + pie chart)

---

## Project Structure

```
├── sitting_posture_detection.ipynb   # Main notebook (Google Colab)
├── features_yolov8.csv               # Extracted geometric features (YOLOv8)
├── features_yolov11.csv              # Extracted geometric features (YOLOv11)
├── runs/
│   └── classify/
│       ├── yolov8_posture/           # YOLOv8-cls training outputs
│       └── yolo11_posture/           # YOLOv11-cls training outputs
└── README.md
```

---

## Setup

This project runs on **Google Colab** with GPU runtime.

### 1. Open the notebook in Colab
see notebook in main branch

### 2. Install dependencies
```python
!pip install ultralytics roboflow supervision -q
```

### 3. Download the dataset
```python

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("dataset-sqm0h").project("sitting-posture-ezkda")
version = project.version(2)
dataset = version.download("multiclass")
```

### 4. Run all cells in order

---

## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.12 |
| PyTorch | 2.10.0 |
| Ultralytics | 8.4.45 |
| scikit-learn | latest |
| Google Colab | Tesla T4 GPU |

---

## References

- Pawitra et al. (2025). *Detection of Ergonomic Sitting Postures in 
  Office Environments*. Journal of Image and Graphics, 13(6), 686–701.
- IEEE Systematic Review on Ergonomic Posture Risk Assessment through 
  Computer Vision and ML. DOI: 10.1109/ACCESS.2024.10772039
- [itakurah/Sitting-Posture-Detection-YOLOv5](https://github.com/itakurah/Sitting-Posture-Detection-YOLOv5) 
  — reference implementation

---

## Author

> Mini-project — Computer Vision & Deep Learning  
> Tools: Google Colab · Ultralytics · Roboflow · scikit-learn
```

