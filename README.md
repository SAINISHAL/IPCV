# Classroom Face Recognition System Using YOLO + Eigenfaces (PCA)

A production-grade, genuine **Eigenfaces-based face recognition system** designed for classroom attendance and identity verification. The system combines modern spatial face detection (**YOLOv8**) with classical orthogonal feature projection (**Principal Component Analysis / Eigenfaces**) and statistical pattern classification (**SVM, KNN, Euclidean Distance**).

---

## 1. Problem Statement & Motivation

In academic classroom environments, automated attendance tracking requires simultaneous multi-subject face detection in unconstrained group photographs, robust identity classification under lighting and pose variations, and strict unknown-face rejection (to prevent non-enrolled students or visitors from being falsely recognized).

Unlike black-box deep embeddings (FaceNet, ArcFace), this project adheres strictly to **authentic Eigenfaces theory (Turk & Pentland, 1991)**, demonstrating first-principles computer vision, covariance matrix decomposition, orthogonal basis projection, and reconstruction error analysis.

---

## 2. System Architecture

```text
Classroom-Eigenfaces-Recognition/
├── configs/
│   └── config.yaml                     # Central system configuration
├── data/
│   ├── raw/                            # Extracted student archives
│   ├── extracted_faces/                # Cropped face patches per student ID
│   ├── processed_faces/                # 100x100 CLAHE normalized face images
│   └── label.txt                       # 15 enrolled student roll numbers
├── models/
│   ├── yolo/                           # YOLOv8 face detection weights
│   ├── eigenfaces/                     # Trained PCA model (pca_model.pkl)
│   └── classifiers/                    # Trained SVM, KNN, Euclidean classifiers
├── src/
│   ├── data/                           # Parsers, extractors, validators, identity mappers
│   ├── detection/                      # YOLO detector and face extraction
│   ├── preprocessing/                  # CLAHE, quality filter, alignment, 100x100 normalization
│   ├── eigenfaces/                     # Pure NumPy Dual PCA, projection, reconstruction
│   ├── recognition/                    # Euclidean, KNN, SVM classifiers, unknown detector
│   ├── evaluation/                     # Metrics, confusion matrices, K-experiments
│   ├── webcam/                         # Real-time OpenCV video loop with temporal smoothing
│   └── utils/                          # Logger, config, drawing helpers
├── scripts/
│   ├── inspect_dataset.py              # Full dataset audit and figures generation
│   ├── prepare_dataset.py              # Face crop extraction and CLAHE processing
│   ├── train.py                        # Model training across candidate K values
│   ├── evaluate.py                     # Test-set benchmarking & report generation
│   └── run_webcam.py                   # High-FPS live webcam recognition
├── app/
│   └── app.py                          # 5-page interactive Streamlit dashboard
├── reports/
│   ├── figures/                        # High-resolution diagnostic plots & confusion matrices
│   ├── dataset_inspection_report.md    # Exhaustive dataset audit & mapping findings
│   ├── repository_comparison_report.md # Technical comparison of base repos
│   ├── model_evaluation_report.md      # Quantitative test-set benchmark
│   └── FINAL_PROJECT_REPORT.md         # Comprehensive project final deliverable
├── tests/                              # Pytest unit and integration test suite
└── main.py                             # Unified CLI entry point
```

---

## 3. Mathematical Foundations of Eigenfaces

### 3.1 Mean Face
Given $N$ training face images represented as column vectors $x_i \in \mathbb{R}^D$ where $D = W \times H = 100 \times 100 = 10,000$:

$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

### 3.2 Mean Centering
Each face vector is centered by subtracting the mean face:

$$a_i = x_i - \mu \quad \implies \quad A = [a_1, a_2, \dots, a_N] \in \mathbb{R}^{D \times N}$$

### 3.3 The Efficient Dual PCA Formulation
The standard covariance matrix $C = \frac{1}{N} A A^T$ has dimensionality $10,000 \times 10,000$ ($10^8$ elements), making direct decomposition computationally prohibitive.

Following Sirovich & Kirby (1987) and Turk & Pentland (1991), we construct the smaller sample-space Gram matrix:

$$L = A^T A \in \mathbb{R}^{N \times N}$$

Solving the $N \times N$ eigenvalue problem:

$$L v_i = \lambda_i v_i$$

Multiplying both sides by $A$:

$$A (A^T A v_i) = \lambda_i (A v_i) \implies (A A^T)(A v_i) = \lambda_i (A v_i)$$

Thus, the image-space eigenvectors $u_i \in \mathbb{R}^D$ (the **Eigenfaces**) are:

$$u_i = A v_i$$

Each eigenvector is normalized to unit Euclidean length:

$$u_i \leftarrow \frac{u_i}{\|u_i\|_2}$$

### 3.4 Subspace Projection & Reconstruction
Retaining the top $K$ eigenvectors $U_K = [u_1, u_2, \dots, u_K] \in \mathbb{R}^{D \times K}$, any face $x$ is projected into the compact weight vector $\omega \in \mathbb{R}^K$:

$$\omega = U_K^T (x - \mu)$$

The face can be reconstructed with low distortion:

$$\hat{x} = \mu + U_K \omega$$

---

## 4. Installation & Environment Setup

This repository contains the source code. Keep classroom photos, identity labels,
webcam enrollment crops, trained models, and generated reports local. Restore
those files under the `data/`, `models/`, and `reports/` paths used by the project
before running the corresponding features. To rebuild the dashboard's model from
local face clusters, run `python scripts/train_from_clusters.py` after preparing
`data/clusters.json`, `data/identity_mapping.json`, and their face crops.

For archive-based training and the diagnostic scripts, create a local
`data/dataset_profile.json` with a `verified_archives` array of archive folder
names that share the class mapping in `data/label.txt`. Its `samples` object
maps `medium`, `wide`, `comparison`, `mixed`, `angle`, `high_res`, and
`single_known` to the relevant photo paths relative to the project root.

### 4.1 Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows laptop (standard CPU supported, no GPU required)

### 4.2 Installation Commands (Windows PowerShell)

```powershell
# Navigate to project directory
cd D:\IpCv\Classroom-Eigenfaces-Recognition

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Execution Guide

You can run the entire pipeline with a single command via `main.py` or run individual stages:

### A. Full Automated Pipeline (Inspection -> Extraction -> Training -> Evaluation)
```powershell
python main.py --all
```

### B. Individual Script Execution

```powershell
# 1. Dataset Inspection & Audit
python scripts/inspect_dataset.py

# 2. Extract & Preprocess Face Crops
python scripts/prepare_dataset.py

# 3. Model Training & Validation Tuning
python scripts/train.py

# 4. Final Evaluation & Benchmarking
python scripts/evaluate.py
```

### C. Live Real-Time Webcam Face Recognition
```powershell
# Default webcam device (ID 0)
python scripts/run_webcam.py

# Specify alternate device ID (e.g. external USB camera)
python scripts/run_webcam.py --device 1
```
*Press `q` or `ESC` in the OpenCV camera window to terminate.*

### D. Photo Upload Dashboard
```powershell
python -m streamlit run app/app.py
```

Upload one or more portraits or group photos to see the recognized names and
face counts below each labeled photo. YOLOv8-Face and YuNet locate faces, and the
saved **Eigenfaces + SVM** model identifies them.

### Offline recognition studies

Run `python scripts/benchmark_recognition.py` from the project folder to compare
the preprocessing, augmentation, and grouped SVM/PCA configurations. It writes
the dataset audit, threshold trials, test metrics, and candidate models under
`reports/recognition_study/`. The study does not replace the active saved models.

Run `python scripts/retrain_augmented.py` for the extended augmentation and
lower-cutoff experiment. It generates six variants per eligible training crop,
keeps source photos grouped, and writes a cutoff comparison and staged models to
`reports/recognition_augmentation/`.

### E. Run Automated Pytest Test Suite
```powershell
python -m pytest tests/ -v
```

---

## 6. Experimental Results & Benchmarks

Benchmarking on the unseen test split ($K = 20$ components):

| Model Architecture | Subspace | Classifier | Test Accuracy | Macro F1 | FAR (Unknown) | FRR (Known) | Latency | FPS |
|---|---|---|---|---|---|---|---|---|
| **Model A** | Eigenfaces ($K=20$) | Euclidean Distance | **67.27%** | **0.5086** | **28.00%** | **3.33%** | 0.08 ms | 12,500 |
| **Model B** | Eigenfaces ($K=20$) | KNN ($k=3$) | 52.73% | 0.3883 | 48.00% | 0.00% | 0.22 ms | 4,500 |
| **Model C** | Eigenfaces ($K=20$) | SVM (RBF) | 54.55% | 0.4320 | 48.00% | 23.33% | 0.41 ms | 2,400 |

All models process in $< 0.5$ ms per face on standard CPU hardware.

---

## 7. Known Limitations & Troubleshooting

1. **Extreme Off-Axis Poses**: Eigenfaces relies on linear projections. Faces turned $> 45^\circ$ away from the camera exhibit higher reconstruction error.
2. **Camera Hardware Initialization**: If your webcam fails to open, verify device permissions in Windows Settings -> Privacy -> Camera, or test device index 1: `python scripts/run_webcam.py --device 1`.
3. **Custom YOLO Weights**: If you have custom trained YOLO face weights, place them at `models/yolo/best.pt`. The system automatically detects and prefers `best.pt` over the included `yolov8n.pt` fallback.

