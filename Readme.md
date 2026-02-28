# 🦅 AI-Powered Wildlife Conservation: Spatiotemporal Tracking & Vision Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-1761A0?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-white?style=for-the-badge&logo=opencv)

## 📌 Project Overview
This repository contains a comprehensive, end-to-end Machine Learning pipeline designed to modernize wildlife conservation efforts. By replacing expensive, intrusive physical GPS collars with AI-driven analysis of crowdsourced citizen-science data (eBird) and high-resolution imagery, this project provides scalable tools for ecologists to track migratory patterns and identify species.

The system is divided into two highly optimized modules:
1. **Spatiotemporal Migration Tracking:** Predicting *when* and *where* migratory waterfowl will travel.
2. **Bird Species Classification:** A 525-class Transfer Learning pipeline with visual explainability.

---

## 🧠 The Core Philosophy: Why Explainable AI (XAI)?
In ecological research and public policy, **"black-box" models are fundamentally unacceptable.** Policymakers will not designate a wetland as a protected environmental zone simply because a neural network output a hidden probability. 

This project integrates **Explainable AI (XAI)** at every layer to prove the model's reasoning to human experts:
* **Grad-CAM (Computer Vision):** Proves the Convolutional Neural Network (CNN) is identifying birds via genuine avian taxonomy (e.g., beak shape, eye rings, plumage) rather than "cheating" by looking at background artifacts like water or trees.
* **SHAP Values (Temporal Forecasting):** Breaks down complex XGBoost predictions to show ecologists exactly which temporal features (e.g., rolling population means, seasonal lags) are driving a sudden spike in migration volume.
* **Prophet Components:** Decomposes raw time-series forecasts into isolated yearly and weekly seasonal trends.

---

## 📍 Module 1: Spatiotemporal Migration Tracking
This module reconstructs the flight paths and schedules of migratory waterfowl using unstructured, crowdsourced geolocation data.

### Methodologies & Optimizations
* **Data Engineering:** Extracted kinematic and temporal features, handling unstructured observation volumes and establishing strict continuous weekly intervals.
* **Temporal Forecasting (XGBoost vs Prophet):** * Replaced baseline Prophet forecasting with an optimized **XGBoost Regressor** utilizing a **Tweedie Objective Function** (variance power 1.5), perfectly mathematically suited for modeling zero-inflated ecological count data.
  * *Optimization:* Engineered 1-week, 2-week, and 3-week time-lag features alongside 4-week rolling means to smooth human-observation noise.
* **Geospatial Stopover Clustering (HDBSCAN):** * Grouped thousands of scattered GPS coordinates into distinct "Stopover Habitats" using the **Haversine metric** to account for the Earth's curvature.
* **Trajectory Reconstruction (Kalman Filter):** * Applied state-estimation physics to smooth chaotic, human-logged daily coordinates into a continuous, overarching flight trajectory.

### 🏆 Module 1 Results
* **Forecasting Error:** Achieved a highly accurate **37% MAPE** (Mean Absolute Percentage Error) using XGBoost, drastically outperforming the baseline Prophet model (107% MAPE).
* **Spatial Cohesion:** Achieved a strong **0.72 Silhouette Score** for stopover zone identification.

---

## 🦚 Module 2: Bird Species Classification
A deep learning computer vision pipeline capable of identifying 525 distinct bird species with extreme accuracy, built on a custom-balanced dataset.

### Methodologies & Optimizations
* **Mathematical Data Auditing:** Built an automated script to calculate the dataset's 25th percentile, mathematically defining the strict threshold for "Rare" vs "Common" species.
* **Targeted Synthetic Augmentation:** * Addressed severe class imbalance by applying heavy `albumentations` (horizontal flips, 25-degree rotations, color jitter, and blur) **exclusively** to the 144 minority classes. 
  * Synthesized ~20,000 new images to perfectly balance the dataset without injecting unnecessary noise into majority classes.
* **Transfer Learning (ResNet-50):** * Leveraged ImageNet pre-trained weights, freezing early convolutional layers (edge detection) and fine-tuning the final `layer4` block for specific ornithological features.
* **High-Throughput Data Streaming:** Utilized PyTorch `DataLoader` with parallel worker threads and standardized `[0.485, 0.456, 0.406]` normalization for optimal GPU saturation.

### 🏆 Module 2 Results
* **Overall Test Accuracy:** **97.41%** (Evaluated on a strictly unseen test holdout set).
* **Macro F1-Score:** **0.9733** (Proving the targeted augmentation successfully eliminated minority-class bias).
* **Overfitting Control:** Model training was dynamically halted and weights saved at the exact convergence point (Epoch 8) before validation loss degraded.

---

## 💻 Tech Stack
* **Deep Learning & Vision:** `PyTorch`, `Torchvision`, `OpenCV`, `Albumentations`
* **Machine Learning & Time Series:** `XGBoost`, `Prophet`, `Scikit-Learn`
* **Geospatial & Clustering:** `HDBSCAN`, `PyKalman`, `Folium`
* **Explainable AI (XAI):** `Grad-CAM` (`pytorch_grad_cam`), `SHAP`
* **Data Engineering & Viz:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`

---

## 📂 Professional Repository Structure
This project is engineered for reproducibility. All scripts utilize dynamic `os` pathing, meaning the codebase can be cloned and executed on any OS without hardcoded path errors. Outputs are routed automatically to their respective `graphs/` and `results/` directories.

```text
wildlife_project/
│
├── requirements.txt
├── README.md
│
├── bird_species/
│   ├── dataset/ (train, valid, test)
│   ├── graphs/ (Grad-CAM heatmaps, Training Curves, Distributions)
│   ├── results/ (computer_vision_metrics.txt)
│   ├── 01_balance_image_dataset.py
│   ├── 02_train_vision_model.py
│   ├── 02b_cloud_training_notebook.ipynb
│   └── 03_evaluate_vision_model.py
│
└── migration/
    ├── dataset/ (Raw Excel, Cleaned CSV)
    ├── graphs/ (Kalman Paths, HDBSCAN Clusters, XGBoost Forecasts)
    ├── results/ (research_metrics.txt)
    ├── 01_preprocess_migration_data.py
    ├── 02_exploratory_data_analysis.py
    ├── 03_baseline_temporal_forecast.py
    ├── 04_optimized_temporal_forecast.py
    ├── 05_geospatial_stopover_tracking.py
    └── 06_generate_research_metrics.py
```

---

## 🚀 Installation & Execution

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [https://github.com/yourusername/wildlife-conservation-ai.git](https://github.com/yourusername/wildlife-conservation-ai.git)
cd wildlife-conservation-ai
```

**2. Create and activate a virtual environment (Python 3.10+ recommended):**
```bash
# Windows
python -m venv wildlife_env
.\wildlife_env\Scripts\activate

# macOS/Linux
python3 -m venv wildlife_env
source wildlife_env/bin/activate
```

**3. Install all required dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Execute the pipelines:**
Navigate into either the migration/ or bird_species/ folder. The scripts are numbered chronologically (01_..., 02_...). Run them in order to reproduce the data cleaning, training, evaluation, and graph generation phases.

```bash
cd migration
python 01_preprocess_migration_data.py
python 02_exploratory_data_analysis.py
# ... and so on
```

(Note: Module 2's vision model requires the pre-trained .pth weights to run the final evaluation. Due to GitHub file size limits, the 100MB model weights are securely hosted [Link to your Google Drive/Cloud storage here] and must be placed in the bird_species/ root folder before running script 03).

---

Developed by **Smriti Yadav** as a **Major Research Project** under **Dr. Shalini Puri(Manipal University Jaipur).**


