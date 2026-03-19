# 🦅 AI-Powered Wildlife Conservation: Spatiotemporal Tracking & Vision Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-1761A0?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-white?style=for-the-badge&logo=opencv)

## 📌 Project Overview
This repository contains a comprehensive, end-to-end Machine Learning pipeline designed to modernize wildlife conservation efforts. By replacing expensive, intrusive physical GPS collars with AI-driven analysis of crowdsourced citizen-science data (eBird) and high-resolution imagery, this project provides scalable tools for ecologists to track migratory patterns and identify species.

The system is divided into two highly optimized modules:
1. **Spatiotemporal Migration Tracking:** Predicting *when* and *where* migratory waterfowl will travel.
2. **Bird Species Classification:** A 525-class Transfer Learning pipeline with rigorous statistical validation and visual explainability.

---

## 🧠 The Core Philosophy: Why Explainable AI (XAI)?
In ecological research and public policy, **"black-box" models are fundamentally unacceptable.** Policymakers will not designate a wetland as a protected environmental zone simply because a neural network output a hidden probability. 

This project integrates high-tier **Explainable AI (XAI)** at every layer to prove the model's reasoning to human experts:
* **Success vs. Failure Grad-CAM (Computer Vision):** Proves the Convolutional Neural Network (CNN) identifies birds via genuine avian taxonomy (e.g., beak shape, eye rings). Includes explicit failure analysis to document known vulnerabilities like spatial occlusion and background camouflage.
* **Game-Theoretic SHAP Values (Temporal Forecasting):** Deploys Shapley Additive exPlanations (Summary and Dependence plots) to break down complex XGBoost predictions, showing ecologists exactly how temporal features (e.g., rolling population means) drive migration spikes.
* **Algorithmic Transparency:** All data pipelines are documented via generated CSV master tables, detailing feature engineering lexicons and dataset balancing statistics.

---

## 📍 Module 1: Spatiotemporal Migration Tracking
This module reconstructs the flight paths and schedules of migratory waterfowl using unstructured, crowdsourced geolocation data.

### Methodologies & Optimizations
* **Data Engineering:** Extracted kinematic and temporal features, handling unstructured observation volumes and establishing strict continuous weekly intervals.
* **Temporal Forecasting (XGBoost vs Prophet):**
  * Replaced baseline Prophet forecasting with an optimized **XGBoost Regressor** utilizing a **Tweedie Objective Function** (variance power 1.1), perfectly mathematically suited for modeling zero-inflated ecological count data.
  * *Optimization:* Engineered 1-week and 2-week time-lag features alongside 4-week rolling means to smooth human-observation noise.
* **Geospatial Stopover Clustering (HDBSCAN):** Grouped thousands of scattered GPS coordinates into distinct "Stopover Habitats" using density-based clustering to filter in-flight "noise," replacing rigid centroid-based K-Means.
* **Forecast Confidence Intervals:** Generated 95% confidence bands (± 1.96 standard deviations) around time-series predictions to mathematically quantify model uncertainty.

### 🏆 Module 1 Results
* **Forecasting Error:** Achieved a highly accurate **37% MAPE** (Mean Absolute Percentage Error) using XGBoost, drastically outperforming the baseline Prophet model (107% MAPE).
* **Spatial Cohesion:** Achieved a strong **0.72 Silhouette Score** for stopover zone identification.
* **SHAP Verification:** SHAP dependency analysis successfully verified that short-term lags and rolling population momentum are the definitive mathematical drivers of migration spikes.

---

## 🦚 Module 2: Bird Species Classification
A deep learning computer vision pipeline capable of identifying 525 distinct bird species with extreme accuracy, mathematically immune to class-imbalance bias.

### Methodologies & Optimizations
* **Mathematical Data Auditing:** Built an automated script to calculate the dataset's 25th percentile, mathematically defining the strict threshold for "Rare" vs "Common" species.
* **Targeted Synthetic Augmentation:** * Addressed severe class imbalance by applying heavy `albumentations` exclusively to the 144 minority classes, bringing the imbalance ratio to a near-perfect 1:1.1 equilibrium.
* **Transfer Learning (ResNet-50):** Leveraged ImageNet pre-trained weights, freezing early convolutional layers and fine-tuning the final `layer4` block for specific ornithological features.
* **5-Fold Bootstrapped Cross-Validation:** Proved architectural stability and immunized the model against dataset-split luck by evaluating across 5 randomized test subsets.

### 🏆 Module 2 Results
* **Overall Test Accuracy:** **97.41%** (Evaluated on a strictly unseen test holdout set).
* **Macro F1-Score:** **0.9733** (Proving the targeted augmentation successfully eliminated minority-class bias).
* **Model Stability:** 5-Fold Cross-Validation yielded a remarkably tight variance of **±0.25% standard deviation**, mathematically proving consistent generalizability.
* **Morphological Twins Analysis:** Analyzed 20 pairs of highly similar congeneric species (e.g., *Downy vs. Hairy Woodpecker*). The augmentation pipeline reduced the top 5 worst misclassification error rates from ~90% down to ~15%.
* **Statistical Significance:** Conducted McNemar's Test on ablation studies to statistically quantify the classifier improvements over raw baseline weights.

---

## 💻 Tech Stack
* **Deep Learning & Vision:** `PyTorch`, `Torchvision`, `OpenCV`, `Albumentations`
* **Machine Learning & Time Series:** `XGBoost`, `Prophet`, `Scikit-Learn`
* **Geospatial & Clustering:** `HDBSCAN`, `PyKalman`, `Folium`
* **Explainable AI (XAI) & Stats:** `Grad-CAM` (`pytorch_grad_cam`), `SHAP`, `Statsmodels`
* **Data Engineering & Viz:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`

---

## 📂 Professional Repository Structure
This project is engineered for reproducibility. All scripts utilize dynamic `os` pathing. Analytical tables (`.csv`) and graphical proofs (`.png`) route automatically to their designated directories.

```text
wildlife_project/
│
├── requirements.txt
├── README.md
│
├── bird_species/
│   ├── dataset/ (train, valid, test)
│   ├── graphs/ (Grad-CAM, 5-Fold Validation, Ablation Distributions)
│   ├── results/ (Master Tables, Morphological Twins, Misclassification Analysis)
│   ├── 1_balance_image_dataset.py
│   ├── 2_train_vision_model.py
│   ├── 3_evaluate_vision_model.py
│   ├── 4_vision_xai_gradcam.py
│   ├── 5_vision_statistical_test.py
│   ├── 6_vision_xai_ablation.py
│   ├── 7_vision_extra_tables.py
│   ├── 8_vision_5fold_validation.py
│   ├── 9_vision_success_failure_xai.py
│   ├── 12_vision_master_analytics.py
│   ├── resnet50_baseline_imbalanced.pth
│   └── resnet50_bird_classifier.pth
│
└── migration/
    ├── dataset/ (Raw Excel, Cleaned CSV)
    ├── graphs/ (Kalman Paths, Confidence Intervals, SHAP Plots)
    ├── results/ (Feature Engineering Lexicon, Data Pipeline Audits)
    ├── 1_preprocess_migration_data.py
    ├── 2_exploratory_data_analysis.py
    ├── 3_baseline_temporal_forecast.py
    ├── 4_optimized_temporal_forecast.py
    ├── 5_geospatial_stopover_tracking.py
    ├── 6_generate_research_metrics.py
    ├── 9_migration_tables.py
    ├── 10_migration_extra_tables.py
    ├── 11_migration_confidence_interval.py
    └── 12_migration_shap_analysis.py
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

Developed by **Smriti Yadav** as a **Major Research Project** under the guidance of  **Dr. Shalini Puri(Manipal University Jaipur).**


