# 🦀 Crab Age Prediction — Kaggle Regression Competition

> **Predict the age of crabs from physical measurements using an optimized gradient boosting ensemble.**  
> Final Kaggle MAE: **1.317** · CV MAE: **1.256** · Top ensemble of LightGBM + XGBoost + CatBoost

---

## 📋 Overview

This project tackles the [Crab Age Prediction](https://www.kaggle.com/competitions/midterm-dsci-6003-spring-2026) Kaggle competition, where the goal is to regress crab age from 8 physical body measurements. Accurate age estimation is valuable in marine biology and aquaculture as a non-invasive alternative to the traditional ring-counting method.

| | |
|---|---|
| **Task** | Regression |
| **Metric** | Mean Absolute Error (MAE) |
| **Train size** | 15,000 samples |
| **Test size** | 10,000 samples |
| **Features (raw → engineered)** | 8 → 20 |
| **CV MAE** | 1.2559 |
| **Kaggle MAE** | 1.31662 |

---

## 🗂️ Repository Structure

```
├── ML_Mid_Final_Submission.ipynb   # Full pipeline notebook
├── train.csv                       # Training data
├── test.csv                        # Test data
├── sample_submission.csv           # Submission format
├── V1-2.csv                        # Final submission file
└── README.md
```

---

## 🔬 Methodology

### 1. Feature Engineering

12 new features were derived from the 8 raw inputs, expanding the feature space to **20 total**:

| Feature | Formula | Rationale |
|---|---|---|
| `Volume` | `Length × Diameter × Height` | Shell volume proxy |
| `Density` | `Weight / (Volume + ε)` | Body density |
| `Shucked_ratio` | `Shucked Weight / (Weight + ε)` | Meat-to-body ratio |
| `Shell_ratio` | `Shell Weight / (Weight + ε)` | Shell fraction |
| `Viscera_ratio` | `Viscera Weight / (Weight + ε)` | Gut fraction |
| `Shell_to_Shucked` | `Shell Weight / (Shucked Weight + ε)` | Shell vs. meat balance |
| `BMI` | `Weight / (Length² + ε)` | Body mass index analog |
| `Weight_remainder` | `Weight − Shucked − Viscera − Shell` | Unexplained weight |
| `Weight_cbrt` | `∛Weight` | Linearizes weight-size relationship |
| `log_Shell` | `log(1 + Shell Weight)` | Log-normalizes skewed weight |
| `Sex_F/M/I` | One-hot encoding | Converts categorical sex to numeric |

### 2. Hyperparameter Tuning

Each model was tuned independently using **Optuna** (30 trials, 5-fold CV, Bayesian optimization):

| Model | Best Trial | CV MAE | Key Params |
|---|---|---|---|
| LightGBM | 4 | **1.2866** | lr=0.0225, depth=9, leaves=117 |
| XGBoost | 18 | 1.2899 | lr=0.0328, depth=4, min_child=22 |
| CatBoost | 28 | 1.2888 | lr=0.0384, depth=8, l2_leaf=3.36 |

### 3. Multi-Seed Training

To reduce variance, each model was trained across **3 random seeds × 5 folds = 15 runs per model** (45 total CV runs):

| Model | Seed 42 | Seed 99 | Seed 2024 |
|---|---|---|---|
| LightGBM | 1.2865 | 1.2897 | 1.2856 |
| XGBoost | 1.2899 | 1.2926 | 1.2912 |
| CatBoost | 1.2888 | 1.2882 | 1.2894 |

### 4. Ensemble Strategy

Three blending approaches were evaluated on out-of-fold predictions:

| Method | MAE |
|---|---|
| LightGBM only (rounded) | 1.2601 |
| Equal average (rounded) | 1.2573 |
| **Weighted avg — LGB 0.45 / CB 0.35 / XGB 0.20 (rounded)** | **1.2559** ✅ |

Integer rounding consistently improved MAE, reflecting the discrete nature of the age target.

---

## 📊 Results

| | MAE |
|---|---|
| **CV (OOF)** | 1.2559 |
| **Kaggle Leaderboard** | 1.31662 |
| CV → Leaderboard gap | +0.061 (+4.8%) |

### Generalization Gap Analysis

The ~4.8% gap between CV and leaderboard MAE is attributable to **mild overfitting of ensemble weights to the OOF set** — the weights were selected based on the same folds used to measure CV performance. Mitigations for future work:

- Use a separate held-out set to tune ensemble weights
- Apply nested cross-validation for unbiased blending estimates
- Use fixed equal weights, which are less prone to OOF overfitting

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-9B59B6?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-F7931E?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.x-FFCC00?style=flat-square&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-3.x-4F8EF7?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy)

---

## 🚀 Reproducing Results

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna

# Run the notebook
jupyter notebook ML_Mid_Final_Submission.ipynb
```

> **Note:** Set `RUN_TUNING = True` on first run to search for optimal hyperparameters. After tuning completes, set to `False` to reload saved params from `best_params.json` and skip directly to training.

---

## 💡 Key Takeaways

- **Rounding matters**: Predicting integer ages by rounding continuous outputs reduced MAE more than any single modeling change
- **Ensemble diversity**: All three models contributed meaningfully — LightGBM (leaf-wise), XGBoost (level-wise), and CatBoost (ordered boosting) offer genuinely different inductive biases
- **Multi-seed averaging**: Running 3 seeds per model smoothed out initialization noise and improved OOF stability
- **Feature ratios > raw features**: Weight composition ratios (shucked, shell, viscera) were the most impactful engineered features

---
