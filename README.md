# Coffee Quality Classification — ML Project

Supervised machine learning project to classify coffee samples as **Specialty** (≥ 80 points) or **Non-Specialty** (< 80 points) using data from the Coffee Quality Institute (CQI).

> Final project for the Machine Learning Applications course.  
> **Author:** Alejandro Zamudio Páez

---

## Problem Statement

Coffee quality evaluation is currently done by certified Q-Graders following CQI protocols — a process that is expensive, subjective, and hard to scale. This project explores whether sensory attributes and production variables can reliably predict the Specialty/Non-Specialty boundary using ML models.

---

## Repository Structure

```
├── coffee_quality_ml.ipynb   # Main notebook (full pipeline)
├── coffee_quality.csv        # Dataset (CQI Arabica, 1,200 samples)
└── README.md
```

---

## Dataset

Based on the [Coffee Quality Institute (CQI) database](https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi), with 1,200 Arabica coffee samples and 17 features:

- **Sensory attributes:** Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean Cup, Sweetness, Cupper Points
- **Production variables:** Country of Origin, Variety, Processing Method, Altitude, Moisture, Category 1 & 2 Defects
- **Target:** Total Cup Points binarized at 80 (Specialty = 1 / Non-Specialty = 0)

Class distribution: **89.8% Specialty / 10.2% Non-Specialty** (imbalanced)

---

## Models

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.9208 | 0.9542 | 0.9888 |
| PCA + Logistic Regression | 0.9250 | 0.9569 | **0.9904** |
| Random Forest | 0.9125 | 0.9536 | 0.9657 |
| RF Optimized (GridSearchCV) | 0.9250 | 0.9600 | 0.9597 |
| XGBoost | 0.9125 | 0.9508 | 0.9221 |
| **XGB Optimized (GridSearchCV)** | **0.9333** | **0.9628** | 0.9479 |

**Best model: XGBoost Optimized** — highest F1-Score and Accuracy.  
All models tuned with GridSearchCV + 5-fold stratified cross-validation.

---

## Key Findings

- **Top 3 most important features:** `Cupper_Points` (0.1509), `Acidity` (0.1159), `Body` (0.1155)
- PCA + Logistic Regression achieves the best AUC-ROC (0.9904), showing strong linear separability after dimensionality reduction
- Main limitation: low recall for Non-Specialty class (0.25) due to class imbalance

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/coffee-quality-ml.git
cd coffee-quality-ml
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 3. Open the notebook
```bash
jupyter notebook coffee_quality_ml_v3.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

---

## Dependencies

| Library | Version |
|---|---|
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| scikit-learn | ≥ 1.2 |
| xgboost | ≥ 1.7 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |

---

## Report

The full IEEE-format report (`informe_ieee_zamudio.docx`) covers:
- Problem statement and objectives
- Dataset description and EDA
- Methodology and preprocessing pipeline
- Model comparison and hyperparameter optimization
- Results, conclusions, and real-world application proposal

---

## References

1. Specialty Coffee Association, "SCA Cupping Protocols," 2015.
2. D. Volpatto, "Coffee Quality Database from CQI," Kaggle, 2018.
3. L. Breiman, "Random Forests," Machine Learning, vol. 45, 2001.
4. T. Chen & C. Guestrin, "XGBoost," KDD, 2016.
5. F. Pedregosa et al., "Scikit-learn," JMLR, vol. 12, 2011.
