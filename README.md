# Predicting 30-day Hospital Readmission from EHR Data

This project predicts 30-day hospital readmission for diabetic patients using structured EHR (Electronic Health Record) data from the UCI Diabetes 130-US Hospitals dataset.

We compare:
- Logistic Regression (L2 and L1)
- A small feed-forward neural network
- Logistic Regression trained on neural-network embeddings

The purpose is to understand how far simple, course-approved models can go on a noisy clinical prediction task.

--------------------------------------------------------------------------------

## 1. Dataset

Source:
UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)

Target:
Binary label representing 30-day readmission.

Class balance:
Positive class around 11 percent  
Negative class around 89 percent  

Features included after processing:
- Age bucket
- Gender
- Race
- Admission and discharge types
- Diagnosis groupings mapped from ICD-9 codes
- Number of diagnoses
- Medication counts and flags
- Laboratory and procedure utilization features

The cleaned dataset is stored at:
data_processed/admissions_features.csv

--------------------------------------------------------------------------------

## 2. Repository Structure

```text
├── data_raw/
│   └── diabetic_data.csv
├── data_processed/
│   ├── admissions_features.csv
│   ├── embeddings_train.npy
│   ├── embeddings_val.npy
│   ├── embeddings_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
├── notebooks/
│   ├── 01_build_uci_cohort.ipynb
│   ├── 02_model_baselines.ipynb
│   ├── 03_model_nn.ipynb
│   └── 04_lr_on_embeddings.ipynb
└── src/
    ├── data_loading.py
    ├── feature_engineering.py
    ├── models_baseline.py
    ├── models_nn.py
    ├── train.py
    └── evaluate.py
```

--------------------------------------------------------------------------------

## 3. Environment and Setup

Dependencies:
- Python 3.9 or higher
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch

To install dependencies:
Use pip install commands for the above libraries.

--------------------------------------------------------------------------------

## 4. How to Run the Pipeline

Step 1: Build the cohort  
Use the notebook 01_build_uci_cohort.ipynb to:
- Load raw CSV
- Clean rows
- Map ICD-9 codes
- Engineer features
- Save admissions_features.csv

Step 2: Baseline models  
Use 02_model_baselines.ipynb to train:
- L2 Logistic Regression
- L1 Logistic Regression

Metrics produced:
AUROC, AUPRC, Accuracy, Confusion Matrix, Classification Report

Step 3: Neural network  
Use 03_model_nn.ipynb to train a small feed-forward network:
Input → Linear 128 → ReLU → Dropout → Linear 64 → ReLU → Dropout → Linear 1

This notebook also:
- Uses BCEWithLogitsLoss with class weighting
- Saves hidden layer embeddings
- Saves y labels for all splits

Step 4: Logistic Regression on embeddings  
Use 04_lr_on_embeddings.ipynb to:
- Load saved embeddings
- Standardize them
- Train Logistic Regression
- Evaluate AUROC, AUPRC, Accuracy
- Plot confusion matrix and classification report

--------------------------------------------------------------------------------

## 5. Results

Logistic Regression (L2):
- AUROC around 0.63 to 0.64
- AUPRC around 0.18 to 0.19
- Accuracy around 0.67

Neural Network:
- AUROC around 0.63 to 0.65
- AUPRC around 0.20 to 0.21
- Accuracy around 0.62
- Higher recall but lower overall accuracy

Logistic Regression on embeddings:
- AUROC around 0.62 to 0.63
- AUPRC around 0.19 to 0.20
- Accuracy around 0.62

Interpretation:
All models are constrained by heavy imbalance and noisy tabular data. Neural network improves recall slightly, but accuracy remains similar.

--------------------------------------------------------------------------------

## 6. Limitations

- Strong class imbalance affects accuracy
- Dataset is noisy and limited to coded EHR fields
- Only simple course-approved models used:
  Logistic Regression and feed-forward neural networks
- More powerful tabular models such as XGBoost were intentionally not used

--------------------------------------------------------------------------------

## 7. Possible Extensions (Not Implemented)

- Better comorbidity or medication feature engineering
- Threshold tuning for accuracy vs recall tradeoff
- Focal loss for imbalance
- More expressive NN architectures
- Out-of-time validation

--------------------------------------------------------------------------------

## 8. Acknowledgments

Dataset from the UCI Machine Learning Repository.  
Project completed for CS 184A.

