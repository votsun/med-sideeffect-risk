# Predicting Readmission Risk from Demographics and Medication Patterns

This repository contains the code and minimal data for our CS184A final project. We predict readmission risk for diabetic patients using a small structured EHR-style dataset from Kaggle (included in this repo).

We compare:
- Logistic Regression (L2)
- Logistic Regression (L1)
- A feed-forward neural network
- Logistic Regression trained on neural-network embeddings

The goal is to evaluate how well simple, course-approved models perform on readmission-risk prediction using engineered demographic, utilization, diagnosis, and simplified medication-pattern features.

---

## Dataset

- Source: Kaggle dataset (included locally)
- Raw file: `data_raw/diabetic_data.csv`
- Processed features used for the demo: `data_processed/admissions_features.csv`
- Target: binary readmission label derived from the dataset’s `readmitted` field

The processed file is included to keep the demo fast and fully reproducible.

---

## Repository Structure

project/
- README.md  
- requirements.txt  
- project.ipynb  
- project.html  
- data_raw/  
  - diabetic_data.csv  
- data_processed/  
  - admissions_features.csv
- notebooks/
  - 01_build_cohort.ipynb
  - 02_model_baselines.ipynb
  - 03_model_nn.ipynb
  - 04_lr_on_embeddings.ipynb
- src/  
  - data_loading.py  
  - feature_engineering.py  
  - models_baseline.py  
  - models_nn.py  
  - train.py  
  - evaluate.py  

---

## Setup

Python 3.9+

Install dependencies:

pip install -r requirements.txt

---

## How to Run

You can run this project either locally or in Google Colab.

### Option A: Run locally (recommended)

1. Create and activate a Python environment (3.9+).
2. Install dependencies:

   pip install -r requirements.txt

3. Launch Jupyter:
   - `jupyter lab`
   - or `jupyter notebook`

4. Open and run:
   - `project.ipynb`
   - Kernel → Restart & Run All

This notebook is a stitched summary of all four analysis notebooks. It:
- Walks through the same sections as:
  - `01_build_cohort.ipynb`
  - `02_model_baselines.ipynb`
  - `03_model_nn.ipynb`
  - `04_lr_on_embeddings.ipynb`
- Loads the processed feature file `data_processed/admissions_features.csv`
- Creates train/val/test splits
- Trains and evaluates:
  - L2 logistic regression
  - L1 logistic regression
  - A small feed-forward neural network
  - Logistic regression on the neural-network embeddings
- Reproduces the key metrics and plots that appear in the report and slides

### Option B: Run in Google Colab

1. Upload the entire `project/` folder to Colab (or upload files individually).
2. Open `project.ipynb`.
3. Install dependencies in a cell:

   !pip install -r requirements.txt

4. Run all cells.

If Colab cannot resolve relative paths, make sure your working directory contains:
- `data_raw/`
- `data_processed/`
- `src/`
- `project.ipynb`

---

## Notes

- The Kaggle dataset is small enough to include in the repository.
- The demo is designed to run end-to-end without external downloads.
- Models and metrics are intentionally limited to course-appropriate methods.

---

## Acknowledgments

Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions/data/discussion).  
Project completed for CS 184A.

