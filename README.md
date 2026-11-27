# Parkinson's Disease Severity Prediction

## Person A: Linear Regression

### Objective

Predict total UPDRS severity scores using biomedical voice measurements
from the UCI Parkinson's Telemonitoring Dataset.

### Dataset

- Source: UCI Machine Learning Repository
- Name: Parkinson's Telemonitoring
- Recordings: 5,875
- Patients: 42
- Input features: 16 biomedical voice measurements
- Target: total_UPDRS

### Methodology

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Examine feature correlations.
4. Split recordings by patient using GroupShuffleSplit.
5. Standardize features using training-set statistics.
6. Train a Linear Regression model.
7. Evaluate using MAE, RMSE and R-squared.
8. Examine coefficients and prediction plots.

### Setup

Create and activate a virtual environment, then install dependencies:

    python -m pip install -r requirements.txt

### Run

    python main.py

### Outputs

The outputs/ directory contains plots, model metrics, predictions,
feature correlations, regression coefficients and patient split details.

### Limitations

The model is an educational research prototype. It predicts severity
scores for held-out patients and is not a clinical diagnostic tool.
The test set contains only a subset of the 42 patients, so performance
estimates may be sensitive to which patients are held out.

### Dataset citation

Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. (2009).
Accurate telemonitoring of Parkinson's disease progression by
non-invasive speech tests. IEEE Transactions on Biomedical Engineering.

Dataset: https://archive.ics.uci.edu/dataset/189/parkinson
