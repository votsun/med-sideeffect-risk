import pandas as pd

def build_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for demographic feature engineering.

    In this project, demographic features (age, sex, etc.)
    were engineered directly in the notebook that builds
    admissions_features.csv.
    """
    return df

def encode_medications_multi_hot(df: pd.DataFrame, vocab) -> pd.DataFrame:
    """
    Placeholder for multi hot encoding of medications.

    Not used in the final pipeline. Medication related
    features were encoded during the admissions_features
    construction step.
    """
    return df