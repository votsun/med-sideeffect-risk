import pandas as pd
from pathlib import Path

def load_processed(path: str | Path) -> pd.DataFrame:
    """
    Load a processed CSV file into a pandas DataFrame.

    Parameters:
    path : str or Path
        Path to the processed CSV file.

    Returns:
    DataFrame
        Loaded table.
    """
    return pd.read_csv(path)