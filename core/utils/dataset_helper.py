import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def split_dataset(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        feature_cols (list): List of column names to be used as features (X).
        target_cols (list): List of column names to be used as target labels (y).
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - X_train (pd.DataFrame): Training set features.
            - X_test (pd.DataFrame): Testing set features.
            - y_train (pd.DataFrame): Training set target labels.
            - y_test (pd.DataFrame): Testing set target labels.
    """
    try:
        # Extract feature and target data
        X = df[feature_cols]
        y = df[target_cols]

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    except KeyError as e:
        raise ValueError(f"Error: Missing columns in the dataset - {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while splitting the dataset: {e}")
