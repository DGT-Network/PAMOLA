"""
Pytest configuration and shared fixtures for privacy model tests.

Provides common fixtures for DataFrame creation, test data generation,
and utility functions used across privacy model test suites.

Run with: pytest -s tests/privacy_models/
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def simple_dataframe():
    """Create a simple test DataFrame with basic structure."""
    return pd.DataFrame({
        "age": [25, 30, 25, 35, 30, 25],
        "city": ["NYC", "LA", "NYC", "NYC", "LA", "NYC"],
        "salary": [50000, 60000, 50000, 70000, 60000, 50000],
    })


@pytest.fixture
def large_dataframe():
    """Create a larger test DataFrame for performance testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 65, size=1000),
        "city": np.random.choice(["NYC", "LA", "SF", "CHI", "BOS"], size=1000),
        "salary": np.random.randint(40000, 150000, size=1000),
        "department": np.random.choice(["HR", "IT", "Sales", "Finance"], size=1000),
    })


@pytest.fixture
def k_anonymity_compliant_data():
    """Create data that satisfies k-anonymity with k=3."""
    return pd.DataFrame({
        "age": [25, 25, 25, 30, 30, 30],
        "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
        "income": ["low", "low", "low", "med", "med", "med"],
    })


@pytest.fixture
def l_diversity_data():
    """Create data with good diversity in sensitive attribute."""
    return pd.DataFrame({
        "age": [25, 25, 25, 30, 30, 30],
        "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
        "disease": ["flu", "cold", "allergy", "diabetes", "heart", "cancer"],
    })


@pytest.fixture
def numeric_dataset():
    """Create a dataset with numeric columns for differential privacy."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000],
        "score": [100, 150, 200, 250, 300, 350],
        "performance": [0.75, 0.80, 0.85, 0.90, 0.95, 0.98],
    })


@pytest.fixture
def mixed_type_dataset():
    """Create a dataset with mixed data types."""
    return pd.DataFrame({
        "id": list(range(1, 11)),
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"],
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "city": ["NYC", "LA", "SF", "CHI", "BOS", "NYC", "LA", "SF", "CHI", "BOS"],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        "active": [True, True, False, True, True, False, True, True, False, True],
    })


@pytest.fixture
def categorical_sensitive_data():
    """Create data for testing categorical sensitive attributes."""
    return pd.DataFrame({
        "age": [20, 20, 20, 30, 30, 30, 40, 40, 40],
        "gender": ["M", "M", "M", "F", "F", "F", "M", "M", "M"],
        "condition": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
    })


@pytest.fixture
def skewed_distribution_data():
    """Create data with skewed distribution for testing."""
    return pd.DataFrame({
        "age": [25] * 5 + [30] * 4 + [35] * 1,
        "city": ["NYC"] * 5 + ["LA"] * 4 + ["SF"] * 1,
        "disease": ["flu"] * 7 + ["cold"] * 2 + ["allergy"] * 1,
    })


@pytest.fixture
def single_row_dataframe():
    """Create a single-row DataFrame."""
    return pd.DataFrame({
        "age": [25],
        "city": ["NYC"],
        "salary": [50000],
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame with expected columns."""
    return pd.DataFrame({
        "age": pd.Series(dtype=int),
        "city": pd.Series(dtype=str),
        "salary": pd.Series(dtype=int),
    })


@pytest.fixture
def dataframe_with_nulls():
    """Create a DataFrame with null values."""
    return pd.DataFrame({
        "age": [25, 30, None, 35, 30, 25],
        "city": ["NYC", "LA", "NYC", None, "LA", "NYC"],
        "salary": [50000, 60000, None, 70000, 60000, 50000],
    })


@pytest.fixture
def dataframe_with_duplicates():
    """Create a DataFrame with many duplicate rows."""
    return pd.DataFrame({
        "age": [25, 25, 25, 25, 30, 30],
        "city": ["NYC", "NYC", "NYC", "NYC", "LA", "LA"],
        "salary": [50000, 50000, 50000, 50000, 60000, 60000],
    })


@pytest.fixture
def dataframe_all_unique():
    """Create a DataFrame with all unique quasi-identifier combinations."""
    return pd.DataFrame({
        "age": [25, 26, 27, 28, 29, 30],
        "city": ["NYC", "LA", "SF", "CHI", "BOS", "DEN"],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000],
    })


@pytest.fixture
def temporal_data():
    """Create data with temporal/date columns."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6),
        "age": [25, 30, 25, 35, 30, 25],
        "city": ["NYC", "LA", "NYC", "NYC", "LA", "NYC"],
        "value": [100, 200, 100, 300, 200, 100],
    })


def create_dataframe_with_k_value(k: int, n_groups: int = 1) -> pd.DataFrame:
    """
    Create a DataFrame with specified k-anonymity level.

    Parameters
    -----------
    k : int
        The k-anonymity level to create.
    n_groups : int
        Number of quasi-identifier groups.

    Returns
    --------
    pd.DataFrame
        DataFrame with k-anonymity property.
    """
    rows = []
    for group in range(n_groups):
        for _ in range(k):
            rows.append({
                "qi_col": f"group_{group}",
                "value": np.random.randint(1, 100),
            })

    return pd.DataFrame(rows)


def create_dataframe_with_l_diversity(
    k: int,
    l: int,
    n_groups: int = 1
) -> pd.DataFrame:
    """
    Create a DataFrame with specified l-diversity level.

    Parameters
    -----------
    k : int
        The k-anonymity level.
    l : int
        The l-diversity level.
    n_groups : int
        Number of quasi-identifier groups.

    Returns
    --------
    pd.DataFrame
        DataFrame with l-diversity property.
    """
    rows = []
    for group in range(n_groups):
        sensitive_values = list(range(l)) * (k // l + 1)
        sensitive_values = sensitive_values[:k]
        for i, sa_val in enumerate(sensitive_values):
            rows.append({
                "qi_col": f"group_{group}",
                "sensitive_attr": f"value_{sa_val}",
                "numeric_col": i,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def privacy_test_setup():
    """Provide a complete test setup with multiple data fixtures."""
    return {
        "simple": pd.DataFrame({
            "age": [25, 25, 30, 30],
            "city": ["NYC", "NYC", "LA", "LA"],
            "disease": ["flu", "cold", "flu", "cold"],
        }),
        "large": pd.DataFrame({
            "id": range(100),
            "age": np.random.randint(20, 65, 100),
            "city": np.random.choice(["NYC", "LA", "SF"], 100),
            "value": np.random.randint(1, 1000, 100),
        }),
        "k3": create_dataframe_with_k_value(3, 2),
        "l2": create_dataframe_with_l_diversity(3, 2, 2),
    }
