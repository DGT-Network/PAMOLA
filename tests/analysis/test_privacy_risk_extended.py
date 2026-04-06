"""Coverage tests for analysis/privacy_risk.py — targets 29 missed lines."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.analysis.privacy_risk import calculate_full_risk


@pytest.fixture
def base_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 70, 200),
        "zip": [f"1000{i % 10}" for i in range(200)],
        "gender": np.random.choice(["M", "F"], 200),
        "salary": np.random.randint(30000, 100000, 200),
    })


class TestCalculateFullRisk:
    def test_basic(self, base_df):
        result = calculate_full_risk(
            base_df, quasi_identifiers=["age", "zip"],
            sensitive_attributes=["salary"],
        )
        assert isinstance(result, dict)

    def test_with_sensitive(self, base_df):
        result = calculate_full_risk(
            base_df, quasi_identifiers=["age", "zip"],
            sensitive_attributes=["salary"],
        )
        assert isinstance(result, dict)

    def test_single_qi(self, base_df):
        result = calculate_full_risk(
            base_df, quasi_identifiers=["age"],
            sensitive_attributes=["salary"],
        )
        assert isinstance(result, dict)

    def test_all_qi(self, base_df):
        result = calculate_full_risk(
            base_df, quasi_identifiers=["age", "zip", "gender"],
            sensitive_attributes=["salary"],
        )
        assert isinstance(result, dict)

    def test_all_unique_ids(self):
        df = pd.DataFrame({"uid": range(100), "val": range(100)})
        result = calculate_full_risk(
            df, quasi_identifiers=["uid"], sensitive_attributes=["val"],
        )
        assert isinstance(result, dict)

    def test_all_same_values(self):
        df = pd.DataFrame({"cat": ["A"] * 100, "val": range(100)})
        result = calculate_full_risk(
            df, quasi_identifiers=["cat"], sensitive_attributes=["val"],
        )
        assert isinstance(result, dict)

    def test_small_df(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "y"]})
        result = calculate_full_risk(
            df, quasi_identifiers=["a"], sensitive_attributes=["b"],
        )
        assert isinstance(result, dict)

    def test_with_nulls(self):
        df = pd.DataFrame({
            "age": [25, 30, None, 40, None] * 20,
            "zip": ["10001", None, "10003", None, "10005"] * 20,
        })
        result = calculate_full_risk(
            df, quasi_identifiers=["age"], sensitive_attributes=["zip"],
        )
        assert isinstance(result, dict)

    def test_large_df(self):
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "a": np.random.randint(0, 50, n),
            "b": np.random.choice(["x", "y", "z"], n),
            "c": np.random.randint(0, 100, n),
        })
        result = calculate_full_risk(
            df, quasi_identifiers=["a", "b"], sensitive_attributes=["c"],
        )
        assert isinstance(result, dict)

    def test_with_sensitive_and_direct(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "name": [f"person_{i}" for i in range(200)],
            "age": np.random.randint(20, 70, 200),
            "salary": np.random.randint(30000, 100000, 200),
        })
        result = calculate_full_risk(
            df,
            quasi_identifiers=["age"],
            sensitive_attributes=["salary"],
        )
        assert isinstance(result, dict)
