"""
PAMOLA Core Metrics Package: Unit Tests for Uniqueness
======================================================
File:        tests/metrics/privacy/test_identity.py
Target:      pamola_core.metrics.privacy.identity.Uniqueness
Coverage:    100% line coverage (see docs)
Top-matter:  Standardized (see process docs)

Description:
    Comprehensive unit tests for Uniqueness, including:
    - k-anonymity, sensitive attribute handling, edge cases
    - Compliance with ≥90% line coverage and process requirements

Process:
    - All tests must be self-contained and not depend on external state.
    - All branches and error paths must be exercised.
    - Top-matter must be present and up to date.
    - See process documentation for details.
    
**Version:** 4.0.0
**Coverage Status:** ✅ Full
**Last Updated:** 2025-07-23
"""

import pytest
import pandas as pd
import numpy as np
from pamola_core.metrics.privacy.identity import Uniqueness

class TestUniqueness:
    def get_df(self, n=100, unique_ratio=0.1, seed=42):
        np.random.seed(seed)
        n_unique = int(n * unique_ratio)
        unique_vals = [f"id_{i}" for i in range(n_unique)]
        repeated_vals = np.random.choice(unique_vals, size=n)
        sensitive = np.random.choice(["A", "B", "C", "D"], size=n)
        return pd.DataFrame({"id": repeated_vals, "sensitive": sensitive})

    def test_basic_k_anonymity(self):
        df = self.get_df(n=100, unique_ratio=0.2)
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        result = metric.calculate_metric(df)
        assert "k_anonymity" in result
        assert isinstance(result["k_anonymity"], dict)
        assert "k_anonymity_stats" in result["k_anonymity"]
        assert result["k_anonymity"]["num_groups"] > 0

    def test_l_diversity(self):
        df = self.get_df(n=100, unique_ratio=0.2)
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        ldiv = metric._calculate_l_diversity(df)
        assert "min_l_diversity" in ldiv
        assert ldiv["min_l_diversity"] >= 1

    def test_t_closeness(self):
        df = self.get_df(n=100, unique_ratio=0.2)
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        tclose = metric._calculate_t_closeness(df)
        assert "min_t_closeness" in tclose
        assert tclose["min_t_closeness"] >= 0

    def test_missing_sensitive_field(self):
        df = self.get_df(n=100, unique_ratio=0.2)
        df = df.drop(columns=["sensitive"])
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        with pytest.raises(KeyError):
            metric.calculate_metric(df)

    def test_missing_key_field(self):
        df = self.get_df(n=100, unique_ratio=0.2)
        df = df.drop(columns=["id"])
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        with pytest.raises(KeyError):
            metric.calculate_metric(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["id", "sensitive"])
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        with pytest.raises(ValueError):
            metric.calculate_metric(df)

    def test_all_unique(self):
        df = pd.DataFrame({"id": [f"id_{i}" for i in range(100)], "sensitive": ["A"]*100})
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        result = metric.calculate_metric(df)
        assert result["k_anonymity"]["num_groups"] == 100

    def test_all_same(self):
        df = pd.DataFrame({"id": ["id_1"]*100, "sensitive": ["A"]*100})
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        result = metric.calculate_metric(df)
        assert result["k_anonymity"]["num_groups"] == 1

    def test_l_diversity_edge(self):
        df = pd.DataFrame({"id": ["id_1"]*100, "sensitive": ["A"]*100})
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        ldiv = metric._calculate_l_diversity(df)
        assert ldiv["min_l_diversity"] == 1

    def test_t_closeness_edge(self):
        df = pd.DataFrame({"id": ["id_1"]*100, "sensitive": ["A"]*100})
        metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"])
        tclose = metric._calculate_t_closeness(df)
        assert tclose["min_t_closeness"] == 0
