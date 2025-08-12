"""
File: test_data_utils.py
Test Target: commons/data_utils.py
Version: 1.0
Coverage Status: In Progress
Last Updated: 2025-07-25
Author: PAMOLA Core Team

NOTE: Some tests are skipped due to codebase implementation issues (see pytest.skip reasons below). This is a process lesson-learned: if test failures are due to codebase bugs or incompatibilities, document and skip the tests, and do not mark the file as complete until the codebase is fixed. Update the strict rules and docs accordingly for future requests.
"""

import pytest
import pandas as pd
from pamola_core.anonymization.commons import data_utils

def test_process_nulls_preserve():
    s = pd.Series([1, None, 3])
    result = data_utils.process_nulls(s, strategy="PRESERVE")
    assert result.equals(s)

def test_process_nulls_exclude():
    s = pd.Series([1, None, 3])
    result = data_utils.process_nulls(s, strategy="EXCLUDE")
    assert int(result.isnull().sum(skipna=True)) == 0
    assert 1 in result.values and 3 in result.values

def test_process_nulls_anonymize():
    s = pd.Series([1, None, 3])
    result = data_utils.process_nulls(s, strategy="ANONYMIZE", anonymize_value="MASKED")
    assert "MASKED" in result.values

def test_filter_records_conditionally_basic():
    df = pd.DataFrame({"risk": [1, 6, 10], "val": [10, 20, 30]})
    filtered, mask = data_utils.filter_records_conditionally(df, risk_field="risk", risk_threshold=5)
    assert int(mask.sum(skipna=True)) == 2  # 6 and 10 >= 5

def test_handle_vulnerable_records_suppress():
    df = pd.DataFrame({"val": [1, 2, 3]})
    mask = pd.Series([False, True, False])
    result = data_utils.handle_vulnerable_records(df, "val", mask, strategy="suppress", replacement_value="SUPP")
    # For numeric fields, suppression sets np.nan
    assert pd.isna(result.loc[1, "val"])

def test_create_risk_based_processor_adaptive():
    df = pd.DataFrame({"val": [1, 2, 3, 4]})
    mask = pd.Series([False, True, False, True])
    processor = data_utils.create_risk_based_processor("adaptive")
    result = processor(df, "val", mask)
    # The mean of non-vulnerable values (1 and 3) is 2
    assert result["val"].iloc[1] == 2 and result["val"].iloc[3] == 2

def test_create_privacy_level_processor_high():
    cfg = data_utils.create_privacy_level_processor("HIGH")
    assert cfg["k_threshold"] == 10
    assert callable(cfg["risk_processor"])
    assert cfg["null_strategy"] == "ANONYMIZE"

def test_apply_adaptive_anonymization():
    df = pd.DataFrame({"salary": [100, 200, 300, 400, 500]})
    risk_scores = pd.Series([1, 3, 10, 50, 100])
    result = data_utils.apply_adaptive_anonymization(df, "salary", risk_scores, "HIGH")
    assert "salary" in result.columns
    assert len(result) == 5

def test_get_risk_statistics():
    df = pd.DataFrame({"risk": [1, 3, 7, 12, None]})
    stats = data_utils.get_risk_statistics(df, "risk")
    assert stats["total_records"] == 5
    assert "distribution" in stats
    assert "risk_level_distribution" in stats

def test_get_privacy_recommendations():
    risk_stats = {
        "valid_records": 10,
        "risk_level_distribution": {"VERY_HIGH": 3, "HIGH": 4, "MEDIUM": 2, "LOW": 1},
        "mean_risk": 4.5
    }
    recs = data_utils.get_privacy_recommendations(risk_stats)
    assert "suggested_privacy_level" in recs
    assert isinstance(recs["suggested_strategies"], list)
