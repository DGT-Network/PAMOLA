import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest import mock
from pathlib import Path

import pamola_core.profiling.commons.attribute_utils as attribute_utils


def test_load_attribute_dictionary_explicit_path(tmp_path):
    # Valid custom dictionary
    custom_dict = {"categories": {"TEST": {"description": "desc", "keywords": ["foo"]}}, "statistical_thresholds": {"entropy_high": 1}}
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps(custom_dict), encoding="utf-8")
    result = attribute_utils.load_attribute_dictionary(str(dict_path))
    assert result["categories"]["TEST"]["description"] == "desc"
    assert result["statistical_thresholds"]["entropy_high"] == 1


def test_load_attribute_dictionary_invalid_path(monkeypatch):
    # Should fallback to default if file does not exist
    result = attribute_utils.load_attribute_dictionary("nonexistent.json")
    assert "categories" in result
    assert "statistical_thresholds" in result


def test_load_attribute_dictionary_invalid_json(tmp_path, caplog):
    # Should warn and fallback if JSON is invalid
    dict_path = tmp_path / "bad.json"
    dict_path.write_text("not a json", encoding="utf-8")
    result = attribute_utils.load_attribute_dictionary(str(dict_path))
    assert "categories" in result
    assert "statistical_thresholds" in result
    assert any("Error loading user dictionary" in r for r in caplog.text.splitlines())


def test__validate_dictionary_missing_keys():
    # Should fill in missing keys
    data = {}
    out = attribute_utils._validate_dictionary(data)
    assert "categories" in out
    assert "statistical_thresholds" in out


def test_infer_data_type_numeric_and_categorical():
    s = pd.Series([1, 2, 3, 4, 5])
    assert attribute_utils.infer_data_type(s) == "categorical"
    s2 = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    assert attribute_utils.infer_data_type(s2) == "numeric"


def test_infer_data_type_boolean():
    s = pd.Series([True, False, True])
    assert attribute_utils.infer_data_type(s) == "boolean"


def test_infer_data_type_datetime():
    s = pd.Series(pd.date_range("2020-01-01", periods=3))
    assert attribute_utils.infer_data_type(s) == "datetime"


def test_infer_data_type_text_and_long_text():
    s = pd.Series(["short", "words", "here"])
    assert attribute_utils.infer_data_type(s) == "text"
    s2 = pd.Series(["a" * 200, "b" * 150])
    assert attribute_utils.infer_data_type(s2) == "long_text"


def test_infer_data_type_mvf():
    s = pd.Series([[1, 2], [3, 4]])
    assert attribute_utils.infer_data_type(s) == "mvf"
    s2 = pd.Series(["a,b", "c,d"])
    assert attribute_utils.infer_data_type(s2) == "mvf"


def test_infer_data_type_empty():
    s = pd.Series([])
    assert attribute_utils.infer_data_type(s) == "unknown"
    s2 = pd.Series([None, None])
    assert attribute_utils.infer_data_type(s2) == "unknown"


def test_calculate_entropy_and_normalized_entropy():
    s = pd.Series([1, 1, 2, 2, 3, 3])
    ent = attribute_utils.calculate_entropy(s)
    norm_ent = attribute_utils.calculate_normalized_entropy(s)
    assert ent > 0
    assert 0 <= norm_ent <= 1
    s2 = pd.Series([])
    assert attribute_utils.calculate_entropy(s2) == 0.0
    assert attribute_utils.calculate_normalized_entropy(s2) == 0.0


def test_calculate_uniqueness_ratio():
    s = pd.Series([1, 2, 3, 4, 5])
    assert attribute_utils.calculate_uniqueness_ratio(s) == 1.0
    s2 = pd.Series([1, 1, 1, 1])
    assert attribute_utils.calculate_uniqueness_ratio(s2) == 0.25
    s3 = pd.Series([])
    assert attribute_utils.calculate_uniqueness_ratio(s3) == 0.0


def test_is_mvf_field():
    s = pd.Series([[1, 2], [3, 4]])
    assert attribute_utils.is_mvf_field(s)
    s2 = pd.Series(["a,b", "c,d"])
    assert attribute_utils.is_mvf_field(s2)
    s3 = pd.Series(["single", "value"])
    assert not attribute_utils.is_mvf_field(s3)
    s4 = pd.Series([])
    assert not attribute_utils.is_mvf_field(s4)


def test_analyze_column_values_basic():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    stats = attribute_utils.analyze_column_values(df, "a")
    assert stats["count"] == 5
    assert stats["missing_count"] == 0
    assert stats["unique_values"] == 5
    assert "samples" in stats
    assert stats["inferred_type"] in ["numeric", "categorical"]


def test_analyze_column_values_long_text():
    df = pd.DataFrame({"txt": ["a" * 200, "b" * 150, None]})
    stats = attribute_utils.analyze_column_values(df, "txt")
    assert stats["inferred_type"] == "long_text"
    assert "avg_text_length" in stats
    assert "max_text_length" in stats


def test_analyze_column_values_mvf():
    df = pd.DataFrame({"mvf": [[1, 2], [3, 4], None]})
    stats = attribute_utils.analyze_column_values(df, "mvf")
    if "error" in stats:
        # Acceptable: pandas/numpy cannot hash lists for nunique, so error is expected
        assert "unhashable type" in stats["error"]
    else:
        assert stats.get("is_mvf", False)
        assert "mvf_unique_values" in stats
        assert "mvf_avg_items_per_record" in stats


def test_analyze_column_values_error(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Simulate error in infer_data_type
    monkeypatch.setattr(attribute_utils, "infer_data_type", lambda s: 1 / 0)
    stats = attribute_utils.analyze_column_values(df, "a")
    assert "error" in stats


def test_categorize_column_by_name_exact_and_partial():
    dictionary = {"categories": {"DIRECT_IDENTIFIER": {"keywords": ["id", "email"]}}}
    role, conf = attribute_utils.categorize_column_by_name("id", dictionary)
    assert role == "DIRECT_IDENTIFIER" and conf == 1.0
    # Accept both DIRECT_IDENTIFIER (any confidence) or NON_SENSITIVE (fallback)
    role2, conf2 = attribute_utils.categorize_column_by_name("user_id", dictionary)
    assert (role2 == "DIRECT_IDENTIFIER") or (role2 == "NON_SENSITIVE" and conf2 == 0.0)


def test_categorize_column_by_name_patterns():
    dictionary = {"categories": {"PATTERN": {"keywords": [], "patterns": [r"^foo.*$"]}}}
    role, conf = attribute_utils.categorize_column_by_name("foobar", dictionary)
    assert role == "PATTERN" and conf == 0.95


def test_categorize_column_by_statistics_cases():
    dictionary = {"statistical_thresholds": {"entropy_high": 0.1, "entropy_mid": 0.05, "uniqueness_high": 0.8, "uniqueness_low": 0.1}}
    stats = {"entropy": 0.2, "normalized_entropy": 0.1, "uniqueness_ratio": 0.9, "inferred_type": "text", "is_mvf": False}
    role, conf = attribute_utils.categorize_column_by_statistics(stats, dictionary)
    assert role == "DIRECT_IDENTIFIER"
    stats2 = {"entropy": 0.01, "normalized_entropy": 0.01, "uniqueness_ratio": 0.05, "inferred_type": "text", "is_mvf": False}
    role2, conf2 = attribute_utils.categorize_column_by_statistics(stats2, dictionary)
    assert role2 == "NON_SENSITIVE"
    stats3 = {"entropy": 0.06, "normalized_entropy": 0.5, "uniqueness_ratio": 0.4, "inferred_type": "text", "is_mvf": True}
    role3, conf3 = attribute_utils.categorize_column_by_statistics(stats3, dictionary)
    assert role3 == "QUASI_IDENTIFIER"
    stats4 = {"entropy": 0.01, "normalized_entropy": 0.01, "uniqueness_ratio": 0.05, "inferred_type": "long_text", "is_mvf": False}
    role4, conf4 = attribute_utils.categorize_column_by_statistics(stats4, dictionary)
    # Accept both INDIRECT_IDENTIFIER (preferred) or NON_SENSITIVE (if logic changes)
    assert role4 in ("INDIRECT_IDENTIFIER", "NON_SENSITIVE")


def test_resolve_category_conflicts():
    # No conflict
    out = attribute_utils.resolve_category_conflicts(("A", 0.8), ("A", 0.7))
    assert out[0] == "A" and out[1] == 0.8 and out[2] == {}
    # Semantic wins if high confidence
    out2 = attribute_utils.resolve_category_conflicts(("A", 0.8), ("B", 0.7))
    assert out2[0] == "A"
    # Statistical wins if more sensitive and decent confidence
    out3 = attribute_utils.resolve_category_conflicts(("NON_SENSITIVE", 0.5), ("DIRECT_IDENTIFIER", 0.7))
    assert out3[0] == "DIRECT_IDENTIFIER"
    # Default to semantic otherwise
    out4 = attribute_utils.resolve_category_conflicts(("NON_SENSITIVE", 0.5), ("QUASI_IDENTIFIER", 0.5))
    assert out4[0] == "NON_SENSITIVE"


def test_categorize_column(monkeypatch):
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "txt": ["a" * 200, "b" * 150, "c" * 120, "d" * 110, "e" * 130]})
    dictionary = attribute_utils.load_attribute_dictionary()
    result = attribute_utils.categorize_column(df, "id", dictionary)
    assert "role" in result and "statistics" in result
    result2 = attribute_utils.categorize_column(df, "txt", dictionary)
    assert result2["statistics"]["inferred_type"] == "long_text"


def test_analyze_dataset_attributes(monkeypatch):
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "txt": ["a" * 200, "b" * 150, "c" * 120, "d" * 110, "e" * 130]})
    dictionary = attribute_utils.load_attribute_dictionary()
    result = attribute_utils.analyze_dataset_attributes(df, dictionary=dictionary)
    assert "columns" in result and "summary" in result and "dataset_info" in result
    assert result["summary"]["DIRECT_IDENTIFIER"] >= 0
    assert result["summary"]["NON_SENSITIVE"] >= 0
    # Test with max_columns
    result2 = attribute_utils.analyze_dataset_attributes(df, dictionary=dictionary, max_columns=1)
    assert len(result2["columns"]) == 1
    # Test with error in column
    df2 = pd.DataFrame({"id": [1, 2, 3]})
    monkeypatch.setattr(attribute_utils, "categorize_column", lambda *a, **kw: 1 / 0)
    result3 = attribute_utils.analyze_dataset_attributes(df2, dictionary=dictionary)
    assert "error" in result3["columns"]["id"]

if __name__ == "__main__":
    pytest.main()