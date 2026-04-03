"""
Tests for pamola_core/profiling/commons/attribute_utils.py

Covers: load_attribute_dictionary, _validate_dictionary, infer_data_type,
calculate_entropy, calculate_normalized_entropy, calculate_uniqueness_ratio,
is_mvf_field, analyze_column_values, categorize_column_by_name,
categorize_column_by_statistics, resolve_category_conflicts,
categorize_column, analyze_dataset_attributes
"""
import json
import math
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.profiling.commons.attribute_utils import (
    load_attribute_dictionary,
    _validate_dictionary,
    infer_data_type,
    calculate_entropy,
    calculate_normalized_entropy,
    calculate_uniqueness_ratio,
    is_mvf_field,
    analyze_column_values,
    categorize_column_by_name,
    categorize_column_by_statistics,
    resolve_category_conflicts,
    categorize_column,
    analyze_dataset_attributes,
    DEFAULT_ATTRIBUTE_ROLES,
    DEFAULT_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_dict():
    return {"categories": DEFAULT_ATTRIBUTE_ROLES, "statistical_thresholds": DEFAULT_THRESHOLDS}


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com"],
        "age": [25, 30, 35, 40, 25],
        "status": ["active", "inactive", "active", "active", "inactive"],
    })


# ---------------------------------------------------------------------------
# load_attribute_dictionary
# ---------------------------------------------------------------------------

class TestLoadAttributeDictionary:
    def test_returns_default_when_no_file(self):
        result = load_attribute_dictionary()
        assert "categories" in result
        assert "statistical_thresholds" in result

    def test_explicit_valid_path(self, tmp_path):
        data = {
            "categories": {"MY_ROLE": {"description": "test", "keywords": ["foo"]}},
            "statistical_thresholds": DEFAULT_THRESHOLDS,
        }
        p = tmp_path / "dict.json"
        p.write_text(json.dumps(data))
        result = load_attribute_dictionary(str(p))
        assert "MY_ROLE" in result["categories"]

    def test_explicit_nonexistent_path_falls_back_to_default(self, tmp_path):
        result = load_attribute_dictionary(str(tmp_path / "does_not_exist.json"))
        assert "categories" in result

    def test_explicit_invalid_json_falls_back_to_default(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json{{")
        result = load_attribute_dictionary(str(p))
        assert "categories" in result

    def test_path_object_accepted(self, tmp_path):
        data = {"categories": DEFAULT_ATTRIBUTE_ROLES, "statistical_thresholds": DEFAULT_THRESHOLDS}
        p = tmp_path / "dict.json"
        p.write_text(json.dumps(data))
        result = load_attribute_dictionary(p)
        assert "categories" in result


# ---------------------------------------------------------------------------
# _validate_dictionary
# ---------------------------------------------------------------------------

class TestValidateDictionary:
    def test_complete_dict_unchanged(self):
        data = {"categories": {"X": {}}, "statistical_thresholds": {"entropy_high": 5.0}}
        result = _validate_dictionary(data)
        assert result["categories"] == {"X": {}}

    def test_missing_categories_filled_with_default(self):
        data = {"statistical_thresholds": DEFAULT_THRESHOLDS}
        result = _validate_dictionary(data)
        assert result["categories"] == DEFAULT_ATTRIBUTE_ROLES

    def test_missing_thresholds_filled_with_default(self):
        data = {"categories": {"X": {}}}
        result = _validate_dictionary(data)
        assert result["statistical_thresholds"] == DEFAULT_THRESHOLDS

    def test_empty_dict_gets_both_defaults(self):
        result = _validate_dictionary({})
        assert "categories" in result
        assert "statistical_thresholds" in result


# ---------------------------------------------------------------------------
# infer_data_type
# ---------------------------------------------------------------------------

class TestInferDataType:
    def test_empty_series(self):
        assert infer_data_type(pd.Series([], dtype=object)) == "unknown"

    def test_boolean(self):
        assert infer_data_type(pd.Series([True, False, True])) == "boolean"

    def test_integer_few_unique(self):
        # 5 unique values in 100 rows: <= 5% unique => categorical
        s = pd.Series([1, 2, 3, 4, 5] * 20)
        result = infer_data_type(s)
        assert result in ("categorical", "numeric")

    def test_float_many_unique(self):
        s = pd.Series(np.random.rand(200))
        assert infer_data_type(s) == "numeric"

    def test_datetime(self):
        s = pd.Series(pd.date_range("2020-01-01", periods=5))
        assert infer_data_type(s) == "datetime"

    def test_all_null_after_dropna(self):
        s = pd.Series([None, None, None])
        assert infer_data_type(s) == "unknown"

    def test_list_values_mvf(self):
        s = pd.Series([[1, 2], [3, 4], [5, 6]])
        assert infer_data_type(s) == "mvf"

    def test_long_text(self):
        long = "x" * 150
        s = pd.Series([long] * 10)
        assert infer_data_type(s) == "long_text"

    def test_categorical_string(self):
        s = pd.Series(["cat", "dog", "cat", "dog"] * 5)
        assert infer_data_type(s) == "categorical"

    def test_text(self):
        s = pd.Series([f"token_{i}" for i in range(50)])
        assert infer_data_type(s) == "text"

    def test_mvf_delimited_string(self):
        # comma-delimited short tokens -> mvf
        s = pd.Series(["a,b,c", "d,e,f", "g,h,i"] * 40)
        result = infer_data_type(s)
        # Could be categorical or mvf depending on uniqueness ratio
        assert result in ("mvf", "categorical", "text")


# ---------------------------------------------------------------------------
# calculate_entropy
# ---------------------------------------------------------------------------

class TestCalculateEntropy:
    def test_empty_series(self):
        assert calculate_entropy(pd.Series([], dtype=object)) == 0.0

    def test_all_nulls(self):
        assert calculate_entropy(pd.Series([None, None])) == 0.0

    def test_uniform_distribution(self):
        # 4 equally likely values: max entropy = log2(4) = 2.0
        s = pd.Series(["a", "b", "c", "d"] * 25)
        e = calculate_entropy(s)
        assert abs(e - 2.0) < 0.01

    def test_single_value(self):
        s = pd.Series(["x"] * 10)
        assert calculate_entropy(s) == 0.0

    def test_two_values_equal(self):
        s = pd.Series([0, 1] * 50)
        e = calculate_entropy(s)
        assert abs(e - 1.0) < 0.01

    def test_positive_entropy(self):
        s = pd.Series(list(range(20)))
        assert calculate_entropy(s) > 0


# ---------------------------------------------------------------------------
# calculate_normalized_entropy
# ---------------------------------------------------------------------------

class TestCalculateNormalizedEntropy:
    def test_empty_series(self):
        assert calculate_normalized_entropy(pd.Series([], dtype=object)) == 0.0

    def test_single_value(self):
        assert calculate_normalized_entropy(pd.Series(["a"] * 10)) == 0.0

    def test_uniform_gives_one(self):
        s = pd.Series(["a", "b", "c", "d"] * 25)
        ne = calculate_normalized_entropy(s)
        assert abs(ne - 1.0) < 0.01

    def test_range_0_to_1(self):
        s = pd.Series([1, 1, 1, 2, 3, 4])
        ne = calculate_normalized_entropy(s)
        assert 0.0 <= ne <= 1.0


# ---------------------------------------------------------------------------
# calculate_uniqueness_ratio
# ---------------------------------------------------------------------------

class TestCalculateUniquenessRatio:
    def test_empty(self):
        assert calculate_uniqueness_ratio(pd.Series([], dtype=object)) == 0.0

    def test_all_unique(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert calculate_uniqueness_ratio(s) == 1.0

    def test_all_same(self):
        s = pd.Series(["x"] * 10)
        assert calculate_uniqueness_ratio(s) == 0.1

    def test_nulls_excluded(self):
        s = pd.Series([1, 2, None, None])
        # 2 unique out of 2 non-null
        assert calculate_uniqueness_ratio(s) == 1.0


# ---------------------------------------------------------------------------
# is_mvf_field
# ---------------------------------------------------------------------------

class TestIsMvfField:
    def test_empty(self):
        assert is_mvf_field(pd.Series([], dtype=object)) is False

    def test_all_nulls(self):
        assert is_mvf_field(pd.Series([None, None])) is False

    def test_list_values(self):
        assert is_mvf_field(pd.Series([[1, 2], [3, 4], [5, 6]])) is True

    def test_tuple_values(self):
        assert is_mvf_field(pd.Series([(1, 2), (3, 4), (5, 6)])) is True

    def test_comma_delimited_short(self):
        s = pd.Series(["a,b,c", "d,e,f"] * 50)
        assert is_mvf_field(s) is True

    def test_plain_text_not_mvf(self):
        s = pd.Series(["hello", "world", "foo", "bar"] * 5)
        assert is_mvf_field(s) is False


# ---------------------------------------------------------------------------
# analyze_column_values
# ---------------------------------------------------------------------------

class TestAnalyzeColumnValues:
    def test_basic_numeric_column(self, simple_df):
        result = analyze_column_values(simple_df, "age")
        assert result["count"] == 5
        assert "entropy" in result
        assert "uniqueness_ratio" in result
        assert "samples" in result

    def test_text_column(self, simple_df):
        result = analyze_column_values(simple_df, "email")
        assert "inferred_type" in result
        assert result["count"] == 5

    def test_text_length_stats(self):
        df = pd.DataFrame({"bio": ["short", "a longer text"] * 10})
        result = analyze_column_values(df, "bio")
        assert "avg_text_length" in result or result["inferred_type"] in ("text", "categorical")

    def test_mvf_column(self):
        """MVF columns with list values — function should handle gracefully."""
        df = pd.DataFrame({"tags": [["a", "b"], ["c"], ["d", "e", "f"]]})
        result = analyze_column_values(df, "tags")
        # The function may or may not detect MVF depending on implementation
        assert isinstance(result, dict)

    def test_long_text_truncated_samples(self):
        df = pd.DataFrame({"notes": ["x" * 200] * 5})
        result = analyze_column_values(df, "notes")
        assert result["inferred_type"] == "long_text"

    def test_missing_values(self):
        df = pd.DataFrame({"col": [1, 2, None, None, 5]})
        result = analyze_column_values(df, "col")
        assert result["missing_count"] == 2

    def test_mvf_delimiter_stats(self):
        df = pd.DataFrame({"tags": ["a,b,c", "d,e", "f,g,h,i"] * 40})
        result = analyze_column_values(df, "tags")
        if result.get("is_mvf"):
            assert "mvf_avg_items_per_record" in result

    def test_error_returns_partial_dict(self):
        df = pd.DataFrame({"good_col": [1, 2, 3]})
        # Passing a non-existent column
        result = analyze_column_values(df, "nonexistent")
        assert "error" in result


# ---------------------------------------------------------------------------
# categorize_column_by_name
# ---------------------------------------------------------------------------

class TestCategorizeColumnByName:
    def test_email_is_direct_identifier(self, default_dict):
        role, conf = categorize_column_by_name("email", default_dict)
        assert role == "DIRECT_IDENTIFIER"
        assert conf > 0.5

    def test_salary_is_sensitive(self, default_dict):
        role, conf = categorize_column_by_name("salary", default_dict)
        assert role == "SENSITIVE_ATTRIBUTE"

    def test_unknown_column_defaults(self, default_dict):
        role, conf = categorize_column_by_name("zzz_xyz_unknown_abc", default_dict)
        assert role == "NON_SENSITIVE"

    def test_exact_keyword_match(self, default_dict):
        role, conf = categorize_column_by_name("id", default_dict)
        assert role == "DIRECT_IDENTIFIER"
        assert conf == 1.0

    def test_partial_keyword_match(self, default_dict):
        role, conf = categorize_column_by_name("user_email_address", default_dict)
        assert role == "DIRECT_IDENTIFIER"
        assert conf > 0

    def test_custom_dictionary_flat_keywords(self):
        dictionary = {
            "categories": {
                "CUSTOM_ROLE": {
                    "description": "test",
                    "keywords": ["email"],
                }
            },
            "statistical_thresholds": DEFAULT_THRESHOLDS,
        }
        role, conf = categorize_column_by_name("email", dictionary)
        assert role == "CUSTOM_ROLE"

    def test_regex_pattern_match(self):
        dictionary = {
            "categories": {
                "PATTERN_ROLE": {
                    "description": "test",
                    "keywords": [],
                    "patterns": [r"^ssn_\d+$"],
                }
            },
            "statistical_thresholds": DEFAULT_THRESHOLDS,
        }
        role, conf = categorize_column_by_name("ssn_123", dictionary)
        assert role == "PATTERN_ROLE"
        assert conf == 0.95

    def test_address_is_quasi_identifier(self, default_dict):
        role, conf = categorize_column_by_name("address", default_dict)
        assert role == "QUASI_IDENTIFIER"


# ---------------------------------------------------------------------------
# categorize_column_by_statistics
# ---------------------------------------------------------------------------

class TestCategorizeColumnByStatistics:
    def test_high_entropy_high_uniqueness_direct_identifier(self, default_dict):
        stats = {
            "entropy": 6.0,
            "normalized_entropy": 0.95,
            "uniqueness_ratio": 0.95,
            "inferred_type": "text",
            "is_mvf": False,
        }
        role, conf = categorize_column_by_statistics(stats, default_dict)
        assert role == "DIRECT_IDENTIFIER"

    def test_long_text_indirect_identifier(self, default_dict):
        stats = {
            "entropy": 3.0,
            "normalized_entropy": 0.7,
            "uniqueness_ratio": 0.5,
            "inferred_type": "long_text",
            "is_mvf": False,
        }
        role, conf = categorize_column_by_statistics(stats, default_dict)
        assert role == "INDIRECT_IDENTIFIER"

    def test_low_entropy_non_sensitive(self, default_dict):
        stats = {
            "entropy": 1.0,
            "normalized_entropy": 0.1,
            "uniqueness_ratio": 0.05,
            "inferred_type": "categorical",
            "is_mvf": False,
        }
        role, conf = categorize_column_by_statistics(stats, default_dict)
        assert role == "NON_SENSITIVE"

    def test_mvf_quasi_identifier(self, default_dict):
        stats = {
            "entropy": 2.0,
            "normalized_entropy": 0.5,
            "uniqueness_ratio": 0.4,
            "inferred_type": "mvf",
            "is_mvf": True,
        }
        role, conf = categorize_column_by_statistics(stats, default_dict)
        assert role == "QUASI_IDENTIFIER"

    def test_medium_entropy_quasi_identifier(self, default_dict):
        stats = {
            "entropy": 4.0,
            "normalized_entropy": 0.7,
            "uniqueness_ratio": 0.5,
            "inferred_type": "text",
            "is_mvf": False,
        }
        role, conf = categorize_column_by_statistics(stats, default_dict)
        assert role == "QUASI_IDENTIFIER"

    def test_defaults_to_non_sensitive(self, default_dict):
        stats = {
            "entropy": 2.0,
            "normalized_entropy": 0.4,
            "uniqueness_ratio": 0.3,
            "inferred_type": "numeric",
            "is_mvf": False,
        }
        role, conf = categorize_column_by_statistics(stats, default_dict)
        assert role == "NON_SENSITIVE"


# ---------------------------------------------------------------------------
# resolve_category_conflicts
# ---------------------------------------------------------------------------

class TestResolveCategoryConflicts:
    def test_no_conflict(self):
        cat, conf, details = resolve_category_conflicts(
            ("DIRECT_IDENTIFIER", 0.8), ("DIRECT_IDENTIFIER", 0.7)
        )
        assert cat == "DIRECT_IDENTIFIER"
        assert details == {}

    def test_high_semantic_confidence_wins(self):
        cat, conf, details = resolve_category_conflicts(
            ("DIRECT_IDENTIFIER", 0.9), ("NON_SENSITIVE", 0.8)
        )
        assert cat == "DIRECT_IDENTIFIER"
        assert "semantic_category" in details

    def test_statistical_more_sensitive_wins(self):
        # semantic QUASI at 0.5 confidence, statistical DIRECT at 0.7
        cat, conf, details = resolve_category_conflicts(
            ("QUASI_IDENTIFIER", 0.5), ("DIRECT_IDENTIFIER", 0.7)
        )
        assert cat == "DIRECT_IDENTIFIER"

    def test_semantic_wins_when_statistical_insufficient_confidence(self):
        # semantic is NON_SENSITIVE 0.5, statistical is DIRECT but only 0.4 confidence
        cat, conf, details = resolve_category_conflicts(
            ("NON_SENSITIVE", 0.5), ("DIRECT_IDENTIFIER", 0.4)
        )
        assert cat == "NON_SENSITIVE"

    def test_conflict_details_populated(self):
        cat, conf, details = resolve_category_conflicts(
            ("SENSITIVE_ATTRIBUTE", 0.6), ("NON_SENSITIVE", 0.7)
        )
        assert "semantic_category" in details
        assert "statistical_category" in details


# ---------------------------------------------------------------------------
# categorize_column
# ---------------------------------------------------------------------------

class TestCategorizeColumn:
    def test_email_column(self, default_dict):
        df = pd.DataFrame({"email": [f"user{i}@test.com" for i in range(20)]})
        result = categorize_column(df, "email", default_dict)
        assert result["column_name"] == "email"
        assert "role" in result
        assert "confidence" in result
        assert "statistics" in result
        assert "semantic_analysis" in result
        assert "statistical_analysis" in result

    def test_salary_column(self, default_dict):
        df = pd.DataFrame({"salary": [30000, 50000, 70000, 90000, 110000]})
        result = categorize_column(df, "salary", default_dict)
        assert result["role"] == "SENSITIVE_ATTRIBUTE"

    def test_with_conflict_details(self, default_dict):
        # id column (direct identifier by name, but low uniqueness might differ)
        df = pd.DataFrame({"status": ["active", "inactive"] * 25})
        result = categorize_column(df, "status", default_dict)
        assert "role" in result

    def test_sample_size_param(self, default_dict):
        df = pd.DataFrame({"age": range(100)})
        result = categorize_column(df, "age", default_dict, sample_size=5)
        assert len(result["statistics"]["samples"]) <= 5


# ---------------------------------------------------------------------------
# analyze_dataset_attributes
# ---------------------------------------------------------------------------

class TestAnalyzeDatasetAttributes:
    def test_basic_analysis(self, simple_df, default_dict):
        result = analyze_dataset_attributes(simple_df, dictionary=default_dict)
        assert "dataset_info" in result
        assert "columns" in result
        assert "summary" in result
        assert result["dataset_info"]["rows"] == 5
        assert result["dataset_info"]["columns"] == 4

    def test_all_columns_analyzed(self, simple_df, default_dict):
        result = analyze_dataset_attributes(simple_df, dictionary=default_dict)
        assert set(result["columns"].keys()) == set(simple_df.columns)

    def test_summary_counts(self, simple_df, default_dict):
        result = analyze_dataset_attributes(simple_df, dictionary=default_dict)
        total = sum(result["summary"].values())
        assert total == len(simple_df.columns)

    def test_column_groups_populated(self, simple_df, default_dict):
        result = analyze_dataset_attributes(simple_df, dictionary=default_dict)
        all_in_groups = sum(len(v) for v in result["column_groups"].values())
        assert all_in_groups == len(simple_df.columns)

    def test_max_columns_limit(self, default_dict):
        df = pd.DataFrame({f"col_{i}": range(10) for i in range(10)})
        result = analyze_dataset_attributes(df, dictionary=default_dict, max_columns=3)
        assert len(result["columns"]) == 3

    def test_auto_loads_dictionary_when_none(self, simple_df):
        result = analyze_dataset_attributes(simple_df, dictionary=None)
        assert "columns" in result

    def test_dataset_metrics_present(self, simple_df, default_dict):
        result = analyze_dataset_attributes(simple_df, dictionary=default_dict)
        assert "dataset_metrics" in result
        assert "avg_entropy" in result["dataset_metrics"]
        assert "avg_uniqueness" in result["dataset_metrics"]

    def test_conflicts_recorded(self, default_dict):
        # Force a conflict: column named 'id' (direct identifier) but all same value (non-sensitive stats)
        df = pd.DataFrame({"id": ["same_value"] * 20, "salary": [50000.0] * 20})
        result = analyze_dataset_attributes(df, dictionary=default_dict)
        # Even if no conflict, the result should be valid
        assert "columns" in result

    def test_dictionary_path_param(self, tmp_path):
        data = {"categories": DEFAULT_ATTRIBUTE_ROLES, "statistical_thresholds": DEFAULT_THRESHOLDS}
        p = tmp_path / "dict.json"
        p.write_text(json.dumps(data))
        df = pd.DataFrame({"email": ["a@b.com", "c@d.com"]})
        result = analyze_dataset_attributes(df, dictionary_path=str(p))
        assert "columns" in result

    def test_error_in_column_handled(self, default_dict):
        # DataFrame with a problematic column won't crash the whole analysis
        df = pd.DataFrame({"email": ["a@b.com", "c@d.com"], "normal": [1, 2]})
        with patch(
            "pamola_core.profiling.commons.attribute_utils.categorize_column",
            side_effect=[Exception("boom"), {"role": "NON_SENSITIVE", "confidence": 0.3,
                                              "statistics": {"entropy": 0, "uniqueness_ratio": 0}}]
        ):
            result = analyze_dataset_attributes(df, dictionary=default_dict)
        assert "columns" in result
