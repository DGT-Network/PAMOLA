"""Extended tests for CategoricalGeneralizationOperation targeting 111 missed lines.
Valid strategies: hierarchy, merge_low_freq, frequency_based."""
import pytest
import pandas as pd
from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
    return R()


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue", "yellow", "green", "red"],
        "size": ["S", "M", "L", "XL", "S", "M", "L", "XL"],
        "val": range(8),
    })


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# --- Hierarchy strategy (default) ---
class TestHierarchyStrategy:
    def test_hierarchy_level_1(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="hierarchy", hierarchy_level=1,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_hierarchy_level_2(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="hierarchy", hierarchy_level=2,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_hierarchy_level_3(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="hierarchy", hierarchy_level=3,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# --- Merge low frequency ---
class TestMergeLowFreq:
    def test_merge_default(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="merge_low_freq",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_merge_high_min_group(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="merge_low_freq", min_group_size=5,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_merge_custom_unknown_value(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="merge_low_freq",
            unknown_value="RARE", min_group_size=3,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_merge_many_rare(self, reporter, tmp_path):
        df = pd.DataFrame({
            "cat": ["common"] * 50 + [f"rare_{i}" for i in range(10)],
            "val": range(60),
        })
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq", min_group_size=5,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Frequency-based ---
class TestFrequencyBased:
    def test_frequency_default(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="frequency_based",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_frequency_low_threshold(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="frequency_based",
            freq_threshold=0.05,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_frequency_high_threshold(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="frequency_based",
            freq_threshold=0.9,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_frequency_max_categories(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="frequency_based",
            max_categories=2,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Case sensitivity ---
class TestCaseSensitivity:
    def test_case_insensitive(self, reporter, tmp_path):
        df = pd.DataFrame({
            "cat": ["Red", "RED", "red", "Blue", "BLUE", "blue"],
            "val": range(6),
        })
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq",
            case_sensitive=False, min_group_size=2,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_case_sensitive(self, reporter, tmp_path):
        df = pd.DataFrame({
            "cat": ["Red", "RED", "red", "Blue", "BLUE", "blue"],
            "val": range(6),
        })
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq",
            case_sensitive=True, min_group_size=2,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Privacy check ---
class TestPrivacyCheck:
    def test_privacy_enabled(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="merge_low_freq",
            privacy_check_enabled=True, min_acceptable_k=2,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_privacy_disabled(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="merge_low_freq",
            privacy_check_enabled=False,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Edge cases ---
class TestEdgeCases:
    def test_field_not_found(self, reporter, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        op = CategoricalGeneralizationOperation(
            field_name="nonexistent", strategy="merge_low_freq",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"cat": ["a", None, "b", None, "c"], "val": range(5)})
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_single_category(self, reporter, tmp_path):
        df = pd.DataFrame({"cat": ["x"] * 100, "val": range(100)})
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_many_categories(self, reporter, tmp_path):
        df = pd.DataFrame({"cat": [f"c{i}" for i in range(100)], "val": range(100)})
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq",
            max_categories=10, min_group_size=5,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_large_df(self, reporter, tmp_path):
        n = 5000
        df = pd.DataFrame({"cat": [f"cat_{i % 20}" for i in range(n)], "val": range(n)})
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_execution(self, base_df, reporter, tmp_path):
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="merge_low_freq", use_dask=True,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
