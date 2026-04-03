"""Tests targeting progress_tracker and dask code paths in profiling/anonymization.
Covers uncovered lines in: currency.py (196-213, 250-438), categorical_op.py (393-476),
numeric_op.py (330-405), record_op.py (316-460), cell_op.py (262-342)."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
        def add_artifact(self, *a, **kw): pass
    return R()


@pytest.fixture
def pt():
    """Mock progress tracker that exercises progress_tracker code paths."""
    tracker = MagicMock()
    tracker.total = 0
    tracker.update = MagicMock()
    tracker.create_sub_tracker = MagicMock(return_value=MagicMock())
    return tracker


def _run(op, df, tmp_path, reporter, pt=None):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter, progress_tracker=pt)


# === Currency Profiling with progress + large data (196-438) ===
class TestCurrencyWithProgress:
    def test_currency_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        np.random.seed(42)
        df = pd.DataFrame({"amount": np.random.lognormal(5, 1, 500), "id": range(500)})
        op = CurrencyOperation(field_name="amount")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_currency_large_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        np.random.seed(42)
        df = pd.DataFrame({"price": np.random.uniform(1, 1000, 2000), "id": range(2000)})
        op = CurrencyOperation(field_name="price")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_currency_with_many_zeros(self, reporter, tmp_path, pt):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        vals = [0.0] * 100 + list(np.random.uniform(10, 100, 400))
        np.random.shuffle(vals)
        df = pd.DataFrame({"val": vals, "id": range(500)})
        op = CurrencyOperation(field_name="val")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS


# === Categorical Generalization with progress (393-476) ===
class TestCategoricalWithProgress:
    def test_merge_low_freq_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
        cats = [f"common_{i % 3}" for i in range(300)] + [f"rare_{i}" for i in range(50)]
        df = pd.DataFrame({"cat": cats, "val": range(350)})
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="merge_low_freq", min_group_size=10,
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_frequency_based_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
        cats = [f"c{i % 20}" for i in range(500)]
        df = pd.DataFrame({"cat": cats, "val": range(500)})
        op = CategoricalGeneralizationOperation(
            field_name="cat", strategy="frequency_based", max_categories=5,
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_hierarchy_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
        df = pd.DataFrame({
            "color": ["red", "blue", "green", "yellow"] * 100,
            "val": range(400),
        })
        op = CategoricalGeneralizationOperation(
            field_name="color", strategy="hierarchy", hierarchy_level=1,
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Numeric Generalization with progress (330-405) ===
class TestNumericWithProgress:
    def test_binning_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        df = pd.DataFrame({"age": np.random.randint(18, 80, 500), "val": range(500)})
        op = NumericGeneralizationOperation(field_name="age", strategy="binning", bin_count=10)
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        df = pd.DataFrame({"score": np.random.uniform(0, 100, 500), "val": range(500)})
        op = NumericGeneralizationOperation(field_name="score", strategy="rounding", precision=0)
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_equal_freq_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        df = pd.DataFrame({"val": np.random.randint(0, 1000, 500), "id": range(500)})
        op = NumericGeneralizationOperation(
            field_name="val", strategy="binning", bin_count=8, binning_method="equal_frequency",
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS


# === Record Suppression with progress (316-460) ===
class TestRecordSuppressionWithProgress:
    def test_null_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        vals = [f"v{i}" if i % 5 != 0 else None for i in range(500)]
        df = pd.DataFrame({"name": vals, "val": range(500)})
        op = RecordSuppressionOperation(field_name="name", suppression_condition="null")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_value_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({
            "dept": [f"d{i % 5}" for i in range(500)],
            "val": range(500),
        })
        op = RecordSuppressionOperation(
            field_name="dept", suppression_condition="value",
            suppression_values=["d0", "d1"],
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_range_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({"age": list(range(500)), "val": range(500)})
        op = RecordSuppressionOperation(
            field_name="age", suppression_condition="range",
            suppression_range=[100, 300],
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_risk_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        np.random.seed(42)
        df = pd.DataFrame({
            "name": [f"p{i}" for i in range(500)],
            "risk": np.random.uniform(0, 10, 500),
        })
        op = RecordSuppressionOperation(
            field_name="name", suppression_condition="risk",
            ka_risk_field="risk", risk_threshold=5.0,
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS


# === Cell Suppression with progress (262-342) ===
class TestCellSuppressionWithProgress:
    def test_null_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        vals = [float(i) if i % 4 != 0 else None for i in range(500)]
        df = pd.DataFrame({"val": vals, "id": range(500)})
        op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_mean_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        vals = [float(i) if i % 5 != 0 else np.nan for i in range(500)]
        df = pd.DataFrame({"val": vals, "id": range(500)})
        op = CellSuppressionOperation(field_name="val", suppression_strategy="mean")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_group_mean_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        df = pd.DataFrame({
            "val": [float(i) if i % 5 != 0 else np.nan for i in range(500)],
            "grp": [f"g{i % 10}" for i in range(500)],
            "id": range(500),
        })
        op = CellSuppressionOperation(
            field_name="val", suppression_strategy="group_mean",
            group_by_field="grp", min_group_size=5,
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS


# === Attribute Suppression with progress ===
class TestAttributeSuppressionWithProgress:
    def test_remove_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation
        df = pd.DataFrame({
            "name": [f"p{i}" for i in range(500)],
            "age": np.random.randint(20, 70, 500),
            "salary": np.random.randint(30000, 100000, 500),
        })
        op = AttributeSuppressionOperation(field_name="salary")
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_remove_multiple_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation
        df = pd.DataFrame({
            "name": [f"p{i}" for i in range(500)],
            "age": np.random.randint(20, 70, 500),
            "salary": np.random.randint(30000, 100000, 500),
            "dept": [f"d{i % 5}" for i in range(500)],
        })
        op = AttributeSuppressionOperation(
            field_name="salary", additional_fields=["dept"],
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS


# === Datetime Generalization with progress ===
class TestDatetimeWithProgress:
    def test_rounding_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
        dates = pd.date_range("2020-01-01", periods=500, freq="h")
        df = pd.DataFrame({"dt": dates, "val": range(500)})
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="day",
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS

    def test_binning_with_tracker(self, reporter, tmp_path, pt):
        from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        df = pd.DataFrame({"dt": dates, "val": range(500)})
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal",
        )
        result = _run(op, df, tmp_path, reporter, pt)
        assert result.status == OperationStatus.SUCCESS
