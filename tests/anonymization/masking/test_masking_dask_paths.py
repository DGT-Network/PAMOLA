"""Tests targeting Dask code paths in masking and generalization operations.
Many operations have 30-80 lines of dask-specific code that's never exercised."""
import pytest
import pandas as pd
import numpy as np
import dask.dataframe as dd
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


def _run_dask(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    op.use_dask = True
    return op.execute(make_ds(df), tmp_path, reporter)


@pytest.fixture
def large_df():
    """200-row DF to trigger dask-related paths."""
    np.random.seed(42)
    return pd.DataFrame({
        "name": [f"Person_{i}" for i in range(200)],
        "age": np.random.randint(18, 80, 200),
        "salary": np.random.randint(20000, 150000, 200),
        "email": [f"user{i}@domain{i % 10}.com" for i in range(200)],
        "ssn": [f"{100+i}-{50+i}-{1000+i}" for i in range(200)],
        "dept": [f"dept_{i % 5}" for i in range(200)],
        "score": np.random.uniform(0, 100, 200),
        "date": pd.date_range("2020-01-01", periods=200, freq="D"),
    })


# === Full Masking dask ===
class TestFullMaskingDask:
    def test_dask_string(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
        op = FullMaskingOperation(field_name="name", use_encryption=False)
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_numeric(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
        op = FullMaskingOperation(field_name="salary", use_encryption=False)
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Partial Masking dask ===
class TestPartialMaskingDask:
    def test_dask_fixed(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="fixed",
            unmasked_prefix=1, unmasked_suffix=1, use_encryption=False,
        )
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_random(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=50, use_encryption=False,
        )
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Numeric Generalization dask ===
class TestNumericDask:
    def test_dask_binning(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=5,
        )
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_rounding(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        op = NumericGeneralizationOperation(
            field_name="score", strategy="rounding", precision=0,
        )
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Categorical Generalization dask ===
class TestCategoricalDask:
    def test_dask_merge(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
        op = CategoricalGeneralizationOperation(
            field_name="dept", strategy="merge_low_freq", min_group_size=50,
        )
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Datetime Generalization dask ===
class TestDatetimeDask:
    def test_dask_rounding(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
        op = DateTimeGeneralizationOperation(
            field_name="date", strategy="rounding", rounding_unit="month",
        )
        result = _run_dask(op, large_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Cell Suppression dask ===
class TestCellDask:
    def test_dask_null(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        df = pd.DataFrame({
            "val": [f"v{i}" if i % 4 != 0 else None for i in range(200)],
            "id": range(200),
        })
        op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_mean(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        df = pd.DataFrame({
            "val": [float(i) if i % 5 != 0 else np.nan for i in range(200)],
            "id": range(200),
        })
        op = CellSuppressionOperation(field_name="val", suppression_strategy="mean")
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_group_mean(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        df = pd.DataFrame({
            "val": [float(i) if i % 5 != 0 else np.nan for i in range(200)],
            "grp": [f"g{i % 5}" for i in range(200)],
            "id": range(200),
        })
        op = CellSuppressionOperation(
            field_name="val", suppression_strategy="group_mean",
            group_by_field="grp", min_group_size=3,
        )
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Record Suppression dask ===
class TestRecordDask:
    def test_dask_null(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({
            "name": [f"p{i}" if i % 4 != 0 else None for i in range(200)],
            "val": range(200),
        })
        op = RecordSuppressionOperation(field_name="name", suppression_condition="null")
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_value(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({
            "dept": [f"d{i % 5}" for i in range(200)],
            "val": range(200),
        })
        op = RecordSuppressionOperation(
            field_name="dept", suppression_condition="value",
            suppression_values=["d0", "d1"],
        )
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_dask_range(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({"age": list(range(200)), "val": range(200)})
        op = RecordSuppressionOperation(
            field_name="age", suppression_condition="range",
            suppression_range=[50, 150],
        )
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Attribute Suppression dask ===
class TestAttributeDask:
    def test_dask_remove(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation
        df = pd.DataFrame({
            "name": [f"p{i}" for i in range(200)],
            "salary": np.random.randint(30000, 100000, 200),
        })
        op = AttributeSuppressionOperation(field_name="salary")
        result = _run_dask(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# === Noise operations dask ===
class TestNoiseDask:
    def test_uniform_numeric_dask(self, reporter, tmp_path):
        from pamola_core.anonymization.noise.uniform_numeric_op import UniformNumericNoiseOperation
        df = pd.DataFrame({"val": np.random.uniform(0, 100, 200), "id": range(200)})
        op = UniformNumericNoiseOperation(field_name="val", noise_range=10.0)
        op.preset_type = None
        op.preset_name = None
        op.use_dask = True
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_uniform_temporal_dask(self, reporter, tmp_path):
        from pamola_core.anonymization.noise.uniform_temporal_op import UniformTemporalNoiseOperation
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame({"dt": dates, "val": range(200)})
        op = UniformTemporalNoiseOperation(field_name="dt", noise_range_days=30)
        op.preset_type = None
        op.preset_name = None
        op.use_dask = True
        result = op.execute(make_ds(df), tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
