"""Tests targeting ENRICH mode and edge case paths across masking + generalization ops.
Many operations have 20-40 lines of ENRICH-specific code that only triggers
when mode="ENRICH" is passed."""
import pytest
import pandas as pd
import numpy as np
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
def large_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "name": [f"Person_{i}" for i in range(n)],
        "age": np.random.randint(18, 80, n),
        "salary": np.random.randint(20000, 150000, n),
        "dept": [f"dept_{i % 8}" for i in range(n)],
        "score": np.random.uniform(0, 100, n),
        "date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "email": [f"user{i}@d{i % 20}.com" for i in range(n)],
    })


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# === Full Masking ENRICH ===
class TestFullMaskingEnrich:
    def test_enrich_string(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
        op = FullMaskingOperation(
            field_name="name", use_encryption=False, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_numeric(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
        op = FullMaskingOperation(
            field_name="salary", use_encryption=False, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_preserve_length(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
        op = FullMaskingOperation(
            field_name="name", preserve_length=True,
            use_encryption=False, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Partial Masking ENRICH ===
class TestPartialMaskingEnrich:
    def test_enrich_fixed(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="fixed",
            unmasked_prefix=2, unmasked_suffix=1,
            use_encryption=False, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_pattern_email(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
        op = PartialMaskingOperation(
            field_name="email", mask_strategy="pattern",
            pattern_type="email", use_encryption=False, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_random(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
        op = PartialMaskingOperation(
            field_name="name", mask_strategy="random",
            mask_percentage=50, use_encryption=False, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Numeric Generalization ENRICH ===
class TestNumericEnrich:
    def test_enrich_binning(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=8,
            mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_rounding(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        op = NumericGeneralizationOperation(
            field_name="score", strategy="rounding", precision=0,
            mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_equal_freq(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
        op = NumericGeneralizationOperation(
            field_name="salary", strategy="binning",
            bin_count=5, binning_method="equal_frequency",
            mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Categorical Generalization ENRICH ===
class TestCategoricalEnrich:
    def test_enrich_merge(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
        op = CategoricalGeneralizationOperation(
            field_name="dept", strategy="merge_low_freq",
            min_group_size=100, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_frequency(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation
        op = CategoricalGeneralizationOperation(
            field_name="dept", strategy="frequency_based",
            max_categories=3, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Datetime Generalization ENRICH ===
class TestDatetimeEnrich:
    def test_enrich_rounding(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
        op = DateTimeGeneralizationOperation(
            field_name="date", strategy="rounding",
            rounding_unit="month", mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_binning(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
        op = DateTimeGeneralizationOperation(
            field_name="date", strategy="binning",
            bin_type="seasonal", mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_component(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation
        op = DateTimeGeneralizationOperation(
            field_name="date", strategy="component",
            keep_components=["year", "month"], mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Noise operations ENRICH ===
class TestNoiseEnrich:
    def test_numeric_noise_enrich(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.noise.uniform_numeric_op import UniformNumericNoiseOperation
        op = UniformNumericNoiseOperation(
            field_name="salary", noise_range=5000.0, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_temporal_noise_enrich(self, large_df, reporter, tmp_path):
        from pamola_core.anonymization.noise.uniform_temporal_op import UniformTemporalNoiseOperation
        op = UniformTemporalNoiseOperation(
            field_name="date", noise_range_days=30, mode="ENRICH",
        )
        result = _run(op, large_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# === Cell/Record Suppression with save_suppressed ===
class TestSuppressionSaveRecords:
    def test_cell_null_save(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
        df = pd.DataFrame({
            "val": [f"v{i}" if i % 3 != 0 else None for i in range(300)],
            "id": range(300),
        })
        op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_record_null_save(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({
            "name": [f"p{i}" if i % 4 != 0 else None for i in range(300)],
            "val": range(300),
        })
        op = RecordSuppressionOperation(
            field_name="name", suppression_condition="null",
            save_suppressed_records=True, suppression_reason_field="_reason",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_record_value_save(self, reporter, tmp_path):
        from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
        df = pd.DataFrame({
            "dept": [f"d{i % 5}" for i in range(300)],
            "val": range(300),
        })
        op = RecordSuppressionOperation(
            field_name="dept", suppression_condition="value",
            suppression_values=["d0", "d1"],
            save_suppressed_records=True,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
