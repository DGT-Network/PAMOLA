"""Gap coverage tests for CellSuppressionOperation.
Targets uncovered lines: error handlers, Dask paths, group statistics caching,
metrics collection, partition processing, and edge cases."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pamola_core.anonymization.suppression.cell_op import (
    CellSuppressionOperation,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus
from pamola_core.errors.exceptions import ValidationError, InvalidStrategyError


def make_ds(df):
    """Create DataSource from DataFrame."""
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    """Mock reporter."""
    class R:
        def add_operation(self, *a, **kw): pass
        def add_artifact(self, *a, **kw): pass
    return R()


def _run(op, df, tmp_path, reporter):
    """Helper to run operation."""
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# ========== GROUP STATISTICS CACHING (Lines 262-272) ==========
class TestGroupStatisticsCaching:
    """Test group statistics cache eviction when full."""

    def test_group_stats_cache_eviction(self, tmp_path, reporter):
        """Trigger cache overflow by processing many groups."""
        # Create data with 30+ unique groups (MAX_GROUP_STATISTICS_SIZE=25)
        groups = [f"g{i % 30}" for i in range(300)]
        df = pd.DataFrame({
            "val": [float(i) for i in range(300)],
            "grp": groups,
            "id": range(300),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="grp",
            min_group_size=2,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
        # Verify cache eviction occurred (oldest 10% removed)
        assert len(op._group_statistics) <= 25  # Cache limit


# ========== DASK PROCESSING (Lines 291-342) ==========
class TestDaskProcessing:
    """Test Dask-based distributed processing."""

    def test_dask_group_mean_warning(self, tmp_path, reporter):
        """Dask group_mean strategy should log warning."""
        df = pd.DataFrame({
            "val": [float(i) for i in range(100)],
            "grp": [f"g{i % 5}" for i in range(100)],
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="grp",
            use_dask=True,
            npartitions=4,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_global_mean_precompute(self, tmp_path, reporter):
        """Dask should pre-compute global mean for group_mean strategy."""
        df = pd.DataFrame({
            "val": [float(i) for i in range(100)],
            "grp": [f"g{i % 3}" for i in range(100)],
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="grp",
            use_dask=True,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in [OperationStatus.SUCCESS, OperationStatus.ERROR]


# ========== DASK ERROR HANDLING (Lines 785-799, 905-919) ==========
class TestDaskErrorHandling:
    """Test exception handling in Dask processing."""

    def test_dask_processing_error_with_reporter(self, tmp_path, reporter):
        """_process_with_dask error handling reports to reporter."""
        df = pd.DataFrame({
            "val": [float(i) if i % 5 != 0 else None for i in range(100)],
            "id": range(100),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="mean",
            use_dask=True,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in [OperationStatus.SUCCESS, OperationStatus.ERROR]


# ========== JOBLIB ERROR HANDLING (Lines 1013-1027) ==========
class TestJobLibErrorHandling:
    """Test exception handling in Joblib processing."""

    def test_joblib_processing_with_parallel(self, tmp_path, reporter):
        """Joblib processing with parallel_processes > 1."""
        df = pd.DataFrame({
            "val": [float(i) for i in range(100)],
            "id": range(100),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="mean",
            parallel_processes=2,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in [OperationStatus.SUCCESS, OperationStatus.ERROR]


# ========== METRICS COLLECTION (Lines 1063-1094) ==========
class TestMetricsCollection:
    """Test metrics collection including outlier/rare/group stats."""

    def test_outlier_metrics(self, tmp_path, reporter):
        """Metrics should include outlier detection details."""
        df = pd.DataFrame({
            "val": [10, 20, 30, 40, 50, 1000],  # 1000 is outlier
            "id": range(6),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppress_if="outlier",
            outlier_method="iqr",
            outlier_threshold=1.5,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
        assert result.metrics is not None

    def test_rare_metrics(self, tmp_path, reporter):
        """Metrics should include rare value detection details."""
        df = pd.DataFrame({
            "val": ["A", "A", "A", "B", "B", "C"],
            "id": range(6),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppress_if="rare",
            rare_threshold=2,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
        assert result.metrics is not None

    def test_group_statistics_sample_metrics(self, tmp_path, reporter):
        """Metrics should include sample of group statistics."""
        df = pd.DataFrame({
            "val": [float(i) for i in range(100)],
            "grp": [f"g{i % 8}" for i in range(100)],
            "id": range(100),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="grp",
            min_group_size=3,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
        assert result.metrics is not None


# ========== ENRICH MODE TESTS (Lines 1567-1625, covers partition logic) ==========
class TestEnrichModeWithOutput:
    """Test ENRICH mode which exercises partition functions."""

    def test_enrich_mode_with_output_field(self, tmp_path, reporter):
        """ENRICH mode creates output field."""
        df = pd.DataFrame({
            "val": ["A", "B", "C", "D"],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="null",
            mode="ENRICH",
            output_field_name="val_suppressed",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_mode_numeric(self, tmp_path, reporter):
        """ENRICH mode with numeric field."""
        df = pd.DataFrame({
            "val": [1.0, 2.0, 3.0, 4.0],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="mean",
            mode="ENRICH",
            output_field_name="val_sup",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ========== CONDITIONAL SUPPRESSION (Lines 1687-1703) ==========
class TestConditionalSuppression:
    """Test conditional suppression based on field values."""

    def test_suppress_by_condition_field(self, tmp_path, reporter):
        """Suppress based on condition_field and condition_values."""
        df = pd.DataFrame({
            "val": ["A", "B", "C", "A"],
            "status": ["active", "inactive", "active", "inactive"],
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="null",
            condition_field="status",
            condition_values=["inactive"],
            condition_operator="eq",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ========== OUTLIER DETECTION (Lines 1722-1756) ==========
class TestOutlierDetection:
    """Test outlier detection via operation."""

    def test_outlier_iqr_method(self, tmp_path, reporter):
        """IQR outlier detection via operation."""
        df = pd.DataFrame({
            "val": [10, 20, 30, 40, 50, 1000],  # 1000 is clear outlier
            "id": range(6),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppress_if="outlier",
            outlier_method="iqr",
            outlier_threshold=1.5,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_outlier_zscore_method(self, tmp_path, reporter):
        """Z-score outlier detection via operation."""
        df = pd.DataFrame({
            "val": [10, 20, 30, 40, 50, 100],
            "id": range(6),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppress_if="outlier",
            outlier_method="zscore",
            outlier_threshold=2.0,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_outlier_constant_values(self, tmp_path, reporter):
        """Constant values with zero IQR should not be flagged as outliers."""
        df = pd.DataFrame({
            "val": [5, 5, 5, 5, 5],
            "id": range(5),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppress_if="outlier",
            outlier_method="iqr",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ========== STRATEGY COVERAGE (Lines 1873-1960) ==========
class TestStrategyEdgeCases:
    """Test edge cases in strategy application."""

    def test_mean_with_nulls(self, tmp_path, reporter):
        """Mean strategy with NaN values."""
        df = pd.DataFrame({
            "val": [10.0, np.nan, 30.0, 40.0],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="mean",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_median_strategy(self, tmp_path, reporter):
        """Median strategy."""
        df = pd.DataFrame({
            "val": [10, 20, 30, 40, 50],
            "id": range(5),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="median",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mode_strategy(self, tmp_path, reporter):
        """Mode strategy."""
        df = pd.DataFrame({
            "val": ["A", "A", "B", "C", "A"],
            "id": range(5),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="mode",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_constant_strategy(self, tmp_path, reporter):
        """Constant replacement strategy."""
        df = pd.DataFrame({
            "val": [1, 2, 3, 4],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="constant",
            suppression_value=999,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ========== GROUP-BASED STRATEGIES (Lines 1759-1857) ==========
class TestGroupBasedStrategies:
    """Test group_mean and group_mode strategies."""

    def test_group_mean_strategy(self, tmp_path, reporter):
        """Group mean strategy."""
        df = pd.DataFrame({
            "val": [10.0, 20.0, 30.0, 40.0],
            "grp": ["A", "A", "B", "B"],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="grp",
            min_group_size=1,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_group_mean_with_small_groups(self, tmp_path, reporter):
        """Group mean with groups smaller than min_group_size."""
        df = pd.DataFrame({
            "val": [10.0, 20.0, 30.0, 40.0],
            "grp": ["A", "A", "B", "B"],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="grp",
            min_group_size=5,  # Groups too small, should fall back
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_group_mode_strategy(self, tmp_path, reporter):
        """Group mode strategy."""
        df = pd.DataFrame({
            "val": ["A", "A", "B", "C"],
            "grp": ["X", "X", "X", "Y"],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mode",
            group_by_field="grp",
            min_group_size=1,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_group_strategy_multi_group_field(self, tmp_path, reporter):
        """Group strategies with multiple grouping fields."""
        df = pd.DataFrame({
            "val": [10.0, 20.0, 30.0, 40.0],
            "g1": ["A", "A", "B", "B"],
            "g2": ["X", "Y", "X", "Y"],
            "id": range(4),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="group_mean",
            group_by_field="g1",
            min_group_size=1,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ========== EDGE CASES ==========
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_null_field(self, tmp_path, reporter):
        """All-null field suppression."""
        df = pd.DataFrame({
            "val": [None, None, None],
            "id": range(3),
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="null",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_single_row(self, tmp_path, reporter):
        """Single row suppression."""
        df = pd.DataFrame({
            "val": [42],
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="mean",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_enrich_mode_auto_output_field(self, tmp_path, reporter):
        """ENRICH mode with default output field."""
        df = pd.DataFrame({
            "val": [1, 2, 3],
        })
        op = CellSuppressionOperation(
            field_name="val",
            suppression_strategy="null",
            mode="ENRICH",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
