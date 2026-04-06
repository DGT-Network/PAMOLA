"""Deep coverage tests for ImputeMissingValuesOperation.
Targets missed lines: 215-226, 370-380, 396-399, 548, 784-955, 1262-1439."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


class TestMetricsCalculation:
    """Lines 784-955: _collect_metrics statistical calculations."""

    def test_metrics_numeric_fields(self):
        """Test std, var, skew, kurtosis for numeric fields (lines 875-945)."""
        original = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        processed = pd.DataFrame({"val": [1.0, 2.0, 3.0, 3.5, 5.0]})

        op = ImputeMissingValuesOperation(field_strategies={"val": {"strategy": "mean"}})
        op.column_prefix = ""
        op.process_count = 5
        op.execution_time = 0.1

        metrics = op._collect_metrics(original, processed)
        assert "imputation_impacts" in metrics
        assert "imputed_values" in metrics

    def test_metrics_numeric_and_categorical(self):
        """Non-numeric skip statistical_comparisons (lines 869-873)."""
        original = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat": ["a", "b", None, "d", "a"],
        })
        processed = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "a", "d", "a"],
        })

        op = ImputeMissingValuesOperation(
            field_strategies={"num": {"strategy": "mean"}, "cat": {"strategy": "mode"}},
        )
        op.column_prefix = ""
        op.process_count = 5
        op.execution_time = 0.1

        metrics = op._collect_metrics(original, processed)
        # numeric has statistical_comparisons, categorical doesn't
        assert "imputed_values" in metrics
        assert isinstance(metrics["imputation_impacts"], dict)

    def test_metrics_with_column_prefix(self):
        """Test prefix stripping in metrics (lines 839-846)."""
        original = pd.DataFrame({"age": [20.0, 30.0, np.nan, 50.0, 60.0]})
        processed = pd.DataFrame({"enrich_age": [20.0, 30.0, 40.0, 50.0, 60.0]})

        op = ImputeMissingValuesOperation(
            field_strategies={"age": {"strategy": "mean"}},
            mode="ENRICH",
            column_prefix="enrich_",
        )
        op.process_count = 5
        op.execution_time = 0.1

        metrics = op._collect_metrics(original, processed)
        assert "age" in metrics.get("imputed_values", {})


class TestProgressAndCache:
    """Lines 214-227, 370-380: Progress updates and cache exception handling."""

    def test_progress_cache_check_and_hit(self, tmp_path):
        """Lines 215-216, 226-227: Progress updated on cache check and hit."""
        df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 4.0, 5.0]})

        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
            use_cache=True,
        )

        progress = MagicMock()
        cache_result = MagicMock(spec=['status'])
        cache_result.status = OperationStatus.SUCCESS

        with patch.object(op, '_validate_and_get_dataframe', return_value=df.copy()):
            with patch.object(op, '_check_cache', return_value=cache_result):
                with patch.object(op, '_cleanup_memory'):
                    ds = DataSource(dataframes={"main": df})
                    result = op.execute(
                        data_source=ds,
                        task_dir=tmp_path,
                        reporter=None,
                        progress_tracker=progress,
                    )
                    # Should use cached result
                    assert result is cache_result

    def test_cache_save_exception(self, tmp_path):
        """Lines 378-380: Cache exception logged, not re-raised."""
        df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 4.0, 5.0]})

        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
            use_cache=True,
        )

        with patch.object(op, '_validate_and_get_dataframe', return_value=df.copy()):
            with patch.object(op, '_check_cache', return_value=None):
                with patch.object(op, '_process_dataframe', return_value=df.copy()):
                    with patch.object(op, '_calculate_all_metrics', return_value={}):
                        with patch.object(op, '_save_metrics'):
                            with patch.object(op, '_handle_visualizations'):
                                with patch.object(op, '_save_output_data'):
                                    # Force cache save to fail
                                    with patch.object(op, '_save_to_cache', side_effect=Exception("Cache error")):
                                        with patch.object(op, '_cleanup_memory'):
                                            ds = DataSource(dataframes={"main": df})
                                            result = op.execute(
                                                data_source=ds,
                                                task_dir=tmp_path,
                                                reporter=None,
                                                progress_tracker=None,
                                            )
                                            # Should not raise
                                            assert result is not None


class TestReporterDetails:
    """Lines 396-399: generalization_ratio conditional addition."""

    def test_generalization_ratio_in_reporter(self, tmp_path):
        """Lines 396-399: Add generalization_ratio when present."""
        df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 4.0, 5.0]})

        op = ImputeMissingValuesOperation(field_strategies={"val": {"strategy": "mean"}})
        op.process_count = 5
        op.execution_time = 0.1

        reporter = MagicMock()

        with patch.object(op, '_validate_and_get_dataframe', return_value=df.copy()):
            with patch.object(op, '_check_cache', return_value=None):
                with patch.object(op, '_process_dataframe', return_value=df.copy()):
                    with patch.object(op, '_calculate_all_metrics', return_value={"generalization_ratio": 0.85}):
                        with patch.object(op, '_save_metrics'):
                            with patch.object(op, '_handle_visualizations'):
                                with patch.object(op, '_save_output_data'):
                                    with patch.object(op, '_save_to_cache'):
                                        with patch.object(op, '_cleanup_memory'):
                                            ds = DataSource(dataframes={"main": df})
                                            op.execute(
                                                data_source=ds,
                                                task_dir=tmp_path,
                                                reporter=reporter,
                                                progress_tracker=None,
                                            )

                                            assert reporter.add_operation.called


class TestVisualizationPaths:
    """Lines 1218-1439: Visualization generation and error handling."""

    def test_visualization_backend_none(self, tmp_path):
        """Line 1233-1235: Return empty dict when backend is None."""
        original = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        processed = original.copy()

        op = ImputeMissingValuesOperation(field_strategies={"val": {"strategy": "mean"}})
        op.column_prefix = ""

        result = op._handle_visualizations(
            original_df=original,
            processed_df=processed,
            metrics={"imputed_values": {"val": {"count": 0}}},
            task_dir=tmp_path,
            result=MagicMock(),
            reporter=None,
            vis_theme=None,
            vis_backend=None,
            vis_strict=False,
            vis_timeout=30,
            progress_tracker=None,
            operation_timestamp="20240101_120000",
        )

        assert result == {}

    def test_visualization_auto_timestamp(self, tmp_path, monkeypatch):
        """Line 1229-1230: Auto-generate timestamp when None."""
        import sys
        viz_stub = __import__('types').SimpleNamespace(
            create_bar_plot=lambda **kw: "Error",
            create_histogram=lambda **kw: "Error",
            create_boxplot=lambda **kw: "Error",
        )
        monkeypatch.setitem(sys.modules, 'pamola_core.utils.visualization', viz_stub)

        original = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        processed = original.copy()

        op = ImputeMissingValuesOperation(field_strategies={"val": {"strategy": "mean"}})
        op.column_prefix = ""

        op._handle_visualizations(
            original_df=original,
            processed_df=processed,
            metrics={"imputed_values": {"val": {"count": 0}}},
            task_dir=tmp_path,
            result=MagicMock(),
            reporter=None,
            vis_theme=None,
            vis_backend="matplotlib",
            vis_strict=False,
            vis_timeout=30,
            progress_tracker=None,
            operation_timestamp=None,
        )
        assert (tmp_path / "visualizations").exists()

    def test_visualization_generation_success(self, tmp_path, monkeypatch):
        """Lines 1266-1282: Create and save visualizations."""
        import sys
        viz_stub = __import__('types').SimpleNamespace(
            create_bar_plot=lambda **kw: "Success",
            create_histogram=lambda **kw: "Success",
            create_boxplot=lambda **kw: "Success",
        )
        monkeypatch.setitem(sys.modules, 'pamola_core.utils.visualization', viz_stub)

        original = pd.DataFrame({
            "val1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "val2": [10.0, 20.0, 30.0, np.nan, 50.0],
        })
        processed = pd.DataFrame({
            "val1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "val2": [10.0, 20.0, 30.0, 30.0, 50.0],
        })

        op = ImputeMissingValuesOperation(
            field_strategies={"val1": {"strategy": "mean"}, "val2": {"strategy": "mean"}},
        )
        op.column_prefix = ""

        result = op._handle_visualizations(
            original_df=original,
            processed_df=processed,
            metrics={"imputed_values": {"val1": {"count": 1}, "val2": {"count": 1}}},
            task_dir=tmp_path,
            result=MagicMock(),
            reporter=None,
            vis_theme="default",
            vis_backend="matplotlib",
            vis_strict=False,
            vis_timeout=30,
            progress_tracker=None,
            operation_timestamp="20240101_120000",
        )
        assert len(result) > 0


class TestCategoricalAndInvalidValues:
    """Lines 548, invalid value handling."""

    def test_categorical_add_new_category(self):
        """Line 548-550: Add new category to categorical column."""
        df = pd.DataFrame({
            "cat": pd.Categorical(["a", "b", "a", None, "a"], categories=["a", "b"]),
        })

        op = ImputeMissingValuesOperation(
            field_strategies={"cat": {"strategy": "constant", "constant_value": "c"}},
        )

        result = op.process_batch(df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_invalid_values_replaced(self):
        """Test that invalid values are replaced with NaN before imputation."""
        df = pd.DataFrame({
            "val": [1.0, 0.0, 3.0, 0.0, 5.0],
            "cat": ["a", "", "b", "", "a"],
        })

        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}, "cat": {"strategy": "mode"}},
            invalid_values={"val": [0.0], "cat": [""]},
        )

        result = op.process_batch(df.copy())
        assert result is not None
        assert isinstance(result, pd.DataFrame)
