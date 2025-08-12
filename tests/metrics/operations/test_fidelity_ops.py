"""
PAMOLA Core Metrics Package: Unit Tests for FidelityOperation
============================================================
File:        tests/metrics/operations/test_fidelity_ops.py
Target:      pamola_core.metrics.operations.fidelity_ops.FidelityOperation
Coverage:    94% line coverage (see docs)
Top-matter:  Standardized (see process docs)

Description:
    Comprehensive unit tests for FidelityOperation and FidelityConfig, including:
    - Constructor, execute, calculate_metrics, visualization, cache
    - Edge cases and error handling
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

"""
Unit tests for pamola_core.metrics.operations.fidelity_ops
Covers: FidelityConfig, FidelityOperation (init, execute, calculate_metrics, _generate_visualizations, _get_cache_parameters)
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pamola_core.metrics.operations.fidelity_ops import FidelityOperation, FidelityConfig
from pamola_core.metrics.fidelity.distribution.kl_divergence import KLDivergence
from pamola_core.metrics.fidelity.distribution.ks_test import KolmogorovSmirnovTest
from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType
from pamola_core.utils.ops.op_result import OperationStatus
from pamola_core.utils.ops.op_config import ConfigError

@pytest.fixture
def dummy_data():
    df1 = pd.DataFrame({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40]})
    df2 = pd.DataFrame({"A": [1, 2, 2, 4], "B": [10, 20, 25, 40]})
    return df1, df2

@pytest.fixture
def dummy_data_source():
    return MagicMock()

@pytest.fixture
def dummy_task_dir(tmp_path):
    return tmp_path

@pytest.fixture
def dummy_reporter():
    return MagicMock()

@pytest.fixture
def dummy_progress_tracker():
    return MagicMock()

@pytest.mark.parametrize("metrics", [[FidelityMetricsType.KS.value], [FidelityMetricsType.KL.value], [FidelityMetricsType.KS.value, FidelityMetricsType.KL.value]])
def test_fidelity_operation_init_and_config(metrics):
    op = FidelityOperation(fidelity_metrics=metrics, columns=["A", "B"])
    assert isinstance(op.config, FidelityConfig)
    assert set(op.fidelity_metrics) == set(metrics)
    assert op.columns == ["A", "B"]
    assert hasattr(op, "calculate_metrics")

@patch("pamola_core.metrics.operations.fidelity_ops.MetricsOperation.execute", return_value=MagicMock(status=OperationStatus.SUCCESS))
def test_execute_success(mock_super_execute, dummy_data_source, dummy_task_dir, dummy_reporter, dummy_progress_tracker):
    op = FidelityOperation()
    result = op.execute(dummy_data_source, dummy_task_dir, dummy_reporter, dummy_progress_tracker)
    assert result.status == OperationStatus.SUCCESS
    mock_super_execute.assert_called_once()

@patch("pamola_core.metrics.operations.fidelity_ops.MetricsOperation.execute", side_effect=Exception("fail"))
def test_execute_failure(mock_super_execute, dummy_data_source, dummy_task_dir, dummy_reporter, dummy_progress_tracker):
    op = FidelityOperation()
    result = op.execute(dummy_data_source, dummy_task_dir, dummy_reporter, dummy_progress_tracker)
    assert result.status == OperationStatus.ERROR
    assert "fail" in result.error_message

@patch("pamola_core.metrics.operations.fidelity_ops.safe_instantiate", side_effect=lambda cls, params: cls(**params))
def test_calculate_metrics_all_supported(mock_safe_instantiate, dummy_data):
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value, FidelityMetricsType.KL.value])
    df1, df2 = dummy_data
    # Patch metric classes to return dummy results
    with patch.object(KLDivergence, "calculate_metric", return_value={"kl_divergence": 0.1, "kl_divergence_bits": 0.2}), \
         patch.object(KolmogorovSmirnovTest, "calculate_metric", return_value={"ks_statistic": 0.3, "p_value": 0.05}):
        results = op.calculate_metrics(df1, df2, fidelity_metrics=[FidelityMetricsType.KS.value, FidelityMetricsType.KL.value], metric_params={})
        assert "ks" in results and "kl" in results
        assert results["ks"]["ks_statistic"] == 0.3
        assert results["kl"]["kl_divergence"] == 0.1

@patch("pamola_core.metrics.operations.fidelity_ops.safe_instantiate", side_effect=lambda cls, params: cls(**params))
def test_calculate_metrics_unsupported_metric(mock_safe_instantiate, dummy_data):
    # Should raise ConfigError at construction
    with pytest.raises(ConfigError, match="not one of.*ks.*kl"):
        FidelityOperation(fidelity_metrics=["unsupported"])

@patch("pamola_core.metrics.operations.fidelity_ops.safe_instantiate", side_effect=lambda cls, params: cls(**params))
def test_calculate_metrics_empty_list(mock_safe_instantiate, dummy_data):
    op = FidelityOperation(fidelity_metrics=[])
    df1, df2 = dummy_data
    with pytest.raises(ValueError, match="No fidelity metrics specified"):
        op.calculate_metrics(df1, df2, fidelity_metrics=[], metric_params={})

def test_get_cache_parameters():
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value], columns=["A"])
    params = op._get_cache_parameters()
    assert "fidelity_metrics" in params and params["fidelity_metrics"] == [FidelityMetricsType.KS.value]
    assert "columns" in params and params["columns"] == ["A"]

@patch("pamola_core.metrics.operations.fidelity_ops.create_bar_plot", return_value="dummy_path")
def test_generate_visualizations_success(mock_create_bar_plot, tmp_path):
    op = FidelityOperation()
    metrics = {"ks": {"ks_statistic": 0.1, "p_value": 0.05}, "kl": {"kl_divergence": 0.2, "kl_divergence_bits": 0.3}}
    paths = op._generate_visualizations(metrics, tmp_path, vis_theme=None, vis_backend="plotly", vis_strict=False, timestamp="20240101")
    assert "ks_metric_bar" in paths or "kl_metric_bar" in paths

@patch("pamola_core.metrics.operations.fidelity_ops.create_bar_plot", side_effect=Exception("viz fail"))
def test_generate_visualizations_failure(mock_create_bar_plot, tmp_path):
    op = FidelityOperation()
    metrics = {"ks": {"ks_statistic": 0.1, "p_value": 0.05}}
    # Should not raise, just log error
    paths = op._generate_visualizations(metrics, tmp_path, vis_theme=None, vis_backend="plotly", vis_strict=False, timestamp="20240101")
    assert isinstance(paths, dict)

def test_generate_visualizations_error_logging(tmp_path):
    op = FidelityOperation()
    metrics = {"ks": {"ks_statistic": 0.1, "p_value": 0.05}}
    # Patch create_bar_plot to return an error string
    with patch("pamola_core.metrics.operations.fidelity_ops.create_bar_plot", return_value="Error: failed to plot"):
        paths = op._generate_visualizations(metrics, tmp_path, vis_theme=None, vis_backend="plotly", vis_strict=False, timestamp="20240101")
        assert isinstance(paths, dict)
        # Should not include ks_metric_bar due to error
        assert "ks_metric_bar" not in paths

def test_generate_visualizations_backend_none(tmp_path):
    op = FidelityOperation()
    metrics = {"ks": {"ks_statistic": 0.1, "p_value": 0.05}}
    # Should skip visualization and return empty dict
    paths = op._generate_visualizations(metrics, tmp_path, vis_theme=None, vis_backend=None, vis_strict=False, timestamp="20240101")
    assert paths == {}

def test_generate_visualizations_exception(tmp_path):
    op = FidelityOperation()
    metrics = {"ks": {"ks_statistic": 0.1, "p_value": 0.05}}
    # Patch create_bar_plot to raise an exception
    with patch("pamola_core.metrics.operations.fidelity_ops.create_bar_plot", side_effect=Exception("plot fail")):
        paths = op._generate_visualizations(metrics, tmp_path, vis_theme=None, vis_backend="plotly", vis_strict=False, timestamp="20240101")
        assert isinstance(paths, dict)

def test_calculate_metrics_with_progress_tracker(dummy_data):
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value])
    df1, df2 = dummy_data
    progress = MagicMock()
    with patch.object(KolmogorovSmirnovTest, "calculate_metric", return_value={"ks_statistic": 0.3, "p_value": 0.05}):
        results = op.calculate_metrics(df1, df2, progress_tracker=progress, fidelity_metrics=[FidelityMetricsType.KS.value], metric_params={})
        assert "ks" in results
        assert progress.update.called

def test_calculate_metrics_with_extra_metric_params(dummy_data):
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value])
    df1, df2 = dummy_data
    params = {FidelityMetricsType.KS.value: {"alpha": 0.01, "extra": 123}}
    with patch.object(KolmogorovSmirnovTest, "calculate_metric", return_value={"ks_statistic": 0.3, "p_value": 0.05}):
        results = op.calculate_metrics(df1, df2, fidelity_metrics=[FidelityMetricsType.KS.value], metric_params=params)
        assert "ks" in results

def test_get_cache_parameters_all_options():
    op = FidelityOperation(
        fidelity_metrics=[FidelityMetricsType.KS.value],
        columns=["A"],
        column_mapping={"A": "A_new"},
        normalize=False,
        confidence_level=0.9,
        optimize_memory=False,
        use_dask=True,
        npartitions=2,
        dask_partition_size="10MB",
        use_cache=False,
        use_encryption=True,
        encryption_mode="age",
        encryption_key="key",
        visualization_theme="dark",
        visualization_backend="matplotlib",
        visualization_strict=True,
        visualization_timeout=60,
    )
    # Manually set extra attributes for full coverage
    op.force_recalculation = True
    op.generate_visualization = True
    params = op._get_cache_parameters()
    assert params["force_recalculation"] is True
    assert params["generate_visualization"] is True

def test_fidelity_config_defaults():
    config = FidelityConfig(fidelity_metrics=[FidelityMetricsType.KS.value, FidelityMetricsType.KL.value])
    assert config["fidelity_metrics"] == [FidelityMetricsType.KS.value, FidelityMetricsType.KL.value]
    assert config.get("columns", []) == []

def test_fidelity_operation_invalid_columns():
    op = FidelityOperation(columns=None)
    assert op.columns == []

def test_fidelity_operation_repr():
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value], columns=["A"])
    assert "FidelityOperation" in repr(op)

def test_fidelity_operation_all_params():
    op = FidelityOperation(
        name="custom_fidelity",
        fidelity_metrics=[FidelityMetricsType.KS.value],
        metric_params={"alpha": 0.01},
        columns=["A"],
        column_mapping={"A": "A_new"},
        normalize=False,
        confidence_level=0.99,
        description="desc",
        optimize_memory=False,
        sample_size=2,
        use_dask=True,
        npartitions=2,
        dask_partition_size="10MB",
        use_cache=False,
        use_encryption=True,
        encryption_mode="age",
        encryption_key="key",
        visualization_theme="dark",
        visualization_backend="matplotlib",
        visualization_strict=True,
        visualization_timeout=60,
    )
    assert op.name == "custom_fidelity"
    assert op.metric_params == {"alpha": 0.01}
    assert op.column_mapping == {"A": "A_new"}
    assert op.normalize is False
    assert op.confidence_level == 0.99
    assert op.description == "desc"
    assert op.optimize_memory is False
    assert op.sample_size == 2
    assert op.use_dask is True
    assert op.npartitions == 2
    assert op.dask_partition_size == "10MB"
    assert op.use_cache is False
    assert op.use_encryption is True
    assert op.encryption_mode == "age"
    assert op.encryption_key == "key"
    assert op.visualization_theme == "dark"
    assert op.visualization_backend == "matplotlib"
    assert op.visualization_strict is True
    assert op.visualization_timeout == 60

def test_calculate_metrics_empty_df():
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value], columns=["A"])
    df1 = pd.DataFrame({"A": []})
    df2 = pd.DataFrame({"A": []})
    result = op.calculate_metrics(df1, df2, fidelity_metrics=[FidelityMetricsType.KS.value], metric_params={})
    assert isinstance(result, dict)

def test_generate_visualizations_unsupported_metric(tmp_path):
    op = FidelityOperation()
    metrics = {"unsupported": {"value": 1.0}}
    paths = op._generate_visualizations(metrics, tmp_path, vis_theme=None, vis_backend="plotly", vis_strict=False, timestamp="20240101")
    assert isinstance(paths, dict)

def test_calculate_metrics_invalid_types():
    op = FidelityOperation(fidelity_metrics=[FidelityMetricsType.KS.value], columns=["A"])
    df1 = "not a dataframe"
    df2 = "not a dataframe"
    with pytest.raises(Exception):
        op.calculate_metrics(df1, df2, fidelity_metrics=[FidelityMetricsType.KS.value], metric_params={})
