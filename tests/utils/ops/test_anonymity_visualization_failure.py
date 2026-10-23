"""CDX-R2-7b: a chart failure inside the k-anonymity profiler reaches the result."""

import pandas as pd

import pamola_core.profiling.analyzers.anonymity as module
from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation
from pamola_core.utils.ops.op_data_source import DataSource


def test_chart_failure_is_reported_in_the_result(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("R2_CHART_FAILED")

    monkeypatch.setattr(module, "create_bar_plot", boom)
    monkeypatch.setattr(module, "create_spider_chart", boom)
    df = pd.DataFrame({"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 1, 2, 2, 3, 3]})
    op = KAnonymityProfilerOperation(
        quasi_identifiers=["x", "y"], analysis_mode="ANALYZE", use_cache=False,
        generate_visualization=True, visualization_backend="matplotlib",
    )
    result = op.execute(DataSource(dataframes={"main": df}), tmp_path, None)
    assert "R2_CHART_FAILED" in result.metrics.get("visualization_error", ""), result.metrics
