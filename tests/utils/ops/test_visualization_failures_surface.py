"""ASB-OPS-7: a chart that cannot be drawn raises, and the failure reaches the result.

Before: the ``create_*`` helpers returned an ``"Error ..."`` string that 84 call sites only logged,
so a missing browser for kaleido gave SUCCESS with no chart and no ``visualization_error``;
``CellSuppressionOperation`` swallowed chart errors too, and registered the comparison path as its
histogram. The plotly PNG renderer is probed once, at request time.
"""

import numpy as np
import pandas as pd
import pytest

import pamola_core.utils.io as io_module
import pamola_core.utils.ops.op_base as op_base
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.errors.exceptions import VisualizationError
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus
from pamola_core.utils.visualization import create_bar_plot


def _frame():
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol", "Dave"] * 10, "age": np.arange(40.0)})


def test_a_chart_helper_raises_instead_of_returning_an_error_string(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("cannot write")

    monkeypatch.setattr(io_module, "_save_visualization_impl", broken)
    with pytest.raises(VisualizationError) as info:
        create_bar_plot({"a": 1, "b": 2}, output_path=tmp_path / "x.png", title="t", backend="matplotlib")
    assert "cannot write" in str(info.value)


_CASES = {
    "masking": lambda: FullMaskingOperation(field_name="name", use_cache=False),
    "cell_suppression_numeric": lambda: CellSuppressionOperation(
        field_name="age", suppression_strategy="null", use_cache=False
    ),
    "profiling": lambda: NumericOperation("age", use_cache=False),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_save_failure_reaches_visualization_error(case, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full R2_SAVE_FAILED")

    monkeypatch.setattr(io_module, "_save_visualization_impl", broken)
    op = _CASES[case]()
    op.generate_visualization = True
    op.visualization_backend = "matplotlib"
    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
    assert "R2_SAVE_FAILED" in result.metrics.get("visualization_error", ""), result.metrics


def test_cell_suppression_registers_the_histogram_it_drew(tmp_path):
    pytest.importorskip("matplotlib")
    op = CellSuppressionOperation(
        field_name="age", suppression_strategy="null", use_cache=False,
        generate_visualization=True, visualization_backend="matplotlib",
    )
    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message
    paths = [str(a.path) for a in result.artifacts if a.category == "visualization"]
    assert len(paths) == len(set(paths)) and len(paths) >= 2     # comparison and histogram, distinct files
    assert any("distribution" in p for p in paths)


def test_plotly_without_a_renderer_is_one_actionable_error_at_request_time(tmp_path, monkeypatch):
    monkeypatch.setattr(op_base, "_PLOTLY_PNG_PROBE", {"reason": "RuntimeError: Kaleido requires Google Chrome"})
    op = FullMaskingOperation(field_name="name", use_cache=False, generate_visualization=True, visualization_backend="plotly")
    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
    assert result.status == OperationStatus.ERROR
    assert "Chrome" in result.error_message and "matplotlib" in result.error_message
    assert result.error_message.count("Kaleido") == 1        # one message, not one per chart
    assert not list(tmp_path.glob("output/*"))                # nothing was processed


def test_the_probe_is_skipped_when_it_is_not_needed(tmp_path, monkeypatch):
    monkeypatch.setattr(op_base, "_PLOTLY_PNG_PROBE", {"reason": "RuntimeError: broken"})
    off = FullMaskingOperation(field_name="name", use_cache=False)               # visualization off
    assert off.execute(DataSource(dataframes={"main": _frame()}), tmp_path / "a", None).status == OperationStatus.SUCCESS
    mpl = FullMaskingOperation(field_name="name", use_cache=False, generate_visualization=True, visualization_backend="matplotlib")
    pytest.importorskip("matplotlib")
    assert mpl.execute(DataSource(dataframes={"main": _frame()}), tmp_path / "b", None).status == OperationStatus.SUCCESS
