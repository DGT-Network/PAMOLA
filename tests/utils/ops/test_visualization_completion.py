"""CDX-AC-12: an explicitly requested visualization is finished when execute() returns.

The render worker used to be joined with ``visualization_timeout``; when the
renderer was slower the operation returned SUCCESS while the thread kept running
and could write artifacts afterwards. Rendering is now waited for (up to the finite
timeout; CDX-R2-7 covers the renderer that exceeds it, see test_visualization_bounded.py)
and a failure is reported in ``result.metrics["visualization_error"]``.
"""

import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.fake_data import FakeNameOperation
from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frame():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave"] * 10,
            "age": [float(v) for v in range(40)],
            "other": range(40),
        }
    )


_CASES = {
    "anonymization_base": (lambda: FullMaskingOperation(field_name="name"), "main"),
    "suppression": (
        lambda: CellSuppressionOperation(field_name="name", suppression_strategy="null"),
        "main",
    ),
    "transformation": (lambda: RemoveFieldsOperation(fields_to_remove=["other"]), "main"),
    "generator": (lambda: FakeNameOperation(field_name="name"), "main"),
    "profiling": (lambda: NumericOperation("age", use_cache=False), "main"),
}


def _viz_threads():
    return [t for t in threading.enumerate() if "viz" in t.name.lower() and t.is_alive()]


def _slow_renderer(state, fail=False, delay=0.6):
    def render(self, *args, **kwargs):
        state["started"] = True
        time.sleep(delay)  # longer than the timeout the operation was given
        if fail:
            raise RuntimeError("renderer exploded")
        state["finished"] = True
        return {}

    return render


@pytest.mark.parametrize("case", sorted(_CASES))
def test_execute_waits_for_a_slow_renderer_despite_a_zero_timeout(case, tmp_path, monkeypatch):
    make, _ = _CASES[case]
    op = make()
    op.generate_visualization = True
    op.visualization_backend = "matplotlib"
    op.visualization_timeout = 0  # not "give up immediately": falls back to the finite default
    state = {}
    monkeypatch.setattr(type(op), "_generate_visualizations", _slow_renderer(state))

    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert state.get("started"), "the visualization step must have run"
    assert state.get("finished"), "execute() returned while the renderer was still running"
    assert _viz_threads() == []
    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert "visualization_error" not in result.metrics


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_failed_renderer_is_reported_in_the_result(case, tmp_path, monkeypatch):
    make, _ = _CASES[case]
    op = make()
    op.generate_visualization = True
    op.visualization_backend = "matplotlib"
    state = {}
    monkeypatch.setattr(type(op), "_generate_visualizations", _slow_renderer(state, fail=True))

    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert _viz_threads() == []
    assert "renderer exploded" in result.metrics.get("visualization_error", ""), result.metrics
    # The data outputs are unaffected by a chart that could not be drawn.
    assert result.status == OperationStatus.SUCCESS, result.error_message


def test_metrics_operation_waits_too(tmp_path, monkeypatch):
    original = pd.DataFrame({"x": [float(i) for i in range(60)]})
    op = FidelityOperation(
        fidelity_metrics=["ks"], columns=["x"], generate_visualization=True,
        visualization_backend="matplotlib", visualization_timeout=10,
    )
    state = {}
    monkeypatch.setattr(type(op), "_generate_visualizations", _slow_renderer(state, delay=1.0))

    result = op.execute(
        DataSource(dataframes={"main": original, "synth": original.copy()}),
        tmp_path, None,
        original_dataset_name="main", transformed_dataset_name="synth",
    )

    assert state.get("finished")
    assert _viz_threads() == []
    assert result.status != OperationStatus.ERROR, result.error_message


def test_a_real_render_is_complete_and_registered_before_return(tmp_path):
    pytest.importorskip("matplotlib")
    op = FullMaskingOperation(
        field_name="name", generate_visualization=True,
        visualization_backend="matplotlib",
    )
    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert result.status == OperationStatus.SUCCESS, result.error_message
    registered = [a for a in result.artifacts if a.category == "visualization"]
    assert registered
    assert all(Path(a.path).exists() for a in registered)
    assert _viz_threads() == []
