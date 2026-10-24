"""CDX-R2-7: rendering is bounded, cancellable and its failures are not swallowed.

Reproductions: probe_viz_failure.py (renderer failure below ``_generate_visualizations`` vanished)
and the blocked-renderer fixture (``execute()`` stayed blocked past ``visualization_timeout``).
"""

import threading
import time

import pandas as pd
import pytest

import pamola_core.anonymization.masking.full_masking_op as masking_module
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus
from pamola_core.utils.vis_helpers.render_guard import (
    guarded_write,
    prepare_render_context,
    wait_for_render,
)


def _frame():
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol", "Dave"] * 10, "age": [float(v) for v in range(40)],
                         "other": range(40)})


_CASES = {
    "anonymization": lambda: FullMaskingOperation(field_name="name"),
    "suppression": lambda: CellSuppressionOperation(field_name="name", suppression_strategy="null"),
    "transformation": lambda: RemoveFieldsOperation(fields_to_remove=["other"]),
    "profiling": lambda: NumericOperation("age", use_cache=False),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_renderer_that_never_finishes_cannot_block_execute(case, tmp_path, monkeypatch):
    op = _CASES[case]()
    op.generate_visualization = True
    op.visualization_backend = "matplotlib"
    op.visualization_timeout = 1
    release = threading.Event()
    late_write = {}

    def stuck(self, *args, **kwargs):
        release.wait(30)
        try:  # what an abandoned renderer would do once it wakes up
            with guarded_write(tmp_path / "late.png"):
                (tmp_path / "late.png").write_bytes(b"x")
            late_write["done"] = True
        except Exception as exc:  # noqa: BLE001
            late_write["refused"] = type(exc).__name__
        return {}

    monkeypatch.setattr(type(op), "_generate_visualizations", stuck)
    started = time.monotonic()
    try:
        result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert elapsed < 20, f"execute() was blocked for {elapsed:.0f}s by a renderer with a 1s timeout"
    assert "VisualizationTimeoutError" in result.metrics.get("visualization_error", ""), result.metrics
    assert not [a for a in result.artifacts if a.category == "visualization"]
    for t in threading.enumerate():
        if "viz" in t.name.lower():
            t.join(5)
    assert late_write.get("refused") == "VisualizationCancelledError", late_write
    assert not (tmp_path / "late.png").exists()


def test_inner_renderer_failure_reaches_the_result(tmp_path):
    for strict in (False, True):
        op = FullMaskingOperation(field_name="name", use_cache=False, generate_visualization=True,
                                  visualization_backend="matplotlib", visualization_strict=strict)
        boom = RuntimeError("R2_RENDERER_FAILED")
        orig = masking_module.create_category_distribution_comparison
        masking_module.create_category_distribution_comparison = lambda *a, **k: (_ for _ in ()).throw(boom)
        try:
            result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path / str(strict), None)
        finally:
            masking_module.create_category_distribution_comparison = orig
        assert "R2_RENDERER_FAILED" in result.metrics.get("visualization_error", ""), (strict, result.metrics)
        assert not [a for a in result.artifacts if a.category == "visualization"]
        assert result.status == OperationStatus.SUCCESS


def test_guard_removes_files_written_before_the_cancellation(tmp_path):
    ctx, run = prepare_render_context()
    target = tmp_path / "early.png"
    gate = threading.Event()

    def worker():
        with guarded_write(target):
            target.write_bytes(b"x")
        gate.wait(30)

    thread = threading.Thread(target=ctx.run, args=(worker,), daemon=True)
    thread.start()
    time.sleep(0.2)
    assert target.exists()
    assert wait_for_render(thread, run, 0.3, interval=0.1) is False
    gate.set()
    assert not target.exists()
    assert run.cancelled


def test_a_nonpositive_timeout_is_still_finite(tmp_path):
    ctx, run = prepare_render_context()
    thread = threading.Thread(target=ctx.run, args=(lambda: None,), daemon=True)
    thread.start()
    assert wait_for_render(thread, run, 0) is True


def test_the_timeout_error_names_the_effective_limit(tmp_path, monkeypatch):
    """ASB-OPS-12: with the timeout unset (0) the message says the 120 s default, not '0s'."""
    from pamola_core.utils.vis_helpers import render_guard

    monkeypatch.setattr(render_guard, "DEFAULT_VISUALIZATION_TIMEOUT", 1.0)   # keep the test short
    release = threading.Event()

    def stuck(self, *args, **kwargs):
        release.wait(30)
        return {}

    op = FullMaskingOperation(field_name="name")
    op.generate_visualization = True
    op.visualization_backend = "matplotlib"
    op.visualization_timeout = 0
    monkeypatch.setattr(type(op), "_generate_visualizations", stuck)
    try:
        result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)
    finally:
        release.set()
    message = result.metrics.get("visualization_error", "")
    assert "VisualizationTimeoutError" in message and "within 1s" in message, message
    assert "within 0s" not in message and "None" not in message
