"""Visualization is opt-in: VIZ-01 (default), VIZ-02 (zero work when off).

The default used to be ``generate_visualization=True``. Every operation then
started a render thread and blocked on it (28-87 % of wall time in the Section 15
baseline) and, on a base install without the ``viz`` extra, logged ~12
WARNING/ERROR lines with tracebacks per call. These tests pin the opposite
contract: a default-constructed operation does no visualization work at all.
"""

import logging
import threading

import pandas as pd
import pytest

from pamola_core.anonymization.generalization.numeric_op import (
    NumericGeneralizationOperation,
)
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.noise.uniform_numeric_op import (
    UniformNumericNoiseOperation,
)
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig

_VIZ_WORDS = ("visuali", "viz", "plotly", "matplotlib", "kaleido", "[diag]")
_PLOT_SUFFIXES = {".png", ".html", ".svg", ".jpg", ".jpeg"}


class _NoopOperation(BaseOperation):
    def execute(self, *args, **kwargs):  # pragma: no cover - never executed
        return OperationResult(status=OperationStatus.SUCCESS)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave"] * 25,
            "age": list(range(20, 120)),
            "other": list(range(100)),
        }
    )


# One operation per code path that carries its own visualization wiring:
# the anonymization base, a subclass with a private copy of the thread block
# (numeric generalization), noise, suppression, transformation and profiling.
_CASES = {
    "full_masking": lambda: FullMaskingOperation(field_name="name"),
    "numeric_generalization": lambda: NumericGeneralizationOperation(
        field_name="age", strategy="binning", bin_count=4
    ),
    "uniform_numeric_noise": lambda: UniformNumericNoiseOperation(
        field_name="age", noise_range=2.0, noise_type="additive"
    ),
    "cell_suppression": lambda: CellSuppressionOperation(
        field_name="name", suppression_strategy="null"
    ),
    "remove_fields": lambda: RemoveFieldsOperation(fields_to_remove=["other"]),
    "numeric_profile": lambda: NumericOperation("age", use_cache=False),
}


def test_base_operation_default_is_off():
    """VIZ-01: the single framework default."""
    assert _NoopOperation().generate_visualization is False


@pytest.mark.parametrize("case", sorted(_CASES))
def test_public_operations_default_is_off(case):
    """VIZ-01: no subclass re-declares the old True default."""
    assert _CASES[case]().generate_visualization is False


def test_config_schema_default_is_off():
    """VIZ-01: UI/config schema agrees with the constructor default."""
    schema = BaseOperationConfig.schema
    assert schema["properties"]["generate_visualization"]["default"] is False


@pytest.mark.parametrize("case", sorted(_CASES))
def test_default_run_does_no_visualization_work(case, tmp_path, monkeypatch, caplog):
    """VIZ-02: no worker thread, no plot file, no viz-related warning/error."""
    started = []
    real_start = threading.Thread.start

    def recording_start(self, *args, **kwargs):
        started.append(self.name)
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", recording_start)

    op = _CASES[case]()
    with caplog.at_level(logging.DEBUG):
        result = op.execute(
            DataSource(dataframes={"main": _frame()}), tmp_path, None
        )

    assert result.status == OperationStatus.SUCCESS, result.error_message

    assert [n for n in started if "viz" in n.lower()] == []
    assert [p for p in tmp_path.rglob("*") if p.suffix.lower() in _PLOT_SUFFIXES] == []
    assert not [
        a for a in result.artifacts if getattr(a, "category", None) == "visualization"
    ]
    noisy = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and any(w in r.getMessage().lower() for w in _VIZ_WORDS)
    ]
    assert noisy == []
    # [DIAG] is the thread machinery's tracing; it must not run when viz is off.
    assert not [r for r in caplog.records if "[DIAG]" in r.getMessage()]


# --- VIZ-03: an explicit request is fulfilled before execute() returns --------


@pytest.mark.parametrize("case", sorted(_CASES))
def test_explicit_visualization_is_complete_when_execute_returns(case, tmp_path):
    """VIZ-03: artifacts exist on disk and are registered, no worker outlives the call."""
    pytest.importorskip("matplotlib")
    op = _CASES[case]()
    op.generate_visualization = True
    # matplotlib needs no browser; plotly PNG export goes through kaleido/Chrome.
    op.visualization_backend = "matplotlib"

    result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert result.status == OperationStatus.SUCCESS, result.error_message
    registered = [a for a in result.artifacts if a.category == "visualization"]
    assert registered, "requested visualization produced no registered artifact"
    for artifact in registered:
        assert artifact.path.exists(), f"registered but not on disk: {artifact.path}"
    assert [t.name for t in threading.enumerate() if "viz" in t.name.lower()] == []


# --- VIZ-04: a base install gets one actionable error, not a traceback storm ---


@pytest.fixture
def no_viz_extra(monkeypatch):
    """Make every plotting backend look uninstalled, as on a base install."""
    import importlib

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in {"matplotlib", "plotly", "kaleido", "seaborn"}:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "pamola_core.utils.optional_deps.importlib.import_module", fake_import
    )


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_missing_viz_extra_yields_exactly_one_actionable_error(
    case, backend, no_viz_extra, tmp_path, caplog
):
    op = _CASES[case]()
    op.generate_visualization = True
    op.visualization_backend = backend

    with caplog.at_level(logging.DEBUG):
        result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert result.status == OperationStatus.ERROR
    assert "pamola-core[viz]" in result.error_message
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(loud) == 1, [r.getMessage() for r in loud]
    assert "pamola-core[viz]" in loud[0].getMessage()
    assert loud[0].exc_info is None, "the error must not drag a traceback along"
    # The check runs before any work: nothing was read, computed or written.
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("case", sorted(_CASES))
def test_missing_viz_extra_is_irrelevant_when_visualization_is_off(
    case, no_viz_extra, tmp_path, caplog
):
    """The default run on a base install: success and no viz-related noise."""
    op = _CASES[case]()
    with caplog.at_level(logging.DEBUG):
        result = op.execute(DataSource(dataframes={"main": _frame()}), tmp_path, None)

    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert not [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and any(w in r.getMessage().lower() for w in _VIZ_WORDS)
    ]
