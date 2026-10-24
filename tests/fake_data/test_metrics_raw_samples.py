"""
SEC-04 / SEC-05 for fake-data operations: the metrics artifact must not carry
the most frequent *original* values unless ``include_raw_samples_in_metrics``
is set.

Pre-fix, ``MetricsCollector.collect_data_stats`` wrote ``value_distribution``
(top 20 values with their shares) of the original data to ``metrics/*.json``.
"""

import importlib

import pandas as pd
import pytest

from pamola_core.fake_data.commons.metrics import MetricsCollector
from pamola_core.fake_data.operations.email_op import FakeEmailOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

PII = [
    "zqcanary.alice@pii-canary.test",
    "zqcanary.bob@pii-canary.test",
    "zqcanary.carol@pii-canary.test",
]


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _metrics_text(task_dir):
    return "".join(
        p.read_text(encoding="utf-8")
        for p in task_dir.rglob("*.json")
        if p.parent.name == "metrics"
    )


def _run(tmp_path, **kwargs):
    df = pd.DataFrame({"id": range(6), "email": PII * 2})
    op = FakeEmailOperation(field_name="email", **kwargs)
    result = op.execute(DataSource(dataframes={"main": df}), tmp_path / "task", _Reporter())
    assert result.status == OperationStatus.SUCCESS, result.error_message
    return op, _metrics_text(tmp_path / "task")


def test_default_metrics_artifact_has_no_original_values(tmp_path):
    op, text = _run(tmp_path)
    assert op.include_raw_samples_in_metrics is False
    assert text, "no metrics artifact written"
    assert "zqcanary" not in text
    assert "value_distribution" not in text


def test_opt_in_restores_the_distribution(tmp_path):
    _, text = _run(tmp_path, include_raw_samples_in_metrics=True)
    assert "zqcanary" in text and "value_distribution" in text


def test_collector_default_keeps_counts_only():
    stats = MetricsCollector().collect_data_stats(pd.Series(PII * 2))
    assert "value_distribution" not in stats
    assert stats["unique_values"] == 3 and stats["total_records"] == 6
    raw = MetricsCollector(include_raw_samples=True).collect_data_stats(pd.Series(PII))
    assert set(raw["value_distribution"]) == set(PII)


@pytest.mark.parametrize("name", ["email", "name", "organization", "phone"])
def test_option_is_declared_in_each_core_schema(name):
    module = importlib.import_module(f"pamola_core.fake_data.schemas.{name}_op_core_schema")
    config = next(
        v
        for v in vars(module).values()
        if isinstance(v, type) and v.__module__ == module.__name__ and hasattr(v, "schema")
    )
    prop = config.schema["allOf"][1]["properties"]["include_raw_samples_in_metrics"]
    assert prop["default"] is False and "WARNING" in prop["description"]
