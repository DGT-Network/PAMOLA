"""ASB-4: DataAttributeProfilerOperation writes raw sample values only on request, reproducibly."""

import json

import pandas as pd

from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

CANARY = "ZQCANARY"


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _df():
    return pd.DataFrame({
        "customer_name": [f"{CANARY}-{i}" for i in range(60)],
        "city": [f"{CANARY}City{i % 7}" for i in range(60)],
        "age": list(range(60)),
    })


def _run(tmp_path, name, **kw):
    op = DataAttributeProfilerOperation(generate_visualization=False, use_cache=False, sample_size=5, **kw)
    out = op.execute(DataSource(dataframes={"main": _df()}), tmp_path / name, _Reporter())
    assert out.status == OperationStatus.SUCCESS, out.error_message
    return {p.name: p.read_text(encoding="utf-8") for p in (tmp_path / name).rglob("*") if p.is_file()}


def test_asb4_default_run_writes_no_raw_values_anywhere(tmp_path):
    files = _run(tmp_path, "off")
    assert files
    assert [n for n, t in files.items() if CANARY in t] == []
    assert not [n for n in files if n.startswith("attribute_sample_")]
    roles = next(t for n, t in files.items() if n.startswith("attribute_roles_"))
    assert all(c["statistics"]["samples"] == [] for c in json.loads(roles)["columns"].values())


def test_asb4_option_declared_off_by_default():
    op = DataAttributeProfilerOperation()
    assert op.include_raw_samples_in_metrics is False
    prop = op.config.schema["allOf"][1]["properties"]["include_raw_samples_in_metrics"]
    assert prop["default"] is False and "WARNING" in prop["description"]


def test_asb4_opt_in_writes_samples_and_they_are_reproducible(tmp_path):
    a = _run(tmp_path, "a", include_raw_samples_in_metrics=True, seed=7)
    b = _run(tmp_path, "b", include_raw_samples_in_metrics=True, seed=7)
    c = _run(tmp_path, "c", include_raw_samples_in_metrics=True)
    e = _run(tmp_path, "e", include_raw_samples_in_metrics=True)  # no seed: a fixed default, still reproducible
    sample = lambda files: next(t for n, t in files.items() if n.startswith("attribute_sample_"))  # noqa: E731
    assert CANARY in sample(a)
    assert json.loads(sample(a)) == json.loads(sample(b))  # same seed, same samples
    assert json.loads(sample(c)) == json.loads(sample(e))
