"""Regression tests for the Codex R2 findings on fix/core10-gates (R2-1 .. R2-4)."""

import json
import logging
import os
import pickle
import time

import pandas as pd
import pytest

from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils.artifact_policy import ArtifactClass
from pamola_core.utils.io_helpers import crypto_utils
from pamola_core.utils.ops.op_data_reader import DataReader
from pamola_core.utils.safe_deserialization import (
    UnsafeDeserializationError,
    UnsupportedBulkPickleError,
)
from pamola_core.utils.tasks.task_reporting import TaskReporter

KEY = "task-key"
MARK = "R2PRIVATEQZ"


def _reporter(tmp_path):
    return TaskReporter("t", "x", "d", tmp_path / "task_report.json")


def _entry(rep, path):
    entries = [a for a in rep.artifacts if os.path.samefile(a["path"], path)]
    assert len(entries) == 1, entries
    return entries[0]


# ---- R2-1: monotonic merge --------------------------------------------------------------
@pytest.mark.parametrize(
    "first,second",
    [
        (dict(artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=False),
         dict(artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)),
        (dict(category="dictionary"),
         dict(artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False)),
        (dict(artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False),
         dict(artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)),
        (dict(artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True),
         dict(artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False)),
    ],
)
def test_r2_1_merge_never_drops_source_values_or_downgrades(tmp_path, first, second):
    f = tmp_path / "m.json"
    f.write_text("{}")
    rep = _reporter(tmp_path)
    rep.add_artifact("json", f, "a", **first)
    rep.add_artifact("json", f, "b", **second)
    e = _entry(rep, f)
    assert e["contains_source_values"] is True
    assert e["artifact_class"] == "diagnostic"
    assert e["not_for_release"] is True


def test_r2_1_protected_mapping_outranks_diagnostic(tmp_path):
    f = tmp_path / "m.json"
    f.write_text("{}")
    rep = _reporter(tmp_path)
    rep.add_artifact("json", f, "a", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    rep.add_artifact("json", f, "b", artifact_class=ArtifactClass.PROTECTED_MAPPING, contains_source_values=True)
    e = _entry(rep, f)
    # R3-2: the mapping class wins, the earlier diagnostic's release restriction stays
    assert e["artifact_class"] == "protected_mapping" and e["not_for_release"] is True


def test_r2_1_symlink_alias_is_the_same_artifact(tmp_path):
    f = tmp_path / "artifact.json"
    f.write_text("{}")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(f)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    rep = _reporter(tmp_path)
    rep.add_artifact("json", f, "a", artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False)
    rep.add_artifact("json", alias, "b", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    e = _entry(rep, f)
    assert e["artifact_class"] == "diagnostic" and e["contains_source_values"] is True


def test_r2_1_case_alias_on_a_case_insensitive_filesystem(tmp_path):
    f = tmp_path / "Case.json"
    f.write_text("{}")
    other = tmp_path / "case.json"
    if not other.exists():
        pytest.skip("case-sensitive filesystem")
    rep = _reporter(tmp_path)
    rep.add_artifact("json", f, "a", artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False)
    rep.add_artifact("json", other, "b", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    assert len([a for a in rep.artifacts if os.path.samefile(a["path"], f)]) == 1


def test_r2_1_reporter_and_manifest_agree_after_real_finalize(tmp_path, monkeypatch):
    from pamola_core.utils.ops.op_result import OperationResult
    from pamola_core.utils.tasks.base_task import BaseTask

    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(tmp_path / "execution_log.json"))
    out = tmp_path / "t"
    out.mkdir()
    f = out / "metrics.json"
    f.write_text(json.dumps({MARK: 1}))
    rep = TaskReporter("t", "x", "d", out / "task_report.json")
    rep.add_artifact("json", f, "op", artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False)
    result = OperationResult()
    result.add_artifact("json", f, "op", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)

    task = BaseTask("t", "x", "d", task_dir=str(out))
    task.task_dir = out
    task.start_time = time.time()
    task.logger = logging.getLogger("r2-1")
    task.reporter = rep
    task.artifacts = result.artifacts
    task.metrics = {}
    task.finalize(True)

    manifest = {e["path"]: e for e in json.loads((out / "manifest.json").read_text())["artifacts"]}
    r_entry = _entry(rep, f)
    m_entry = manifest[str(f)]
    for key in ("artifact_class", "contains_source_values", "not_for_release"):
        assert r_entry[key] == m_entry[key], key
    assert r_entry["artifact_class"] == "diagnostic" and r_entry["not_for_release"] is True


# ---- R2-2: memory-only reads --------------------------------------------------------------
def _store():
    s = MappingStore()
    s.add_mapping("email", MARK + "@private.test", "fixed@example.test", False)
    return s


def test_r2_2_mapping_reads_never_stage_plaintext(tmp_path, monkeypatch):
    path = tmp_path / "m.json"
    _store().save_json(path, encryption_key=KEY)
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr(
        crypto_utils, "decrypt_file",
        lambda *a, **k: pytest.fail("a decrypted file would be staged on disk"),
    )
    full = MappingStore()
    full.load(path, encryption_key=KEY)
    assert full.mappings["email"][MARK + "@private.test"] == "fixed@example.test"
    inc = MappingStore()
    inc.add_mapping("email", "other@x.test", "o@example.test", False)
    assert inc.load(path, overwrite_existing=False, encryption_key=KEY)["email"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_r2_2_a_crash_while_reading_leaves_no_plaintext(tmp_path, monkeypatch):
    path = tmp_path / "m.json"
    _store().save_json(path, encryption_key=KEY)
    from pamola_core.utils.crypto_helpers.providers.simple_provider import SimpleProvider

    real = SimpleProvider.decrypt_data

    def crash_after_decrypt(self, *a, **k):
        real(self, *a, **k)
        raise KeyboardInterrupt

    monkeypatch.setattr(SimpleProvider, "decrypt_data", crash_after_decrypt)
    with pytest.raises(KeyboardInterrupt):
        MappingStore().load(path, encryption_key=KEY)
    for p in tmp_path.iterdir():
        assert MARK.encode() not in p.read_bytes()


def test_r2_2_hierarchy_reader_decrypts_in_memory(tmp_path, monkeypatch):
    from pamola_core.anonymization.commons.hierarchy_dictionary import HierarchyDictionary

    path = tmp_path / "hierarchy.json"
    crypto_utils.write_encrypted_json(
        {"levels": ["group"], "data": {MARK: {"group": "safe"}}}, path, KEY
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr(
        crypto_utils, "decrypt_file", lambda *a, **k: pytest.fail("staged plaintext")
    )
    h = HierarchyDictionary()
    h.load_from_file(path, encryption_key=KEY)
    assert h.get_hierarchy(MARK) == "safe" or h.get_hierarchy(MARK)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---- R2-3 -----------------------------------------------------------------------------
def test_r2_3_task_products_get_a_class_and_agree_in_manifest(tmp_path, monkeypatch):
    from pamola_core.utils.tasks.base_task import BaseTask

    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(tmp_path / "execution_log.json"))
    out = tmp_path / "t"
    out.mkdir()
    rep = TaskReporter("t", "x", "d", out / "task_report.json")
    task = BaseTask("t", "x", "d", task_dir=str(out))
    task.task_dir = out
    task.start_time = time.time()
    task.logger = logging.getLogger("r2-3")
    task.reporter = rep
    task.artifacts = []
    task.metrics = {}
    task.finalize(True)
    manifest = {os.path.basename(e["path"]): e for e in json.loads((out / "manifest.json").read_text())["artifacts"]}
    for name in ("task_report.json", "manifest.json"):
        assert manifest[name]["artifact_class"] == "diagnostic", name
        assert manifest[name]["contains_source_values"] is True
        assert manifest[name]["not_for_release"] is True


def test_r2_3_config_of_a_failed_operation_is_registered(tmp_path):
    from pamola_core.fake_data.operations.email_op import FakeEmailOperation
    from pamola_core.utils.ops.op_data_source import DataSource

    op = FakeEmailOperation(field_name="absent")
    res = op.execute(
        DataSource(dataframes={"main": pd.DataFrame({"email": ["a@b.test"]})}), tmp_path / "t", None
    )
    assert res.status.name == "ERROR"
    cfg = [a for a in res.artifacts if a.path.name == "config.json"]
    assert cfg and cfg[0].artifact_class is ArtifactClass.DIAGNOSTIC


def test_r2_3_reporter_entry_copies_the_results_class(tmp_path, monkeypatch):
    from pamola_core.utils.ops.op_result import OperationResult
    from pamola_core.utils.tasks.base_task import BaseTask

    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(tmp_path / "execution_log.json"))
    out = tmp_path / "t"
    out.mkdir()
    f = out / "metrics.json"
    f.write_text("{}")
    rep = TaskReporter("t", "x", "d", out / "task_report.json")
    rep.add_artifact("json", f, "reporter call without class")  # derived: data
    result = OperationResult()
    result.add_artifact("json", f, "op", category="metrics")  # aggregate_metric
    task = BaseTask("t", "x", "d", task_dir=str(out))
    task.task_dir = out
    task.start_time = time.time()
    task.logger = logging.getLogger("r2-3b")
    task.reporter = rep
    task.artifacts = result.artifacts
    task.metrics = {}
    task.finalize(True)
    assert _entry(rep, f)["artifact_class"] == "aggregate_metric"


# ---- R2-4 -----------------------------------------------------------------------------
class _Boom:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _evil(tmp_path, name):
    marker = tmp_path / ("ran_" + name)
    path = tmp_path / name
    with open(path, "wb") as f:
        pickle.dump(_Boom(marker), f)
    return path, marker


def test_r2_4_read_similar_files_raises_the_typed_refusal(tmp_path):
    from pamola_core.utils import io as pamola_io

    path, marker = _evil(tmp_path, "a.pkl")
    with pytest.raises(UnsafeDeserializationError):
        pamola_io.read_similar_files(tmp_path, pattern="*.pkl", ignore_errors=False)
    assert not marker.exists()


@pytest.mark.parametrize("shape", ["list", "dict"])
def test_r2_4_explicit_format_does_not_hide_a_pickle_in_multi_file_sources(tmp_path, shape):
    path, marker = _evil(tmp_path, "a.pkl")
    source = [path] if shape == "list" else {"x": path}
    with pytest.raises(UnsafeDeserializationError):
        DataReader().read_dataframe(source, file_format="csv")
    assert not marker.exists()


def test_r2_4_opted_in_pickle_with_dask_or_multi_file_is_an_explicit_unsupported_error(tmp_path):
    path = tmp_path / "ok.pkl"
    pd.DataFrame({"x": [1, 2]}).to_pickle(path)
    for kwargs in (dict(use_dask=True), dict()):
        source = path if kwargs else [path]
        with pytest.raises(UnsupportedBulkPickleError):
            DataReader().read_dataframe(source, allow_unsafe_pickle=True, **kwargs)
    renamed = tmp_path / "ok.bin"
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(UnsupportedBulkPickleError):
        DataReader().read_dataframe(renamed, file_format="pickle", use_dask=True, allow_unsafe_pickle=True)


def test_r2_4_single_file_opt_in_still_loads(tmp_path):
    path = tmp_path / "ok.pkl"
    pd.DataFrame({"x": [1, 2]}).to_pickle(path)
    with pytest.warns(RuntimeWarning):
        df, err = DataReader().read_dataframe(path, allow_unsafe_pickle=True)
    assert err is None and len(df) == 2
