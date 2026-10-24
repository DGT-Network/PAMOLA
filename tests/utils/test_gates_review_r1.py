"""Regression tests for the Codex R1 findings on fix/core10-gates (F1-F9)."""

import json
import os
import pickle

import pandas as pd
import pytest

from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils.artifact_policy import ArtifactClass, ArtifactPolicyError
from pamola_core.utils.io_helpers import crypto_utils
from pamola_core.utils.ops.op_data_reader import DataReader
from pamola_core.utils.safe_deserialization import UnsafeDeserializationError

KEY = "task-key"


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


def _store():
    s = MappingStore()
    s.add_mapping("email", "R1MAPPINGPRIVATE@private.test", "fixed@example.test", False)
    return s


class _Reporter:
    def __init__(self):
        self.artifacts = []

    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        self.artifacts.append(k)


# ---- F4 / F5 / F6 ---------------------------------------------------------------------
def test_f4_encrypted_mapping_write_never_puts_plaintext_on_disk(tmp_path, monkeypatch):
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        with open(src, "rb") as handle:
            seen.append(handle.read())
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    monkeypatch.setattr(
        crypto_utils,
        "encrypt_file",
        lambda *a, **k: pytest.fail("a plaintext staging file would be encrypted"),
    )
    path = tmp_path / "m.json"
    _store().save_json(path, encryption_key=KEY)
    assert seen and b"R1MAPPINGPRIVATE" not in seen[0]
    assert b"R1MAPPINGPRIVATE" not in path.read_bytes()
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"], "leftover file"


def test_f4_a_crash_before_the_replace_leaves_no_plaintext(tmp_path, monkeypatch):
    def crash(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        _store().save_json(tmp_path / "m.json", encryption_key=KEY)
    for p in tmp_path.iterdir():
        assert b"R1MAPPINGPRIVATE" not in p.read_bytes()


def test_f5_io_module_with_a_key_is_rejected_before_writing(tmp_path):
    from pamola_core.errors.exceptions import InvalidParameterError
    from pamola_core.utils import io as pamola_io

    path = tmp_path / "m.json"
    with pytest.raises(InvalidParameterError):
        _store().save_json(path, io_module=pamola_io, encryption_key=KEY)
    assert not path.exists()


def test_f6_incremental_encrypted_load_decrypts_and_a_bad_envelope_raises(tmp_path):
    from pamola_core.errors.exceptions import ValidationError

    path = tmp_path / "m.json"
    _store().save_json(path, encryption_key=KEY)
    target = MappingStore()
    target.add_mapping("email", "other@x.test", "o@example.test", False)
    stats = target.load(path, overwrite_existing=False, encryption_key=KEY)
    assert stats.get("email") == 1
    assert target.mappings["email"]["R1MAPPINGPRIVATE@private.test"] == "fixed@example.test"
    with pytest.raises(ValidationError):  # the envelope is not a mapping schema
        MappingStore().load(path, overwrite_existing=False)
    with pytest.raises(Exception):
        MappingStore().load(path, overwrite_existing=False, encryption_key="wrong")


# ---- F7 -------------------------------------------------------------------------------
def _email_op(**kw):
    from pamola_core.fake_data.operations.email_op import FakeEmailOperation

    return FakeEmailOperation(
        field_name="email", consistency_mechanism="mapping", key="k", **kw
    )


def test_f7_plain_json_input_loads_with_encrypted_output_and_failures_propagate(tmp_path):
    plain = tmp_path / "in.json"
    _store().save_json(plain)  # plain text input, as older runs wrote
    op = _email_op(
        mapping_store_path=str(plain),
        save_mapping=True,
        use_encryption=True,
        encryption_key=KEY,
    )
    assert op.mapping_store.mappings["email"]["R1MAPPINGPRIVATE@private.test"]
    pkl, marker = _evil(tmp_path, "in.pkl")
    with pytest.raises(UnsafeDeserializationError):
        _email_op(mapping_store_path=str(pkl))
    assert not marker.exists()
    enc = tmp_path / "enc.json"
    _store().save_json(enc, encryption_key=KEY)
    with pytest.raises(Exception):
        _email_op(mapping_store_path=str(enc), use_encryption=True, encryption_key="wrong")


def test_f7_save_mapping_without_a_store_is_a_typed_failure(tmp_path):
    op = _email_op(save_mapping=True, use_encryption=True, encryption_key=KEY)
    op.mapping_store = None

    class _R:
        artifacts = []

    with pytest.raises(ArtifactPolicyError):
        op._save_protected_mapping(tmp_path, _R())


# ---- F9 -------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["d.pkl", "d.pickle", "d.PKL"])
@pytest.mark.parametrize("use_dask", [False, True])
def test_f9_reader_refuses_pickle_before_dispatch_and_raises(tmp_path, name, use_dask):
    path, marker = _evil(tmp_path, name)
    with pytest.raises(UnsafeDeserializationError):
        DataReader().read_dataframe(path, use_dask=use_dask)
    with pytest.raises(UnsafeDeserializationError):
        DataReader().read_dataframe(path, file_format="PICKLE", use_dask=use_dask)
    assert not marker.exists()


def test_f9_io_normalizes_pkl_format(tmp_path):
    from pamola_core.utils import io as pamola_io

    path = tmp_path / "d.pkl"
    pd.DataFrame({"x": [1]}).to_pickle(path)
    with pytest.raises(UnsafeDeserializationError):
        pamola_io.read_dataframe(path, file_format="pkl")
    with pytest.warns(RuntimeWarning):
        df = pamola_io.read_dataframe(path, file_format="PKL", allow_unsafe_pickle=True)
    assert len(df) == 1


# ---- F2 / F3 --------------------------------------------------------------------------
def test_f2_split_by_id_metrics_have_counts_only(tmp_path):
    from pamola_core.transformations.splitting.split_by_id_values_op import (
        SplitByIDValuesOperation,
    )
    from pamola_core.utils.ops.op_data_source import DataSource

    df = pd.DataFrame({"id": [f"R1QX7CANARY{i}" for i in range(20)], "v": range(20)})
    op = SplitByIDValuesOperation(id_field="id", number_of_partitions=2)
    res = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert res.status.name == "SUCCESS", res.error_message
    for p in (tmp_path / "t" / "metrics").rglob("*.json"):
        text = p.read_text(encoding="utf-8")
        assert "R1QX7CANARY" not in text and "included_records" not in text
    assert "R1QX7CANARY" not in json.dumps(res.metrics, default=str)


def test_f3_fake_email_default_metrics_have_no_domain_labels(tmp_path):
    from pamola_core.utils.ops.op_data_source import DataSource

    df = pd.DataFrame(
        {"id": range(6), "email": ["a@r1qx7canary.test", "b@r1qx7canary.test"] * 3}
    )
    op = _email_op(preserve_domain_ratio=1.0)
    res = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert res.status.name == "SUCCESS", res.error_message
    found = False
    for p in (tmp_path / "t" / "metrics").rglob("*.json"):
        found = True
        assert "r1qx7canary" not in p.read_text(encoding="utf-8").lower()
    assert found


# ---- F1 / F8 --------------------------------------------------------------------------
def test_f1_reporter_keeps_one_entry_per_path_and_the_assigned_class(tmp_path):
    from pamola_core.utils.tasks.task_reporting import TaskReporter

    rep = TaskReporter(task_id="t", task_type="x", description="d", report_path=tmp_path / "r.json")
    f = tmp_path / "g.json"
    f.write_text("{}")
    rep.add_artifact("json", f, "unclassified", category="output")
    rep.add_artifact(
        "json",
        f,
        "classified",
        category="metrics",
        artifact_class=ArtifactClass.DIAGNOSTIC,
        contains_source_values=True,
    )
    rep.add_artifact("json", f, "again", category="metrics")
    entries = [a for a in rep.artifacts if a["path"] == str(f)]
    assert len(entries) == 1
    assert entries[0]["artifact_class"] == "diagnostic"
    assert entries[0]["contains_source_values"] and entries[0]["not_for_release"]


def test_f8_pseudonymization_mapping_config_and_backup_are_registered(tmp_path):
    from pamola_core.anonymization.pseudonymization.mapping_op import (
        ConsistentMappingPseudonymizationOperation,
    )
    from pamola_core.utils.ops.op_data_source import DataSource

    df = pd.DataFrame({"id": [f"u{i}" for i in range(10)]})
    for _ in range(2):  # the second run backs the mapping up
        op = ConsistentMappingPseudonymizationOperation(
            field_name="id", mapping_encryption_key="a1" * 32, pseudonym_type="uuid"
        )
        res = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
        assert res.status.name == "SUCCESS", res.error_message
    by_name = {a.path.name: a for a in res.artifacts}
    maps = [a for n, a in by_name.items() if ".enc" in n]
    assert maps and all(a.artifact_class is ArtifactClass.PROTECTED_MAPPING for a in maps)
    assert any(".bak." in n for n in by_name), sorted(by_name)
    assert by_name["config.json"].artifact_class is ArtifactClass.DIAGNOSTIC
