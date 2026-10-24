"""Regression tests for the Codex R3 findings on fix/core10-gates (R3-1 .. R3-5)."""

import json
import logging
import os
import time

import pytest

from pamola_core.errors.exceptions import DecryptionError
from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils.artifact_policy import ArtifactClass
from pamola_core.utils.io_helpers import crypto_utils
from pamola_core.utils.io_helpers.crypto_utils import (
    UnsupportedMemoryOnlyModeError,
    read_encrypted_json,
)
from pamola_core.utils.tasks.task_reporting import TaskReporter

KEY = "task-key"
INJECTED = {"email": [{"original": "attacker", "synthetic": "replacement"}]}
PAYLOAD = {"mappings": {"email": [{"original": "real", "synthetic": "fixed"}]}}


def _envelope(tmp_path, name="e.json"):
    env = crypto_utils.encrypt_data(json.dumps(PAYLOAD).encode(), key=KEY, mode="simple")
    path = tmp_path / name
    path.write_text(json.dumps(env))
    return path, env


def _write(path, doc):
    path.write_text(json.dumps(doc))
    return path


# ---- R3-1 -----------------------------------------------------------------------------
def test_r3_1_removing_mode_and_injecting_mappings_is_rejected_for_any_key(tmp_path):
    _, env = _envelope(tmp_path)
    bad = dict(env)
    bad.pop("mode")
    bad["mappings"] = INJECTED
    path = _write(tmp_path / "bad.json", bad)
    for key in (KEY, "wrong"):
        with pytest.raises(DecryptionError):
            read_encrypted_json(path, key)
        with pytest.raises(DecryptionError):
            MappingStore().load(path, encryption_key=key)


def test_r3_1_mode_none_with_crypto_fields_is_rejected(tmp_path):
    _, env = _envelope(tmp_path)
    bad = dict(env, mode="none", data=json.dumps({"mappings": INJECTED}))
    path = _write(tmp_path / "none.json", bad)
    for key in (KEY, "wrong"):
        with pytest.raises(DecryptionError):
            read_encrypted_json(path, key)


def test_r3_1_a_legacy_envelope_without_mode_decrypts_and_a_wrong_key_fails(tmp_path):
    from pamola_core.utils import crypto as legacy

    env = legacy.encrypt_data(json.dumps(PAYLOAD), KEY)
    path = _write(tmp_path / "legacy.json", env if isinstance(env, dict) else json.loads(env))
    assert "mode" not in json.loads(path.read_text())
    assert read_encrypted_json(path, KEY) == PAYLOAD
    with pytest.raises(DecryptionError):
        read_encrypted_json(path, "wrong")


def test_r3_1_current_envelope_round_trips_wrong_key_and_tampering_fail(tmp_path):
    path, env = _envelope(tmp_path)
    assert read_encrypted_json(path, KEY) == PAYLOAD
    with pytest.raises(DecryptionError):
        read_encrypted_json(path, "wrong")
    tampered = dict(env)
    tampered["data"] = tampered["data"][:-4] + ("AAAA" if tampered["data"][-4:] != "AAAA" else "BBBB")
    with pytest.raises(DecryptionError):
        read_encrypted_json(_write(tmp_path / "t.json", tampered), KEY)


def test_r3_1_only_a_document_without_envelope_keys_is_plain_json(tmp_path):
    plain = _write(tmp_path / "p.json", PAYLOAD)
    assert read_encrypted_json(plain, KEY) == PAYLOAD
    assert read_encrypted_json(plain, None) == PAYLOAD
    hierarchy = _write(tmp_path / "h.json", {"levels": ["g"], "data": {"x": {"g": "y"}}})
    assert read_encrypted_json(hierarchy, KEY)["data"] == {"x": {"g": "y"}}  # data as an object
    with pytest.raises(DecryptionError):  # a ciphertext string is an envelope key
        read_encrypted_json(_write(tmp_path / "s.json", {"mappings": INJECTED, "data": "AAAA"}), KEY)


def test_r3_1_age_is_an_explicit_typed_unsupported_error(tmp_path):
    age = tmp_path / "a.age"
    age.write_bytes(b"age-encryption.org/v1\n-> X25519 abc\n")
    with pytest.raises(UnsupportedMemoryOnlyModeError):
        read_encrypted_json(age, KEY)
    env = _write(tmp_path / "m.json", {"mode": "age", "data": "x"})
    with pytest.raises(UnsupportedMemoryOnlyModeError):
        read_encrypted_json(env, KEY)


# ---- R3-2 / R3-3 -----------------------------------------------------------------------
def _reporter(tmp_path):
    return TaskReporter("t", "x", "d", tmp_path / "task_report.json")


def _one(rep, path):
    entries = [a for a in rep.artifacts if os.path.samefile(a["path"], path)]
    assert len(entries) == 1, entries
    return entries[0]


@pytest.mark.parametrize("order", ["diagnostic_first", "mapping_first"])
def test_r3_2_protected_mapping_never_clears_the_release_restriction(tmp_path, order):
    f = tmp_path / "x.csv"
    f.write_text("a")
    rep = _reporter(tmp_path)
    diag = dict(artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    pm = dict(artifact_class=ArtifactClass.PROTECTED_MAPPING, contains_source_values=True)
    first, second = (diag, pm) if order == "diagnostic_first" else (pm, diag)
    rep.add_artifact("csv", f, "a", **first)
    rep.add_artifact("csv", f, "b", **second)
    e = _one(rep, f)
    assert e["artifact_class"] == "protected_mapping"
    assert e["contains_source_values"] is True
    assert e["not_for_release"] is True


def test_r3_3_an_inferred_class_never_overrides_an_explicit_one(tmp_path):
    f = tmp_path / "out.csv"
    f.write_text("a")
    rep = _reporter(tmp_path)
    rep.add_artifact("csv", f, "data", artifact_class=ArtifactClass.DATA, contains_source_values=True)
    rep.add_artifact("csv", f, "inferred", category="dataset")  # unknown category -> diagnostic
    e = _one(rep, f)
    assert e["artifact_class"] == "data"
    assert e["not_for_release"] is False


def test_r3_3_explicit_conflicts_still_use_the_monotonic_rule(tmp_path):
    f = tmp_path / "m.json"
    f.write_text("{}")
    rep = _reporter(tmp_path)
    rep.add_artifact("json", f, "a", artifact_class=ArtifactClass.DATA, contains_source_values=True)
    rep.add_artifact("json", f, "b", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    e = _one(rep, f)
    assert e["artifact_class"] == "diagnostic" and e["not_for_release"] is True


# ---- R3-4 -----------------------------------------------------------------------------
def test_r3_4_aliases_registered_before_the_file_exists_are_reconciled(tmp_path):
    a = tmp_path / "Future.json"
    b = tmp_path / "future.json"
    rep = _reporter(tmp_path)
    rep.add_artifact("json", a, "a", artifact_class=ArtifactClass.AGGREGATE_METRIC, contains_source_values=False)
    rep.add_artifact("json", b, "b", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    a.write_text("{}")
    if not b.exists():
        pytest.skip("case-sensitive filesystem: the two names are different files")
    rep.reconcile_aliases()
    entries = [e for e in rep.artifacts if os.path.samefile(e["path"], a)]
    assert len(entries) == 1
    assert entries[0]["artifact_class"] == "diagnostic" and entries[0]["not_for_release"] is True
    saved = rep.generate_report()
    assert len([e for e in saved["artifacts"] if str(e["path"]).lower().endswith("future.json")]) == 1


def test_r3_4_a_registration_merges_every_matching_alias(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(f)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    rep = _reporter(tmp_path)
    # two entries that were not yet recognisable as one file
    rep.artifacts.append(
        {"path": str(tmp_path / "ghost1"), "artifact_class": "aggregate_metric",
         "contains_source_values": False, "not_for_release": False, "class_explicit": True}
    )
    rep.artifacts.append(
        {"path": str(tmp_path / "ghost2"), "artifact_class": "aggregate_metric",
         "contains_source_values": False, "not_for_release": False, "class_explicit": True}
    )
    rep.artifacts[0]["path"] = str(f)
    rep.artifacts[1]["path"] = str(link)  # aliases of one file, never merged
    rep.add_artifact("json", f, "diag", artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True)
    assert len(rep.artifacts) == 1
    assert rep.artifacts[0]["artifact_class"] == "diagnostic"


# ---- R3-5 -----------------------------------------------------------------------------
def test_r3_5_a_pre_registered_task_report_is_in_the_manifest_exactly_once(tmp_path, monkeypatch):
    from pamola_core.utils.tasks.base_task import BaseTask

    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(tmp_path / "execution_log.json"))
    out = tmp_path / "t"
    out.mkdir()
    rep = TaskReporter("t", "x", "d", out / "task_report.json")
    rep.add_artifact(
        "json", out / "task_report.json", "pre-registered", category="report",
        artifact_class=ArtifactClass.DIAGNOSTIC, contains_source_values=True,
    )
    task = BaseTask("t", "x", "d", task_dir=str(out))
    task.task_dir = out
    task.start_time = time.time()
    task.logger = logging.getLogger("r3-5")
    task.reporter = rep
    task.artifacts = []
    task.metrics = {}
    assert task.finalize(True)
    manifest = json.loads((out / "manifest.json").read_text())["artifacts"]
    reports = [e for e in manifest if os.path.basename(e["path"]) == "task_report.json"]
    assert len(reports) == 1
    assert reports[0]["artifact_class"] == "diagnostic" and reports[0]["not_for_release"] is True


def test_r3_op_base_registration_typeerror_is_logged(tmp_path, caplog):
    from pamola_core.utils.ops.op_base import _register_written_files
    from pamola_core.utils.ops.op_result import OperationResult

    (tmp_path / "config.json").write_text("{}")

    class _Reporter:
        def add_artifact(self, *a, **k):
            raise TypeError("old signature")

    with caplog.at_level(logging.WARNING):
        _register_written_files(OperationResult(), tmp_path, _Reporter())
    assert any("could not register" in r.getMessage() for r in caplog.records)
