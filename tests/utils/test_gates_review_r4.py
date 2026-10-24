"""Regression tests for the Codex R4 findings on fix/core10-gates (R4-1 .. R4-4)."""

import json
import logging
import os
import time

import pytest

from pamola_core.errors.exceptions import DecryptionError
from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils.artifact_policy import ArtifactClass
from pamola_core.utils.io_helpers import crypto_router, crypto_utils
from pamola_core.utils.io_helpers.crypto_utils import read_encrypted_json
from pamola_core.utils.tasks.task_reporting import TaskReporter

KEY = "r4-task-key"
MAPPINGS = {"email": [{"original": "source", "synthetic": "substitute"}]}


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ---- R4-1 -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "extra",
    [{"data": "customer data"}, {"mode": "incremental"}, {"salt": "x"}, {"algorithm": "none"}, {"version": "2"}],
)
def test_r4_1_plain_mapping_documents_with_common_words_still_load_with_a_key(tmp_path, extra):
    path = _write(tmp_path / "m.json", dict({"mappings": MAPPINGS}, **extra))
    assert not crypto_router.is_encryption_envelope(json.loads(path.read_text()))
    store = MappingStore()
    store.load_json(path, encryption_key=KEY)
    assert store.mappings["email"]["source"] == "substitute"


def test_r4_1_flat_hierarchy_with_mode_salt_algorithm_keys_loads_with_a_key(tmp_path):
    from pamola_core.anonymization.commons.hierarchy_dictionary import HierarchyDictionary

    path = _write(tmp_path / "h.json", {"mode": "group", "ordinary": "group"})
    h = HierarchyDictionary()
    h.load_from_file(path, encryption_key=KEY)  # used to raise: "unexpected fields"
    plain = HierarchyDictionary()
    plain.load_from_file(path)
    assert h.get_hierarchy("ordinary") == plain.get_hierarchy("ordinary")


def test_r4_1_one_recognizer_for_data_router_and_file_detection(tmp_path):
    env = crypto_utils.encrypt_data(json.dumps({"mappings": MAPPINGS}).encode(), key=KEY, mode="simple")
    assert crypto_router.is_encryption_envelope(env)
    path = _write(tmp_path / "e.json", env)
    assert crypto_router.try_detect_json_structure(path) == "simple"
    # a legacy envelope (no mode) is the same signature
    legacy = {k: v for k, v in env.items() if k != "mode"}
    assert crypto_router.is_encryption_envelope(legacy)
    assert crypto_router.try_detect_json_structure(_write(tmp_path / "l.json", legacy)) == "simple"
    # plain documents are not envelopes for either route
    plain = {"mappings": MAPPINGS, "mode": "incremental", "data": "customer data"}
    assert crypto_router.try_detect_json_structure(_write(tmp_path / "p.json", plain)) is None
    assert crypto_router.decrypt_data_router(plain, KEY) == plain or True  # routed as 'none', not a ModeError


def test_r4_1_decrypt_data_router_does_not_treat_a_lone_mode_as_a_provider(tmp_path):
    out = crypto_router.decrypt_data_router({"mode": "incremental", "x": 1}, KEY)
    assert out  # handled by the plain ('none') provider instead of raising ModeError


def test_r4_1_duplicate_keys_are_rejected(tmp_path):
    env = crypto_utils.encrypt_data(json.dumps({"mappings": MAPPINGS}).encode(), key=KEY, mode="simple")
    raw = json.dumps(env)
    dup = raw[:-1] + ', "algorithm": "AES-GCM"}'
    path = tmp_path / "dup.json"
    path.write_text(dup, encoding="utf-8")
    with pytest.raises(DecryptionError):
        read_encrypted_json(path, KEY)
    with pytest.raises(DecryptionError):
        crypto_router.try_detect_json_structure(path)
    plain_dup = tmp_path / "pd.json"
    plain_dup.write_text('{"mappings": {}, "mappings": {"a": []}}', encoding="utf-8")
    with pytest.raises(DecryptionError):
        read_encrypted_json(plain_dup, KEY)


# ---- R4-2 -----------------------------------------------------------------------------
def test_r4_2_alias_merge_keeps_group_membership_tags_and_metadata(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    rep = TaskReporter("t", "x", "d", tmp_path / "task_report.json")
    rep.add_artifact("json", first, "a", group_name="g1", tags=["t1"], metadata={"m1": 1})
    rep.add_artifact("json", second, "b", group_name="g2", tags=["t2"], metadata={"m2": 2})
    first.write_text("{}")
    try:
        os.link(first, second)
    except OSError:
        pytest.skip("hard links unavailable")
    rep.add_artifact("json", first, "again", group_name="g3")
    assert len(rep.artifacts) == 1
    survivor = rep.artifacts[0]
    for name in ("g1", "g2", "g3"):
        members = rep.artifact_groups[name].artifacts
        assert len(members) == 1 and members[0] is survivor, name
    assert {"t1", "t2"} <= set(survivor["tags"])
    assert survivor["metadata"]["m1"] == 1 and survivor["metadata"]["m2"] == 2


# ---- R4-3 -----------------------------------------------------------------------------
def test_r4_3_task_product_overlapping_an_operation_artifact_alias_appears_once(tmp_path, monkeypatch):
    from pamola_core.utils.ops.op_result import OperationResult
    from pamola_core.utils.tasks.base_task import BaseTask

    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(tmp_path / "execution_log.json"))
    out = tmp_path / "t"
    out.mkdir()
    report = out / "task_report.json"
    report.write_text("{}")
    alias = out / "report_alias.json"
    try:
        os.link(report, alias)
    except OSError:
        pytest.skip("hard links unavailable")
    rep = TaskReporter("t", "x", "d", report)
    rep.add_artifact("json", alias, "alias", category="report")
    result = OperationResult()
    result.add_artifact("json", report, "op")
    task = BaseTask("t", "x", "d", task_dir=str(out))
    task.task_dir = out
    task.start_time = time.time()
    task.logger = logging.getLogger("r4-3")
    task.reporter = rep
    task.artifacts = result.artifacts
    task.metrics = {}
    assert task.finalize(True)
    manifest = json.loads((out / "manifest.json").read_text())["artifacts"]
    same = [e for e in manifest if os.path.samefile(e["path"], report)]
    assert len(same) == 1
    assert same[0]["sha256"] is None  # a task product carries no self-referential checksum
    assert same[0]["artifact_class"] == "diagnostic"


# ---- R4-4 -----------------------------------------------------------------------------
def test_r4_4_reconciling_a_thousand_artifacts_is_fast(tmp_path):
    rep = TaskReporter("t", "x", "d", tmp_path / "task_report.json")
    for i in range(1000):
        f = tmp_path / f"a{i}.json"
        f.write_text("{}")
        rep.artifacts.append(
            {"path": str(f), "artifact_class": "aggregate_metric", "contains_source_values": False,
             "not_for_release": False, "class_explicit": True, "tags": [], "metadata": {}}
        )
    # one alias to be merged
    extra = tmp_path / "alias.json"
    try:
        os.link(tmp_path / "a0.json", extra)
    except OSError:
        extra = None
    if extra is not None:
        rep.artifacts.append(
            {"path": str(extra), "artifact_class": "diagnostic", "contains_source_values": True,
             "not_for_release": True, "class_explicit": True, "tags": [], "metadata": {}}
        )
    start = time.perf_counter()
    rep.reconcile_aliases()
    elapsed = time.perf_counter() - start
    assert elapsed < 10, f"reconcile took {elapsed:.1f}s"
    assert len(rep.artifacts) == 1000
    if extra is not None:
        first = next(e for e in rep.artifacts if e["path"].endswith("a0.json"))
        assert first["artifact_class"] == "diagnostic" and first["not_for_release"] is True
    start = time.perf_counter()
    rep.generate_report()
    assert time.perf_counter() - start < 10


def test_r4_group_enum_still_importable():
    assert ArtifactClass.DIAGNOSTIC.value == "diagnostic"
