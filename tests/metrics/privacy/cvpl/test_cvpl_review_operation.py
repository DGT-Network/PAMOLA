"""Operation-level regressions of the CVPL review (CDX-CVPL-5, 6, 15): cache identity,
persisted artifacts and the public result contract."""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from pamola_core.metrics import CVPLLinkageOperation
from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

GT = {"source_id_column": "id", "release_id_column": "id", "scope": "complete"}
BASE = dict(
    qids=[{"name": "g", "generalized": False}], auxiliary=["x"], qid_as_features=False,
    ground_truth=GT, seed=0,
)


def frame(g, x, ids):
    return pd.DataFrame({"g": g, "x": x, "id": ids})


def op_run(tmp_path, s, r, name="run", **params):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    op = CVPLLinkageOperation(**{**BASE, **params})
    out = op.execute(DataSource(dataframes={"s": s, "r": r}), d, None,
                     original_dataset_name="s", transformed_dataset_name="r")
    return out, d


# ======================================================================================
# CDX-CVPL-5 / 6 / 15: operation level
# ======================================================================================
def test_cdx_cvpl_5_frozen_projection_state_is_part_of_the_cache_identity(tmp_path):
    train1 = frame(["A", "A"], [-1, 1], ["a", "b"])
    train2 = frame(["A", "A"], [9, 11], ["a", "b"])
    p1 = run_cvpl(train1, train1, CVPLConfig.from_dict(BASE)).projection
    p2 = run_cvpl(train2, train2, CVPLConfig.from_dict(BASE)).projection
    assert p1.fingerprint() != p2.fingerprint()
    s, r = frame(["A"], [1], ["a"]), frame(["A"], [-1], ["a"])
    first, _ = op_run(tmp_path, s, r, "cache", fit_strategy="frozen", frozen_projection=p1, use_cache=True)
    second, _ = op_run(tmp_path, s, r, "cache", fit_strategy="frozen", frozen_projection=p2, use_cache=True)
    fresh, _ = op_run(tmp_path, s, r, "nocache", fit_strategy="frozen", frozen_projection=p2, use_cache=False)
    assert first.metrics["cvpl_lr"] == 0.0
    assert second.metrics["cvpl_lr"] == fresh.metrics["cvpl_lr"] == 1.0
    assert not second.metrics.get("cached")
    again, _ = op_run(tmp_path, s, r, "cache", fit_strategy="frozen", frozen_projection=p2, use_cache=True)
    assert again.metrics["cvpl_lr"] == 1.0  # the same fit does hit the cache


CANARY = "CANARY_JANE_DOE_1978_04_15"


def test_cdx_cvpl_6_hierarchy_values_are_never_persisted(tmp_path, caplog):
    s = frame([CANARY, "other"], [-1, 1], ["a", "b"])
    r = frame(["city", "other"], [-1, 1], ["a", "b"])
    with caplog.at_level(logging.DEBUG):
        out, d = op_run(
            tmp_path, s, r, "pii", use_cache=True,
            qids=[{"name": "g", "kind": "categorical", "hierarchy": {CANARY: "city"}}])
    assert out.status == OperationStatus.SUCCESS, out.error_message
    assert out.metrics["cvpl_lr"] == 1.0  # the hierarchy still worked
    leaks = [str(p) for p in Path(d).rglob("*") if p.is_file() and CANARY.encode() in p.read_bytes()]
    assert leaks == []
    assert CANARY not in "\n".join(rec.getMessage() for rec in caplog.records)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["qids"][0]["hierarchy_size"] == 1 and len(cfg["qids"][0]["hierarchy_digest"]) == 64


def test_cdx_cvpl_6_two_different_hierarchies_do_not_share_a_cache_entry(tmp_path):
    s = frame(["Lyon", "Paris"], [-1, 1], ["a", "b"])
    r = frame(["South", "North"], [-1, 1], ["a", "b"])
    good, _ = op_run(tmp_path, s, r, "h", use_cache=True,
                     qids=[{"name": "g", "kind": "categorical", "hierarchy": {"Lyon": "South", "Paris": "North"}}])
    swapped, _ = op_run(tmp_path, s, r, "h", use_cache=True,
                        qids=[{"name": "g", "kind": "categorical", "hierarchy": {"Lyon": "North", "Paris": "South"}}])
    assert good.metrics["cvpl_lr"] == 1.0
    assert swapped.metrics["cvpl_lr"] == 0.0 and not swapped.metrics.get("cached")


def test_cdx_cvpl_15_operation_result_keeps_statuses_bounds_and_nulls(tmp_path):
    s = frame(["A", "B", "C"], [-1, 0, 1], ["a", "b", "c"])
    r = frame(["A", "B", "C"], [-1, 1, 0], ["a", "b", "c"])
    out, _ = op_run(tmp_path, s, r, "api")
    m = out.metrics
    assert m["cvpl_id_uniform_gt"] is None  # null survives
    assert m["cvpl_id_uniform_gt_status"]["status"] == "bounded"
    assert m["cvpl_id_uniform_gt_status"]["upper"] == 1.0
    assert m["cvpl_lr_status"]["lower"] == pytest.approx(1 / 3)
    assert "blocking_ladder_trajectory" in m and "parameters" in m and "projection" in m


def test_cdx_cvpl_15_no_ground_truth_nulls_are_keys_of_the_public_result(tmp_path):
    out, _ = op_run(tmp_path, frame(["A"], [1], ["a"]), frame(["A"], [1], ["a"]), "nogt", ground_truth=None)
    for key in ("cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau", "gt_coverage"):
        assert key in out.metrics and out.metrics[key] is None
    assert out.metrics["cvpl_top1_success_gt_status"]["reason"] == "no_ground_truth"


def test_cdx_cvpl_15_a_cached_run_is_a_successful_run(tmp_path):
    s, r = frame(["A", "B"], [-1, 1], ["a", "b"]), frame(["A", "B"], [-1, 1], ["a", "b"])
    first, _ = op_run(tmp_path, s, r, "again", use_cache=True)
    second, _ = op_run(tmp_path, s, r, "again", use_cache=True)
    assert first.status == OperationStatus.SUCCESS
    assert second.status == OperationStatus.SUCCESS  # used to come back as PENDING
    assert second.metrics["cvpl_lr"] == first.metrics["cvpl_lr"]
