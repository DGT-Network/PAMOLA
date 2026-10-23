"""CVPLLinkageOperation: framework contract (alignment, cache), artifacts and privacy."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pamola_core.errors.exceptions import InvalidParameterError
from pamola_core.metrics import CVPLLinkageOperation, FidelityOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

from .helpers import GT, anchors, make_df, release_frame, source_frame

OPS = {
    "qids": [{"name": "g", "generalized": False}], "auxiliary": ["x"], "qid_as_features": False,
    "tau": 0.5, "ground_truth": GT,
}


def _run(op, source, release, task_dir, **kw):
    return op.execute(
        DataSource(dataframes={"main": source, "rel": release}), task_dir, None,
        original_dataset_name="main", transformed_dataset_name="rel", **kw,
    )


def _metrics_file(task_dir: Path) -> dict:
    path = next((task_dir / "metrics").glob("CVPLLinkageOperation_*.json"))
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def test_cvpl_operation_is_registered_exported_and_opts_out_of_alignment():
    import pamola_core
    assert pamola_core.CVPLLinkageOperation is CVPLLinkageOperation
    assert CVPLLinkageOperation.requires_aligned_inputs is False
    assert FidelityOperation.requires_aligned_inputs is True  # every other metric is unchanged


def test_cvpl_framework_alignment_does_not_alter_the_populations(tmp_path):
    """Unrelated indices and different row counts must reach CVPL intact (execution level)."""
    src = source_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(6)], index_start=10)
    rel = release_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")] + anchors(), index_start=100)
    assert not set(src.index) & set(rel.index)
    res = _run(CVPLLinkageOperation(**OPS), src, rel, tmp_path)
    assert res.status == OperationStatus.SUCCESS, res.error_message
    assert res.metrics["n_original"] == 6          # the intersection of the indices is empty
    assert res.metrics["n_released"] == 8
    assert res.metrics["cvpl_lr"] == pytest.approx(1.0)
    m = _metrics_file(tmp_path)
    assert m["n_original"] == 6 and m["algorithm_version"] == "1.0"


def test_cvpl_operation_does_not_touch_dtypes_of_large_inputs(tmp_path):
    n = 12_000  # above the base class' memory-optimisation threshold
    rng = np.random.default_rng(0)
    src = pd.DataFrame({"g": rng.choice(["A", "B"], n), "x": rng.normal(size=n).astype("float64"),
                        "sid": [f"s{i}" for i in range(n)]})
    rel = src.iloc[:2000].rename(columns={"sid": "rid"})
    seen = {}
    import pamola_core.metrics.operations.cvpl_ops as mod
    real = mod.run_cvpl

    def spy(source, release, cfg, frozen_projection=None):
        seen["x"] = (source["x"].dtype, release["x"].dtype)
        return real(source, release, cfg, frozen_projection)

    mod.run_cvpl = spy
    try:
        res = _run(CVPLLinkageOperation(**{**OPS, "max_comparisons": 10**9}), src, rel, tmp_path)
    finally:
        mod.run_cvpl = real
    assert res.status == OperationStatus.SUCCESS, res.error_message
    assert seen["x"] == (np.dtype("float64"), np.dtype("float64"))


def test_cvpl_c14_cache_is_recomputed_when_only_the_release_changes(tmp_path):
    src = source_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(4)])
    rel1 = release_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")] + anchors())
    rel2 = rel1.copy()
    rel2["rid"] = ["q0", "q1"] + list(rel1["rid"].iloc[2:])  # same index, same row count, other subjects

    first = _run(CVPLLinkageOperation(use_cache=True, **OPS), src, rel1, tmp_path)
    second = _run(CVPLLinkageOperation(use_cache=True, **OPS), src, rel2, tmp_path)
    assert first.status == second.status == OperationStatus.SUCCESS
    assert not first.metrics.get("cached") and not second.metrics.get("cached")
    assert first.metrics["cvpl_id_uniform_gt"] != second.metrics["cvpl_id_uniform_gt"]
    # an identical request is served from the cache
    third = _run(CVPLLinkageOperation(use_cache=True, **OPS), src, rel1, tmp_path)
    assert third.metrics.get("cached") is True


def test_cvpl_c14_cache_identity_includes_the_cvpl_parameters(tmp_path):
    src = source_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(4)])
    rel = release_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")] + anchors())
    _run(CVPLLinkageOperation(use_cache=True, **OPS), src, rel, tmp_path)
    other = _run(CVPLLinkageOperation(use_cache=True, **{**OPS, "tau": 0.99}), src, rel, tmp_path)
    assert not other.metrics.get("cached")
    again_gt_changed = rel.copy()
    again_gt_changed.loc[again_gt_changed.index[0], "rid"] = "s3"  # the ground truth itself changed
    changed = _run(CVPLLinkageOperation(use_cache=True, **OPS), src, again_gt_changed, tmp_path)
    assert not changed.metrics.get("cached")


def test_cvpl_empty_release_is_a_measured_zero_and_empty_source_is_not_measured(tmp_path):
    src = source_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")])
    empty_rel = release_frame([("A", 1.0, "s0")]).iloc[0:0]
    res = _run(CVPLLinkageOperation(**OPS), src, empty_rel, tmp_path)
    assert res.status == OperationStatus.SUCCESS, res.error_message
    m = _metrics_file(tmp_path)
    assert m["cvpl_lr"] == 0.0 and m["cvpl_lr_status"]["status"] == "measured"
    assert m["cvpl_id_uniform_gt"] == 0.0 and m["cvpl_id_uniform_gt_status"]["status"] == "measured"
    assert m["n_released"] == 0 and m["n_original"] == 2

    out2 = tmp_path / "second"
    out2.mkdir()
    res2 = _run(CVPLLinkageOperation(**OPS), src.iloc[0:0], release_frame([("A", 1.0, "s0")]), out2)
    assert res2.status == OperationStatus.SUCCESS, res2.error_message
    m2 = _metrics_file(out2)
    assert m2["cvpl_lr"] is None and m2["cvpl_lr_status"] == {
        "value": None, "status": "not_measured", "reason": "empty_source"}


def test_cvpl_missing_column_is_a_typed_error_result(tmp_path):
    src = source_frame([("A", 1.0, "s0")]).drop(columns="x")
    rel = release_frame([("A", 1.0, "s0")])
    res = _run(CVPLLinkageOperation(**OPS), src, rel, tmp_path)
    assert res.status == OperationStatus.ERROR
    assert "x" in res.error_message


@pytest.mark.parametrize("kwargs", [{"columns": ["g"]}, {"column_mapping": {"g": "h"}}, {"sample_size": 10}])
def test_cvpl_operation_rejects_parameters_that_do_not_apply(kwargs):
    with pytest.raises(InvalidParameterError):
        CVPLLinkageOperation(**OPS, **kwargs)


def test_cvpl_operation_validates_the_configuration_at_construction():
    with pytest.raises(InvalidParameterError):
        CVPLLinkageOperation(qids=["g"], tau=2.0)
    with pytest.raises(TypeError):  # qids is a required argument (validate-config reports it too)
        CVPLLinkageOperation(auxiliary=["x"])
    with pytest.raises(InvalidParameterError):
        CVPLLinkageOperation(qids=["g"], fit_strategy="frozen")


def test_cvpl_operation_reports_resource_limit_without_raising(tmp_path):
    src = source_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(20)])
    rel = release_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(20)])
    res = _run(CVPLLinkageOperation(**{**OPS, "max_comparisons": 10}), src, rel, tmp_path)
    assert res.status == OperationStatus.SUCCESS, res.error_message
    m = _metrics_file(tmp_path)
    assert m["cvpl_lr"] is None
    assert m["cvpl_lr_status"]["reason"] == "resource_limit"
    assert m["cvpl_id_uniform_gt"] is None


# ---------------------------------------------------------------------------------------
# privacy of the artifacts
# ---------------------------------------------------------------------------------------
def test_cvpl_canary_no_values_or_identifiers_in_any_artifact_or_log(tmp_path, caplog):
    sensitive = {
        "qid": "CANARY_QID_alice_smith", "feat": "CANARY_FEATURE_diagnosis", "sid": "CANARY_SUBJECT_0042",
        "rid": "CANARY_RELEASE_0042",
    }
    rng = np.random.default_rng(1)
    n = 30
    src = pd.DataFrame({
        "g": [f"{sensitive['qid']}_{i % 3}" for i in range(n)],
        "x": rng.normal(size=n),
        "dx": [f"{sensitive['feat']}_{i % 5}" for i in range(n)],
        "sid": [f"{sensitive['sid']}_{i}" for i in range(n)],
    })
    rel = src.sample(20, random_state=2).rename(columns={"sid": "rid"})
    rel["rid"] = rel["rid"].str.replace("SUBJECT", "RELEASE")
    op = CVPLLinkageOperation(
        qids=[{"name": "g", "generalized": False}], auxiliary=["x", "dx"], tau=0.5,
        ground_truth=GT, use_cache=True, seed=0,
    )
    with caplog.at_level(logging.DEBUG):
        res = _run(op, src, rel, tmp_path)
    assert res.status == OperationStatus.SUCCESS, res.error_message

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any("metrics" in str(p) for p in files) and any("cache" in str(p) for p in files)
    blobs = [p.read_bytes().decode("utf-8", errors="ignore") for p in files]
    blobs += [r.getMessage() for r in caplog.records]
    blobs += [str(m) for m in res.metrics.values()]
    for token in ("CANARY_QID", "CANARY_FEATURE", "CANARY_SUBJECT", "CANARY_RELEASE", "alice_smith"):
        assert not [b for b in blobs if token in b], f"{token} leaked"
    # the numbers that are allowed out are all there
    m = _metrics_file(tmp_path)
    assert {"cvpl_lr", "cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau",
            "gt_coverage", "algorithm_version", "direction", "linkage_unit"} <= set(m)


def test_cvpl_ground_truth_is_not_a_feature(tmp_path):
    """Changing only the ground-truth identifiers cannot change what the attacker sees."""
    src = make_df(60, 1)
    rel = make_df(60, 2)
    base = {"qids": [{"name": "age", "bin_width": 10}, "gender", "region_level1"], "auxiliary": ["income"], "tau": 0.9}
    from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl
    a = run_cvpl(src.rename(columns={"person_id": "sid"}), rel.rename(columns={"person_id": "rid"}),
                 CVPLConfig.from_dict({**base, "ground_truth": GT}))
    shuffled = rel.copy()
    shuffled["person_id"] = shuffled["person_id"].sample(frac=1.0, random_state=7).to_numpy()
    b = run_cvpl(src.rename(columns={"person_id": "sid"}), shuffled.rename(columns={"person_id": "rid"}),
                 CVPLConfig.from_dict({**base, "ground_truth": GT}))
    for key in ("cvpl_lr", "n_linked", "cvpl_weighted_lr_legacy"):
        assert a[key] == b[key]
    assert a["projection"]["feature_columns"] == ["age", "gender", "region_level1", "income"]
