"""Regression tests for the second adversarial review of the CVPL implementation (CDX-CVPL-R2-n).

Every test reproduces a reviewer fixture (``codex_scratch/cvpl_r2``: final_probes,
operation_probes, rewrite_oracle, extended_paths, memory_encoding, tiny_uncertainty) with the
reviewer's expected numbers. Where the reviewer accepted a deliberately loose bound the test
asserts containment of the reviewer's feasible value, not equality.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

GT = {"source_id_column": "id", "release_id_column": "id", "scope": "complete"}
BASE = dict(
    qids=[{"name": "g", "generalized": False}], auxiliary=["x"], qid_as_features=False,
    ground_truth=GT, seed=0, min_components=1, max_components=1,
)
B = 2 ** 53


def frame(g, x, ids):
    return pd.DataFrame(dict(g=g, x=x, id=ids))


_TRAIN = frame(["A", "A"], [-1.0, 1.0], ["a", "b"])
PROJ = run_cvpl(_TRAIN, _TRAIN, CVPLConfig.from_dict(BASE)).projection


def evaluate(s, r, params=None, projection=PROJ, **kw):
    cfg = CVPLConfig.from_dict({**BASE, "fit_strategy": "frozen", **(params or {})})
    return run_cvpl(s, r, cfg, frozen_projection=projection, **kw).metrics


def bounds(m, key):
    """(lower, upper) of a measure whatever its status."""
    st = m[key + "_status"]
    if st["status"] == "measured":
        return m[key], m[key]
    return st["lower"], st["upper"]


# ======================================================================================
# CDX-CVPL-R2-4: exact integer endpoints and bin identities everywhere
# ======================================================================================
@pytest.mark.parametrize(
    "source,release,extra,lr",
    [
        # Codex: source B, release interval [B+1, B+2] -> incompatible
        (B, f"[{B + 1},{B + 2}]", {}, 0.0),
        # Codex: source B+1, release [B+1, B+1] -> compatible
        (B + 1, f"[{B + 1},{B + 1}]", {}, 1.0),
        # Codex: source B+1, release pd.Interval(B, B+1, closed="both") -> compatible
        (B + 1, pd.Interval(B, B + 1, closed="both"), {}, 1.0),
        # Codex: source B, release B+1, bin_width=1 -> different bins
        (B, B + 1, {"bin_width": 1}, 0.0),
        # neighbours of the same width-2 bin / of different width-2 bins
        (B + 2, B + 3, {"bin_width": 2}, 1.0),
        (B + 1, B + 2, {"bin_width": 2}, 0.0),
        (B, B, {"bin_width": 1}, 1.0),
        # open-ended and half-open string intervals keep every digit
        (B + 1, f"{B + 1}+", {}, 1.0),
        (B, f"{B + 1}+", {}, 0.0),
        (B + 1, f"({B + 1},{B + 5}]", {}, 0.0),
        (B + 2, f"({B + 1},{B + 5}]", {}, 1.0),
    ],
)
def test_cdx_cvpl_r2_4_exact_integer_endpoints_intervals_and_bins(source, release, extra, lr):
    s, r = frame([source], [1.0], ["a"]), frame([release], [1.0], ["a"])
    m = evaluate(s, r, dict(qids=[dict(name="g", kind="numeric", **extra)]))
    assert m["cvpl_lr"] == lr
    assert m["cvpl_id_uniform_gt"] == lr  # the single candidate is the right person


def test_cdx_cvpl_r2_4_interval_dtype_column_with_large_integers():
    s = frame([B + 1], [1.0], ["a"])
    r = frame(pd.arrays.IntervalArray.from_arrays([B + 1], [B + 1], closed="both"), [1.0], ["a"])
    assert evaluate(s, r, dict(qids=[dict(name="g", kind="numeric")]))["cvpl_lr"] == 1.0
    r = frame(pd.arrays.IntervalArray.from_arrays([B + 2], [B + 3], closed="both"), [1.0], ["a"])
    assert evaluate(s, r, dict(qids=[dict(name="g", kind="numeric")]))["cvpl_lr"] == 0.0


def test_cdx_cvpl_r2_4_union_counts_a_pair_once_with_binned_large_integers():
    """Codex ``bin_same_value_union``: one source, one release, both g=B, h=A, width 1, ladder
    [[g],[h]]. The explicit union contains ONE pair; the bin's float upper end used to collapse
    onto its lower end, so the overlap predicate rejected the pair and rung 2 counted it again."""
    s, r = frame([B], [1.0], ["a"]), frame([B], [1.0], ["a"])
    s["h"], r["h"] = "A", "A"
    params = dict(
        qids=[dict(name="g", kind="numeric", bin_width=1), dict(name="h", generalized=False)],
        blocking_ladder=[["g"], ["h"]],
    )
    m = evaluate(s, r, params)
    assert m["headline_basis"] == "union_of_rungs"
    assert m["n_compared"] == 1  # the pair, once
    assert m["cvpl_lr"] == 1.0
    assert m["cvpl_id_uniform_subjects_gt"] == 1.0  # was 0.5
    assert m["cvpl_id_uniform_rows_gt"] == 1.0
    assert m["cvpl_top1_success_gt"] == 1.0  # was 0.5


@pytest.mark.parametrize("pair", [(B, B + 1, 1), (B + 1, B + 1, 1), (B + 1, B + 2, 2), (B + 3, B + 2, 2),
                                  (3, 4, 2), (3.5, 3.6, 1), (-5, -4, 3)])
def test_cdx_cvpl_r2_4_plain_key_partition_equals_the_shared_predicate(pair):
    """The optimised plain-key partition and the pairwise predicate used for the union
    exclusion must say the same thing about every pair."""
    from pamola_core.metrics.privacy.cvpl.compat import _group_compat, _plain_keys, encode_qid
    from pamola_core.metrics.privacy.cvpl.config import QIDSpec

    a, b, w = pair
    cfg = CVPLConfig.from_dict({**BASE, "qids": [dict(name="g", kind="numeric", bin_width=w)]})
    enc = encode_qid(cfg.qids[0], pd.Series([a, b, a], dtype=object), pd.Series([a, b, b], dtype=object), cfg)
    ks, kr = _plain_keys(enc, "s"), _plain_keys(enc, "r")
    rows = np.arange(3)
    same_key = np.ones((3, 3), dtype=bool)
    for x, y in zip(ks, kr):
        same_key &= x[:, None] == y[None, :]
    assert (same_key == _group_compat(enc, rows, rows, "equal_null")).all()


# ======================================================================================
# operation-level helpers
# ======================================================================================
def op_run(tmp_path, s, r, name="run", **params):
    from pamola_core.metrics import CVPLLinkageOperation
    from pamola_core.utils.ops.op_data_source import DataSource

    d = tmp_path / name
    d.mkdir(exist_ok=True)
    op = CVPLLinkageOperation(**{**BASE, **params})
    out = op.execute(DataSource(dataframes={"s": s, "r": r}), d, None,
                     original_dataset_name="s", transformed_dataset_name="r")
    return out, d


def _all_text(d, out):
    chunks = [json.dumps(out.metrics, default=str)]
    for p in d.rglob("*"):
        if p.is_file():
            chunks.append(p.read_bytes().decode("utf-8", errors="ignore"))
    return "\n".join(chunks)


# ======================================================================================
# CDX-CVPL-R2-3: the projection fingerprint covers the ORDERED encoded feature layout
# ======================================================================================
def test_cdx_cvpl_r2_3_feature_order_is_part_of_the_fingerprint_and_the_cache(tmp_path):
    t = pd.DataFrame(dict(g=["A"] * 4, x=[1.0, -1.0, 0.0, 0.0], y=[0.0, 0.0, 1.0, -1.0], id=list("abcd")))
    ps = [run_cvpl(t, t, CVPLConfig.from_dict({**BASE, "auxiliary": aux})).projection for aux in (["x", "y"], ["y", "x"])]
    assert list(ps[0].kinds) == ["x", "y"] and list(ps[1].kinds) == ["y", "x"]
    assert ps[0].components.tolist() == ps[1].components.tolist() == [[0.0], [1.0]]  # same arrays
    assert ps[0].fingerprint() != ps[1].fingerprint()  # different physical attributes
    s = pd.DataFrame(dict(g=["A"], x=[1.0], y=[1.0], id=["a"]))
    r = pd.DataFrame(dict(g=["A"], x=[1.0], y=[-1.0], id=["a"]))
    kw = dict(auxiliary=["x", "y"], fit_strategy="frozen")
    first, _ = op_run(tmp_path, s, r, "ord", frozen_projection=ps[0], use_cache=True, **kw)
    second, _ = op_run(tmp_path, s, r, "ord", frozen_projection=ps[1], use_cache=True, **kw)
    fresh, _ = op_run(tmp_path, s, r, "ord_fresh", frozen_projection=ps[1], use_cache=False, **kw)
    assert first.metrics["cvpl_lr"] == 0.0
    assert fresh.metrics["cvpl_lr"] == 1.0
    assert second.metrics["cvpl_lr"] == 1.0  # was the cached 0


def test_cdx_cvpl_r2_3_fingerprint_serialisation_is_unambiguous():
    from pamola_core.metrics.privacy.cvpl.projection import CVPLProjection, ProjectionInfo

    def proj(cats):
        info = ProjectionInfo(2, 1, None, 1, "frozen", ["c"], {}, {})
        return CVPLProjection(
            kinds={"c": "categorical"}, num_mean={}, num_scale={}, categories={"c": cats},
            has_other={"c": False}, pca_mean=np.zeros(len(cats)), components=np.ones((len(cats), 1)), info=info)

    # a joined string cannot tell ["ab", "c"] from ["a", "bc"]
    assert proj(["ab", "c"]).fingerprint() != proj(["a", "bc"]).fingerprint()
    assert proj(["a", "b"]).fingerprint() == proj(["a", "b"]).fingerprint()


# ======================================================================================
# CDX-CVPL-R2-5: custom wildcard tokens are input values and are never published
# ======================================================================================
CANARY_TOKEN = "RAW_WILDCARD_R2_BOB_1966"


def test_cdx_cvpl_r2_5_wildcard_token_is_absent_from_every_surface(tmp_path, caplog):
    s = frame([CANARY_TOKEN, "ordinary"], [-1.0, 1.0], ["RAW_GT_R2_PERSON_001", "another"])
    r = s.copy()
    with caplog.at_level("DEBUG"):
        out, d = op_run(tmp_path, s, r, "wild", wildcard_tokens=["*", CANARY_TOKEN], use_cache=True)
    assert str(out.status).endswith("SUCCESS")
    text = _all_text(d, out)
    assert CANARY_TOKEN not in text
    assert CANARY_TOKEN not in "\n".join(rec.getMessage() for rec in caplog.records)
    params = out.metrics["parameters"]
    assert "wildcard_tokens" not in params
    assert params["wildcard_token_count"] == 2 and len(params["wildcard_tokens_digest"]) == 64


def test_cdx_cvpl_r2_5_digest_keeps_the_cache_identity(tmp_path):
    s, r = frame(["X1"], [1.0], ["a"]), frame(["Y"], [1.0], ["a"])
    first, _ = op_run(tmp_path, s, r, "tok", wildcard_tokens=["*", "X1"], fit_strategy="frozen",
                      frozen_projection=PROJ, use_cache=True)
    second, _ = op_run(tmp_path, s, r, "tok", wildcard_tokens=["*", "X2"], fit_strategy="frozen",
                       frozen_projection=PROJ, use_cache=True)
    assert first.metrics["cvpl_lr"] == 1.0  # X1 is a wildcard: compatible with Y
    assert second.metrics["cvpl_lr"] == 0.0  # X1 is an ordinary label: not Y


# ======================================================================================
# CDX-CVPL-R2-1: top-1 / precision bounds account for a shifted maximum and tie band
# ======================================================================================
def _top_bound_fixtures():
    t = pd.DataFrame(dict(g=["A"] * 4, x=[1.0, -1.0, 0.0, 0.0], y=[0.0, 0.0, 1.0, -1.0], id=list("abcd")))
    proj = run_cvpl(t, t, CVPLConfig.from_dict({**BASE, "auxiliary": ["x", "y"], "min_components": 2,
                                                 "max_components": 2})).projection
    s = pd.DataFrame(dict(g=["A"], x=[1.0], y=[0.0], id=["a"]))
    xs = [0.8, 0.8 - 7e-10, 0.8 - 7e-10, 0.0]
    r = pd.DataFrame(dict(g=["A"] * 4, x=xs, y=[math.sqrt(1 - v * v) if v else 0.0 for v in xs], id=list("abcd")))
    params = dict(auxiliary=["x", "y"], min_components=2, max_components=2, tau=0.5)
    return proj, s, r, params


def test_cdx_cvpl_r2_1_top1_and_precision_bounds_contain_the_tie_shifting_completion():
    proj, s, r, params = _top_bound_fixtures()
    m = evaluate(s, r, params, projection=proj)  # d: zero vector, cosine undefined
    # Codex: resolving d to cosine .8 + 7e-10 makes the new maximum exclude b and c: a and d
    # are tied and the correct share is 1/2 - outside the old [0, 1/3]
    r2 = r.copy()
    r2.loc[3, ["x", "y"]] = [1e-6 * (0.8 + 7e-10), 1e-6 * math.sqrt(1 - (0.8 + 7e-10) ** 2)]
    done = evaluate(s, r2, params, projection=proj)
    assert done["cvpl_top1_success_gt"] == pytest.approx(0.5)
    assert done["precision_at_tau"] == pytest.approx(0.5)
    for key in ("cvpl_top1_success_gt", "precision_at_tau"):
        assert m[key] is None and m[key + "_status"]["status"] == "bounded"
        lo, hi = bounds(m, key)
        assert lo <= 0.5 <= hi, (key, lo, hi)
        assert hi == 1.0  # deliberately loose: an unknown candidate can alter a tie of the correct subject
        assert m[key + "_status"]["n_uncertain"] >= 1


def test_cdx_cvpl_r2_1_an_unknown_row_that_cannot_touch_the_correct_tie_adds_no_looseness():
    proj, s, r, params = _top_bound_fixtures()
    r = r.copy()
    r.loc[0, "id"], r.loc[1, "id"] = "z1", "a"
    r.loc[1, ["x", "y"]] = [0.7, math.sqrt(1 - 0.49)]  # the correct subject scores below the tie band
    m = evaluate(s, r, params, projection=proj)
    lo, hi = bounds(m, "cvpl_top1_success_gt")
    assert hi == 0.0


# ======================================================================================
# CDX-CVPL-R2-8: uncertainty flags come from membership/count logic, not from float noise
# ======================================================================================
def test_cdx_cvpl_r2_8_tiny_uncertainty_is_not_reported_as_measured():
    k = 1_000_001
    s = pd.DataFrame(dict(g=["A"], x=[1.0], id=[0]))
    r = pd.DataFrame(dict(g=["A"] * (k + 1), x=np.r_[np.ones(k), 0.0], id=np.arange(k + 1)))
    m = evaluate(s, r)
    st = m["cvpl_id_uniform_gt_status"]
    assert st["status"] == "bounded" and st["n_uncertain"] == 1
    assert m["cvpl_id_uniform_gt"] is None
    assert st["lower"] == pytest.approx(9.99998000004e-7, rel=1e-12)
    assert st["upper"] == pytest.approx(9.99999000001e-7, rel=1e-12)


# ======================================================================================
# CDX-CVPL-R2-2: uncertainty on EVERY published surface
# ======================================================================================
def _undefined_unlinked():
    return (frame(["A", "B", "C"], [-1.0, 0.0, 1.0], ["a", "b", "c"]),
            frame(["A", "B", "C"], [-1.0, 1.0, 0.0], ["a", "b", "c"]))


def _third_to_one(lo, hi):
    return lo == pytest.approx(1 / 3) and hi == pytest.approx(1.0)


def test_cdx_cvpl_r2_2_every_surface_carries_the_existential_bounds():
    s, r = _undefined_unlinked()
    m = run_cvpl(s, r, CVPLConfig.from_dict(BASE)).metrics
    assert m["cvpl_lr"] is None and m["cvpl_lr_status"]["status"] == "bounded"
    assert _third_to_one(m["cvpl_lr_lower"], m["cvpl_lr_upper"])
    assert m["cvpl_lr_union"] is None and m["cvpl_lr_union_status"]["status"] == "bounded"
    assert _third_to_one(m["cvpl_lr_union_lower"], m["cvpl_lr_union_upper"])
    last = m["candidate_union_trajectory"][-1]
    assert last["cvpl_lr_union"] is None and last["status"] == "bounded"
    assert _third_to_one(last["cvpl_lr_union_lower"], last["cvpl_lr_union_upper"])
    entry = m["blocking_ladder_trajectory"][-1]
    assert entry["cumulative_lr"] is None and entry["cumulative_lr_status"] == "bounded"
    rung = m["cvpl_lr_by_rung"][0]  # never "measured" with a null value
    assert rung["cvpl_lr"] is None and rung["status"] == "bounded"
    assert _third_to_one(rung["cvpl_lr_lower"], rung["cvpl_lr_upper"])
    assert m["cvpl_weighted_lr_legacy"] is None  # the same fixed weights for lower and upper
    assert m["cvpl_weighted_lr_legacy_status"]["status"] == "bounded"
    assert _third_to_one(m["cvpl_weighted_lr_legacy_lower"], m["cvpl_weighted_lr_legacy_upper"])
    # the documented headline == trajectory equality holds for the bounds too
    assert (m["cvpl_lr_lower"], m["cvpl_lr_upper"]) == (last["cvpl_lr_union_lower"], last["cvpl_lr_union_upper"])


def test_cdx_cvpl_r2_2_union_headline_of_a_non_nested_ladder_carries_the_bounds():
    s, r = _undefined_unlinked()
    s["h"], r["h"] = ["H0", "H1", "H2"], ["H0", "H1", "H2"]  # rung h: the same pairs as rung g
    params = dict(qids=[dict(name="g", generalized=False), dict(name="h", generalized=False)],
                  blocking_ladder=[["g"], ["h"]])
    m = run_cvpl(s, r, CVPLConfig.from_dict({**BASE, **params})).metrics
    assert m["headline_basis"] == "union_of_rungs"
    assert m["cvpl_lr"] is None and m["cvpl_lr_status"]["status"] == "bounded"
    assert _third_to_one(m["cvpl_lr_lower"], m["cvpl_lr_upper"])
    last = m["candidate_union_trajectory"][-1]
    assert (m["cvpl_lr"], m["cvpl_lr_lower"], m["cvpl_lr_upper"]) == (
        last["cvpl_lr_union"], last["cvpl_lr_union_lower"], last["cvpl_lr_union_upper"])
    assert m["cvpl_weighted_lr_legacy"] is None and m["cvpl_weighted_lr_legacy_status"]["status"] == "bounded"
    assert m["n_compared"] == 0 or m["n_compared"] <= 3  # a pair shared by both rungs is counted once


def test_cdx_cvpl_r2_2_legacy_weights_are_the_same_for_both_bounds():
    # source A has 1 candidate and is linked, source B has 2 candidates and none is defined:
    # weights 1 and 1/2
    s = frame(["A", "B"], [-1.0, 0.0], ["a", "b"])
    r = frame(["A", "B", "B"], [-1.0, 1.0, 1.0], ["a", "b", "q"])
    m = run_cvpl(s, r, CVPLConfig.from_dict(BASE)).metrics
    assert m["cvpl_weighted_lr_legacy_lower"] == pytest.approx(1 / 1.5)
    assert m["cvpl_weighted_lr_legacy_upper"] == pytest.approx(1.0)


def test_cdx_cvpl_r2_2_operation_result_publishes_no_lower_bound_as_a_measure(tmp_path):
    s, r = _undefined_unlinked()
    out, _ = op_run(tmp_path, s, r, "unc")
    m = out.metrics
    for key in ("cvpl_lr", "cvpl_lr_union", "cvpl_weighted_lr_legacy"):
        assert m[key] is None, key
    assert m["candidate_union_trajectory"][-1]["cvpl_lr_union"] is None
    assert m["cvpl_lr_by_rung"][0]["status"] == "bounded"


# ======================================================================================
# CDX-CVPL-R2-6: identification per rung, headline = largest over the declared profiles
# ======================================================================================
def _ladder_framing():
    s = frame(["A"], [1.0], ["a"])
    r = frame(["A"] * 5, [1.0] * 5, list("abcde"))
    s["h"] = ["own"]
    r["h"] = ["own", "other", "other", "other", "other"]
    params = dict(qids=[dict(name="g", generalized=False), dict(name="h", generalized=False)],
                  blocking_ladder=[["g", "h"], ["g"]])
    return s, r, params


def test_cdx_cvpl_r2_6_headline_is_the_largest_over_rungs_and_names_its_rung():
    s, r, params = _ladder_framing()
    m = evaluate(s, r, params)
    strict, loose = m["blocking_ladder_trajectory"]
    assert strict["measures"]["cvpl_id_uniform_gt"] == 1.0 and strict["measures"]["cvpl_top1_success_gt"] == 1.0
    assert loose["measures"]["cvpl_id_uniform_gt"] == pytest.approx(0.2)
    assert loose["measures"]["cvpl_top1_success_gt"] == pytest.approx(0.2)
    assert m["cvpl_lr"] == 1.0  # existential: nested ladder, the last rung is the union
    # the loose-rung value 0.2 is no longer the headline: the strict profile wins
    assert m["cvpl_id_uniform_gt"] == 1.0 and m["cvpl_id_uniform_basis_rung"] == 0
    assert m["cvpl_id_uniform_gt_status"]["basis_rung"] == 0
    assert m["cvpl_top1_success_gt"] == 1.0 and m["cvpl_top1_success_gt_basis_rung"] == 0
    assert m["cvpl_id_uniform_subjects_gt_basis_rung"] == 0 and m["cvpl_id_uniform_rows_gt_basis_rung"] == 0
    assert m["precision_at_tau"] == 1.0 and m["precision_at_tau_basis_rung"] == 0
    assert [e["rung"] for e in m["identification_by_rung"]] == [0, 1]


def test_cdx_cvpl_r2_6_basis_names_both_the_strategy_and_the_rung():
    s = frame(["A"], [1.0], ["a"])
    r = frame(["A"] * 5, [1.0] * 5, ["a", "a", "a", "b", "c"])
    r["h"] = ["own", "own", "own", "own", "x"]
    s["h"] = ["own"]
    params = dict(qids=[dict(name="g", generalized=False), dict(name="h", generalized=False)],
                  blocking_ladder=[["g"], ["g", "h"]])
    m = evaluate(s, r, params)
    m0 = m["blocking_ladder_trajectory"][0]["measures"]
    assert m0["cvpl_id_uniform_rows_gt"] == pytest.approx(3 / 5)
    assert m0["cvpl_id_uniform_subjects_gt"] == pytest.approx(1 / 3)
    assert m["cvpl_id_uniform_gt"] == pytest.approx(3 / 4)  # rung 1: rows 3/4 beat subjects 1/2
    assert m["cvpl_id_uniform_basis"] == "rows" and m["cvpl_id_uniform_basis_rung"] == 1


def test_cdx_cvpl_r2_6_population_maximum_of_the_two_fixed_strategies():
    s = frame(["A", "B"], [1.0, 1.0], ["a", "b"])
    r = frame(["A"] * 5 + ["B"] * 5, [1.0] * 10, ["a", "a", "a", "a", "x", "b", "c", "c", "c", "c"])
    m = evaluate(s, r)
    assert m["cvpl_id_uniform_subjects_gt"] == pytest.approx(0.5)
    assert m["cvpl_id_uniform_rows_gt"] == pytest.approx(0.5)
    assert m["cvpl_id_uniform_gt"] == pytest.approx(0.5)  # not the per-record optimum 0.65


def test_cdx_cvpl_r2_6_an_unmeasured_rung_withholds_the_identification_headline():
    s, r, params = _ladder_framing()
    r = r.copy()
    r.loc[1, "h"] = "unknown-generalized-label"  # strict rung: a generalized label with no hierarchy
    params["qids"][1] = dict(name="h")
    m = evaluate(s, r, params)
    assert m["cvpl_lr"] == 1.0  # existential headline: the last (loosest) rung is measured
    for key in ("cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau"):
        assert m[key] is None and m[key + "_status"]["status"] in ("not_applicable", "not_measured")


# ======================================================================================
# CDX-CVPL-R2-7: exact-fallback classes are not an N x F expansion; refusal comes first
# ======================================================================================
def _wide_zero_fixture(n, k=50):
    base = dict(qids=[dict(name="g", generalized=False)], auxiliary=["x", "c"], qid_as_features=False,
                ground_truth=None, min_components=1, max_components=1, max_categories=k + 1,
                fit_strategy="source_only", chunk_pairs=10)
    t = pd.DataFrame(dict(g=["A"] * (2 * k), x=np.tile([-1.0, 1.0], k), c=np.repeat([f"c{i}" for i in range(k)], 2)))
    proj = run_cvpl(t, t, CVPLConfig.from_dict(base)).projection
    s = pd.DataFrame(dict(g=["A"] * n, x=np.zeros(n), c=[f"c{i % k}" for i in range(n)]))
    return base, proj, s


def test_cdx_cvpl_r2_7_budget_refusal_happens_before_any_fallback_preparation(monkeypatch):
    from pamola_core.metrics.privacy.cvpl import engine

    base, proj, s = _wide_zero_fixture(200)

    def boom(*a, **k):
        raise AssertionError("fallback classes prepared although the budget refuses the rung")

    monkeypatch.setattr(engine, "_encoded_classes", boom)
    m = run_cvpl(s, s, CVPLConfig.from_dict({**base, "fit_strategy": "frozen", "max_comparisons": 1}),
                 frozen_projection=proj).metrics
    assert m["cvpl_lr_status"]["status"] == "not_measured"
    assert m["cvpl_lr_status"]["reason"] == "resource_limit"


def test_cdx_cvpl_r2_7_fallback_classes_are_exact_and_not_dense():
    import tracemalloc

    from pamola_core.metrics.privacy.cvpl.engine import _encoded_classes
    from pamola_core.metrics.privacy.cvpl.projection import categorical_values, numeric_values

    k, n = 500, 2000
    base, proj, s = _wide_zero_fixture(n, k)
    cols = {"x": ("numeric", numeric_values(s["x"])), "c": ("categorical", categorical_values(s["c"]))}
    deg = np.ones(n, dtype=bool)
    tracemalloc.start()
    cls_s, cls_r = _encoded_classes(proj, cols, cols, deg, deg)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    dense = n * proj.info.n_features * 8  # one side's N x F expansion
    assert proj.info.n_features >= k
    assert peak < dense / 8, (peak, dense)  # the old code allocated >= 4x `dense`
    assert len(np.unique(cls_s)) == k and (cls_s == cls_r).all()
    # the classes are exactly the equivalence classes of the dense encoded vectors
    dense_x = proj.phi_rows(cols, np.arange(n)) + 0.0
    _, inverse = np.unique(dense_x, axis=0, return_inverse=True)
    assert (pd.factorize(inverse.reshape(-1))[0] == pd.factorize(cls_s)[0]).all()


def test_cdx_cvpl_r2_7_exact_equal_pairs_still_link():
    base, proj, s = _wide_zero_fixture(100)
    m = run_cvpl(s, s, CVPLConfig.from_dict({**base, "fit_strategy": "frozen"}), frozen_projection=proj).metrics
    assert m["cvpl_lr"] == 1.0 and m["n_exact_fallback_records"] == 100
