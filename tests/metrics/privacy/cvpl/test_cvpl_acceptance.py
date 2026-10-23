"""CVPL acceptance cases C01-C10, C13-C15, C17, C18 (ADR-CVPL-001 section 7).

One test per case, named ``test_cvpl_cNN_...``. Cases that need an operation run
(cache, framework alignment) are in ``test_cvpl_operation.py``.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

from .helpers import EXACT_G, GT, anchors, joint_mean, make_df, measure, release_frame, source_frame


# --- C01-C03: what the two GT measures say about the candidate set -----------------
def test_cvpl_c01_five_candidates_one_true():
    src = source_frame([("A", 1.0, "s0")])
    rel = release_frame(
        [("A", 1.0, "s0"), ("A", 1.1, "t1"), ("A", 1.2, "t2"), ("A", 1.3, "t3"), ("A", 1.4, "t4")]
        + anchors()
    )
    assert joint_mean(src, rel) < 1.0  # all five candidates are on the source's side
    r = measure(src, rel)
    assert r["cvpl_lr"] == 1.0
    assert r["cvpl_id_uniform_gt"] == pytest.approx(0.2)
    assert r["cvpl_id_uniform_gt_status"]["status"] == "measured"
    # existence of a link is not the probability of the right one
    assert r["cvpl_id_uniform_gt"] < r["cvpl_lr"]


def test_cvpl_c02_five_candidates_no_true_one():
    src = source_frame([("A", 1.0, "s0")])
    rel = release_frame([("A", 1.0 + 0.1 * i, f"t{i}") for i in range(5)] + anchors())
    r = measure(src, rel)
    assert r["cvpl_lr"] == 1.0
    assert r["cvpl_id_uniform_gt"] == 0.0
    # a measured zero, with a complete GT behind it - not an unknown
    assert r["cvpl_id_uniform_gt_status"]["status"] == "measured"
    assert r["cvpl_top1_success_gt"] == 0.0


def test_cvpl_c03_single_true_candidate_without_competitors():
    src = source_frame([("A", 1.0, "s0")])
    rel = release_frame([("A", 1.0, "s0")] + anchors())
    r = measure(src, rel)
    assert r["cvpl_id_uniform_gt"] == 1.0
    assert r["cvpl_top1_success_gt"] == 1.0
    assert r["precision_at_tau"] == 1.0


# --- C04: bypassing the blocking or the threshold earns nothing --------------------
def test_cvpl_c04_true_pair_outside_blocking_or_below_tau_scores_zero():
    # (a) the subject's own row sits in another block
    src = source_frame([("A", 1.0, "s0")])
    rel = release_frame([("B", 1.0, "s0"), ("A", 1.1, "t1")] + anchors())
    r = measure(src, rel)
    assert r["cvpl_lr"] == 1.0
    assert r["cvpl_id_uniform_gt"] == 0.0
    # (b) in the block but on the other side of the joint mean: cosine -1 < tau
    src = source_frame([("A", 1.0, "s0")])
    rel = release_frame([("A", -1000.0, "s0"), ("A", 1.1, "t1"), ("A", 1.2, "t2")] + anchors())
    m = joint_mean(src, rel)
    assert -1000.0 < m < 1.0
    r = measure(src, rel)
    assert r["cvpl_lr"] == 1.0  # the two look-alikes pass
    assert r["cvpl_id_uniform_gt"] == 0.0  # the true pair does not: no credit for bypassing tau


# --- C05: no ground truth is not zero risk ------------------------------------------
def test_cvpl_c05_missing_ground_truth_is_not_measured_not_zero():
    src = source_frame([("A", 1.0, "s0")]).drop(columns="sid")
    rel = release_frame([("A", 1.0, None), ("A", 1.1, None)] + anchors()).drop(columns="rid")
    r = measure(src, rel, ground_truth=None)
    assert r["cvpl_lr"] == 1.0
    for key in ("cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau"):
        assert r[key] is None
        st = r[f"{key}_status"]
        assert st["status"] == "not_measured" and st["reason"] == "no_ground_truth"
    assert r["ground_truth"]["status"] == "absent"
    assert r["gt_coverage"] is None


def test_cvpl_c05_partial_ground_truth_reports_only_the_known_subset():
    src = source_frame([("A", 1.0, "s0"), ("A", 2.0, None)])
    rel = release_frame([("A", 1.0, "s0"), ("A", 1.5, "t1")] + anchors())
    r = measure(src, rel)
    assert r["cvpl_id_uniform_gt"] is None
    st = r["cvpl_id_uniform_gt_status"]
    assert st["status"] == "not_measured" and st["reason"].startswith("ground_truth_incomplete")
    assert st["known_subset_denominator"] == 1  # own denominator, not the population
    assert r["gt_coverage"] == pytest.approx(0.5)


# --- C06: suppression keeps the original denominator ----------------------------------
def test_cvpl_c06_suppression_keeps_the_source_denominator():
    n = 100
    src = source_frame([(f"G{i}", 1.0 + i / 1000, f"s{i}") for i in range(n)])
    kept = [i for i in range(n) if i % 10 < 7]  # 30 of the 100 subjects are suppressed
    rel = release_frame([(f"G{i}", 1.0 + i / 1000, f"s{i}") for i in kept] + anchors())
    assert len(kept) == 70
    r = measure(src, rel)
    assert r["n_original"] == 100
    assert r["cvpl_lr"] == pytest.approx(0.70)
    assert r["cvpl_id_uniform_gt"] == pytest.approx(0.70)
    assert r["cvpl_id_uniform_gt_status"]["denominator"] == 100
    assert r["cvpl_top1_success_gt"] == pytest.approx(0.70)
    assert r["precision_at_tau"] == pytest.approx(1.0)  # of the 70 declared links, all right
    assert r["precision_at_tau_status"]["denominator"] == 70


# --- C07: generalization ----------------------------------------------------------------
def _age_frames(release_ages):
    src = pd.DataFrame({"age": [21, 31, 41, 51], "x": [1.0, 1.1, 1.2, 1.3], "sid": list("abcd")})
    rel = pd.DataFrame({
        "age": list(release_ages) + ["[90-99]"] * 4,
        "x": [1.0, 1.1, 1.2, 1.3] + [-100.0, -101.0, -102.0, -103.0],
        "rid": list("abcd") + ["z1", "z2", "z3", "z4"],
    })
    return src, rel


@pytest.mark.parametrize("ages", [
    ["[20-29]", "[30-39]", "[40-49]", "[50-59]"],
    ["20-29", "30-39", "40-49", "50-59"],
    ["[20, 30)", "[30, 40)", "[40, 50)", "[50, 60)"],
    [pd.Interval(20, 29, closed="both"), pd.Interval(30, 39, closed="both"),
     pd.Interval(40, 49, closed="both"), pd.Interval(50, 59, closed="both")],
])
def test_cvpl_c07_numeric_interval_keeps_the_pair_compatible(ages):
    src, rel = _age_frames(ages)
    cfg = CVPLConfig.from_dict({"qids": ["age"], "auxiliary": ["x"], "qid_as_features": False,
                                "tau": 0.5, "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    assert r["qid_status"]["age"]["applicable"] is True
    assert r["cvpl_lr"] == 1.0
    assert r["cvpl_id_uniform_gt"] == 1.0  # each age falls in exactly its own interval
    assert r["n_with_candidates"] == 4


def test_cvpl_c07_interval_bounds_are_honoured():
    src = pd.DataFrame({"age": [29, 30], "x": [1.0, 1.1], "sid": ["a", "b"]})
    rel = pd.DataFrame({"age": ["[20, 30)", "[30, 40)", "[90-99]"], "x": [1.0, 1.1, -100.0],
                        "rid": ["a", "b", "z"]})
    cfg = CVPLConfig.from_dict({"qids": ["age"], "auxiliary": ["x"], "qid_as_features": False,
                                "tau": 0.5, "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    # 29 is in [20,30) only, 30 in [30,40) only: half-open bounds respected
    assert r["cvpl_id_uniform_gt"] == 1.0


def test_cvpl_c07_categorical_hierarchy_makes_generalized_values_compatible():
    src = pd.DataFrame({"city": ["Paris", "Lyon", "Nice"], "x": [1.0, 1.1, 1.2], "sid": ["a", "b", "c"]})
    rel = pd.DataFrame({"city": ["North", "South", "South", "Z"], "x": [1.0, 1.1, 1.2, -100.0],
                        "rid": ["a", "b", "c", "z"]})
    # 'Z' is the filler row's own label: a generalized value must be NAMED by the hierarchy
    hierarchy = {"Paris": "North", "Lyon": "South", "Nice": "South", "North": "France", "South": "France",
                 "Z": "Z-root"}
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": "city", "kind": "categorical", "hierarchy": hierarchy}],
        "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5, "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    assert r["cvpl_lr"] == 1.0
    # Lyon and Nice share the released value 'South': two candidates each, one true
    assert r["cvpl_id_uniform_gt"] == pytest.approx((1 + 0.5 + 0.5) / 3)


def test_cvpl_c07_generalization_without_hierarchy_is_not_applicable_never_exact_match():
    src = pd.DataFrame({"city": ["Paris", "Lyon", "Nice"], "x": [1.0, 1.1, 1.2], "sid": ["a", "b", "c"]})
    rel = pd.DataFrame({"city": ["North", "South", "South", "Z"], "x": [1.0, 1.1, 1.2, -100.0],
                        "rid": ["a", "b", "c", "z"]})
    cfg = CVPLConfig.from_dict({"qids": ["city"], "auxiliary": ["x"], "qid_as_features": False,
                                "tau": 0.5, "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    assert r["qid_status"]["city"]["applicable"] is False
    assert r["qid_status"]["city"]["limitation"] == "generalized_values_without_hierarchy"
    # no number is reported: an exact-match 0 would read as "protected"
    assert r["cvpl_lr"] is None
    assert r["cvpl_lr_status"]["status"] == "not_applicable"
    assert "city" in r["cvpl_lr_status"]["reason"]
    # a ladder that does not need the QID still measures
    cfg2 = CVPLConfig.from_dict({
        "qids": ["city"], "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5,
        "blocking_ladder": [["city"], []], "ground_truth": GT})
    r2 = run_cvpl(src, rel, cfg2)
    assert r2["blocking_ladder_trajectory"][0]["status"] == "not_applicable"
    assert r2["blocking_ladder_trajectory"][1]["status"] == "measured"


# --- C08: a zero projected vector -----------------------------------------------------------
def test_cvpl_c08_equal_encoded_vectors_link_deterministically():
    # D' = D, x = 1, 2, 3: the middle record equals the joint mean, so its projection is
    # the zero vector in both datasets. BEST scored 2/3 here (a false negative).
    src = source_frame([("A", 1.0, "s0"), ("A", 2.0, "s1"), ("A", 3.0, "s2")])
    rel = release_frame([("A", 1.0, "s0"), ("A", 2.0, "s1"), ("A", 3.0, "s2")])
    r = measure(src, rel, tau=1.0)
    assert r["degenerate_vectors"] == {"source": 1, "release": 1}
    assert r["n_exact_fallback"] == 1
    assert r["n_undetermined"] == 0
    # 1-D cosine: the middle one links only to itself by the fallback; the outer two to
    # the rows on their own side of the mean, not to the opposite one
    assert r["cvpl_lr"] == 1.0


def test_cvpl_c08_unequal_vectors_with_undefined_cosine_are_undetermined_not_unlinked():
    src = source_frame([("A", 2.0, "s0")])
    rel = release_frame([("A", 1.0, "t1"), ("A", 3.0, "t2")])
    assert joint_mean(src, rel) == 2.0  # the source is the zero vector
    r = measure(src, rel)
    assert r["n_exact_fallback"] == 0
    assert r["n_undetermined"] == 1
    # an undefined candidate may matter: no point value, only the interval
    assert r["cvpl_lr"] is None
    assert r["cvpl_lr_status"]["status"] == "bounded"
    assert r["cvpl_lr_lower"] == 0.0 and r["cvpl_lr_upper"] == 1.0
    assert r["cvpl_lr_status"]["upper"] == 1.0
    assert "n_undetermined" in r["undefined_policy"]


def test_cvpl_c08_a_fully_degenerate_representation_is_not_measured():
    src = source_frame([("A", 5.0, "s0"), ("A", 5.0, "s1")])
    rel = release_frame([("A", 5.0, "t0")])
    r = measure(src, rel, ground_truth=None)
    # constant feature: every projected vector is zero; the exact fallback links the
    # identical encoded vectors, so this is measured through the fallback, not skipped
    assert r["n_exact_fallback"] == 2 and r["cvpl_lr"] == 1.0


# --- C09: duplicates of one subject ------------------------------------------------------
def test_cvpl_c09_duplicated_release_row_does_not_dilute_a_subject():
    src = source_frame([("A", 1.0, "s0")])
    once = release_frame([("A", 1.0, "s0"), ("A", 1.1, "t1")] + anchors())
    dup = release_frame([("A", 1.0, "s0"), ("A", 1.0, "s0"), ("A", 1.1, "t1")] + anchors())
    r1, r2 = measure(src, once), measure(src, dup)
    assert r1["cvpl_id_uniform_subjects_gt"] == pytest.approx(0.5)
    # rows would give 2/3; subjects give 1/2 again (subject-level counting under a FIXED
    # representation: source_only/frozen keep it fixed, joint fitting would move it)
    assert r2["cvpl_id_uniform_subjects_gt"] == pytest.approx(0.5)
    assert r2["cvpl_id_uniform_rows_gt"] == pytest.approx(2 / 3)
    assert r2["cvpl_id_uniform_gt"] == pytest.approx(2 / 3)  # headline = max of the two
    assert r2["ground_truth"]["release_subject_duplicates"] is True
    assert r2["cvpl_top1_success_gt"] == pytest.approx(r1["cvpl_top1_success_gt"])


# --- C10: nested ladders -------------------------------------------------------------------
def _ladder_frames():
    src = pd.DataFrame({
        "g": ["A", "B"], "h": [1, 1], "x": [1.0, 1.1], "sid": ["s0", "s1"],
    })
    rel = pd.DataFrame({
        "g": ["A", "A", "A", "A", "B", "Z", "Z", "Z", "Z"],
        "h": [1, 2, 2, 2, 2, 9, 9, 9, 9],
        "x": [1.0, 1.05, 1.06, 1.07, 1.1, -100, -101, -102, -103],
        "rid": ["s0", "t1", "t2", "t3", "t4", "z1", "z2", "z3", "z4"],
    })
    return src, rel


def test_cvpl_c10_nested_ladder_lr_never_decreases_while_p_id_may_fall():
    src, rel = _ladder_frames()
    cfg = CVPLConfig.from_dict({
        "qids": [EXACT_G, "h"], "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5,
        "blocking_ladder": [["g", "h"], ["g"], []], "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    traj = r["blocking_ladder_trajectory"]
    lrs = [e["measures"]["cvpl_lr"] for e in traj]
    ps = [e["measures"]["cvpl_id_uniform_subjects_gt"] for e in traj]
    assert lrs == sorted(lrs)  # Theorem 1: existential linkage is monotone
    assert lrs == [0.5, 1.0, 1.0]  # s1 gains a candidate once h is relaxed
    assert ps[0] == pytest.approx(0.5)   # (s0 -> {s0}) 1, (s1 -> {}) 0
    assert ps[1] < 1.0 and ps[1] != ps[0]
    assert ps[2] < ps[0]  # more competitors: the correct-identity probability FELL
    assert r["ladder"]["nested"] is True and r["ladder"]["monotone_lr_theorem_applies"] is True
    assert r["ladder"]["transitions"] == ["relaxing", "relaxing"]
    assert r["headline_rung"] == 2 and r["cvpl_lr"] == lrs[-1]


# --- C13: high-cardinality QID and attributes -------------------------------------------------
def test_cvpl_c13_high_cardinality_is_encoded_in_bounded_width_not_dropped():
    rng = np.random.default_rng(0)
    n = 400
    zips = [f"Z{i:04d}" for i in range(n)]
    notes = rng.choice([f"note{i}" for i in range(300)], n)
    src = pd.DataFrame({"zip": zips, "note": notes, "age": rng.integers(20, 70, n), "sid": [f"s{i}" for i in range(n)]})
    rel = src.copy().rename(columns={"sid": "rid"})
    cfg = CVPLConfig.from_dict({
        "qids": ["zip"], "auxiliary": ["note", "age"], "tau": 0.9, "max_categories": 50, "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    enc = r["projection"]["encoding"]
    assert set(enc) == {"zip", "note"}                    # nothing silently dropped
    assert enc["zip"]["distinct_values"] == 400 and enc["zip"]["bounded"] is True
    assert enc["zip"]["encoded_width"] == 50              # bounded encoding
    assert r["projection"]["feature_columns"] == ["zip", "note", "age"]
    assert r["cvpl_lr_status"]["status"] == "measured"
    # the unique QID value still blocks exactly: a row's only candidate is its own copy
    assert r["cvpl_id_uniform_gt"] == pytest.approx(1.0)


def test_cvpl_c13_oversized_representation_is_a_limit_not_a_silent_drop():
    n = 30
    src = pd.DataFrame({f"c{j}": [f"v{i}" for i in range(n)] for j in range(3)})
    src["g"] = "A"
    cfg = CVPLConfig.from_dict({"qids": ["g"], "auxiliary": ["c0", "c1", "c2"], "max_categories": 2,
                                "qid_as_features": False, "tau": 0.5})
    r = run_cvpl(src, src.copy(), cfg)
    assert r["projection"]["encoding"]["c0"]["encoded_width"] == 2


# --- C15: independent indices, seed 0, null / wildcard policy ------------------------------------
def test_cvpl_c15_independent_indices_do_not_change_the_denominator():
    src = source_frame([("A", 1.0, "s0"), ("A", 1.1, "s1"), ("B", 1.2, "s2")], index_start=10)
    rel = release_frame([("A", 1.0, "s0"), ("B", 1.2, "s2")] + anchors(), index_start=10_000)
    assert not set(src.index) & set(rel.index)
    r = measure(src, rel)
    assert r["n_original"] == 3 and r["n_released"] == 2 + 6
    assert r["cvpl_lr"] == 1.0  # s1 also finds s0's released row (same block, same side)
    assert r["cvpl_id_uniform_gt"] == pytest.approx(2 / 3)  # s0 and s2 right, s1 wrong
    # an overlapping but different index gives the same answer
    rel_shifted = rel.copy()
    rel_shifted.index = [11, 12] + list(range(500, 506))
    r2 = measure(src, rel_shifted)
    assert r2["cvpl_id_uniform_gt"] == r["cvpl_id_uniform_gt"] and r2["n_original"] == 3


def test_cvpl_c15_seed_zero_is_a_seed():
    src, rel = make_df(60, 1), make_df(60, 2)
    cfg = {"qids": [{"name": "age", "bin_width": 10}, "gender", "region_level1"],
           "auxiliary": ["income"], "tau": 0.9}
    r0 = run_cvpl(src, rel, CVPLConfig.from_dict({**cfg, "seed": 0}))
    assert r0["seed"] == 0 and r0["parameters"]["seed"] == 0
    r_none = run_cvpl(src, rel, CVPLConfig.from_dict(cfg))
    assert r_none["seed"] is None
    r1 = run_cvpl(src, rel, CVPLConfig.from_dict({**cfg, "seed": 0}))
    assert r0["cvpl_lr"] == r1["cvpl_lr"]


@pytest.mark.parametrize(("policy", "lo", "hi", "zero"), [
    ("equal_null", 2, 2, 0.0),   # null matches only null; '*' matches everything
    ("wildcard", 3, 7, 0.0),     # null behaves like '*' (a null source matches everything)
    ("incompatible", 0, 2, 0.5), # a null on either side matches nothing
])
def test_cvpl_c15_null_and_wildcard_policy_is_explicit(policy, lo, hi, zero):
    src = pd.DataFrame({"g": ["A", None], "x": [1.0, 1.1]}, index=[5, 6])
    rel = pd.DataFrame({"g": ["A", "*", None], "x": [1.0, 1.05, 1.1]}, index=[70, 71, 72])
    rel = pd.concat([rel, pd.DataFrame({"g": ["Z"] * 4, "x": [-100.0, -101.0, -102.0, -103.0]}, index=[80, 81, 82, 83])])
    cfg = CVPLConfig.from_dict({"qids": [EXACT_G], "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5,
                                "null_policy": policy})
    r = run_cvpl(src, rel, cfg)
    summary = r["candidate_count_summary"]
    assert (summary["min"], summary["max"]) == (lo, hi)
    assert summary["zero_fraction"] == zero
    qs = r["qid_status"]["g"]
    assert qs["source_null"] == 1 and qs["release_wildcard"] == 1 and qs["release_null"] == 1


# --- C17: resource budget ---------------------------------------------------------------------------
def test_cvpl_c17_over_budget_rung_is_not_measured_with_a_typed_reason_never_zero():
    src = source_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(6)])
    rel = release_frame([("A", 1.0 + i / 10, f"s{i}") for i in range(6)] + [("B", 1.0, f"b{i}") for i in range(6)])
    cfg = CVPLConfig.from_dict({
        "qids": [EXACT_G], "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5,
        "blocking_ladder": [["g"], []], "ground_truth": GT, "max_comparisons": 50})
    r = run_cvpl(src, rel, cfg)
    t = r["blocking_ladder_trajectory"]
    assert t[0]["status"] == "measured" and t[0]["planned_comparisons"] == 36
    assert t[1]["status"] == "not_measured" and t[1]["reason"] == "resource_limit"
    assert t[1]["planned_comparisons"] == 72
    # the headline is the last rung: unmeasured, with the reason - not a safe-looking 0
    assert r["cvpl_lr"] is None
    assert r["cvpl_lr_status"] == {"value": None, "status": "not_measured", "reason": "resource_limit"}
    assert r["cvpl_id_uniform_gt"] is None


def test_cvpl_c17_group_level_limit_is_also_reported_not_raised():
    src = pd.DataFrame({"g": [f"{i}" for i in range(200)], "x": np.arange(200.0)})
    rel = pd.DataFrame({"g": ["*"] + [f"{i}" for i in range(200)], "x": np.arange(201.0)})
    cfg = CVPLConfig.from_dict({"qids": [EXACT_G], "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5,
                                "max_group_pairs": 1000})
    r = run_cvpl(src, rel, cfg)
    assert r["blocking_ladder_trajectory"][0]["reason"] == "resource_limit"
    assert r["cvpl_lr"] is None


# --- C18: the historical goldens are legacy compatibility fixtures --------------------------------------
_QIDS = [{"name": "age", "bin_width": 10}, "gender", "region_level1"]


@pytest.mark.parametrize(("name", "second_seed", "tau", "lr", "linked", "weighted", "components"), [
    ("G1", None, 0.5, 1.0, 60, 1.0, 4),
    ("G3", 2, 0.9, 0.6, 36, 0.581294, 4),
    ("G5", 999, 0.95, 0.333333, 20, 0.29803, 5),
])
def test_cvpl_c18_legacy_goldens_G1_G3_G5(name, second_seed, tau, lr, linked, weighted, components):
    """Numbers produced by the BEST ATK-CVPL-001 implementation on the same data and
    parameters (identifier excluded from the features). They are compatibility fixtures
    for the legacy weighted score - not a statement that the numbers measure correct
    identification: G3 compares two *independent* populations and still links 60 %."""
    d = make_df(60, 1)
    dp = d.copy() if second_seed is None else make_df(60, second_seed)
    cfg = CVPLConfig.from_dict({"qids": _QIDS, "auxiliary": ["income"], "tau": tau, "seed": 42})
    r = run_cvpl(d, dp, cfg)
    assert r["cvpl_lr"] == pytest.approx(lr, abs=1e-6)
    assert r["n_linked"] == linked
    assert r["cvpl_weighted_lr_legacy"] == pytest.approx(weighted, abs=1e-6)
    assert r["projection"]["n_components"] == components
    assert r["metric_semantics"]["cvpl_weighted_lr_legacy"] == "weighted_existential_linkage_v1"


@pytest.mark.parametrize(("name", "tau"), [("G2", 0.8), ("G4", 0.8)])
def test_cvpl_c18_legacy_goldens_G2_G4_projection_only(name, tau):
    """G2/G4 leave ``person_id`` in the feature set (60 one-hot columns bounded to 50).

    LEGACY-ONLY DIVERGENCE, accepted and pinned (CDX-CVPL-16). The projection is reproduced
    exactly (28 components, 0.903622 of the variance), but the historical linked counts are
    NOT: BEST linked 21 records (LR 0.35, legacy weighted rate 0.319195), this implementation
    links 19 (LR 0.316667, weighted 0.276781). 49 of the one-hot columns have identical
    variance, so the PCA basis inside that degenerate eigenspace is arbitrary: cutting the
    kept components through it rotates the cosine geometry, and a different (equally valid)
    solver legitimately links a different number of pairs. The numbers below are what THIS
    implementation produces (deterministic: the covariance eigen-decomposition with canonical
    signs); they pin it against accidental drift and must not be read as reference values or
    as evidence about the BEST numbers. The fixture is not a correctness oracle."""
    d, dp = make_df(60, 1), make_df(60, 2)
    cfg = CVPLConfig.from_dict({"qids": _QIDS, "auxiliary": ["income", "person_id"], "tau": tau, "seed": 42})
    r = run_cvpl(d, dp, cfg)
    assert r["projection"]["n_components"] == 28
    assert r["projection"]["variance_retained"] == pytest.approx(0.903622, abs=1e-6)
    assert r["projection"]["encoding"]["person_id"]["bounded"] is True
    assert r["n_undetermined"] == 0
    assert r["n_linked"] == 19
    assert r["cvpl_lr"] == pytest.approx(0.3166666667, abs=1e-9)
    assert r["cvpl_weighted_lr_legacy"] == pytest.approx(0.2767808592, abs=1e-9)
