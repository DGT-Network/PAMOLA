"""Regression tests for the adversarial review of the CVPL implementation (CDX-CVPL-n).

Every test reproduces one of the reviewer's fixtures (``codex_scratch/cvpl``) with the
reviewer's expected numbers. Where a fixture used to give a measured-looking wrong number
the assertion is the corrected value, a ``bounded`` status with bounds, or an explicit
``not_applicable`` / ``not_measured``.
"""

import json

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

GT = {"source_id_column": "id", "release_id_column": "id", "scope": "complete"}
BASE = dict(
    qids=[{"name": "g", "generalized": False}], auxiliary=["x"], qid_as_features=False,
    ground_truth=GT, seed=0,
)


def frame(g, x, ids):
    return pd.DataFrame({"g": g, "x": x, "id": ids})


def cv(s, r, **params):
    return run_cvpl(s, r, CVPLConfig.from_dict({**BASE, **params}))


# ======================================================================================
# CDX-CVPL-1: one admissibility predicate for planning, scanning and the union
# ======================================================================================
def test_cdx_cvpl_1_union_null_wildcard_fixture():
    s = frame([None, None], [-1, 1], ["a", "b"])
    s["h"] = ["A", "B"]
    r = frame(["*", "*"], [-1, 1], ["a", "b"])
    r["h"] = ["A", "B"]
    res = cv(s, r, qids=[{"name": "g", "generalized": False}, {"name": "h", "generalized": False}],
             blocking_ladder=[["g"], ["h"]], null_policy="incompatible")
    # rung 1 has no admissible pair (null is admissible with nothing); rung 2 links both
    assert res["headline_basis"] == "union_of_rungs"
    assert res["cvpl_lr"] == res["cvpl_id_uniform_gt"] == res["cvpl_top1_success_gt"] == 1.0
    traj = res["candidate_union_trajectory"]
    assert traj[0]["cvpl_lr_union"] == 0.0 and traj[-1]["cvpl_lr_union"] == 1.0
    assert res["cvpl_lr"] == traj[-1]["cvpl_lr_union"]  # headline == trajectory end


def _adm(a, b, policy):
    an, bn = a is None, b is None
    if policy == "incompatible" and (an or bn):
        return False
    if policy == "wildcard" and (an or bn):
        return True
    if a == "*" or b == "*":
        return True
    if an and bn:
        return policy == "equal_null"
    if an or bn:
        return False
    return a == b


@pytest.mark.parametrize("policy", ["equal_null", "wildcard", "incompatible"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cdx_cvpl_1_union_equals_explicit_pair_set_union(policy, seed):
    rng = np.random.default_rng(seed)
    n, m = 25, 40
    pick = lambda k: rng.choice(np.array(["A", "B", "*", None], dtype=object), k)
    s = pd.DataFrame({"p": pick(n), "q": pick(n), "x": rng.normal(size=n), "id": [f"s{i}" for i in range(n)]})
    r = pd.DataFrame({"p": pick(m), "q": pick(m), "x": rng.normal(size=m), "id": rng.integers(0, 30, m).astype(str)})
    r["id"] = np.where(r["id"].astype(int) < 25, "s" + r["id"], "z" + r["id"])
    mean = float(np.concatenate([s["x"], r["x"]]).mean())
    sign_s, sign_r = np.sign(s["x"].to_numpy() - mean), np.sign(r["x"].to_numpy() - mean)

    # existential linkage is evaluated over the UNION of the two single-QID candidate sets;
    # the identification measures are evaluated per rung (each a declared attacker profile)
    # and the headline is the largest value over the rungs (CDX-CVPL-R2-6)
    lr = 0.0
    per_rung = {"p": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0]}  # subjects, rows, top1
    for i in range(n):
        union = [j for j in range(m)
                 if sign_s[i] == sign_r[j] and (_adm(s.p[i], r.p[j], policy) or _adm(s.q[i], r.q[j], policy))]
        lr += bool(union)
        for col in ("p", "q"):
            rows = [j for j in range(m) if sign_s[i] == sign_r[j] and _adm(s[col][i], r[col][j], policy)]
            if not rows:
                continue
            subj = {r.id[j] for j in rows}
            own = s.id[i] in subj
            per_rung[col][0] += own / len(subj)
            per_rung[col][1] += sum(r.id[j] == s.id[i] for j in rows) / len(rows)
            per_rung[col][2] += own / len(subj)  # every passing cosine is +1: all tie
    uni_s = max(v[0] for v in per_rung.values())
    uni_r = max(v[1] for v in per_rung.values())
    top = max(v[2] for v in per_rung.values())
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": "p", "generalized": False}, {"name": "q", "generalized": False}],
        "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5, "null_policy": policy,
        "blocking_ladder": [["p"], ["q"]], "ground_truth": GT})
    res = run_cvpl(s, r, cfg)
    assert res["headline_basis"] == "union_of_rungs"
    assert res["cvpl_lr"] == pytest.approx(lr / n, abs=1e-12)
    assert res["cvpl_lr"] == pytest.approx(res["candidate_union_trajectory"][-1]["cvpl_lr_union"], abs=1e-12)
    assert res["cvpl_id_uniform_subjects_gt"] == pytest.approx(uni_s / n, abs=1e-12)
    assert res["cvpl_id_uniform_rows_gt"] == pytest.approx(uni_r / n, abs=1e-12)
    assert res["cvpl_top1_success_gt"] == pytest.approx(top / n, abs=1e-12)


# ======================================================================================
# CDX-CVPL-2: open-ended intervals -> finite encoding; non-finite never reaches the threshold
# ======================================================================================
def test_cdx_cvpl_2_open_ended_intervals_are_encoded_finitely():
    s = frame([61, 71], [1, 2], ["a", "b"])
    r = frame(["60+", "70+"], [1, 2], ["a", "b"])
    res = cv(s, r, qids=[{"name": "g", "kind": "numeric"}], auxiliary=[], qid_as_features=True)
    assert res["cvpl_lr"] == res["cvpl_id_uniform_gt"] == res["cvpl_top1_success_gt"] == 1.0
    assert res["projection"]["variance_retained"] is not None
    json.dumps(res.to_dict(), allow_nan=False, default=str)  # no NaN anywhere in the result


def test_cdx_cvpl_2_non_finite_representation_is_not_measured_never_a_zero(monkeypatch):
    from pamola_core.metrics.privacy.cvpl.projection import CVPLProjection

    real = CVPLProjection.project

    def poisoned(self, cols, n_rows, chunk_rows=None):
        out = real(self, cols, n_rows, chunk_rows)
        out[0, 0] = np.nan
        return out

    monkeypatch.setattr(CVPLProjection, "project", poisoned)
    res = cv(frame(["A", "A"], [-1, 1], ["a", "b"]), frame(["A", "A"], [-1, 1], ["a", "b"]))
    assert res["cvpl_lr"] is None
    assert res["cvpl_lr_status"]["reason"] == "non_finite_representation"
    assert res["cvpl_id_uniform_gt"] is None


def test_cdx_cvpl_2_infinite_point_values_are_invalid_not_infinite_geometry():
    s = frame(["A", "A"], [-1, 1], ["a", "b"])
    s["h"] = [np.inf, 5.0]
    r = frame(["A", "A"], [-1, 1], ["a", "b"])
    r["h"] = [np.inf, 5.0]
    res = cv(s, r, auxiliary=["x", "h"])
    json.dumps(res.to_dict(), allow_nan=False, default=str)
    assert res["projection"]["imputed_values"]["h"]["source"] == 1


# ======================================================================================
# CDX-CVPL-3: ground-truth identifier equality
# ======================================================================================
def test_cdx_cvpl_3_numeric_ids_compare_as_numbers():
    s = frame(["A", "B"], [-1, 1], [1, 2])
    r = frame(["A", "B"], [-1, 1], [1.0, 2.0])
    res = cv(s, r)
    assert res["cvpl_lr"] == res["cvpl_id_uniform_gt"] == res["cvpl_top1_success_gt"] == res["precision_at_tau"] == 1.0
    assert res["gt_coverage"] == 1.0


def test_cdx_cvpl_3_numeric_and_text_identifiers_are_not_conflated():
    s = frame(["A", "B"], [-1, 1], [1, 2])
    r = frame(["A", "B"], [-1, 1], ["1", "2"])
    res = cv(s, r)
    assert res["ground_truth"]["status"] == "invalid"
    assert res["ground_truth"]["reason"].startswith("identifier_types_incompatible")
    assert res["cvpl_id_uniform_gt"] is None
    assert res["cvpl_id_uniform_gt_status"]["status"] == "not_measured"
    assert res["cvpl_lr"] == 1.0  # exposure itself does not need the ground truth


def test_cdx_cvpl_3_unsupported_identifier_types_are_rejected_not_coerced():
    s = frame(["A", "B"], [-1, 1], [True, False])
    r = frame(["A", "B"], [-1, 1], [True, False])
    res = cv(s, r)
    assert res["ground_truth"]["status"] == "invalid"
    assert "unsupported_identifier" in res["ground_truth"]["reason"]
    s2 = frame(["A", "B"], [-1, 1], [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])
    assert cv(s2, s2.copy())["ground_truth"]["status"] == "invalid"


def test_cdx_cvpl_3_large_integer_and_float_ids():
    big = 2**60
    s = frame(["A", "B"], [-1, 1], [big, big + 1])
    r = frame(["A", "B"], [-1, 1], [big, big + 1])
    assert cv(s, r)["cvpl_id_uniform_gt"] == 1.0
    # float 3.0 is the integer 3, 3.5 stays itself
    s = frame(["A", "B"], [-1, 1], [3, 4])
    r = frame(["A", "B"], [-1, 1], [3.0, 4.5])
    assert cv(s, r)["cvpl_id_uniform_gt"] == pytest.approx(0.5)


# ======================================================================================
# CDX-CVPL-4: undefined cosines -> bounds, never a measured point value
# ======================================================================================
def test_cdx_cvpl_4_undefined_unlinked_is_bounded_not_measured_one_third():
    s = frame(["A", "B", "C"], [-1, 0, 1], ["a", "b", "c"])
    r = frame(["A", "B", "C"], [-1, 1, 0], ["a", "b", "c"])
    res = cv(s, r)
    third = 1 / 3
    assert res["n_undetermined"] == 2
    assert res["cvpl_lr"] is None and res["cvpl_lr_status"]["status"] == "bounded"
    assert (res["cvpl_lr_lower"], res["cvpl_lr_upper"]) == (pytest.approx(third), 1.0)
    for key in ("cvpl_id_uniform_subjects_gt", "cvpl_id_uniform_rows_gt", "cvpl_id_uniform_gt",
                "cvpl_top1_success_gt"):
        st = res[f"{key}_status"]
        assert res[key] is None and st["status"] == "bounded", key
        assert st["lower"] == pytest.approx(third) and st["upper"] == pytest.approx(1.0), key
        assert st["n_uncertain"] >= 2
    p = res["precision_at_tau_status"]
    assert res["precision_at_tau"] is None and p["status"] == "bounded"
    assert p["lower"] == pytest.approx(third) and p["upper"] == 1.0


def test_cdx_cvpl_4_undefined_hidden_candidate_is_not_lost_behind_a_false_link():
    s = frame(["A", "B"], [-2, 2], ["a", "b"])
    r = frame(["A", "A", "B"], [-2, 0, 2], ["wrong", "a", "b"])
    res = cv(s, r)
    assert res["cvpl_lr"] == 1.0  # the false link already proves existential linkage
    assert res["n_undetermined"] == 0
    # ... but the undefined TRUE candidate still decides the identification measures
    for key, (lo, hi) in {
        "cvpl_id_uniform_subjects_gt": (0.5, 0.75), "cvpl_id_uniform_rows_gt": (0.5, 0.75),
        "cvpl_id_uniform_gt": (0.5, 0.75), "cvpl_top1_success_gt": (0.5, 1.0),
    }.items():
        st = res[f"{key}_status"]
        assert res[key] is None and st["status"] == "bounded", key
        assert (st["lower"], st["upper"]) == (pytest.approx(lo), pytest.approx(hi)), key
        assert st["n_uncertain"] == 1, key


def test_cdx_cvpl_4_a_measure_no_unknown_can_touch_stays_a_point_value():
    s = frame(["A", "B"], [-1, 1], ["a", "b"])
    r = frame(["A", "B"], [-1, 1], ["a", "b"])
    res = cv(s, r)
    assert res["cvpl_id_uniform_gt_status"]["status"] == "measured"
    assert res["cvpl_id_uniform_gt_status"]["n_uncertain"] == 0


# ======================================================================================
# CDX-CVPL-7: tie tolerance never admits a candidate that failed the threshold
# ======================================================================================
def test_cdx_cvpl_7_tie_tolerance_cannot_readmit_a_below_tau_candidate():
    s = pd.DataFrame({"g": ["A", "Z", "Z", "Z"], "x": [1, -1, 0, 0], "y": [0, 0, 1, -1],
                      "id": ["a", "b", "c", "d"]})
    c = 1 - 5e-10
    r = pd.DataFrame({"g": ["A", "A"], "x": [1, c], "y": [0, np.sqrt(1 - c * c)], "id": ["wrong", "a"]})
    res = cv(s, r, auxiliary=["x", "y"], fit_strategy="source_only", tau=1.0, min_components=2, max_components=2)
    assert res["cvpl_lr"] == 0.25
    assert res["cvpl_id_uniform_gt"] == 0.0
    assert res["cvpl_top1_success_gt"] == 0.0
    assert res["precision_at_tau"] == 0.0


# ======================================================================================
# CDX-CVPL-8: partial ground truth never invents a known subset
# ======================================================================================
def test_cdx_cvpl_8_anonymous_release_rows_make_membership_unknown():
    s = frame(["A"], [1], ["a"])
    r = frame(["A", "A"], [1, 1], ["a", None])
    res = cv(s, r)
    assert res["cvpl_id_uniform_gt"] is None
    assert res["gt_coverage"] == 0.0
    st = res["cvpl_id_uniform_gt_status"]
    assert st["status"] == "not_measured"
    assert st["known_subset_value"] is None and st["known_subset_denominator"] == 0
    assert st["known_subset_reason"] == "release_membership_unknown"
    assert res["ground_truth"]["release_membership_unknown"] is True


def test_cdx_cvpl_8_a_fully_identified_release_still_gives_a_known_subset():
    s = frame(["A", "A"], [1, 1.1], ["a", None])
    r = frame(["A", "A"], [1, 1.1], ["a", "q"])
    res = cv(s, r)
    st = res["cvpl_id_uniform_gt_status"]
    assert res["gt_coverage"] == 0.5
    assert st["known_subset_denominator"] == 1 and st["known_subset_value"] is not None
    assert res["cvpl_id_uniform_gt"] is None  # still no population figure


# ======================================================================================
# CDX-CVPL-9: exact numeric equality, invalid != missing
# ======================================================================================
def test_cdx_cvpl_9_signed_zero_is_one_value():
    s = frame([-0.0, 1.0], [-1, 1], ["a", "b"])
    r = frame([0.0, 1.0], [-1, 1], ["a", "b"])
    res = cv(s, r, qids=[{"name": "g", "kind": "numeric"}])
    assert res["cvpl_lr"] == res["cvpl_id_uniform_gt"] == res["cvpl_top1_success_gt"] == 1.0


def test_cdx_cvpl_9_integers_beyond_float_precision_stay_distinct():
    s = frame([2**53], [-1], ["a"])
    r = frame([2**53 + 1], [-1], ["b"])
    assert cv(s, r, qids=[{"name": "g", "kind": "numeric"}])["cvpl_lr"] == 0.0
    r_same = frame([2**53], [-1], ["a"])
    same = cv(s, r_same, qids=[{"name": "g", "kind": "numeric"}])
    assert same["cvpl_lr"] == 1.0
    # ... also as a bin / interval endpoint comparison between two huge neighbours
    s2 = frame([2**60, 2**60 + 1], [-1, 1], ["a", "b"])
    r2 = frame([2**60, 2**60 + 1], [-1, 1], ["a", "b"])
    out = cv(s2, r2, qids=[{"name": "g", "kind": "numeric"}])
    assert out["cvpl_id_uniform_gt"] == 1.0


def test_cdx_cvpl_9_an_invalid_release_value_is_not_a_missing_value():
    spec = [{"name": "g", "kind": "numeric", "generalized": False}]
    res = cv(frame([None], [1], ["a"]), frame(["bad"], [1], ["a"]), qids=spec)
    assert res["cvpl_lr"] == 0.0
    # missing still matches missing; invalid matches nothing, not even invalid
    assert cv(frame([None], [1], ["a"]), frame([None], [1], ["a"]), qids=spec)["cvpl_lr"] == 1.0
    assert cv(frame(["bad"], [1], ["a"]), frame(["bad"], [1], ["a"]), qids=spec)["cvpl_lr"] == 0.0


def test_cdx_cvpl_9_an_invalid_source_value_is_not_a_missing_value():
    spec = [{"name": "g", "kind": "numeric", "generalized": False}]
    res = cv(frame(["bad"], [1], ["a"]), frame([None], [1], ["a"]), qids=spec)
    assert res["cvpl_lr"] == 0.0
    assert res["qid_status"]["g"]["source_unparsable"] == 1


# ======================================================================================
# CDX-CVPL-10: no literal value collides with an internal tag
# ======================================================================================
def test_cdx_cvpl_10_literal_missing_marker_is_a_distinct_category():
    s = frame(["A"] * 3, [None, "__missing__", "X"], ["a", "b", "c"])
    res = cv(s, s.copy())
    assert res["cvpl_id_uniform_gt"] == res["cvpl_top1_success_gt"] == 1.0
    # the control with an ordinary label was always 1
    ctrl = s.assign(x=["MISSING_CATEGORY", "__missing__", "X"])
    assert cv(ctrl, ctrl.copy())["cvpl_id_uniform_gt"] == 1.0


def test_cdx_cvpl_10_internal_tags_cannot_be_typed_by_a_user():
    from pamola_core.metrics.privacy.cvpl.projection import categorical_values

    vals = categorical_values(pd.Series(["__missing__", None, "__other__", "o:", "m:", "v:x"]))
    assert len(set(vals.tolist())) == 6


def test_cdx_cvpl_10_literal_other_bucket_label_does_not_merge_with_the_overflow_bucket():
    labels = [f"c{i}" for i in range(8)] + ["__other__"]
    s = frame(["A"] * 9, labels, [f"s{i}" for i in range(9)])
    res = cv(s, s.copy(), max_categories=4)
    assert res["bucketed_features"][0]["distinct_values"] == 9
    # the literal '__other__' is one ordinary (rare) value, counted like the others
    assert res["bucketed_features"][0]["distinct_values_collapsed"] == 9 - 3


# ======================================================================================
# CDX-CVPL-11: hierarchy coverage
# ======================================================================================
def test_cdx_cvpl_11_irrelevant_hierarchy_does_not_make_generalized_values_applicable():
    s = frame(["Lyon", "Paris"], [-1, 1], ["a", "b"])
    r = frame(["South", "North"], [-1, 1], ["a", "b"])
    res = cv(s, r, qids=[{"name": "g", "kind": "categorical", "generalized": True,
                          "hierarchy": {"Elsewhere": "Other"}}])
    assert res["qid_status"]["g"]["applicable"] is False
    assert res["qid_status"]["g"]["limitation"] == "generalized_values_not_in_hierarchy"
    assert res["cvpl_lr"] is None
    assert res["cvpl_lr_status"]["status"] == "not_applicable"


def test_cdx_cvpl_11_a_hierarchy_that_names_the_values_is_applicable():
    s = frame(["Lyon", "Paris"], [-1, 1], ["a", "b"])
    r = frame(["South", "North"], [-1, 1], ["a", "b"])
    res = cv(s, r, qids=[{"name": "g", "kind": "categorical",
                          "hierarchy": {"Lyon": "South", "Paris": "North"}}])
    assert res["cvpl_lr"] == 1.0


# ======================================================================================
# CDX-CVPL-14: an empty candidate set settles before projection degeneracy
# ======================================================================================
def test_cdx_cvpl_14_completely_suppressed_release_is_a_measured_zero():
    res = cv(frame(["A"], [1], ["a"]), frame([], [], []))
    assert res["cvpl_lr"] == res["cvpl_id_uniform_gt"] == res["cvpl_top1_success_gt"] == 0.0
    assert res["cvpl_id_uniform_gt_status"]["status"] == "measured"
    assert res["cvpl_id_uniform_gt_status"]["denominator"] == 1


def test_cdx_cvpl_14_no_admissible_candidate_with_a_constant_source_is_a_measured_zero():
    res = cv(frame(["A"], [1], ["a"]), frame(["B"], [1], ["a"]))
    assert res["cvpl_lr"] == 0.0 and res["cvpl_lr_status"]["status"] == "measured"
    assert res["cvpl_id_uniform_gt"] == 0.0


# ======================================================================================
# CDX-CVPL-17: both uniform strategies; the headline is the larger population value
# ======================================================================================
def test_cdx_cvpl_17_subjects_rows_and_headline_on_the_fixed_subject_count_fixture():
    s = frame(["A"], [1], ["a"])
    r = frame(["A"] * 3, [1] * 3, ["a", "a", "b"])
    res = cv(s, r)
    assert res["cvpl_id_uniform_subjects_gt"] == pytest.approx(0.5)
    assert res["cvpl_id_uniform_rows_gt"] == pytest.approx(2 / 3)
    assert res["cvpl_id_uniform_gt"] == pytest.approx(2 / 3)
    assert res["cvpl_id_uniform_basis"] == "rows"
    st = res["cvpl_id_uniform_gt_status"]
    assert st["status"] == "measured" and st["basis"] == "rows"


def test_cdx_cvpl_17_headline_is_never_below_either_strategy():
    rng = np.random.default_rng(5)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 20
        s = pd.DataFrame({"g": rng.choice(["A", "B"], n), "x": rng.normal(size=n), "id": [f"s{i}" for i in range(n)]})
        r = pd.DataFrame({"g": rng.choice(["A", "B"], 30), "x": rng.normal(size=30),
                          "id": [f"s{i}" for i in rng.integers(0, 25, 30)]})
        res = cv(s, r, tau=0.3)
        head = res["cvpl_id_uniform_gt"]
        assert head >= res["cvpl_id_uniform_subjects_gt"] - 1e-12
        assert head >= res["cvpl_id_uniform_rows_gt"] - 1e-12
        # complete GT, defined cosines: identification never exceeds existential linkage
        assert head <= res["cvpl_lr"] + 1e-12
        assert res["cvpl_top1_success_gt"] <= res["cvpl_lr"] + 1e-12
        assert head == pytest.approx(max(res["cvpl_id_uniform_subjects_gt"], res["cvpl_id_uniform_rows_gt"]))
        assert res["cvpl_id_uniform_basis"] in ("subjects", "rows")


def test_cdx_cvpl_17_no_ground_truth_publishes_none_for_all_strategies():
    res = cv(frame(["A"], [1], ["a"]), frame(["A"], [1], ["a"]), ground_truth=None)
    for key in ("cvpl_id_uniform_subjects_gt", "cvpl_id_uniform_rows_gt", "cvpl_id_uniform_gt"):
        assert res[key] is None and res[f"{key}_status"]["reason"] == "no_ground_truth"


# ======================================================================================
# CDX-CVPL-13: duplicates and the fitted representation
# ======================================================================================
def test_cdx_cvpl_13_duplicate_release_rows_move_a_joint_fit():
    s = frame(["A"], [1], ["a"])
    r = frame(["A", "A", "Z", "Z"], [1, 2, -2, 5.5], ["a", "b", "c", "d"])
    r_dup = pd.concat([r, r.iloc[[2]]], ignore_index=True)
    before, after = cv(s, r), cv(s, r_dup)
    assert before["cvpl_id_uniform_gt"] == 1.0 and before["cvpl_top1_success_gt"] == 1.0
    # joint fit: the duplicated row outside the candidate set shifts the mean, the geometry,
    # and with it the result (documented: invariance needs a fixed representation)
    assert after["cvpl_id_uniform_gt"] == 0.5 and after["cvpl_top1_success_gt"] == 0.5


def test_cdx_cvpl_13_duplicate_release_rows_do_not_move_a_fixed_representation():
    s = frame(["A"], [1], ["a"])
    r = frame(["A", "A", "Z", "Z"], [1, 2, -2, 5.5], ["a", "b", "c", "d"])
    r_dup = pd.concat([r, r.iloc[[2]]], ignore_index=True)
    before, after = cv(s, r, fit_strategy="source_only"), cv(s, r_dup, fit_strategy="source_only")
    for key in ("cvpl_lr", "cvpl_id_uniform_subjects_gt", "cvpl_top1_success_gt"):
        assert before[key] == after[key]
    # a duplicate of a CANDIDATE row changes the row-uniform strategy but not the subject one
    cand_dup = pd.concat([r, r.iloc[[0]]], ignore_index=True)
    assert cv(s, cand_dup, fit_strategy="source_only")["cvpl_id_uniform_subjects_gt"] == \
        before["cvpl_id_uniform_subjects_gt"]


# ======================================================================================
# CDX-CVPL-12: bounded working memory
# ======================================================================================
def _watch_keys(monkeypatch):
    import pamola_core.metrics.privacy.cvpl.engine as eng

    seen = {"max_keys": 0}
    original = eng._distinct

    def watching(keys, m, S, own):
        if keys:
            seen["max_keys"] = max(seen["max_keys"], sum(len(k) for k in keys))
        return original(keys, m, S, own)

    monkeypatch.setattr(eng, "_distinct", watching)
    return seen


@pytest.mark.parametrize("n", [250, 500, 1000])
def test_cdx_cvpl_12_exact_fallback_state_is_per_chunk_not_per_pair(monkeypatch, n):
    seen = _watch_keys(monkeypatch)
    s = frame(["A"] * n, [1.0] * n, np.arange(n))
    res = cv(s, s.copy(), chunk_pairs=10)
    assert res["n_exact_fallback_pairs"] == n * n  # every pair counted, none retained
    assert res["n_compared"] == 0  # cosine evaluations only; fallback work is its own count
    # one record per chunk: its distinct-subject keys are bounded by the subjects, not N*N
    assert seen["max_keys"] <= 2 * n


@pytest.mark.parametrize("n", [250, 500, 1000])
def test_cdx_cvpl_12_duplicate_subject_accumulators_do_not_grow_with_the_pair_count(monkeypatch, n):
    seen = _watch_keys(monkeypatch)
    s = frame(["A"] * n, np.linspace(1, 2, n), np.arange(n))
    r = pd.concat([s, s.iloc[[0]]], ignore_index=True)
    cv(s, r, chunk_pairs=10, fit_strategy="source_only")
    assert seen["max_keys"] <= 2 * n


def test_cdx_cvpl_12_planning_streams_compatibility_and_refuses_early(monkeypatch):
    from pamola_core.metrics.privacy.cvpl import compat

    cells = {"max": 0, "calls": 0}
    original = compat.RungPlan._compat_row

    def watching(self, part, g0, g1):
        cells["calls"] += 1
        cells["max"] = max(cells["max"], (g1 - g0) * len(part.rep_r))
        return original(self, part, g0, g1)

    monkeypatch.setattr(compat.RungPlan, "_compat_row", watching)
    n = 3000
    s = pd.DataFrame({"g": [str(i) for i in range(n)], "x": np.arange(n, dtype=float), "id": np.arange(n)})
    r = pd.concat([s, pd.DataFrame({"g": ["*"], "x": [-1.0], "id": [-1]})], ignore_index=True)
    res = cv(s, r, qids=[{"name": "g", "generalized": False}], max_comparisons=10)
    assert res["cvpl_lr_status"]["reason"] == "resource_limit"
    assert cells["max"] <= 4_000_000  # never the 9M-cell matrix
    assert cells["calls"] == 1  # the budget stopped planning after the first tile


