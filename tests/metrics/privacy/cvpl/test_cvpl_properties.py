"""CVPL properties: an independent oracle, Theorem 1, determinism, configuration, parsing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pamola_core.errors.exceptions import InvalidParameterError
from pamola_core.metrics.privacy.cvpl import CVPLConfig, QIDSpec, run_cvpl
from pamola_core.metrics.privacy.cvpl.compat import (
    STATE_VALUE,
    STATE_WILD,
    classify_transition,
    encode_qid,
)

from .helpers import GT, make_df


# ---------------------------------------------------------------------------------------
# an independent, deliberately naive oracle (python loops, scikit-learn PCA)
# ---------------------------------------------------------------------------------------
def _oracle(src, rel, qids, aux, cat_aux, tau, rungs, union=False):
    feats_num = ["n1"] + [c for c in aux if c.startswith("n")]
    feats_num = list(dict.fromkeys(feats_num))
    cats = list(qids) + list(cat_aux)
    both = pd.concat([src, rel], ignore_index=True)
    num = both[feats_num].to_numpy(float)
    num = np.where(np.isnan(num), np.nanmean(num, axis=0), num)
    num = StandardScaler().fit_transform(num)
    ohe = OneHotEncoder(sparse_output=False).fit(both[cats].astype(str))
    x = np.hstack([num, ohe.transform(both[cats].astype(str))])
    n_src = len(src)
    max_k = max(1, min(len(both), x.shape[1], 50))
    pca = PCA(n_components=max_k, svd_solver="full").fit(x)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = max(1, min(max(3, min(int(np.searchsorted(cum, 0.9)) + 1, max_k)), max_k))
    z = pca.transform(x)[:, :k]
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.where(nrm == 0, 1, nrm)
    zs, zr = z[:n_src], z[n_src:]
    out = []
    for rung in rungs:
        linked, uni, top1, declared = 0, 0.0, 0.0, 0
        for i in range(n_src):
            passing = []
            for j in range(len(rel)):
                ok = False
                for one in (rungs if union else [rung]):
                    this = True
                    for q in one:
                        a, b = src[q].iloc[i], rel[q].iloc[j]
                        if not (a == b or b == "*" or a == "*"):
                            this = False
                            break
                    ok = ok or this
                if ok:
                    sim = float(zs[i] @ zr[j])
                    if sim >= tau - 1e-12:
                        passing.append((j, sim))
            if passing:
                linked += 1
            subjects = {rel["rid"].iloc[j] for j, _ in passing}
            sid = src["sid"].iloc[i]
            if subjects:
                uni += (sid in subjects) / len(subjects)
                best = max(s for _, s in passing)
                tied = {rel["rid"].iloc[j] for j, s in passing if s >= best - 1e-9}
                top1 += (sid in tied) / len(tied)
                declared += 1
        out.append({"lr": linked / n_src, "uni": uni / n_src, "top1": top1 / n_src,
                    "prec": (top1 / declared) if declared else None})
    return out


def _random_frames(seed, n=40, m=70, dup=True):
    rng = np.random.default_rng(seed)
    src = pd.DataFrame({
        "a": rng.choice(["a1", "a2", "a3"], n), "b": rng.choice(["b1", "b2"], n),
        "n1": rng.normal(size=n), "n2": rng.normal(size=n), "c": rng.choice(["c1", "c2", "c3", "c4"], n),
        "sid": [f"s{i}" for i in range(n)],
    })
    base = src.sample(m // 2, random_state=seed).copy()
    base["n1"] += rng.normal(scale=0.3, size=len(base))
    noise = pd.DataFrame({
        "a": rng.choice(["a1", "a2", "a3", "*"], m - len(base)), "b": rng.choice(["b1", "b2"], m - len(base)),
        "n1": rng.normal(size=m - len(base)), "n2": rng.normal(size=m - len(base)),
        "c": rng.choice(["c1", "c2", "c3", "c4"], m - len(base)),
        "sid": [f"x{i}" for i in range(m - len(base))],
    })
    rel = pd.concat([base, noise], ignore_index=True).rename(columns={"sid": "rid"})
    if dup:  # repeated rows of the same subject
        rel = pd.concat([rel, rel.iloc[:6]], ignore_index=True)
    rel.index = rng.permutation(len(rel)) + 5000
    return src, rel


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("tau", [0.3, 0.8])
def test_engine_matches_an_independent_oracle(seed, tau):
    src, rel = _random_frames(seed)
    rungs = [["a", "b"], ["a"], []]
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": "a", "generalized": False}, {"name": "b", "generalized": False}],
        "auxiliary": ["n1", "n2", "c"], "tau": tau, "blocking_ladder": rungs, "ground_truth": GT,
    })
    result = run_cvpl(src, rel, cfg)
    expected = _oracle(src, rel, ["a", "b"], ["n1", "n2", "c"], ["c"], tau, rungs)
    for entry, want in zip(result["blocking_ladder_trajectory"], expected):
        got = entry["measures"]
        assert got["cvpl_lr"] == pytest.approx(want["lr"], abs=1e-12)
        assert got["cvpl_id_uniform_subjects_gt"] == pytest.approx(want["uni"], abs=1e-9)
    # identification headline: the largest value over the declared rungs (CDX-CVPL-R2-6)
    assert result["cvpl_top1_success_gt"] == pytest.approx(max(e["top1"] for e in expected), abs=1e-9)
    precisions = [e["prec"] for e in expected if e["prec"] is not None]
    if not precisions:
        assert result["precision_at_tau"] is None
    else:
        assert result["precision_at_tau"] == pytest.approx(max(precisions), abs=1e-9)
    assert result["ground_truth"]["release_subject_duplicates"] is True


# ---------------------------------------------------------------------------------------
# Theorem 1
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("tau", [0.2, 0.6, 0.95])
def test_cvpl_theorem1_lr_is_monotone_along_a_nested_ladder(seed, tau):
    src, rel = _random_frames(seed, n=30, m=60)
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": q, "generalized": False} for q in ("a", "b", "c")],
        "auxiliary": ["n1", "n2"], "tau": tau,
        "blocking_ladder": [["a", "b", "c"], ["a", "b"], ["a"], []], "ground_truth": GT,
    })
    r = run_cvpl(src, rel, cfg)
    assert r["ladder"]["transitions"] == ["relaxing"] * 3 and r["ladder"]["nested"]
    lrs = [e["measures"]["cvpl_lr"] for e in r["blocking_ladder_trajectory"]]
    assert lrs == sorted(lrs), lrs
    cands = [e["measures"]["n_with_candidates"] for e in r["blocking_ladder_trajectory"]]
    assert cands == sorted(cands)
    # the cumulative union equals the last rung for a relaxing ladder
    assert r["cvpl_lr_union"] == lrs[-1]


def test_cvpl_non_nested_ladder_is_flagged_and_reports_the_union():
    src, rel = _random_frames(1, n=30, m=60)
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": q, "generalized": False} for q in ("a", "b")],
        "auxiliary": ["n1", "n2"], "tau": 0.6, "blocking_ladder": [["a"], ["b"]], "ground_truth": GT,
    })
    r = run_cvpl(src, rel, cfg)
    assert r["ladder"]["transitions"] == ["non_nested"]
    assert r["ladder"]["nested"] is False and r["ladder"]["monotone_lr_theorem_applies"] is False
    union = [e["cvpl_lr_union"] for e in r["candidate_union_trajectory"]]
    own = [e["measures"]["cvpl_lr"] for e in r["blocking_ladder_trajectory"]]
    assert union[0] == own[0] and union[1] >= max(own)  # the union can only grow
    assert r["cvpl_lr"] == own[-1]  # the headline is the last declared rung, not the union


@pytest.mark.parametrize(("a", "b", "expected"), [
    (["a", "b"], ["a"], "relaxing"), (["a"], ["a", "b"], "tightening"),
    (["a"], ["a"], "equal"), (["a"], ["b"], "non_nested"), (["a", "b"], [], "relaxing"),
])
def test_cvpl_transition_classification(a, b, expected):
    assert classify_transition(a, b) == expected


# ---------------------------------------------------------------------------------------
# determinism and fit strategies
# ---------------------------------------------------------------------------------------
def _strip(d):
    d = dict(d)
    d.pop("elapsed_seconds", None)
    return d


@pytest.mark.parametrize("seed", [0, 42])
def test_cvpl_is_deterministic_under_a_seed(seed):
    src, rel = make_df(60, 1), make_df(60, 2)
    rel = rel.sample(frac=1.0, random_state=3)  # row order of the release is irrelevant
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": "age", "bin_width": 10}, "gender", "region_level1"], "auxiliary": ["income"],
        "tau": 0.9, "seed": seed, "ground_truth": None,
    })
    a, b = run_cvpl(src, rel, cfg), run_cvpl(src, rel.iloc[::-1], cfg)
    assert _strip(a.metrics) == _strip(run_cvpl(src, rel, cfg).metrics)
    assert a["seed"] == seed
    assert a["cvpl_lr"] == b["cvpl_lr"] and a["n_linked"] == b["n_linked"]  # order-independent


def test_cvpl_fit_strategies_are_explicit_and_frozen_is_reusable():
    src = make_df(60, 1)
    rel_a, rel_b = make_df(60, 2), make_df(60, 3)
    base = {"qids": [{"name": "age", "bin_width": 10}, "gender", "region_level1"], "auxiliary": ["income"], "tau": 0.9}
    joint_a = run_cvpl(src, rel_a, CVPLConfig.from_dict(base))
    joint_b = run_cvpl(src, rel_b, CVPLConfig.from_dict(base))
    only = run_cvpl(src, rel_a, CVPLConfig.from_dict({**base, "fit_strategy": "source_only"}))
    assert only["projection"]["fit_strategy"] == "source_only"
    assert only["projection"]["n_fit_rows"] == 60 and joint_a["projection"]["n_fit_rows"] == 120
    # a frozen projection from the source reproduces source_only for any release
    frozen_cfg = CVPLConfig.from_dict({**base, "fit_strategy": "frozen"})
    f_a = run_cvpl(src, rel_a, frozen_cfg, frozen_projection=only.projection)
    assert f_a["n_linked"] == only["n_linked"]
    f_b = run_cvpl(src, rel_b, frozen_cfg, frozen_projection=only.projection)
    assert f_b["projection"]["n_components"] == only["projection"]["n_components"]
    with pytest.raises(InvalidParameterError):
        run_cvpl(src, rel_a, frozen_cfg)
    assert joint_b["projection"]["fit_strategy"] == "joint"


# ---------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"qids": []},
    {"qids": ["a", "a"]},
    {"qids": ["a"], "tau": 0.0},
    {"qids": ["a"], "tau": 1.5},
    {"qids": ["a"], "auxiliary": ["a"]},
    {"qids": ["a"], "fit_strategy": "target_only"},
    {"qids": ["a"], "direction": "release_to_reference"},
    {"qids": ["a"], "blocking_ladder": [["zzz"]]},
    {"qids": ["a"], "blocking_ladder": [["a"], ["a"]]},
    {"qids": ["a"], "null_policy": "drop"},
    {"qids": ["a"], "seed": True},
    {"qids": ["a"], "unknown_option": 1},
    {"qids": ["a"], "ground_truth": {"source_id_column": "a", "release_id_column": "rid"}},
    {"qids": [{"name": "a", "kind": "categorical", "generalized": True}]},
    {"qids": [{"name": "a", "kind": "numeric", "hierarchy": {"x": "y"}}]},
])
def test_cvpl_invalid_configuration_is_a_typed_error(bad):
    with pytest.raises(InvalidParameterError):
        CVPLConfig.from_dict(bad)


def test_cvpl_ground_truth_columns_can_never_be_features():
    with pytest.raises(InvalidParameterError, match="evaluation-only"):
        CVPLConfig.from_dict({"qids": ["g"], "auxiliary": ["sid"],
                              "ground_truth": {"source_id_column": "sid", "release_id_column": "rid"}})


def test_cvpl_params_round_trip():
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": "age", "bin_width": 10}, "city"], "auxiliary": ["x"], "tau": 0.7, "seed": 0,
        "blocking_ladder": [["age", "city"], ["age"]], "ground_truth": GT})
    assert CVPLConfig.from_dict(cfg.to_params()) == cfg


def test_cvpl_missing_and_unnamed_columns_are_never_added_implicitly():
    src = pd.DataFrame({"g": ["A"] * 4, "x": [1.0, 2, 3, 4], "extra": ["u", "v", "w", "t"]})
    rel = src.copy()
    cfg = CVPLConfig.from_dict({"qids": [{"name": "g", "generalized": False}], "auxiliary": ["x"], "tau": 0.5})
    r = run_cvpl(src, rel, cfg)
    assert r["projection"]["feature_columns"] == ["g", "x"]
    with pytest.raises(Exception, match="nope"):
        run_cvpl(src, rel, CVPLConfig.from_dict({"qids": ["g"], "auxiliary": ["nope"]}))


# ---------------------------------------------------------------------------------------
# value parsing
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(("text", "lo", "hi"), [
    ("[20-29]", 20, 29), ("20-29", 20, 29), ("[20, 30)", 20, 30), ("(20, 30]", 20, 30),
    ("20..29", 20, 29), ("20 to 29", 20, 29), ("[-5--1]", -5, -1), ("60+", 60, np.inf), ("[1.5-2.5]", 1.5, 2.5),
])
def test_cvpl_interval_strings(text, lo, hi):
    cfg = CVPLConfig.from_dict({"qids": ["age"]})
    enc = encode_qid(QIDSpec("age", kind="numeric"), pd.Series([lo]), pd.Series([text]), cfg)
    assert enc.state_r[0] == STATE_VALUE and enc.lo_r[0] == lo and enc.hi_r[0] == hi
    assert enc.applicable and enc.has_interval


def test_cvpl_wildcard_and_unparsable_numeric_values():
    cfg = CVPLConfig.from_dict({"qids": ["age"]})
    enc = encode_qid(QIDSpec("age", kind="numeric"), pd.Series([21, 22]), pd.Series(["*", "young"]), cfg)
    assert enc.state_r[0] == STATE_WILD
    assert enc.applicable is False and enc.limitation == "release_values_not_numeric_or_interval"


def test_cvpl_top1_ties_are_broken_uniformly_in_expectation():
    from .helpers import anchors, measure, release_frame, source_frame
    src = source_frame([("A", 1.0, "s0")])
    rel = release_frame([("A", 1.0, "s0"), ("A", 1.2, "t1")] + anchors())
    r = measure(src, rel)
    # both candidates have cosine exactly 1: a two-way tie
    assert r["cvpl_top1_success_gt"] == pytest.approx(0.5)
    assert r["precision_at_tau"] == pytest.approx(0.5)


# --- conformance with the original CVPL project's own tests (test_toy_dataset.py) ---------
def test_original_cvpl_self_linkage_is_high():
    d = make_df(60, 1)
    cfg = CVPLConfig.from_dict({"qids": [{"name": "age", "bin_width": 10}, "gender", "region_level1"],
                                "auxiliary": ["income"], "tau": 0.9})
    assert run_cvpl(d, d.copy(), cfg)["cvpl_lr"] > 0.5


def test_original_cvpl_threshold_monotonicity():
    d, dp = make_df(60, 1), make_df(60, 2)
    base = {"qids": [{"name": "age", "bin_width": 10}, "gender", "region_level1"], "auxiliary": ["income"]}
    lrs = [run_cvpl(d, dp, CVPLConfig.from_dict({**base, "tau": t}))["cvpl_lr"] for t in (0.3, 0.6, 0.9, 0.99)]
    assert lrs == sorted(lrs, reverse=True)  # a higher threshold never links more


def test_original_cvpl_small_dataset_and_no_ground_truth_do_not_fail():
    d = make_df(5, 1)
    cfg = CVPLConfig.from_dict({"qids": ["gender"], "auxiliary": ["income"], "tau": 0.5})
    r = run_cvpl(d, make_df(5, 2), cfg)
    assert 0 <= r["cvpl_lr"] <= 1


# --- the union headline (non-nested ladders) ----------------------------------------------------
def _union_cfg(rungs, **kw):
    return CVPLConfig.from_dict({
        "qids": [{"name": q, "generalized": False} for q in ("a", "b", "c")],
        "auxiliary": ["n1", "n2"], "tau": 0.5, "blocking_ladder": rungs, "ground_truth": GT, **kw})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cvpl_non_nested_headline_is_the_union_of_candidate_sets_all_measures(seed):
    src, rel = _random_frames(seed)
    rungs = [["a"], ["b"]]
    r = run_cvpl(src, rel, _union_cfg(rungs))
    assert r["ladder"]["non_nested"] is True and r["headline_basis"] == "union_of_rungs"
    want = _oracle(src, rel, ["a", "b", "c"], ["n1", "n2"], [], 0.5, rungs, union=True)[-1]
    # (the oracle features: n1, n2 numeric; qids a, b one-hot - same as the engine's)
    assert r["cvpl_lr"] == pytest.approx(want["lr"], abs=1e-12)
    # identification is evaluated PER RUNG (each rung a declared attacker profile); the
    # headline is the largest value over the rungs, not a value of the union
    per = _oracle(src, rel, ["a", "b", "c"], ["n1", "n2"], [], 0.5, rungs)
    assert r["cvpl_id_uniform_subjects_gt"] == pytest.approx(max(e["uni"] for e in per), abs=1e-9)
    assert r["cvpl_top1_success_gt"] == pytest.approx(max(e["top1"] for e in per), abs=1e-9)
    precisions = [e["prec"] for e in per if e["prec"] is not None]
    if precisions:
        assert r["precision_at_tau"] == pytest.approx(max(precisions), abs=1e-9)
    # the headline lr is the final value of the union trajectory, and never below any rung
    assert r["cvpl_lr"] == r["candidate_union_trajectory"][-1]["cvpl_lr_union"] == r["cvpl_lr_union"]
    by_rung = [e["cvpl_lr"] for e in r["cvpl_lr_by_rung"]]
    assert r["cvpl_lr"] >= max(by_rung) and len(by_rung) == 2
    assert r["headline_rung"] is None


@pytest.mark.parametrize("seed", range(4))
def test_cvpl_nested_ladder_last_rung_equals_the_union_all_measures(seed):
    """Theorem 1's premise: on a relaxing ladder the last rung IS the union - checked by
    forcing the union evaluation and comparing every headline measure."""
    src, rel = _random_frames(seed)
    cfg = _union_cfg([["a", "b", "c"], ["a", "b"], ["a"]])
    last = run_cvpl(src, rel, cfg)
    forced = run_cvpl(src, rel, cfg, force_union=True)
    assert last["headline_basis"] == "last_rung" and forced["headline_basis"] == "union_of_rungs"
    for key in ("cvpl_lr", "n_linked", "n_with_candidates", "n_compared", "cvpl_id_uniform_gt",
                "cvpl_top1_success_gt", "precision_at_tau", "cvpl_weighted_lr_legacy"):
        assert last[key] == pytest.approx(forced[key], abs=1e-12), key
    assert last["candidate_count_summary"] == forced["candidate_count_summary"]


def test_cvpl_tightening_or_mixed_ladders_also_use_the_union():
    src, rel = _random_frames(1)
    r = run_cvpl(src, rel, _union_cfg([["a"], ["a", "b"]]))
    assert r["ladder"]["transitions"] == ["tightening"] and r["headline_basis"] == "union_of_rungs"
    # a tightening rung adds no pairs: the union is the first rung
    assert r["cvpl_lr"] == r["cvpl_lr_by_rung"][0]["cvpl_lr"]


def test_cvpl_union_over_budget_is_not_measured_while_rungs_are_still_reported():
    src, rel = _random_frames(2)
    full = run_cvpl(src, rel, _union_cfg([["a"], ["b"]]))
    each = [e["planned_comparisons"] for e in full["blocking_ladder_trajectory"]]
    limit = max(each) + 1  # every rung fits, their sum does not
    assert sum(each) > limit
    r = run_cvpl(src, rel, _union_cfg([["a"], ["b"]], max_comparisons=limit))
    assert all(e["status"] == "measured" for e in r["blocking_ladder_trajectory"])
    assert r["cvpl_lr"] is None and r["cvpl_lr_status"]["reason"] == "resource_limit"
    assert "union_detail" in r.metrics


def test_cvpl_union_with_a_not_applicable_rung_is_not_applicable():
    src = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "n1": [1.0, 2.0], "n2": [1.0, 2.0], "sid": ["s0", "s1"]})
    rel = pd.DataFrame({"a": ["x", "y"], "b": ["P-general", "Q-general"], "n1": [1.0, 2.0], "n2": [1.0, 2.0], "rid": ["s0", "s1"]})
    cfg = CVPLConfig.from_dict({
        "qids": [{"name": "a", "generalized": False}, "b"], "auxiliary": ["n1", "n2"],
        "blocking_ladder": [["a"], ["b"]], "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    assert r["blocking_ladder_trajectory"][0]["status"] == "measured"
    assert r["blocking_ladder_trajectory"][1]["status"] == "not_applicable"
    assert r["cvpl_lr_status"]["status"] == "not_applicable"
    assert "QID 'b'" in r["cvpl_lr_status"]["reason"]


# --- visibility of bucketing and of fixes -------------------------------------------------------------
def test_cvpl_bucketing_is_reported_never_silent():
    rng = np.random.default_rng(0)
    n = 300
    src = pd.DataFrame({"g": "A", "tag": [f"t{i}" for i in range(n)], "x": rng.normal(size=n)})
    cfg = CVPLConfig.from_dict({"qids": [{"name": "g", "generalized": False}], "auxiliary": ["tag", "x"],
                                "max_categories": 20, "tau": 0.5, "qid_as_features": False})
    r = run_cvpl(src, src.copy(), cfg)
    (b,) = r["bucketed_features"]
    assert b == {"feature": "tag", "distinct_values": 300, "encoded_width": 20,
                 "distinct_values_collapsed": 281, "rows_in_other_bucket": 562}
    (w,) = r["warnings"]
    assert "'tag'" in w and "300" in w and "281" in w and "t1" not in w.replace("tag", "")  # names and counts only
    clean = run_cvpl(src.head(10), src.head(10).copy(), cfg)
    assert clean["bucketed_features"] == [] and clean["warnings"] == []


def test_cvpl_not_applicable_reason_names_the_qid_and_the_fix():
    src = pd.DataFrame({"city": ["Paris", "Lyon"], "x": [1.0, 2.0]})
    rel = pd.DataFrame({"city": ["North", "South"], "x": [1.0, 2.0]})
    r = run_cvpl(src, rel, CVPLConfig.from_dict({"qids": ["city"], "auxiliary": ["x"]}))
    reason = r["cvpl_lr_status"]["reason"]
    assert "QID 'city'" in reason and "hierarchy" in reason and "generalized=False" in reason
    assert r["qid_status"]["city"]["fix"] == reason
