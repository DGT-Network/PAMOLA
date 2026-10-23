"""Counterexamples from the independent review of the earlier CVPL draft.

Each test pins how the CORE metric behaves where the BEST-compatible proxy was
shown to mislead, and what the result lets a reader see.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

from .helpers import GT, make_df

_QIDS = [{"name": "age", "bin_width": 10}, "gender", "region_level1"]


def _g(**kw):
    return CVPLConfig.from_dict({"qids": [{"name": "g", "generalized": False}], "auxiliary": ["x"], "tau": 0.9, **kw})


def test_independent_populations_link_but_identify_nobody():
    """G3: two independent 60-row populations. Look-alikes exist (cvpl_lr 0.6); with disjoint
    subject identifiers the ground truth says nobody is the same person: 0, measured."""
    src = make_df(60, 1).rename(columns={"person_id": "sid"})
    rel = make_df(60, 2).rename(columns={"person_id": "rid"})
    rel["rid"] = "other_" + rel["rid"]
    cfg = CVPLConfig.from_dict({"qids": _QIDS, "auxiliary": ["income"], "tau": 0.9, "ground_truth": GT})
    r = run_cvpl(src, rel, cfg)
    assert r["cvpl_lr"] == pytest.approx(0.6)
    assert r["cvpl_id_uniform_gt"] == 0.0 and r["cvpl_id_uniform_gt_status"]["status"] == "measured"
    assert r["cvpl_top1_success_gt"] == 0.0
    assert r["precision_at_tau"] == 0.0


def test_singleton_blocks_are_visible_even_when_cosine_rejects_them():
    """Blocking alone can reveal a bijection. The cosine step may still reject the pair; the
    candidate summary keeps that evidence visible instead of reading as 'no exposure'."""
    src = pd.DataFrame({"g": ["A", "B"], "x": [1.0, 2.0]})
    rel = pd.DataFrame({"g": ["A", "B"], "x": [101.0, 102.0]})
    r = run_cvpl(src, rel, _g())
    assert r["n_with_candidates"] == 2
    assert r["candidate_count_summary"]["min"] == r["candidate_count_summary"]["max"] == 1
    assert r["candidate_count_summary"]["zero_fraction"] == 0.0


def test_joint_fit_depends_on_unrelated_rows_and_source_only_does_not():
    src = pd.DataFrame({"g": ["A", "A"], "x": [1.0, 2.0]})
    rel = pd.DataFrame({"g": ["A", "A"], "x": [3.0, 4.0]})
    extra = pd.concat([rel, pd.DataFrame({"g": ["Z", "Z"], "x": [100.0, 101.0]})], ignore_index=True)
    joint_a, joint_b = run_cvpl(src, rel, _g()), run_cvpl(src, extra, _g())
    assert joint_a["cvpl_lr"] != joint_b["cvpl_lr"]  # unrelated rows moved the representation
    assert joint_a["projection"]["fit_strategy"] == "joint"
    so_a = run_cvpl(src, rel, _g(fit_strategy="source_only"))
    so_b = run_cvpl(src, extra, _g(fit_strategy="source_only"))
    assert so_a["cvpl_lr"] == so_b["cvpl_lr"]  # the explicit mode for comparing releases


def test_duplicate_candidates_change_the_legacy_weight_but_not_exposure():
    """Review example: repeating a released row moves the inverse-candidate-count weighted
    score from 1/2 to 1/3 although the same two records link. The existential rate does not move."""
    src = pd.DataFrame({"g": ["A", "A", "B", "B"], "x": [-1.0, -2.0, 1.0, 2.0]})
    rel = pd.DataFrame({"g": ["A", "B"], "x": [-1.0, -2.0]})
    dup = pd.concat([rel, rel.iloc[:1]], ignore_index=True)
    cfg = _g(qid_as_features=False)
    a, b = run_cvpl(src, rel, cfg), run_cvpl(src, dup, cfg)
    assert a["cvpl_lr"] == b["cvpl_lr"] == 0.5
    assert a["cvpl_weighted_lr_legacy"] == pytest.approx(1 / 2)
    assert b["cvpl_weighted_lr_legacy"] == pytest.approx(1 / 3)


def test_subject_level_probability_ignores_duplicated_rows():
    src = pd.DataFrame({"g": ["A", "A"], "x": [1.0, 1.1], "sid": ["s0", "s1"]})
    rel = pd.DataFrame({"g": ["A", "A", "Z", "Z", "Z", "Z"], "x": [1.0, 1.1, -50.0, -51.0, -52.0, -53.0],
                        "rid": ["s0", "s1", "z0", "z1", "z2", "z3"]})
    dup = pd.concat([rel, rel.iloc[:1]], ignore_index=True)
    a = run_cvpl(src, rel, _g(ground_truth=GT, tau=0.2, qid_as_features=False))
    b = run_cvpl(src, dup, _g(ground_truth=GT, tau=0.2, qid_as_features=False))
    assert a["cvpl_id_uniform_gt"] == pytest.approx(b["cvpl_id_uniform_gt"]) == pytest.approx(0.5)


def test_a_unique_categorical_feature_is_never_dropped_under_any_fit_strategy():
    src = pd.DataFrame({"g": ["A"] * 4, "token": list("abcd"), "x": [1.0, 2.0, 3.0, 4.0]})
    for strategy in ("joint", "source_only"):
        r = run_cvpl(src, src.copy(), _g(auxiliary=["token", "x"], fit_strategy=strategy))
        assert "token" in r["projection"]["feature_columns"]
        assert r["projection"]["encoding"]["token"]["distinct_values"] == 4


def test_the_diagonal_hierarchy_example_links_through_the_declared_rule():
    """originals aged 21, 31, 41, 51 against released [20-29] ... [50-59]: BEST compared nothing."""
    src = pd.DataFrame({"age": [21, 31, 41, 51], "x": [1.0, 1.1, 1.2, 1.3]})
    rel = pd.DataFrame({"age": ["[20-29]", "[30-39]", "[40-49]", "[50-59]", "[90-99]"], "x": [1.0, 1.1, 1.2, 1.3, -50.0]})
    r = run_cvpl(src, rel, CVPLConfig.from_dict({"qids": ["age"], "auxiliary": ["x"], "qid_as_features": False, "tau": 0.5}))
    assert r["cvpl_lr"] == 1.0
    assert r["candidate_count_summary"]["min"] == 1
