"""CVPL PCA stability: invariance to row order, and the (near-)degenerate eigenspace diagnostic."""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

FEATURES = ["f0", "f1", "f2", "f3", "f4", "f5"]


def _config(**overrides):
    params = dict(
        qids=[{"name": "g", "generalized": False}],
        auxiliary=FEATURES,
        qid_as_features=False,
        tau=0.9,
        ground_truth={"source_id_column": "sid", "release_id_column": "rid"},
    )
    params.update(overrides)
    return CVPLConfig.from_dict(params)


def _frames(source_values, release_values):
    n, m = len(source_values), len(release_values)
    src = pd.DataFrame(source_values, columns=FEATURES)
    src["g"], src["sid"] = "A", [f"s{i}" for i in range(n)]
    rel = pd.DataFrame(release_values, columns=FEATURES)
    rel["g"], rel["rid"] = "A", [f"s{i}" if i < n else f"x{i}" for i in range(m)]
    return src, rel


def _correlated(n=300, seed=0):
    """Features with a clearly separated spectrum (one strong common factor)."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 3))
    mixing = rng.normal(size=(3, len(FEATURES))) * np.array([[3.0], [1.5], [0.7]])
    values = factors @ mixing + rng.normal(scale=0.3, size=(n, len(FEATURES)))
    noisy = values + rng.normal(scale=0.2, size=values.shape)
    return _frames(values, noisy)


def _orthonormal_source(n=240, seed=0):
    """Source columns exactly orthogonal with equal variance: every eigenvalue is 1."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, len(FEATURES))))
    q = q - q.mean(axis=0)
    q, _ = np.linalg.qr(q)
    values = q * np.sqrt(n - 1)
    release = values + rng.normal(scale=0.05, size=values.shape)
    return _frames(values, release)


def test_a_separated_spectrum_is_not_flagged():
    src, rel = _correlated()
    r = run_cvpl(src, rel, _config(variance_retained=0.8, min_components=1))
    proj = r["projection"]
    assert proj["degenerate_eigenspace"] is False
    assert proj["cutoff_in_degenerate_eigenspace"] is False
    assert proj["min_relative_eigengap"] > 1e-3
    assert not any("degenerate" in w for w in r["warnings"])


def test_results_are_invariant_to_row_permutation_when_the_spectrum_is_separated():
    src, rel = _correlated()
    cfg = _config(variance_retained=0.8, min_components=1)
    base = run_cvpl(src, rel, cfg)
    rng = np.random.default_rng(42)
    shuffled = run_cvpl(
        src.iloc[rng.permutation(len(src))], rel.iloc[rng.permutation(len(rel))], cfg
    )
    for key in ("cvpl_lr", "n_linked", "n_with_candidates", "cvpl_id_uniform_gt",
                "cvpl_top1_success_gt", "precision_at_tau"):
        assert shuffled[key] == pytest.approx(base[key], abs=1e-12), key
    assert shuffled["projection"]["n_components"] == base["projection"]["n_components"]
    np.testing.assert_allclose(
        np.abs(shuffled.projection.components), np.abs(base.projection.components), atol=1e-8
    )


def test_a_cutoff_inside_a_degenerate_eigenspace_is_flagged_with_a_warning():
    # all six eigenvalues equal; keeping 50% of the variance keeps 3 of the 6 tied directions
    src, rel = _orthonormal_source()
    r = run_cvpl(
        src, rel, _config(fit_strategy="source_only", variance_retained=0.5, min_components=1)
    )
    proj = r["projection"]
    assert proj["n_components"] == 3
    assert proj["degenerate_eigenspace"] is True
    assert proj["cutoff_in_degenerate_eigenspace"] is True
    assert proj["min_relative_eigengap"] < 1e-6
    assert any("degenerate" in w for w in r["warnings"])


def test_a_degenerate_eigenspace_wholly_inside_the_retained_part_is_flagged_but_harmless():
    src, rel = _orthonormal_source()
    r = run_cvpl(
        src, rel, _config(fit_strategy="source_only", variance_retained=1.0, min_components=1,
                          max_components=6)
    )
    proj = r["projection"]
    assert proj["n_components"] == 6
    assert proj["degenerate_eigenspace"] is True  # tied eigenvalues among the retained ones
    assert proj["cutoff_in_degenerate_eigenspace"] is False  # nothing is dropped: no ambiguity
    assert not any("degenerate" in w for w in r["warnings"])


def test_rotating_inside_a_fully_retained_eigenspace_leaves_the_result_unchanged():
    src, rel = _orthonormal_source()
    cfg = _config(fit_strategy="source_only", variance_retained=1.0, min_components=1,
                  max_components=6)
    base = run_cvpl(src, rel, cfg)
    rng = np.random.default_rng(7)
    rot, _ = np.linalg.qr(rng.normal(size=(6, 6)))  # an orthogonal map of the feature axes
    rotated_src, rotated_rel = src.copy(), rel.copy()
    rotated_src[FEATURES] = src[FEATURES].values @ rot
    rotated_rel[FEATURES] = rel[FEATURES].values @ rot
    other = run_cvpl(rotated_src, rotated_rel, cfg)
    # phi standardizes each rotated column, but a rotation of an isotropic cloud stays isotropic
    for key in ("cvpl_lr", "n_linked"):
        assert other[key] == pytest.approx(base[key], abs=0.02), key


def test_the_result_states_the_evaluation_model_and_the_experimental_status():
    src, rel = _correlated(n=60)
    r = run_cvpl(src, rel, _config(variance_retained=0.8, min_components=1))
    assert r["evaluation_model"] == "white_box_maximum_over_declared_profiles_and_fixed_strategies"
    assert r["maturity"] == "experimental"
    text = r["interpretation"]
    assert "maximum over the declared attacker profiles" in text
    assert "not a bound for an arbitrary attacker" in text


def test_r1_03_a_cutoff_inside_the_numerical_null_eigenspace_is_flagged():
    from pamola_core.metrics.privacy.cvpl.projection import _eigengap_diagnostics

    # the eigenvalues of a rank-one standardized covariance as the solver returns them
    residues = np.array([6.0845, 3.44e-16, 1.35e-47, 0.0, 0.0, 0.0])
    degenerate, cutoff, gap = _eigengap_diagnostics(residues, 2)
    assert (degenerate, cutoff, gap) == (True, True, 0.0)
    # keeping only the positive component cuts between a resolved eigenvalue and the null
    # cluster: not ambiguous
    degenerate, cutoff, _ = _eigengap_diagnostics(residues, 1)
    assert cutoff is False


def test_r1_03_codex_rank_one_source_warns():
    rng = np.random.default_rng(918)
    n = 72
    t = np.linspace(-2, 2, n)
    source_values = np.tile(t[:, None], (1, len(FEATURES)))  # six identical columns: rank one
    release_values = source_values + rng.normal(0, 0.8, source_values.shape)
    src, rel = _frames(source_values, release_values)
    r = run_cvpl(
        src, rel, _config(fit_strategy="source_only", tau=0.96, min_components=2, max_components=2)
    )
    assert r["projection"]["cutoff_in_degenerate_eigenspace"] is True
    assert any("degenerate" in w for w in r["warnings"])
