"""DCR-01..03: bounded backends, deterministic selection, typed size limit.

The exact DCR search used to build the whole n_synthetic x n_original matrix
(~80 GB at 100K x 100K, ~7 TiB at 1M x 1M). These tests pin that

* the block-wise exact search returns the values of the dense matrix,
* FAISS is only selected where it returns the same values (and where it does not,
  the difference is recorded rather than hidden),
* backend selection depends only on configuration and size,
* an input beyond the limit raises ``ResourceError`` before any distance is
  computed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

from pamola_core.errors.exceptions import ResourceError
from pamola_core.metrics.operations.privacy_ops import PrivacyMetricOperation
from pamola_core.metrics.privacy import distance as distance_module
from pamola_core.metrics.privacy.distance import (
    _AUTO_FAISS_MIN_COMPARISONS,
    _FAISS_EXACT,
    DistanceToClosestRecord,
)

faiss_required = pytest.mark.skipif(
    distance_module.faiss is None, reason="faiss-cpu is not installed"
)


def _arrays(n_orig=600, n_trans=400, d=6, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_orig, d)), rng.normal(size=(n_trans, d)) + 0.2


def _dense_reference(orig, trans, metric, aggregation, k_neighbors=3):
    """The pre-change implementation: one full matrix, then aggregate."""
    distances = pairwise_distances(trans, orig, metric=metric)
    if aggregation == "min":
        return np.min(distances, axis=1)
    if aggregation == "mean_k":
        k = min(k_neighbors, distances.shape[1])
        return np.mean(np.sort(distances, axis=1)[:, :k], axis=1)
    return np.percentile(distances, 10, axis=1)


@pytest.mark.parametrize("aggregation", ["min", "mean_k", "percentile"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_blockwise_exact_search_equals_the_dense_matrix(
    metric, aggregation, monkeypatch
):
    orig, trans = _arrays()
    # A tiny budget forces many row blocks; the answer must not depend on it.
    monkeypatch.setattr(distance_module, "_WORKING_MEMORY_MIB", 0.02)
    calls = []
    real_chunked = distance_module.pairwise_distances_chunked

    def counting(*args, **kwargs):
        for block in real_chunked(*args, **kwargs):
            calls.append(len(block))
            yield block

    monkeypatch.setattr(distance_module, "pairwise_distances_chunked", counting)

    dcr = DistanceToClosestRecord(
        distance_metric=metric, aggregation=aggregation, use_faiss=False
    )
    got = dcr._calculate_dcr_sklearn(orig, trans)

    assert len(calls) > 1, "the test must exercise more than one row block"
    assert got.shape == (len(trans),)
    np.testing.assert_allclose(
        got, _dense_reference(orig, trans, metric, aggregation), rtol=0, atol=1e-12
    )


def test_blockwise_mean_k_with_k_at_least_the_number_of_records():
    orig, trans = _arrays(n_orig=4)
    dcr = DistanceToClosestRecord(
        aggregation="mean_k", k_neighbors=10, use_faiss=False
    )
    np.testing.assert_allclose(
        dcr._calculate_dcr_sklearn(orig, trans),
        _dense_reference(orig, trans, "euclidean", "mean_k", k_neighbors=10),
        atol=1e-12,
    )


@faiss_required
@pytest.mark.parametrize(("metric", "aggregation"), sorted(_FAISS_EXACT))
def test_faiss_equals_the_exact_search_where_it_is_selected(metric, aggregation):
    orig, trans = _arrays()
    dcr = DistanceToClosestRecord(
        distance_metric=metric, aggregation=aggregation, k_neighbors=3
    )
    exact = dcr._calculate_dcr_sklearn(orig, trans)
    approx = dcr._calculate_dcr_faiss(orig, trans)

    # FAISS works in float32; same neighbours, same values to that precision.
    np.testing.assert_allclose(approx, exact, rtol=1e-4, atol=1e-5)


@faiss_required
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_faiss_percentile_falls_back_to_the_exact_search(metric):
    """FAISS sees only the k <= 10 nearest neighbours, so a percentile of all distances is
    computed by the exact search instead - and the result says so."""
    assert (metric, "percentile") not in _FAISS_EXACT
    orig, trans = _arrays()
    dcr = DistanceToClosestRecord(distance_metric=metric, aggregation="percentile")
    exact = dcr._calculate_dcr_sklearn(orig, trans)
    other = dcr._calculate_dcr_faiss(orig, trans)
    np.testing.assert_allclose(other, exact)
    assert dcr._backend_fallback["from"] == "faiss"


@pytest.mark.parametrize(
    ("kwargs", "n_orig", "n_trans", "faiss_present", "expected"),
    [
        # small input: exact search
        ({}, 1_000, 1_000, True, "sklearn"),
        # large and FAISS-equivalent: FAISS
        ({}, 100_000, 100_000, True, "faiss"),
        ({"distance_metric": "cosine"}, 100_000, 100_000, True, "faiss"),
        (
            {"distance_metric": "cosine", "aggregation": "mean_k"},
            100_000,
            100_000,
            True,
            "faiss",
        ),
        # large but FAISS would change the numbers: bounded exact search
        ({"aggregation": "mean_k"}, 100_000, 100_000, True, "faiss"),
        ({"aggregation": "percentile"}, 100_000, 100_000, True, "sklearn"),
        ({"distance_metric": "manhattan"}, 100_000, 100_000, True, "sklearn"),
        # explicit choices win
        ({"use_faiss": False}, 100_000, 100_000, True, "sklearn"),
        ({"use_faiss": True}, 10, 10, True, "faiss"),
        # FAISS missing: never "faiss"
        ({}, 100_000, 100_000, False, "sklearn"),
        ({"use_faiss": True}, 100_000, 100_000, False, "sklearn"),
    ],
)
def test_backend_selection_is_a_function_of_configuration_and_size(
    kwargs, n_orig, n_trans, faiss_present, expected, monkeypatch
):
    if faiss_present:
        monkeypatch.setattr(distance_module, "faiss", object())
    else:
        monkeypatch.setattr(distance_module, "faiss", None)
    dcr = DistanceToClosestRecord(**kwargs)
    assert dcr._select_backend(n_orig, n_trans) == expected
    assert dcr._select_backend(n_orig, n_trans) == expected  # deterministic


def test_the_automatic_threshold_is_the_documented_one():
    assert _AUTO_FAISS_MIN_COMPARISONS == 10_000 * 10_000


# --- DCR-03: the guard --------------------------------------------------------


def test_size_limit_message_is_actionable_for_a_million_records():
    dcr = DistanceToClosestRecord()
    with pytest.raises(ResourceError) as excinfo:
        dcr._check_size_limit(1_000_000, 1_000_000, 20, "sklearn")
    message = str(excinfo.value)
    assert "1,000,000,000,000" in message
    assert "sample" in message
    assert "max_comparisons" in message


def test_size_limit_is_checked_before_any_distance_is_computed(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("a distance backend ran despite the size limit")

    monkeypatch.setattr(distance_module, "pairwise_distances_chunked", explode)
    if distance_module.faiss is not None:
        monkeypatch.setattr(DistanceToClosestRecord, "_calculate_dcr_faiss", explode)
    orig = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=list("abc"))
    trans = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 3)), columns=list("abc"))

    with pytest.raises(ResourceError):
        DistanceToClosestRecord(max_comparisons=100).calculate_metric(orig, trans)


def test_limit_applies_to_every_backend_choice(monkeypatch):
    orig = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)), columns=list("ab"))
    trans = orig.copy()
    for use_faiss in (None, True, False):
        with pytest.raises(ResourceError):
            DistanceToClosestRecord(use_faiss=use_faiss, max_comparisons=10).calculate_metric(
                orig, trans
            )


def test_default_limits_leave_100k_and_refuse_1m_for_the_exact_search():
    dcr = DistanceToClosestRecord()
    dcr._check_size_limit(100_000, 100_000, 10, "sklearn")  # the 100K target
    dcr._check_size_limit(1_000_000, 1_000_000, 10, "faiss")  # bounded and fast
    with pytest.raises(ResourceError):
        dcr._check_size_limit(1_000_000, 1_000_000, 10, "sklearn")


def test_limit_can_be_raised():
    orig = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)), columns=list("ab"))
    result = DistanceToClosestRecord(max_comparisons=10**6).calculate_metric(orig, orig.copy())
    assert result["dcr_statistics"]["min"] == pytest.approx(0.0, abs=1e-6)


def test_privacy_operation_surfaces_the_typed_limit_error():
    """The operation must not hide ResourceError behind a generic ValidationError."""
    orig = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)), columns=list("ab"))
    op = PrivacyMetricOperation(
        privacy_metrics=["dcr"], metric_params={"dcr": {"max_comparisons": 10}}
    )
    with pytest.raises(ResourceError):
        op.calculate_metrics(
            orig, orig.copy(), privacy_metrics=["dcr"], metric_params=op.metric_params
        )


def test_default_run_is_unchanged_for_a_small_input():
    """Same numbers as the dense implementation for the default configuration."""
    rng = np.random.default_rng(5)
    orig = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    trans = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    result = DistanceToClosestRecord().calculate_metric(orig, trans)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(orig.to_numpy())
    reference = _dense_reference(
        scaler.transform(orig.to_numpy()),
        scaler.transform(trans.to_numpy()),
        "euclidean",
        "min",
    )
    assert result["dcr_statistics"]["mean"] == pytest.approx(float(reference.mean()))
    assert result["dcr_statistics"]["p5"] == pytest.approx(float(np.percentile(reference, 5)))
