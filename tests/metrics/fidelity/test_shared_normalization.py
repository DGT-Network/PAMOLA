"""CDX-R2-13: fidelity metrics normalize the two compared inputs with ONE transformation.

Before: KS z-scored each input on its own, so a pure shift / rescale ([0..99] vs [1000..1099])
looked identical (statistic 0, "no difference").
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.commons.normalize import normalize_pair_sklearn
from pamola_core.metrics.fidelity.distribution.kl_divergence import KLDivergence
from pamola_core.metrics.fidelity.distribution.ks_test import KolmogorovSmirnovTest


def _shifted():
    base = np.arange(100.0)
    return pd.DataFrame({"x": base}), pd.DataFrame({"x": base + 1000.0})


@pytest.mark.parametrize("normalize", [True, "zscore", "minmax"])
def test_ks_sees_a_shift_with_every_normalization(normalize):
    original, shifted = _shifted()
    stat = KolmogorovSmirnovTest(normalize=normalize).calculate_metric(original, shifted)["ks_statistic"]
    assert stat == pytest.approx(1.0)


def test_ks_identical_inputs_stay_zero():
    original, _ = _shifted()
    assert KolmogorovSmirnovTest(normalize=True).calculate_metric(original, original.copy())["ks_statistic"] == 0


def test_ks_aggregated_path_shares_the_transformation():
    keys = [f"k{i}" for i in range(30)]
    a = pd.DataFrame({"k": keys, "v": np.arange(30.0)})
    b = pd.DataFrame({"k": keys, "v": np.arange(30.0) * 50 + 500})
    plain = KolmogorovSmirnovTest(key_fields=["k"], value_field="v", normalize=False).calculate_metric(a, b)
    shared = KolmogorovSmirnovTest(key_fields=["k"], value_field="v", normalize=True).calculate_metric(a, b)
    assert shared["ks_statistic"] > 0 and plain["ks_statistic"] > 0


@pytest.mark.parametrize("method", ["zscore", "minmax"])
def test_pair_helper_uses_combined_statistics(method):
    first, second = np.arange(10.0), np.arange(10.0) + 100
    n1, n2 = normalize_pair_sklearn(first, second, method)
    assert not np.allclose(n1, n2)
    assert n1.mean() < n2.mean()


def test_pair_helper_probability_is_per_distribution():
    n1, n2 = normalize_pair_sklearn(np.array([1.0, 3.0]), np.array([2.0, 2.0]), "probability")
    assert n1.sum() == pytest.approx(1.0) and n2.sum() == pytest.approx(1.0)


def test_kl_with_zscore_keeps_the_difference_between_distributions():
    a = pd.Series(["a"] * 90 + ["b"] * 10)
    b = pd.Series(["a"] * 10 + ["b"] * 90)
    result = KLDivergence(normalize="zscore").calculate_metric(pd.DataFrame({"x": a}), pd.DataFrame({"x": b}))
    assert result["kl_divergence"] > 0.1
