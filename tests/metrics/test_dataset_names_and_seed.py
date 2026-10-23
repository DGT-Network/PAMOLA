"""ASB-OPS-6: a metric never compares a dataset with itself by default, and the sample is chosen by the operation seed.

Before: both dataset names defaulted to "main" (DCR 0, KS 0 - a silent self-comparison), and
``_sample_aligned`` always used ``random_state=42`` although ``seed`` is part of the cache key.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frame(n=200, seed=0):
    return pd.DataFrame({"x": np.random.default_rng(seed).normal(size=n)})


def test_the_same_dataset_twice_is_a_typed_error_by_default(tmp_path):
    result = FidelityOperation(fidelity_metrics=["ks"], use_cache=False).execute(
        DataSource(dataframes={"main": _frame()}), tmp_path, None
    )
    assert result.status == OperationStatus.ERROR
    assert "same dataset" in result.error_message and "'main'" in result.error_message


def test_explicit_names_work(tmp_path):
    result = FidelityOperation(fidelity_metrics=["ks"], use_cache=False).execute(
        DataSource(dataframes={"a": _frame(), "b": _frame(seed=1)}), tmp_path, None,
        original_dataset_name="a", transformed_dataset_name="b",
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message


def test_one_dataset_is_allowed_when_a_column_mapping_compares_different_columns(tmp_path):
    frame = pd.DataFrame({"x": _frame()["x"], "x_enriched": _frame(seed=2)["x"]})
    result = FidelityOperation(
        fidelity_metrics=["ks"], use_cache=False, columns=["x"], column_mapping={"x": "x_enriched"}
    ).execute(DataSource(dataframes={"main": frame}), tmp_path, None)
    assert result.status == OperationStatus.SUCCESS, result.error_message


def _sample(seed, tmp_path_factory, name):
    op = FidelityOperation(fidelity_metrics=["ks"], use_cache=False, sample_size=50, seed=seed)
    original, transformed = _frame(500, 0), _frame(500, 1)
    original.index = transformed.index = range(500)
    captured = {}
    real = op._sample_aligned

    def spy(*args, **kwargs):
        captured["random_state"] = kwargs.get("random_state")
        sampled = real(*args, **kwargs)
        captured["rows"] = sampled[0]["x"].round(6).tolist()
        return sampled

    op._sample_aligned = spy
    result = op.execute(
        DataSource(dataframes={"o": original, "t": transformed}), tmp_path_factory.mktemp(name), None,
        original_dataset_name="o", transformed_dataset_name="t",
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    return captured


def test_the_operation_seed_selects_the_sample(tmp_path_factory):
    one, one_again, other = (_sample(1, tmp_path_factory, "a"), _sample(1, tmp_path_factory, "b"), _sample(2, tmp_path_factory, "c"))
    assert one["random_state"] == 1 and other["random_state"] == 2
    assert one["rows"] == one_again["rows"] and one["rows"] != other["rows"]


def test_without_a_seed_the_sample_stays_what_it_was(tmp_path_factory):
    assert _sample(None, tmp_path_factory, "d")["random_state"] == 42
