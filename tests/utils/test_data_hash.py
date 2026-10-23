"""CDX-AC-6: the data hash used as cache identity covers ALL of the data.

``generate_data_hash`` used to hash 5,000 evenly spaced rows plus column sums, and
fell back to "type + length" on any error. Changing an unsampled row, or the index
labels, left the key unchanged, so a cached score was served for different data.
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.utils import helpers
from pamola_core.utils.helpers import generate_data_hash
from pamola_core.utils.ops.op_data_source import DataSource


def _frame(n=20_000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=n), "s": rng.choice(list("abcd"), n)})


def test_identical_content_has_identical_hash():
    df = _frame()
    assert generate_data_hash(df) == generate_data_hash(df.copy())


def test_a_change_outside_the_old_sample_changes_the_hash():
    df = _frame()
    other = df.copy()
    # 20,000 rows sampled at 5,000 points: row 1 was never looked at, and a swap
    # keeps the column sum/min/max unchanged.
    other.loc[1, "x"], other.loc[2, "x"] = df.loc[2, "x"], df.loc[1, "x"]
    assert generate_data_hash(df) != generate_data_hash(other)


def test_a_changed_string_value_changes_the_hash():
    df = _frame()
    other = df.copy()
    other.loc[3, "s"] = "z"
    assert generate_data_hash(df) != generate_data_hash(other)


def test_reversed_index_labels_change_the_hash():
    df = _frame(100)
    other = df.copy()
    other.index = other.index[::-1]
    assert generate_data_hash(df) != generate_data_hash(other)


def test_row_order_dtype_and_column_names_matter():
    df = _frame(100)
    assert generate_data_hash(df) != generate_data_hash(df.iloc[::-1])
    assert generate_data_hash(df) != generate_data_hash(df.astype({"x": "float32"}))
    assert generate_data_hash(df) != generate_data_hash(df.rename(columns={"x": "y"}))


def test_unhashable_cells_raise_instead_of_a_lossy_fallback():
    from pamola_core.utils.helpers import UnhashableDataError

    with pytest.raises(UnhashableDataError):
        generate_data_hash(pd.DataFrame({"v": [[1, 2], [3]]}))


def test_series_and_dask_are_covered_too():
    s = _frame()["x"]
    t = s.copy()
    t.iloc[1] += 1e-9
    assert generate_data_hash(s) != generate_data_hash(t)

    df = _frame(1000)
    d1 = dd.from_pandas(df, npartitions=3)
    d2 = dd.from_pandas(df.assign(x=df["x"].where(df.index != 1, 0.5)), npartitions=3)
    assert generate_data_hash(d1) != generate_data_hash(d2)


def test_chunk_size_does_not_change_the_hash(monkeypatch):
    df = _frame(5000)
    reference = generate_data_hash(df)
    monkeypatch.setattr(helpers, "_HASH_CHUNK_ROWS", 7)
    assert generate_data_hash(df) == reference


def test_a_hashing_failure_is_not_turned_into_a_weak_key(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("cannot hash")

    monkeypatch.setattr(helpers.pd.util, "hash_pandas_object", boom)
    with pytest.raises(RuntimeError):
        generate_data_hash(_frame(10))


def test_unsupported_type_raises():
    with pytest.raises(Exception, match="Unsupported data type"):
        generate_data_hash([1, 2, 3])


# --- the metric cache built on it --------------------------------------------


def _run(original, transformed, task_dir):
    op = FidelityOperation(fidelity_metrics=["ks"], use_cache=True, columns=["x"])
    return op.execute(
        DataSource(dataframes={"main": original, "synth": transformed}),
        task_dir,
        None,
        original_dataset_name="main",
        transformed_dataset_name="synth",
    )


def test_metric_cache_misses_when_only_unsampled_rows_differ(tmp_path):
    original = _frame()[["x"]]
    transformed = original.copy()
    changed = transformed.copy()
    changed.loc[1, "x"] += 3.0  # a row the 5,000-point sample never reached

    _run(original, transformed, tmp_path)
    second = _run(original, changed, tmp_path)
    assert not second.metrics.get("cached")
    third = _run(original, transformed.copy(), tmp_path)
    assert third.metrics.get("cached") is True


def test_metric_cache_is_disabled_when_the_data_cannot_be_hashed(tmp_path, monkeypatch):
    original = _frame(200)[["x"]]
    monkeypatch.setattr(
        helpers.pd.util,
        "hash_pandas_object",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cannot hash")),
    )
    first = _run(original, original.copy(), tmp_path)
    second = _run(original, original.copy(), tmp_path)

    assert first.status.value != "error" and second.status.value != "error"
    assert not second.metrics.get("cached")
    assert not list((tmp_path / "cache").rglob("*.json"))
