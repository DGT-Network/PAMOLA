"""Pickle is never loaded implicitly (docs/en/core/security_loading_files.md)."""

import os
import pickle

import pandas as pd
import pytest

from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.utils import io as pamola_io
from pamola_core.utils.safe_deserialization import UnsafeDeserializationError


class _Boom:
    """Unpickling this object runs code: it creates a marker file."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _malicious_pickle(path, marker):
    with open(path, "wb") as f:
        pickle.dump(_Boom(marker), f)


def test_mapping_store_refuses_pickle_by_default_and_does_not_execute_it(tmp_path):
    marker = tmp_path / "executed.txt"
    evil = tmp_path / "map.pkl"
    _malicious_pickle(evil, marker)
    store = MappingStore()
    with pytest.raises(UnsafeDeserializationError):
        store.load_pickle(evil)
    with pytest.raises(UnsafeDeserializationError):
        store.load(evil)  # format inferred from .pkl
    with pytest.raises(UnsafeDeserializationError):
        store.load(evil, format="pickle")
    assert not marker.exists(), "the pickle payload must not run"


def test_mapping_store_pickle_needs_the_explicit_flag_and_warns(tmp_path, caplog):
    store = MappingStore()
    store.mappings = {"f": {"a": "b"}}
    store.reverse_mappings = {"f": {"b": "a"}}
    path = tmp_path / "map.pkl"
    store.save_pickle(path)
    other = MappingStore()
    with pytest.warns(RuntimeWarning, match="allow_unsafe_pickle"):
        other.load(path, allow_unsafe_pickle=True)
    assert other.mappings == {"f": {"a": "b"}}


def test_read_dataframe_refuses_pickle_by_default_and_does_not_execute_it(tmp_path):
    marker = tmp_path / "executed.txt"
    evil = tmp_path / "data.pkl"
    _malicious_pickle(evil, marker)
    with pytest.raises(UnsafeDeserializationError):
        pamola_io.read_dataframe(evil)
    assert not marker.exists()


def test_read_dataframe_pickle_with_opt_in_still_works(tmp_path):
    path = tmp_path / "data.pkl"
    pd.DataFrame({"x": [1, 2]}).to_pickle(path)
    with pytest.warns(RuntimeWarning):
        df = pamola_io.read_dataframe(path, allow_unsafe_pickle=True)
    assert len(df) == 2


def test_safe_formats_do_not_need_the_flag(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)
    assert len(pamola_io.read_dataframe(path)) == 2
