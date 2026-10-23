"""
Regressions for the adversarial review of the merged A-C work (CDX-AC-3/4/5/7).

Each test reproduces the defect as described in the review.
"""

import json
import logging

import pandas as pd
import pytest

from pamola_core.anonymization.commons import processing_utils as anon_pu
from pamola_core.anonymization.commons.mapping_storage import MappingStorage
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.fake_data.commons import processing_utils as fake_pu
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

KEY = "ab" * 32


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _identity(chunk, **_kwargs):
    return chunk


def _mapping_op(path, **kwargs):
    return ConsistentMappingPseudonymizationOperation(
        field_name="email",
        mapping_encryption_key=KEY,
        mapping_file=str(path),
        **kwargs,
    )


# --- CDX-AC-5: index preserved by chunked / parallel processing -------------


@pytest.mark.parametrize("module", [anon_pu, fake_pu], ids=["anonymization", "fake_data"])
def test_chunk_processing_keeps_caller_index(module):
    df = pd.DataFrame({"v": [1, 2, 3]}, index=[100, 200, 300])
    out, ok = module.process_dataframe_using_chunk(df, _identity, chunk_size=2)
    assert ok
    assert list(out.index) == [100, 200, 300]


@pytest.mark.parametrize("module", [anon_pu, fake_pu], ids=["anonymization", "fake_data"])
def test_parallel_processing_keeps_caller_index(module):
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5]}, index=[100, 200, 300, 400, 500])
    out, ok = module.process_dataframe_using_joblib(
        df, _identity, chunk_size=2, parallel_processes=2
    )
    assert ok
    assert list(out.index) == [100, 200, 300, 400, 500]


def test_operation_pipeline_keeps_index_through_chunks(tmp_path):
    op = _mapping_op(tmp_path / "m.enc")
    op.chunk_size = 2
    op._initialize_mapping((tmp_path / "t").mkdir() or tmp_path / "t")
    df = pd.DataFrame({"email": list("abcde")}, index=[100, 200, 300, 400, 500])
    out = op._process_data_with_config(df.copy())
    assert list(out.index) == [100, 200, 300, 400, 500]


# --- CDX-AC-4: duplicate index labels ---------------------------------------


def test_duplicate_index_labels_get_their_own_pseudonyms(tmp_path):
    op = _mapping_op(tmp_path / "m.enc")
    (tmp_path / "t").mkdir()
    op._initialize_mapping(tmp_path / "t")
    batch = pd.DataFrame({"email": ["alice", "bob", None]}, index=[7, 7, 7])

    out = op.process_batch(batch)

    values = out["email"].tolist()
    assert values[0] != values[1]
    assert pd.isna(values[2])
    assert values[0] == op._mapping["alice"] and values[1] == op._mapping["bob"]
    assert list(out.index) == [7, 7, 7]


def test_duplicate_index_labels_across_chunks(tmp_path):
    op = _mapping_op(tmp_path / "m.enc")
    op.chunk_size = 2
    (tmp_path / "t").mkdir()
    op._initialize_mapping(tmp_path / "t")
    names = ["alice", "bob", "alice", None, "carol", "bob"]
    df = pd.DataFrame({"email": names}, index=[1, 1, 1, 2, 2, 2])

    out = op._process_data_with_config(df.copy())

    mapping = op._mapping
    expected = [mapping.get(n) if n is not None else None for n in names]
    assert [None if pd.isna(v) else v for v in out["email"]] == expected
    assert len(set(mapping.values())) == 3
    assert list(out.index) == [1, 1, 1, 2, 2, 2]


def test_mapping_is_reversible_for_each_row_with_duplicate_labels(tmp_path):
    op = _mapping_op(tmp_path / "m.enc")
    (tmp_path / "t").mkdir()
    op._initialize_mapping(tmp_path / "t")
    batch = pd.DataFrame({"email": ["alice", "bob", "carol"]}, index=[0, 0, 0])
    out = op.process_batch(batch)
    assert [op.get_reverse_mapping(p) for p in out["email"]] == ["alice", "bob", "carol"]


# --- CDX-AC-3: process-parallel mapping allocation --------------------------


def test_process_parallel_execution_is_rejected_not_silently_wrong(tmp_path):
    path = tmp_path / "m.enc"
    op = _mapping_op(path, use_vectorization=True, parallel_processes=2, chunk_size=3)
    df = pd.DataFrame({"email": [f"user{i}" for i in range(12)]})

    result = op.execute(DataSource(dataframes={"main": df}), tmp_path / "a", _Reporter())

    assert result.status == OperationStatus.ERROR
    assert "parallel_processes" in result.error_message
    assert "single process" in result.error_message
    assert not result.get_artifacts_by_type("csv")


def test_single_process_vectorized_mapping_is_reversible_across_instances(tmp_path):
    path = tmp_path / "m.enc"
    values = [f"user{i}" for i in range(12)]
    df = pd.DataFrame({"email": values})
    op = _mapping_op(path, use_vectorization=True, parallel_processes=1, chunk_size=3)
    result = op.execute(DataSource(dataframes={"main": df}), tmp_path / "a", _Reporter())
    assert result.status == OperationStatus.SUCCESS, result.error_message
    out = pd.read_csv(result.get_artifacts_by_type("csv")[0].path)["email"].tolist()

    assert len(set(out)) == 12
    stored = MappingStorage(path, bytes.fromhex(KEY)).load()
    assert {stored[v] for v in values} == set(out)
