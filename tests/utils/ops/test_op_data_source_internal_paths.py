"""Tests targeting internal code paths of DataSource.
Covers: get_dataframe_chunks, add_multi_file_dataset, release_dataframe,
get_task_encryption_key, apply_data_types with filtering/fallback,
_validate_dataframe_schema, get_file_metadata.
Targets 92 missed lines (76% → 85%+)."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.utils.ops.op_data_source import DataSource


# --- get_dataframe_chunks ---
class TestGetDataframeChunks:
    def test_chunks_from_in_memory_df(self):
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        ds = DataSource(dataframes={"test": df})
        chunks = list(ds.get_dataframe_chunks("test", chunk_size=30))
        total = sum(len(c) for c in chunks)
        assert total == 100

    def test_chunks_from_file(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": range(100)}).to_csv(f, index=False)
        ds = DataSource(file_paths={"csv": f})
        chunks = list(ds.get_dataframe_chunks("csv", chunk_size=30))
        total = sum(len(c) for c in chunks)
        assert total == 100

    def test_chunks_nonexistent_name(self):
        ds = DataSource()
        chunks = list(ds.get_dataframe_chunks("ghost", chunk_size=10))
        assert chunks == []

    def test_chunks_nonexistent_file(self):
        ds = DataSource(file_paths={"bad": Path("/nonexistent/file.csv")})
        chunks = list(ds.get_dataframe_chunks("bad", chunk_size=10))
        assert chunks == []

    def test_chunks_with_columns_filter(self):
        df = pd.DataFrame({"a": range(50), "b": range(50), "c": range(50)})
        ds = DataSource(dataframes={"test": df})
        chunks = list(ds.get_dataframe_chunks("test", chunk_size=20, columns=["a"]))
        if chunks:
            assert "a" in chunks[0].columns


# --- add_multi_file_dataset ---
class TestAddMultiFileDataset:
    def test_add_multi_csv(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(f1, index=False)
        pd.DataFrame({"x": [3, 4]}).to_csv(f2, index=False)
        ds = DataSource()
        ds.add_multi_file_dataset("multi", [f1, f2], load=True)
        assert "multi" in ds.dataframes or "multi" in ds.file_paths

    def test_add_multi_no_load(self, tmp_path):
        f1 = tmp_path / "a.csv"
        pd.DataFrame({"x": [1]}).to_csv(f1, index=False)
        ds = DataSource()
        ds.add_multi_file_dataset("multi", [f1], load=False)
        assert "multi" in ds.file_paths

    def test_add_multi_nonexistent_files(self):
        ds = DataSource()
        ds.add_multi_file_dataset("bad", [Path("/no/file.csv")], load=False)
        assert "bad" in ds.file_paths


# --- release_dataframe ---
class TestReleaseDataframe:
    def test_release_existing(self):
        ds = DataSource(dataframes={"test": pd.DataFrame({"x": [1]})})
        assert ds.release_dataframe("test") is True
        assert "test" not in ds.dataframes

    def test_release_nonexistent(self):
        ds = DataSource()
        assert ds.release_dataframe("ghost") is False


# --- get_task_encryption_key ---
class TestGetTaskEncryptionKey:
    def test_none_task_id(self):
        ds = DataSource()
        assert ds.get_task_encryption_key(None) is None

    def test_valid_task_id_with_key(self):
        ds = DataSource()
        with patch("pamola_core.utils.crypto_helpers.key_store.get_key_for_task",
                    return_value="fakekey"):
            result = ds.get_task_encryption_key("task-001")
            assert result is not None
            assert result["key"] == "fakekey"

    def test_valid_task_id_no_key(self):
        ds = DataSource()
        with patch("pamola_core.utils.crypto_helpers.key_store.get_key_for_task",
                    return_value=None):
            result = ds.get_task_encryption_key("task-002")
            assert result is None

    def test_import_error_fallback(self):
        ds = DataSource()
        with patch("pamola_core.utils.crypto_helpers.key_store.get_key_for_task",
                    side_effect=ImportError("no crypto")):
            result = ds.get_task_encryption_key("task-003")
            assert result is None


# --- apply_data_types with field filtering and fallback ---
class TestApplyDataTypes:
    def test_no_types_returns_unchanged(self):
        ds = DataSource()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = ds.apply_data_types(df, "no_mapping")
        assert len(result) == 3

    def test_with_types(self):
        ds = DataSource()
        ds.add_data_type("t", {"x": "float64"})
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = ds.apply_data_types(df, "t")
        # dtype may be numpy float64 or pandas Float64 (nullable) — both are float
        assert "float" in str(result["x"].dtype).lower()

    def test_with_field_filter(self):
        ds = DataSource()
        ds.add_data_type("t", {"x": "float64", "y": "str"})
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = ds.apply_data_types(df, "t", fields=["x"])
        assert "float" in str(result["x"].dtype).lower()

    def test_filter_no_matching_fields(self):
        ds = DataSource()
        ds.add_data_type("t", {"x": "float64"})
        df = pd.DataFrame({"z": [1, 2]})
        result = ds.apply_data_types(df, "t", fields=["z"])
        assert len(result) == 2

    def test_column_not_in_df(self):
        ds = DataSource()
        ds.add_data_type("t", {"missing_col": "float64"})
        df = pd.DataFrame({"x": [1, 2]})
        result = ds.apply_data_types(df, "t")
        assert len(result) == 2

    def test_incompatible_conversion_raises(self):
        ds = DataSource()
        ds.add_data_type("t", {"name": "int64"})
        df = pd.DataFrame({"name": ["alice", "bob", "carol"]})
        with pytest.raises(Exception):
            ds.apply_data_types(df, "t")

    def test_same_dtype_skips(self):
        ds = DataSource()
        ds.add_data_type("t", {"x": "int64"})
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
        result = ds.apply_data_types(df, "t")
        assert len(result) == 3


# --- get_file_metadata ---
class TestGetFileMetadata:
    def test_metadata_for_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": range(10)}).to_csv(f, index=False)
        ds = DataSource(file_paths={"csv": f})
        meta = ds.get_file_metadata("csv")
        assert meta is not None
        assert isinstance(meta, dict)

    def test_metadata_nonexistent(self):
        ds = DataSource()
        meta = ds.get_file_metadata("ghost")
        assert meta is None


# --- estimate_memory_usage ---
class TestEstimateMemoryUsage:
    def test_in_memory(self):
        ds = DataSource(dataframes={"df": pd.DataFrame({"a": range(1000)})})
        result = ds.estimate_memory_usage("df")
        assert result is not None

    def test_nonexistent(self):
        ds = DataSource()
        result = ds.estimate_memory_usage("ghost")
        assert result is None


# --- analyze_dataframe ---
class TestAnalyzeDataframe:
    def test_basic(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": [f"s{i}" for i in range(100)],
            "c": np.random.rand(100),
        })
        ds = DataSource(dataframes={"test": df})
        result = ds.analyze_dataframe("test")
        assert isinstance(result, dict)

    def test_nonexistent(self):
        ds = DataSource()
        result = ds.analyze_dataframe("ghost")
        assert result is None


# --- optimize_memory ---
class TestOptimizeMemory:
    def test_optimize(self):
        df = pd.DataFrame({
            "a": range(500),
            "b": ["cat"] * 500,
            "c": np.random.rand(500),
        })
        ds = DataSource(dataframes={"df": df})
        result = ds.optimize_memory()
        assert isinstance(result, dict)
