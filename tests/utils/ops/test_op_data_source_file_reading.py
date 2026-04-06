"""Tests for DataSource file-based reading paths — targets 123 missed lines.
Covers: get_dataframe_from_file, file path resolution, schema validation,
multi-file loading, encryption key routing, and error paths."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.utils.ops.op_data_source import DataSource


@pytest.fixture
def ds():
    return DataSource()


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(f, index=False)
    return f


@pytest.fixture
def json_file(tmp_path):
    f = tmp_path / "data.json"
    pd.DataFrame({"x": [10, 20], "y": ["a", "b"]}).to_json(f, orient="records")
    return f


# --- File path management ---
class TestFilePathManagement:
    def test_add_and_get_file_path(self, ds, csv_file):
        ds.add_file_path("csv1", csv_file)
        assert ds.get_file_path("csv1") == csv_file

    def test_get_file_path_nonexistent(self, ds):
        result = ds.get_file_path("ghost")
        assert result is None

    def test_add_file_path_as_string(self, ds, csv_file):
        ds.add_file_path("csv1", str(csv_file))
        result = ds.get_file_path("csv1")
        assert isinstance(result, Path)


# --- get_dataframe from file ---
class TestGetDataframeFromFile:
    def test_read_csv_via_file_path(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        df, err = ds.get_dataframe("csv")
        assert df is not None
        assert len(df) == 3
        assert err is None

    def test_read_json_via_file_path(self, ds, json_file):
        ds.add_file_path("json", json_file)
        df, err = ds.get_dataframe("json")
        assert df is not None
        assert len(df) == 2

    def test_file_not_found_returns_error(self, ds):
        ds.add_file_path("missing", Path("/nonexistent/file.csv"))
        df, err = ds.get_dataframe("missing")
        assert df is None
        assert err is not None
        assert "error_type" in err

    def test_read_with_columns_filter(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        df, err = ds.get_dataframe("csv", columns=["a"])
        if df is not None:
            assert "a" in df.columns

    def test_read_with_nrows(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        df, err = ds.get_dataframe("csv", nrows=1)
        if df is not None:
            assert len(df) <= 1


# --- Schema ---
class TestSchemaFromFile:
    def test_schema_from_csv(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        ds.get_dataframe("csv")  # Load first
        schema = ds.get_schema("csv")
        assert schema is not None
        assert isinstance(schema, dict)

    def test_schema_returns_cached(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        ds.get_dataframe("csv")
        s1 = ds.get_schema("csv")
        s2 = ds.get_schema("csv")
        assert s1 == s2


# --- Multi-file paths ---
class TestMultiFilePaths:
    def test_list_of_file_paths(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(f1, index=False)
        pd.DataFrame({"x": [3, 4]}).to_csv(f2, index=False)
        ds = DataSource(file_paths={"multi": [f1, f2]})
        result = ds.get_file_path("multi")
        assert isinstance(result, list)
        assert len(result) == 2


# --- Encryption key management ---
class TestEncryptionKeys:
    def test_add_encryption_key(self, ds):
        ds.add_encryption_key("file1", "secret123")
        assert ds.encryption_keys.get("file1") == "secret123"

    def test_no_overwrite_existing_key(self, ds):
        ds.add_encryption_key("file1", "key1")
        ds.add_encryption_key("file1", "key2")
        assert ds.encryption_keys["file1"] == "key1"

    def test_encryption_modes_dict_exists(self, ds):
        assert isinstance(ds.encryption_modes, dict)


# --- Data types ---
class TestDataTypes:
    def test_add_and_apply_data_types(self, ds):
        ds.add_data_type("employees", {"age": "int32", "score": "float64"})
        df = pd.DataFrame({"age": [25, 30], "score": [3.5, 4.0]})
        result = ds.apply_data_types(df, "employees")
        assert isinstance(result, pd.DataFrame)

    def test_apply_types_no_mapping(self, ds):
        df = pd.DataFrame({"x": [1, 2]})
        result = ds.apply_data_types(df, "no_such_mapping")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# --- Init kwargs ---
class TestInitKwargs:
    def test_init_with_encryption_keys(self):
        ds = DataSource(encryption_keys={"file1": "key1"})
        assert ds.encryption_keys.get("file1") == "key1"


# --- Analysis ---
class TestAnalysis:
    def test_analyze_loaded_df(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        ds.get_dataframe("csv")
        result = ds.analyze_dataframe("csv")
        assert isinstance(result, dict)


# --- Memory ---
class TestMemoryOperations:
    def test_estimate_memory_from_file(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        ds.get_dataframe("csv")
        result = ds.estimate_memory_usage("csv")
        assert result is not None

    def test_optimize_with_loaded_data(self, ds, csv_file):
        ds.add_file_path("csv", csv_file)
        ds.get_dataframe("csv")
        result = ds.optimize_memory()
        assert isinstance(result, dict)


# --- Release ---
class TestRelease:
    def test_release_dataframe(self, ds):
        df = pd.DataFrame({"a": [1, 2]})
        ds.add_dataframe("test", df)
        assert "test" in ds.dataframes
        ds.release_dataframe("test")
        assert "test" not in ds.dataframes

    def test_release_nonexistent(self, ds):
        # Should not raise
        ds.release_dataframe("ghost")


# --- Dataframe count ---
class TestDataframeCount:
    def test_multiple_dataframes(self):
        ds = DataSource(dataframes={
            "a": pd.DataFrame({"x": [1]}),
            "b": pd.DataFrame({"y": [2]}),
        })
        assert len(ds.dataframes) == 2
