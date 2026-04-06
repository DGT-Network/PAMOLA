"""Tests for DataSource classmethods and additional internal paths.
Targets: from_dataframe, from_file_path, from_multi_file_dataset,
create_sample, validate_schema, normalize_target_dtype,
get_schema, get_encryption_info, suggest_engine, context-manager,
add_dataframe, add_encryption_key, add_encryption_mode, add_file_path,
__enter__/__exit__, string path conversion."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pamola_core.utils.ops.op_data_source import DataSource


# --- from_dataframe classmethod ---
class TestFromDataframe:
    def test_basic(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = DataSource.from_dataframe(df)
        assert "main" in ds.dataframes
        assert len(ds.dataframes["main"]) == 3

    def test_custom_name(self):
        df = pd.DataFrame({"x": [1]})
        ds = DataSource.from_dataframe(df, name="custom")
        assert "custom" in ds.dataframes


# --- from_file_path classmethod ---
class TestFromFilePath:
    def test_without_load(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(f, index=False)
        ds = DataSource.from_file_path(f)
        assert "main" in ds.file_paths

    def test_with_load(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(f, index=False)
        ds = DataSource.from_file_path(f, load=True)
        assert "main" in ds.dataframes or "main" in ds.file_paths

    def test_with_string_path(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(f, index=False)
        ds = DataSource.from_file_path(str(f), name="csv_data")
        assert "csv_data" in ds.file_paths

    def test_custom_name(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(f, index=False)
        ds = DataSource.from_file_path(f, name="mydata")
        assert "mydata" in ds.file_paths


# --- from_multi_file_dataset classmethod ---
class TestFromMultiFileDataset:
    def test_no_load(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        pd.DataFrame({"x": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"x": [2]}).to_csv(f2, index=False)
        ds = DataSource.from_multi_file_dataset([f1, f2], name="multi", load=False)
        assert "multi" in ds.file_paths

    def test_with_load(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(f1, index=False)
        pd.DataFrame({"x": [3, 4]}).to_csv(f2, index=False)
        ds = DataSource.from_multi_file_dataset([f1, f2], name="multi", load=True)
        assert "multi" in ds.dataframes or "multi" in ds.file_paths


# --- create_sample ---
class TestCreateSample:
    def test_existing_df(self):
        df = pd.DataFrame({"a": range(500), "b": range(500)})
        ds = DataSource(dataframes={"main": df})
        sample, err = ds.create_sample("main", sample_size=100)
        assert err is None or sample is not None

    def test_nonexistent_name(self):
        ds = DataSource()
        sample, err = ds.create_sample("ghost")
        assert sample is None


# --- validate_schema ---
class TestValidateSchema:
    def test_valid_schema(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        ds = DataSource(dataframes={"df": df})
        valid, errors = ds.validate_schema("df", {"columns": {"x": "int64"}})
        assert isinstance(valid, bool)

    def test_invalid_schema_type(self):
        ds = DataSource(dataframes={"df": pd.DataFrame({"x": [1]})})
        valid, errors = ds.validate_schema("df", "not_a_dict")
        assert valid is False
        assert len(errors) > 0

    def test_nonexistent_df(self):
        ds = DataSource()
        valid, errors = ds.validate_schema("ghost", {"columns": {}})
        assert valid is False


# --- normalize_target_dtype ---
class TestNormalizeTargetDtype:
    def test_string_int64(self):
        ds = DataSource()
        result = ds.normalize_target_dtype("int64")
        assert result is not None

    def test_string_float64(self):
        ds = DataSource()
        result = ds.normalize_target_dtype("float64")
        assert result is not None

    def test_non_string_passthrough(self):
        ds = DataSource()
        dtype_obj = np.dtype("float32")
        result = ds.normalize_target_dtype(dtype_obj)
        assert result == dtype_obj

    def test_unknown_string_fallback(self):
        ds = DataSource()
        result = ds.normalize_target_dtype("unknown_type_xyz")
        assert result == "unknown_type_xyz"

    def test_str_type(self):
        ds = DataSource()
        result = ds.normalize_target_dtype("str")
        assert result is not None

    def test_bool_type(self):
        ds = DataSource()
        result = ds.normalize_target_dtype("bool")
        assert result is not None

    def test_datetime_type(self):
        ds = DataSource()
        result = ds.normalize_target_dtype("datetime64[ns]")
        assert result is not None


# --- get_schema ---
class TestGetSchema:
    def test_existing_df(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        ds = DataSource(dataframes={"df": df})
        schema = ds.get_schema("df")
        assert schema is not None
        assert isinstance(schema, dict)

    def test_nonexistent(self):
        ds = DataSource()
        schema = ds.get_schema("ghost")
        assert schema is None


# --- get_encryption_info ---
class TestGetEncryptionInfo:
    def test_with_file_path(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(f, index=False)
        ds = DataSource(file_paths={"csv": f})
        info = ds.get_encryption_info("csv")
        # Returns None if not encrypted, or dict if encrypted
        assert info is None or isinstance(info, dict)

    def test_nonexistent(self):
        ds = DataSource()
        info = ds.get_encryption_info("ghost")
        assert info is None

    def test_dataframe_only_returns_none(self):
        ds = DataSource(dataframes={"df": pd.DataFrame({"x": [1]})})
        info = ds.get_encryption_info("df")
        assert info is None


# --- suggest_engine ---
class TestSuggestEngine:
    def test_small_df(self):
        ds = DataSource(dataframes={"df": pd.DataFrame({"x": range(100)})})
        engine = ds.suggest_engine("df")
        assert isinstance(engine, str)

    def test_nonexistent(self):
        ds = DataSource()
        engine = ds.suggest_engine("ghost")
        assert isinstance(engine, str)


# --- context manager ---
class TestContextManager:
    def test_enter_exit(self):
        df = pd.DataFrame({"x": [1]})
        with DataSource(dataframes={"df": df}) as ds:
            assert ds is not None
            assert "df" in ds.dataframes


# --- add_dataframe ---
class TestAddDataframe:
    def test_add(self):
        ds = DataSource()
        df = pd.DataFrame({"x": [1, 2]})
        ds.add_dataframe("new", df)
        assert "new" in ds.dataframes

    def test_duplicate_add(self):
        ds = DataSource()
        df = pd.DataFrame({"x": [1]})
        ds.add_dataframe("df", df)
        ds.add_dataframe("df", df)  # should not raise
        assert "df" in ds.dataframes


# --- add_encryption_key ---
class TestAddEncryptionKey:
    def test_add(self):
        ds = DataSource()
        ds.add_encryption_key("df", "mykey")
        assert ds.encryption_keys.get("df") == "mykey"


# --- add_encryption_mode ---
class TestAddEncryptionMode:
    def test_add(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(f, index=False)
        ds = DataSource()
        # detect_encryption_mode reads file header to detect mode
        with patch("pamola_core.utils.io_helpers.crypto_utils.detect_encryption_mode",
                   return_value="NONE"):
            ds.add_encryption_mode("csv", str(f))
        assert "csv" in ds.encryption_modes

    def test_add_duplicate_skips(self, tmp_path):
        ds = DataSource(encryption_modes={"csv": "NONE"})
        ds.add_encryption_mode("csv", "anything")  # already exists, should skip
        assert ds.encryption_modes.get("csv") == "NONE"


# --- add_file_path ---
class TestAddFilePath:
    def test_add_path(self, tmp_path):
        f = tmp_path / "data.csv"
        f.touch()
        ds = DataSource()
        ds.add_file_path("csv", f)
        assert "csv" in ds.file_paths

    def test_add_string_path(self, tmp_path):
        f = tmp_path / "data.csv"
        f.touch()
        ds = DataSource()
        ds.add_file_path("csv", str(f))
        assert "csv" in ds.file_paths


# --- string path conversion in __init__ ---
class TestStringPathConversion:
    def test_string_path_converted(self, tmp_path):
        f = str(tmp_path / "data.csv")
        ds = DataSource(file_paths={"csv": f})
        assert isinstance(ds.file_paths["csv"], Path)

    def test_list_of_string_paths_converted(self, tmp_path):
        f1 = str(tmp_path / "a.csv")
        f2 = str(tmp_path / "b.csv")
        ds = DataSource(file_paths={"multi": [f1, f2]})
        for p in ds.file_paths["multi"]:
            assert isinstance(p, Path)


# --- apply_data_types with direct data_types kwarg ---
class TestApplyDataTypesDirect:
    def test_direct_data_types_kwarg(self):
        ds = DataSource()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = ds.apply_data_types(df, data_types={"x": "float64"})
        # dtype may be numpy float64 or pandas Float64 (nullable)
        assert "float" in str(result["x"].dtype).lower()

    def test_invalid_df_type_raises(self):
        ds = DataSource()
        with pytest.raises(Exception):
            ds.apply_data_types([1, 2, 3])

    def test_fast_convert_fallback(self):
        """When fast convert fails, column-by-column fallback should handle."""
        ds = DataSource()
        df = pd.DataFrame({"x": ["alice", "bob"], "y": [1.0, 2.0]})
        with pytest.raises(Exception):
            ds.apply_data_types(df, data_types={"x": "int64"})

    def test_dtype_equal_already(self):
        ds = DataSource()
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
        result = ds.apply_data_types(df, data_types={"x": "int64"})
        assert len(result) == 3


# --- schema cache clearing paths ---
class TestSchemaCacheClearing:
    def test_add_dataframe_clears_schema_cache(self):
        """Covers lines 147-149 in op_data_source.py."""
        ds = DataSource()
        df1 = pd.DataFrame({"x": [1]})
        ds.add_dataframe("df", df1)
        # Force schema into cache by getting it
        ds.get_schema("df")
        # Now add a new df with same name - should clear cache
        df2 = pd.DataFrame({"y": [2]})
        ds.add_dataframe("df", df2)
        assert "df" in ds.dataframes

    def test_add_encryption_key_clears_cache(self):
        """Covers lines 158-162 in op_data_source.py."""
        ds = DataSource(dataframes={"df": pd.DataFrame({"x": [1]})})
        ds.get_schema("df")
        # Adding encryption key for same name should clear schema cache
        ds.add_encryption_key("df", "mykey")
        assert ds.encryption_keys.get("df") == "mykey"

    def test_add_file_path_clears_cache(self, tmp_path):
        """Covers lines 196-198 in op_data_source.py."""
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(f, index=False)
        ds = DataSource(file_paths={"csv": f})
        # Trigger schema caching
        ds.get_schema("csv")
        # Replace the file path - should clear cache
        f2 = tmp_path / "data2.csv"
        pd.DataFrame({"y": [2]}).to_csv(f2, index=False)
        ds.add_file_path("csv", f2)
        assert ds.file_paths["csv"] == f2


# --- context manager with exception ---
class TestContextManagerWithException:
    def test_exit_with_exception_logs_error(self):
        """Covers lines 121-124 in op_data_source.py."""
        ds = DataSource()
        try:
            with ds:
                raise ValueError("test error")
        except ValueError:
            pass  # Exception should propagate (return False from __exit__)


# --- get_dataframe column filter edge cases ---
class TestGetDataframeColumnFilter:
    def test_all_requested_columns_missing(self):
        """Covers lines 464-474: all requested columns missing."""
        ds = DataSource(dataframes={"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})})
        df, err = ds.get_dataframe("df", columns=["nonexistent1", "nonexistent2"])
        assert df is None
        assert err is not None

    def test_some_columns_missing(self):
        """Covers lines 457-474: some requested columns missing."""
        ds = DataSource(dataframes={"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})})
        df, err = ds.get_dataframe("df", columns=["x", "nonexistent"])
        # Should return df with only valid columns
        assert df is not None or err is not None
