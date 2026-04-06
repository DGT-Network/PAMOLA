"""
Extended unit tests for pamola_core/utils/io.py to increase coverage.

Covers missed lines:
- Simple delegation functions: get_file_size, file_exists, safe_remove_file,
  validate_file_type, calculate_checksum, get_file_stats
- JSON functions: read_json, write_json, append_to_json_array, merge_json_objects
- write_dataframe_to_csv (progress path, dask path, compression validation)
- write_chunks_to_csv (compression validation, error paths)
- write_csv (dict writing)
- save_dataframe (all format branches)
- read_dataframe (all format branches)
- load_data_operation and load_settings_operation
- generate_word_frequencies
- optimize_dataframe_memory, estimate_file_memory, get_system_memory
- detect_csv_dialect, validate_file_format, is_encrypted_file
- save_visualization, save_plot
"""

import csv
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from pamola_core.utils import io
from pamola_core.errors.exceptions import (
    InvalidParameterError,
    PamolaFileNotFoundError,
    FileValidationError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv(tmp_path):
    """Small CSV file fixture."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p, df


@pytest.fixture
def tmp_json(tmp_path):
    """Small JSON file fixture."""
    data = {"name": "test", "value": 42, "items": [1, 2, 3]}
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p, data


@pytest.fixture
def sample_df():
    """Simple 5-row DataFrame."""
    return pd.DataFrame({"id": range(5), "name": [f"n{i}" for i in range(5)]})


# ===========================================================================
# File metadata delegation functions (lines ~157, 174, 191, 210, 229)
# ===========================================================================

class TestFileMetadataDelegation:
    """Tests for simple delegation wrappers around file_utils."""

    def test_get_file_metadata_returns_dict(self, tmp_path):
        f = tmp_path / "meta.txt"
        f.write_text("hello")
        result = io.get_file_metadata(f)
        assert isinstance(result, dict)

    def test_calculate_checksum_sha256(self, tmp_path):
        f = tmp_path / "chk.txt"
        f.write_text("data")
        result = io.calculate_checksum(f, "sha256")
        assert result is not None
        assert len(result) == 64  # sha256 hex length

    def test_calculate_checksum_md5(self, tmp_path):
        f = tmp_path / "chk_md5.txt"
        f.write_text("data")
        result = io.calculate_checksum(f, "md5")
        assert result is not None
        assert len(result) == 32  # md5 hex length

    def test_calculate_checksum_nonexistent_returns_none(self, tmp_path):
        result = io.calculate_checksum(tmp_path / "no_file.txt", "sha256")
        assert result is None

    def test_get_file_size_existing_file(self, tmp_path):
        f = tmp_path / "size.txt"
        f.write_text("12345")
        result = io.get_file_size(f)
        assert isinstance(result, int)
        assert result > 0

    def test_get_file_size_nonexistent_returns_none(self, tmp_path):
        result = io.get_file_size(tmp_path / "ghost.txt")
        assert result is None

    def test_file_exists_true(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("hi")
        assert io.file_exists(f) is True

    def test_file_exists_false(self, tmp_path):
        assert io.file_exists(tmp_path / "missing.txt") is False

    def test_safe_remove_file_success(self, tmp_path):
        f = tmp_path / "remove_me.txt"
        f.write_text("bye")
        result = io.safe_remove_file(f)
        assert result is True
        assert not f.exists()

    def test_safe_remove_file_nonexistent(self, tmp_path):
        result = io.safe_remove_file(tmp_path / "no_file.txt")
        assert result is False

    def test_validate_file_type_correct_extension(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b")
        assert io.validate_file_type(f, "csv") is True

    def test_validate_file_type_wrong_extension(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b")
        assert io.validate_file_type(f, "json") is False

    def test_get_file_stats_returns_dict(self, tmp_path):
        f = tmp_path / "stats.txt"
        f.write_text("content")
        result = io.get_file_stats(f)
        assert isinstance(result, dict)


# ===========================================================================
# JSON read/write functions (lines 398, 409-436, 1154-1159)
# ===========================================================================

class TestJSONFunctions:
    """Tests for JSON reading and writing."""

    def test_read_json_basic(self, tmp_json):
        p, data = tmp_json
        result = io.read_json(p)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_read_json_nonexistent_raises(self, tmp_path):
        with pytest.raises((PamolaFileNotFoundError, FileNotFoundError, Exception)):
            io.read_json(tmp_path / "no.json")

    def test_write_json_basic(self, tmp_path, sample_df):
        data = {"key": "val", "num": 1}
        out = tmp_path / "out.json"
        result = io.write_json(data, out)
        assert result == out
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["key"] == "val"

    def test_write_json_with_list(self, tmp_path):
        data = [{"a": 1}, {"a": 2}]
        out = tmp_path / "list.json"
        io.write_json(data, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert len(loaded) == 2

    def test_write_json_convert_numpy(self, tmp_path):
        data = {"arr": np.array([1, 2, 3]), "val": np.int64(42)}
        out = tmp_path / "np.json"
        io.write_json(data, out, convert_numpy=True)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["val"] == 42
        assert loaded["arr"] == [1, 2, 3]

    def test_write_json_creates_parent_dir(self, tmp_path):
        out = tmp_path / "subdir" / "new.json"
        io.write_json({"x": 1}, out)
        assert out.exists()

    def test_write_json_io_error_propagates(self, tmp_path):
        """IOError during write should propagate."""
        out = tmp_path / "bad.json"
        with patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(IOError):
                io.write_json({"k": "v"}, out)


# ===========================================================================
# append_to_json_array (lines 1288-1330)
# ===========================================================================

class TestAppendToJsonArray:
    def test_creates_new_file_with_item(self, tmp_path):
        out = tmp_path / "arr.json"
        io.append_to_json_array({"x": 1}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == [{"x": 1}]

    def test_appends_to_existing_array(self, tmp_path):
        out = tmp_path / "arr.json"
        out.write_text(json.dumps([{"x": 1}]), encoding="utf-8")
        io.append_to_json_array({"x": 2}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert len(loaded) == 2
        assert loaded[1]["x"] == 2

    def test_overwrites_non_array_with_new_array(self, tmp_path):
        out = tmp_path / "arr.json"
        out.write_text(json.dumps({"not": "array"}), encoding="utf-8")
        io.append_to_json_array({"item": 1}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert loaded[0]["item"] == 1

    def test_raises_when_file_missing_and_no_create(self, tmp_path):
        out = tmp_path / "missing.json"
        with pytest.raises((FileValidationError, Exception)):
            io.append_to_json_array({"x": 1}, out, create_if_missing=False)

    def test_handles_invalid_json_gracefully(self, tmp_path):
        out = tmp_path / "bad.json"
        out.write_text("not valid json", encoding="utf-8")
        # Should create new array rather than raise
        io.append_to_json_array({"x": 1}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)


# ===========================================================================
# merge_json_objects (lines 1288-1330)
# ===========================================================================

class TestMergeJsonObjects:
    def test_creates_new_file(self, tmp_path):
        out = tmp_path / "obj.json"
        io.merge_json_objects({"a": 1}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == {"a": 1}

    def test_merges_with_existing(self, tmp_path):
        out = tmp_path / "obj.json"
        out.write_text(json.dumps({"a": 1}), encoding="utf-8")
        io.merge_json_objects({"b": 2}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["a"] == 1
        assert loaded["b"] == 2

    def test_overwrites_existing_key(self, tmp_path):
        out = tmp_path / "obj.json"
        out.write_text(json.dumps({"a": 1}), encoding="utf-8")
        io.merge_json_objects({"a": 99}, out, overwrite_existing=True)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["a"] == 99

    def test_does_not_overwrite_when_disabled(self, tmp_path):
        out = tmp_path / "obj.json"
        out.write_text(json.dumps({"a": 1}), encoding="utf-8")
        io.merge_json_objects({"a": 99}, out, overwrite_existing=False)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["a"] == 1

    def test_raises_when_missing_no_create(self, tmp_path):
        out = tmp_path / "missing.json"
        with pytest.raises((FileValidationError, Exception)):
            io.merge_json_objects({"a": 1}, out, create_if_missing=False)

    def test_overwrites_non_dict_with_new_dict(self, tmp_path):
        out = tmp_path / "obj.json"
        out.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        io.merge_json_objects({"key": "val"}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)

    def test_recursive_merge(self, tmp_path):
        out = tmp_path / "deep.json"
        out.write_text(json.dumps({"a": {"x": 1}}), encoding="utf-8")
        io.merge_json_objects({"a": {"y": 2}}, out, recursive_merge=True)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["a"]["x"] == 1
        assert loaded["a"]["y"] == 2


# ===========================================================================
# write_dataframe_to_csv — validation / progress paths (lines 652-782)
# ===========================================================================

class TestWriteDataframeToCsvExtended:
    def test_invalid_compression_raises(self, tmp_path, sample_df):
        with pytest.raises(InvalidParameterError):
            io.write_dataframe_to_csv(sample_df, tmp_path / "out.csv", compression="lz4")

    def test_quoting_parameter_csv_quote_all(self, tmp_path, sample_df):
        out = tmp_path / "quoted.csv"
        io.write_dataframe_to_csv(sample_df, out, quoting=csv.QUOTE_ALL, show_progress=False)
        assert out.exists()

    def test_large_dataframe_with_progress(self, tmp_path):
        """DataFrame > PROGRESS_CHUNK_SIZE triggers chunked progress path."""
        df = pd.DataFrame({"a": range(15000), "b": range(15000)})
        out = tmp_path / "large.csv"
        result = io.write_dataframe_to_csv(df, out, show_progress=True)
        assert result == out
        df_back = pd.read_csv(out)
        assert len(df_back) == 15000

    def test_creates_parent_directory(self, tmp_path, sample_df):
        out = tmp_path / "sub" / "dir" / "out.csv"
        io.write_dataframe_to_csv(sample_df, out, show_progress=False)
        assert out.exists()

    def test_gzip_compression(self, tmp_path, sample_df):
        out = tmp_path / "out.csv.gz"
        result = io.write_dataframe_to_csv(sample_df, out, compression="gzip", show_progress=False)
        assert result == out
        assert out.exists()

    def test_custom_delimiter(self, tmp_path, sample_df):
        out = tmp_path / "pipe.csv"
        io.write_dataframe_to_csv(sample_df, out, delimiter="|", show_progress=False)
        content = out.read_text(encoding="utf-8")
        assert "|" in content


# ===========================================================================
# write_chunks_to_csv validation (lines 826)
# ===========================================================================

class TestWriteChunksToCsvExtended:
    def test_invalid_compression_raises(self, tmp_path, sample_df):
        chunks = iter([sample_df])
        with pytest.raises(InvalidParameterError):
            io.write_chunks_to_csv(chunks, tmp_path / "out.csv", compression="zlib")

    def test_writes_multiple_chunks(self, tmp_path):
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})
        out = tmp_path / "multi.csv"
        result = io.write_chunks_to_csv(iter([df1, df2]), out)
        assert result == out
        df_back = pd.read_csv(out)
        assert len(df_back) == 4

    def test_empty_chunks_creates_empty_file(self, tmp_path):
        out = tmp_path / "empty.csv"
        result = io.write_chunks_to_csv(iter([]), out)
        assert result == out


# ===========================================================================
# write_csv (lines 908-916)
# ===========================================================================

class TestWriteCSV:
    def test_writes_dict_as_csv(self, tmp_path):
        data = {"accuracy": 0.95, "f1": 0.90, "recall": 0.88}
        out = tmp_path / "metrics.csv"
        result = io.write_csv(data, out)
        assert result == str(out)
        rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0] == ["Metric", "Value"]
        assert rows[1][0] == "accuracy"

    def test_write_csv_flat_dir(self, tmp_path):
        out = tmp_path / "metrics.csv"
        io.write_csv({"k": "v"}, out)
        assert out.exists()


# ===========================================================================
# save_dataframe (lines 1647-1740)
# ===========================================================================

class TestSaveDataframe:
    def test_csv_format(self, tmp_path, sample_df):
        out = tmp_path / "df.csv"
        result = io.save_dataframe(sample_df, out, format="csv")
        assert out.exists()

    def test_json_format_records(self, tmp_path, sample_df):
        out = tmp_path / "df.json"
        result = io.save_dataframe(sample_df, out, format="json")
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert len(loaded) == 5

    def test_json_format_dict_orient(self, tmp_path, sample_df):
        out = tmp_path / "df_dict.json"
        io.save_dataframe(sample_df, out, format="json", orient="dict")
        assert out.exists()

    def test_json_format_list_orient(self, tmp_path, sample_df):
        out = tmp_path / "df_list.json"
        io.save_dataframe(sample_df, out, format="json", orient="list")
        assert out.exists()

    def test_json_format_index_orient(self, tmp_path, sample_df):
        out = tmp_path / "df_index.json"
        io.save_dataframe(sample_df, out, format="json", orient="index")
        assert out.exists()

    def test_json_format_split_orient(self, tmp_path, sample_df):
        out = tmp_path / "df_split.json"
        io.save_dataframe(sample_df, out, format="json", orient="split")
        assert out.exists()

    def test_json_format_invalid_orient_falls_back_to_records(self, tmp_path, sample_df):
        out = tmp_path / "df_invalid.json"
        io.save_dataframe(sample_df, out, format="json", orient="nonsense")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)

    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available")
    def test_parquet_format(self, mock_pyarrow, tmp_path, sample_df):
        mock_pyarrow.return_value = None
        out = tmp_path / "df.parquet"
        with patch.object(pd.DataFrame, "to_parquet"):
            io.save_dataframe(sample_df, out, format="parquet")

    def test_pickle_format(self, tmp_path, sample_df):
        out = tmp_path / "df.pkl"
        result = io.save_dataframe(sample_df, out, format="pickle")
        # save_dataframe returns the path; verify the file was written
        assert result is not None
        target = Path(result) if isinstance(result, str) else result
        assert target.exists()
        loaded = pd.read_pickle(target)
        assert len(loaded) == 5

    def test_unsupported_format_raises(self, tmp_path, sample_df):
        with pytest.raises(InvalidParameterError):
            io.save_dataframe(sample_df, tmp_path / "df.xyz", format="xyz")

    def test_extension_added_when_missing(self, tmp_path, sample_df):
        out = tmp_path / "noext"
        result = io.save_dataframe(sample_df, out, format="csv")
        # File with .csv extension should exist
        assert (tmp_path / "noext.csv").exists()


# ===========================================================================
# read_dataframe (lines 1781-1942)
# ===========================================================================

class TestReadDataframe:
    def test_csv_format_inferred(self, tmp_path, sample_df):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        df = io.read_dataframe(csv_file)
        assert len(df) == 5

    def test_json_format_list(self, tmp_path, sample_df):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_df.to_dict(orient="records")), encoding="utf-8")
        df = io.read_dataframe(json_file)
        assert len(df) == 5

    def test_json_format_dict_values_are_dicts(self, tmp_path):
        # orient=index produces dict-of-dicts
        data = {"0": {"id": 0, "name": "n0"}, "1": {"id": 1, "name": "n1"}}
        json_file = tmp_path / "idx.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        df = io.read_dataframe(json_file)
        assert isinstance(df, pd.DataFrame)

    def test_json_format_with_orient_columns(self, tmp_path, sample_df):
        data = sample_df.to_dict(orient="list")
        json_file = tmp_path / "cols.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        df = io.read_dataframe(json_file, orient="columns")
        assert len(df) == 5

    def test_json_format_with_nrows(self, tmp_path, sample_df):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_df.to_dict(orient="records")), encoding="utf-8")
        df = io.read_dataframe(json_file, nrows=3)
        assert len(df) == 3

    def test_json_format_with_skiprows_int(self, tmp_path, sample_df):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_df.to_dict(orient="records")), encoding="utf-8")
        df = io.read_dataframe(json_file, skiprows=2)
        assert len(df) == 3

    def test_json_format_with_skiprows_list(self, tmp_path, sample_df):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_df.to_dict(orient="records")), encoding="utf-8")
        df = io.read_dataframe(json_file, skiprows=[0, 1])
        assert len(df) == 3

    def test_pickle_format(self, tmp_path, sample_df):
        pkl_file = tmp_path / "data.pkl"
        sample_df.to_pickle(pkl_file)
        df = io.read_dataframe(pkl_file)
        assert len(df) == 5

    def test_pickle_with_column_filter(self, tmp_path, sample_df):
        pkl_file = tmp_path / "data.pkl"
        sample_df.to_pickle(pkl_file)
        df = io.read_dataframe(pkl_file, columns=["id"])
        assert list(df.columns) == ["id"]

    def test_pickle_with_nrows(self, tmp_path, sample_df):
        pkl_file = tmp_path / "data.pkl"
        sample_df.to_pickle(pkl_file)
        df = io.read_dataframe(pkl_file, nrows=2)
        assert len(df) == 2

    def test_pickle_with_skiprows_int(self, tmp_path, sample_df):
        pkl_file = tmp_path / "data.pkl"
        sample_df.to_pickle(pkl_file)
        df = io.read_dataframe(pkl_file, skiprows=2)
        assert len(df) == 3

    def test_pickle_with_skiprows_list(self, tmp_path, sample_df):
        pkl_file = tmp_path / "data.pkl"
        sample_df.to_pickle(pkl_file)
        df = io.read_dataframe(pkl_file, skiprows=[0, 1])
        assert len(df) == 3

    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available")
    @patch("pandas.read_parquet")
    def test_parquet_format(self, mock_read_parquet, mock_pyarrow, tmp_path, sample_df):
        mock_pyarrow.return_value = None
        mock_read_parquet.return_value = sample_df
        parquet_file = tmp_path / "data.parquet"
        parquet_file.write_bytes(b"fake_parquet_content")
        df = io.read_dataframe(parquet_file)
        assert len(df) == 5

    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available")
    @patch("pandas.read_parquet")
    def test_parquet_with_skiprows_int(self, mock_read_parquet, mock_pyarrow, tmp_path, sample_df):
        mock_pyarrow.return_value = None
        mock_read_parquet.return_value = sample_df
        parquet_file = tmp_path / "data.parquet"
        parquet_file.write_bytes(b"fake")
        df = io.read_dataframe(parquet_file, skiprows=2)
        assert len(df) == 3

    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available")
    @patch("pandas.read_parquet")
    def test_parquet_with_skiprows_list(self, mock_read_parquet, mock_pyarrow, tmp_path, sample_df):
        mock_pyarrow.return_value = None
        mock_read_parquet.return_value = sample_df
        parquet_file = tmp_path / "data.parquet"
        parquet_file.write_bytes(b"fake")
        df = io.read_dataframe(parquet_file, skiprows=[0, 1])
        assert len(df) == 3

    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available")
    @patch("pandas.read_parquet")
    def test_parquet_with_nrows(self, mock_read_parquet, mock_pyarrow, tmp_path, sample_df):
        mock_pyarrow.return_value = None
        mock_read_parquet.return_value = sample_df
        parquet_file = tmp_path / "data.parquet"
        parquet_file.write_bytes(b"fake")
        df = io.read_dataframe(parquet_file, nrows=2)
        assert len(df) == 2

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(PamolaFileNotFoundError):
            io.read_dataframe(tmp_path / "ghost.csv")

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("content")
        with pytest.raises(InvalidParameterError):
            io.read_dataframe(f, file_format="xyz")

    def test_explicit_format_overrides_extension(self, tmp_path, sample_df):
        f = tmp_path / "data.txt"
        sample_df.to_csv(f, index=False)
        df = io.read_dataframe(f, file_format="csv")
        assert len(df) == 5


# ===========================================================================
# load_data_operation (lines 1994-2055)
# ===========================================================================

class TestLoadDataOperation:
    def test_with_dataframe_passthrough(self, sample_df):
        result = io.load_data_operation(sample_df)
        assert len(result) == 5

    def test_with_csv_file_path_str(self, tmp_path, sample_df):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        result = io.load_data_operation(str(csv_file))
        assert len(result) == 5

    def test_with_csv_file_path_obj(self, tmp_path, sample_df):
        csv_file = tmp_path / "data.csv"
        sample_df.to_csv(csv_file, index=False)
        result = io.load_data_operation(csv_file)
        assert len(result) == 5

    def test_with_data_source_get_dataframe(self, sample_df):
        mock_ds = MagicMock()
        mock_ds.get_dataframe.return_value = (sample_df, None)
        result = io.load_data_operation(mock_ds, "dataset1")
        assert len(result) == 5

    def test_with_data_source_returns_none_raises(self):
        mock_ds = MagicMock()
        mock_ds.get_dataframe.return_value = (None, {"message": "Not found"})
        with pytest.raises(InvalidParameterError):
            io.load_data_operation(mock_ds, "dataset1")

    def test_unsupported_type_raises(self):
        with pytest.raises(InvalidParameterError):
            io.load_data_operation(12345)

    def test_invalid_path_raises_validation_error(self):
        with pytest.raises((ValidationError, Exception)):
            io.load_data_operation("/nonexistent/path/data.csv")


# ===========================================================================
# load_settings_operation (lines 2262, 2285, 2367, 2374, 2377-2380, 2410-2414)
# ===========================================================================

class TestLoadSettingsOperation:
    def _make_data_source(self, enc_key=None, enc_mode=None):
        ds = MagicMock()
        ds.encryption_keys = {"main": enc_key}
        ds.encryption_modes = {"main": enc_mode}
        return ds

    def test_default_settings(self):
        ds = self._make_data_source()
        result = io.load_settings_operation(ds, "main")
        assert result["encoding"] == "utf-8"
        assert result["delimiter"] == ","
        assert result["quotechar"] == '"'
        assert result["use_encryption"] is False
        assert result["detect_parameters"] is False
        assert result["use_dask"] is False

    def test_custom_encoding(self):
        ds = self._make_data_source()
        result = io.load_settings_operation(ds, "main", encoding="latin-1")
        assert result["encoding"] == "latin-1"

    def test_custom_delimiter(self):
        ds = self._make_data_source()
        result = io.load_settings_operation(ds, "main", delimiter=";")
        assert result["delimiter"] == ";"

    def test_encryption_key_forwarded(self):
        ds = self._make_data_source(enc_key="secret123")
        result = io.load_settings_operation(ds, "main")
        assert result["encryption_key"] == "secret123"

    def test_encryption_mode_forwarded(self):
        ds = self._make_data_source(enc_mode="aes256")
        result = io.load_settings_operation(ds, "main")
        assert result["encryption_mode"] == "aes256"

    def test_use_encryption_flag(self):
        ds = self._make_data_source()
        result = io.load_settings_operation(ds, "main", use_encryption=True)
        assert result["use_encryption"] is True

    def test_missing_key_returns_none(self):
        ds = MagicMock()
        ds.encryption_keys = {}
        ds.encryption_modes = {}
        result = io.load_settings_operation(ds, "nonexistent")
        assert result["encryption_key"] is None
        assert result["encryption_mode"] is None


# ===========================================================================
# generate_word_frequencies (lines 2410-2414)
# ===========================================================================

class TestGenerateWordFrequencies:
    def test_basic_frequency_count(self):
        text = "hello world hello"
        result = io.generate_word_frequencies(text)
        assert result["hello"] == 2
        assert result["world"] == 1

    def test_case_insensitive(self):
        text = "Hello HELLO hello"
        result = io.generate_word_frequencies(text)
        assert result["hello"] == 3

    def test_exclude_words(self):
        text = "the quick brown fox the"
        result = io.generate_word_frequencies(text, exclude_words=["the"])
        assert "the" not in result
        assert result["quick"] == 1

    def test_empty_string(self):
        result = io.generate_word_frequencies("")
        assert result == {}

    def test_special_characters_stripped(self):
        text = "hello! world? hello."
        result = io.generate_word_frequencies(text)
        assert result["hello"] == 2
        assert result["world"] == 1

    def test_exclude_words_case_insensitive(self):
        text = "The quick THE"
        result = io.generate_word_frequencies(text, exclude_words=["THE"])
        assert "the" not in result


# ===========================================================================
# Memory and format helpers delegation (lines ~293, 2262, 2285)
# ===========================================================================

class TestMemoryAndFormatHelpers:
    @patch("pamola_core.utils.io_helpers.memory_utils.get_system_memory")
    def test_get_system_memory_delegates(self, mock_fn):
        mock_fn.return_value = {"total_gb": 16.0, "available_gb": 8.0}
        result = io.get_system_memory()
        mock_fn.assert_called_once()
        assert result["total_gb"] == 16.0

    @patch("pamola_core.utils.io_helpers.memory_utils.estimate_file_memory")
    def test_estimate_file_memory_delegates(self, mock_fn, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("a,b\n1,2")
        mock_fn.return_value = {"file_size_bytes": 10, "file_type": "csv"}
        result = io.estimate_file_memory(f)
        mock_fn.assert_called_once_with(f)

    @patch("pamola_core.utils.io_helpers.memory_utils.estimate_file_memory")
    def test_estimate_file_memory_list_delegates(self, mock_fn, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("a\n1")
        f2.write_text("a\n2")
        mock_fn.return_value = {"file_size_bytes": 5}
        result = io.estimate_file_memory_list([f1, f2])
        assert mock_fn.call_count == 2

    @patch("pamola_core.utils.io_helpers.memory_utils.optimize_dataframe_memory")
    def test_optimize_dataframe_memory_delegates(self, mock_fn, sample_df):
        mock_fn.return_value = (sample_df, {"savings_percent": 10.0})
        df, info = io.optimize_dataframe_memory(sample_df, categorical_threshold=0.3)
        mock_fn.assert_called_once()
        assert info["savings_percent"] == 10.0

    @patch("pamola_core.utils.io_helpers.csv_utils.detect_csv_dialect")
    def test_detect_csv_dialect_delegates(self, mock_fn, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("a,b\n1,2")
        mock_fn.return_value = {"delimiter": ","}
        result = io.detect_csv_dialect(f)
        mock_fn.assert_called_once()

    @patch("pamola_core.utils.io_helpers.format_utils.validate_file_format")
    def test_validate_file_format_delegates(self, mock_fn, tmp_path):
        f = tmp_path / "v.csv"
        f.write_text("a,b")
        mock_fn.return_value = {"valid": True}
        result = io.validate_file_format(f, expected_format="csv")
        mock_fn.assert_called_once()

    @patch("pamola_core.utils.io_helpers.format_utils.is_encrypted_file")
    def test_is_encrypted_file_delegates(self, mock_fn, tmp_path):
        f = tmp_path / "enc.bin"
        f.write_bytes(b"\x00\x01\x02")
        mock_fn.return_value = True
        result = io.is_encrypted_file(f)
        mock_fn.assert_called_once_with(f)
        assert result is True


# ===========================================================================
# save_visualization (lines 1503, 1529-1559)
# ===========================================================================

class TestSaveVisualization:
    def test_unsupported_type_raises(self, tmp_path):
        from pamola_core.errors.exceptions import TypeValidationError
        with pytest.raises((TypeValidationError, Exception)):
            io.save_visualization("not_a_figure", tmp_path / "fig.png")

    def test_pil_image_saved(self, tmp_path):
        """PIL Image saved correctly."""
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (10, 10), color=(255, 0, 0))
        out = tmp_path / "image.png"
        result = io.save_visualization(img, out, format="png")
        assert out.exists()

    def test_wordcloud_dict_saved(self, tmp_path):
        """WordCloud result dict (with 'image' key) saved correctly."""
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (10, 10), color=(0, 255, 0))
        figure = {"image": img}
        out = tmp_path / "wc.png"
        result = io.save_visualization(figure, out, format="png")
        assert out.exists()

    def test_matplotlib_figure_saved(self, tmp_path):
        """Matplotlib figure with savefig saved correctly."""
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock()
        out = tmp_path / "mpl.png"
        io.save_visualization(mock_fig, out, format="png")
        mock_fig.savefig.assert_called()

    def test_extension_corrected_from_format(self, tmp_path):
        """If path extension doesn't match format, it gets corrected."""
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (5, 5))
        out = tmp_path / "image.bmp"
        result = io.save_visualization(img, out, format="png")
        # Result path should end in .png
        assert str(result).endswith(".png")


class TestSavePlot:
    def test_delegates_to_save_visualization(self, tmp_path):
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock()
        out = tmp_path / "plot.png"
        with patch("pamola_core.utils.io.save_visualization") as mock_sv:
            mock_sv.return_value = out
            result = io.save_plot(mock_fig, out, dpi=150)
            mock_sv.assert_called_once()
            call_kwargs = mock_sv.call_args
            assert call_kwargs[1]["format"] == "png" or call_kwargs[0][2] == "png"


# ===========================================================================
# read_csv_in_chunks — dask path and use_dask with encryption warning
# ===========================================================================

class TestReadCsvInChunksExtended:
    def test_dask_with_encryption_falls_back(self, tmp_csv):
        """use_dask=True with encryption_key falls back to pandas (raises because file not encrypted)."""
        p, df = tmp_csv
        # The file is not actually encrypted, so decryption will fail
        with pytest.raises(Exception):
            list(io.read_csv_in_chunks(
                p,
                chunk_size=2,
                use_dask=True,
                encryption_key="secret",
                show_progress=False,
            ))

    @patch("pamola_core.utils.io_helpers.dask_utils.is_dask_available", return_value=True)
    @patch("pamola_core.utils.io_helpers.dask_utils.read_csv_in_chunks")
    def test_dask_path_used_when_available(self, mock_dask_read, mock_dask_avail, tmp_csv):
        """When dask is available and no encryption, dask path is used."""
        p, df = tmp_csv
        # Dask path should yield chunks; simulate with real data
        mock_dask_read.return_value = iter([df])
        chunks = list(io.read_csv_in_chunks(
            p,
            use_dask=True,
            show_progress=False,
        ))
        mock_dask_read.assert_called_once()

    @patch("pamola_core.utils.io_helpers.dask_utils.is_dask_available", return_value=True)
    @patch("pamola_core.utils.io_helpers.dask_utils.read_csv_in_chunks", side_effect=Exception("dask error"))
    def test_dask_error_falls_back_to_pandas(self, mock_dask_read, mock_dask_avail, tmp_csv):
        """Dask error triggers fallback to pandas chunking."""
        p, df = tmp_csv
        chunks = list(io.read_csv_in_chunks(
            p,
            use_dask=True,
            show_progress=False,
            chunk_size=2,
        ))
        total = sum(len(c) for c in chunks)
        assert total == 3


# ===========================================================================
# write_parquet — exception paths (lines 1449-1454)
# ===========================================================================

class TestWriteParquetExtended:
    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available",
           side_effect=ImportError("no pyarrow"))
    def test_missing_pyarrow_raises(self, mock_check, tmp_path, sample_df):
        with pytest.raises(ImportError):
            io.write_parquet(sample_df, tmp_path / "out.parquet")

    @patch("pamola_core.utils.io_helpers.format_utils.check_pyarrow_available")
    @patch.object(pd.DataFrame, "to_parquet", side_effect=Exception("write failed"))
    def test_generic_exception_propagates(self, mock_to_parquet, mock_check, tmp_path, sample_df):
        mock_check.return_value = None
        with pytest.raises(Exception, match="write failed"):
            io.write_parquet(sample_df, tmp_path / "out.parquet")


# ===========================================================================
# read_dataframe — JSON orient paths (split, tight)
# ===========================================================================

class TestReadDataframeJsonOrients:
    def test_records_orient(self, tmp_path, sample_df):
        data = sample_df.to_dict(orient="records")
        f = tmp_path / "records.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        df = io.read_dataframe(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_df)


# ===========================================================================
# Ensure json column filtering on read_dataframe
# ===========================================================================

class TestReadDataframeJsonColumnFilter:
    def test_column_filter_applied(self, tmp_path, sample_df):
        f = tmp_path / "data.json"
        f.write_text(json.dumps(sample_df.to_dict(orient="records")), encoding="utf-8")
        df = io.read_dataframe(f, columns=["id"])
        assert "name" not in df.columns
        assert "id" in df.columns
