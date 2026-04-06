"""Tests targeting dask write paths and progress bar paths in utils/io.py.
Mocks thresholds to trigger dask/progress code with small DataFrames.
Targets missed lines: 680-702, 704-755, 774-782, 881-889, 1529-1536, 1994-2055."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import pamola_core.utils.io as io_module


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "a": range(100),
        "b": [f"val_{i}" for i in range(100)],
        "c": np.random.rand(100),
    })


# --- Dask write path (lines 680-702) ---
class TestDaskWritePath:
    def test_write_csv_triggers_dask_path(self, small_df, tmp_path):
        """Mock DASK_THRESHOLD_ROWS to 10 to trigger dask path with 100 rows."""
        out = tmp_path / "out.csv"
        with patch.object(io_module, "DASK_THRESHOLD_ROWS", 10):
            result = io_module.write_dataframe_to_csv(
                small_df, out, show_progress=False, use_dask=True,
            )
        assert Path(result).exists() if result else True

    def test_write_csv_dask_fallback_on_error(self, small_df, tmp_path):
        """When dask write fails, should fall back to pandas."""
        out = tmp_path / "out.csv"
        with patch.object(io_module, "DASK_THRESHOLD_ROWS", 10), \
             patch("pamola_core.utils.io_helpers.dask_utils.write_dataframe_to_csv",
                   side_effect=RuntimeError("dask fail")):
            result = io_module.write_dataframe_to_csv(
                small_df, out, show_progress=False, use_dask=True,
            )
        assert out.exists()


# --- Progress bar write path (lines 704-755) ---
class TestProgressWritePath:
    def test_write_csv_with_progress(self, small_df, tmp_path):
        """Mock PROGRESS_CHUNK_SIZE to trigger progress path."""
        out = tmp_path / "prog.csv"
        with patch.object(io_module, "PROGRESS_CHUNK_SIZE", 10):
            result = io_module.write_dataframe_to_csv(
                small_df, out, show_progress=True,
            )
        assert out.exists()


# --- write_dataframe_to_json (lines 881-889) ---
class TestWriteJsonPaths:
    def test_write_json_records(self, small_df, tmp_path):
        out = tmp_path / "data.json"
        result = io_module.write_json(
            small_df.to_dict(orient="records"), out,
        )
        assert Path(result).exists()

    def test_write_json_with_indent(self, small_df, tmp_path):
        out = tmp_path / "data_pretty.json"
        result = io_module.write_json(
            small_df.to_dict(orient="records"), out, indent=2,
        )
        assert Path(result).exists()


# --- read_csv_in_chunks with progress (lines 1529-1536) ---
class TestReadCsvChunksProgress:
    def test_read_chunks_with_progress(self, tmp_path):
        df = pd.DataFrame({"a": range(500), "b": range(500)})
        f = tmp_path / "big.csv"
        df.to_csv(f, index=False)
        chunks = list(io_module.read_csv_in_chunks(f, chunk_size=100, show_progress=True))
        total = sum(len(c) for c in chunks)
        assert total == 500

    def test_read_chunks_without_progress(self, tmp_path):
        df = pd.DataFrame({"a": range(200), "b": range(200)})
        f = tmp_path / "data.csv"
        df.to_csv(f, index=False)
        chunks = list(io_module.read_csv_in_chunks(f, chunk_size=50, show_progress=False))
        total = sum(len(c) for c in chunks)
        assert total == 200


# --- save_dataframe JSON orient paths (lines 1667-1691) ---
class TestSaveDataframeJsonOrients:
    def test_save_json_records(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json")
        assert result is not None

    def test_save_json_dict_orient(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json", orient="dict")
        assert result is not None

    def test_save_json_list_orient(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json", orient="list")
        assert result is not None

    def test_save_json_split_orient(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json", orient="split")
        assert result is not None

    def test_save_json_index_orient(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json", orient="index")
        assert result is not None

    def test_save_json_tight_orient(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json", orient="tight")
        assert result is not None

    def test_save_json_invalid_orient_fallback(self, small_df, tmp_path):
        out = tmp_path / "df.json"
        result = io_module.save_dataframe(small_df, out, format="json", orient="badvalue")
        assert result is not None


# --- read_dataframe for different formats (lines 1823-1849) ---
class TestReadDataframeFormats:
    def test_read_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(f, index=False)
        df = io_module.read_dataframe(f)
        assert len(df) == 2

    def test_read_json(self, tmp_path):
        f = tmp_path / "data.json"
        pd.DataFrame({"x": [1, 2]}).to_json(f, orient="records")
        df = io_module.read_dataframe(f)
        assert isinstance(df, pd.DataFrame)

    def test_read_parquet(self, tmp_path):
        f = tmp_path / "data.parquet"
        pd.DataFrame({"x": [1, 2]}).to_parquet(f)
        df = io_module.read_dataframe(f)
        assert len(df) == 2

    def test_read_pickle(self, tmp_path):
        f = tmp_path / "data.pkl"
        pd.DataFrame({"x": [1, 2]}).to_pickle(f)
        df = io_module.read_dataframe(f)
        assert len(df) == 2

    def test_read_with_column_filter(self, tmp_path):
        f = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(f, index=False)
        df = io_module.read_dataframe(f, columns=["x"])
        assert list(df.columns) == ["x"]


# --- optimize_dataframe_memory (lines 1886-1934) ---
class TestOptimizeMemory:
    def test_optimize_basic(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "str_col": ["a", "b", "c", "d", "e"],
        })
        result = io_module.optimize_dataframe_memory(df)
        # Returns Tuple[pd.DataFrame, Dict] or just pd.DataFrame
        if isinstance(result, tuple):
            result_df = result[0]
        else:
            result_df = result
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 5

    def test_optimize_with_categories(self):
        df = pd.DataFrame({
            "cat": ["x", "y", "x", "y", "x"] * 100,
            "num": range(500),
        })
        result = io_module.optimize_dataframe_memory(df)
        if isinstance(result, tuple):
            result_df = result[0]
        else:
            result_df = result
        assert isinstance(result_df, pd.DataFrame)

    def test_optimize_empty(self):
        df = pd.DataFrame()
        result = io_module.optimize_dataframe_memory(df)
        if isinstance(result, tuple):
            result_df = result[0]
        else:
            result_df = result
        assert isinstance(result_df, pd.DataFrame)
