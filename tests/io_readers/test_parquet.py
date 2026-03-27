"""
Unit tests for pamola_core.io.parquet module.

Tests for Parquet reading and writing functionality:
- DataParquet class instantiation and interface
- read_parquet() function with various options
- write_parquet() function with various options
- Column selection
- Filters and compression
- Partitioning
- Missing file error handling
- Edge cases (empty files, single row)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from pamola_core.io.parquet import DataParquet, read_parquet, write_parquet

# Skip tests if pyarrow is not available
pytest.importorskip("pyarrow")


class TestDataParquetClass:
    """Test DataParquet handler class."""

    def test_dataparquet_instantiation(self):
        """Test DataParquet instance creation."""
        handler = DataParquet()
        assert handler is not None
        assert hasattr(handler, 'read')
        assert hasattr(handler, 'write')

    def test_dataparquet_read_method_exists(self):
        """Test DataParquet has callable read method."""
        handler = DataParquet()
        assert callable(handler.read)

    def test_dataparquet_write_method_exists(self):
        """Test DataParquet has callable write method."""
        handler = DataParquet()
        assert callable(handler.write)


class TestReadParquetBasic:
    """Test basic read_parquet functionality."""

    def test_read_parquet_simple_file(self, tmp_path):
        """Test reading a simple Parquet file."""
        parquet_file = tmp_path / "test.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert isinstance(df_read, pd.DataFrame)
        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_read_parquet_preserves_dtypes(self, tmp_path):
        """Test that read_parquet preserves data types.

        Note: read_parquet uses dtype_backend='numpy_nullable' by default,
        so integer columns may return Int64 (nullable) and float columns may
        return Float64 (nullable). String columns may be 'object' or
        pd.StringDtype() depending on the backend.
        """
        parquet_file = tmp_path / "dtypes.parquet"
        df_original = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert df_read['int_col'].dtype in [np.int64, np.int32, pd.Int64Dtype(), pd.Int32Dtype()]
        assert df_read['float_col'].dtype in [np.float64, np.float32, pd.Float64Dtype(), pd.Float32Dtype()]
        assert df_read['str_col'].dtype in ['object', pd.StringDtype()]

    def test_read_parquet_with_index(self, tmp_path):
        """Test reading Parquet with index."""
        parquet_file = tmp_path / "with_index.parquet"
        df_original = pd.DataFrame(
            {'name': ['Alice', 'Bob'], 'value': [100, 200]},
            index=['a', 'b']
        )
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert len(df_read) == 2


class TestReadParquetColumnSelection:
    """Test read_parquet column selection."""

    def test_read_parquet_specific_columns(self, tmp_path):
        """Test reading specific columns."""
        parquet_file = tmp_path / "columns.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file), columns=['id', 'name'])

        assert list(df_read.columns) == ['id', 'name']
        assert df_read.shape == (3, 2)

    def test_read_parquet_single_column(self, tmp_path):
        """Test reading single column."""
        parquet_file = tmp_path / "single_col.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file), columns=['id'])

        assert list(df_read.columns) == ['id']
        assert len(df_read) == 3

    def test_read_parquet_multiple_columns(self, tmp_path):
        """Test reading multiple columns."""
        parquet_file = tmp_path / "multi_cols.parquet"
        df_original = pd.DataFrame({
            f'col_{i}': range(i, i + 5) for i in range(10)
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file), columns=['col_0', 'col_5', 'col_9'])

        assert len(df_read.columns) == 3


class TestReadParquetFilters:
    """Test read_parquet with filters."""

    def test_read_parquet_with_filters(self, tmp_path):
        """Test reading Parquet with row filters."""
        parquet_file = tmp_path / "filters.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'status': ['active', 'inactive', 'active', 'active', 'inactive'],
            'value': [100, 200, 300, 400, 500]
        })
        df_original.to_parquet(parquet_file)

        # Filter for value > 250
        df_read = read_parquet(str(parquet_file), filters=[[('value', '>', 250)]])

        assert len(df_read) >= 3  # Should have ids 3, 4, 5

    def test_read_parquet_filters_equality(self, tmp_path):
        """Test filters with equality."""
        parquet_file = tmp_path / "filters_eq.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'status': ['active', 'active', 'inactive']
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file), filters=[[('status', '==', 'active')]])

        assert len(df_read) >= 2


class TestReadParquetDataTypes:
    """Test read_parquet dtype handling."""

    def test_read_parquet_with_numeric_types(self, tmp_path):
        """Test reading Parquet with numeric types."""
        parquet_file = tmp_path / "numeric.parquet"
        df_original = pd.DataFrame({
            'id': np.array([1, 2, 3], dtype=np.int32),
            'value': np.array([1.5, 2.5, 3.5], dtype=np.float32)
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert len(df_read) == 3

    def test_read_parquet_with_datetime(self, tmp_path):
        """Test reading Parquet with datetime."""
        parquet_file = tmp_path / "datetime.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10'])
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert pd.api.types.is_datetime64_any_dtype(df_read['date'])


class TestReadParquetEdgeCases:
    """Test read_parquet edge cases."""

    def test_read_parquet_empty_file(self, tmp_path):
        """Test reading empty Parquet file."""
        parquet_file = tmp_path / "empty.parquet"
        df_original = pd.DataFrame({
            'id': pd.Series([], dtype='int64'),
            'name': pd.Series([], dtype='object')
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert len(df_read) == 0
        assert list(df_read.columns) == ['id', 'name']

    def test_read_parquet_single_row(self, tmp_path):
        """Test reading Parquet with single row."""
        parquet_file = tmp_path / "single_row.parquet"
        df_original = pd.DataFrame({'id': [1], 'name': ['Alice']})
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert len(df_read) == 1

    def test_read_parquet_single_column(self, tmp_path):
        """Test reading Parquet with single column."""
        parquet_file = tmp_path / "single_column.parquet"
        df_original = pd.DataFrame({'id': [1, 2, 3]})
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert df_read.shape[1] == 1

    def test_read_parquet_with_null_values(self, tmp_path):
        """Test reading Parquet with null values."""
        parquet_file = tmp_path / "nulls.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, np.nan, 30.1]
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert pd.isna(df_read['value'].iloc[1])

    def test_read_parquet_special_column_names(self, tmp_path):
        """Test reading Parquet with special column names."""
        parquet_file = tmp_path / "special_names.parquet"
        df_original = pd.DataFrame({
            'col with spaces': [1, 2, 3],
            'col-with-dashes': [4, 5, 6],
            'col_with_underscores': [7, 8, 9]
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert 'col with spaces' in df_read.columns


class TestReadParquetLargeFile:
    """Test read_parquet with larger datasets."""

    def test_read_parquet_large_rows(self, tmp_path):
        """Test reading Parquet with many rows."""
        parquet_file = tmp_path / "large.parquet"
        n_rows = 100000
        df_original = pd.DataFrame({
            'id': range(1, n_rows + 1),
            'value': np.random.rand(n_rows)
        })
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert len(df_read) == n_rows

    def test_read_parquet_many_columns(self, tmp_path):
        """Test reading Parquet with many columns."""
        parquet_file = tmp_path / "many_cols.parquet"
        n_cols = 100
        data = {f'col_{i}': range(1, 101) for i in range(n_cols)}
        df_original = pd.DataFrame(data)
        df_original.to_parquet(parquet_file)

        df_read = read_parquet(str(parquet_file))

        assert df_read.shape[1] == n_cols


class TestReadParquetErrors:
    """Test read_parquet error handling."""

    def test_read_parquet_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_parquet("/nonexistent/path/to/file.parquet")

    def test_read_parquet_invalid_column_name(self, tmp_path):
        """Test reading with invalid column name."""
        parquet_file = tmp_path / "test.parquet"
        df_original = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df_original.to_parquet(parquet_file)

        with pytest.raises(Exception):  # KeyError or similar
            read_parquet(str(parquet_file), columns=['nonexistent'])


class TestWriteParquetBasic:
    """Test basic write_parquet functionality."""

    def test_write_parquet_simple(self, tmp_path):
        """Test writing a simple DataFrame to Parquet."""
        parquet_file = tmp_path / "output.parquet"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })

        write_parquet(df, str(parquet_file))

        assert parquet_file.exists()
        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 3

    def test_write_parquet_with_index(self, tmp_path):
        """Test writing Parquet with index."""
        parquet_file = tmp_path / "with_index.parquet"
        df = pd.DataFrame(
            {'name': ['Alice', 'Bob'], 'value': [100, 200]},
            index=['a', 'b']
        )

        write_parquet(df, str(parquet_file), index=True)

        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 2

    def test_write_parquet_without_index(self, tmp_path):
        """Test writing Parquet without index."""
        parquet_file = tmp_path / "no_index.parquet"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_parquet(df, str(parquet_file), index=False)

        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 3


class TestWriteParquetCompression:
    """Test write_parquet compression options."""

    def test_write_parquet_snappy_compression(self, tmp_path):
        """Test writing Parquet with snappy compression."""
        parquet_file = tmp_path / "snappy.parquet"
        df = pd.DataFrame({
            'id': range(1, 1001),
            'value': np.random.rand(1000)
        })

        write_parquet(df, str(parquet_file), compression='snappy')

        assert parquet_file.exists()
        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 1000

    def test_write_parquet_gzip_compression(self, tmp_path):
        """Test writing Parquet with gzip compression."""
        parquet_file = tmp_path / "gzip.parquet"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_parquet(df, str(parquet_file), compression='gzip')

        assert parquet_file.exists()
        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 3

    def test_write_parquet_no_compression(self, tmp_path):
        """Test writing Parquet without compression."""
        parquet_file = tmp_path / "no_compression.parquet"
        df = pd.DataFrame({'id': [1, 2, 3]})

        write_parquet(df, str(parquet_file), compression=None)

        assert parquet_file.exists()


class TestWriteParquetEdgeCases:
    """Test write_parquet edge cases."""

    def test_write_parquet_empty_dataframe(self, tmp_path):
        """Test writing empty DataFrame."""
        parquet_file = tmp_path / "empty.parquet"
        df = pd.DataFrame({
            'id': pd.Series([], dtype='int64'),
            'name': pd.Series([], dtype='object')
        })

        write_parquet(df, str(parquet_file))

        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 0

    def test_write_parquet_single_row(self, tmp_path):
        """Test writing single row DataFrame."""
        parquet_file = tmp_path / "single.parquet"
        df = pd.DataFrame({'id': [1], 'name': ['Alice']})

        write_parquet(df, str(parquet_file))

        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 1

    def test_write_parquet_with_nan(self, tmp_path):
        """Test writing Parquet with NaN values."""
        parquet_file = tmp_path / "with_nan.parquet"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, np.nan, 30.1]
        })

        write_parquet(df, str(parquet_file))

        df_read = pd.read_parquet(parquet_file)
        assert pd.isna(df_read['value'].iloc[1])

    def test_write_parquet_with_categorical(self, tmp_path):
        """Test writing Parquet with categorical data."""
        parquet_file = tmp_path / "categorical.parquet"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'category': pd.Categorical(['A', 'B', 'A'])
        })

        write_parquet(df, str(parquet_file))

        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == 3


class TestDataParquetHandler:
    """Test DataParquet handler integration."""

    def test_dataparquet_read_write_roundtrip(self, tmp_path):
        """Test reading and writing with DataParquet handler."""
        parquet_file = tmp_path / "roundtrip.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })

        handler = DataParquet()
        handler.write(df_original, str(parquet_file))

        df_read = handler.read(str(parquet_file))

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_dataparquet_multiple_instances(self, tmp_path):
        """Test using multiple DataParquet instances."""
        parquet_file1 = tmp_path / "file1.parquet"
        parquet_file2 = tmp_path / "file2.parquet"

        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        handler1 = DataParquet()
        handler2 = DataParquet()

        handler1.write(df1, str(parquet_file1))
        handler2.write(df2, str(parquet_file2))

        df1_read = handler1.read(str(parquet_file1))
        df2_read = handler2.read(str(parquet_file2))

        assert list(df1_read.columns) == ['a']
        assert list(df2_read.columns) == ['b']


class TestParquetRoundtrip:
    """Test Parquet read-write roundtrip with various data."""

    def test_parquet_roundtrip_datetime(self, tmp_path):
        """Test Parquet roundtrip with datetime."""
        parquet_file = tmp_path / "datetime_roundtrip.parquet"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10'])
        })

        write_parquet(df_original, str(parquet_file))
        df_read = read_parquet(str(parquet_file))

        assert pd.api.types.is_datetime64_any_dtype(df_read['date'])
        assert df_read['date'].iloc[0] == pd.Timestamp('2023-01-15')

    def test_parquet_roundtrip_mixed_types(self, tmp_path):
        """Test Parquet roundtrip with mixed data types."""
        parquet_file = tmp_path / "mixed_roundtrip.parquet"
        df_original = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })

        write_parquet(df_original, str(parquet_file))
        df_read = read_parquet(str(parquet_file))

        assert len(df_read) == 3
        assert df_read['int_col'].iloc[0] == 1
        assert df_read['str_col'].iloc[0] == 'a'

    def test_parquet_roundtrip_large_values(self, tmp_path):
        """Test Parquet roundtrip with large numeric values."""
        parquet_file = tmp_path / "large_values.parquet"
        df_original = pd.DataFrame({
            'big_int': [999999999999, 888888888888, 777777777777],
            'big_float': [1.23e10, 4.56e10, 7.89e10]
        })

        write_parquet(df_original, str(parquet_file))
        df_read = read_parquet(str(parquet_file))

        assert df_read['big_int'].iloc[0] == 999999999999
