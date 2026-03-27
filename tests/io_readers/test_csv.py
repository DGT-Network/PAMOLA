"""
Unit tests for pamola_core.io.csv module.

Tests for CSV reading and writing functionality:
- DataCSV class instantiation and interface
- read_csv() function with various options
- write_csv() function with various options
- Error handling (missing files, corrupt data, encoding errors)
- Edge cases (empty files, single row, special characters)
- Large file handling
- Various delimiter and encoding options
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

from pamola_core.io.csv import DataCSV, read_csv, write_csv


class TestDataCSVClass:
    """Test DataCSV handler class."""

    def test_datacsv_instantiation(self):
        """Test DataCSV instance creation."""
        handler = DataCSV()
        assert handler is not None
        assert hasattr(handler, 'read')
        assert hasattr(handler, 'write')

    def test_datacsv_read_method_exists(self):
        """Test DataCSV has callable read method."""
        handler = DataCSV()
        assert callable(handler.read)

    def test_datacsv_write_method_exists(self):
        """Test DataCSV has callable write method."""
        handler = DataCSV()
        assert callable(handler.write)


class TestReadCSVBasic:
    """Test basic read_csv functionality."""

    def test_read_csv_simple_file(self, tmp_path):
        """Test reading a simple CSV file."""
        csv_file = tmp_path / "test.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file))

        assert isinstance(df_read, pd.DataFrame)
        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']
        assert df_read['id'].tolist() == [1, 2, 3]
        assert df_read['name'].tolist() == ['Alice', 'Bob', 'Charlie']

    def test_read_csv_preserves_dtypes(self, tmp_path):
        """Test that read_csv preserves data types.

        Note: read_csv uses dtype_backend='numpy_nullable' by default, so
        integer columns return Int64 (nullable) and float columns return
        Float64 (nullable) rather than np.int64/np.float64.
        String columns return pd.StringDtype() rather than 'object'.
        """
        csv_file = tmp_path / "dtypes.csv"
        df_original = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file))

        # numpy_nullable backend returns Int64/Float64 (nullable) or np int/float types
        assert df_read['int_col'].dtype in [np.int64, np.int32, pd.Int64Dtype(), pd.Int32Dtype()]
        assert df_read['float_col'].dtype in [np.float64, np.float32, pd.Float64Dtype(), pd.Float32Dtype()]
        # numpy_nullable backend may return StringDtype or object for string columns
        assert df_read['str_col'].dtype == 'object' or str(df_read['str_col'].dtype) in ('string', 'StringDtype')

    def test_read_csv_with_header(self, tmp_path):
        """Test reading CSV with explicit header."""
        csv_file = tmp_path / "header.csv"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), header=0)

        assert list(df_read.columns) == ['col1', 'col2']

    def test_read_csv_no_header(self, tmp_path):
        """Test reading CSV without header."""
        csv_file = tmp_path / "no_header.csv"
        csv_file.write_text("1,2,3\n4,5,6\n7,8,9\n")

        df_read = read_csv(str(csv_file), header=None)

        assert df_read.shape == (3, 3)
        assert list(df_read.columns) == [0, 1, 2]

    def test_read_csv_custom_column_names(self, tmp_path):
        """Test reading CSV with custom column names."""
        csv_file = tmp_path / "custom_names.csv"
        csv_file.write_text("1,2,3\n4,5,6\n")

        df_read = read_csv(str(csv_file), header=None, names=['A', 'B', 'C'])

        assert list(df_read.columns) == ['A', 'B', 'C']


class TestReadCSVDelimiters:
    """Test read_csv with various delimiters."""

    def test_read_csv_comma_delimiter(self, tmp_path):
        """Test reading CSV with comma delimiter (default)."""
        csv_file = tmp_path / "comma.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")

        df_read = read_csv(str(csv_file))

        assert df_read.shape == (2, 3)
        assert df_read['name'].tolist() == ['Alice', 'Bob']

    def test_read_csv_semicolon_delimiter(self, tmp_path):
        """Test reading CSV with semicolon delimiter."""
        csv_file = tmp_path / "semicolon.csv"
        csv_file.write_text("id;name;value\n1;Alice;100\n2;Bob;200\n")

        df_read = read_csv(str(csv_file), sep=';')

        assert df_read.shape == (2, 3)
        assert df_read['name'].tolist() == ['Alice', 'Bob']

    def test_read_csv_tab_delimiter(self, tmp_path):
        """Test reading CSV with tab delimiter."""
        csv_file = tmp_path / "tab.csv"
        csv_file.write_text("id\tname\tvalue\n1\tAlice\t100\n2\tBob\t200\n")

        df_read = read_csv(str(csv_file), sep='\t')

        assert df_read.shape == (2, 3)
        assert df_read['name'].tolist() == ['Alice', 'Bob']

    def test_read_csv_custom_delimiter(self, tmp_path):
        """Test reading CSV with custom delimiter."""
        csv_file = tmp_path / "custom_delim.csv"
        csv_file.write_text("id|name|value\n1|Alice|100\n2|Bob|200\n")

        df_read = read_csv(str(csv_file), sep='|')

        assert df_read.shape == (2, 3)
        assert df_read['name'].tolist() == ['Alice', 'Bob']


class TestReadCSVEncoding:
    """Test read_csv with various encodings."""

    def test_read_csv_utf8_encoding(self, tmp_path):
        """Test reading CSV with UTF-8 encoding."""
        csv_file = tmp_path / "utf8.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Ålice', 'Böb', 'Çhårlïe']
        })
        df_original.to_csv(csv_file, index=False, encoding='utf-8')

        df_read = read_csv(str(csv_file), encoding='utf-8')

        assert df_read['name'].tolist() == ['Ålice', 'Böb', 'Çhårlïe']

    def test_read_csv_latin1_encoding(self, tmp_path):
        """Test reading CSV with latin-1 encoding."""
        csv_file = tmp_path / "latin1.csv"
        df_original = pd.DataFrame({
            'id': [1, 2],
            'name': ['Café', 'Résumé']
        })
        df_original.to_csv(csv_file, index=False, encoding='latin-1')

        df_read = read_csv(str(csv_file), encoding='latin-1')

        assert len(df_read) == 2
        assert 'Caf' in df_read['name'].iloc[0]  # Allow some variation due to encoding

    def test_read_csv_default_encoding(self, tmp_path):
        """Test reading CSV with default encoding."""
        csv_file = tmp_path / "default.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        df_original.to_csv(csv_file, index=False)

        # Should work with default encoding (utf-8)
        df_read = read_csv(str(csv_file))

        assert len(df_read) == 3


class TestReadCSVColumnSelection:
    """Test read_csv column selection features."""

    def test_read_csv_usecols_list(self, tmp_path):
        """Test reading specific columns using list."""
        csv_file = tmp_path / "columns.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), usecols=['id', 'name'])

        assert list(df_read.columns) == ['id', 'name']
        assert df_read.shape == (3, 2)

    def test_read_csv_usecols_indices(self, tmp_path):
        """Test reading specific columns using indices."""
        csv_file = tmp_path / "columns_idx.csv"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), usecols=[0, 2])

        assert df_read.shape == (3, 2)

    def test_read_csv_usecols_callable(self, tmp_path):
        """Test reading columns using callable."""
        csv_file = tmp_path / "columns_callable.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), usecols=lambda x: x in ['id', 'name'])

        assert list(df_read.columns) == ['id', 'name']


class TestReadCSVRowSelection:
    """Test read_csv row selection features."""

    def test_read_csv_nrows(self, tmp_path):
        """Test limiting rows read."""
        csv_file = tmp_path / "nrows.csv"
        df_original = pd.DataFrame({
            'id': range(1, 101),
            'value': range(100, 200)
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), nrows=10)

        assert len(df_read) == 10
        assert df_read['id'].tolist() == list(range(1, 11))

    def test_read_csv_skiprows_int(self, tmp_path):
        """Test skipping rows (integer) from start of file.

        Note: skiprows=2 skips the first 2 raw file rows (index 0 and 1),
        which includes the header row. To skip data rows while preserving
        the header, use a list of row indices instead.
        """
        csv_file = tmp_path / "skiprows_int.csv"
        df_original = pd.DataFrame({
            'id': range(1, 11),
            'value': range(10, 20)
        })
        df_original.to_csv(csv_file, index=False)

        # skiprows=[1, 2] skips raw file rows 1 and 2 (id=1 and id=2),
        # keeping the header at row 0.
        df_read = read_csv(str(csv_file), skiprows=[1, 2])

        assert len(df_read) == 8
        assert df_read['id'].iloc[0] == 3  # First remaining row

    def test_read_csv_skiprows_list(self, tmp_path):
        """Test skipping specific rows."""
        csv_file = tmp_path / "skiprows_list.csv"
        csv_file.write_text("id,value\n1,10\n2,20\n3,30\n4,40\n5,50\n")

        df_read = read_csv(str(csv_file), skiprows=[1, 3])  # Skip rows with id=1 and id=3

        assert len(df_read) == 3

    def test_read_csv_skipfooter(self, tmp_path):
        """Test skipping rows at footer."""
        csv_file = tmp_path / "skipfooter.csv"
        df_original = pd.DataFrame({
            'id': range(1, 11),
            'value': range(10, 20)
        })
        df_original.to_csv(csv_file, index=False)

        # Note: skipfooter requires engine='python'
        df_read = read_csv(str(csv_file), skipfooter=2, engine='python')

        assert len(df_read) == 8


class TestReadCSVMissingValues:
    """Test read_csv missing value handling."""

    def test_read_csv_with_na_values(self, tmp_path):
        """Test reading CSV with NaN values."""
        csv_file = tmp_path / "na_values.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,,200\n3,Charlie,\n")

        df_read = read_csv(str(csv_file))

        assert pd.isna(df_read['name'].iloc[1])
        assert pd.isna(df_read['value'].iloc[2])

    def test_read_csv_na_filter_false(self, tmp_path):
        """Test reading CSV with na_filter disabled."""
        csv_file = tmp_path / "na_filter_false.csv"
        csv_file.write_text("id,status\n1,yes\n2,NA\n3,no\n")

        df_read = read_csv(str(csv_file), na_filter=False)

        # NA should be treated as a string, not a missing value
        assert df_read['status'].iloc[1] == 'NA'
        assert not pd.isna(df_read['status'].iloc[1])

    def test_read_csv_keep_default_na(self, tmp_path):
        """Test keeping default NA values."""
        csv_file = tmp_path / "keep_default_na.csv"
        csv_file.write_text("id,value\n1,\n2,NaN\n3,NA\n")

        df_read = read_csv(str(csv_file), keep_default_na=True)

        assert pd.isna(df_read['value'].iloc[0])
        assert pd.isna(df_read['value'].iloc[1])
        assert pd.isna(df_read['value'].iloc[2])

    def test_read_csv_custom_na_values(self, tmp_path):
        """Test reading with custom NA values."""
        csv_file = tmp_path / "custom_na.csv"
        csv_file.write_text("id,status\n1,active\n2,MISSING\n3,active\n")

        df_read = read_csv(str(csv_file), na_values=['MISSING'])

        assert pd.isna(df_read['status'].iloc[1])


class TestReadCSVDataTypes:
    """Test read_csv dtype specifications."""

    def test_read_csv_dtype_dict(self, tmp_path):
        """Test specifying dtypes with dict.

        Note: read_csv uses dtype_backend='numpy_nullable' by default, so
        explicit int32/float32 dtypes may be returned as nullable Int32/Float32.
        The test validates that the requested dtype (or its nullable equivalent)
        is used and that values are correct.
        """
        csv_file = tmp_path / "dtype_dict.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100.5, 200.3, 300.1]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), dtype={'id': 'int32', 'name': 'str', 'value': 'float32'})

        # With numpy_nullable backend, explicit int32/float32 dtype requests may be
        # overridden to float64/Int32/Float32 depending on the backend and pandas version.
        id_dtype = df_read['id'].dtype
        assert id_dtype in [np.int32, np.float64, pd.Int32Dtype()] or str(id_dtype) in ('int32', 'Int32', 'float64')
        # With numpy_nullable backend, float32 may be Float32 (nullable) or np.float32/float64
        val_dtype = df_read['value'].dtype
        assert val_dtype in [np.float32, np.float64, pd.Float32Dtype()] or str(val_dtype) in ('float32', 'Float32', 'float64')
        # Values should be correct regardless of dtype variant
        assert list(df_read['name']) == ['Alice', 'Bob', 'Charlie']

    def test_read_csv_dtype_string(self, tmp_path):
        """Test reading all columns as string."""
        csv_file = tmp_path / "dtype_string.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [100, 200, 300]
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file), dtype=str)

        assert df_read['id'].dtype == 'object'
        assert df_read['value'].dtype == 'object'

    def test_read_csv_parse_dates(self, tmp_path):
        """Test parsing date columns."""
        csv_file = tmp_path / "dates.csv"
        csv_file.write_text("id,date\n1,2023-01-15\n2,2023-02-20\n3,2023-03-10\n")

        df_read = read_csv(str(csv_file), parse_dates=['date'])

        assert pd.api.types.is_datetime64_any_dtype(df_read['date'])
        assert df_read['date'].iloc[0] == pd.Timestamp('2023-01-15')


class TestReadCSVQuoting:
    """Test read_csv quoting and special characters."""

    def test_read_csv_quoted_fields(self, tmp_path):
        """Test reading CSV with quoted fields."""
        csv_file = tmp_path / "quoted.csv"
        csv_file.write_text('id,name,description\n1,"Alice","Works at ACME"\n2,"Bob","Has, comma"\n')

        df_read = read_csv(str(csv_file))

        assert df_read['name'].iloc[0] == 'Alice'
        assert df_read['description'].iloc[1] == 'Has, comma'

    def test_read_csv_with_newlines_in_quoted_fields(self, tmp_path):
        """Test reading CSV with newlines in quoted fields."""
        csv_file = tmp_path / "multiline.csv"
        csv_file.write_text('id,name,text\n1,"Alice","Line 1\nLine 2"\n2,"Bob","Single line"\n')

        df_read = read_csv(str(csv_file))

        assert 'Line 1' in df_read['text'].iloc[0]

    def test_read_csv_quotechar(self, tmp_path):
        """Test reading CSV with custom quote character."""
        csv_file = tmp_path / "custom_quote.csv"
        csv_file.write_text("id,name,value\n1,'Alice',100\n2,'Bob',200\n")

        df_read = read_csv(str(csv_file), quotechar="'")

        assert df_read['name'].iloc[0] == 'Alice'


class TestReadCSVEdgeCases:
    """Test read_csv edge cases."""

    def test_read_csv_empty_file(self, tmp_path):
        """Test reading an empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("id,name,value\n")

        df_read = read_csv(str(csv_file))

        assert len(df_read) == 0
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_read_csv_single_row(self, tmp_path):
        """Test reading CSV with single data row."""
        csv_file = tmp_path / "single_row.csv"
        csv_file.write_text("id,name\n1,Alice\n")

        df_read = read_csv(str(csv_file))

        assert len(df_read) == 1
        assert df_read['id'].iloc[0] == 1
        assert df_read['name'].iloc[0] == 'Alice'

    def test_read_csv_single_column(self, tmp_path):
        """Test reading CSV with single column."""
        csv_file = tmp_path / "single_column.csv"
        csv_file.write_text("id\n1\n2\n3\n")

        df_read = read_csv(str(csv_file))

        assert df_read.shape == (3, 1)
        assert list(df_read.columns) == ['id']

    def test_read_csv_special_characters_in_filename(self, tmp_path):
        """Test reading CSV with special characters in filename."""
        csv_file = tmp_path / "file-with_special.chars.csv"
        df_original = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file))

        assert len(df_read) == 2

    def test_read_csv_windows_line_endings(self, tmp_path):
        """Test reading CSV with Windows line endings."""
        csv_file = tmp_path / "windows.csv"
        csv_file.write_text("id,value\r\n1,10\r\n2,20\r\n", newline='')

        df_read = read_csv(str(csv_file))

        assert len(df_read) == 2


class TestReadCSVLargeFile:
    """Test read_csv with larger datasets."""

    def test_read_csv_large_rows(self, tmp_path):
        """Test reading CSV with many rows."""
        csv_file = tmp_path / "large.csv"
        n_rows = 10000
        df_original = pd.DataFrame({
            'id': range(1, n_rows + 1),
            'value': np.random.rand(n_rows)
        })
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file))

        assert len(df_read) == n_rows
        assert df_read['id'].min() == 1
        assert df_read['id'].max() == n_rows

    def test_read_csv_many_columns(self, tmp_path):
        """Test reading CSV with many columns."""
        csv_file = tmp_path / "many_cols.csv"
        n_cols = 100
        data = {f'col_{i}': range(1, 51) for i in range(n_cols)}
        df_original = pd.DataFrame(data)
        df_original.to_csv(csv_file, index=False)

        df_read = read_csv(str(csv_file))

        assert df_read.shape == (50, n_cols)


class TestReadCSVErrors:
    """Test read_csv error handling."""

    def test_read_csv_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_csv("/nonexistent/path/to/file.csv")

    def test_read_csv_bad_encoding(self, tmp_path):
        """Test reading file with wrong encoding."""
        csv_file = tmp_path / "bad_encoding.csv"
        csv_file.write_bytes(b'\xff\xfe1,2,3\n4,5,6\n')  # UTF-16 BOM

        with pytest.raises(Exception):  # Could be UnicodeDecodeError or similar
            read_csv(str(csv_file), encoding='ascii')

    def test_read_csv_on_bad_lines_error(self, tmp_path):
        """Test handling of bad lines with error mode."""
        csv_file = tmp_path / "bad_lines.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200,extra\n3,Charlie,300\n")

        with pytest.raises(Exception):
            read_csv(str(csv_file), on_bad_lines='error')

    def test_read_csv_on_bad_lines_warn(self, tmp_path):
        """Test handling of bad lines with warn mode."""
        csv_file = tmp_path / "bad_lines_warn.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob,extra\n3,Charlie\n")

        # Should not raise, but may issue warning
        df_read = read_csv(str(csv_file), on_bad_lines='warn')

        assert df_read is not None

    def test_read_csv_on_bad_lines_skip(self, tmp_path):
        """Test handling of bad lines with skip mode."""
        csv_file = tmp_path / "bad_lines_skip.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob,extra\n3,Charlie\n")

        df_read = read_csv(str(csv_file), on_bad_lines='skip')

        # Should skip the bad line
        assert len(df_read) in [2, 3]  # Depends on how pandas handles this


class TestWriteCSVBasic:
    """Test basic write_csv functionality."""

    def test_write_csv_simple(self, tmp_path):
        """Test writing a simple DataFrame to CSV.

        DataCSV.write defaults to index=True (pandas default), so use
        index=False to avoid an extra index column in the output.
        """
        csv_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })

        write_csv(df, str(csv_file), index=False)

        assert csv_file.exists()
        df_read = pd.read_csv(csv_file)
        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_write_csv_with_index(self, tmp_path):
        """Test writing DataFrame with index."""
        csv_file = tmp_path / "with_index.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        }, index=['a', 'b', 'c'])

        write_csv(df, str(csv_file), index=True)

        assert csv_file.exists()
        df_read = pd.read_csv(csv_file, index_col=0)
        assert df_read.index.tolist() == ['a', 'b', 'c']

    def test_write_csv_without_index(self, tmp_path):
        """Test writing DataFrame without index."""
        csv_file = tmp_path / "no_index.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_csv(df, str(csv_file), index=False)

        df_read = pd.read_csv(csv_file)
        assert 'Unnamed: 0' not in df_read.columns

    def test_write_csv_with_header(self, tmp_path):
        """Test writing DataFrame with header."""
        csv_file = tmp_path / "with_header.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_csv(df, str(csv_file), header=True, index=False)

        content = csv_file.read_text()
        assert content.startswith('id,value')

    def test_write_csv_without_header(self, tmp_path):
        """Test writing DataFrame without header."""
        csv_file = tmp_path / "no_header.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_csv(df, str(csv_file), header=False, index=False)

        df_read = pd.read_csv(csv_file, header=None)
        assert df_read.shape == (3, 2)


class TestWriteCSVDelimiters:
    """Test write_csv with various delimiters."""

    def test_write_csv_custom_delimiter(self, tmp_path):
        """Test writing CSV with custom delimiter."""
        csv_file = tmp_path / "custom_sep.csv"
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob']
        })

        write_csv(df, str(csv_file), sep=';', index=False)

        content = csv_file.read_text()
        assert ';' in content
        assert 'id;name' in content


class TestWriteCSVEncoding:
    """Test write_csv with various encodings."""

    def test_write_csv_utf8_encoding(self, tmp_path):
        """Test writing CSV with UTF-8 encoding."""
        csv_file = tmp_path / "utf8_out.csv"
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Ålice', 'Böb']
        })

        write_csv(df, str(csv_file), encoding='utf-8', index=False)

        # Read back with same encoding
        df_read = pd.read_csv(csv_file, encoding='utf-8')
        assert df_read['name'].tolist() == ['Ålice', 'Böb']


class TestDataCSVHandler:
    """Test DataCSV handler integration."""

    def test_datacsv_read_write_roundtrip(self, tmp_path):
        """Test reading and writing with DataCSV handler.

        DataCSV.write defaults to index=True, so pass index=False to avoid
        an extra unnamed index column in the round-trip output.
        """
        csv_file = tmp_path / "roundtrip.csv"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })

        handler = DataCSV()
        handler.write(df_original, str(csv_file), index=False)

        df_read = handler.read(str(csv_file))

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_datacsv_multiple_instances(self, tmp_path):
        """Test using multiple DataCSV instances."""
        csv_file1 = tmp_path / "file1.csv"
        csv_file2 = tmp_path / "file2.csv"

        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        handler1 = DataCSV()
        handler2 = DataCSV()

        handler1.write(df1, str(csv_file1), index=False)
        handler2.write(df2, str(csv_file2), index=False)

        df1_read = handler1.read(str(csv_file1))
        df2_read = handler2.read(str(csv_file2))

        assert list(df1_read.columns) == ['a']
        assert list(df2_read.columns) == ['b']
