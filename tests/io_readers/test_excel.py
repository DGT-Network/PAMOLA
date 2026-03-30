"""
Unit tests for pamola_core.io.excel module.

Tests for Excel reading and writing functionality:
- DataExcel class instantiation and interface
- read_excel() function with various options
- write_excel() function with various options
- Multiple sheets
- Header handling
- Data type specifications
- Missing file error handling
- Edge cases (empty sheets, single row/column)
"""

import pytest
import pandas as pd
import numpy as np

from pamola_core.io.excel import DataExcel, read_excel, write_excel

# Skip tests if openpyxl is not available
pytest.importorskip("openpyxl")


class TestDataExcelClass:
    """Test DataExcel handler class."""

    def test_dataexcel_instantiation(self):
        """Test DataExcel instance creation."""
        handler = DataExcel()
        assert handler is not None
        assert hasattr(handler, 'read')
        assert hasattr(handler, 'write')

    def test_dataexcel_read_method_exists(self):
        """Test DataExcel has callable read method."""
        handler = DataExcel()
        assert callable(handler.read)

    def test_dataexcel_write_method_exists(self):
        """Test DataExcel has callable write method."""
        handler = DataExcel()
        assert callable(handler.write)


class TestReadExcelBasic:
    """Test basic read_excel functionality."""

    def test_read_excel_simple_file(self, tmp_path):
        """Test reading a simple Excel file."""
        xlsx_file = tmp_path / "test.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert isinstance(df_read, pd.DataFrame)
        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_read_excel_default_sheet(self, tmp_path):
        """Test reading Excel file with default sheet (first sheet)."""
        xlsx_file = tmp_path / "default_sheet.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        df_original.to_excel(xlsx_file, sheet_name='Sheet1', index=False)

        df_read = read_excel(str(xlsx_file))

        assert len(df_read) == 3

    def test_read_excel_with_header(self, tmp_path):
        """Test reading Excel with explicit header."""
        xlsx_file = tmp_path / "header.xlsx"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), header=0)

        assert list(df_read.columns) == ['col1', 'col2']

    def test_read_excel_no_header(self, tmp_path):
        """Test reading Excel without header."""
        xlsx_file = tmp_path / "no_header.xlsx"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), header=None)

        assert df_read.shape[0] >= 3  # At least data rows

    def test_read_excel_custom_column_names(self, tmp_path):
        """Test reading Excel with custom column names."""
        xlsx_file = tmp_path / "custom_names.xlsx"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), names=['A', 'B'])

        assert list(df_read.columns) == ['A', 'B']


class TestReadExcelMultipleSheets:
    """Test read_excel with multiple sheets."""

    def test_read_excel_specific_sheet_by_name(self, tmp_path):
        """Test reading specific sheet by name."""
        xlsx_file = tmp_path / "multiple_sheets.xlsx"

        with pd.ExcelWriter(xlsx_file) as writer:
            pd.DataFrame({'id': [1, 2], 'value': [10, 20]}).to_excel(
                writer, sheet_name='Sheet1', index=False
            )
            pd.DataFrame({'name': ['A', 'B'], 'score': [100, 90]}).to_excel(
                writer, sheet_name='Sheet2', index=False
            )

        df_read = read_excel(str(xlsx_file), sheet_name='Sheet2')

        assert list(df_read.columns) == ['name', 'score']
        assert len(df_read) == 2

    def test_read_excel_specific_sheet_by_index(self, tmp_path):
        """Test reading specific sheet by index."""
        xlsx_file = tmp_path / "sheet_by_index.xlsx"

        with pd.ExcelWriter(xlsx_file) as writer:
            pd.DataFrame({'id': [1, 2], 'value': [10, 20]}).to_excel(
                writer, sheet_name='Sheet1', index=False
            )
            pd.DataFrame({'name': ['A', 'B'], 'score': [100, 90]}).to_excel(
                writer, sheet_name='Sheet2', index=False
            )

        df_read = read_excel(str(xlsx_file), sheet_name=1)

        assert list(df_read.columns) == ['name', 'score']

    def test_read_excel_all_sheets(self, tmp_path):
        """Test reading all sheets as dictionary."""
        xlsx_file = tmp_path / "all_sheets.xlsx"

        with pd.ExcelWriter(xlsx_file) as writer:
            pd.DataFrame({'id': [1, 2]}).to_excel(writer, sheet_name='Sheet1', index=False)
            pd.DataFrame({'name': ['A', 'B']}).to_excel(writer, sheet_name='Sheet2', index=False)

        sheets = read_excel(str(xlsx_file), sheet_name=None)

        assert isinstance(sheets, dict)
        assert 'Sheet1' in sheets
        assert 'Sheet2' in sheets

    def test_read_excel_multiple_sheets_list(self, tmp_path):
        """Test reading multiple sheets by list."""
        xlsx_file = tmp_path / "multi_sheets_list.xlsx"

        with pd.ExcelWriter(xlsx_file) as writer:
            pd.DataFrame({'id': [1, 2]}).to_excel(writer, sheet_name='Sheet1', index=False)
            pd.DataFrame({'name': ['A', 'B']}).to_excel(writer, sheet_name='Sheet2', index=False)
            pd.DataFrame({'value': [10, 20]}).to_excel(writer, sheet_name='Sheet3', index=False)

        sheets = read_excel(str(xlsx_file), sheet_name=[0, 2])

        assert isinstance(sheets, dict)
        assert len(sheets) == 2


class TestReadExcelColumnSelection:
    """Test read_excel column selection."""

    def test_read_excel_usecols_list(self, tmp_path):
        """Test reading specific columns using list."""
        xlsx_file = tmp_path / "usecols_list.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), usecols=['id', 'name'])

        assert list(df_read.columns) == ['id', 'name']

    def test_read_excel_usecols_range(self, tmp_path):
        """Test reading column range."""
        xlsx_file = tmp_path / "usecols_range.xlsx"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'col4': [10, 11, 12]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), usecols='A:C')

        assert df_read.shape[1] >= 3

    def test_read_excel_usecols_indices(self, tmp_path):
        """Test reading columns by indices."""
        xlsx_file = tmp_path / "usecols_idx.xlsx"
        df_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), usecols=[0, 2])

        assert df_read.shape[1] == 2


class TestReadExcelRowSelection:
    """Test read_excel row selection."""

    def test_read_excel_nrows(self, tmp_path):
        """Test limiting rows read."""
        xlsx_file = tmp_path / "nrows.xlsx"
        df_original = pd.DataFrame({
            'id': range(1, 101),
            'value': range(100, 200)
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), nrows=10)

        assert len(df_read) == 10

    def test_read_excel_skiprows_int(self, tmp_path):
        """Test skipping rows (integer)."""
        xlsx_file = tmp_path / "skiprows_int.xlsx"
        df_original = pd.DataFrame({
            'id': range(1, 11),
            'value': range(10, 20)
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), skiprows=2)

        assert len(df_read) <= 10

    def test_read_excel_skipfooter(self, tmp_path):
        """Test skipping rows at footer."""
        xlsx_file = tmp_path / "skipfooter.xlsx"
        df_original = pd.DataFrame({
            'id': range(1, 11),
            'value': range(10, 20)
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), skipfooter=2)

        assert len(df_read) <= 9


class TestReadExcelDataTypes:
    """Test read_excel dtype handling."""

    def test_read_excel_dtype_dict(self, tmp_path):
        """Test specifying dtypes with dict.

        Note: read_excel uses dtype_backend='numpy_nullable' by default, so
        integer columns may return Int64 (nullable) rather than np.int64.
        """
        xlsx_file = tmp_path / "dtype_dict.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100.5, 200.3, 300.1]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), dtype={'id': int, 'name': str})

        assert df_read['id'].dtype in [np.int64, np.int32, np.int16, pd.Int64Dtype(), pd.Int32Dtype()]

    def test_read_excel_parse_dates(self, tmp_path):
        """Test parsing date columns."""
        xlsx_file = tmp_path / "dates.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10'])
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), parse_dates=['date'])

        assert pd.api.types.is_datetime64_any_dtype(df_read['date'])


class TestReadExcelMissingValues:
    """Test read_excel missing value handling."""

    def test_read_excel_with_na_values(self, tmp_path):
        """Test reading Excel with NaN values."""
        xlsx_file = tmp_path / "na_values.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', None, 'Charlie'],
            'value': [100, 200, None]
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert pd.isna(df_read['name'].iloc[1])
        assert pd.isna(df_read['value'].iloc[2])

    def test_read_excel_na_filter(self, tmp_path):
        """Test NA filtering."""
        xlsx_file = tmp_path / "na_filter.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'status': ['active', None, 'active']
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file), na_filter=True)

        assert pd.isna(df_read['status'].iloc[1])


class TestReadExcelEdgeCases:
    """Test read_excel edge cases."""

    def test_read_excel_empty_sheet(self, tmp_path):
        """Test reading an empty sheet."""
        xlsx_file = tmp_path / "empty.xlsx"

        with pd.ExcelWriter(xlsx_file) as writer:
            pd.DataFrame({'col1': [], 'col2': []}).to_excel(
                writer, sheet_name='Sheet1', index=False
            )

        df_read = read_excel(str(xlsx_file))

        assert len(df_read) == 0

    def test_read_excel_single_row(self, tmp_path):
        """Test reading Excel with single data row."""
        xlsx_file = tmp_path / "single_row.xlsx"
        df_original = pd.DataFrame({'id': [1], 'name': ['Alice']})
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert len(df_read) == 1

    def test_read_excel_single_column(self, tmp_path):
        """Test reading Excel with single column."""
        xlsx_file = tmp_path / "single_column.xlsx"
        df_original = pd.DataFrame({'id': [1, 2, 3]})
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert df_read.shape[1] == 1

    def test_read_excel_with_index_column(self, tmp_path):
        """Test reading Excel with index as column."""
        xlsx_file = tmp_path / "with_index.xlsx"
        df_original = pd.DataFrame(
            {'name': ['Alice', 'Bob'], 'value': [100, 200]},
            index=['a', 'b']
        )
        df_original.to_excel(xlsx_file)

        df_read = read_excel(str(xlsx_file), index_col=0)

        assert df_read.index.tolist() == ['a', 'b']

    def test_read_excel_special_characters(self, tmp_path):
        """Test reading Excel with special characters."""
        xlsx_file = tmp_path / "special_chars.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['Hello@World', 'Test#123', 'Value$99']
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert df_read['text'].iloc[0] == 'Hello@World'


class TestReadExcelLargeFile:
    """Test read_excel with larger datasets."""

    def test_read_excel_large_rows(self, tmp_path):
        """Test reading Excel with many rows."""
        xlsx_file = tmp_path / "large.xlsx"
        n_rows = 5000
        df_original = pd.DataFrame({
            'id': range(1, n_rows + 1),
            'value': np.random.rand(n_rows)
        })
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert len(df_read) == n_rows

    def test_read_excel_many_columns(self, tmp_path):
        """Test reading Excel with many columns."""
        xlsx_file = tmp_path / "many_cols.xlsx"
        n_cols = 50
        data = {f'col_{i}': range(1, 51) for i in range(n_cols)}
        df_original = pd.DataFrame(data)
        df_original.to_excel(xlsx_file, index=False)

        df_read = read_excel(str(xlsx_file))

        assert df_read.shape[1] == n_cols


class TestReadExcelErrors:
    """Test read_excel error handling."""

    def test_read_excel_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_excel("/nonexistent/path/to/file.xlsx")

    def test_read_excel_invalid_sheet_name(self, tmp_path):
        """Test reading with invalid sheet name."""
        xlsx_file = tmp_path / "sheets.xlsx"

        with pd.ExcelWriter(xlsx_file) as writer:
            pd.DataFrame({'id': [1, 2]}).to_excel(writer, sheet_name='Sheet1', index=False)

        with pytest.raises(Exception):  # ValueError or KeyError
            read_excel(str(xlsx_file), sheet_name='NonExistent')


class TestWriteExcelBasic:
    """Test basic write_excel functionality."""

    def test_write_excel_simple(self, tmp_path):
        """Test writing a simple DataFrame to Excel."""
        xlsx_file = tmp_path / "output.xlsx"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })

        write_excel(df, str(xlsx_file))

        assert xlsx_file.exists()
        df_read = pd.read_excel(xlsx_file)
        assert len(df_read) == 3

    def test_write_excel_with_sheet_name(self, tmp_path):
        """Test writing Excel with custom sheet name."""
        xlsx_file = tmp_path / "custom_sheet.xlsx"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_excel(df, str(xlsx_file), sheet_name='MyData')

        # Verify sheet name
        sheets = pd.read_excel(xlsx_file, sheet_name=None)
        assert 'MyData' in sheets

    def test_write_excel_with_index(self, tmp_path):
        """Test writing Excel with index."""
        xlsx_file = tmp_path / "with_index.xlsx"
        df = pd.DataFrame(
            {'name': ['Alice', 'Bob'], 'value': [100, 200]},
            index=['a', 'b']
        )

        write_excel(df, str(xlsx_file), index=True)

        df_read = pd.read_excel(xlsx_file, index_col=0)
        assert df_read.index.tolist() == ['a', 'b']

    def test_write_excel_without_index(self, tmp_path):
        """Test writing Excel without index."""
        xlsx_file = tmp_path / "no_index.xlsx"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_excel(df, str(xlsx_file), index=False)

        df_read = pd.read_excel(xlsx_file)
        assert 'Unnamed: 0' not in df_read.columns

    def test_write_excel_with_header(self, tmp_path):
        """Test writing Excel with header."""
        xlsx_file = tmp_path / "with_header.xlsx"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_excel(df, str(xlsx_file), header=True, index=False)

        df_read = pd.read_excel(xlsx_file)
        assert list(df_read.columns) == ['id', 'value']

    def test_write_excel_without_header(self, tmp_path):
        """Test writing Excel without header."""
        xlsx_file = tmp_path / "no_header.xlsx"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        write_excel(df, str(xlsx_file), header=False, index=False)

        df_read = pd.read_excel(xlsx_file, header=None)
        assert df_read.shape == (3, 2)


class TestWriteExcelFormatting:
    """Test write_excel formatting options."""

    def test_write_excel_start_position(self, tmp_path):
        """Test writing Excel with custom start position."""
        xlsx_file = tmp_path / "start_pos.xlsx"
        df = pd.DataFrame({
            'id': [1, 2],
            'value': [10, 20]
        })

        write_excel(df, str(xlsx_file), startrow=2, startcol=1, index=False)

        assert xlsx_file.exists()

    def test_write_excel_columns_subset(self, tmp_path):
        """Test writing specific columns."""
        xlsx_file = tmp_path / "columns.xlsx"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })

        write_excel(df, str(xlsx_file), columns=['id', 'name'], index=False)

        df_read = pd.read_excel(xlsx_file)
        assert list(df_read.columns) == ['id', 'name']


class TestWriteExcelMultipleSheets:
    """Test writing multiple sheets to Excel."""

    def test_write_excel_multiple_sheets(self, tmp_path):
        """Test writing multiple sheets using ExcelWriter."""
        xlsx_file = tmp_path / "multiple_sheets.xlsx"

        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'name': ['A', 'B'], 'score': [100, 90]})

        with pd.ExcelWriter(xlsx_file) as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)

        sheets = pd.read_excel(xlsx_file, sheet_name=None)
        assert len(sheets) == 2


class TestDataExcelHandler:
    """Test DataExcel handler integration."""

    def test_dataexcel_read_write_roundtrip(self, tmp_path):
        """Test reading and writing with DataExcel handler.

        DataExcel.write defaults to index=True, so pass index=False to avoid
        an extra unnamed index column in the round-trip output.
        """
        xlsx_file = tmp_path / "roundtrip.xlsx"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })

        handler = DataExcel()
        handler.write(df_original, str(xlsx_file), index=False)

        df_read = handler.read(str(xlsx_file))

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_dataexcel_multiple_instances(self, tmp_path):
        """Test using multiple DataExcel instances.

        DataExcel.write defaults to index=True, so pass index=False to avoid
        extra unnamed index columns in the output.
        """
        xlsx_file1 = tmp_path / "file1.xlsx"
        xlsx_file2 = tmp_path / "file2.xlsx"

        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        handler1 = DataExcel()
        handler2 = DataExcel()

        handler1.write(df1, str(xlsx_file1), index=False)
        handler2.write(df2, str(xlsx_file2), index=False)

        df1_read = handler1.read(str(xlsx_file1))
        df2_read = handler2.read(str(xlsx_file2))

        assert list(df1_read.columns) == ['a']
        assert list(df2_read.columns) == ['b']
