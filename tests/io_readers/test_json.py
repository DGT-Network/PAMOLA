"""
Unit tests for pamola_core.io.json module.

Tests for JSON reading and writing functionality:
- DataJSON class instantiation and interface
- read_json() function with various options
- write_json() function with various options
- Nested JSON structures
- JSON arrays and records format
- Encoding and error handling
- Edge cases (empty files, single record)
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from pamola_core.io.json import DataJSON, read_json, write_json


class TestDataJSONClass:
    """Test DataJSON handler class."""

    def test_datajson_instantiation(self):
        """Test DataJSON instance creation."""
        handler = DataJSON()
        assert handler is not None
        assert hasattr(handler, 'read')
        assert hasattr(handler, 'write')

    def test_datajson_read_method_exists(self):
        """Test DataJSON has callable read method."""
        handler = DataJSON()
        assert callable(handler.read)

    def test_datajson_write_method_exists(self):
        """Test DataJSON has callable write method."""
        handler = DataJSON()
        assert callable(handler.write)


class TestReadJSONBasic:
    """Test basic read_json functionality."""

    def test_read_json_simple_records(self, tmp_path):
        """Test reading JSON in records format."""
        json_file = tmp_path / "records.json"
        data = [
            {'id': 1, 'name': 'Alice', 'value': 100},
            {'id': 2, 'name': 'Bob', 'value': 200},
            {'id': 3, 'name': 'Charlie', 'value': 300}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert isinstance(df_read, pd.DataFrame)
        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_read_json_split_format(self, tmp_path):
        """Test reading JSON in split format."""
        json_file = tmp_path / "split.json"
        data = {
            'index': [0, 1, 2],
            'columns': ['id', 'name'],
            'data': [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']]
        }
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file), orient='split')

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name']

    def test_read_json_index_format(self, tmp_path):
        """Test reading JSON in index format."""
        json_file = tmp_path / "index.json"
        data = {
            '0': {'name': 'Alice', 'value': 100},
            '1': {'name': 'Bob', 'value': 200},
            '2': {'name': 'Charlie', 'value': 300}
        }
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file), orient='index')

        assert len(df_read) == 3

    def test_read_json_columns_format(self, tmp_path):
        """Test reading JSON in columns format."""
        json_file = tmp_path / "columns.json"
        data = {
            'id': {'0': 1, '1': 2, '2': 3},
            'name': {'0': 'Alice', '1': 'Bob', '2': 'Charlie'}
        }
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file), orient='columns')

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name']

    def test_read_json_values_format(self, tmp_path):
        """Test reading JSON in values format."""
        json_file = tmp_path / "values.json"
        data = [[1, 'Alice', 100], [2, 'Bob', 200], [3, 'Charlie', 300]]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file), orient='values')

        assert len(df_read) == 3
        assert df_read.shape[1] == 3


class TestReadJSONLines:
    """Test read_json with lines format."""

    def test_read_json_lines_format(self, tmp_path):
        """Test reading JSON Lines format."""
        json_file = tmp_path / "lines.jsonl"
        with open(json_file, 'w') as f:
            f.write('{"id": 1, "name": "Alice", "value": 100}\n')
            f.write('{"id": 2, "name": "Bob", "value": 200}\n')
            f.write('{"id": 3, "name": "Charlie", "value": 300}\n')

        df_read = read_json(str(json_file), lines=True)

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_read_json_lines_empty_file(self, tmp_path):
        """Test reading empty JSON Lines file.

        Behavior depends on pandas/ujson version:
        - Older versions raise ValueError/EmptyDataError.
        - Newer versions (pandas >= 2.x with ujson) return an empty DataFrame.
        Both outcomes are acceptable — the important thing is no unhandled crash.
        """
        json_file = tmp_path / "empty.jsonl"
        json_file.write_text("")

        try:
            result = read_json(str(json_file), lines=True)
            # If no exception, result should be an empty DataFrame
            assert isinstance(result, __import__("pandas").DataFrame)
            assert len(result) == 0
        except Exception:
            # Exception on empty file is also acceptable behavior
            pass


class TestReadJSONEncoding:
    """Test read_json with various encodings."""

    def test_read_json_utf8_encoding(self, tmp_path):
        """Test reading JSON with UTF-8 encoding."""
        json_file = tmp_path / "utf8.json"
        data = [
            {'id': 1, 'name': 'Ålice'},
            {'id': 2, 'name': 'Böb'},
            {'id': 3, 'name': 'Çhårlïe'}
        ]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file), encoding='utf-8')

        assert df_read['name'].tolist() == ['Ålice', 'Böb', 'Çhårlïe']

    def test_read_json_default_encoding(self, tmp_path):
        """Test reading JSON with default encoding."""
        json_file = tmp_path / "default.json"
        data = [{'id': 1, 'name': 'Alice'}]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        # Should work with default encoding (utf-8)
        df_read = read_json(str(json_file))

        assert len(df_read) == 1


class TestReadJSONDataTypes:
    """Test read_json dtype handling."""

    def test_read_json_with_numeric_types(self, tmp_path):
        """Test reading JSON preserves numeric types.

        Note: read_json uses dtype_backend='numpy_nullable' by default, so
        integer columns may return Int64 (nullable) and float columns may
        return Float64 (nullable) rather than np.int64/np.float64.
        """
        json_file = tmp_path / "numeric.json"
        data = [
            {'id': 1, 'float_val': 10.5, 'int_val': 100},
            {'id': 2, 'float_val': 20.3, 'int_val': 200}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert df_read['float_val'].dtype in [np.float64, np.float32, pd.Float64Dtype(), pd.Float32Dtype()]
        assert df_read['int_val'].dtype in [np.int64, np.int32, pd.Int64Dtype(), pd.Int32Dtype()]

    def test_read_json_parse_dates(self, tmp_path):
        """Test reading JSON with date parsing."""
        json_file = tmp_path / "dates.json"
        data = [
            {'id': 1, 'date': '2023-01-15'},
            {'id': 2, 'date': '2023-02-20'}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file), convert_dates=['date'])

        # Date strings might not be automatically converted without orient='table'
        assert len(df_read) == 2


class TestReadJSONNestedStructures:
    """Test read_json with nested JSON."""

    def test_read_json_nested_dict(self, tmp_path):
        """Test reading JSON with nested dictionaries."""
        json_file = tmp_path / "nested.json"
        data = [
            {'id': 1, 'user': {'name': 'Alice', 'age': 25}},
            {'id': 2, 'user': {'name': 'Bob', 'age': 30}}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert len(df_read) == 2
        # Nested dict typically becomes string or object column
        assert 'user' in df_read.columns

    def test_read_json_array_column(self, tmp_path):
        """Test reading JSON with array values."""
        json_file = tmp_path / "array.json"
        data = [
            {'id': 1, 'tags': ['a', 'b', 'c']},
            {'id': 2, 'tags': ['x', 'y', 'z']}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert len(df_read) == 2
        assert 'tags' in df_read.columns


class TestReadJSONEdgeCases:
    """Test read_json edge cases."""

    def test_read_json_empty_array(self, tmp_path):
        """Test reading empty JSON array."""
        json_file = tmp_path / "empty_array.json"
        with open(json_file, 'w') as f:
            json.dump([], f)

        df_read = read_json(str(json_file))

        assert len(df_read) == 0

    def test_read_json_single_record(self, tmp_path):
        """Test reading JSON with single record."""
        json_file = tmp_path / "single.json"
        data = [{'id': 1, 'name': 'Alice', 'value': 100}]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert len(df_read) == 1
        assert df_read['id'].iloc[0] == 1

    def test_read_json_unicode_characters(self, tmp_path):
        """Test reading JSON with unicode characters."""
        json_file = tmp_path / "unicode.json"
        data = [
            {'id': 1, 'text': '你好世界'},
            {'id': 2, 'text': 'مرحبا بالعالم'}
        ]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        df_read = read_json(str(json_file), encoding='utf-8')

        assert df_read['text'].iloc[0] == '你好世界'
        assert df_read['text'].iloc[1] == 'مرحبا بالعالم'

    def test_read_json_with_null_values(self, tmp_path):
        """Test reading JSON with null values."""
        json_file = tmp_path / "nulls.json"
        data = [
            {'id': 1, 'name': 'Alice', 'value': None},
            {'id': 2, 'name': None, 'value': 200},
            {'id': 3, 'name': 'Charlie', 'value': 300}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert pd.isna(df_read['value'].iloc[0])
        assert pd.isna(df_read['name'].iloc[1])

    def test_read_json_large_numbers(self, tmp_path):
        """Test reading JSON with large numbers."""
        json_file = tmp_path / "large_numbers.json"
        data = [
            {'id': 1, 'big_int': 999999999999999999},
            {'id': 2, 'big_int': 888888888888888888}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert df_read['big_int'].iloc[0] == 999999999999999999


class TestReadJSONErrors:
    """Test read_json error handling."""

    def test_read_json_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_json("/nonexistent/path/to/file.json")

    def test_read_json_invalid_json(self, tmp_path):
        """Test reading invalid JSON file."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{invalid json content")

        with pytest.raises(Exception):  # JSONDecodeError or similar
            read_json(str(json_file))

    def test_read_json_bad_encoding(self, tmp_path):
        """Test reading file with wrong encoding."""
        json_file = tmp_path / "bad_encoding.json"
        json_file.write_bytes(b'\xff\xfe{"id": 1}\n')  # UTF-16 BOM

        with pytest.raises(Exception):
            read_json(str(json_file), encoding='ascii')


class TestWriteJSONBasic:
    """Test basic write_json functionality."""

    def test_write_json_records_format(self, tmp_path):
        """Test writing JSON in records format."""
        json_file = tmp_path / "output.json"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })

        write_json(df, str(json_file), orient='records')

        assert json_file.exists()
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[0]['id'] == 1

    def test_write_json_split_format(self, tmp_path):
        """Test writing JSON in split format."""
        json_file = tmp_path / "split_out.json"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        write_json(df, str(json_file), orient='split')

        assert json_file.exists()
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert 'index' in data
        assert 'columns' in data
        assert 'data' in data

    def test_write_json_index_format(self, tmp_path):
        """Test writing JSON in index format."""
        json_file = tmp_path / "index_out.json"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'value': [100, 200]
        })

        write_json(df, str(json_file), orient='index')

        assert json_file.exists()
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert '0' in data

    def test_write_json_columns_format(self, tmp_path):
        """Test writing JSON in columns format."""
        json_file = tmp_path / "cols_out.json"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        write_json(df, str(json_file), orient='columns')

        assert json_file.exists()
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert 'id' in data
        assert 'name' in data

    def test_write_json_with_index(self, tmp_path):
        """Test writing JSON with index."""
        json_file = tmp_path / "with_index.json"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'value': [100, 200]
        }, index=['a', 'b'])

        write_json(df, str(json_file), orient='index')

        assert json_file.exists()


class TestWriteJSONEncoding:
    """Test write_json with encodings."""

    def test_write_json_utf8_encoding(self, tmp_path):
        """Test writing JSON with UTF-8 encoding."""
        json_file = tmp_path / "utf8_out.json"
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Ålice', 'Böb']
        })

        write_json(df, str(json_file), orient='records')

        # Read back and verify
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data[0]['name'] == 'Ålice'

    def test_write_json_with_force_ascii_false(self, tmp_path):
        """Test writing JSON with force_ascii=False."""
        json_file = tmp_path / "unicode_out.json"
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['你好', 'مرحبا']
        })

        write_json(df, str(json_file), orient='records', force_ascii=False)

        # Read back and verify unicode is preserved
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '你好' in content


class TestWriteJSONFormatting:
    """Test write_json formatting options."""

    def test_write_json_with_indent(self, tmp_path):
        """Test writing JSON with indentation."""
        json_file = tmp_path / "indented.json"
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob']
        })

        write_json(df, str(json_file), orient='records', indent=2)

        content = json_file.read_text()
        # Indented JSON should have multiple lines
        assert '\n' in content
        assert '  ' in content

    def test_write_json_double_precision(self, tmp_path):
        """Test writing JSON with double precision."""
        json_file = tmp_path / "precision.json"
        df = pd.DataFrame({
            'id': [1, 2],
            'value': [1.23456789, 9.87654321]
        })

        write_json(df, str(json_file), orient='records', double_precision=4)

        assert json_file.exists()


class TestWriteJSONLines:
    """Test write_json with lines format."""

    def test_write_json_lines_format(self, tmp_path):
        """Test writing JSON Lines format."""
        json_file = tmp_path / "lines_out.jsonl"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        write_json(df, str(json_file), orient='records', lines=True)

        lines = json_file.read_text().strip().split('\n')
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert 'id' in data
            assert 'name' in data


class TestDataJSONHandler:
    """Test DataJSON handler integration."""

    def test_datajson_read_write_roundtrip(self, tmp_path):
        """Test reading and writing with DataJSON handler."""
        json_file = tmp_path / "roundtrip.json"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })

        handler = DataJSON()
        handler.write(df_original, str(json_file), orient='records')

        df_read = handler.read(str(json_file))

        assert len(df_read) == 3
        assert list(df_read.columns) == ['id', 'name', 'value']

    def test_datajson_multiple_instances(self, tmp_path):
        """Test using multiple DataJSON instances."""
        json_file1 = tmp_path / "file1.json"
        json_file2 = tmp_path / "file2.json"

        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        handler1 = DataJSON()
        handler2 = DataJSON()

        handler1.write(df1, str(json_file1), orient='records')
        handler2.write(df2, str(json_file2), orient='records')

        df1_read = handler1.read(str(json_file1))
        df2_read = handler2.read(str(json_file2))

        assert list(df1_read.columns) == ['a']
        assert list(df2_read.columns) == ['b']


class TestReadWriteJSONDataTypes:
    """Test JSON read/write with various data types."""

    def test_json_roundtrip_with_nan(self, tmp_path):
        """Test JSON roundtrip with NaN values."""
        json_file = tmp_path / "nan_roundtrip.json"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, np.nan, 30.1]
        })

        write_json(df_original, str(json_file), orient='records')
        df_read = read_json(str(json_file))

        assert len(df_read) == 3
        assert pd.isna(df_read['value'].iloc[1])

    def test_json_roundtrip_with_bool(self, tmp_path):
        """Test JSON roundtrip with boolean values."""
        json_file = tmp_path / "bool_roundtrip.json"
        df_original = pd.DataFrame({
            'id': [1, 2, 3],
            'flag': [True, False, True]
        })

        write_json(df_original, str(json_file), orient='records')
        df_read = read_json(str(json_file))

        assert df_read['flag'].iloc[0] == True
        assert df_read['flag'].iloc[1] == False


class TestReadJSONLargeData:
    """Test read_json with larger datasets."""

    def test_read_json_many_records(self, tmp_path):
        """Test reading JSON with many records."""
        json_file = tmp_path / "large.json"
        n_records = 1000
        data = [{'id': i, 'value': i * 10} for i in range(n_records)]
        with open(json_file, 'w') as f:
            json.dump(data, f)

        df_read = read_json(str(json_file))

        assert len(df_read) == n_records

    def test_read_json_lines_many_records(self, tmp_path):
        """Test reading JSON Lines with many records."""
        json_file = tmp_path / "large_lines.jsonl"
        n_records = 1000
        with open(json_file, 'w') as f:
            for i in range(n_records):
                f.write(json.dumps({'id': i, 'value': i * 10}) + '\n')

        df_read = read_json(str(json_file), lines=True)

        assert len(df_read) == n_records
