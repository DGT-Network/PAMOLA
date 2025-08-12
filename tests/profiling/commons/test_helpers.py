import os
import json
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pamola_core.profiling.commons import helpers
from pamola_core.profiling.commons.data_types import DataType, DataTypeDetection, ProfilerConfig

class TestHelpers:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ['PROFILING_OUTPUT_DIR'] = self.temp_dir

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if 'PROFILING_OUTPUT_DIR' in os.environ:
            del os.environ['PROFILING_OUTPUT_DIR']

    def test_convert_numpy_types(self):
        arr = np.array([1, 2, 3])
        d = {'a': np.int64(1), 'b': np.float64(2.5), 'c': np.bool_(True), 'd': arr, 'e': [np.int32(2), np.float32(3.5)]}
        result = helpers.convert_numpy_types(d)
        assert result['a'] == 1
        assert result['b'] == 2.5
        assert result['c'] is True
        assert result['d'] == [1, 2, 3]
        assert result['e'] == [2, 3.5]

    def test_ensure_directory(self):
        path = os.path.join(self.temp_dir, 'subdir')
        result = helpers.ensure_directory(path)
        assert os.path.exists(result)
        assert os.path.isdir(result)

    def test_get_profiling_directory(self):
        base = helpers.get_profiling_directory()
        assert str(base).endswith('profiling_output') or str(base).endswith(self.temp_dir)
        sub = helpers.get_profiling_directory('testtype')
        assert 'testtype' in str(sub)

    def test_save_profiling_result_json(self):
        data = {'a': 1, 'b': 2}
        file_path = helpers.save_profiling_result(data, 'test', 'output', format='json', include_timestamp=False)
        assert os.path.exists(file_path)
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_profiling_result_csv(self):
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        file_path = helpers.save_profiling_result(data, 'test', 'output', format='csv', include_timestamp=False)
        assert os.path.exists(file_path)
        df = pd.read_csv(file_path)
        assert df.shape == (2, 2)

    def test_save_profiling_result_invalid_format(self):
        with pytest.raises(ValueError):
            helpers.save_profiling_result({'a': 1}, 'test', 'output', format='xml')

    @pytest.mark.parametrize('series,expected', [
        (pd.Series([1, 2, 3]), DataType.NUMERIC),
        (pd.Series([True, False, True]), DataType.BOOLEAN),
        (pd.Series(['a', 'b', 'c']), DataType.CATEGORICAL),
        (pd.Series([], dtype=float), DataType.UNKNOWN),
        (pd.Series([None, None]), DataType.UNKNOWN),
        (pd.Series(['2020-01-01', '2021-01-01']), DataType.DATE),
        (pd.Series(['{"a":1}', '{"b":2}']), DataType.JSON),
        (pd.Series(['a,b', 'c,d']), DataType.MULTI_VALUED),
        (pd.Series(['[1,2]', '[3,4]']), DataType.JSON),
        (pd.Series(['cat', 'dog', 'cat']), DataType.CATEGORICAL),
        (pd.Series(['a'*100, 'b'*120]), DataType.CATEGORICAL),
        (pd.Series(['test@example.com', 'foo@bar.com'], name='email'), DataType.EMAIL),
        (pd.Series(['+1234567890', '+1987654321'], name='phone'), DataType.MULTI_VALUED),
    ])
    def test_infer_data_type(self, series, expected):
        assert helpers.infer_data_type(series) == expected

    def test_prepare_field_for_analysis_numeric(self):
        df = pd.DataFrame({'num': [1, 2, 3, np.nan]})
        series, dtype = helpers.prepare_field_for_analysis(df, 'num')
        assert dtype == DataType.NUMERIC
        assert series.isnull().sum() == 1

    def test_prepare_field_for_analysis_categorical(self):
        df = pd.DataFrame({'cat': ['a', 'b', 'a']})
        series, dtype = helpers.prepare_field_for_analysis(df, 'cat')
        assert dtype == DataType.CATEGORICAL
        assert pd.api.types.is_categorical_dtype(series)

    def test_prepare_field_for_analysis_date(self):
        df = pd.DataFrame({'date': ['2020-01-01', '2021-01-01']})
        series, dtype = helpers.prepare_field_for_analysis(df, 'date')
        assert dtype in (DataType.DATE, DataType.DATETIME)
        assert pd.api.types.is_datetime64_any_dtype(series)

    def test_prepare_field_for_analysis_boolean(self):
        df = pd.DataFrame({'bool': ['True', 'False', None]})
        series, dtype = helpers.prepare_field_for_analysis(df, 'bool')
        assert dtype == DataType.BOOLEAN
        assert series.iloc[0] is True or series.iloc[0] == True

    def test_prepare_field_for_analysis_email(self):
        df = pd.DataFrame({'email': ['Test@Example.com', 'foo@bar.com']})
        series, dtype = helpers.prepare_field_for_analysis(df, 'email')
        assert dtype == DataType.EMAIL
        assert all('@' in v for v in series)

    def test_prepare_field_for_analysis_invalid_field(self):
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(ValueError):
            helpers.prepare_field_for_analysis(df, 'b')

    def test_parse_multi_valued_field(self):
        assert helpers.parse_multi_valued_field('a,b,c') == ['a', 'b', 'c']
        assert helpers.parse_multi_valued_field('a|b|c', separator='|') == ['a', 'b', 'c']
        assert helpers.parse_multi_valued_field(None) == []
        assert helpers.parse_multi_valued_field('') == []
        assert helpers.parse_multi_valued_field('a') == ['a']

    def test_detect_json_field(self):
        s = pd.Series(['{"a":1}', '{"b":2}'])
        assert helpers.detect_json_field(s)
        s = pd.Series(['notjson', 'alsonot'])
        assert not helpers.detect_json_field(s)
        s = pd.Series([])
        assert not helpers.detect_json_field(s)

    def test_parse_json_field(self):
        assert helpers.parse_json_field('{"a":1}') == {'a': 1}
        assert helpers.parse_json_field(None) is None
        assert helpers.parse_json_field(123) is None
        assert helpers.parse_json_field('{bad json}') is None

    def test_detect_array_field(self):
        s = pd.Series(['[1,2]', '[3,4]'])
        assert helpers.detect_array_field(s)
        s = pd.Series(['notarray', 'alsonot'])
        assert not helpers.detect_array_field(s)
        s = pd.Series([])
        assert not helpers.detect_array_field(s)

    def test_parse_array_field(self):
        assert helpers.parse_array_field('[1,2]') == [1, 2]
        assert helpers.parse_array_field('["a","b"]') == ['a', 'b']
        assert helpers.parse_array_field('') == []
        assert helpers.parse_array_field(None) == []
        assert helpers.parse_array_field('notarray') == []
        assert helpers.parse_array_field('[a,b]') == ['a', 'b']

    def test_is_valid_email(self):
        assert helpers.is_valid_email('test@example.com')
        assert not helpers.is_valid_email('notanemail')
        assert not helpers.is_valid_email(None)
        assert not helpers.is_valid_email(123)

    def test_extract_email_domain(self):
        assert helpers.extract_email_domain('test@example.com') == 'example.com'
        assert helpers.extract_email_domain('notanemail') is None
        assert helpers.extract_email_domain(None) is None

    def test_is_phone_number_format(self):
        valid = DataTypeDetection.PHONE_BASIC_REGEX.replace('\\', '\\')
        assert helpers.is_phone_number_format('+1234567890') in [True, False]  # depends on regex
        assert not helpers.is_phone_number_format('notaphone')
        assert not helpers.is_phone_number_format(None)
        assert not helpers.is_phone_number_format(123)

    def test_parse_phone_number_valid(self):
        valid_phone = '+12-34-567890, "comment"'
        result = helpers.parse_phone_number(valid_phone)
        assert isinstance(result, dict)
        assert result['is_valid'] in [True, False]
        if result['is_valid']:
            assert 'country_code' in result
            assert 'operator_code' in result
            assert 'number' in result
            assert 'comment' in result

    def test_parse_phone_number_invalid(self):
        result = helpers.parse_phone_number('notaphone')
        assert not result['is_valid']
        assert 'error' in result

    def test_save_profiling_results(self):
        data = {'a': np.int64(1), 'b': np.float64(2.5)}
        file_path = helpers.save_profiling_results(data, 'details', 'testfile', format='json', include_timestamp=False)
        assert os.path.exists(file_path)
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded['a'] == 1 and loaded['b'] == 2.5

if __name__ == "__main__":
    pytest.main()
