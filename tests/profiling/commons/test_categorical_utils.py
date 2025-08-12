import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pamola_core.profiling.commons import categorical_utils as cu

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        'cat': ['a', 'b', 'a', 'c', 'b', 'a', None, 'd', 'e', 'e', 'e', 'e'],
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'empty': [None]*12
    })


def test_analyze_categorical_field_valid(simple_df):
    result = cu.analyze_categorical_field(simple_df, 'cat', top_n=3, min_frequency=1)
    assert result['field_name'] == 'cat'
    assert result['total_records'] == 12
    assert result['non_null_values'] == 11
    assert 'top_values' in result
    assert 'entropy' in result
    assert 'value_dictionary' in result
    assert result['unique_values'] == 5
    assert result['null_values'] == 1
    assert result['null_percent'] > 0
    assert 'distribution_type' in result
    assert 'skewness' in result
    assert 'concentration' in result


def test_analyze_categorical_field_missing_column(simple_df):
    result = cu.analyze_categorical_field(simple_df, 'not_a_column')
    assert 'error' in result
    assert 'not_a_column' in result['error']


def test_analyze_categorical_field_empty_df():
    df = pd.DataFrame({'cat': []})
    result = cu.analyze_categorical_field(df, 'cat')
    assert result['total_records'] == 0
    assert result['non_null_values'] == 0
    assert result['null_values'] == 0
    assert result['unique_values'] == 0
    assert result['top_values'] == {}
    assert result['percent_covered_by_top'] == 0
    assert result['entropy'] == 0
    assert result['cardinality_ratio'] == 0
    assert result['value_dictionary']['total_unique_values'] == 0


def test_analyze_categorical_field_kwargs_disable_features(simple_df):
    result = cu.analyze_categorical_field(simple_df, 'cat', analyze_distribution=False, detect_anomalies=False)
    assert 'distribution_type' not in result
    assert 'anomalies' not in result


def test_create_value_dictionary_valid(simple_df):
    d = cu.create_value_dictionary(simple_df, 'cat', min_frequency=1)
    assert d['field_name'] == 'cat'
    assert d['total_unique_values'] > 0
    assert isinstance(d['dictionary_data'], list)
    assert all('frequency' in x for x in d['dictionary_data'])
    assert all('percent' in x for x in d['dictionary_data'])


def test_create_value_dictionary_min_frequency(simple_df):
    d = cu.create_value_dictionary(simple_df, 'cat', min_frequency=3)
    assert all(x['frequency'] >= 3 for x in d['dictionary_data'])


def test_create_value_dictionary_empty(simple_df):
    d = cu.create_value_dictionary(simple_df, 'empty')
    assert d['total_unique_values'] == 0
    assert d['dictionary_data'] == []


def test_create_value_dictionary_invalid_column(simple_df):
    with mock.patch.object(cu.logger, 'error') as mock_log:
        d = cu.create_value_dictionary(simple_df, 'not_a_column')
        assert 'error' in d
        assert d['total_unique_values'] == 0
        assert d['dictionary_data'] == []
        assert mock_log.called


def test_analyze_distribution_characteristics_typical():
    vc = pd.Series([5, 3, 2], index=['a', 'b', 'c'])
    res = cu.analyze_distribution_characteristics(vc, non_null_values=10, unique_values=3)
    assert 'distribution_type' in res
    assert 'skewness' in res
    assert 'concentration' in res
    assert res['distribution_type'] in [
        'single_value', 'highly_concentrated', 'concentrated', 'moderately_distributed', 'well_distributed', 'unknown']


def test_analyze_distribution_characteristics_empty():
    vc = pd.Series([], dtype=int)
    res = cu.analyze_distribution_characteristics(vc, non_null_values=0, unique_values=0)
    assert res['distribution_type'] == 'empty'
    assert res['skewness'] == 0
    assert res['concentration'] == 0


def test_detect_anomalies_single_char_and_numeric():
    df = pd.DataFrame({'cat': ['a', 'b', '1', '2', 'aa', 'bb', 'c', 'd', 'e', 'f', 'g', 'h']})
    vc = df['cat'].value_counts()
    anomalies = cu.detect_anomalies(df, 'cat', vc, min_frequency=1)
    assert 'single_char_values' in anomalies
    assert 'numeric_like_strings' in anomalies
    assert isinstance(anomalies['single_char_values'], dict)
    assert isinstance(anomalies['numeric_like_strings'], dict)


def test_detect_anomalies_potential_typos():
    df = pd.DataFrame({'cat': ['apple', 'appl', 'apple', 'aple', 'banana', 'banan', 'banana', 'bananna']})
    vc = df['cat'].value_counts()
    anomalies = cu.detect_anomalies(df, 'cat', vc, min_frequency=1)
    assert 'potential_typos' in anomalies
    assert any('similar_to' in v for v in anomalies['potential_typos'].values())


def test_detect_anomalies_none_found():
    df = pd.DataFrame({'cat': ['apple', 'banana', 'cherry', 'date']})
    vc = df['cat'].value_counts()
    anomalies = cu.detect_anomalies(df, 'cat', vc, min_frequency=1)
    assert anomalies == {}


def test_levenshtein_distance_basic():
    assert cu.levenshtein_distance('kitten', 'sitting') == 3
    assert cu.levenshtein_distance('flaw', 'lawn') == 2
    assert cu.levenshtein_distance('', 'abc') == 3
    assert cu.levenshtein_distance('abc', '') == 3
    assert cu.levenshtein_distance('abc', 'abc') == 0


def test_estimate_resources_typical(simple_df):
    res = cu.estimate_resources(simple_df, 'cat')
    assert 'estimated_memory_mb' in res
    assert 'estimated_time_seconds' in res
    assert 'complex_distribution' in res
    assert 'unique_value_count' in res
    assert 'total_records' in res


def test_estimate_resources_missing_column(simple_df):
    res = cu.estimate_resources(simple_df, 'not_a_column')
    assert 'error' in res
    assert 'not_a_column' in res['error']


def test_analyze_multiple_categorical_fields_valid(simple_df):
    res = cu.analyze_multiple_categorical_fields(simple_df, ['cat', 'num', 'empty'])
    assert isinstance(res, dict)
    assert 'cat' in res and 'num' in res and 'empty' in res
    assert 'field_name' in res['cat']
    assert 'field_name' in res['num']
    assert 'field_name' in res['empty']


def test_analyze_multiple_categorical_fields_with_error(simple_df):
    res = cu.analyze_multiple_categorical_fields(simple_df, ['cat', 'not_a_column'])
    assert 'cat' in res
    assert 'not_a_column' in res
    assert 'error' in res['not_a_column']


def test_analyze_multiple_categorical_fields_empty_fields(simple_df):
    res = cu.analyze_multiple_categorical_fields(simple_df, [])
    assert res == {}


def test_analyze_multiple_categorical_fields_invalid_input():
    df = pd.DataFrame({'cat': ['a', 'b']})
    with pytest.raises(TypeError):
        cu.analyze_multiple_categorical_fields(df, None)


def test_analyze_categorical_field_invalid_types():
    with pytest.raises(AttributeError):
        cu.analyze_categorical_field(None, 'cat')
    with pytest.raises(AttributeError):
        cu.analyze_categorical_field('not_a_df', 'cat')


def test_create_value_dictionary_invalid_types():
    # Should not raise, but return a dict with 'error' key
    d1 = cu.create_value_dictionary(None, 'cat')
    assert 'error' in d1
    d2 = cu.create_value_dictionary('not_a_df', 'cat')
    assert 'error' in d2


def test_estimate_resources_invalid_types():
    with pytest.raises(AttributeError):
        cu.estimate_resources(None, 'cat')
    with pytest.raises(AttributeError):
        cu.estimate_resources('not_a_df', 'cat')


def test_levenshtein_distance_edge_cases():
    assert cu.levenshtein_distance('', '') == 0
    assert cu.levenshtein_distance('a', '') == 1
    assert cu.levenshtein_distance('', 'a') == 1
    assert cu.levenshtein_distance('a', 'b') == 1

if __name__ == "__main__":
    pytest.main()