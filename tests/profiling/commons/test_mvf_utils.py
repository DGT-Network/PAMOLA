import pytest
import pandas as pd
from pamola_core.profiling.commons import mvf_utils
import logging

# Silence logger for test output clarity
logging.getLogger(mvf_utils.__name__).setLevel(logging.CRITICAL)

def test_parse_mvf_valid_cases():
    # JSON array
    assert mvf_utils.parse_mvf('["A", "B"]', format_type='json') == ["A", "B"]
    # Array string
    assert mvf_utils.parse_mvf("['A', 'B']", format_type='array_string') == ["A", "B"]
    # CSV
    assert mvf_utils.parse_mvf('A, B', format_type='csv') == ["A", "B"]
    # Auto-detect JSON
    assert mvf_utils.parse_mvf('["A", "B"]') == ["A", "B"]
    # Auto-detect array string
    assert mvf_utils.parse_mvf("['A', 'B']") == ["A", "B"]
    # Auto-detect CSV
    assert mvf_utils.parse_mvf('A, B') == ["A", "B"]
    # Dict as JSON
    assert mvf_utils.parse_mvf('{"A": 1, "B": 2}', format_type='json') == ["A", "B"]
    # Non-string value
    assert mvf_utils.parse_mvf(123) == ["123"]
    # Quoted CSV (should split as a single value if quotes are not handled)
    assert mvf_utils.parse_mvf('"A,1",B', format_type='csv') == ['"A', '1"', 'B']

def test_parse_mvf_edge_cases():
    # None, NaN, empty string
    assert mvf_utils.parse_mvf(None) == []
    assert mvf_utils.parse_mvf(float('nan')) == []
    assert mvf_utils.parse_mvf('') == []
    # Empty array representations
    assert mvf_utils.parse_mvf('[]') == []
    assert mvf_utils.parse_mvf('None') == []
    assert mvf_utils.parse_mvf('nan') == []
    # Array with empty inner content
    assert mvf_utils.parse_mvf('[]', format_type='array_string') == []
    # Array with quoted separator
    assert mvf_utils.parse_mvf('["A,1", "B"]', format_type='json') == ["A,1", "B"]
    # Array with custom separator
    assert mvf_utils.parse_mvf('A|B', format_type='csv', separator='|') == ["A", "B"]

def test_parse_mvf_invalid_cases():
    # Invalid JSON (should fallback to separator split)
    assert mvf_utils.parse_mvf('[A, B]', format_type='json') == ["A", "B"]
    # Invalid array string (should fallback to separator split)
    assert mvf_utils.parse_mvf('[A, B]', format_type='array_string') == ["A", "B"]
    # Malformed input
    assert mvf_utils.parse_mvf('["A", B') == ['["A"', 'B']
    # Unknown format_type
    assert mvf_utils.parse_mvf('A,B', format_type='unknown') == ["A", "B"]

def test_detect_mvf_format():
    assert mvf_utils.detect_mvf_format(["['A', 'B']"]*10 + [None]*5) == 'array_string'
    assert mvf_utils.detect_mvf_format(['["A", "B"]']*10) == 'json'
    assert mvf_utils.detect_mvf_format(['A, B']*10) == 'csv'
    assert mvf_utils.detect_mvf_format([None, 123, 'foo']*10) == 'unknown'
    assert mvf_utils.detect_mvf_format([]) == 'unknown'

def test_standardize_mvf_format():
    # List
    assert mvf_utils.standardize_mvf_format('["A", "B"]', 'list') == ["A", "B"]
    # JSON
    assert mvf_utils.standardize_mvf_format('["A", "B"]', 'json') == '["A", "B"]'
    # CSV
    assert mvf_utils.standardize_mvf_format('["A", "B"]', 'csv') == 'A, B'
    # Array string
    assert mvf_utils.standardize_mvf_format('["A", "B"]', 'array_string') == "['A', 'B']"
    # Unknown target format
    assert mvf_utils.standardize_mvf_format('["A", "B"]', 'foo') == ["A", "B"]
    # Escaping single quotes: the function escapes single quotes with \\', so the output will be double-escaped
    result = mvf_utils.standardize_mvf_format("['A'B', 'C']", 'array_string')
    assert result == "['A\\'B', 'C']" or result == "['\\'A\\'B\\'', '\\'C\\'']"

def test_analyze_mvf_field_valid():
    df = pd.DataFrame({
        'mvf': [
            '["A", "B"]',
            '["A"]',
            '["B"]',
            '[]',
            None,
            '["A", "B"]',
            '["C"]',
            '["A", "B"]',
        ]
    })
    result = mvf_utils.analyze_mvf_field(df, 'mvf')
    assert result['field_name'] == 'mvf'
    assert result['total_records'] == 8
    assert result['null_count'] == 1
    assert result['unique_values'] == 3
    assert 'values_analysis' in result
    assert 'combinations_analysis' in result
    assert 'value_counts_distribution' in result
    assert result['empty_arrays_count'] == 1
    assert result['unique_combinations'] >= 1
    assert result['avg_values_per_record'] > 0

def test_analyze_mvf_field_invalid():
    df = pd.DataFrame({'foo': [1, 2, 3]})
    result = mvf_utils.analyze_mvf_field(df, 'bar')
    assert 'error' in result

def test_create_value_dictionary():
    df = pd.DataFrame({'mvf': ['["A", "B"]', '["A"]', '["B"]', '[]', None, '["A", "B"]']})
    out = mvf_utils.create_value_dictionary(df, 'mvf', min_frequency=1)
    assert set(out.columns) == {'value', 'frequency', 'percentage'}
    assert out['frequency'].sum() == 6
    assert all(out['frequency'] >= 1)
    # Field not found
    out2 = mvf_utils.create_value_dictionary(df, 'foo')
    assert out2.empty

def test_create_combinations_dictionary():
    df = pd.DataFrame({'mvf': ['["A", "B"]', '["A"]', '["B"]', '[]', None, '["A", "B"]']})
    out = mvf_utils.create_combinations_dictionary(df, 'mvf', min_frequency=1)
    assert set(out.columns) == {'combination', 'frequency', 'percentage'}
    # The sum of frequencies should match the number of non-null records (5, since one is None and one is empty array)
    assert out['frequency'].sum() == 5
    assert all(out['frequency'] >= 1)
    # Field not found
    out2 = mvf_utils.create_combinations_dictionary(df, 'foo')
    assert out2.empty

def test_analyze_value_count_distribution():
    df = pd.DataFrame({'mvf': ['["A", "B"]', '["A"]', '["B"]', '[]', None, '["A", "B"]']})
    out = mvf_utils.analyze_value_count_distribution(df, 'mvf')
    assert isinstance(out, dict)
    assert '0' in out and '1' in out and '2' in out
    # Field not found
    out2 = mvf_utils.analyze_value_count_distribution(df, 'foo')
    assert out2 == {}

def test_estimate_resources():
    df = pd.DataFrame({'mvf': ['["A", "B"]']*100 + ['["A"]']*50 + [None]*10})
    out = mvf_utils.estimate_resources(df, 'mvf')
    assert out['field_name'] == 'mvf'
    assert out['total_records'] == 160
    assert out['non_null_count'] == 150
    assert 'detected_format' in out
    assert 'estimated_avg_values_per_record' in out
    assert 'estimated_memory_mb' in out
    assert 'estimated_time_seconds' in out
    # Field not found
    out2 = mvf_utils.estimate_resources(df, 'foo')
    assert 'error' in out2

if __name__ == "__main__":
    pytest.main()
