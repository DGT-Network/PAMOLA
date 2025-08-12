import pytest
import pandas as pd
import numpy as np
from pamola_core.profiling.commons import group_utils
from unittest import mock


def make_group_df(fields, values):
    return pd.DataFrame([{k: v[i] for k, v in fields.items()} for i in range(len(next(iter(fields.values()))))])


def test_calculate_field_variation_valid():
    df = pd.DataFrame({'a': [1, 2, 3, 4]})
    result = group_utils.calculate_field_variation(df, 'a')
    assert result == 1.0


def test_calculate_field_variation_identical():
    df = pd.DataFrame({'a': [5, 5, 5]})
    result = group_utils.calculate_field_variation(df, 'a')
    assert result == 0.0


def test_calculate_field_variation_single_row():
    df = pd.DataFrame({'a': [1]})
    assert group_utils.calculate_field_variation(df, 'a') == 0.0


def test_calculate_field_variation_nulls_as_value():
    df = pd.DataFrame({'a': [1, None, 1, 2]})
    result = group_utils.calculate_field_variation(df, 'a', handle_nulls='as_value')
    assert 0 < result < 1


def test_calculate_field_variation_nulls_exclude():
    df = pd.DataFrame({'a': [1, None, 1, 2]})
    result = group_utils.calculate_field_variation(df, 'a', handle_nulls='exclude')
    assert 0 < result < 1


def test_calculate_field_variation_mvf_field():
    df = pd.DataFrame({'a': ["[1,2]", "[2,1]", "[3]"]})
    result = group_utils.calculate_field_variation(df, 'a')
    assert result == 0.5


def test_calculate_field_variation_invalid_mvf():
    df = pd.DataFrame({'a': ["notalist", "notalist", "notalist"]})
    result = group_utils.calculate_field_variation(df, 'a')
    assert result == 0.0


def test_calculate_weighted_variation_valid():
    df = pd.DataFrame({'a': [1, 2, 2], 'b': [3, 3, 4]})
    weights = {'a': 0.7, 'b': 0.3}
    result = group_utils.calculate_weighted_variation(df, weights)
    assert 0 <= result <= 1


def test_calculate_weighted_variation_empty_fields():
    df = pd.DataFrame({'a': [1, 2]})
    result = group_utils.calculate_weighted_variation(df, {}, handle_nulls='as_value')
    assert result == 0.0


def test_calculate_weighted_variation_zero_weight():
    df = pd.DataFrame({'a': [1, 2]})
    result = group_utils.calculate_weighted_variation(df, {'a': 0.0})
    assert result == 0.0


def test_calculate_change_frequency_valid():
    df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [1, 1, 1, 1]})
    result = group_utils.calculate_change_frequency(df, ['a', 'b'])
    assert result['a'] > 0
    assert result['b'] == 0.0


def test_calculate_change_frequency_single_row():
    df = pd.DataFrame({'a': [1]})
    result = group_utils.calculate_change_frequency(df, ['a'])
    assert result['a'] == 0.0


def test_calculate_change_frequency_nulls_exclude():
    df = pd.DataFrame({'a': [1, None, 2, 2]})
    result = group_utils.calculate_change_frequency(df, ['a'], handle_nulls='exclude')
    assert 'a' in result


def test_create_identifier_hash_valid():
    row = pd.Series({'x': 'foo', 'y': 'bar'})
    result = group_utils.create_identifier_hash(row, ['x', 'y'])
    assert isinstance(result, str)
    assert len(result) == 32


def test_create_identifier_hash_missing_field():
    row = pd.Series({'x': 'foo'})
    result = group_utils.create_identifier_hash(row, ['x', 'y'])
    assert result == 'acbd18db4cc2f85cedef654fccc4a4d8'


def test_analyze_cross_groups_valid():
    df = pd.DataFrame({
        'resume_id': [1, 1, 2, 2],
        'first_name': ['A', 'A', 'B', 'B'],
        'last_name': ['X', 'X', 'Y', 'Y'],
        'birth_day': [1, 1, 2, 2]
    })
    result = group_utils.analyze_cross_groups(df, 'resume_id', ['first_name', 'last_name', 'birth_day'])
    assert 'primary_group_field' in result
    assert 'cross_group_count' in result


def test_analyze_cross_groups_missing_primary():
    df = pd.DataFrame({'a': [1]})
    result = group_utils.analyze_cross_groups(df, 'resume_id', ['a'])
    assert 'error' in result


def test_analyze_cross_groups_missing_secondary():
    df = pd.DataFrame({'resume_id': [1]})
    result = group_utils.analyze_cross_groups(df, 'resume_id', ['foo'])
    assert 'error' in result


def test_extract_group_metadata_valid():
    df = pd.DataFrame({'a': [1, 2, 2, None]})
    result = group_utils.extract_group_metadata(df, ['a'])
    assert result['group_size'] == 4
    assert 'a_unique_count' in result
    assert 'a_most_common' in result
    assert 'a_null_count' in result


def test_extract_group_metadata_missing_field():
    df = pd.DataFrame({'a': [1, 2]})
    result = group_utils.extract_group_metadata(df, ['b'])
    assert result['group_size'] == 2
    assert all('b' not in k for k in result)


def test_analyze_collapsibility_valid():
    results = [
        {'group_id': 1, 'variation': 0.1, 'size': 3, 'field_variations': {'a': 0.1}},
        {'group_id': 2, 'variation': 0.3, 'size': 2, 'field_variations': {'a': 0.3}},
    ]
    res = group_utils.analyze_collapsibility(results, threshold=0.2)
    assert res['collapsible_groups_count'] == 1
    assert res['collapsible_records_count'] == 3


def test_analyze_collapsibility_empty():
    res = group_utils.analyze_collapsibility([], threshold=0.2)
    assert res['collapsible_groups_count'] == 0
    assert res['collapsible_records_count'] == 0


def test_identify_change_patterns_valid():
    results = [
        {'field_variations': {'a': 0.1, 'b': 0.9}},
        {'field_variations': {'a': 0.2, 'b': 0.8}},
        {'field_variations': {'a': 0.3, 'b': 0.7}},
    ]
    res = group_utils.identify_change_patterns(results)
    assert 'field_statistics' in res
    assert 'field_correlations' in res


def test_calculate_variation_distribution_valid():
    variations = [0.1, 0.2, 0.5, 0.7, 0.9]
    res = group_utils.calculate_variation_distribution(variations, bins=5)
    assert isinstance(res, dict)
    assert sum(res.values()) == len(variations)


def test_calculate_variation_distribution_empty():
    res = group_utils.calculate_variation_distribution([], bins=5)
    assert res == {}


def test_analyze_group_in_chunks_valid():
    df = pd.DataFrame({
        'group': [1]*3 + [2]*3,
        'a': [1, 2, 3, 4, 5, 6],
        'b': [1, 1, 1, 2, 2, 2]
    })
    fields_weights = {'a': 0.5, 'b': 0.5}
    results, stats = group_utils.analyze_group_in_chunks(df, 'group', fields_weights, chunk_size=2, min_group_size=2)
    assert isinstance(results, list)
    assert isinstance(stats, dict)
    assert stats['analyzed_groups'] == 2


def test_analyze_group_in_chunks_min_group_size():
    df = pd.DataFrame({'group': [1, 2], 'a': [1, 2], 'b': [1, 2]})
    fields_weights = {'a': 1.0}
    results, stats = group_utils.analyze_group_in_chunks(df, 'group', fields_weights, chunk_size=1, min_group_size=3)
    assert results == []
    assert stats['analyzed_groups'] == 0


def test_estimate_resources_valid():
    df = pd.DataFrame({'group': [1, 1, 2, 2], 'a': [1, 2, 3, 4]})
    fields_weights = {'a': 1.0}
    res = group_utils.estimate_resources(df, 'group', fields_weights)
    assert 'estimated_memory_mb' in res
    assert 'estimated_time_seconds' in res
    assert 'group_count' in res


def test_estimate_resources_missing_group_field():
    df = pd.DataFrame({'a': [1, 2]})
    fields_weights = {'a': 1.0}
    res = group_utils.estimate_resources(df, 'group', fields_weights)
    assert 'error' in res


def test_calculate_field_variation_invalid_field():
    df = pd.DataFrame({'a': [1, 2]})
    with pytest.raises(KeyError):
        group_utils.calculate_field_variation(df, 'b')


def test_calculate_weighted_variation_invalid_field():
    df = pd.DataFrame({'a': [1, 2]})
    weights = {'b': 1.0}
    result = group_utils.calculate_weighted_variation(df, weights)
    assert result == 0.0


def test_analyze_group_in_chunks_invalid_field():
    df = pd.DataFrame({'group': [1, 2], 'a': [1, 2]})
    fields_weights = {'b': 1.0}
    results, stats = group_utils.analyze_group_in_chunks(df, 'group', fields_weights)
    assert all('b' not in r['field_variations'] for r in results)


def test_create_identifier_hash_nan():
    row = pd.Series({'x': np.nan, 'y': 'foo'})
    result = group_utils.create_identifier_hash(row, ['x', 'y'])
    assert result is None


def test_analyze_cross_groups_handle_nulls_exclude():
    df = pd.DataFrame({
        'resume_id': [1, 2, 2],
        'first_name': ['A', None, 'B'],
        'last_name': ['X', 'Y', 'Y'],
        'birth_day': [1, 2, 2]
    })
    result = group_utils.analyze_cross_groups(df, 'resume_id', ['first_name', 'last_name', 'birth_day'], handle_nulls='exclude')
    assert 'primary_group_field' in result


def test_extract_group_metadata_empty():
    df = pd.DataFrame({'a': []})
    result = group_utils.extract_group_metadata(df, ['a'])
    assert result['group_size'] == 0


def test_analyze_collapsibility_all_above_threshold():
    results = [
        {'group_id': 1, 'variation': 0.5, 'size': 3, 'field_variations': {'a': 0.5}},
        {'group_id': 2, 'variation': 0.6, 'size': 2, 'field_variations': {'a': 0.6}},
    ]
    res = group_utils.analyze_collapsibility(results, threshold=0.2)
    assert res['collapsible_groups_count'] == 0
    assert res['collapsible_records_count'] == 0


def test_identify_change_patterns_empty():
    res = group_utils.identify_change_patterns([])
    assert 'field_statistics' in res
    assert 'field_correlations' in res
    assert res['field_statistics'] == {}


def test_calculate_variation_distribution_bins():
    variations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    res = group_utils.calculate_variation_distribution(variations, bins=2)
    assert isinstance(res, dict)
    assert sum(res.values()) == len(variations)


def test_analyze_group_in_chunks_large_chunk():
    df = pd.DataFrame({'group': [1]*10, 'a': list(range(10)), 'b': list(range(10))})
    fields_weights = {'a': 1.0, 'b': 1.0}
    results, stats = group_utils.analyze_group_in_chunks(df, 'group', fields_weights, chunk_size=100)
    assert stats['analyzed_groups'] == 1


def test_estimate_resources_large():
    df = pd.DataFrame({'group': [1]*1000, 'a': list(range(1000))})
    fields_weights = {'a': 1.0}
    res = group_utils.estimate_resources(df, 'group', fields_weights)
    assert res['use_chunks_recommended'] is False or res['use_chunks_recommended'] is True


if __name__ == "__main__":
    pytest.main()
