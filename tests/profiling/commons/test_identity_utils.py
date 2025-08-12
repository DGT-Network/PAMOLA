import pytest
import pandas as pd
from pamola_core.profiling.commons import identity_utils

import hashlib
from unittest import mock

def test_calculate_hash_valid_md5():
    values = [1, 'abc', None]
    result = identity_utils.calculate_hash(values, algorithm="md5")
    expected = hashlib.md5('1abc'.encode()).hexdigest().upper()
    assert result == expected

def test_calculate_hash_valid_sha1():
    values = [1, 'abc', None]
    result = identity_utils.calculate_hash(values, algorithm="sha1")
    expected = hashlib.sha1('1abc'.encode()).hexdigest().upper()
    assert result == expected

def test_calculate_hash_valid_sha256():
    values = [1, 'abc', None]
    result = identity_utils.calculate_hash(values, algorithm="sha256")
    expected = hashlib.sha256('1abc'.encode()).hexdigest().upper()
    assert result == expected

def test_calculate_hash_invalid_algorithm():
    values = [1, 2, 3]
    with pytest.raises(ValueError):
        identity_utils.calculate_hash(values, algorithm="notarealhash")

def test_calculate_hash_empty_values():
    values = []
    result = identity_utils.calculate_hash(values)
    expected = hashlib.md5(''.encode()).hexdigest().upper()
    assert result == expected

def test_compute_identifier_stats_valid():
    df = pd.DataFrame({
        'id': [1, 2, 2, 3, None],
        'entity': ['a', 'b', 'c', 'd', 'e']
    })
    stats = identity_utils.compute_identifier_stats(df, 'id', 'entity')
    assert stats['total_records'] == 5
    assert stats['unique_identifiers'] == 3
    assert stats['null_identifiers'] == 1
    assert stats['coverage_percentage'] == 80.0
    assert 'avg_entities_per_id' in stats
    assert 'max_entities_per_id' in stats
    assert 'one_to_one_count' in stats
    assert 'one_to_many_count' in stats
    assert 'avg_ids_per_entity' in stats
    assert 'max_ids_per_entity' in stats
    assert stats['uniqueness_ratio'] == 3/5

def test_compute_identifier_stats_missing_id_field():
    df = pd.DataFrame({'foo': [1, 2, 3]})
    stats = identity_utils.compute_identifier_stats(df, 'id')
    assert stats['error'].startswith('Field id not found')
    assert stats['total_records'] == 3
    assert stats['unique_identifiers'] == 0
    assert stats['null_identifiers'] == 0
    assert stats['coverage_percentage'] == 0

def test_compute_identifier_stats_no_entity_field():
    df = pd.DataFrame({'id': [1, 2, 2, 3, None]})
    stats = identity_utils.compute_identifier_stats(df, 'id', 'entity')
    assert 'avg_entities_per_id' not in stats
    assert stats['total_records'] == 5

def test_compute_identifier_stats_empty_df():
    df = pd.DataFrame({'id': []})
    stats = identity_utils.compute_identifier_stats(df, 'id')
    assert stats['total_records'] == 0
    assert stats['unique_identifiers'] == 0
    assert stats['null_identifiers'] == 0
    assert stats['coverage_percentage'] == 0
    assert stats['uniqueness_ratio'] == 0

def test_analyze_identifier_distribution_valid():
    df = pd.DataFrame({'id': [1, 2, 2, 3, 3, 3], 'entity': ['a', 'b', 'c', 'd', 'e', 'f']})
    result = identity_utils.analyze_identifier_distribution(df, 'id', 'entity', top_n=2)
    assert result['total_identifiers'] == 3
    assert result['total_records'] == 6
    assert result['max_count'] >= result['min_count']
    assert isinstance(result['distribution'], dict)
    assert len(result['top_examples']) == 2

def test_analyze_identifier_distribution_missing_id_field():
    df = pd.DataFrame({'foo': [1, 2, 3]})
    result = identity_utils.analyze_identifier_distribution(df, 'id')
    assert result['error'].startswith('Field id not found')
    assert result['total_records'] == 3

def test_analyze_identifier_distribution_missing_entity_field():
    df = pd.DataFrame({'id': [1, 2, 2, 3]})
    with mock.patch.object(identity_utils.logger, 'warning') as mock_warn:
        result = identity_utils.analyze_identifier_distribution(df, 'id', 'entity')
        assert result['total_identifiers'] == 3
        assert result['total_records'] == 4
        mock_warn.assert_called()

def test_analyze_identifier_distribution_empty_df():
    df = pd.DataFrame({'id': []})
    result = identity_utils.analyze_identifier_distribution(df, 'id')
    assert result['total_identifiers'] == 0
    assert result['total_records'] == 0
    assert result['max_count'] == 0
    assert result['min_count'] == 0
    assert result['avg_count'] == 0
    assert result['median_count'] == 0
    assert result['distribution'] == {}
    assert result['top_examples'] == []

def test_analyze_identifier_consistency_valid():
    df = pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'ref1': ['a', 'a', 'b', 'b', 'c', 'c'],
        'ref2': [10, 10, 20, 20, 30, 30]
    })
    result = identity_utils.analyze_identifier_consistency(df, 'id', ['ref1', 'ref2'])
    assert result['total_records'] == 6
    assert result['total_combinations'] == 3
    assert result['consistent_combinations'] == 3
    assert result['inconsistent_combinations'] == 0
    assert result['match_percentage'] == 100
    assert result['mismatch_count'] == 0
    assert result['mismatch_examples'] == []

def test_analyze_identifier_consistency_inconsistent():
    df = pd.DataFrame({
        'id': [1, 2, 2, 3],
        'ref1': ['a', 'a', 'a', 'b'],
        'ref2': [10, 10, 10, 20]
    })
    result = identity_utils.analyze_identifier_consistency(df, 'id', ['ref1', 'ref2'])
    assert result['inconsistent_combinations'] == 1
    assert result['mismatch_count'] == 1
    assert len(result['mismatch_examples']) == 1
    assert result['mismatch_examples'][0]['count'] == 3

def test_analyze_identifier_consistency_missing_id_field():
    df = pd.DataFrame({'foo': [1, 2, 3], 'ref1': ['a', 'b', 'c']})
    result = identity_utils.analyze_identifier_consistency(df, 'id', ['ref1'])
    assert result['error'].startswith('Field id not found')
    assert result['total_records'] == 3

def test_analyze_identifier_consistency_no_reference_fields():
    df = pd.DataFrame({'id': [1, 2, 3]})
    result = identity_utils.analyze_identifier_consistency(df, 'id', ['foo', 'bar'])
    assert result['error'].startswith('None of the reference fields')
    assert result['total_records'] == 3

def test_analyze_identifier_consistency_empty_df():
    df = pd.DataFrame({'id': [], 'ref1': []})
    result = identity_utils.analyze_identifier_consistency(df, 'id', ['ref1'])
    assert result['total_records'] == 0
    assert result['total_combinations'] == 0
    assert result['consistent_combinations'] == 0
    assert result['inconsistent_combinations'] == 0
    assert result['match_percentage'] == 0
    assert result['mismatch_count'] == 0
    assert result['mismatch_examples'] == []

def test_find_cross_matches_valid():
    df = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'ref1': ['a', 'a', 'a', 'b', 'b'],
        'ref2': [10, 10, 10, 20, 20]
    })
    result = identity_utils.find_cross_matches(df, 'id', ['ref1', 'ref2'])
    assert result['total_records'] == 5
    assert result['total_cross_matches'] == 2
    assert len(result['cross_match_examples']) == 2
    assert result['cross_match_examples'][0]['count'] == 3

def test_find_cross_matches_missing_id_field():
    df = pd.DataFrame({'foo': [1, 2, 3], 'ref1': ['a', 'b', 'c']})
    result = identity_utils.find_cross_matches(df, 'id', ['ref1'])
    assert result['error'].startswith('Field id not found')
    assert result['total_records'] == 3

def test_find_cross_matches_no_reference_fields():
    df = pd.DataFrame({'id': [1, 2, 3]})
    result = identity_utils.find_cross_matches(df, 'id', ['foo', 'bar'])
    assert result['error'].startswith('None of the reference fields')
    assert result['total_records'] == 3

def test_find_cross_matches_empty_df():
    df = pd.DataFrame({'id': [], 'ref1': []})
    result = identity_utils.find_cross_matches(df, 'id', ['ref1'])
    assert result['total_records'] == 0
    assert result['total_cross_matches'] == 0
    assert result['cross_match_examples'] == []

def test_find_cross_matches_all_unique():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'ref1': ['a', 'b', 'c'],
        'ref2': [10, 20, 30]
    })
    result = identity_utils.find_cross_matches(df, 'id', ['ref1', 'ref2'])
    assert result['total_cross_matches'] == 0
    assert result['cross_match_examples'] == []

if __name__ == "__main__":
    pytest.main()
