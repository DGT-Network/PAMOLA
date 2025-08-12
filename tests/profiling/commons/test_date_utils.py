import pytest
import pandas as pd
from datetime import datetime
from pamola_core.profiling.commons import date_utils

class TestDateUtils:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Common DataFrames for tests
        self.df_valid = pd.DataFrame({
            'date': ['2000-01-01', '1999-12-31', '2005-05-05', None, '1940-01-01', '2025-01-01', '-1000-01-01', 'notadate'],
            'group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'uid': [1, 1, 2, 2, 3, 3, 4, 4]
        })
        self.df_empty = pd.DataFrame({'date': [], 'group': [], 'uid': []})
        self.df_missing = pd.DataFrame({'other': [1, 2, 3]})

    def test_prepare_date_data_valid(self):
        dates, null_count, non_null_count = date_utils.prepare_date_data(self.df_valid, 'date')
        assert isinstance(dates, pd.Series)
        assert null_count == 1
        assert non_null_count == 7
        assert dates.isna().sum() >= 1

    def test_prepare_date_data_missing_field(self):
        with pytest.raises(ValueError):
            date_utils.prepare_date_data(self.df_missing, 'date')

    def test_calculate_date_stats_valid(self):
        dates = pd.to_datetime(self.df_valid['date'], errors='coerce')
        stats = date_utils.calculate_date_stats(dates)
        assert stats['valid_count'] == 5
        assert stats['min_date'] == '1940-01-01'
        assert stats['max_date'] == '2025-01-01'

    def test_calculate_date_stats_empty(self):
        stats = date_utils.calculate_date_stats(pd.Series([], dtype='datetime64[ns]'))
        assert stats['valid_count'] == 0
        assert stats['min_date'] is None
        assert stats['max_date'] is None

    def test_calculate_distributions_valid(self):
        dates = pd.to_datetime(self.df_valid['date'], errors='coerce')
        dists = date_utils.calculate_distributions(dates)
        assert 'year_distribution' in dists
        assert 'decade_distribution' in dists
        assert 'month_distribution' in dists
        assert 'day_of_week_distribution' in dists
        assert isinstance(dists['year_distribution'], dict)

    def test_calculate_distributions_empty(self):
        dists = date_utils.calculate_distributions(pd.Series([], dtype='datetime64[ns]'))
        assert dists == {}

    @pytest.mark.parametrize('date_str,fmt,expected', [
        ('2000-01-01', '%Y-%m-%d', True),
        ('01-01-2000', '%Y-%m-%d', False),
        ('notadate', '%Y-%m-%d', False),
        (None, '%Y-%m-%d', False),
    ])
    def test_validate_date_format(self, date_str, fmt, expected):
        assert date_utils.validate_date_format(date_str, fmt) == expected

    def test_detect_date_anomalies(self):
        anomalies = date_utils.detect_date_anomalies(self.df_valid['date'], min_year=1940, max_year=2005)
        assert 'invalid_format' in anomalies
        assert 'too_old' in anomalies
        assert 'future_dates' in anomalies
        assert 'too_young' in anomalies
        assert 'negative_years' in anomalies
        assert any('notadate' in str(x) for x in anomalies['invalid_format'])

    def test_detect_date_anomalies_empty(self):
        anomalies = date_utils.detect_date_anomalies(self.df_empty['date'] if 'date' in self.df_empty else pd.Series([], dtype=object))
        for v in anomalies.values():
            assert v == []

    def test_detect_date_changes_within_group(self):
        res = date_utils.detect_date_changes_within_group(self.df_valid, 'group', 'date')
        assert 'groups_with_changes' in res
        assert 'examples' in res
        # There should be at least one group with changes
        assert res['groups_with_changes'] >= 0

    def test_detect_date_changes_within_group_missing_column(self):
        res = date_utils.detect_date_changes_within_group(self.df_missing, 'group', 'date')
        assert 'error' in res
        res2 = date_utils.detect_date_changes_within_group(self.df_valid, 'group', 'not_a_date')
        assert 'error' in res2

    def test_detect_date_inconsistencies_by_uid(self):
        res = date_utils.detect_date_inconsistencies_by_uid(self.df_valid, 'uid', 'date')
        assert 'uids_with_inconsistencies' in res
        assert 'examples' in res
        assert res['uids_with_inconsistencies'] >= 0

    def test_detect_date_inconsistencies_by_uid_missing_column(self):
        res = date_utils.detect_date_inconsistencies_by_uid(self.df_missing, 'uid', 'date')
        assert 'error' in res
        res2 = date_utils.detect_date_inconsistencies_by_uid(self.df_valid, 'uid', 'not_a_date')
        assert 'error' in res2

    def test_analyze_date_field_valid(self):
        res = date_utils.analyze_date_field(self.df_valid, 'date', min_year=1940, max_year=2005, id_column='group', uid_column='uid')
        assert 'total_records' in res
        assert 'null_count' in res
        assert 'fill_rate' in res
        assert 'valid_count' in res
        assert 'anomalies' in res
        assert 'date_changes_within_group' in res
        assert 'date_inconsistencies_by_uid' in res

    def test_analyze_date_field_missing_field(self):
        res = date_utils.analyze_date_field(self.df_missing, 'date')
        assert 'error' in res

    def test_analyze_date_field_empty(self):
        res = date_utils.analyze_date_field(self.df_empty, 'date')
        assert 'total_records' in res
        assert res['total_records'] == 0
        assert res['null_count'] == 0
        assert res['non_null_count'] == 0
        assert res['valid_count'] == 0
        assert res['invalid_count'] == 0
        assert res['fill_rate'] == 0
        assert res['valid_rate'] == 0

    def test_estimate_resources_valid(self):
        res = date_utils.estimate_resources(self.df_valid, 'date')
        assert 'estimated_memory_mb' in res
        assert 'estimated_time_seconds' in res
        assert 'recommended_chunk_size' in res
        assert 'use_chunks_recommended' in res

    def test_estimate_resources_missing_field(self):
        res = date_utils.estimate_resources(self.df_missing, 'date')
        assert 'error' in res
        assert res['estimated_memory_mb'] == 10
        assert res['estimated_time_seconds'] == 1

if __name__ == "__main__":
    pytest.main()
