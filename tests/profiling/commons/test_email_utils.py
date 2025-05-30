import pytest
import pandas as pd
import numpy as np
from pamola_core.profiling.commons import email_utils

class TestEmailUtils:
    # is_valid_email
    def test_is_valid_email_valid(self):
        assert email_utils.is_valid_email('user@example.com')
        assert email_utils.is_valid_email('user.name+tag@sub.domain.co')

    def test_is_valid_email_invalid(self):
        assert not email_utils.is_valid_email('userexample.com')
        assert not email_utils.is_valid_email('user@.com')
        assert not email_utils.is_valid_email('user@com')
        assert not email_utils.is_valid_email('user@domain.')
        assert not email_utils.is_valid_email('')
        assert not email_utils.is_valid_email(None)
        assert not email_utils.is_valid_email(np.nan)
        assert not email_utils.is_valid_email(12345)

    # extract_email_domain
    def test_extract_email_domain_valid(self):
        assert email_utils.extract_email_domain('user@example.com') == 'example.com'
        assert email_utils.extract_email_domain('user@EXAMPLE.COM') == 'example.com'
        assert email_utils.extract_email_domain(' user@sub.domain.com ') == 'sub.domain.com'

    def test_extract_email_domain_invalid(self):
        assert email_utils.extract_email_domain('userexample.com') is None
        assert email_utils.extract_email_domain('user@') is None
        assert email_utils.extract_email_domain('@domain.com') == 'domain.com'
        assert email_utils.extract_email_domain('') is None
        assert email_utils.extract_email_domain(None) is None
        assert email_utils.extract_email_domain(123) is None

    # detect_personal_patterns
    def test_detect_personal_patterns_typical(self):
        emails = pd.Series([
            'john.doe@example.com',
            'jane_doe@example.com',
            'doe.john@example.com',
            'doe_jane@example.com',
            'janedoe@example.com',
            'doejane@example.com',
            'invalidemail',
            None,
            np.nan
        ])
        result = email_utils.detect_personal_patterns(emails)
        assert 'total_valid_emails' in result
        assert result['total_valid_emails'] == 6
        assert isinstance(result['pattern_counts'], dict)
        assert isinstance(result['pattern_percentages'], dict)

    def test_detect_personal_patterns_empty(self):
        emails = pd.Series([])
        result = email_utils.detect_personal_patterns(emails)
        assert result['total_valid_emails'] == 0
        for v in result['pattern_counts'].values():
            assert v == 0
        for v in result['pattern_percentages'].values():
            assert v == 0

    def test_detect_personal_patterns_all_invalid(self):
        emails = pd.Series(['', None, np.nan, 'notanemail'])
        result = email_utils.detect_personal_patterns(emails)
        assert result['total_valid_emails'] == 0

    # analyze_email_field
    def test_analyze_email_field_valid(self):
        df = pd.DataFrame({
            'email': [
                'john.doe@example.com',
                'jane_doe@example.com',
                'user@domain.com',
                'invalid',
                None,
                np.nan
            ]
        })
        stats = email_utils.analyze_email_field(df, 'email', top_n=2)
        assert stats['total_rows'] == 6
        assert stats['null_count'] == 2
        assert stats['valid_count'] == 3
        assert 'top_domains' in stats
        assert isinstance(stats['personal_patterns'], dict)

    def test_analyze_email_field_missing_column(self):
        df = pd.DataFrame({'other': ['a', 'b']})
        stats = email_utils.analyze_email_field(df, 'email')
        assert 'error' in stats

    def test_analyze_email_field_all_invalid(self):
        df = pd.DataFrame({'email': ['notanemail', '', None]})
        stats = email_utils.analyze_email_field(df, 'email')
        assert stats['valid_count'] == 0
        assert stats['invalid_count'] == 2

    # create_domain_dictionary
    def test_create_domain_dictionary_valid(self):
        df = pd.DataFrame({'email': [
            'a@x.com', 'b@x.com', 'c@y.com', 'd@z.com', None, np.nan
        ]})
        result = email_utils.create_domain_dictionary(df, 'email', min_count=1)
        assert result['field_name'] == 'email'
        assert result['total_domains'] == 3
        assert result['total_emails'] == 4
        assert any(d['domain'] == 'x.com' for d in result['domains'])
        assert all('percentage' in d for d in result['domains'])

    def test_create_domain_dictionary_min_count(self):
        df = pd.DataFrame({'email': [
            'a@x.com', 'b@x.com', 'c@y.com', 'd@z.com', 'e@z.com', 'f@z.com'
        ]})
        result = email_utils.create_domain_dictionary(df, 'email', min_count=3)
        assert result['total_domains'] == 1
        assert result['domains'][0]['domain'] == 'z.com'

    def test_create_domain_dictionary_missing_column(self):
        df = pd.DataFrame({'other': ['a', 'b']})
        result = email_utils.create_domain_dictionary(df, 'email')
        assert 'error' in result

    def test_create_domain_dictionary_all_invalid(self):
        df = pd.DataFrame({'email': ['notanemail', '', None]})
        result = email_utils.create_domain_dictionary(df, 'email')
        assert result['total_domains'] == 0
        assert result['total_emails'] == 0
        assert result['domains'] == []

    # estimate_resources
    def test_estimate_resources_valid(self):
        df = pd.DataFrame({'email': [
            'a@x.com', 'b@x.com', 'c@y.com', 'd@z.com', None, np.nan
        ]})
        result = email_utils.estimate_resources(df, 'email')
        assert result['total_rows'] == 6
        assert result['non_null_count'] == 4
        assert result['estimated_valid_emails'] == int(4 * 0.95)
        assert 'estimated_unique_domains' in result
        assert 'estimated_memory_mb' in result
        assert 'estimated_processing_time_sec' in result

    def test_estimate_resources_missing_column(self):
        df = pd.DataFrame({'other': ['a', 'b']})
        result = email_utils.estimate_resources(df, 'email')
        assert 'error' in result

    def test_estimate_resources_all_null(self):
        df = pd.DataFrame({'email': [None, np.nan]})
        result = email_utils.estimate_resources(df, 'email')
        assert result['non_null_count'] == 0
        assert result['estimated_valid_emails'] == 0
        assert result['estimated_unique_domains'] == 0

if __name__ == "__main__":
    pytest.main()
