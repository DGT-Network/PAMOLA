import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from pamola_core.profiling.commons import currency_utils

class TestCurrencyUtils:
    # is_currency_field
    @pytest.mark.parametrize("field_name,expected", [
        ("total_amount", True),
        ("price_usd", True),
        ("salary", True),
        ("random_field", False),
        ("", False),
        (None, False),
    ])
    def test_is_currency_field(self, field_name, expected):
        if field_name is None:
            with pytest.raises(TypeError):
                currency_utils.is_currency_field(field_name)
        else:
            assert currency_utils.is_currency_field(field_name) == expected

    # extract_currency_symbol
    @pytest.mark.parametrize("value,expected_value,expected_code", [
        ("$100", "100", "USD"),
        ("100$", "100", "USD"),
        ("100 €", "100", "EUR"),
        ("EUR 100", "100", "EUR"),
        ("100 EUR", "100", "EUR"),
        ("100", "100", None),
        (None, "", None),
        (123, "123", None),
        ("C$ 200", "200", "CAD"),
        ("R$100", "100", "BRL"),
        ("zł 50", "50", "PLN"),
        ("50 zł", "50", "PLN"),
    ])
    def test_extract_currency_symbol(self, value, expected_value, expected_code):
        cleaned, code = currency_utils.extract_currency_symbol(value)
        assert cleaned == expected_value
        assert code == expected_code

    # normalize_currency_value
    @pytest.mark.parametrize("value,locale,expected_val,expected_code,expected_valid", [
        ("$1,234.56", "en_US", 1234.56, "USD", True),
        ("1 234,56 €", "fr_FR", 1234.56, "EUR", True),
        ("-£2,000.00", "en_GB", -2000.0, None, True),
        ("1000", "en_US", 1000.0, None, True),
        (None, "en_US", None, None, False),
        ("", "en_US", None, None, False),
        (1234, "en_US", 1234.0, None, True),
        (Decimal("12.34"), "en_US", 12.34, None, True),
        ("notanumber", "en_US", None, None, False),
        ("CHF 1'234.56", "en_US", 1234.56, "CHF", True),
        ("1.234,56 €", "de_DE", 1234.56, "EUR", True),
        ("-1 234,56 ₽", "ru_RU", -1234.56, "RUB", True),
    ])
    def test_normalize_currency_value(self, value, locale, expected_val, expected_code, expected_valid):
        val, code, valid = currency_utils.normalize_currency_value(value, locale)
        if expected_val is None:
            assert val is None
        else:
            assert val == pytest.approx(expected_val)
        assert code == expected_code
        assert valid == expected_valid

    # parse_currency_field
    def test_parse_currency_field_valid(self):
        df = pd.DataFrame({
            'amount': ["$100", "$200", "$300", None, "notanumber"]
        })
        series, counts = currency_utils.parse_currency_field(df, 'amount')
        assert list(series)[:3] == [100.0, 200.0, 300.0]
        assert pd.isna(series[3])
        assert pd.isna(series[4])
        assert counts == {'USD': 3}
        assert hasattr(series, 'currencies')
        assert hasattr(series, 'valid_flags')
        assert series.currencies[:3] == ["USD", "USD", "USD"]
        assert series.valid_flags[:3] == [True, True, True]

    def test_parse_currency_field_invalid_column(self):
        df = pd.DataFrame({'foo': [1, 2, 3]})
        with pytest.raises(ValueError):
            currency_utils.parse_currency_field(df, 'bar')

    # analyze_currency_stats
    def test_analyze_currency_stats_normal(self):
        arr = pd.Series([1.0, 2.0, 3.0, 0.0, -1.0, None])
        counts = {'USD': 3, 'EUR': 2}
        stats = currency_utils.analyze_currency_stats(arr, counts)
        assert stats['min'] == -1.0
        assert stats['max'] == 3.0
        assert stats['mean'] == pytest.approx(1.0)
        assert stats['median'] == pytest.approx(1.0)
        assert stats['std'] > 0
        assert stats['valid_count'] == 5
        assert stats['zero_count'] == 1
        assert stats['negative_count'] == 1
        assert stats['zero_percentage'] == pytest.approx(20.0)
        assert stats['negative_percentage'] == pytest.approx(20.0)
        assert stats['currency_distribution'] == counts
        assert stats['multi_currency'] is True
        assert 'skewness' in stats
        assert 'kurtosis' in stats

    def test_analyze_currency_stats_empty(self):
        arr = pd.Series([None, None])
        stats = currency_utils.analyze_currency_stats(arr)
        assert stats['min'] is None
        assert stats['max'] is None
        assert stats['mean'] is None
        assert stats['median'] is None
        assert stats['std'] is None
        assert stats['valid_count'] == 0
        assert stats['zero_count'] == 0
        assert stats['negative_count'] == 0
        assert stats['zero_percentage'] == 0.0
        assert stats['negative_percentage'] == 0.0
        assert stats['currency_distribution'] == {}
        assert stats['multi_currency'] is False

    # detect_currency_from_sample
    def test_detect_currency_from_sample(self):
        df = pd.DataFrame({'amount': ["$100", "€200", "€300", "USD 400", "100", None]})
        code = currency_utils.detect_currency_from_sample(df, 'amount', sample_size=5)
        assert code in ("USD", "EUR")

    def test_detect_currency_from_sample_unknown(self):
        df = pd.DataFrame({'amount': ["foo", "bar", None]})
        code = currency_utils.detect_currency_from_sample(df, 'amount')
        assert code == 'UNKNOWN'

    def test_detect_currency_from_sample_missing_column(self):
        df = pd.DataFrame({'foo': [1, 2, 3]})
        code = currency_utils.detect_currency_from_sample(df, 'bar')
        assert code == 'UNKNOWN'

    # generate_currency_samples
    def test_generate_currency_samples_normal(self):
        stats = {
            'min': 0.0,
            'max': 100.0,
            'mean': 50.0,
            'std': 10.0,
            'valid_count': 10,
            'zero_count': 1,
            'negative_count': 2
        }
        samples = currency_utils.generate_currency_samples(stats, count=5)
        assert 'normal' in samples
        assert len(samples['normal']) == 5
        assert 'boundary' in samples
        assert samples['boundary'] == [0.0, 100.0]
        assert 'special' in samples
        assert 0.0 in samples['special']
        assert any(x < 0 for x in samples['special'])

    def test_generate_currency_samples_empty(self):
        stats = {'valid_count': 0}
        samples = currency_utils.generate_currency_samples(stats)
        assert samples == {}

    # create_empty_currency_stats
    def test_create_empty_currency_stats(self):
        stats = currency_utils.create_empty_currency_stats()
        assert stats['min'] is None
        assert stats['max'] is None
        assert stats['mean'] is None
        assert stats['valid_count'] == 0
        assert stats['currency_distribution'] == {}
        assert stats['multi_currency'] is False
        assert 'outliers' in stats
        assert 'normality' in stats

    # Edge: test parse_currency_field with all invalid values
    def test_parse_currency_field_all_invalid(self):
        df = pd.DataFrame({'amount': [None, "notanumber", "", " "]})
        series, counts = currency_utils.parse_currency_field(df, 'amount')
        assert all(x is None for x in series)
        assert counts == {}

    # Edge: test analyze_currency_stats with np.ndarray
    def test_analyze_currency_stats_numpy(self):
        arr = np.array([1.0, 2.0, 3.0, 0.0, -1.0, np.nan])
        stats = currency_utils.analyze_currency_stats(arr)
        assert stats['min'] == -1.0
        assert stats['max'] == 3.0
        assert stats['valid_count'] == 5

    # Edge: test generate_currency_samples with std=0
    def test_generate_currency_samples_zero_std(self):
        stats = {'min': 10.0, 'max': 10.0, 'mean': 10.0, 'std': 0.0, 'valid_count': 5, 'zero_count': 0, 'negative_count': 0}
        samples = currency_utils.generate_currency_samples(stats, count=3)
        assert 'normal' in samples
        assert all(x == 10.0 for x in samples['normal']) or all(10.0 <= x <= 10.0 for x in samples['normal'])
        assert samples['boundary'] == [10.0, 10.0]

    # Edge: test create_empty_currency_stats structure
    def test_create_empty_currency_stats_structure(self):
        stats = currency_utils.create_empty_currency_stats()
        assert isinstance(stats['outliers'], dict)
        assert isinstance(stats['normality'], dict)

    # Invalid: test normalize_currency_value with weird string
    def test_normalize_currency_value_weird_string(self):
        val, code, valid = currency_utils.normalize_currency_value("abc$def", "en_US")
        assert val is None
        assert code == "USD" or code is None
        assert not valid

if __name__ == "__main__":
    pytest.main()