import pytest
import pandas as pd
from unittest import mock
from pamola_core.profiling.commons import currency_analysis

class TestCurrencyAnalysis:
    def setup_method(self):
        self.df_valid = pd.DataFrame({
            'amount': [100, 200, 300, None, 500],
            'currency': ['USD', 'EUR', 'RUB', 'USD', None]
        })
        self.exchange_rates = {'USD': 70, 'EUR': 80, 'RUB': 1}
        self.base_currency = 'RUB'

    def test_analyze_currency_field_no_currency_column(self):
        with mock.patch('pamola_core.profiling.analyzers.numeric.NumericAnalyzer') as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze.return_value.stats = {'mean': 250, 'std': 158.11}
            result = currency_analysis.analyze_currency_field(self.df_valid, 'amount', None)
            assert 'mean' in result
            assert 'std' in result
            assert 'currency_dist' not in result

    def test_analyze_currency_field_missing_value_field(self):
        df = pd.DataFrame({'foo': [1, 2, 3]})
        result = currency_analysis.analyze_currency_field(df, 'not_present')
        assert 'error' in result
        assert 'not_present' in result['error']

    def test_analyze_salary_field_delegates(self):
        with mock.patch('pamola_core.profiling.commons.currency_analysis.analyze_currency_field') as mock_func:
            mock_func.return_value = {'test': 1}
            result = currency_analysis.analyze_salary_field(self.df_valid, 'amount', 'currency')
            assert result == {'test': 1}
            mock_func.assert_called_once()

    def test_convert_currencies_valid(self):
        result = currency_analysis.convert_currencies(
            self.df_valid, 'amount', 'currency', self.exchange_rates, self.base_currency
        )
        # Only non-None values and currencies in exchange_rates or base_currency
        assert isinstance(result, pd.Series)
        assert len(result) == 3  # Only 3 valid rows for conversion
        assert all(isinstance(x, (int, float)) for x in result)

    def test_convert_currencies_empty(self):
        df = pd.DataFrame({'amount': [], 'currency': []})
        result = currency_analysis.convert_currencies(df, 'amount', 'currency', self.exchange_rates)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_convert_currencies_missing_exchange_rate(self):
        df = pd.DataFrame({'amount': [100], 'currency': ['GBP']})
        result = currency_analysis.convert_currencies(df, 'amount', 'currency', self.exchange_rates)
        assert result.empty

    def test_convert_currencies_missing_fields(self):
        df = pd.DataFrame({'foo': [1]})
        with pytest.raises(KeyError):
            currency_analysis.convert_currencies(df, 'amount', 'currency', self.exchange_rates)

    def test_convert_currencies_nan_values(self):
        df = pd.DataFrame({'amount': [None, 100], 'currency': ['USD', None]})
        result = currency_analysis.convert_currencies(df, 'amount', 'currency', self.exchange_rates)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def teardown_method(self):
        pass

if __name__ == "__main__":
    pytest.main()