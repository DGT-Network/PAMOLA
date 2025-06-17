import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pamola_core.profiling.analyzers.currency import CurrencyAnalyzer, CurrencyOperation
from pamola_core.utils.ops.op_result import OperationStatus


class DummyReporter:
    def add_operation(self, *args, **kwargs): pass
    def add_artifact(self, *args, **kwargs): pass


@pytest.fixture
def currency_df():
    return pd.DataFrame({
        'amount': ['$10.00', '$20.00', 'N/A', '--', '$30.00', None, 'invalid', '-$5.00', '$0.00']
    })


@pytest.fixture
def tmp_csv(currency_df, tmp_path):
    csv_path = tmp_path / 'test_currency.csv'
    currency_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def dummy_reporter():
    return DummyReporter()


def mock_currency_analysis_dependencies():
    return patch.multiple(
        'pamola_core.profiling.commons.currency_utils',
        parse_currency_field=MagicMock(return_value=(
            pd.Series([10.0, 20.0, None, None, 30.0, None, None, -5.0, 0.0], name="amount"),
            {'USD': 9}
        )),
        analyze_currency_stats=MagicMock(return_value={
            'min': -5.0, 'max': 30.0, 'mean': 11.25, 'negative_count': 1, 'zero_count': 1
        }),
        generate_currency_samples=MagicMock(return_value=[10.0, 20.0])
    )


def mock_numeric_util_dependencies():
    return patch.multiple(
        'pamola_core.profiling.commons.numeric_utils',
        calculate_percentiles=MagicMock(return_value={'50': 10.0}),
        calculate_histogram=MagicMock(return_value=[(0, 10)]),
        detect_outliers=MagicMock(return_value={'count': 0}),
        test_normality=MagicMock(return_value={'is_normal': True})
    )


def assert_result_metrics(result):
    assert result['valid_count'] == 5
    assert result['null_count'] == 3
    assert result['invalid_count'] == 1
    assert result['stats']['min'] == -5.0
    assert result['stats']['max'] == 30.0
    assert result['stats']['mean'] == 11.25
    assert result['stats']['percentiles']['50'] == 10.0
    assert result['stats']['histogram'] == [(0, 10)]
    assert result['stats']['outliers']['count'] == 0
    assert result['stats']['normality']['is_normal'] is True
    assert result['stats']['samples'] == [10.0, 20.0]
    assert 'semantic_notes' in result['stats']


def test_currency_analyzer_dataframe_success(currency_df):
    analyzer = CurrencyAnalyzer()
    with mock_currency_analysis_dependencies(), mock_numeric_util_dependencies():
        result = analyzer.analyze(currency_df, 'amount')
        assert result['field_name'] == 'amount'
        assert_result_metrics(result)


@pytest.mark.parametrize("input_type", ["dataframe", "csv"])
def test_currency_operation_execute_success(input_type, currency_df, tmp_csv, tmp_path, dummy_reporter):
    data_source = MagicMock()
    data_source.__class__.__name__ = 'DataSource'
    input_data = currency_df if input_type == "dataframe" else pd.read_csv(tmp_csv)

    with patch('pamola_core.profiling.analyzers.currency.load_data_operation', return_value=input_data), \
         patch.object(CurrencyAnalyzer, 'analyze', return_value={
             'field_name': 'amount',
             'valid_count': 5,
             'null_count': 3,
             'invalid_count': 1,
             'stats': {
                 'min': -5.0, 'max': 30.0, 'mean': 11.25,
                 'percentiles': {'50': 10.0},
                 'histogram': [(0, 10)],
                 'outliers': {'count': 0},
                 'normality': {'is_normal': True},
                 'samples': [10.0, 20.0],
                 'semantic_notes': []
             },
             'currency_counts': {'USD': 9},
             'multi_currency': False
         }) as mock_analyze, \
         patch('pamola_core.profiling.analyzers.currency.write_json') as mock_json, \
         patch('pamola_core.profiling.analyzers.currency.write_dataframe_to_csv') as mock_csv, \
         patch('pamola_core.profiling.analyzers.currency.create_histogram') as mock_hist, \
         patch('pamola_core.profiling.analyzers.currency.create_boxplot') as mock_box, \
         patch('pamola_core.profiling.analyzers.currency.create_correlation_pair') as mock_corr:

        op = CurrencyOperation(field_name='amount')
        result = op.execute(data_source, tmp_path, dummy_reporter)
        assert result.status == OperationStatus.SUCCESS
        assert mock_analyze.called
        assert mock_json.called
        assert mock_csv.called
        assert mock_hist.called or mock_box.called or mock_corr.called


def test_currency_analyzer_handles_malformed_and_missing(currency_df):
    analyzer = CurrencyAnalyzer()
    with mock_currency_analysis_dependencies(), mock_numeric_util_dependencies():
        result = analyzer.analyze(currency_df, 'amount')
        assert_result_metrics(result)
