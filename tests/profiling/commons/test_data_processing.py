import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pamola_core.profiling.commons import data_processing

class TestDataProcessing:
    def setup_method(self):
        self.df_numeric = pd.DataFrame({
            'a': [1, 2, 3, np.nan, '4', 'bad', None],
            'b': ['x', 'y', 'z', None, '', 'foo', 'bar'],
            'date': pd.to_datetime(['2020-01-01', None, '2021-01-01', 'notadate', pd.NaT, '2022-01-01', '2023-01-01'], errors='coerce'),
            'email': ['test@example.com', 'foo@bar.com', None, '', 'notanemail', 'user@domain.com', np.nan],
        })

    def test_prepare_numeric_data_valid(self):
        series, null_count, non_null_count = data_processing.prepare_numeric_data(self.df_numeric, 'a')
        assert isinstance(series, pd.Series)
        assert null_count == 2  # np.nan and None
        assert non_null_count == 5
        assert set(series.values) == {1, 2, 3, 4}

    def test_prepare_numeric_data_field_not_exist(self):
        series, null_count, non_null_count = data_processing.prepare_numeric_data(self.df_numeric, 'not_a_field')
        assert series.empty
        assert null_count == 0
        assert non_null_count == 0

    def test_prepare_numeric_data_all_null(self):
        df = pd.DataFrame({'a': [None, np.nan, None]})
        series, null_count, non_null_count = data_processing.prepare_numeric_data(df, 'a')
        assert series.empty
        assert null_count == 3
        assert non_null_count == 0

    def test_prepare_numeric_data_non_numeric(self):
        df = pd.DataFrame({'a': ['foo', 'bar', None]})
        series, null_count, non_null_count = data_processing.prepare_numeric_data(df, 'a')
        assert series.empty
        assert null_count == 1
        assert non_null_count == 2

    def test_prepare_field_for_analysis_numeric(self):
        # The 'a' column in self.df_numeric contains mixed types (int, str, nan, 'bad'), so pandas treats it as object dtype.
        # The function will return 'text' for object dtype unless all values are numeric.
        # To test the numeric case, use a DataFrame with only numeric values.
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        series, dtype = data_processing.prepare_field_for_analysis(df, 'a')
        assert dtype == 'numeric'
        assert isinstance(series, pd.Series)

    def test_prepare_field_for_analysis_date(self):
        series, dtype = data_processing.prepare_field_for_analysis(self.df_numeric, 'date')
        assert dtype == 'date'
        assert isinstance(series, pd.Series)

    def test_prepare_field_for_analysis_email(self):
        series, dtype = data_processing.prepare_field_for_analysis(self.df_numeric, 'email')
        assert dtype == 'email'
        assert isinstance(series, pd.Series)

    def test_prepare_field_for_analysis_text(self):
        series, dtype = data_processing.prepare_field_for_analysis(self.df_numeric, 'b')
        assert dtype == 'text'
        assert isinstance(series, pd.Series)

    def test_prepare_field_for_analysis_field_not_exist(self):
        series, dtype = data_processing.prepare_field_for_analysis(self.df_numeric, 'not_a_field')
        assert series.empty
        assert dtype == 'unknown'

    def test_prepare_field_for_analysis_object_unknown(self):
        df = pd.DataFrame({'a': [None, None]})
        series, dtype = data_processing.prepare_field_for_analysis(df, 'a')
        assert dtype == 'text' or dtype == 'unknown'

    @mock.patch('pamola_core.utils.progress.process_dataframe_in_chunks')
    @mock.patch('pamola_core.profiling.commons.numeric_utils.combine_chunk_results')
    def test_handle_large_dataframe_valid(self, mock_combine, mock_chunks):
        mock_chunks.return_value = [{'result': 1}, {'result': 2}]
        mock_combine.return_value = {'result': 3}
        def op(chunk, field, **kwargs):
            return {'result': chunk[field].sum()}
        df = pd.DataFrame({'a': [1, 2, 3, 4]})
        result = data_processing.handle_large_dataframe(df, 'a', op, chunk_size=2)
        assert result == {'result': 3}
        mock_chunks.assert_called()
        mock_combine.assert_called_with([{'result': 1}, {'result': 2}])

    @mock.patch('pamola_core.utils.progress.process_dataframe_in_chunks')
    @mock.patch('pamola_core.profiling.commons.numeric_utils.combine_chunk_results')
    def test_handle_large_dataframe_field_not_exist(self, mock_combine, mock_chunks):
        mock_chunks.return_value = []
        mock_combine.return_value = {}
        def op(chunk, field, **kwargs):
            return {'result': 0}
        df = pd.DataFrame({'b': [1, 2]})
        with pytest.raises(KeyError):
            data_processing.handle_large_dataframe(df, 'a', op)

    def test_handle_large_dataframe_invalid_operation(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        # The function expects a callable, passing None will raise a TypeError when called inside a lambda
        def op(chunk, field, **kwargs):
            return None
        # Instead of passing None, pass a function that raises TypeError
        with pytest.raises(TypeError):
            data_processing.handle_large_dataframe(df, 'a', lambda *args, **kwargs: (_ for _ in ()).throw(TypeError()))

    def test_prepare_numeric_data_invalid_df(self):
        # If df is None, accessing df.columns will raise AttributeError
        with pytest.raises(AttributeError):
            data_processing.prepare_numeric_data(None, 'a')

    def test_prepare_field_for_analysis_invalid_df(self):
        # If df is None, accessing df.columns will raise AttributeError
        with pytest.raises(AttributeError):
            data_processing.prepare_field_for_analysis(None, 'a')

if __name__ == "__main__":
    pytest.main()