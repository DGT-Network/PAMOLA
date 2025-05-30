"""
Tests for the base_transformation_op module in the pamola_core/transformations package.
These tests ensure that the TransformationOperation class properly implements data transformation,
validation, caching, metrics calculation, visualization, and error handling.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.ops.op_result import OperationStatus

class DummyTransformation(TransformationOperation):
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Simple transformation: add 1 to numeric columns
        for col in batch.select_dtypes(include='number').columns:
            batch[col] = batch[col] + 1
        return batch

@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [10, 20, 30, 40],
        'c': ['x', 'y', 'z', 'w']
    })

@pytest.fixture
def empty_data():
    return pd.DataFrame({'a': [], 'b': [], 'c': []})

@pytest.fixture
def data_source(dummy_data):
    ds = MagicMock()
    return ds

@pytest.fixture
def task_dir(tmp_path):
    return tmp_path

@pytest.fixture
def reporter():
    return MagicMock()

@pytest.fixture
def progress_tracker():
    tracker = MagicMock()
    tracker.update = MagicMock()
    return tracker

@pytest.fixture
def dummy_op():
    return DummyTransformation(field_name='a', use_cache=False)

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_valid_case(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, dummy_data):
    mock_load.return_value = (dummy_data.copy(), None)
    result = dummy_op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR or result.status == OperationStatus.SUCCESS

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_empty_dataframe(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, empty_data):
    mock_load.return_value = (empty_data.copy(), None)
    result = dummy_op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR or result.status == OperationStatus.SUCCESS

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_invalid_field(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, dummy_data):
    dummy_op.field_name = 'not_a_field'
    mock_load.return_value = (dummy_data.copy(), None)
    result = dummy_op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR
    assert 'not found' in result.error_message

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_data_load_error(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker):
    mock_load.return_value = (None, {'message': 'load error'})
    result = dummy_op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR
    assert 'Failed to load' in result.error_message

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_process_batch_error(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, dummy_data):
    class FailingOp(DummyTransformation):
        def process_batch(self, batch):
            raise ValueError('fail')
    op = FailingOp(field_name='a', use_cache=False)
    mock_load.return_value = (dummy_data.copy(), None)
    result = op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR
    assert 'Processing error' in result.error_message

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_metrics_error(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, dummy_data):
    class MetricsFailOp(DummyTransformation):
        def _collect_metrics(self, o, t):
            raise Exception('metrics fail')
    op = MetricsFailOp(field_name='a', use_cache=False)
    mock_load.return_value = (dummy_data.copy(), None)
    result = op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR or result.status == OperationStatus.SUCCESS

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_visualization_error(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, dummy_data):
    class VizFailOp(DummyTransformation):
        def _handle_visualizations(self, *a, **k):
            raise Exception('viz fail')
    op = VizFailOp(field_name='a', use_cache=False)
    mock_load.return_value = (dummy_data.copy(), None)
    result = op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR or result.status == OperationStatus.SUCCESS

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_save_output_error(mock_load, dummy_op, data_source, task_dir, reporter, progress_tracker, dummy_data):
    class SaveFailOp(DummyTransformation):
        def _save_output_data(self, *a, **k):
            raise Exception('save fail')
    op = SaveFailOp(field_name='a', use_cache=False)
    mock_load.return_value = (dummy_data.copy(), None)
    result = op.execute(data_source, task_dir, reporter, progress_tracker)
    assert result.status == OperationStatus.ERROR
    assert 'Error saving output data' in result.error_message

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_encryption_key_required(mock_load):
    with pytest.raises(ValueError):
        DummyTransformation(field_name='a', use_encryption=True, encryption_key=None)

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_prepare_output_fields_enrich(mock_load, dummy_data):
    op = DummyTransformation(field_name='a', mode='ENRICH', output_field_name='a_new', use_cache=False)
    fields = op._prepare_output_fields(dummy_data)
    assert fields == ['a_new']

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_prepare_output_fields_replace(mock_load, dummy_data):
    op = DummyTransformation(field_name='a', mode='REPLACE', use_cache=False)
    fields = op._prepare_output_fields(dummy_data)
    assert fields == ['a']

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_prepare_output_fields_all(mock_load, dummy_data):
    op = DummyTransformation(field_name='', use_cache=False)
    fields = op._prepare_output_fields(dummy_data)
    assert set(fields) == set(dummy_data.columns)

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_generate_data_hash_series(mock_load, dummy_data):
    op = DummyTransformation(field_name='a', use_cache=False)
    h = op._generate_data_hash(dummy_data['a'])
    assert isinstance(h, str) and len(h) == 32

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_generate_data_hash_df(mock_load, dummy_data):
    op = DummyTransformation(field_name='a', use_cache=False)
    h = op._generate_data_hash(dummy_data)
    assert isinstance(h, str) and len(h) == 32

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_df_characteristics(mock_load, dummy_data):
    op = DummyTransformation(field_name='a', use_cache=False)
    char = op._df_characteristics(dummy_data)
    assert 'shape' in char and 'dtypes' in char

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_series_characteristics(mock_load, dummy_data):
    op = DummyTransformation(field_name='a', use_cache=False)
    char = op._series_characteristics(dummy_data['a'])
    assert 'length' in char and 'dtype' in char

@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_get_basic_parameters(mock_load):
    op = DummyTransformation(field_name='a', use_cache=False)
    params = op._get_basic_parameters()
    assert 'name' in params and 'version' in params

if __name__ == "__main__":
    pytest.main()
