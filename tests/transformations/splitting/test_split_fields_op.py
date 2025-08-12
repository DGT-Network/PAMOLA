"""
Tests for the split_fields_op module in the pamola_core/transformations/splitting package.
These tests ensure that the SplitFieldsOperation class properly implements field splitting,
parameter validation, metrics collection, caching, and output saving, including edge and error cases.
"""
import os
from datetime import datetime
import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch
from pamola_core.transformations.splitting.split_fields_op import SplitFieldsOperation, OutputFormat
from pamola_core.utils.ops.op_result import OperationStatus, OperationResult

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'a': [10, 20, 30],
        'b': [100, 200, 300],
        'c': ['x', 'y', 'z']
    })

@pytest.fixture
def field_groups():
    return {
        'group1': ['a', 'b'],
        'group2': ['c']
    }

@pytest.fixture
def operation(field_groups):
    op = SplitFieldsOperation(
        id_field='id',
        field_groups=field_groups,
        include_id_field=True,
        output_format=OutputFormat.CSV.value,
        save_output=True,
        use_cache=False
    )
    # Set default attributes for direct method calls
    op.timestamp = 'testts'
    op._input_dataset = 'test.csv'
    op.start_time = 0
    op.end_time = 1
    return op

class DummyDataSource:
    pass

class DummyReporter:
    def __init__(self):
        self.operations = []
    def add_operation(self, *args, **kwargs):
        self.operations.append((args, kwargs))

class DummyProgress:
    def __init__(self):
        self.total = 0
        self.updates = []
    def update(self, val, info):
        self.updates.append((val, info))

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation')
def test_execute_valid_case(mock_load_data, mock_load_settings, operation, sample_dataframe):
    mock_load_data.return_value = sample_dataframe
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = Path('test_task_dir')
    with patch.object(operation, '_prepare_directories', return_value={'output': task_dir, 'metrics': task_dir}):
        with patch.object(operation, '_save_output') as mock_save_output, \
             patch.object(operation, '_save_metrics') as mock_save_metrics, \
             patch.object(operation, '_generate_visualizations') as mock_vis, \
             patch.object(operation, '_save_cache') as mock_cache:
            result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
            assert isinstance(result, OperationResult)
            assert result.status == OperationStatus.SUCCESS
            assert mock_save_output.called
            assert mock_save_metrics.called
            assert mock_vis.called
            # Remove cache assertion: use_cache=False in fixture
            assert any('Operation completed successfully' in str(op[1]) for op in reporter.operations)

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation')
def test_execute_empty_dataframe(mock_load_data, mock_load_settings, operation):
    mock_load_data.return_value = pd.DataFrame()
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = Path('test_task_dir')
    with patch.object(operation, '_prepare_directories', return_value={'output': task_dir, 'metrics': task_dir}):
        result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
        assert result.status == OperationStatus.ERROR
        assert 'validate input parameters failed' in (result.error_message or '').lower()

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation')
def test_execute_invalid_field_groups(mock_load_data, mock_load_settings, operation, sample_dataframe):
    # field_groups has a field not in df
    operation.field_groups = {'group1': ['not_a_field']}
    mock_load_data.return_value = sample_dataframe
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = Path('test_task_dir')
    with patch.object(operation, '_prepare_directories', return_value={'output': task_dir, 'metrics': task_dir}):
        result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
        assert result.status == OperationStatus.ERROR
        assert 'validate input parameters failed' in (result.error_message or '').lower()

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation')
def test_execute_cache_hit(mock_load_data, mock_load_settings, operation, sample_dataframe):
    mock_load_data.return_value = sample_dataframe
    progress = DummyProgress()
    task_dir = Path('test_task_dir')
    # Patch _get_cache to return a dict as the real cache does
    fake_cache = OperationResult(
        status=OperationStatus.SUCCESS,
        metrics={},
        error_message=None,
        execution_time=1.0,
        error_trace=None,
        artifacts=[
                {'type': 'csv', 'path': 'foo.csv', 'description': 'desc', 'category': 'output', 'tags': []}
            ]
        )
    with patch.object(operation, '_prepare_directories', return_value={'output': task_dir, 'metrics': task_dir}):
        with patch.object(operation, '_get_cache', return_value=fake_cache) as mock_cache:
            operation.use_cache = True
            result = operation.execute(DummyDataSource(), task_dir, None, progress)
            assert isinstance(result, OperationResult)
            assert mock_cache.called

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation')
def test_execute_exception_handling(mock_load_data, mock_load_settings, operation, sample_dataframe):
    mock_load_data.side_effect = Exception('load error')
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = Path('test_task_dir')
    with patch.object(operation, '_prepare_directories', return_value={'output': task_dir, 'metrics': task_dir}):
        result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
        assert result.status == OperationStatus.ERROR
        assert 'load error' in (result.error_message or '')

def test_process_data_normal(operation, sample_dataframe):
    result = operation._process_data(sample_dataframe)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'group1', 'group2'}
    assert all(isinstance(df, pd.DataFrame) for df in result.values())
    # id field should be included
    assert 'id' in result['group1'].columns
    assert 'id' in result['group2'].columns

def test_process_data_no_id(operation, sample_dataframe):
    operation.include_id_field = False
    result = operation._process_data(sample_dataframe)
    assert 'id' not in result['group1'].columns or 'id' not in result['group2'].columns

def test_collect_metrics(operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    # Ensure required attributes are set
    operation._input_dataset = 'test.csv'
    operation.start_time = 0
    operation.end_time = 1
    metrics = operation._collect_metrics(sample_dataframe, output)
    assert isinstance(metrics, dict)
    assert metrics['total_input_records'] == 3
    assert metrics['number_of_splits'] == 2
    assert 'split_info' in metrics

def test_save_metrics_success(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    # Ensure required attributes are set
    operation._input_dataset = 'test.csv'
    operation.start_time = 0
    operation.end_time = 1
    metrics = operation._collect_metrics(sample_dataframe, output)
    result = OperationResult(status=OperationStatus.SUCCESS)
    with patch('pamola_core.transformations.splitting.split_fields_op.write_json') as mock_write_json:
        path = operation._save_metrics(metrics, tmp_path, result)
        assert mock_write_json.called
        assert path.exists() or isinstance(path, Path)

def test_save_metrics_failure(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    # Ensure required attributes are set
    operation._input_dataset = 'test.csv'
    operation.start_time = 0
    operation.end_time = 1
    metrics = operation._collect_metrics(sample_dataframe, output)
    result = OperationResult(status=OperationStatus.SUCCESS)
    with patch('pamola_core.transformations.splitting.split_fields_op.write_json', side_effect=Exception('fail')):
        with pytest.raises(Exception):
            operation._save_metrics(metrics, tmp_path, result)

def test_save_output_csv(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.timestamp = 'testts'
    with patch('pamola_core.transformations.splitting.split_fields_op.write_dataframe_to_csv') as mock_write_csv, \
         patch('pamola_core.transformations.splitting.split_fields_op.get_encryption_mode', return_value=None):
        operation._save_output(output, tmp_path, result)
        assert mock_write_csv.called

def test_save_output_json_encrypted(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = OutputFormat.JSON.value
    operation.timestamp = 'testts'
    with patch('pamola_core.transformations.splitting.split_fields_op.crypto_utils.encrypt_file') as mock_encrypt, \
         patch('pamola_core.transformations.splitting.split_fields_op.get_encryption_mode', return_value=None), \
         patch('pamola_core.transformations.splitting.split_fields_op.directory_utils.safe_remove_temp_file') as mock_rm, \
         patch('pandas.DataFrame.to_json') as mock_to_json:
        operation._save_output(output, tmp_path, result, use_encryption=True, encryption_key='key')
        assert mock_encrypt.called
        assert mock_rm.called
        assert mock_to_json.called

def test_save_output_json_no_encryption(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = OutputFormat.JSON.value
    operation.timestamp = 'testts'
    with patch('pamola_core.transformations.splitting.split_fields_op.get_encryption_mode', return_value=None):
        # Should not raise
        operation._save_output(output, tmp_path, result, use_encryption=False, encryption_key=None)

def test_save_output_unsupported_format(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = 'xml'  # unsupported
    operation.timestamp = 'testts'
    with patch('pamola_core.transformations.splitting.split_fields_op.get_encryption_mode', return_value=None):
        # Should not raise
        operation._save_output(output, tmp_path, result)

def test_generate_visualizations_all(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.timestamp = 'testts'
    with patch('pamola_core.transformations.splitting.split_fields_op.create_bar_plot', return_value=str(tmp_path/'bar.png')) as mock_bar, \
         patch('pamola_core.transformations.splitting.split_fields_op.plot_field_subset_network', return_value=str(tmp_path/'net.png')) as mock_net:
        operation._generate_visualizations(sample_dataframe, output, tmp_path, result)
        assert mock_bar.called
        assert mock_net.called

def test_generate_visualizations_empty_output(tmp_path, operation, sample_dataframe):
    result = OperationResult(status=OperationStatus.SUCCESS)
    # output_data is empty dict
    with patch('pamola_core.transformations.splitting.split_fields_op.create_bar_plot') as mock_bar:
        operation._generate_visualizations(sample_dataframe, {}, tmp_path, result)
        assert not mock_bar.called

def test_save_cache_success(tmp_path, operation, sample_dataframe):
    operation._original_df = sample_dataframe
    result = OperationResult(status=OperationStatus.SUCCESS)
    with patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.generate_cache_key', return_value='key'), \
         patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.save_cache') as mock_save:
        operation._save_cache(tmp_path, result)
        assert mock_save.called

def test_save_cache_failure(tmp_path, operation, sample_dataframe):
    operation._original_df = sample_dataframe
    result = OperationResult(status=OperationStatus.SUCCESS)
    with patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.generate_cache_key', return_value='key'), \
         patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.save_cache', side_effect=Exception('fail')):
        # Should not raise
        operation._save_cache(tmp_path, result)

def test_get_cache_hit(operation, sample_dataframe):
    with patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.generate_cache_key', return_value='key'), \
         patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.get_cache', return_value={
            'result': {
                'status': 'SUCCESS',
                'metrics': {},
                'error_message': None,
                'execution_time': 1.0,
                'error_trace': None,
                'artifacts': [
                    {'type': 'csv', 'path': 'foo.csv', 'description': 'desc', 'category': 'output', 'tags': []}
                ]
            }
         }):
        result = operation._get_cache(sample_dataframe)
        assert isinstance(result, OperationResult)
        assert result.status == OperationStatus.SUCCESS

def test_get_cache_miss(operation, sample_dataframe):
    with patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.generate_cache_key', return_value='key'), \
         patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.get_cache', return_value={}):
        result = operation._get_cache(sample_dataframe)
        assert result is None

def test_get_cache_exception(operation, sample_dataframe):
    with patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.generate_cache_key', return_value='key'), \
         patch('pamola_core.transformations.splitting.split_fields_op.operation_cache.get_cache', side_effect=Exception('fail')):
        result = operation._get_cache(sample_dataframe)
        assert result is None

def test_get_cache_parameters(operation):
    params = operation._get_cache_parameters(id_field='id', field_groups={'g': ['a']}, output_format='csv')
    assert params['id_field'] == 'id'
    assert params['field_groups'] == {'g': ['a']}
    assert params['output_format'] == 'csv'

def test_generate_data_hash_normal(operation, sample_dataframe):
    h = operation._generate_data_hash(sample_dataframe)
    assert isinstance(h, str)
    assert len(h) == 32  # md5

def test_generate_data_hash_exception(operation):
    # Pass object that will fail
    class BadDF:
        @property
        def columns(self):
            raise Exception('fail')
        @property
        def shape(self):
            return (0, 0)
        @property
        def dtypes(self):
            return []
    h = operation._generate_data_hash(BadDF())
    assert isinstance(h, str)
    assert len(h) == 32

def test_set_input_parameters(operation):
    operation._set_input_parameters(id_field='idx', field_groups={'g': ['a']}, include_id_field=False,
                                    generate_visualization=False, save_output=False, output_format='json',
                                    include_timestamp=False, use_cache=False, force_recalculation=True,
                                    use_dask=True, npartitions=2, use_vectorization=True, parallel_processes=2,
                                    visualization_backend='matplotlib', visualization_theme='dark',
                                    visualization_strict=True, use_encryption=True, encryption_key='k')
    assert operation.id_field == 'idx'
    assert operation.field_groups == {'g': ['a']}
    assert operation.include_id_field is False
    assert operation.generate_visualization is False
    assert operation.save_output is False
    assert operation.output_format == 'json'
    assert operation.include_timestamp is False
    assert operation.use_cache is False
    assert operation.force_recalculation is True
    assert operation.use_dask is True
    assert operation.npartitions == 2
    assert operation.use_vectorization is True
    assert operation.parallel_processes == 2
    assert operation.visualization_backend == 'matplotlib'
    assert operation.visualization_theme == 'dark'
    assert operation.visualization_strict is True
    assert operation.use_encryption is True
    assert operation.encryption_key == 'k'
    assert operation.timestamp == ''

def test_validate_input_parameters_valid(operation, sample_dataframe):
    assert operation._validate_input_parameters(sample_dataframe) is True

def test_validate_input_parameters_missing_field(operation, sample_dataframe):
    operation.field_groups = {'g': ['not_a_field']}
    assert operation._validate_input_parameters(sample_dataframe) is False

def test_validate_input_parameters_missing_id(operation, sample_dataframe):
    operation.include_id_field = True
    operation.id_field = 'not_id'
    assert operation._validate_input_parameters(sample_dataframe) is False

def test_validate_input_parameters_empty_groups(operation, sample_dataframe):
    operation.field_groups = {}
    assert operation._validate_input_parameters(sample_dataframe) is False

def test_load_data_and_validate_input_parameters_valid(operation, sample_dataframe):
    with patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={}), \
         patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation', return_value=sample_dataframe):
        df, valid = operation._load_data_and_validate_input_parameters(DummyDataSource(), dataset_name='main')
        assert valid is True
        assert isinstance(df, pd.DataFrame)

def test_load_data_and_validate_input_parameters_invalid(operation):
    with patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={}), \
         patch('pamola_core.transformations.splitting.split_fields_op.load_data_operation', return_value=None):
        df, valid = operation._load_data_and_validate_input_parameters(DummyDataSource(), dataset_name='main')
        assert valid is False
        assert df is None

def test_compute_total_steps(operation):
    steps = operation._compute_total_steps(use_cache=True, force_recalculation=False, save_output=True, generate_visualization=True)
    assert steps >= 7

if __name__ == "__main__":
    pytest.main()
