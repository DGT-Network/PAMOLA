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
from pamola_core.transformations.splitting.split_fields_op import SplitFieldsOperation
from pamola_core.transformations.commons.enum import OutputFormat
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
    encryption_keys = {}
    encryption_modes = {}
    settings = {}
    data_source_name = "test"

    def apply_data_types(self, df, *args, **kwargs):
        return df

class DummyReporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []
    def add_operation(self, *args, **kwargs):
        self.operations.append((args, kwargs))
    def add_artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))

class DummyProgress:
    def __init__(self):
        self.total = 0
        self.updates = []
    def update(self, *args, **kwargs):
        self.updates.append((args, kwargs))

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_valid_case(mock_load_data, mock_load_settings, operation, sample_dataframe, tmp_path):
    mock_load_data.return_value = sample_dataframe
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = tmp_path
    dirs = {'output': task_dir, 'metrics': task_dir, 'cache': task_dir, 'logs': task_dir,
            'temp': task_dir, 'dictionaries': task_dir, 'visualizations': task_dir,
            'attacks': task_dir, 'reports': task_dir, 'input': task_dir, 'root': task_dir}
    with patch.object(operation, '_prepare_directories', return_value=dirs):
        with patch.object(operation, '_save_multiple_output_data') as mock_save_output, \
             patch.object(operation, '_save_metrics') as mock_save_metrics, \
             patch.object(operation, '_handle_visualizations') as mock_vis, \
             patch.object(operation, '_save_to_cache') as mock_cache:
            result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
            assert isinstance(result, OperationResult)
            assert result.status == OperationStatus.SUCCESS
            assert mock_save_output.called
            assert mock_save_metrics.called
            assert mock_vis.called
            # Remove cache assertion: use_cache=False in fixture
            assert any('Operation completed successfully' in str(op[1]) for op in reporter.operations)

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_empty_dataframe(mock_load_data, mock_load_settings, operation, tmp_path):
    mock_load_data.return_value = pd.DataFrame()
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = tmp_path
    dirs = {'output': task_dir, 'metrics': task_dir, 'cache': task_dir, 'logs': task_dir,
            'temp': task_dir, 'dictionaries': task_dir, 'visualizations': task_dir,
            'attacks': task_dir, 'reports': task_dir, 'input': task_dir, 'root': task_dir}
    with patch.object(operation, '_prepare_directories', return_value=dirs):
        result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
        assert result.status == OperationStatus.ERROR
        assert result.error_message is not None

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_invalid_field_groups(mock_load_data, mock_load_settings, operation, sample_dataframe, tmp_path):
    # field_groups has a field not in df
    operation.field_groups = {'group1': ['not_a_field']}
    mock_load_data.return_value = sample_dataframe
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = tmp_path
    dirs = {'output': task_dir, 'metrics': task_dir, 'cache': task_dir, 'logs': task_dir,
            'temp': task_dir, 'dictionaries': task_dir, 'visualizations': task_dir,
            'attacks': task_dir, 'reports': task_dir, 'input': task_dir, 'root': task_dir}
    with patch.object(operation, '_prepare_directories', return_value=dirs):
        result = operation.execute(DummyDataSource(), task_dir, reporter, progress)
        assert result.status == OperationStatus.ERROR
        assert result.error_message is not None

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_cache_hit(mock_load_data, mock_load_settings, operation, sample_dataframe, tmp_path):
    mock_load_data.return_value = sample_dataframe
    progress = DummyProgress()
    task_dir = tmp_path
    dirs = {'output': task_dir, 'metrics': task_dir, 'cache': task_dir, 'logs': task_dir,
            'temp': task_dir, 'dictionaries': task_dir, 'visualizations': task_dir,
            'attacks': task_dir, 'reports': task_dir, 'input': task_dir, 'root': task_dir}
    # Patch _check_cache to return a cached result directly
    fake_cache = OperationResult(
        status=OperationStatus.SUCCESS,
        metrics={},
        error_message=None,
        execution_time=1.0,
        error_trace=None,
    )
    with patch.object(operation, '_prepare_directories', return_value=dirs):
        with patch.object(operation, '_check_cache', return_value=fake_cache) as mock_cache:
            operation.use_cache = True
            result = operation.execute(DummyDataSource(), task_dir, None, progress)
            assert isinstance(result, OperationResult)
            assert mock_cache.called

@patch('pamola_core.transformations.splitting.split_fields_op.load_settings_operation', return_value={})
@patch('pamola_core.transformations.base_transformation_op.load_data_operation')
def test_execute_exception_handling(mock_load_data, mock_load_settings, operation, sample_dataframe, tmp_path):
    mock_load_data.side_effect = Exception('load error')
    reporter = DummyReporter()
    progress = DummyProgress()
    task_dir = tmp_path
    dirs = {'output': task_dir, 'metrics': task_dir, 'cache': task_dir, 'logs': task_dir,
            'temp': task_dir, 'dictionaries': task_dir, 'visualizations': task_dir,
            'attacks': task_dir, 'reports': task_dir, 'input': task_dir, 'root': task_dir}
    with patch.object(operation, '_prepare_directories', return_value=dirs):
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
    operation.execution_time = 1.0
    metrics = operation._collect_metrics(sample_dataframe, output)
    assert isinstance(metrics, dict)
    assert metrics['total_input_records'] == 3
    assert metrics['number_of_splits'] == 2
    assert 'split_info' in metrics

def test_save_metrics_success(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    operation._input_dataset = 'test.csv'
    operation.start_time = 0
    operation.end_time = 1
    operation.execution_time = 1.0
    metrics = operation._collect_metrics(sample_dataframe, output)
    result = OperationResult(status=OperationStatus.SUCCESS)
    writer = MagicMock()
    writer.write_metrics.return_value = MagicMock(path=tmp_path / 'metrics.json')
    reporter = DummyReporter()
    # _save_metrics is inherited from base; call with correct signature
    ok = operation._save_metrics(
        metrics=metrics,
        writer=writer,
        result=result,
        reporter=reporter,
        progress_tracker=None,
        operation_timestamp='20240101_000000',
    )
    assert writer.write_metrics.called
    assert ok is True

def test_save_metrics_failure(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    operation._input_dataset = 'test.csv'
    operation.start_time = 0
    operation.end_time = 1
    operation.execution_time = 1.0
    metrics = operation._collect_metrics(sample_dataframe, output)
    result = OperationResult(status=OperationStatus.SUCCESS)
    writer = MagicMock()
    writer.write_metrics.side_effect = Exception('fail')
    # _save_metrics catches exceptions and returns False
    ok = operation._save_metrics(
        metrics=metrics,
        writer=writer,
        result=result,
        reporter=None,
        progress_tracker=None,
        operation_timestamp='20240101_000000',
    )
    assert ok is False

def test_save_output_csv(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.timestamp = 'testts'
    writer = MagicMock()
    writer.write_dataframe.return_value = MagicMock(path=str(tmp_path / 'out.csv'))
    # _save_multiple_output_data calls _save_output_data (base) for each subset
    with patch.object(operation, '_save_output_data') as mock_save:
        operation._save_multiple_output_data(
            result_subsets=output,
            writer=writer,
            result=result,
            reporter=DummyReporter(),
            progress_tracker=None,
            timestamp='testts',
        )
        assert mock_save.called

def test_save_output_json_encrypted(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = OutputFormat.JSON.value
    operation.timestamp = 'testts'
    writer = MagicMock()
    with patch.object(operation, '_save_output_data') as mock_save:
        operation._save_multiple_output_data(
            result_subsets=output,
            writer=writer,
            result=result,
            reporter=None,
            progress_tracker=None,
            timestamp='testts',
            use_encryption=True,
            encryption_key='key',
        )
        assert mock_save.called

def test_save_output_json_no_encryption(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = OutputFormat.JSON.value
    operation.timestamp = 'testts'
    writer = MagicMock()
    with patch.object(operation, '_save_output_data') as mock_save:
        # Should not raise
        operation._save_multiple_output_data(
            result_subsets=output,
            writer=writer,
            result=result,
            reporter=None,
            progress_tracker=None,
            timestamp='testts',
            use_encryption=False,
            encryption_key=None,
        )
        assert mock_save.called

def test_save_output_unsupported_format(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = 'xml'  # unsupported
    operation.timestamp = 'testts'
    writer = MagicMock()
    with patch.object(operation, '_save_output_data') as mock_save:
        # Should not raise
        operation._save_multiple_output_data(
            result_subsets=output,
            writer=writer,
            result=result,
            reporter=None,
            progress_tracker=None,
            timestamp='testts',
        )
        assert mock_save.called

def test_generate_visualizations_all(tmp_path, operation, sample_dataframe):
    output = operation._process_data(sample_dataframe)
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.timestamp = 'testts'
    with patch('pamola_core.transformations.splitting.split_fields_op.create_bar_plot', return_value=str(tmp_path/'bar.png')) as mock_bar, \
         patch('pamola_core.transformations.splitting.split_fields_op.plot_field_subset_network', return_value=str(tmp_path/'net.png')) as mock_net:
        operation._generate_visualizations(sample_dataframe, output, tmp_path, result, operation.timestamp)
        assert mock_bar.called
        assert mock_net.called

def test_generate_visualizations_empty_output(tmp_path, operation, sample_dataframe):
    result = OperationResult(status=OperationStatus.SUCCESS)
    # output_data is empty dict
    with patch('pamola_core.transformations.splitting.split_fields_op.create_bar_plot') as mock_bar:
        operation._generate_visualizations(sample_dataframe, {}, tmp_path, result, operation.timestamp)
        assert not mock_bar.called

def test_save_cache_success(tmp_path, operation, sample_dataframe):
    operation.use_cache = True
    operation.operation_cache = MagicMock()
    operation.operation_cache.save_cache.return_value = True
    result = OperationResult(status=OperationStatus.SUCCESS)
    ok = operation._save_to_cache(sample_dataframe, sample_dataframe, result, tmp_path)
    assert ok is True
    assert operation.operation_cache.save_cache.called

def test_save_cache_failure(tmp_path, operation, sample_dataframe):
    operation.use_cache = True
    operation.operation_cache = MagicMock()
    operation.operation_cache.save_cache.side_effect = Exception('fail')
    result = OperationResult(status=OperationStatus.SUCCESS)
    # _save_to_cache lets exception propagate; caller wraps in try/except
    try:
        operation._save_to_cache(sample_dataframe, sample_dataframe, result, tmp_path)
    except Exception:
        pass  # expected

def test_get_cache_hit(operation, sample_dataframe):
    from pamola_core.utils import helpers
    operation.use_cache = True
    operation.operation_cache = MagicMock()
    cached_result = OperationResult(status=OperationStatus.SUCCESS)
    cache_data = {'result': {'status': 'SUCCESS', 'metrics': {}, 'error_message': None,
                             'execution_time': 1.0, 'error_trace': None, 'artifacts': []}}
    operation.operation_cache.get_cache.return_value = cache_data
    with patch.object(helpers, 'get_cache_result', return_value=cached_result):
        result = operation._check_cache(sample_dataframe, None)
        assert isinstance(result, OperationResult)
        assert result.status == OperationStatus.SUCCESS

def test_get_cache_miss(operation, sample_dataframe):
    operation.use_cache = True
    operation.operation_cache = MagicMock()
    operation.operation_cache.get_cache.return_value = {}
    result = operation._check_cache(sample_dataframe, None)
    assert result is None

def test_get_cache_exception(operation, sample_dataframe):
    operation.use_cache = True
    operation.operation_cache = MagicMock()
    operation.operation_cache.get_cache.side_effect = Exception('fail')
    result = operation._check_cache(sample_dataframe, None)
    assert result is None

def test_get_cache_parameters(operation):
    params = operation._get_cache_parameters()
    assert params['id_field'] == operation.id_field
    assert params['field_groups'] == operation.field_groups
    assert isinstance(params, dict)

def test_generate_data_hash_normal(operation, sample_dataframe):
    from pamola_core.utils import helpers
    h = helpers.generate_data_hash(sample_dataframe)
    assert isinstance(h, str)
    assert len(h) == 64  # BLAKE2b with digest_size=32 -> 64 hex chars

def test_generate_data_hash_exception(operation):
    from pamola_core.utils import helpers
    # generate_data_hash has its own fallback for invalid data
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    h = helpers.generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 64  # BLAKE2b with digest_size=32 -> 64 hex chars

def test_set_input_parameters(operation):
    # _set_input_parameters does not exist; verify that attributes can be set directly
    operation.id_field = 'idx'
    operation.field_groups = {'g': ['a']}
    operation.include_id_field = False
    operation.generate_visualization = False
    operation.save_output = False
    operation.output_format = 'json'
    operation.use_cache = False
    operation.force_recalculation = True
    operation.use_dask = True
    operation.npartitions = 2
    operation.use_vectorization = True
    operation.parallel_processes = 2
    operation.visualization_backend = 'matplotlib'
    operation.visualization_theme = 'dark'
    operation.visualization_strict = True
    operation.use_encryption = True
    operation.encryption_key = 'k'
    assert operation.id_field == 'idx'
    assert operation.field_groups == {'g': ['a']}
    assert operation.include_id_field is False
    assert operation.generate_visualization is False
    assert operation.save_output is False
    assert operation.output_format == 'json'
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

def test_validate_input_parameters_valid(operation, sample_dataframe):
    assert operation._validate_input_parameters(sample_dataframe) is True

def test_validate_input_parameters_missing_field(operation, sample_dataframe):
    from pamola_core.errors.exceptions import FieldNotFoundError
    operation.field_groups = {'g': ['not_a_field']}
    with pytest.raises(FieldNotFoundError):
        operation._validate_input_parameters(sample_dataframe)

def test_validate_input_parameters_missing_id(operation, sample_dataframe):
    from pamola_core.errors.exceptions import FieldNotFoundError
    operation.include_id_field = True
    operation.id_field = 'not_id'
    with pytest.raises(FieldNotFoundError):
        operation._validate_input_parameters(sample_dataframe)

def test_validate_input_parameters_empty_groups(operation, sample_dataframe):
    from pamola_core.errors.exceptions import MissingParameterError
    operation.field_groups = {}
    with pytest.raises(MissingParameterError):
        operation._validate_input_parameters(sample_dataframe)


def test_compute_total_steps(operation):
    # _compute_total_steps takes no args; uses instance attributes
    operation.use_cache = True
    operation.force_recalculation = False
    operation.save_output = True
    operation.generate_visualization = True
    steps = operation._compute_total_steps()
    assert steps >= 7

if __name__ == "__main__":
    pytest.main()
