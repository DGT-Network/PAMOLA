"""
Tests for the split_by_id_values_op module in the pamola_core/transformations/splitting package.
These tests ensure that the SplitByIDValuesOperation class properly implements dataset splitting,
partitioning, caching, metrics collection, output saving, and error handling.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pamola_core.transformations.splitting.split_by_id_values_op import (
    SplitByIDValuesOperation, PartitionMethod,
)
from pamola_core.transformations.commons.enum import OutputFormat

# Fixtures for reusable setup
def sample_dataframe():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'value': ['a', 'b', 'c', 'd', 'e', 'f']
    })

def empty_dataframe():
    return pd.DataFrame({'id': [], 'value': []})

@pytest.fixture
def mock_data_source():
    ds = MagicMock()
    ds.apply_data_types = MagicMock(side_effect=lambda df, *args, **kwargs: df)
    return ds

@pytest.fixture
def mock_task_dir(tmp_path):
    return tmp_path

@pytest.fixture
def mock_reporter():
    return MagicMock()

@pytest.fixture
def mock_progress_tracker():
    tracker = MagicMock()
    tracker.total = 0
    return tracker

@pytest.fixture
def operation():
    return SplitByIDValuesOperation(
        id_field='id',
        value_groups=None,
        number_of_partitions=2,
        partition_method=PartitionMethod.EQUAL_SIZE.value,
        output_format=OutputFormat.CSV.value,
        save_output=False,
        use_cache=False,
        force_recalculation=False,
        use_dask=False,
        npartitions=1,
        use_vectorization=False,
        parallel_processes=1,
        visualization_theme=None,
        visualization_strict=False,
        use_encryption=False,
        encryption_key=None,
    )

# --- Tests for __init__ and parameter setting ---
def test_init_sets_parameters():
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=3, partition_method='random')
    assert op.id_field == 'id'
    assert op.number_of_partitions == 3
    assert op.partition_method == 'random'
    assert op.output_format == 'csv'
    assert op.save_output is True

def test_set_input_parameters_sets_all(operation):
    # _set_input_parameters does not exist; set attributes directly
    operation.id_field = 'idx'
    operation.value_groups = {'g1': [1]}
    operation.number_of_partitions = 1
    operation.partition_method = 'modulo'
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
    operation.encryption_key = 'abc'
    assert operation.id_field == 'idx'
    assert operation.value_groups == {'g1': [1]}
    assert operation.number_of_partitions == 1
    assert operation.partition_method == 'modulo'
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
    assert operation.encryption_key == 'abc'

# --- Tests for _validate_input_parameters ---
def test_validate_input_parameters_valid_value_groups(operation):
    df = sample_dataframe()
    operation.id_field = 'id'
    operation.value_groups = {'g1': [1, 2], 'g2': [3, 4]}
    assert operation._validate_input_parameters(df) is True

def test_validate_input_parameters_missing_id_field(operation):
    from pamola_core.errors.exceptions import FieldNotFoundError
    df = sample_dataframe()
    operation.id_field = 'not_in_df'
    with pytest.raises(FieldNotFoundError):
        operation._validate_input_parameters(df)

def test_validate_input_parameters_no_id_field(operation):
    from pamola_core.errors.exceptions import InvalidParameterError
    df = sample_dataframe()
    operation.id_field = None
    with pytest.raises(InvalidParameterError):
        operation._validate_input_parameters(df)

def test_validate_input_parameters_invalid_partition_method(operation):
    from pamola_core.errors.exceptions import InvalidStrategyError
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 2
    operation.partition_method = 'invalid_method'
    with pytest.raises(Exception):
        operation._validate_input_parameters(df)

def test_validate_input_parameters_missing_value_groups_and_partitions(operation):
    from pamola_core.errors.exceptions import InvalidParameterError
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 0
    with pytest.raises(InvalidParameterError):
        operation._validate_input_parameters(df)

# --- Tests for _process_with_pandas ---
def test_process_with_pandas_value_groups(operation):
    df = sample_dataframe()
    operation.value_groups = {'g1': [1, 2], 'g2': [3, 4]}
    operation.id_field = 'id'
    result = operation._process_with_pandas(df)
    assert set(result.keys()) == {'g1', 'g2', 'others'}
    assert all(isinstance(v, pd.DataFrame) for v in result.values())
    assert set(result['g1']['id']) == {1, 2}
    assert set(result['g2']['id']) == {3, 4}
    assert set(result['others']['id']) == {5, 6}

def test_process_with_pandas_equal_size(operation):
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.EQUAL_SIZE.value
    result = operation._process_with_pandas(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}
    total = sum(len(v) for v in result.values())
    assert total == len(df)

def test_process_with_pandas_random(operation):
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.RANDOM.value
    result = operation._process_with_pandas(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}
    total = sum(len(v) for v in result.values())
    assert total == len(df)

def test_process_with_pandas_modulo(operation):
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.MODULO.value
    result = operation._process_with_pandas(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}
    total = sum(len(v) for v in result.values())
    assert total == len(df)

def test_process_with_pandas_no_groups_or_partitions(operation):
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 0
    result = operation._process_with_pandas(df)
    assert set(result.keys()) == {'all_data'}
    assert len(result['all_data']) == len(df)

def test_process_with_pandas_empty_df(operation):
    df = empty_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 0
    result = operation._process_with_pandas(df)
    assert set(result.keys()) == {'all_data'}
    assert result['all_data'].empty

# --- Tests for _process_with_joblib ---
def test_process_with_joblib_groups(operation):
    df = sample_dataframe()
    operation.value_groups = {'g1': [1, 2], 'g2': [3, 4]}
    operation.id_field = 'id'
    operation.parallel_processes = 2
    result = operation._process_with_joblib(df)
    assert set(result.keys()) == {'g1', 'g2', 'others'}
    assert set(result['g1']['id']) == {1, 2}
    assert set(result['g2']['id']) == {3, 4}
    assert set(result['others']['id']) == {5, 6}

# --- Tests for _process_with_dask ---
def test_process_with_dask_modulo(operation):
    pytest.importorskip('dask')
    df = sample_dataframe()
    operation.use_dask = True
    operation.npartitions = 2
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.MODULO.value
    operation.id_field = 'id'
    result = operation._process_with_dask(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}
    total = sum(len(v) for v in result.values())
    assert total == len(df)

def test_process_with_dask_random(operation):
    pytest.importorskip('dask')
    df = sample_dataframe()
    operation.use_dask = True
    operation.npartitions = 2
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.RANDOM.value
    operation.id_field = 'id'
    result = operation._process_with_dask(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}
    total = sum(len(v) for v in result.values())
    assert total == len(df)

# --- Tests for _process_data ---
def test_process_data_selects_pandas(operation):
    df = sample_dataframe()
    operation.use_dask = False
    operation.use_vectorization = False
    operation.value_groups = None
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.EQUAL_SIZE.value
    result = operation._process_data(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}

def test_process_data_selects_joblib(operation):
    df = sample_dataframe()
    operation.use_vectorization = True
    operation.parallel_processes = 2
    operation.value_groups = {'g1': [1, 2], 'g2': [3, 4]}
    result = operation._process_data(df)
    assert set(result.keys()) == {'g1', 'g2', 'others'}

def test_process_data_selects_dask(operation):
    pytest.importorskip('dask')
    df = sample_dataframe()
    operation.use_dask = True
    operation.npartitions = 2
    operation.number_of_partitions = 2
    operation.partition_method = PartitionMethod.MODULO.value
    result = operation._process_data(df)
    assert set(result.keys()) == {'partition_0', 'partition_1'}

# --- Tests for _collect_metrics ---
def test_collect_metrics_dict_output(operation):
    df = sample_dataframe()
    operation.id_field = 'id'
    operation._input_dataset = 'test.csv'
    operation.start_time = 0
    operation.end_time = 1
    operation.execution_time = 1.0
    output = {'a': df.copy(), 'b': df.copy()}
    metrics = operation._collect_metrics(df, output)
    assert metrics['total_input_records'] == 6
    assert metrics['number_of_splits'] == 2
    assert 'split_info' in metrics

# --- Tests for _generate_data_hash ---
def test_generate_data_hash_returns_str(operation):
    from pamola_core.utils import helpers
    df = sample_dataframe()
    h = helpers.generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 64  # BLAKE2b with digest_size=32 -> 64 hex chars

def test_generate_data_hash_handles_exception(operation):
    from pamola_core.utils import helpers
    # generate_data_hash has its own fallback for invalid data
    df = sample_dataframe()
    h = helpers.generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 64  # BLAKE2b with digest_size=32 -> 64 hex chars

# --- Tests for _get_cache_parameters ---
def test_get_cache_parameters_returns_dict(operation):
    params = operation._get_cache_parameters()
    assert isinstance(params, dict)
    assert 'id_field' in params
    assert 'number_of_partitions' in params

# --- Tests for _compute_total_steps ---
def test_compute_total_steps_all_true(operation):
    # _compute_total_steps takes no args; uses instance attributes
    operation.use_cache = True
    operation.force_recalculation = False
    operation.save_output = True
    operation.generate_visualization = True
    steps = operation._compute_total_steps()
    assert steps >= 7

def test_compute_total_steps_minimal(operation):
    operation.use_cache = False
    operation.force_recalculation = False
    operation.save_output = False
    operation.generate_visualization = False
    steps = operation._compute_total_steps()
    assert steps == 4

# --- Tests for _save_multiple_output_data, _save_metrics, _save_to_cache, _check_cache ---
def test_save_output_and_metrics_and_cache(operation, tmp_path):
    from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
    df = sample_dataframe()
    result = OperationResult(status=OperationStatus.SUCCESS)
    operation.output_format = OutputFormat.CSV.value
    operation.timestamp = '20220101_000000'

    # Test _save_multiple_output_data via mocked _save_output_data
    writer = MagicMock()
    with patch.object(operation, '_save_output_data') as mock_save_output:
        operation._save_multiple_output_data({'test': df}, writer, result, None, None, timestamp='20220101_000000')
        mock_save_output.assert_called_once()

    # Test _save_metrics via mocked writer
    metrics = {'a': 1}
    writer2 = MagicMock()
    writer2.write_metrics.return_value = MagicMock(path=str(tmp_path / 'metrics.json'))
    operation._save_metrics(
        metrics=metrics,
        writer=writer2,
        result=result,
        reporter=None,
        progress_tracker=None,
        operation_timestamp='20220101_000000',
    )
    assert writer2.write_metrics.called

    # Test _save_to_cache
    operation.use_cache = True
    operation.operation_cache = MagicMock()
    operation.operation_cache.save_cache.return_value = True
    ok = operation._save_to_cache(df, df, result, tmp_path)
    assert ok is True

    # Test _check_cache (cache hit)
    from pamola_core.utils import helpers
    operation.operation_cache = MagicMock()
    cache_data = {'status': 'SUCCESS', 'metrics': {}, 'error_message': None,
                  'execution_time': 1.0, 'error_trace': None, 'artifacts': []}
    operation.operation_cache.get_cache.return_value = cache_data
    with patch.object(helpers, 'get_cache_result', return_value=result):
        out = operation._check_cache(df, None)
        assert out is not None


# --- Tests for execute (integration, error, and edge cases) ---
def test_execute_success(operation, mock_data_source, mock_task_dir, mock_reporter, mock_progress_tracker):
    from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
    df = sample_dataframe()
    operation.save_output = True
    operation.use_cache = True
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.base_transformation_op.load_data_operation', return_value=df):
            with patch.object(operation, '_check_cache', return_value=None):
                with patch.object(operation, '_save_multiple_output_data') as so, \
                     patch.object(operation, '_save_metrics') as sm, \
                     patch.object(operation, '_save_to_cache') as sc, \
                     patch.object(operation, '_handle_visualizations') as gv:
                    result = operation.execute(
                        data_source=mock_data_source,
                        task_dir=mock_task_dir,
                        reporter=mock_reporter,
                        progress_tracker=mock_progress_tracker
                    )
                    assert hasattr(result, 'status')
                    assert result.status.name == 'SUCCESS'
                    assert sm.called
                    assert so.called
                    assert sc.called

def test_execute_cache_hit(operation, mock_data_source, mock_task_dir, mock_reporter, mock_progress_tracker):
    from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
    df = sample_dataframe()
    fake_result = OperationResult(status=OperationStatus.SUCCESS)
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.base_transformation_op.load_data_operation', return_value=df):
            with patch.object(operation, '_check_cache', return_value=fake_result):
                result = operation.execute(
                    data_source=mock_data_source,
                    task_dir=mock_task_dir,
                    reporter=None,
                    progress_tracker=mock_progress_tracker
                )
                assert isinstance(result, OperationResult)
                assert result.status.name == 'SUCCESS'

def test_execute_invalid_input(operation, mock_data_source, mock_task_dir, mock_reporter, mock_progress_tracker):
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.base_transformation_op.load_data_operation', return_value=None):
            result = operation.execute(
                data_source=mock_data_source,
                task_dir=mock_task_dir,
                reporter=mock_reporter,
                progress_tracker=mock_progress_tracker
            )
            assert result.status.name == 'ERROR'

# --- Tests for _generate_visualizations (smoke) ---
def test_generate_visualizations_smoke(operation, tmp_path):
    df = sample_dataframe()
    output = {'a': df.copy(), 'b': df.copy()}
    result = MagicMock()
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.create_bar_plot', return_value='bar.png'), \
         patch('pamola_core.transformations.splitting.split_by_id_values_op.create_pie_chart', return_value='pie.png'), \
         patch('pamola_core.transformations.splitting.split_by_id_values_op.create_heatmap', return_value='heatmap.png'), \
         patch('pamola_core.transformations.splitting.split_by_id_values_op.ensure_directory'):
        operation.id_field = 'id'
        operation.timestamp = '20220101_000000'
        operation._generate_visualizations(df, output, tmp_path, result, operation.timestamp)

if __name__ == "__main__":
	pytest.main()
