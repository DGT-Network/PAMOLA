"""
Tests for the split_by_id_values_op module in the pamola_core/transformations/splitting package.
These tests ensure that the SplitByIDValuesOperation class properly implements dataset splitting,
partitioning, caching, metrics collection, output saving, and error handling.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pamola_core.transformations.splitting.split_by_id_values_op import (
    SplitByIDValuesOperation, PartitionMethod, OutputFormat
)

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
    return MagicMock()

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
        visualization_backend=None,
        visualization_theme=None,
        visualization_strict=False,
        use_encryption=False,
        encryption_key=None,
        encryption_mode=None
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
    operation._set_input_parameters(
        id_field='idx', value_groups={'g1': [1]}, number_of_partitions=1,
        partition_method='modulo', generate_visualization=False, save_output=False,
        output_format='json', include_timestamp=False, use_cache=False, force_recalculation=True,
        use_dask=True, npartitions=2, use_vectorization=True, parallel_processes=2,
        visualization_backend='matplotlib', visualization_theme='dark', visualization_strict=True,
        use_encryption=True, encryption_key='abc')
    assert operation.id_field == 'idx'
    assert operation.value_groups == {'g1': [1]}
    assert operation.number_of_partitions == 1
    assert operation.partition_method == 'modulo'
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
    assert operation.encryption_key == 'abc'
    assert isinstance(operation.timestamp, str)

# --- Tests for _validate_input_parameters ---
def test_validate_input_parameters_valid_value_groups(operation):
    df = sample_dataframe()
    operation.id_field = 'id'
    operation.value_groups = {'g1': [1, 2], 'g2': [3, 4]}
    assert operation._validate_input_parameters(df) is True

def test_validate_input_parameters_missing_id_field(operation):
    df = sample_dataframe()
    operation.id_field = 'not_in_df'
    assert operation._validate_input_parameters(df) is False

def test_validate_input_parameters_no_id_field(operation):
    df = sample_dataframe()
    operation.id_field = None
    assert operation._validate_input_parameters(df) is False

def test_validate_input_parameters_invalid_partition_method(operation):
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 2
    operation.partition_method = 'invalid_method'
    assert operation._validate_input_parameters(df) is False

def test_validate_input_parameters_missing_value_groups_and_partitions(operation):
    df = sample_dataframe()
    operation.value_groups = None
    operation.number_of_partitions = 0
    assert operation._validate_input_parameters(df) is False

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
    output = {'a': df.copy(), 'b': df.copy()}
    metrics = operation._collect_metrics(df, output)
    assert metrics['input_dataset'] == 'test.csv'
    assert metrics['total_input_records'] == 6
    assert metrics['number_of_splits'] == 2
    assert 'split_info' in metrics

# --- Tests for _generate_data_hash ---
def test_generate_data_hash_returns_str(operation):
    df = sample_dataframe()
    h = operation._generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 32

def test_generate_data_hash_handles_exception(operation):
    class BadDF:
        columns = ['id']
        shape = (1, 1)
        dtypes = ['int']  # Add dtypes to match fallback in code
        def __getitem__(self, key):
            raise Exception('fail')
    h = operation._generate_data_hash(BadDF())
    assert isinstance(h, str)
    assert len(h) == 32

# --- Tests for _get_cache_parameters ---
def test_get_cache_parameters_returns_dict(operation):
    params = operation._get_cache_parameters(id_field='id', value_groups=None, number_of_partitions=2)
    assert isinstance(params, dict)
    assert params['id_field'] == 'id'
    assert params['number_of_partitions'] == 2

# --- Tests for _compute_total_steps ---
def test_compute_total_steps_all_true(operation):
    steps = operation._compute_total_steps(
        use_cache=True, force_recalculation=False, save_output=True, generate_visualization=True)
    assert steps >= 7

def test_compute_total_steps_minimal(operation):
    steps = operation._compute_total_steps(
        use_cache=False, force_recalculation=False, save_output=False, generate_visualization=False)
    assert steps == 4

# --- Tests for _save_output, _save_metrics, _save_cache, _get_cache ---
def test_save_output_and_metrics_and_cache(operation, tmp_path):
    df = sample_dataframe()
    result = MagicMock()
    result.artifacts = []
    operation.output_format = OutputFormat.CSV.value
    operation.timestamp = '20220101_000000'
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.write_dataframe_to_csv') as wcsv:
        operation._save_output({'test': df}, tmp_path, result)
        wcsv.assert_called_once()
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.write_json') as wjson:
        metrics = {'a': 1}
        operation.timestamp = '20220101_000000'
        operation._save_metrics(metrics, tmp_path, result)
        wjson.assert_called_once()
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.operation_cache') as oc:
        operation._original_df = df
        operation._save_cache(tmp_path, result)
        assert oc.save_cache.called
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.operation_cache') as oc:
        oc.generate_cache_key.return_value = 'key'
        oc.get_cache.return_value = {'result': {'status': 'SUCCESS', 'metrics': {}, 'artifacts': []}}
        out = operation._get_cache(df)
        assert out is not None

# --- Tests for _load_data_and_validate_input_parameters ---
def test_load_data_and_validate_input_parameters_valid(operation, mock_data_source):
    df = sample_dataframe()
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_data_operation', return_value=df):
            out_df, valid = operation._load_data_and_validate_input_parameters(mock_data_source)
            assert valid is True
            assert out_df.equals(df)

def test_load_data_and_validate_input_parameters_invalid(operation, mock_data_source):
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_data_operation', return_value=None):
            out_df, valid = operation._load_data_and_validate_input_parameters(mock_data_source)
            assert valid is False
            assert out_df is None

# --- Tests for execute (integration, error, and edge cases) ---
def test_execute_success(operation, mock_data_source, mock_task_dir, mock_reporter, mock_progress_tracker):
    df = sample_dataframe()
    operation.save_output = True  # Ensure _save_output is called
    operation.use_cache = True   # Ensure _save_cache is called
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_data_operation', return_value=df):
            with patch.object(operation, '_get_cache', return_value=None):
                with patch.object(operation, '_save_output') as so, \
                     patch.object(operation, '_save_metrics') as sm, \
                     patch.object(operation, '_save_cache') as sc, \
                     patch.object(operation, '_generate_visualizations') as gv:
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
                    assert gv.called

def test_execute_cache_hit(operation, mock_data_source, mock_task_dir, mock_reporter, mock_progress_tracker):
    from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
    df = sample_dataframe()
    # Provide a valid artifact dict so .get() works in the code
    fake_result = OperationResult(status=OperationStatus.SUCCESS, artifacts=[{"artifact_type": "json", "path": "dummy", "description": "", "category": "output", "tags": []}])
    with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_settings_operation', return_value={}):
        with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_data_operation', return_value=df):
            with patch.object(operation, '_get_cache', return_value=fake_result):
                operation._save_cache = lambda *a, **kw: None
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
        with patch('pamola_core.transformations.splitting.split_by_id_values_op.load_data_operation', return_value=None):
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
        operation._generate_visualizations(df, output, tmp_path, result)

if __name__ == "__main__":
	pytest.main()
