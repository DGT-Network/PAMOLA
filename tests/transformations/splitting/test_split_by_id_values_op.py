"""
Unit tests for SplitByIDValuesOperation in split_by_id_values_op.py

These tests verify the functionality of SplitByIDValuesOperation, including
partitioning by value groups, equal size, random, modulo, output creation, metrics, error handling, and parameter validation.

Run with:
    pytest tests/transformations/splitting/test_split_by_id_values_op.py
"""
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock
import pytest
import pandas as pd
import sys
import gc
from pamola_core.transformations.splitting.split_by_id_values_op import (
    SplitByIDValuesOperation, PartitionMethod, OutputFormat
)
from pamola_core.utils.ops.op_result import OperationStatus

class DummyDataSource:
    def __init__(self, df):
        self.df = df
    def load(self, *args, **kwargs):
        return self.df

def dummy_load_data_operation(data_source, dataset_name):
    return data_source.df

class DummyReporter:
    def __init__(self):
        self.operations = []
    def add_operation(self, name, status, details=None):
        self.operations.append((name, status, details))

class DummyProgressTracker:
    def __init__(self):
        self.updates = []
    def update(self, step, info):
        self.updates.append((step, info))

@pytest.fixture(scope="function")
def temp_task_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    gc.collect()
    for _ in range(3):
        try:
            shutil.rmtree(d)
            break
        except PermissionError:
            gc.collect()

@pytest.fixture(scope="function")
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10, 20, 30, 40, 50, 60]
    })

@pytest.fixture(scope="function")
def empty_df():
    return pd.DataFrame({'id': [], 'value': []})

@pytest.fixture(autouse=True)
def patch_load_data(monkeypatch):
    monkeypatch.setattr(
        'pamola_core.transformations.splitting.split_by_id_values_op.load_data_operation',
        dummy_load_data_operation
    )

def test_valid_value_groups_split(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', value_groups={'A': [1, 2], 'B': [3, 4]}, output_format='csv')
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.SUCCESS
    assert 'A' in op._process_data(sample_df)
    assert 'B' in op._process_data(sample_df)
    assert 'others' in op._process_data(sample_df)
    assert result.metrics['number_of_splits'] == 3
    assert os.path.exists(temp_task_dir / 'output')
    assert os.path.exists(temp_task_dir / 'metrics')
    assert os.path.exists(temp_task_dir / 'visualizations')

def test_valid_partition_equal_size(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, partition_method=PartitionMethod.EQUAL_SIZE.value)
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.SUCCESS
    out = op._process_data(sample_df)
    assert len(out) == 2
    assert all(isinstance(df, pd.DataFrame) for df in out.values())

def test_valid_partition_random(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=3, partition_method=PartitionMethod.RANDOM.value)
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.SUCCESS
    out = op._process_data(sample_df)
    assert len(out) == 3

def test_valid_partition_modulo(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, partition_method=PartitionMethod.MODULO.value)
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.SUCCESS
    out = op._process_data(sample_df)
    assert len(out) == 2

def test_no_split_returns_all_data(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id')
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    out = op._process_data(sample_df)
    assert 'all_data' in out
    assert out['all_data'].equals(sample_df)

def test_empty_dataframe(temp_task_dir, empty_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2)
    ds = DummyDataSource(empty_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.ERROR
    assert 'empty' in result.error_message

def test_invalid_id_field(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='not_a_column', number_of_partitions=2)
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.ERROR
    assert 'not found' in result.error_message or 'invalid' in result.error_message

def test_invalid_partition_method(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, partition_method='invalid_method')
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    result = op.execute(ds, temp_task_dir, reporter, tracker)
    assert result.status == OperationStatus.ERROR
    assert 'invalid' in result.error_message or 'Unsupported partition method' in result.error_message

def test_value_groups_with_missing_ids(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', value_groups={'A': [1, 99]}, output_format='csv')
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    with mock.patch.object(op.logger, 'warning') as mock_warn:
        result = op.execute(ds, temp_task_dir, reporter, tracker)
        assert result.status == OperationStatus.SUCCESS

def test_save_output_creates_files(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, output_format='csv')
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    op.execute(ds, temp_task_dir, reporter, tracker)
    output_dir = temp_task_dir / 'output'
    assert output_dir.exists()
    assert any(f.suffix == '.csv' for f in output_dir.iterdir())

def test_save_metrics_creates_file(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, output_format='csv')
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    op.execute(ds, temp_task_dir, reporter, tracker)
    metrics_dir = temp_task_dir / 'metrics'
    assert metrics_dir.exists()
    assert any(f.suffix == '.json' for f in metrics_dir.iterdir())

def test_generate_visualizations_creates_files(temp_task_dir, sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, output_format='csv', generate_visualization=True)
    ds = DummyDataSource(sample_df)
    reporter = DummyReporter()
    tracker = DummyProgressTracker()
    op.execute(ds, temp_task_dir, reporter, tracker)
    vis_dir = temp_task_dir / 'visualizations'
    assert vis_dir.exists()
    assert any(f.suffix == '.png' for f in vis_dir.iterdir())

def test_collect_metrics_structure(sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2)
    out = op._process_data(sample_df)
    op.input_dataset = 'main.csv'
    op.start_time = 0
    op.end_time = 1
    metrics = op._collect_metrics(sample_df, out)
    assert 'operation_type' in metrics
    assert 'input_dataset' in metrics
    assert 'number_of_splits' in metrics
    assert 'split_info' in metrics

def test_generate_data_hash_changes_with_data(sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2)
    hash1 = op._generate_data_hash(sample_df)
    df2 = sample_df.copy()
    df2.loc[0, 'value'] = 999
    hash2 = op._generate_data_hash(df2)
    assert hash1 != hash2

def test_get_cache_parameters(sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2)
    # Patch missing attributes to avoid AttributeError
    if not hasattr(op, 'field_groups'):
        op.field_groups = None
    if not hasattr(op, 'include_id_field'):
        op.include_id_field = None
    params = op._get_cache_parameters()
    assert isinstance(params, dict)
    assert 'operation' in params
    assert 'id_field' in params

def test_validate_parameters_missing_id_field(sample_df):
    op = SplitByIDValuesOperation(id_field=None, number_of_partitions=2)
    assert not op._validate_parameters(sample_df)

def test_validate_parameters_invalid_partition_method(sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2, partition_method='not_a_method')
    assert not op._validate_parameters(sample_df)

def test_validate_parameters_value_groups_warns(sample_df):
    op = SplitByIDValuesOperation(id_field='id', value_groups={'A': [1, 999]})
    with mock.patch.object(op.logger, 'warning') as mock_warn:
        op._validate_parameters(sample_df)
        mock_warn.assert_called()

def test_set_common_operation_parameters_sets_all(sample_df):
    op = SplitByIDValuesOperation(id_field='id', number_of_partitions=2)
    op._set_common_operation_parameters(
        id_field='id', value_groups={'A': [1]}, number_of_partitions=3,
        partition_method=PartitionMethod.EQUAL_SIZE.value, output_format=OutputFormat.CSV.value,
        force_recalculation=True, generate_visualization=False, include_timestamp=False,
        save_output=False, parallel_processes=2, batch_size=500, use_cache=True, use_dask=True,
        use_encryption=True, encryption_key='key')
    assert op.id_field == 'id'
    assert op.value_groups == {'A': [1]}
    assert op.number_of_partitions == 3
    assert op.partition_method == PartitionMethod.EQUAL_SIZE.value
    assert op.output_format == OutputFormat.CSV.value
    assert op.force_recalculation is True
    assert op.generate_visualization is False
    assert op.include_timestamp is False
    assert op.save_output is False
    assert op.parallel_processes == 2
    assert op.batch_size == 500
    assert op.use_cache is True
    assert op.use_dask is True
    assert op.use_encryption is True
    assert op.encryption_key == 'key'

if __name__ == "__main__":
    pytest.main()
