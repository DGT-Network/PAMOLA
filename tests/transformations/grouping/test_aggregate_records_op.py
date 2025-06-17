"""
Unit tests for AggregateRecordsOperation in aggregate_records_op.py

These tests verify the functionality of AggregateRecordsOperation and related grouping operations,
including aggregation logic, cache handling, metrics, output, and error handling.

Run with:
    pytest tests/transformations/grouping/test_aggregate_records_op.py
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from pamola_core.transformations.grouping.aggregate_records_op import (
    AggregateRecordsOperation,
    create_aggregate_records_operation,
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

class DummyDataSource:
    pass

class DummyReporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []
    def add_operation(self, operation, details=None):
        self.operations.append((operation, details))
    def add_artifact(self, type_, path, desc):
        self.artifacts.append((type_, path, desc))

class DummyProgress:
    def __init__(self):
        self.updates = []
        self.total = 0
    def update(self, n, d):
        self.updates.append((n, d))

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 1, 2, 2, 3],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
    })

@pytest.fixture
def empty_df():
    return pd.DataFrame({'A': [], 'B': [], 'C': []})

@pytest.fixture
def op(tmp_path):
    return AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum', 'mean'], 'C': ['max']},
        custom_aggregations={},
        output_format='csv',
        use_cache=False,
        use_encryption=False,
        use_dask=False,
    )

def test_valid_case(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum', 'mean'], 'C': ['max']},
        custom_aggregations={},
        output_format='csv',
        use_cache=False,
        use_encryption=False,
        use_dask=False,
    )
    with patch('pamola_core.transformations.grouping.aggregate_records_op.load_data_operation', return_value=sample_df):
        with patch('pamola_core.transformations.grouping.aggregate_records_op.aggregate_dataframe', side_effect=lambda df, **kwargs: df.groupby('A').agg({'B': ['sum', 'mean'], 'C': ['max']})):
            with patch('pamola_core.transformations.grouping.aggregate_records_op.DataWriter') as MockWriter:
                writer = MockWriter.return_value
                writer.write_metrics.return_value = MagicMock(path=tmp_path/'metrics.json')
                writer.write_dataframe.return_value = MagicMock(path=tmp_path/'output.csv')
                result = op.execute(
                    data_source=DummyDataSource(),
                    task_dir=tmp_path,
                    reporter=DummyReporter(),
                    progress_tracker=DummyProgress(),
                )
                assert isinstance(result, OperationResult)
                assert result.status == OperationStatus.SUCCESS

def test_edge_case_empty_input(empty_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
        custom_aggregations={},
        output_format='csv',
        use_cache=False,
        use_encryption=False,
        use_dask=False,
    )
    with patch('pamola_core.transformations.grouping.aggregate_records_op.load_data_operation', return_value=empty_df):
        with patch('pamola_core.transformations.grouping.aggregate_records_op.aggregate_dataframe', side_effect=lambda df, **kwargs: df):
            with patch('pamola_core.transformations.grouping.aggregate_records_op.DataWriter') as MockWriter:
                writer = MockWriter.return_value
                writer.write_metrics.return_value = MagicMock(path=tmp_path/'metrics.json')
                writer.write_dataframe.return_value = MagicMock(path=tmp_path/'output.csv')
                result = op.execute(
                    data_source=DummyDataSource(),
                    task_dir=tmp_path,
                    reporter=DummyReporter(),
                    progress_tracker=DummyProgress(),
                )
                assert isinstance(result, OperationResult)
                assert result.status == OperationStatus.SUCCESS

def test_invalid_input_group_by():
    with pytest.raises(ValueError):
        AggregateRecordsOperation(
            group_by_fields=[],
            aggregations={'B': ['sum']},
        )._validate_input_params([], {'B': ['sum']})

def test_invalid_aggregation_func():
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['not_a_func']},
    )
    with pytest.raises(ValueError):
        op._validate_input_params(['A'], {'B': ['not_a_func']})

def test_invalid_custom_aggregation_func():
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
        custom_aggregations={'B': ['not_a_func']},
    )
    with pytest.raises(ValueError):
        op._validate_input_params(['A'], {'B': ['sum']}, {'B': ['not_a_func']})

def test_check_cache_no_cache(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
        use_cache=True,
    )
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = None
    result = op._check_cache(sample_df)
    assert result is None

def test_check_cache_hit(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
        use_cache=True,
    )
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = {
        'metrics': {'total_input_records': 5},
        'timestamp': '2025-01-01T00:00:00',
    }
    result = op._check_cache(sample_df)
    assert isinstance(result, OperationResult)
    assert result.status == OperationStatus.SUCCESS
    assert result.metrics['cached'] is True

def test_save_to_cache_success(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
        use_cache=True,
    )
    op.operation_cache = MagicMock()
    op.operation_cache.save_cache.return_value = True
    assert op._save_to_cache(sample_df, sample_df, tmp_path, metrics={'foo': 1}) is True

def test_save_to_cache_fail(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
        use_cache=True,
    )
    op.operation_cache = MagicMock()
    op.operation_cache.save_cache.side_effect = Exception('fail')
    assert op._save_to_cache(sample_df, sample_df, tmp_path, metrics={'foo': 1}) is False

def test_cleanup_memory(sample_df):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    op._temp_data = sample_df
    op.operation_cache = MagicMock()
    op._cleanup_memory(processed_df=sample_df, df=sample_df)
    assert op._temp_data is None
    assert op.operation_cache is None

def test_generate_visualizations(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    with patch('pamola_core.transformations.grouping.aggregate_records_op.sample_large_dataset', side_effect=lambda df, max_samples: df):
        with patch('pamola_core.transformations.grouping.aggregate_records_op.generate_record_count_per_group_vis', return_value={'rec': tmp_path/'rec.png'}):
            with patch('pamola_core.transformations.grouping.aggregate_records_op.generate_aggregation_comparison_vis', return_value={'agg': tmp_path/'agg.png'}):
                with patch('pamola_core.transformations.grouping.aggregate_records_op.generate_group_size_distribution_vis', return_value={'dist': tmp_path/'dist.png'}):
                    result = op._generate_visualizations(
                        df=sample_df,
                        processed_df=sample_df,
                        task_dir=tmp_path,
                        result=OperationResult(status=OperationStatus.SUCCESS),
                        reporter=DummyReporter(),
                    )
                    assert 'rec' in result and 'agg' in result and 'dist' in result

def test_save_output_data(sample_df, tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    writer = MagicMock()
    writer.write_dataframe.return_value = MagicMock(path=tmp_path/'output.csv')
    result = OperationResult(status=OperationStatus.SUCCESS)
    reporter = DummyReporter()
    op._save_output_data(
        result_df=sample_df,
        task_dir=tmp_path,
        include_timestamp_in_filenames=True,
        is_encryption_required=False,
        writer=writer,
        result=result,
        reporter=reporter,
        progress_tracker=DummyProgress(),
    )
    assert any('output.csv' in str(a.path) for a in result.artifacts)

def test_prepare_directories(tmp_path):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    dirs = op._prepare_directories(tmp_path)
    assert all(Path(d).exists() for d in dirs.values())

def test_update_progress_tracker():
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    progress = DummyProgress()
    op._update_progress_tracker(5, 2, 'step', progress)
    assert progress.total == 5
    assert progress.updates[-1][1]['step'] == 'step'

def test_collect_metrics(sample_df):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    op.start_time = 0
    op.end_time = 1
    processed_df = sample_df.groupby('A').agg({'B': 'sum'})
    metrics = op._collect_metrics(sample_df, processed_df)
    assert 'total_input_records' in metrics
    assert 'num_groups' in metrics

def test_collect_aggregate_metrics(sample_df):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    processed_df = sample_df.groupby('A').agg({'B': 'sum'})
    metrics = op._collect_aggregate_metrics(sample_df, processed_df)
    assert 'num_groups' in metrics
    assert 'group_size_min' in metrics
    assert 'aggregated_field_stats' in metrics

def test_process_batch(sample_df):
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    with patch('pamola_core.transformations.grouping.aggregate_records_op.logger.warning') as mock_warn:
        out = op.process_batch(sample_df)
        mock_warn.assert_called_once()
    assert out.equals(sample_df)

def test_process_value():
    op = AggregateRecordsOperation(
        group_by_fields=['A'],
        aggregations={'B': ['sum']},
    )
    with patch('pamola_core.transformations.grouping.aggregate_records_op.logger.warning') as mock_warn:
        out = op._process_value(42)
        mock_warn.assert_called_once()
    assert out == 42

def test_create_aggregate_records_operation():
    op = create_aggregate_records_operation(group_by_fields=['A'], aggregations={'B': ['sum']})
    assert isinstance(op, AggregateRecordsOperation)

if __name__ == "__main__":
    pytest.main()
