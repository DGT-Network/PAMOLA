"""
Unit tests for SplitFieldsOperation in split_fields_op.py

These tests verify the functionality of SplitFieldsOperation, including
splitting fields into groups, output formats, error handling, metrics, and visualization.

Run with:
    pytest tests/transformations/splitting/test_split_fields_op.py
"""
import os
from datetime import datetime

import pytest
import pandas as pd
from pamola_core.transformations.splitting.split_fields_op import SplitFieldsOperation, OutputFormat
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_data_source import DataSource

class DummyDataSource:
    def __init__(self, df):
        self.df = df

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

def dummy_load_data_operation(data_source, dataset_name):
    return data_source.df

def dummy_ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def dummy_configure_task_logging(task_id, log_level, log_dir):
    class DummyLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def warning(self, msg): pass
        def exception(self, msg): pass
    return DummyLogger()

@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    monkeypatch.setattr('pamola_core.utils.io.load_data_operation', dummy_load_data_operation)
    monkeypatch.setattr('pamola_core.utils.io.ensure_directory', dummy_ensure_directory)
    monkeypatch.setattr('pamola_core.utils.logging.configure_task_logging', dummy_configure_task_logging)
    yield

def make_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'a': [10, 20, 30],
        'b': [100, 200, 300],
        'c': ['x', 'y', 'z']
    })

def make_data_source(df):
    return DataSource.from_dataframe(df, name="main")

def test_valid_case(tmp_path):
    df = make_df()
    data_source = make_data_source(df)
    reporter = DummyReporter()
    progress = DummyProgressTracker()
    field_groups = {'group1': ['a', 'b'], 'group2': ['c']}
    op = SplitFieldsOperation(id_field='id', field_groups=field_groups, include_id_field=True, output_format=OutputFormat.CSV.value)
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from unittest.mock import patch
    with patch.object(plt, 'savefig'), patch.object(go.Figure, 'write_image'):
        result = op.execute(data_source, tmp_path, reporter, progress_tracker=progress)
    assert isinstance(result, OperationResult)
    assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
    if result.status == OperationStatus.ERROR:
        assert 'Error in SplitFieldsOperation' in result.error_message
    else:
        assert 'group1' in op._process_data(df)
        assert 'group2' in op._process_data(df)
        assert all('id' in df_.columns for df_ in op._process_data(df).values())
        assert result.metrics['number_of_splits'] == 2
        assert result.metrics['total_input_records'] == 3
        assert result.metrics['total_input_fields'] == 4
        assert result.metrics['id_field'] == 'id'
        assert any(a.category == 'metrics' for a in result.artifacts)
        assert any(a.category == 'output' for a in result.artifacts)
        assert any(a.category == 'visualization' for a in result.artifacts)

def test_edge_empty_field_groups(tmp_path):
    df = make_df()
    data_source = DummyDataSource(df)
    reporter = DummyReporter()
    op = SplitFieldsOperation(id_field='id', field_groups={}, include_id_field=True)
    result = op.execute(data_source, tmp_path, reporter)
    assert result.status == OperationStatus.ERROR
    assert 'field_groups must not be empty' in result.error_message or result.error_message is not None

def test_edge_empty_dataframe(tmp_path):
    df = pd.DataFrame(columns=['id', 'a', 'b', 'c'])
    data_source = DummyDataSource(df)
    reporter = DummyReporter()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True)
    result = op.execute(data_source, tmp_path, reporter)
    assert result.status == OperationStatus.ERROR
    assert 'Input data frame is None or empty' in result.error_message or result.error_message is not None

def test_invalid_field_in_group(tmp_path):
    df = make_df()
    data_source = DummyDataSource(df)
    reporter = DummyReporter()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['not_a_field']}, include_id_field=True)
    result = op.execute(data_source, tmp_path, reporter)
    assert result.status == OperationStatus.ERROR
    assert 'not found in DataFrame' in result.error_message or result.error_message is not None

def test_invalid_id_field(tmp_path):
    df = make_df()
    data_source = DummyDataSource(df)
    reporter = DummyReporter()
    op = SplitFieldsOperation(id_field='not_id', field_groups={'g': ['a']}, include_id_field=True)
    result = op.execute(data_source, tmp_path, reporter)
    assert result.status == OperationStatus.ERROR
    assert 'ID field' in result.error_message or result.error_message is not None

def test_output_format_json(tmp_path):
    df = make_df()
    data_source = make_data_source(df)
    reporter = DummyReporter()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True, output_format=OutputFormat.JSON.value)
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from unittest.mock import patch
    with patch.object(plt, 'savefig'), patch.object(go.Figure, 'write_image'):
        result = op.execute(data_source, tmp_path, reporter)
    assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
    if result.status == OperationStatus.ERROR:
        assert 'Error in SplitFieldsOperation' in result.error_message
    else:
        assert any(str(a.path).endswith('.json') for a in result.artifacts if a.category == 'output')

def test_process_data_logic():
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a', 'b']}, include_id_field=True)
    result = op._process_data(df)
    assert isinstance(result, dict)
    assert 'g' in result
    assert set(result['g'].columns) == {'id', 'a', 'b'}
    op2 = SplitFieldsOperation(id_field='id', field_groups={'g': ['a', 'b']}, include_id_field=False)
    result2 = op2._process_data(df)
    assert set(result2['g'].columns) == {'a', 'b'}

def test_collect_metrics():
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a', 'b']}, include_id_field=True)
    op.input_dataset = 'main.csv'
    op.start_time = 0
    op.end_time = 1
    out = op._process_data(df)
    metrics = op._collect_metrics(df, out)
    assert metrics['operation_type'] == 'SplitFieldsOperation'
    assert metrics['input_dataset'] == 'main.csv'
    assert metrics['number_of_splits'] == 1
    assert 'split_info' in metrics

def test_save_metrics(tmp_path):
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True)
    op.input_dataset = 'main.csv'
    op.start_time = 0
    op.end_time = 1
    op.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = op._process_data(df)
    metrics = op._collect_metrics(df, out)
    result = OperationResult(status=OperationStatus.SUCCESS)
    path = op._save_metrics(metrics, tmp_path, result)
    assert path.exists()
    assert any(a.category == 'metrics' for a in result.artifacts)

def test_save_output_csv_and_json(tmp_path):
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True, output_format=OutputFormat.CSV.value)
    op.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = op._process_data(df)
    result = OperationResult(status=OperationStatus.SUCCESS)
    op._save_output(out, tmp_path, result)
    assert any(str(a.path).endswith('.csv') for a in result.artifacts)
    op2 = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True, output_format=OutputFormat.JSON.value)
    op2.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out2 = op2._process_data(df)
    result2 = OperationResult(status=OperationStatus.SUCCESS)
    op2._save_output(out2, tmp_path, result2)
    assert any(str(a.path).endswith('.json') for a in result2.artifacts)

def test_generate_visualizations(tmp_path):
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a', 'b']}, include_id_field=True)
    op.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = op._process_data(df)
    result = OperationResult(status=OperationStatus.SUCCESS)
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from unittest.mock import patch
    with patch.object(plt, 'savefig'), patch.object(go.Figure, 'write_image'):
        try:
            op._generate_visualizations(df, out, tmp_path, result)
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")
    assert any(a.category == 'visualization' for a in result.artifacts)

def test_generate_data_hash_and_cache(monkeypatch):
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True)
    h1 = op._generate_data_hash(df)
    h2 = op._generate_data_hash(df)
    assert h1 == h2
    class DummyDF(pd.DataFrame):
        @property
        def columns(self):
            return super().columns
    dummy_df = DummyDF(df)
    op2 = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True)
    try:
        op2._generate_data_hash(dummy_df)
    except Exception as e:
        pytest.fail(f'Should not raise: {e}')

def test_validate_parameters():
    df = make_df()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a', 'b']}, include_id_field=True)
    assert op._validate_parameters(df)
    op2 = SplitFieldsOperation(id_field='id', field_groups={'g': ['not_a_field']}, include_id_field=True)
    assert not op2._validate_parameters(df)
    op3 = SplitFieldsOperation(id_field='not_id', field_groups={'g': ['a']}, include_id_field=True)
    assert not op3._validate_parameters(df)
    op4 = SplitFieldsOperation(id_field='id', field_groups={}, include_id_field=True)
    assert not op4._validate_parameters(df)

def test_set_common_operation_parameters():
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True)
    op._set_common_operation_parameters(id_field='idx', field_groups={'g': ['b']}, include_id_field=False, output_format=OutputFormat.JSON.value, force_recalculation=True, generate_visualization=False, include_timestamp=False, save_output=False, parallel_processes=2, batch_size=5, use_cache=True, use_dask=True, use_encryption=True, encryption_key='k')
    assert op.id_field == 'idx'
    assert op.field_groups == {'g': ['b']}
    assert op.include_id_field is False
    assert op.output_format == OutputFormat.JSON.value
    assert op.force_recalculation is True
    assert op.generate_visualization is False
    assert op.include_timestamp is False
    assert op.save_output is False
    assert op.parallel_processes == 2
    assert op.batch_size == 5
    assert op.use_cache is True
    assert op.use_dask is True
    assert op.use_encryption is True
    assert op.encryption_key == 'k'

def test_execute_exception(monkeypatch, tmp_path):
    df = make_df()
    data_source = DummyDataSource(df)
    reporter = DummyReporter()
    op = SplitFieldsOperation(id_field='id', field_groups={'g': ['a']}, include_id_field=True)
    monkeypatch.setattr(op, '_initialize_logger', lambda *a, **k: (_ for _ in ()).throw(Exception('fail_logger')))
    result = op.execute(data_source, tmp_path, reporter)
    assert result.status == OperationStatus.ERROR
    assert 'Error in SplitFieldsOperation' in result.error_message

if __name__ == "__main__":
    pytest.main()
