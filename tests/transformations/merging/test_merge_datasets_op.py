"""
Unit tests for MergeDatasetsOperation in merge_datasets_op.py

These tests verify the functionality of MergeDatasetsOperation, including
various join types, relationship detection, cache, metrics, error handling, and output.

Run with:
    pytest tests/transformations/merging/test_merge_datasets_op.py
"""
import pytest
import pandas as pd
from unittest import mock
from pathlib import Path
from pamola_core.transformations.merging.merge_datasets_op import (
    MergeDatasetsOperation, create_merge_datasets_operation
)
from pamola_core.common.enum.relationship_type import RelationshipType
from pamola_core.utils.ops.op_result import OperationStatus

class DummyDataSource:
    def __init__(self, data_dict):
        self.data_dict = data_dict
    def get(self, name):
        return self.data_dict.get(name)

class DummyReporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []
    def add_operation(self, operation, details=None):
        self.operations.append((operation, details))
    def add_artifact(self, artifact_type, path, description):
        self.artifacts.append((artifact_type, path, description))

@pytest.fixture
def sample_dfs():
    left = pd.DataFrame({
        'id': [1, 2, 3],
        'val': ['a', 'b', 'c']
    })
    right = pd.DataFrame({
        'id': [2, 3, 4],
        'val2': ['x', 'y', 'z']
    })
    return left, right

@pytest.fixture
def empty_dfs():
    return pd.DataFrame({'id': [], 'val': []}), pd.DataFrame({'id': [], 'val2': []})

@pytest.fixture
def tmp_task_dir(tmp_path):
    return tmp_path

@pytest.mark.usefixtures("sample_dfs", "tmp_task_dir")
class TestMergeDatasetsOperation:
    def setup_method(self):
        self.left, self.right = pd.DataFrame({
            'id': [1, 2, 3],
            'val': ['a', 'b', 'c']
        }), pd.DataFrame({
            'id': [2, 3, 4],
            'val2': ['x', 'y', 'z']
        })
        self.data_source = DummyDataSource({'main': self.left, 'lookup': self.right})
        self.task_dir = Path("/tmp/test_merge_op")
        self.reporter = DummyReporter()

    def test_valid_case_inner_join(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        op._validate_input_params = lambda *a, **kw: None
        op._get_dataset = lambda source, name: self.left if name == 'main' else self.right
        op._detect_relationship_type_auto = lambda *a, **k: RelationshipType.ONE_TO_ONE.value
        op._validate_relationship = lambda *a, **k: None
        op._check_cache = lambda df: None
        op._collect_metrics = lambda **kwargs: {'dummy': 1}
        op._generate_visualizations = lambda *a, **k: {}
        op._save_output_data = lambda **kwargs: None
        op._save_to_cache = lambda **kwargs: None
        op._cleanup_memory = lambda *a, **k: None
        result = op.execute(self.data_source, tmp_task_dir, self.reporter)
        assert result.status == OperationStatus.SUCCESS
        assert any(getattr(a, 'description', '').find('metrics') != -1 for a in result.artifacts)

    def test_valid_case_left_join(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='left',
            relationship_type='one-to-one',
            use_cache=False
        )
        op._validate_input_params = lambda *a, **kw: None
        op._get_dataset = lambda source, name: self.left if name == 'main' else self.right
        op._detect_relationship_type_auto = lambda *a, **k: RelationshipType.ONE_TO_ONE.value
        op._validate_relationship = lambda *a, **k: None
        op._check_cache = lambda df: None
        op._collect_metrics = lambda **kwargs: {'dummy': 1}
        op._generate_visualizations = lambda *a, **k: {}
        op._save_output_data = lambda **kwargs: None
        op._save_to_cache = lambda **kwargs: None
        op._cleanup_memory = lambda *a, **k: None
        result = op.execute(self.data_source, tmp_task_dir, self.reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_edge_case_empty_inputs(self, tmp_path):
        left, right = pd.DataFrame({'id': [], 'val': []}), pd.DataFrame({'id': [], 'val2': []})
        data_source = DummyDataSource({'main': left, 'lookup': right})
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        op._validate_input_params = lambda *a, **kw: None
        op._get_dataset = lambda source, name: left if name == 'main' else right
        op._detect_relationship_type_auto = lambda *a, **k: RelationshipType.ONE_TO_ONE.value
        op._validate_relationship = lambda *a, **k: None
        op._check_cache = lambda df: None
        op._collect_metrics = lambda **kwargs: {'dummy': 1}
        op._generate_visualizations = lambda *a, **k: {}
        op._save_output_data = lambda **kwargs: None
        op._save_to_cache = lambda **kwargs: None
        op._cleanup_memory = lambda *a, **k: None
        result = op.execute(data_source, tmp_path, self.reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_invalid_input_missing_left_key(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key=None,
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        with pytest.raises(ValueError):
            op._validate_input_params('one-to-one', None, 'main', 'lookup', None)

    def test_invalid_input_missing_right(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name=None,
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        with pytest.raises(ValueError):
            op._validate_input_params('one-to-one', 'id', 'main', None, None)

    def test_invalid_relationship_type(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='invalid',
            use_cache=False
        )
        with pytest.raises(ValueError):
            op._validate_input_params('invalid', 'id', 'main', 'lookup', None)

    def test_auto_relationship_detection(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='auto',
            use_cache=False
        )
        op._validate_input_params = lambda *a, **kw: None
        op._get_dataset = lambda source, name: self.left if name == 'main' else self.right
        op._detect_relationship_type_auto = lambda *a, **k: RelationshipType.ONE_TO_ONE.value
        op._validate_relationship = lambda *a, **k: None
        op._check_cache = lambda df: None
        op._collect_metrics = lambda **kwargs: {'dummy': 1}
        op._generate_visualizations = lambda *a, **k: {}
        op._save_output_data = lambda **kwargs: None
        op._save_to_cache = lambda **kwargs: None
        op._cleanup_memory = lambda *a, **k: None
        result = op.execute(self.data_source, tmp_task_dir, self.reporter)
        assert op.relationship_type in [RelationshipType.ONE_TO_ONE.value, RelationshipType.ONE_TO_MANY.value]
        assert result.status == OperationStatus.SUCCESS

    def test_many_to_many_relationship_error(self, tmp_task_dir):
        left = pd.DataFrame({'id': [1, 1, 2], 'val': ['a', 'b', 'c']})
        right = pd.DataFrame({'id': [2, 2, 3], 'val2': ['x', 'y', 'z']})
        data_source = DummyDataSource({'main': left, 'lookup': right})
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='auto',
            use_cache=False
        )
        result = op.execute(data_source, tmp_task_dir, self.reporter)
        assert result.status == OperationStatus.ERROR

    def test_check_cache_returns_none_when_disabled(self):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        df = self.left
        assert op._check_cache(df) is None

    def test_save_to_cache_and_check_cache(self, tmp_task_dir):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=True
        )
        op.operation_cache = mock.Mock()
        op._save_to_cache(self.left, self.right, tmp_task_dir, metrics={'foo': 1})
        op.operation_cache.get_cache.return_value = {'metrics': {'foo': 1}, 'timestamp': 'now'}
        result = op._check_cache(self.left)
        assert result is not None
        assert result.metrics['cached']

    def test_cleanup_memory(self):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        op._temp_data = pd.DataFrame({'a': [1]})
        op.right_df = pd.DataFrame({'b': [2]})
        op.operation_cache = mock.Mock()
        op._cleanup_memory(self.left, self.right, self.left)
        assert op._temp_data is None
        assert op.right_df is None
        assert op.operation_cache is None

    def test_get_dataset_with_none(self):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one-to-one',
            use_cache=False
        )
        assert op._get_dataset(self.data_source, None) is None

    def test_validate_input_params(self):
        op = MergeDatasetsOperation(
            left_dataset_name='main',
            right_dataset_name='lookup',
            left_key='id',
            right_key='id',
            join_type='inner',
            relationship_type='one_to_one',
            use_cache=False
        )
        op._validate_input_params('one_to_one', 'id', 'main', 'lookup', None)
        with pytest.raises(ValueError):
            op._validate_input_params('invalid', 'id', 'main', 'lookup', None)
        with pytest.raises(ValueError):
            op._validate_input_params('one_to_one', None, 'main', 'lookup', None)
        with pytest.raises(ValueError):
            op._validate_input_params('one_to_one', 'id', None, 'lookup', None)
        with pytest.raises(ValueError):
            op._validate_input_params('one_to_one', 'id', 'main', None, None)

if __name__ == "__main__":
    pytest.main()
