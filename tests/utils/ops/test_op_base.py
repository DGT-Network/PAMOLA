"""
Unit tests for pamola_core.utils.ops.op_base module.

These tests verify the functionality of the BaseOperation class and its derivatives,
including configuration serialization, operation registration, and execution lifecycle.

Run with: pytest -s tests/utils/ops/test_op_base.py
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from pamola_core.utils.ops.op_base import BaseOperation, FieldOperation, DataFrameOperation, ConfigSaveError, OperationScope
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class DummyReporter:
    """Simple reporter for testing."""

    def __init__(self):
        self.operations = []

    def add_operation(self, name, status=None, details=None):
        self.operations.append({
            'name': name,
            'status': status,
            'details': details
        })


# Renamed test operation class to avoid pytest collection warning
class OperationForTest(BaseOperation):
    """Test operation for unit tests."""

    def __init__(self, **kwargs):
        super().__init__(
            name="test_operation",
            description="Test operation for unit tests",
            config=OperationConfig(test_param="test_value"),
            **kwargs
        )

    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        # Access configuration
        test_param = self.config.get("test_param")

        # Prepare directories
        dirs = self._prepare_directories(task_dir)

        # Write a test file
        test_file = dirs["output"] / "test_output.txt"
        with open(test_file, "w") as f:
            f.write(f"Test output with param: {test_param}")

        # Return successful result
        result = OperationResult(status=OperationStatus.SUCCESS)
        result.add_artifact("test_output", test_file)
        result.add_metric("test_metric", 42)

        return result


# Renamed failing operation class to avoid pytest collection warning
class FailingOperationForTest(BaseOperation):
    """Test operation that fails during execution."""

    def __init__(self, **kwargs):
        super().__init__(
            name="failing_operation",
            description="Operation that fails",
            **kwargs
        )

    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        raise ValueError("Simulated failure")


# Renamed field operation class to avoid pytest collection warning
class FieldOpForTest(FieldOperation):
    """Test field operation."""

    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        return OperationResult(status=OperationStatus.SUCCESS)


# Renamed dataframe operation class to avoid pytest collection warning
class DataFrameOpForTest(DataFrameOperation):
    """Test DataFrame operation."""

    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        return OperationResult(status=OperationStatus.SUCCESS)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_data_source():
    """Create a mocked DataSource."""
    mock_ds = mock.MagicMock(spec=DataSource)
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    # Setup get_dataframe to return the DataFrame
    mock_ds.get_dataframe.return_value = (df, None)
    return mock_ds


class TestBaseOperation:
    """Tests for the BaseOperation class."""

    def test_initialization(self):
        """Test that BaseOperation initializes correctly."""
        op = OperationForTest()

        assert op.name == "test_operation"
        assert op.description == "Test operation for unit tests"
        assert op.version == "1.0.0"
        assert op.use_encryption is False
        assert op.encryption_key is None
        assert op.use_vectorization is False
        assert isinstance(op.config, OperationConfig)

    def test_save_config(self, temp_dir):
        """Test that save_config correctly serializes and writes config."""
        # Create operation
        op = OperationForTest()

        # Save config
        op.save_config(temp_dir)

        # Check that config file exists
        config_path = temp_dir / "config.json"
        assert config_path.exists()

        # Check config content
        with open(config_path, 'r') as f:
            config = json.load(f)

        assert config["operation_name"] == op.name
        assert config["version"] == op.version
        assert config["test_param"] == "test_value"

    def test_save_config_atomic(self, temp_dir):
        """Test that save_config is atomic and handles errors."""
        op = OperationForTest()

        # Mock open to fail
        with mock.patch('builtins.open', side_effect=IOError("Simulated IO error")):
            with pytest.raises(ConfigSaveError):
                op.save_config(temp_dir)

        # Check that no temp file remains
        temp_path = temp_dir / "config.json.tmp"
        assert not temp_path.exists()

    def test_prepare_directories(self, temp_dir):
        """Test that _prepare_directories creates all required directories."""
        op = OperationForTest()
        dirs = op._prepare_directories(temp_dir)

        # Check that all directories were created
        assert (temp_dir / "output").is_dir()
        assert (temp_dir / "dictionaries").is_dir()
        assert (temp_dir / "visualizations").is_dir()
        assert (temp_dir / "logs").is_dir()

        # Check that dirs dictionary has correct paths
        assert dirs["output"] == temp_dir / "output"
        assert dirs["dictionaries"] == temp_dir / "dictionaries"
        assert dirs["visualizations"] == temp_dir / "visualizations"
        assert dirs["logs"] == temp_dir / "logs"

    def test_run_successful(self, temp_dir, mock_data_source):
        """Test successful operation execution through run."""
        # Create operation and reporter
        op = OperationForTest()
        reporter = DummyReporter()

        # Run operation
        result = op.run(
            data_source=mock_data_source,
            task_dir=temp_dir,
            reporter=reporter,
            track_progress=True
        )

        # Check result
        assert result.status == OperationStatus.SUCCESS

        # Fix: Check if an artifact's path's name contains "test_output"
        found = False
        for artifact in result.artifacts:
            if hasattr(artifact, 'path') and "test_output" in str(artifact.path):
                found = True
                break
        assert found, "test_output artifact not found"

        assert "test_metric" in result.metrics
        assert result.metrics["test_metric"] == 42
        assert result.execution_time is not None

        # Check reporter
        assert len(reporter.operations) == 2  # Start and end operations
        assert reporter.operations[0]["name"] == "test_operation"
        assert reporter.operations[1]["name"] == "test_operation completed"
        assert reporter.operations[1]["status"] == "success"

        # Check files
        config_path = temp_dir / "config.json"
        assert config_path.exists()
        output_path = temp_dir / "output" / "test_output.txt"
        assert output_path.exists()

        # Check execution time
        assert op.get_execution_time() is not None

    def test_run_failing(self, temp_dir, mock_data_source):
        """Test failing operation execution through run."""
        # Create operation and reporter
        op = FailingOperationForTest()
        reporter = DummyReporter()

        # Run operation
        result = op.run(
            data_source=mock_data_source,
            task_dir=temp_dir,
            reporter=reporter,
            track_progress=True
        )

        # Check result
        assert result.status == OperationStatus.ERROR
        assert result.error_message == "Simulated failure"
        assert result.execution_time is not None

        # Check reporter
        assert len(reporter.operations) == 2  # Start and end operations
        assert reporter.operations[0]["name"] == "failing_operation"
        assert reporter.operations[1]["name"] == "failing_operation failed"
        assert reporter.operations[1]["status"] == "error"

        # Check config was still saved
        config_path = temp_dir / "config.json"
        assert config_path.exists()

    def test_encryption_warning(self, temp_dir, mock_data_source):
        """Test warning when encryption is requested without key."""
        # Create operation with encryption but no key
        op = OperationForTest(use_encryption=True)
        reporter = DummyReporter()

        # Mock logger
        with mock.patch.object(op, 'logger') as mock_logger:
            # Run operation
            op.run(
                data_source=mock_data_source,
                task_dir=temp_dir,
                reporter=reporter
            )

            # Check warning
            mock_logger.warning.assert_any_call(
                "Encryption requested but no key provided, disabling encryption"
            )

    def test_vectorization_warning(self, temp_dir, mock_data_source):
        """Test warning when vectorization is requested without enough processes."""
        # Create operation with vectorization
        op = OperationForTest(use_vectorization=True)
        reporter = DummyReporter()

        # Mock logger
        with mock.patch.object(op, 'logger') as mock_logger:
            # Run operation with parallel_processes=1
            op.run(
                data_source=mock_data_source,
                task_dir=temp_dir,
                reporter=reporter,
                parallel_processes=1
            )

            # Check warning
            mock_logger.warning.assert_any_call(
                "Vectorization requested but parallel_processes <= 1, disabling vectorization"
            )

    def test_context_manager(self, temp_dir, mock_data_source):
        """Test using operation as context manager."""
        reporter = DummyReporter()

        with OperationForTest() as op:
            result = op.run(
                data_source=mock_data_source,
                task_dir=temp_dir,
                reporter=reporter
            )
            assert result.status == OperationStatus.SUCCESS

    def test_get_version(self):
        """Test get_version method."""
        op = OperationForTest()
        assert op.get_version() == "1.0.0"

        # Set custom version
        op.version = "2.1.0"
        assert op.get_version() == "2.1.0"


class TestOperationScope:
    """Tests for the OperationScope class."""

    def test_initialization(self):
        """Test that OperationScope initializes correctly."""
        # Empty scope
        scope = OperationScope()
        assert scope.fields == []
        assert scope.datasets == []
        assert scope.field_groups == {}

        # Scope with parameters
        scope = OperationScope(
            fields=["field1", "field2"],
            datasets=["ds1"],
            field_groups={"group1": ["field1", "field2"]}
        )
        assert scope.fields == ["field1", "field2"]
        assert scope.datasets == ["ds1"]
        assert scope.field_groups == {"group1": ["field1", "field2"]}

    def test_add_field(self):
        """Test add_field method."""
        scope = OperationScope()
        scope.add_field("field1")
        assert scope.fields == ["field1"]

        # Adding same field twice should not duplicate
        scope.add_field("field1")
        assert scope.fields == ["field1"]

        scope.add_field("field2")
        assert scope.fields == ["field1", "field2"]

    def test_add_dataset(self):
        """Test add_dataset method."""
        scope = OperationScope()
        scope.add_dataset("ds1")
        assert scope.datasets == ["ds1"]

        # Adding same dataset twice should not duplicate
        scope.add_dataset("ds1")
        assert scope.datasets == ["ds1"]

        scope.add_dataset("ds2")
        assert scope.datasets == ["ds1", "ds2"]

    def test_add_field_group(self):
        """Test add_field_group method."""
        scope = OperationScope()
        scope.add_field_group("group1", ["field1", "field2"])
        assert scope.field_groups == {"group1": ["field1", "field2"]}

        # Adding another group
        scope.add_field_group("group2", ["field3"])
        assert scope.field_groups == {
            "group1": ["field1", "field2"],
            "group2": ["field3"]
        }

        # Overwriting existing group
        scope.add_field_group("group1", ["field4"])
        assert scope.field_groups == {
            "group1": ["field4"],
            "group2": ["field3"]
        }

    def test_has_field(self):
        """Test has_field method."""
        scope = OperationScope(fields=["field1", "field2"])
        assert scope.has_field("field1") is True
        assert scope.has_field("field2") is True
        assert scope.has_field("field3") is False

    def test_has_dataset(self):
        """Test has_dataset method."""
        scope = OperationScope(datasets=["ds1", "ds2"])
        assert scope.has_dataset("ds1") is True
        assert scope.has_dataset("ds2") is True
        assert scope.has_dataset("ds3") is False

    def test_has_field_group(self):
        """Test has_field_group method."""
        scope = OperationScope(field_groups={
            "group1": ["field1", "field2"],
            "group2": ["field3"]
        })
        assert scope.has_field_group("group1") is True
        assert scope.has_field_group("group2") is True
        assert scope.has_field_group("group3") is False

    def test_get_fields_in_group(self):
        """Test get_fields_in_group method."""
        scope = OperationScope(field_groups={
            "group1": ["field1", "field2"],
            "group2": ["field3"]
        })
        assert scope.get_fields_in_group("group1") == ["field1", "field2"]
        assert scope.get_fields_in_group("group2") == ["field3"]
        assert scope.get_fields_in_group("group3") == []  # Non-existent group returns empty list

    def test_to_dict(self):
        """Test to_dict method."""
        scope = OperationScope(
            fields=["field1", "field2"],
            datasets=["ds1"],
            field_groups={"group1": ["field1", "field2"]}
        )
        data = scope.to_dict()

        assert data["fields"] == ["field1", "field2"]
        assert data["datasets"] == ["ds1"]
        assert data["field_groups"] == {"group1": ["field1", "field2"]}

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "fields": ["field1", "field2"],
            "datasets": ["ds1"],
            "field_groups": {"group1": ["field1", "field2"]}
        }
        scope = OperationScope.from_dict(data)

        assert scope.fields == ["field1", "field2"]
        assert scope.datasets == ["ds1"]
        assert scope.field_groups == {"group1": ["field1", "field2"]}


class TestFieldOperation:
    """Tests for the FieldOperation class."""

    def test_initialization(self):
        """Test that FieldOperation initializes correctly."""
        op = FieldOpForTest(field_name="test_field")

        assert op.name == "test_field analysis"
        assert "Analysis of test_field field" in op.description
        assert op.field_name == "test_field"
        assert op.scope.fields == ["test_field"]

    def test_add_related_field(self):
        """Test add_related_field method."""
        op = FieldOpForTest(field_name="test_field")
        op.add_related_field("related_field")

        assert "test_field" in op.scope.fields
        assert "related_field" in op.scope.fields
        assert len(op.scope.fields) == 2

    def test_validate_field_existence(self):
        """Test validate_field_existence method."""
        op = FieldOpForTest(field_name="test_field")

        # Field exists
        df = pd.DataFrame({
            "test_field": [1, 2, 3],
            "other_field": ["a", "b", "c"]
        })
        assert op.validate_field_existence(df) is True

        # Field does not exist
        df = pd.DataFrame({
            "other_field": ["a", "b", "c"]
        })
        assert op.validate_field_existence(df) is False


class TestDataFrameOperation:
    """Tests for the DataFrameOperation class."""

    def test_initialization(self):
        """Test that DataFrameOperation initializes correctly."""
        op = DataFrameOpForTest(name="test_df_op")

        assert op.name == "test_df_op"
        assert isinstance(op.scope, OperationScope)

    def test_add_field_group(self):
        """Test add_field_group method."""
        op = DataFrameOpForTest(name="test_df_op")
        op.add_field_group("group1", ["field1", "field2"])

        assert op.scope.has_field_group("group1") is True
        assert op.scope.get_fields_in_group("group1") == ["field1", "field2"]

    def test_get_field_group(self):
        """Test get_field_group method."""
        op = DataFrameOpForTest(name="test_df_op")
        op.add_field_group("group1", ["field1", "field2"])

        assert op.get_field_group("group1") == ["field1", "field2"]
        assert op.get_field_group("group2") == []  # Non-existent group returns empty list


if __name__ == "__main__":
    pytest.main()