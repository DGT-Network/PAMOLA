"""
Tests for the base_task module in the pamola_core/utils/tasks package.

These tests ensure that the BaseTask class properly implements the task lifecycle,
operation handling, error management, and other pamola core functionality.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.tasks.base_task import (
    BaseTask
)


# Mock classes for testing
class MockArtifact:
    """Mock artifact for testing."""

    def __init__(self, path, artifact_type, description):
        self.path = path
        self.artifact_type = artifact_type
        self.description = description
        self.category = "test"
        self.tags = ["test"]
        self.metadata = {}


class MockOperation:
    """Mock operation for testing."""

    def __init__(self, name="MockOperation", should_fail=False, **kwargs):
        self.name = name
        self.should_fail = should_fail
        self.kwargs = kwargs
        self.run_called = False

    def run(self, **kwargs):
        self.run_called = True
        if self.should_fail:
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message="Operation failed intentionally",
                execution_time=0.1,
                artifacts=[],
                metrics={}
            )
        return OperationResult(
            status=OperationStatus.SUCCESS,
            error_message=None,
            execution_time=0.1,
            artifacts=[MockArtifact("test.json", "json", "Test artifact")],
            metrics={"metric1": 10, "metric2": "value"}
        )


class TestTask(BaseTask):
    """Concrete task implementation for testing."""

    def __init__(self, task_id="test_task", task_type="test_type", description="Test Task"):
        super().__init__(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_datasets={"test_data": "test_data.csv"},
            auxiliary_datasets={"aux_data": "aux_data.csv"}
        )
        self.configure_operations_called = False

    def configure_operations(self):
        self.configure_operations_called = True
        self.add_operation("MockOperation")


class TestTaskWithError(BaseTask):
    """Task implementation that raises an error during operation execution."""

    def __init__(self):
        super().__init__(
            task_id="error_task",
            task_type="test_type",
            description="Task with error"
        )

    def configure_operations(self):
        self.add_operation("MockOperation", should_fail=True)


class TestBaseTask:
    """Tests for the BaseTask class."""

    @pytest.fixture
    def setup_environment(self):
        """Setup environment for testing."""
        with TemporaryDirectory() as temp_dir:
            # Create mock directory structure
            temp_path = Path(temp_dir)
            configs_dir = temp_path / "configs"
            configs_dir.mkdir()
            data_dir = temp_path / "DATA"
            data_dir.mkdir()
            logs_dir = temp_path / "logs"
            logs_dir.mkdir()

            # Create mock project config
            project_config = {
                "data_repository": "DATA",
                "log_level": "INFO",
                "tasks": {
                    "test_task": {
                        "continue_on_error": True
                    }
                }
            }

            with open(configs_dir / "prj_config.json", "w") as f:
                json.dump(project_config, f)

            # Create task directories
            processed_dir = data_dir / "processed" / "test_type" / "test_task"
            processed_dir.mkdir(parents=True)

            # Yield the temporary path for tests to use
            yield temp_path

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for testing."""
        # Mock key functions to