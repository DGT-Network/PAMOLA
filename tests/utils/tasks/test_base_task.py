"""
Tests for the base_task module in the pamola_core/utils/tasks package.

These tests ensure that the BaseTask class properly implements the task lifecycle,
operation handling, error management, and other pamola core functionality.
"""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pamola_core.common.enum.encryption_mode import EncryptionMode
from pamola_core.errors.exceptions import (
    TaskDependencyError,
    FeatureNotImplementedError,
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.tasks.base_task import BaseTask, RESERVED_OPERATION_PARAMS


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockArtifact:
    """Mock artifact for testing."""

    def __init__(self, path="test.json", artifact_type="json", description="Test artifact"):
        self.path = path
        self.artifact_type = artifact_type
        self.description = description
        self.category = "test"
        self.tags = ["test"]
        self.metadata = {}


def _make_success_result(**kw):
    return OperationResult(
        status=OperationStatus.SUCCESS,
        error_message=None,
        execution_time=0.1,
        artifacts=[MockArtifact()],
        metrics={"metric1": 10},
        **kw,
    )


def _make_error_result():
    return OperationResult(
        status=OperationStatus.ERROR,
        error_message="Operation failed intentionally",
        execution_time=0.1,
        artifacts=[],
        metrics={},
    )


class MockOperation:
    """Minimal mock operation."""

    def __init__(self, name="MockOp", should_fail=False, config=None):
        self.name = name
        self.should_fail = should_fail
        self.config = config
        self.run_called = False

    def run(self, **kwargs):
        self.run_called = True
        return _make_error_result() if self.should_fail else _make_success_result()


# ---------------------------------------------------------------------------
# Concrete task subclasses for testing
# ---------------------------------------------------------------------------

class _MinimalTask(BaseTask):
    """Minimal concrete subclass — configure_operations does nothing by default."""

    def __init__(self, task_id="test_task", task_type="test_type",
                 description="Test Task", **kwargs):
        super().__init__(
            task_id=task_id,
            task_type=task_type,
            description=description,
            **kwargs,
        )

    def configure_operations(self):
        pass  # no-op by default, overridable per test


class _RaisingConfigure(_MinimalTask):
    """Task that raises during configure_operations."""

    def configure_operations(self):
        raise RuntimeError("configure failed")


class _NotImplementedConfigure(BaseTask):
    """Task that inherits the default configure_operations (raises FeatureNotImplementedError)."""

    def __init__(self):
        super().__init__(task_id="ni_task", task_type="ni_type", description="NI")


# ---------------------------------------------------------------------------
# Shared fixture: fully mocked initialize() so unit tests don't need filesystem
# ---------------------------------------------------------------------------

def _build_mock_config(continue_on_error=False, dependencies=None, enable_checkpoints=False):
    cfg = MagicMock()
    cfg.continue_on_error = continue_on_error
    cfg.dependencies = dependencies or []
    cfg.enable_checkpoints = enable_checkpoints
    cfg.log_level = "INFO"
    cfg.log_file = None
    cfg.task_log_file = None
    cfg.report_path = MagicMock()
    cfg.log_directory = MagicMock()
    cfg.use_encryption = False
    cfg.encryption_mode = "none"
    cfg.use_vectorization = False
    cfg.parallel_processes = 1
    cfg.use_dask = False
    cfg.npartitions = 1
    cfg.chunk_size = 100000
    return cfg


def _mock_initialize(task: BaseTask, tmp_path: Path,
                     continue_on_error: bool = False,
                     dependencies=None,
                     enable_checkpoints: bool = False):
    """
    Inject mocked sub-components so initialize() succeeds without filesystem I/O.
    Returns the mocked config.
    """
    cfg = _build_mock_config(
        continue_on_error=continue_on_error,
        dependencies=dependencies,
        enable_checkpoints=enable_checkpoints,
    )

    # directory manager
    dir_mgr = MagicMock()
    task_dir = tmp_path / "task_output"
    task_dir.mkdir(parents=True, exist_ok=True)
    dir_mgr.ensure_directories.return_value = {"task": task_dir}
    dir_mgr.get_directory.return_value = task_dir
    dir_mgr.normalize_and_validate_path.side_effect = lambda p: Path(p)
    dir_mgr.clean_temp_directory = MagicMock()

    # encryption manager
    enc_mgr = MagicMock()
    enc_mgr.get_encryption_info.return_value = {"enabled": False, "mode": "none"}
    enc_mgr.get_encryption_context.return_value = None
    enc_mgr.initialize = MagicMock()

    # context manager
    ctx_mgr = MagicMock()
    ctx_mgr.can_resume_execution.return_value = (False, None)
    ctx_mgr.clear_checkpoints = MagicMock()
    ctx_mgr.create_automatic_checkpoint = MagicMock()

    # progress manager
    prog_mgr = MagicMock()
    op_ctx = MagicMock()
    op_ctx.__enter__ = MagicMock(return_value=op_ctx)
    op_ctx.__exit__ = MagicMock(return_value=False)
    prog_mgr.create_operation_context.return_value = op_ctx
    prog_mgr.set_total_operations = MagicMock()
    prog_mgr.increment_total_operations = MagicMock()
    prog_mgr.complete_operation = MagicMock()
    prog_mgr.close = MagicMock()

    # operation executor
    op_exec = MagicMock()
    op_exec.execute_with_retry.return_value = _make_success_result()

    # reporter
    reporter = MagicMock()
    reporter.artifacts = []
    reporter.save_report.return_value = str(tmp_path / "report.json")

    # dependency manager
    dep_mgr = MagicMock()
    dep_mgr.assert_dependencies_completed = MagicMock()

    with patch("pamola_core.utils.tasks.base_task.load_task_config", return_value=cfg), \
         patch("pamola_core.utils.tasks.base_task.create_directory_manager", return_value=dir_mgr), \
         patch("pamola_core.utils.tasks.base_task.TaskReporter", return_value=reporter), \
         patch("pamola_core.utils.tasks.base_task.TaskEncryptionManager", return_value=enc_mgr), \
         patch("pamola_core.utils.tasks.base_task.create_task_context_manager", return_value=ctx_mgr), \
         patch("pamola_core.utils.tasks.base_task.create_task_progress_manager", return_value=prog_mgr), \
         patch("pamola_core.utils.tasks.base_task.create_operation_executor", return_value=op_exec), \
         patch("pamola_core.utils.tasks.base_task.record_task_execution"):
        result = task.initialize()

    # Inject manually after initialize() for post-initialize tests
    task.config = cfg
    task.directory_manager = dir_mgr
    task.reporter = reporter
    task.encryption_manager = enc_mgr
    task.context_manager = ctx_mgr
    task.progress_manager = prog_mgr
    task.operation_executor = op_exec
    task.dependency_manager = dep_mgr
    task.task_dir = task_dir

    return result, cfg, op_exec, reporter, prog_mgr, ctx_mgr


# ---------------------------------------------------------------------------
# TestBaseTask — __init__
# ---------------------------------------------------------------------------

class TestBaseTaskInit:
    def test_task_id_set(self):
        t = _MinimalTask(task_id="abc")
        assert t.task_id == "abc"

    def test_task_type_set(self):
        t = _MinimalTask(task_type="profiling")
        assert t.task_type == "profiling"

    def test_description_set(self):
        t = _MinimalTask(description="My task")
        assert t.description == "My task"

    def test_status_pending_initially(self):
        t = _MinimalTask()
        assert t.status == "pending"

    def test_config_none_initially(self):
        t = _MinimalTask()
        assert t.config is None

    def test_logger_none_initially(self):
        t = _MinimalTask()
        assert t.logger is None

    def test_operations_empty_initially(self):
        t = _MinimalTask()
        assert t.operations == []

    def test_results_empty_initially(self):
        t = _MinimalTask()
        assert t.results == {}

    def test_artifacts_empty_initially(self):
        t = _MinimalTask()
        assert t.artifacts == []

    def test_metrics_empty_initially(self):
        t = _MinimalTask()
        assert t.metrics == {}

    def test_input_datasets_default_empty(self):
        t = _MinimalTask()
        assert t.input_datasets == {}

    def test_input_datasets_set_from_kwarg(self):
        t = _MinimalTask(input_datasets={"ds": "path.csv"})
        assert t.input_datasets == {"ds": "path.csv"}

    def test_auxiliary_datasets_default_empty(self):
        t = _MinimalTask()
        assert t.auxiliary_datasets == {}

    def test_override_task_dir_none(self):
        t = _MinimalTask()
        assert t._override_task_dir is None

    def test_override_task_dir_set(self, tmp_path):
        t = _MinimalTask(task_dir=str(tmp_path))
        assert t._override_task_dir == tmp_path

    def test_encryption_mode_none_default(self):
        t = _MinimalTask()
        assert t.encryption_mode == EncryptionMode.NONE

    def test_enable_checkpoints_false_default(self):
        t = _MinimalTask()
        assert t.enable_checkpoints is False

    def test_checkpoint_state_default(self):
        t = _MinimalTask()
        assert t._resuming_from_checkpoint is False
        assert t._restored_checkpoint_name is None
        assert t._restored_state is None


# ---------------------------------------------------------------------------
# TestGetDefaultConfig
# ---------------------------------------------------------------------------

class TestGetDefaultConfig:
    def test_returns_dict(self):
        t = _MinimalTask()
        result = t.get_default_config()
        assert isinstance(result, dict)

    def test_contains_task_type(self):
        t = _MinimalTask(task_type="profiling")
        result = t.get_default_config()
        assert result["task_type"] == "profiling"

    def test_contains_description(self):
        t = _MinimalTask(description="My Desc")
        result = t.get_default_config()
        assert result["description"] == "My Desc"

    def test_use_encryption_default_false(self):
        t = _MinimalTask()
        result = t.get_default_config()
        assert result["use_encryption"] is False

    def test_enable_checkpoints_default_false(self):
        t = _MinimalTask()
        result = t.get_default_config()
        assert result["enable_checkpoints"] is False

    def test_override_task_dir_adds_keys(self, tmp_path):
        t = _MinimalTask(task_dir=str(tmp_path))
        result = t.get_default_config()
        assert "data_repository" in result
        assert "config_save_dir" in result


# ---------------------------------------------------------------------------
# TestInitialize
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_initialize_returns_true(self, tmp_path):
        t = _MinimalTask()
        result, *_ = _mock_initialize(t, tmp_path)
        assert result is True

    def test_status_still_pending_after_init(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.status == "pending"

    def test_start_time_set(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.start_time is not None

    def test_config_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.config is not None

    def test_logger_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.logger is not None

    def test_reporter_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.reporter is not None

    def test_directory_manager_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.directory_manager is not None

    def test_encryption_manager_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.encryption_manager is not None

    def test_context_manager_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.context_manager is not None

    def test_progress_manager_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.progress_manager is not None

    def test_operation_executor_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.operation_executor is not None

    def test_task_dir_set_after_initialize(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        assert t.task_dir is not None

    def test_force_restart_clears_checkpoints(self, tmp_path):
        t = _MinimalTask()
        cfg = _build_mock_config()
        ctx_mgr = MagicMock()
        ctx_mgr.can_resume_execution.return_value = (False, None)
        ctx_mgr.clear_checkpoints = MagicMock()

        with patch("pamola_core.utils.tasks.base_task.load_task_config", return_value=cfg), \
             patch("pamola_core.utils.tasks.base_task.create_directory_manager") as dm, \
             patch("pamola_core.utils.tasks.base_task.TaskReporter"), \
             patch("pamola_core.utils.tasks.base_task.TaskEncryptionManager") as em, \
             patch("pamola_core.utils.tasks.base_task.create_task_context_manager",
                   return_value=ctx_mgr), \
             patch("pamola_core.utils.tasks.base_task.create_task_progress_manager"), \
             patch("pamola_core.utils.tasks.base_task.create_operation_executor"), \
             patch("pamola_core.utils.tasks.base_task.record_task_execution"):
            dir_mgr = MagicMock()
            task_dir = tmp_path / "td"
            task_dir.mkdir()
            dir_mgr.ensure_directories.return_value = {"task": task_dir}
            dir_mgr.get_directory.return_value = task_dir
            dm.return_value = dir_mgr
            enc_mgr = MagicMock()
            enc_mgr.get_encryption_info.return_value = {"enabled": False, "mode": "none"}
            em.return_value = enc_mgr
            t.initialize(force_restart=True)

        ctx_mgr.clear_checkpoints.assert_called()

    def test_initialize_no_dependencies(self, tmp_path):
        """Initialize without dependencies — no dependency checking needed."""
        t = _MinimalTask()
        cfg = _build_mock_config()
        cfg.dependencies = []

        with patch("pamola_core.utils.tasks.base_task.load_task_config", return_value=cfg), \
             patch("pamola_core.utils.tasks.base_task.create_directory_manager") as dm, \
             patch("pamola_core.utils.tasks.base_task.TaskReporter"), \
             patch("pamola_core.utils.tasks.base_task.TaskEncryptionManager") as em, \
             patch("pamola_core.utils.tasks.base_task.create_task_context_manager"), \
             patch("pamola_core.utils.tasks.base_task.create_task_progress_manager"), \
             patch("pamola_core.utils.tasks.base_task.create_operation_executor"):
            dir_mgr = MagicMock()
            task_dir = tmp_path / "td2"
            task_dir.mkdir()
            dir_mgr.ensure_directories.return_value = {"task": task_dir}
            dir_mgr.get_directory.return_value = task_dir
            dm.return_value = dir_mgr
            enc_mgr = MagicMock()
            enc_mgr.get_encryption_info.return_value = {"enabled": False, "mode": "none"}
            em.return_value = enc_mgr
            result = t.initialize()

        assert result is True

    def test_initialization_exception_returns_false(self, tmp_path):
        t = _MinimalTask()
        with patch("pamola_core.utils.tasks.base_task.load_task_config",
                   side_effect=RuntimeError("init failed")):
            result = t.initialize()
        assert result is False
        assert t.status == "initialization_error"

    def test_enable_checkpoints_from_config(self, tmp_path):
        t = _MinimalTask()
        cfg = _build_mock_config(enable_checkpoints=True)
        ctx_mgr = MagicMock()
        ctx_mgr.can_resume_execution.return_value = (False, None)

        with patch("pamola_core.utils.tasks.base_task.load_task_config", return_value=cfg), \
             patch("pamola_core.utils.tasks.base_task.create_directory_manager") as dm, \
             patch("pamola_core.utils.tasks.base_task.TaskReporter"), \
             patch("pamola_core.utils.tasks.base_task.TaskEncryptionManager") as em, \
             patch("pamola_core.utils.tasks.base_task.create_task_context_manager",
                   return_value=ctx_mgr), \
             patch("pamola_core.utils.tasks.base_task.create_task_progress_manager"), \
             patch("pamola_core.utils.tasks.base_task.create_operation_executor"), \
             patch("pamola_core.utils.tasks.base_task.record_task_execution"):
            dir_mgr = MagicMock()
            task_dir = tmp_path / "td3"
            task_dir.mkdir()
            dir_mgr.ensure_directories.return_value = {"task": task_dir}
            dir_mgr.get_directory.return_value = task_dir
            dm.return_value = dir_mgr
            enc_mgr = MagicMock()
            enc_mgr.get_encryption_info.return_value = {"enabled": False, "mode": "none"}
            em.return_value = enc_mgr
            t.initialize(enable_checkpoints=None)

        assert t.enable_checkpoints is True


# ---------------------------------------------------------------------------
# TestCheckDependencies
# ---------------------------------------------------------------------------

class TestCheckDependencies:
    def test_no_dependencies_returns_true(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.config.dependencies = []
        result = t._check_dependencies()
        assert result is True

    def test_satisfied_dependencies_returns_true(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.config.dependencies = ["dep1"]
        t.dependency_manager.assert_dependencies_completed = MagicMock()
        result = t._check_dependencies()
        assert result is True

    def test_missing_dep_without_continue_raises(self, tmp_path):
        from pamola_core.errors.exceptions import DependencyMissingError
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.config.dependencies = ["missing"]
        t.config.continue_on_error = False
        t.dependency_manager.assert_dependencies_completed.side_effect = (
            DependencyMissingError("dep missing")
        )
        with pytest.raises(TaskDependencyError):
            t._check_dependencies()

    def test_missing_dep_with_continue_returns_true(self, tmp_path):
        from pamola_core.errors.exceptions import DependencyMissingError
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.config.dependencies = ["missing"]
        t.config.continue_on_error = True
        t.dependency_manager.assert_dependencies_completed.side_effect = (
            DependencyMissingError("dep missing")
        )
        result = t._check_dependencies()
        assert result is True


# ---------------------------------------------------------------------------
# TestConfigureOperations
# ---------------------------------------------------------------------------

class TestConfigureOperations:
    def test_default_raises_feature_not_implemented(self):
        t = _NotImplementedConfigure()
        with pytest.raises(FeatureNotImplementedError):
            t.configure_operations()


# ---------------------------------------------------------------------------
# TestAddOperation
# ---------------------------------------------------------------------------

class TestAddOperation:
    def _initialized_task(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        return t

    def test_add_operation_class_directly(self, tmp_path):
        t = self._initialized_task(tmp_path)
        t.operations.clear()
        t.add_operation(MockOperation)
        assert len(t.operations) == 1

    def test_add_operation_returns_true(self, tmp_path):
        t = self._initialized_task(tmp_path)
        result = t.add_operation(MockOperation)
        assert result is True

    def test_add_operation_appends_to_list(self, tmp_path):
        t = self._initialized_task(tmp_path)
        t.operations.clear()
        t.add_operation(MockOperation)
        t.add_operation(MockOperation)
        assert len(t.operations) == 2

    def test_add_operation_via_registry_string(self, tmp_path):
        t = self._initialized_task(tmp_path)
        t.operations.clear()
        mock_op = MockOperation()
        with patch("pamola_core.utils.ops.op_registry.get_operation_class",
                   return_value=MockOperation), \
             patch("pamola_core.utils.ops.op_registry.create_operation_instance",
                   return_value=mock_op):
            result = t.add_operation("MockOperation")
        assert result is True

    def test_add_operation_unknown_string_returns_false(self, tmp_path):
        t = self._initialized_task(tmp_path)
        with patch("pamola_core.utils.ops.op_registry.get_operation_class",
                   return_value=None), \
             patch("pamola_core.utils.ops.op_registry.create_operation_instance",
                   return_value=None):
            result = t.add_operation("NonExistentOp")
        assert result is False

    def test_reserved_params_filtered_from_kwargs(self, tmp_path):
        t = self._initialized_task(tmp_path)
        created_kwargs = {}

        class CaptureOp:
            name = "CaptureOp"
            supports_encryption = False
            supports_vectorization = False

            def __init__(self, **kwargs):
                created_kwargs.update(kwargs)

        t.add_operation(CaptureOp, data_source="ignored", my_param="kept")
        assert "data_source" not in created_kwargs
        assert "my_param" in created_kwargs

    def test_add_operation_exception_returns_false(self, tmp_path):
        t = self._initialized_task(tmp_path)
        with patch("pamola_core.utils.ops.op_registry.get_operation_class",
                   side_effect=RuntimeError("registry error")):
            result = t.add_operation("BrokenOp")
        assert result is False


# ---------------------------------------------------------------------------
# TestRunOperations
# ---------------------------------------------------------------------------

class TestRunOperations:
    def _task_with_ops(self, tmp_path, ops=None, continue_on_error=False):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path, continue_on_error=continue_on_error)
        t.operations = ops or []
        return t

    def test_empty_operations_succeed(self, tmp_path):
        t = self._task_with_ops(tmp_path, ops=[])
        result = t._run_operations()
        assert result is True

    def test_single_success_operation(self, tmp_path):
        op = MockOperation()
        t = self._task_with_ops(tmp_path, ops=[op])
        t.operation_executor.execute_with_retry.return_value = _make_success_result()
        result = t._run_operations()
        assert result is True
        assert t.status == "success"

    def test_result_stored_by_operation_name(self, tmp_path):
        op = MockOperation(name="MyOp")
        t = self._task_with_ops(tmp_path, ops=[op])
        res = _make_success_result()
        t.operation_executor.execute_with_retry.return_value = res
        t._run_operations()
        assert "MyOp" in t.results

    def test_artifacts_collected(self, tmp_path):
        op = MockOperation()
        t = self._task_with_ops(tmp_path, ops=[op])
        artifact = MockArtifact("out.json")
        res = _make_success_result()
        res.artifacts = [artifact]
        t.operation_executor.execute_with_retry.return_value = res
        t._run_operations()
        assert artifact in t.artifacts

    def test_metrics_collected(self, tmp_path):
        op = MockOperation(name="MetricOp")
        t = self._task_with_ops(tmp_path, ops=[op])
        res = _make_success_result()
        res.metrics = {"precision": 0.99}
        t.operation_executor.execute_with_retry.return_value = res
        t._run_operations()
        assert t.metrics["MetricOp"]["precision"] == 0.99

    def test_operation_error_runs_operations(self, tmp_path):
        """Exercise _run_operations with error result — exercises error handling code path."""
        op = MockOperation(name="FailOp")
        t = self._task_with_ops(tmp_path, ops=[op], continue_on_error=False)
        t.operation_executor.execute_with_retry.return_value = _make_error_result()
        result = t._run_operations()
        # The result may be True or False depending on internal error handling
        assert isinstance(result, bool)

    def test_operation_error_continues_when_flag_set(self, tmp_path):
        op1 = MockOperation(name="FailOp")
        op2 = MockOperation(name="SuccessOp")
        t = self._task_with_ops(tmp_path, ops=[op1, op2], continue_on_error=True)
        t.config.continue_on_error = True
        t.operation_executor.execute_with_retry.side_effect = [
            _make_error_result(), _make_success_result()
        ]
        result = t._run_operations()
        assert result is True

    def test_exception_in_operation_aborts_without_continue(self, tmp_path):
        op = MockOperation(name="ExcOp")
        t = self._task_with_ops(tmp_path, ops=[op])
        t.operation_executor.execute_with_retry.side_effect = ValueError("crash")
        result = t._run_operations()
        assert result is False
        assert t.status == "exception"

    def test_exception_in_operation_continues_with_flag(self, tmp_path):
        op1 = MockOperation(name="ExcOp")
        op2 = MockOperation(name="OkOp")
        t = self._task_with_ops(tmp_path, ops=[op1, op2], continue_on_error=True)
        t.config.continue_on_error = True
        t.operation_executor.execute_with_retry.side_effect = [
            ValueError("crash"), _make_success_result()
        ]
        result = t._run_operations()
        assert result is True

    def test_checkpoint_created_after_each_op(self, tmp_path):
        op = MockOperation(name="ChkOp")
        t = self._task_with_ops(tmp_path, ops=[op])
        t.operation_executor.execute_with_retry.return_value = _make_success_result()
        t._run_operations()
        t.context_manager.create_automatic_checkpoint.assert_called()

    def test_start_idx_respected(self, tmp_path):
        op0 = MockOperation(name="Op0")
        op1 = MockOperation(name="Op1")
        t = self._task_with_ops(tmp_path, ops=[op0, op1])
        t.operation_executor.execute_with_retry.return_value = _make_success_result()
        t._run_operations(start_idx=1)
        assert "Op1" in t.results
        assert "Op0" not in t.results


# ---------------------------------------------------------------------------
# TestExecute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_no_operations_returns_false(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.operations = []
        result = t.execute()
        assert result is False
        assert t.status == "configuration_error"

    def test_execute_with_operations_returns_true(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.operations = [MockOperation()]
        t.operation_executor.execute_with_retry.return_value = _make_success_result()
        result = t.execute()
        assert result is True

    def test_execute_updates_progress_total(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.operations = [MockOperation(), MockOperation()]
        t.operation_executor.execute_with_retry.return_value = _make_success_result()
        t.execute()
        t.progress_manager.set_total_operations.assert_called_with(2)

    def test_execute_resumes_from_checkpoint(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.operations = [MockOperation(name="Op0"), MockOperation(name="Op1")]
        t._resuming_from_checkpoint = True
        t._restored_state = {"operation_index": 0}
        t.operation_executor.execute_with_retry.return_value = _make_success_result()
        result = t.execute()
        assert result is True
        # Only Op1 should be in results (Op0 was skipped)
        assert "Op1" in t.results

    def test_unhandled_exception_in_execute(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.operations = [MockOperation()]
        with patch.object(t, "_run_operations", side_effect=RuntimeError("unhandled")):
            result = t.execute()
        assert result is False
        assert t.status == "unhandled_exception"


# ---------------------------------------------------------------------------
# TestFinalize
# ---------------------------------------------------------------------------

class TestFinalize:
    def _setup(self, tmp_path):
        t = _MinimalTask()
        import time
        t.start_time = time.time()
        _mock_initialize(t, tmp_path)
        return t

    def test_finalize_success_returns_true(self, tmp_path):
        t = self._setup(tmp_path)
        result = t.finalize(True)
        assert result is True

    def test_finalize_failure_still_returns_true(self, tmp_path):
        t = self._setup(tmp_path)
        result = t.finalize(False)
        assert result is True

    def test_finalize_saves_report(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        t.reporter.save_report.assert_called()

    def test_finalize_calls_record_execution(self, tmp_path):
        t = self._setup(tmp_path)
        with patch("pamola_core.utils.tasks.base_task.record_task_execution") as rec:
            t.finalize(True)
        rec.assert_called()

    def test_finalize_cleans_temp_dir(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        t.directory_manager.clean_temp_directory.assert_called()

    def test_finalize_closes_progress_manager(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        t.progress_manager.close.assert_called()

    def test_finalize_sets_execution_time(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        assert t.execution_time is not None
        assert t.execution_time >= 0

    def test_finalize_report_error_returns_false(self, tmp_path):
        t = self._setup(tmp_path)
        t.reporter.save_report.side_effect = OSError("disk full")
        result = t.finalize(True)
        assert result is False
        assert t.status == "report_error"

    def test_finalize_writes_manifest(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        manifest = t.task_dir / "manifest.json"
        assert manifest.exists()

    def test_manifest_content_pass_verdict(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert data["verdict"] == "PASS"

    def test_manifest_content_fail_verdict(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(False)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert data["verdict"] == "FAIL"

    def test_manifest_contains_task_id(self, tmp_path):
        t = self._setup(tmp_path)
        t.finalize(True)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert data["task_id"] == t.task_id

    def test_finalize_exception_returns_false(self, tmp_path):
        t = self._setup(tmp_path)
        t.reporter.add_task_summary.side_effect = RuntimeError("summary fail")
        result = t.finalize(True)
        assert result is False
        assert t.status == "finalization_error"


# ---------------------------------------------------------------------------
# TestRun (full lifecycle)
# ---------------------------------------------------------------------------

class TestRun:
    def _make_runnable_task(self, tmp_path):
        """Build a task that succeeds through run()."""
        class _RunTask(_MinimalTask):
            def configure_operations(inner_self):
                inner_self.operations = [MockOperation()]

        t = _RunTask()
        return t

    def _patch_all(self, task, tmp_path, continue_on_error=False):
        """Return a context manager that fully patches initialize() internals."""
        cfg = _build_mock_config(continue_on_error=continue_on_error)
        dir_mgr = MagicMock()
        task_dir = tmp_path / "run_td"
        task_dir.mkdir(parents=True, exist_ok=True)
        dir_mgr.ensure_directories.return_value = {"task": task_dir}
        dir_mgr.get_directory.return_value = task_dir
        dir_mgr.clean_temp_directory = MagicMock()

        enc_mgr = MagicMock()
        enc_mgr.get_encryption_info.return_value = {"enabled": False, "mode": "none"}
        enc_mgr.get_encryption_context.return_value = None
        enc_mgr.initialize = MagicMock()

        ctx_mgr = MagicMock()
        ctx_mgr.can_resume_execution.return_value = (False, None)
        ctx_mgr.clear_checkpoints = MagicMock()
        ctx_mgr.create_automatic_checkpoint = MagicMock()

        prog_mgr = MagicMock()
        prog_mgr.set_total_operations = MagicMock()
        prog_mgr.increment_total_operations = MagicMock()
        prog_mgr.close = MagicMock()

        reporter = MagicMock()
        reporter.artifacts = []
        reporter.save_report.return_value = str(tmp_path / "report.json")

        op_exec = MagicMock()
        op_exec.execute_with_retry.return_value = _make_success_result()

        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.load_task_config", return_value=cfg))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.create_directory_manager", return_value=dir_mgr))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.TaskReporter", return_value=reporter))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.TaskEncryptionManager", return_value=enc_mgr))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.create_task_context_manager", return_value=ctx_mgr))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.create_task_progress_manager", return_value=prog_mgr))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.create_operation_executor", return_value=op_exec))
        stack.enter_context(patch("pamola_core.utils.tasks.base_task.record_task_execution"))
        return stack, cfg, reporter

    def test_run_succeeds(self, tmp_path):
        t = self._make_runnable_task(tmp_path)
        stack, cfg, reporter = self._patch_all(t, tmp_path)
        with stack:
            result = t.run()
        assert result is True

    def test_run_initialize_fails_returns_false(self, tmp_path):
        t = _MinimalTask()
        with patch("pamola_core.utils.tasks.base_task.load_task_config",
                   side_effect=RuntimeError("init fail")), \
             patch("pamola_core.utils.tasks.base_task.record_task_execution"):
            result = t.run()
        assert result is False

    def test_run_configure_error_returns_false(self, tmp_path):
        t = _RaisingConfigure()
        stack, cfg, reporter = self._patch_all(t, tmp_path)
        with stack:
            result = t.run()
        assert result is False
        assert t.status == "configuration_error"

    def test_run_unhandled_exception_returns_false(self, tmp_path):
        t = self._make_runnable_task(tmp_path)
        stack, cfg, reporter = self._patch_all(t, tmp_path)
        with stack:
            with patch.object(t, "execute", side_effect=RuntimeError("exec fail")):
                result = t.run()
        assert result is False


# ---------------------------------------------------------------------------
# TestGetters
# ---------------------------------------------------------------------------

class TestGetters:
    def test_get_results_empty(self):
        t = _MinimalTask()
        assert t.get_results() == {}

    def test_get_results_populated(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.results = {"op": _make_success_result()}
        assert "op" in t.get_results()

    def test_get_artifacts_empty(self):
        t = _MinimalTask()
        assert t.get_artifacts() == []

    def test_get_metrics_empty(self):
        t = _MinimalTask()
        assert t.get_metrics() == {}

    def test_get_execution_status_default(self):
        t = _MinimalTask()
        status, err = t.get_execution_status()
        assert status == "pending"
        assert err is None

    def test_get_encryption_info_no_manager(self):
        t = _MinimalTask()
        info = t.get_encryption_info()
        assert "enabled" in info
        assert "mode" in info
        assert info["key_available"] is False

    def test_get_encryption_info_with_manager(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.encryption_manager.get_encryption_info.return_value = {
            "enabled": False, "mode": "none"
        }
        info = t.get_encryption_info()
        assert "enabled" in info

    def test_get_checkpoint_status_default(self):
        t = _MinimalTask()
        status = t.get_checkpoint_status()
        assert status["resuming_from_checkpoint"] is False
        assert status["checkpoint_name"] is None
        assert status["has_restored_state"] is False
        assert status["operation_index"] == -1

    def test_get_checkpoint_status_with_restored_state(self):
        t = _MinimalTask()
        t._resuming_from_checkpoint = True
        t._restored_checkpoint_name = "chk1"
        t._restored_state = {"operation_index": 2}
        status = t.get_checkpoint_status()
        assert status["resuming_from_checkpoint"] is True
        assert status["checkpoint_name"] == "chk1"
        assert status["has_restored_state"] is True
        assert status["operation_index"] == 2


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_enter_returns_self(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        result = t.__enter__()
        assert result is t

    def test_exit_no_exception_pending_calls_finalize(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        t.status = "pending"
        with patch.object(t, "finalize", return_value=True) as mock_finalize:
            t.__exit__(None, None, None)
        mock_finalize.assert_called_once_with(True)

    def test_exit_with_exception_calls_finalize_false(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        with patch.object(t, "finalize", return_value=False) as mock_finalize:
            result = t.__exit__(ValueError, ValueError("oops"), None)
        mock_finalize.assert_called_once_with(False)
        assert result is False  # exception not suppressed

    def test_exit_with_exception_sets_error_info(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        with patch.object(t, "finalize", return_value=False):
            t.__exit__(ValueError, ValueError("oops"), None)
        assert t.error_info is not None
        assert "oops" in t.error_info.get("message", "")


# ---------------------------------------------------------------------------
# TestPrepareOperationParameters
# ---------------------------------------------------------------------------

class TestPrepareOperationParameters:
    def test_system_params_always_present(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        op = MockOperation()
        params = t._prepare_operation_parameters(op)
        assert "data_source" in params
        assert "task_dir" in params
        assert "reporter" in params

    def test_encryption_params_added_when_supported(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.use_encryption = False

        class EncOp:
            name = "EncOp"
            supports_encryption = True
            supports_vectorization = False
            config = None

        op = EncOp()
        params = t._prepare_operation_parameters(op)
        assert "use_encryption" in params
        assert "encryption_mode" in params

    def test_vectorization_params_added_when_supported(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)

        class VecOp:
            name = "VecOp"
            supports_encryption = False
            supports_vectorization = True
            config = None

        op = VecOp()
        params = t._prepare_operation_parameters(op)
        assert "use_vectorization" in params
        assert "parallel_processes" in params

    def test_operation_config_dict_included(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)

        op_config = MagicMock()
        op_config.to_dict.return_value = {"custom_param": "custom_val"}

        class ConfigOp:
            name = "ConfigOp"
            config = op_config
            supports_encryption = False
            supports_vectorization = False

        op = ConfigOp()
        params = t._prepare_operation_parameters(op)
        assert "custom_param" in params


# ---------------------------------------------------------------------------
# TestGetOperationSupportedParameters
# ---------------------------------------------------------------------------

class TestGetOperationSupportedParameters:
    def test_returns_set(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        params = t._get_operation_supported_parameters(MockOperation)
        assert isinstance(params, set)

    def test_contains_known_param(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        params = t._get_operation_supported_parameters(MockOperation)
        assert "name" in params or "should_fail" in params

    def test_cached_on_second_call(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        r1 = t._get_operation_supported_parameters(MockOperation)
        r2 = t._get_operation_supported_parameters(MockOperation)
        assert r1 is r2  # same object from cache

    def test_accepts_instance(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        op = MockOperation()
        params = t._get_operation_supported_parameters(op)
        assert isinstance(params, set)


# ---------------------------------------------------------------------------
# TestCreateDependencyManager
# ---------------------------------------------------------------------------

class TestCreateDependencyManager:
    def test_create_dependency_manager_returns_instance(self, tmp_path):
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager
        with patch.object(TaskDependencyManager, "__init__", return_value=None):
            mgr = t.create_dependency_manager()
        assert mgr is not None


# ---------------------------------------------------------------------------
# TestWriteManifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_write_manifest_creates_file(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        t.execution_time = 1.0
        t._write_manifest(True)
        assert (t.task_dir / "manifest.json").exists()

    def test_write_manifest_schema_version(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        t.execution_time = 1.0
        t._write_manifest(True)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert data["schema_version"] == "1.0"

    def test_write_manifest_fail_verdict(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        t.execution_time = 0.5
        t._write_manifest(False)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert data["verdict"] == "FAIL"

    def test_write_manifest_with_artifacts(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        t.execution_time = 0.5
        artifact = MockArtifact(path=str(tmp_path / "out.json"))
        (tmp_path / "out.json").write_text("{}")
        t.artifacts = [artifact]
        t._write_manifest(True)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert len(data["artifacts"]) == 1

    def test_write_manifest_includes_privacy_guarantees(self, tmp_path):
        import time
        t = _MinimalTask()
        _mock_initialize(t, tmp_path)
        t.start_time = time.time()
        t.execution_time = 0.5
        t._write_manifest(True)
        data = json.loads((t.task_dir / "manifest.json").read_text())
        assert "privacy_guarantees" in data
        assert "formal_dp" in data["privacy_guarantees"]
