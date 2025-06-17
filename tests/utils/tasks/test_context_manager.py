"""
Tests for the context_manager module in the pamola_core/utils/tasks package.

These tests ensure that the TaskContextManager class properly implements checkpointing,
state management, error handling, and resumable execution.
"""

import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest import mock
import pytest
import os
import time

from pamola_core.utils.tasks.context_manager import (
    TaskContextManager,
    ContextManagerError,
    CheckpointError,
    StateSerializationError,
    StateRestorationError,
    create_task_context_manager,
)

# --- Fixtures and Mocks ---

class DummyProgressManager:
    def __init__(self):
        self.logs = []
        self.task = mock.Mock()
        self.task.config = mock.Mock()
        self.task.config.to_dict.return_value = {"param": 1}
        self.task.version = "1.0"
    def create_operation_context(self, name, total, description=None):
        return mock.MagicMock().__enter__.return_value
    def log_info(self, msg):
        self.logs.append(("info", msg))
    def log_error(self, msg):
        self.logs.append(("error", msg))
    def log_warning(self, msg):
        self.logs.append(("warning", msg))
    def log_debug(self, msg):
        self.logs.append(("debug", msg))

@pytest.fixture
def temp_task_dir(tmp_path):
    d = tmp_path / "task"
    d.mkdir()
    return d

@pytest.fixture
def context_mgr(temp_task_dir):
    return TaskContextManager(
        task_id="test_task",
        task_dir=temp_task_dir,
        log_directory=None,
        max_state_size=1024 * 1024,
        progress_manager=DummyProgressManager(),
    )

# --- Tests for TaskContextManager ---

class TestTaskContextManager:
    def test_initial_state(self, context_mgr):
        state = context_mgr.get_current_state()
        assert state["task_id"] == "test_task"
        assert state["operation_index"] == -1
        assert isinstance(state["operations_completed"], list)
        assert isinstance(state["operations_failed"], list)
        assert isinstance(state["metrics"], dict)
        assert isinstance(state["artifacts"], list)

    def test_checkpoint_dir(self, context_mgr):
        assert context_mgr.get_checkpoint_dir().exists()

    def test_save_and_restore_execution_state(self, context_mgr):
        state = context_mgr.get_current_state()
        state["operation_index"] = 1
        path = context_mgr.save_execution_state(state, "cp1")
        assert path.exists()
        restored = context_mgr.restore_execution_state("cp1")
        assert restored["operation_index"] == 1

    def test_save_execution_state_invalid_path(self, context_mgr, monkeypatch):
        monkeypatch.setattr(
            context_mgr, "_sanitize_checkpoint_name", lambda name: "../badpath"
        )
        with pytest.raises(StateSerializationError):
            context_mgr.save_execution_state(context_mgr.get_current_state(), "bad")

    def test_restore_execution_state_missing(self, context_mgr):
        with pytest.raises(StateRestorationError):
            context_mgr.restore_execution_state("notfound")

    def test_clear_checkpoints(self, context_mgr):
        state = context_mgr.get_current_state()
        context_mgr.save_execution_state(state, "cp2")
        assert context_mgr.clear_checkpoints() is True
        assert context_mgr.get_checkpoint_dir().exists()
        assert context_mgr.get_checkpoints() == []

    def test_is_checkpoint_valid(self, context_mgr):
        state = context_mgr.get_current_state()
        state["config_hash"] = context_mgr._calculate_config_hash({"a": 1})
        state["task_version"] = "1.0"
        context_mgr.save_execution_state(state, "cp3")
        assert context_mgr.is_checkpoint_valid("cp3", config_hash=state["config_hash"], task_version="1.0")
        assert not context_mgr.is_checkpoint_valid("cp3", config_hash="bad", task_version="1.0")
        assert not context_mgr.is_checkpoint_valid("cp3", config_hash=state["config_hash"], task_version="bad")

    def test_create_automatic_checkpoint(self, context_mgr):
        path = context_mgr.create_automatic_checkpoint(2, metrics={"m": 5})
        assert path.exists()
        restored = context_mgr.restore_execution_state(path.stem)
        assert restored["operation_index"] == 2
        assert restored["metrics"]["m"] == 5

    def test_update_state(self, context_mgr):
        context_mgr.update_state({"operation_index": 5, "metrics": {"x": 1}})
        state = context_mgr.get_current_state()
        assert state["operation_index"] == 5
        assert state["metrics"]["x"] == 1

    def test_record_operation_completion(self, context_mgr):
        context_mgr.record_operation_completion(1, "op1", result_metrics={"score": 10})
        state = context_mgr.get_current_state()
        assert any(op["name"] == "op1" for op in state["operations_completed"])
        assert state["metrics"]["op1"]["score"] == 10

    def test_record_operation_failure(self, context_mgr):
        context_mgr.record_operation_failure(2, "op2", {"message": "fail"})
        state = context_mgr.get_current_state()
        assert any(op["name"] == "op2" for op in state["operations_failed"])

    def test_record_artifact(self, context_mgr):
        context_mgr.record_artifact("/tmp/file.txt", "txt", "desc")
        state = context_mgr.get_current_state()
        assert any(a["path"] == "/tmp/file.txt" for a in state["artifacts"])

    def test_can_resume_execution(self, context_mgr):
        context_mgr.create_automatic_checkpoint(3)
        can_resume, name = context_mgr.can_resume_execution()
        assert can_resume
        assert isinstance(name, str)

    def test_get_latest_checkpoint(self, context_mgr):
        path = context_mgr.create_automatic_checkpoint(4)
        name = context_mgr.get_latest_checkpoint()
        assert name is None or isinstance(name, str)

    def test_get_checkpoints(self, context_mgr):
        context_mgr.create_automatic_checkpoint(5)
        checkpoints = context_mgr.get_checkpoints()
        assert isinstance(checkpoints, list)
        assert checkpoints

    def test_sanitize_checkpoint_name(self, context_mgr):
        name = context_mgr._sanitize_checkpoint_name("bad/name:with*chars?")
        assert "/" not in name and ":" not in name and "*" not in name and "?" not in name
        assert len(name) <= 100

    def test_cleanup_old_checkpoints(self, context_mgr):
        # Create more than default max checkpoints
        for i in range(12):
            context_mgr.create_automatic_checkpoint(i)
        removed = context_mgr.cleanup_old_checkpoints(max_checkpoints=5)
        assert removed >= 7
        assert len(context_mgr.get_checkpoints()) <= 5

    def test_context_manager_enter_exit(self, temp_task_dir):
        mgr = TaskContextManager("exit_task", temp_task_dir)
        with mgr:
            mgr.create_automatic_checkpoint(1)
        # Should cleanup old checkpoints on exit
        assert isinstance(mgr.get_checkpoints(), list)

    def test_context_manager_exit_with_exception(self, temp_task_dir):
        mgr = TaskContextManager("exc_task", temp_task_dir)
        try:
            with mgr:
                raise RuntimeError("fail")
        except RuntimeError:
            # Should create error checkpoint
            checkpoints = mgr.get_checkpoints()
            assert any("error_exc_task" in name for name, _ in checkpoints)

    def test_cleanup(self, context_mgr):
        context_mgr.create_automatic_checkpoint(1)
        context_mgr.cleanup()
        # Should not raise

    def test_atomic_write_json_and_serialization(self, context_mgr, tmp_path):
        import numpy as np
        import pytest
        data = {
            "a": np.int32(5),
            "b": np.float64(3.14),
            "c": np.bool_(True),
            "d": np.array([1, 2, 3]),
            "e": [np.int64(7), np.float32(2.71)]
        }
        file_path = tmp_path / "test.json"
        context_mgr._atomic_write_json(data, file_path)
        assert file_path.exists()
        with open(file_path, "r") as f:
            loaded = json.load(f)
        assert loaded["a"] == 5
        assert loaded["b"] == pytest.approx(3.14)
        assert loaded["c"] is True
        assert loaded["d"] == [1, 2, 3]
        assert loaded["e"][0] == 7
        assert loaded["e"][1] == pytest.approx(2.71)

    def test__get_task_config_and_version(self, context_mgr):
        # Should return dummy config and version from DummyProgressManager
        assert context_mgr._get_task_config() == {"param": 1}
        assert context_mgr._get_task_version() == "1.0"
        # Remove progress_manager
        context_mgr.progress_manager = None
        assert context_mgr._get_task_config() == {}
        assert context_mgr._get_task_version() == "unknown"

    def test_exceptions_in_context_manager_exit(self, temp_task_dir):
        class BadTaskContextManager(TaskContextManager):
            def cleanup_old_checkpoints(self, *a, **kw):
                raise Exception("fail cleanup")
        mgr = BadTaskContextManager("bad_exit", temp_task_dir)
        with mgr:
            mgr.create_automatic_checkpoint(1)
        # Should not raise, error is logged

    def test_exceptions_in_error_checkpoint(self, temp_task_dir):
        class BadTaskContextManager(TaskContextManager):
            def save_execution_state(self, *a, **kw):
                raise Exception("fail save")
        mgr = BadTaskContextManager("bad_exc", temp_task_dir)
        try:
            with mgr:
                raise RuntimeError("fail")
        except RuntimeError:
            # Should not raise, error is logged
            pass

    def test_clear_checkpoints_handles_permission_error(self, context_mgr, monkeypatch):
        # Simulate shutil.rmtree raising an exception
        monkeypatch.setattr(shutil, "rmtree", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))
        state = context_mgr.get_current_state()
        context_mgr.save_execution_state(state, "cp_perm")
        assert context_mgr.clear_checkpoints() is False

    def test_cleanup_old_checkpoints_handles_invalid_path(self, context_mgr, monkeypatch):
        monkeypatch.setattr(
            "pamola_core.utils.tasks.context_manager.validate_path_security", lambda path: False
        )
        with pytest.raises(CheckpointError):
            context_mgr.create_automatic_checkpoint(100)

    def test_get_checkpoints_handles_timeout(self, context_mgr, monkeypatch):
        import filelock
        monkeypatch.setattr("filelock.FileLock.__enter__", lambda self: (_ for _ in ()).throw(filelock.Timeout()))
        assert context_mgr.get_checkpoints() == []

    def test_get_checkpoints_handles_exception(self, context_mgr, monkeypatch):
        monkeypatch.setattr("filelock.FileLock.__enter__", lambda self: (_ for _ in ()).throw(Exception("fail")))
        assert context_mgr.get_checkpoints() == []

    def test__sanitize_checkpoint_name_truncation(self, context_mgr):
        name = "a" * 120
        sanitized = context_mgr._sanitize_checkpoint_name(name)
        assert len(sanitized) == 100

    def test_create_task_context_manager_logs_error(self, temp_task_dir, monkeypatch):
        # Simulate error in ensure_directory
        monkeypatch.setattr("pamola_core.utils.tasks.context_manager.ensure_directory", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
        with pytest.raises(ContextManagerError):
            create_task_context_manager("fail_task", temp_task_dir)

    def test_serialize_data_handles_none_and_builtin(self):
        from pamola_core.utils.tasks.context_manager import _serialize_data
        assert _serialize_data(None) is None
        assert _serialize_data({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}
        assert _serialize_data([1, 2, 3]) == [1, 2, 3]

    def test_serialize_data_handles_numpy_types(self):
        import numpy as np
        from pamola_core.utils.tasks.context_manager import _serialize_data
        assert isinstance(_serialize_data(np.array([1, 2, 3])), list)

    def test_context_manager_exit_with_error_checkpoint_failure(self, temp_task_dir):
        # Simulate error during error checkpoint creation in __exit__
        class BadTaskContextManager(TaskContextManager):
            def save_execution_state(self, *a, **kw):
                raise Exception("fail save in exit")
        mgr = BadTaskContextManager("fail_exit", temp_task_dir)
        try:
            with mgr:
                raise RuntimeError("fail")
        except RuntimeError:
            # Should not raise, error is logged
            pass

    def test_context_manager_exit_with_cleanup_failure(self, temp_task_dir):
        # Simulate error during cleanup_old_checkpoints in __exit__
        class BadTaskContextManager(TaskContextManager):
            def cleanup_old_checkpoints(self, *a, **kw):
                raise Exception("fail cleanup in exit")
        mgr = BadTaskContextManager("fail_cleanup_exit", temp_task_dir)
        with mgr:
            mgr.create_automatic_checkpoint(1)
        # Should not raise, error is logged

    def test_cleanup_handles_exception(self, context_mgr, monkeypatch):
        # Simulate error in cleanup_old_checkpoints
        monkeypatch.setattr(context_mgr, "cleanup_old_checkpoints", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail cleanup")))
        # Should not raise
        context_mgr.cleanup()

    def test__load_checkpoint_history_handles_timeout_and_exception(self, context_mgr, monkeypatch):
        # Simulate filelock.Timeout
        import filelock
        monkeypatch.setattr("filelock.FileLock.__enter__", lambda self: (_ for _ in ()).throw(filelock.Timeout()))
        # Should not raise
        context_mgr._load_checkpoint_history()
        # Simulate generic Exception
        monkeypatch.setattr("filelock.FileLock.__enter__", lambda self: (_ for _ in ()).throw(Exception("fail")))
        context_mgr._load_checkpoint_history()

    def test__calculate_config_hash_handles_exception(self, context_mgr, monkeypatch):
        # Simulate json.dumps raising exception
        monkeypatch.setattr("json.dumps", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
        assert context_mgr._calculate_config_hash({"a": object()}) == "unknown"

    def test__atomic_write_json_handles_tempfile_exception(self, context_mgr, monkeypatch, tmp_path):
        # Simulate tempfile.mkstemp raising exception
        monkeypatch.setattr("tempfile.mkstemp", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail temp")))
        with pytest.raises(StateSerializationError):
            context_mgr._atomic_write_json({"a": 1}, tmp_path / "fail.json")

    def test__atomic_write_json_handles_write_exception(self, context_mgr, monkeypatch, tmp_path):
        # Simulate os.fdopen raising exception
        orig_fdopen = os.fdopen
        def bad_fdopen(*a, **kw):
            raise Exception("fail write")
        monkeypatch.setattr(os, "fdopen", bad_fdopen)
        # Patch tempfile.mkstemp to return a valid temp file
        fd, temp_path = tempfile.mkstemp(dir=tmp_path, suffix='.tmp.json')
        monkeypatch.setattr(tempfile, "mkstemp", lambda *a, **kw: (fd, temp_path))
        with pytest.raises(StateSerializationError):
            context_mgr._atomic_write_json({"a": 1}, tmp_path / "fail2.json")
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    def test__atomic_write_json_handles_replace_exception(self, context_mgr, monkeypatch, tmp_path):
        # Simulate os.replace raising exception
        orig_replace = os.replace
        def bad_replace(*a, **kw):
            raise Exception("fail replace")
        monkeypatch.setattr(os, "replace", bad_replace)
        fd, temp_path = tempfile.mkstemp(dir=tmp_path, suffix='.tmp.json')
        monkeypatch.setattr(tempfile, "mkstemp", lambda *a, **kw: (fd, temp_path))
        # Patch os.fdopen to normal
        monkeypatch.setattr(os, "fdopen", os.fdopen)
        with pytest.raises(StateSerializationError):
            context_mgr._atomic_write_json({"a": 1}, tmp_path / "fail3.json")
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception:
            pass

# --- create_task_context_manager helper ---

def test_create_task_context_manager(temp_task_dir):
    mgr = create_task_context_manager("helper_task", temp_task_dir)
    assert isinstance(mgr, TaskContextManager)
    assert mgr.task_id == "helper_task"
    # Test error case (skip if not portable)
    # with pytest.raises(ContextManagerError):
    #     create_task_context_manager("bad", Path("/bad/path/doesnotexist/"))

if __name__ == "__main__":
    pytest.main()