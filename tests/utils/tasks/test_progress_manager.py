"""
Tests for the progress_manager module in the pamola_core/utils/tasks package.

These tests ensure that the ProgressTracker, NoOpProgressTracker, TaskProgressManager, and ProgressContext classes
properly implement progress tracking, logging, metrics, and error handling.
"""

import logging
import sys
import time
import types
import pytest
from unittest import mock
from pamola_core.utils.tasks import progress_manager

# --- Fixtures and Mocks ---

class DummyLogger:
    def __init__(self):
        self.records = []
        self.handlers = []
    def log(self, level, msg):
        self.records.append((level, msg))
    def debug(self, msg):
        self.records.append((logging.DEBUG, msg))
    def info(self, msg):
        self.records.append((logging.INFO, msg))
    def warning(self, msg):
        self.records.append((logging.WARNING, msg))
    def error(self, msg):
        self.records.append((logging.ERROR, msg))
    def critical(self, msg):
        self.records.append((logging.CRITICAL, msg))

class DummyReporter:
    def __init__(self):
        self.calls = []
    def add_operation(self, name, status, details):
        self.calls.append((name, status, details))

@pytest.fixture
def dummy_logger():
    logger = DummyLogger()
    # Add a dummy handler to test _check_logger_handlers
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.handlers.append(handler)
    return logger

@pytest.fixture
def dummy_reporter():
    return DummyReporter()

# --- Tests for NoOpProgressTracker ---

def test_noop_progress_tracker_basic():
    tracker = progress_manager.NoOpProgressTracker(
        total=10, description="desc", unit="items", position=0, leave=True, parent=None, color=None
    )
    tracker.update(steps=2)
    tracker.set_description("new desc")
    tracker.set_postfix({"foo": 1})
    tracker.close()
    assert tracker.metrics["current"] == 2
    assert "execution_time" in tracker.metrics
    assert "peak_memory_mb" in tracker.metrics
    assert tracker.description == "new desc"


def test_noop_progress_tracker_context_manager():
    tracker = progress_manager.NoOpProgressTracker(1, "desc")
    with tracker as t:
        t.update(1)
    assert tracker.metrics["current"] == 1
    assert "execution_time" in tracker.metrics

# --- Tests for ProgressTracker ---

def test_progress_tracker_basic(monkeypatch):
    # Patch tqdm to avoid actual output
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    tracker = progress_manager.ProgressTracker(
        total=5, description="desc", unit="items", position=0, leave=True, parent=None, color=None, disable=False
    )
    tracker.update(steps=2, postfix={"foo": 1})
    tracker.set_description("desc2")
    tracker.set_postfix({"bar": 2})
    tracker.clear()
    tracker.refresh()
    tracker.close()
    assert tracker.description == "desc2"
    assert hasattr(tracker, "metrics")

def test_progress_tracker_context_manager(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    tracker = progress_manager.ProgressTracker(2, "desc")
    with tracker as t:
        t.update(1)
    assert hasattr(tracker, "metrics")
    
def test_progress_tracker_close_failure_color(monkeypatch):
    class DummyPbar:
        def __init__(self):
            self.n = 1
            self.colour = None
            self.bar_format = None
            self.closed = False
        def update(self, steps): pass
        def set_postfix(self, **kwargs): pass
        def set_description(self, desc): pass
        def close(self): self.closed = True
        def clear(self): pass
        def refresh(self): pass
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    tracker = progress_manager.ProgressTracker(1, "desc")
    tracker.pbar = DummyPbar()
    tracker.close(failed=True)
    assert tracker.pbar.colour == "red" or tracker.pbar.bar_format is not None
    
def test_progress_tracker_close_color_exception(monkeypatch):
    class DummyPbar:
        def __init__(self):
            self.n = 1
        def update(self, steps): pass
        def set_postfix(self, **kwargs): pass
        def set_description(self, desc): pass
        def close(self): pass
        def clear(self): pass
        def refresh(self): pass
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    tracker = progress_manager.ProgressTracker(1, "desc")
    tracker.pbar = DummyPbar()
    # Patch setattr to raise
    monkeypatch.setattr(DummyPbar, "__setattr__", lambda self, name, value: (_ for _ in ()).throw(Exception("fail")))
    # Should not raise
    tracker.close(failed=True)

# --- Tests for TaskProgressManager ---

def test_task_progress_manager_basic(dummy_logger, dummy_reporter, monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    mgr = progress_manager.TaskProgressManager(
        task_id="tid", task_type="ttype", logger=dummy_logger, reporter=dummy_reporter, total_operations=2, quiet=False
    )
    mgr.set_total_operations(3)
    mgr.increment_total_operations(2)
    op = mgr.start_operation("op1", 2, description="desc")
    mgr.update_operation("op1", 1, postfix={"foo": 1})
    mgr.complete_operation("op1", success=True, metrics={"m": 1})
    mgr.log_info("info")
    mgr.log_warning("warn")
    mgr.log_error("err")
    mgr.log_debug("dbg")
    mgr.log_critical("crit", preserve_progress=True)
    ctx = mgr.create_operation_context("op2", 1)
    assert isinstance(ctx, progress_manager.ProgressContext)
    metrics = mgr.get_metrics()
    assert metrics["task_id"] == "tid"
    mgr.close()


def test_task_progress_manager_quiet_mode(dummy_logger):
    mgr = progress_manager.TaskProgressManager(
        task_id="tid2", task_type="ttype2", logger=dummy_logger, reporter=None, total_operations=0, quiet=True
    )
    op = mgr.start_operation("op2", 0)
    assert isinstance(op, progress_manager.NoOpProgressTracker)
    op = mgr.start_operation("op3", 2)
    assert isinstance(op, progress_manager.NoOpProgressTracker)
    mgr.complete_operation("op3", success=False)
    mgr.close()


def test_task_progress_manager_logger_stdout_warning():
    logger = DummyLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.handlers.append(handler)
    mgr = progress_manager.TaskProgressManager(
        task_id="tid3", task_type="ttype3", logger=logger, reporter=None, total_operations=1, quiet=False
    )
    # Should log a warning about stdout
    assert any(logging.WARNING == rec[0] for rec in logger.records)
    mgr.close()


def test_task_progress_manager_edge_cases(dummy_logger, monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    mgr = progress_manager.TaskProgressManager(
        task_id="tid4", task_type="ttype4", logger=dummy_logger, reporter=None, total_operations=0, quiet=False
    )
    # update_operation with missing op
    mgr.update_operation("notfound", 1)
    # complete_operation with missing op
    mgr.complete_operation("notfound", success=True)
    # log_message with preserve_progress False
    mgr.log_message(logging.INFO, "msg", preserve_progress=False)
    mgr.close()


def test_task_progress_manager_set_total_operations_exception(dummy_logger, monkeypatch):
    mgr = progress_manager.TaskProgressManager("tid", "ttype", dummy_logger, None, 1, False)
    mgr.main_progress = mock.Mock()
    type(mgr.main_progress).pbar = mock.PropertyMock(return_value=mock.Mock())
    def raise_exc(*a, **k): raise Exception("fail")
    mgr.main_progress.pbar.total = 1
    monkeypatch.setattr(mgr.main_progress.pbar, "refresh", raise_exc)
    mgr.set_total_operations(2)  # Should log a warning, not raise


def test_task_progress_manager_increment_total_operations_exception(dummy_logger, monkeypatch):
    mgr = progress_manager.TaskProgressManager("tid", "ttype", dummy_logger, None, 1, False)
    mgr.main_progress = mock.Mock()
    type(mgr.main_progress).pbar = mock.PropertyMock(return_value=mock.Mock())
    def raise_exc(*a, **k): raise Exception("fail")
    mgr.main_progress.pbar.total = 1
    monkeypatch.setattr(mgr.main_progress.pbar, "refresh", raise_exc)
    mgr.increment_total_operations(1)  # Should log a warning, not raise


def test_task_progress_manager_complete_operation_update_main_progress_exception(dummy_logger, monkeypatch):
    mgr = progress_manager.TaskProgressManager("tid", "ttype", dummy_logger, None, 1, False)
    mgr.main_progress = mock.Mock()
    def raise_exc(*a, **k): raise Exception("fail")
    mgr.main_progress.update = raise_exc
    mgr.active_operations["op"] = mock.Mock(metrics={})
    mgr.complete_operation("op", success=True)  # Should log debug, not raise


def test_task_progress_manager_close_exception(dummy_logger):
    mgr = progress_manager.TaskProgressManager("tid", "ttype", dummy_logger, None, 1, False)
    bad_progress = mock.Mock()
    bad_progress.close.side_effect = Exception("fail")
    mgr.active_operations["bad"] = bad_progress
    mgr.main_progress = mock.Mock()
    mgr.main_progress.close.side_effect = Exception("fail")
    mgr.close()  # Should log debug, not raise


def test_task_progress_manager_log_message_stderr(dummy_logger, capsys):
    mgr = progress_manager.TaskProgressManager("tid", "ttype", dummy_logger, None, 1, True)
    mgr.main_progress = None
    mgr.log_message(logging.WARNING, "warn to stderr")
    captured = capsys.readouterr()
    assert "warn to stderr" in captured.err

# --- Tests for ProgressContext ---

def test_progress_context_normal(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    mgr = progress_manager.TaskProgressManager(
        task_id="tid5", task_type="ttype5", logger=DummyLogger(), reporter=None, total_operations=1, quiet=False
    )
    with progress_manager.ProgressContext(mgr, "op", 1) as tracker:
        tracker.update(1)
    # Should call complete_operation
    assert mgr.operations_completed == 1


def test_progress_context_empty(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    mgr = progress_manager.TaskProgressManager(
        task_id="tid6", task_type="ttype6", logger=DummyLogger(), reporter=None, total_operations=1, quiet=False
    )
    with progress_manager.ProgressContext(mgr, "op", 0) as tracker:
        tracker.update(1)
    # Should not increment operations_completed
    assert mgr.operations_completed == 0


def test_progress_context_exception(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    mgr = progress_manager.TaskProgressManager(
        task_id="tid7", task_type="ttype7", logger=DummyLogger(), reporter=None, total_operations=1, quiet=False
    )
    try:
        with progress_manager.ProgressContext(mgr, "op", 1) as tracker:
            raise ValueError("fail")
    except ValueError:
        pass
    # Should call complete_operation with success=False
    assert mgr.operations_completed == 1


def test_progress_context_exit_complete_operation_exception(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    class BadManager(progress_manager.TaskProgressManager):
        def complete_operation(self, *a, **k): raise Exception("fail")
    mgr = BadManager("tid", "ttype", DummyLogger(), None, 1, False)
    ctx = progress_manager.ProgressContext(mgr, "op", 1)
    ctx.tracker = mock.Mock(metrics={})
    ctx.empty_operation = False
    ctx.__exit__(None, None, None)  # Should log error, not raise


def test_progress_context_exit_tracker_close_exception(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    class BadLogger(DummyLogger):
        def error(self, msg): self.records.append(("error", msg))
    mgr = progress_manager.TaskProgressManager("tid", "ttype", BadLogger(), None, 1, False)
    ctx = progress_manager.ProgressContext(mgr, "op", 0)
    ctx.tracker = mock.Mock()
    ctx.tracker.close.side_effect = Exception("fail")
    ctx.empty_operation = True
    ctx.__exit__(None, None, None)  # Should log error, not raise

# --- Tests for create_task_progress_manager ---

def test_create_task_progress_manager(monkeypatch, dummy_logger):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    mgr = progress_manager.create_task_progress_manager(
        task_id="tid8", task_type="ttype8", logger=dummy_logger, reporter=None, total_operations=1, quiet=None
    )
    assert isinstance(mgr, progress_manager.TaskProgressManager)


def test_create_task_progress_manager_quiet_env(monkeypatch, dummy_logger):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    mgr = progress_manager.create_task_progress_manager(
        task_id="tid9", task_type="ttype9", logger=dummy_logger, reporter=None, total_operations=1, quiet=None
    )
    assert mgr.quiet is True


def test_create_task_progress_manager_ci_env(monkeypatch, dummy_logger):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(progress_manager.os, "environ", {"CI": "true"})
    mgr = progress_manager.create_task_progress_manager(
        task_id="tid10", task_type="ttype10", logger=dummy_logger, reporter=None, total_operations=1, quiet=None
    )
    assert mgr.quiet is True

# --- Invalid input and error cases ---

def test_progress_tracker_invalid_inputs(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    tracker = progress_manager.ProgressTracker(
        total=0, description="desc", unit="items", position=0, leave=True, parent=None, color=None, disable=False
    )
    tracker.update(steps=0, postfix=None)
    tracker.close()
    # Should not raise


def test_noop_progress_tracker_invalid_inputs():
    tracker = progress_manager.NoOpProgressTracker(
        total=0, description="desc", unit="items", position=0, leave=True, parent=None, color=None
    )
    tracker.update(steps=0, postfix=None)
    tracker.close()
    # Should not raise


def test_task_progress_manager_invalid_inputs(dummy_logger):
    mgr = progress_manager.TaskProgressManager(
        task_id="tid11", task_type="ttype11", logger=dummy_logger, reporter=None, total_operations=0, quiet=True
    )
    mgr.set_total_operations(0)
    mgr.increment_total_operations(0)
    mgr.close()


def test_noop_progress_tracker_parent_update_exception():
    class BadParent:
        def update(self, steps): raise Exception("fail")
    tracker = progress_manager.NoOpProgressTracker(1, "desc", parent=BadParent())
    tracker.close()  # Should not raise


def test_progress_tracker_parent_update_exception(monkeypatch):
    monkeypatch.setattr(progress_manager, "tqdm", mock.MagicMock())
    class BadParent:
        def update(self, steps): raise Exception("fail")
    tracker = progress_manager.ProgressTracker(1, "desc", parent=BadParent())
    tracker.close()  # Should not raise   
    
if __name__ == "__main__":
    pytest.main()
