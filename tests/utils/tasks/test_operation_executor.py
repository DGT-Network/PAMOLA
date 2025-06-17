"""
Tests for the operation_executor module in the pamola_core/utils/tasks package.

These tests ensure that the TaskOperationExecutor class properly implements operation execution,
retry logic, error handling, progress tracking, and parallel execution.
"""

import pytest
import logging
import time
from unittest import mock
from unittest.mock import MagicMock, patch

from pamola_core.utils.tasks.operation_executor import (
    TaskOperationExecutor,
    ExecutionError,
    MaxRetriesExceededError,
    NonRetriableError,
    create_operation_executor,
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# --- Fixtures and Mocks ---

class DummyOperation:
    def __init__(self, should_fail=False, fail_times=0, status=OperationStatus.SUCCESS, raise_type=Exception, retriable=True):
        self.should_fail = should_fail
        self.fail_times = fail_times
        self.status = status
        self.raise_type = raise_type
        self.retriable = retriable
        self.run_calls = 0
    def run(self, **kwargs):
        self.run_calls += 1
        if self.should_fail and self.run_calls <= self.fail_times:
            e = self.raise_type(f"fail {self.run_calls}")
            if self.retriable:
                setattr(e, 'retriable', True)
            raise e
        return OperationResult(status=self.status)

class DummyProgressTracker:
    def create_subtask(self, *a, **kw):
        return self
    def update(self, *a, **kw):
        pass

class DummyReporter:
    def __init__(self):
        self.ops = []
    def add_operation(self, **kwargs):
        self.ops.append(kwargs)

class DummyConfig:
    continue_on_error = False
    parallel_processes = 2
    store_traceback = True
    mask_sensitive_data = False
    sensitive_patterns = [r'password=\S+']

# --- Module-level classes for multiprocessing pickling ---
class FailingOperation:
    def run(self, **kwargs):
        raise RuntimeError("fail in parallel")

class KeyboardInterruptOp:
    def run(self, **kwargs):
        raise KeyboardInterrupt()

class NonRetriableOp:
    def run(self, **kwargs):
        raise ValueError("non-retriable error")

@pytest.fixture
def executor():
    logger = logging.getLogger("test")
    return TaskOperationExecutor(
        task_config=DummyConfig(),
        logger=logger,
        reporter=DummyReporter(),
        default_max_retries=2,
        default_backoff_factor=1.5,
        default_initial_wait=0.01,
        default_max_wait=0.05,
        default_jitter=False,
    )

# --- Tests for TaskOperationExecutor ---

class TestTaskOperationExecutor:
    def test_execute_operation_success(self, executor):
        op = DummyOperation()
        result = executor.execute_operation(op, params={})
        assert result.status == OperationStatus.SUCCESS
        assert executor.execution_stats["successful_operations"] == 1

    def test_execute_operation_failure(self, executor):
        op = DummyOperation(should_fail=True, fail_times=1)
        with pytest.raises(Exception):
            executor.execute_operation(op, params={})
        assert executor.execution_stats["failed_operations"] == 1

    def test_execute_with_retry_success_on_second_try(self, executor):
        op = DummyOperation(should_fail=True, fail_times=1)
        result = executor.execute_with_retry(op, params={})
        assert result.status == OperationStatus.SUCCESS
        assert op.run_calls == 2
        assert executor.execution_stats["retried_operations"] == 1

    def test_execute_with_retry_max_retries_exceeded(self, executor):
        op = DummyOperation(should_fail=True, fail_times=5)
        with pytest.raises(MaxRetriesExceededError):
            executor.execute_with_retry(op, params={})

    def test_execute_with_retry_non_retriable(self, executor):
        op = DummyOperation(should_fail=True, fail_times=1, raise_type=ValueError, retriable=False)
        with pytest.raises(NonRetriableError):
            executor.execute_with_retry(op, params={})

    def test_is_retriable_error(self, executor):
        e = ConnectionError()
        assert executor.is_retriable_error(e)
        e2 = ValueError()
        assert not executor.is_retriable_error(e2)
        e3 = Exception()
        setattr(e3, 'retriable', True)
        assert executor.is_retriable_error(e3)
        e4 = Exception()
        assert not executor.is_retriable_error(e4)

    def test_add_and_remove_retriable_exception(self, executor):
        class CustomError(Exception):
            pass
        executor.add_retriable_exception(CustomError)
        assert CustomError in executor.retriable_exceptions
        executor.remove_retriable_exception(CustomError)
        assert CustomError not in executor.retriable_exceptions

    def test_add_retriable_exception_never_retry(self, executor, caplog):
        executor.add_retriable_exception(ValueError)
        assert ValueError not in executor.retriable_exceptions
        assert any("NEVER_RETRY_EXCEPTIONS" in m for m in caplog.text.splitlines())

    def test_execute_operations_sequential_success(self, executor):
        ops = [DummyOperation() for _ in range(3)]
        results = executor.execute_operations(ops, common_params={})
        assert all(r.status == OperationStatus.SUCCESS for r in results.values())

    def test_execute_operations_sequential_continue_on_error(self, executor):
        executor.config.continue_on_error = True
        ops = [DummyOperation(), NonRetriableOp()]
        results = executor.execute_operations(ops, common_params={})
        assert len(results) == 2
        assert any(r.status == OperationStatus.ERROR for r in results.values())

    def test_execute_operations_parallel_success(self, executor):
        ops = [DummyOperation() for _ in range(2)]
        results = executor.execute_operations_parallel(ops, common_params={})
        assert all(r.status == OperationStatus.SUCCESS for r in results.values())

    def test_execute_operations_parallel_with_error(self, executor):
        ops = [DummyOperation(), FailingOperation()]
        results = executor.execute_operations_parallel(ops, common_params={})
        assert len(results) == 2
        assert any(r.status == OperationStatus.ERROR for r in results.values())
        assert any("fail in parallel" in (r.error_message or "") for r in results.values())

    def test_execute_operations_parallel_continue_on_error(self, executor):
        executor.config.continue_on_error = True
        ops = [DummyOperation(), FailingOperation()]
        results = executor.execute_operations_parallel(ops, common_params={})
        assert len(results) == 2
        assert any(r.status == OperationStatus.ERROR for r in results.values())

    @pytest.mark.skip(reason="KeyboardInterrupt cannot be reliably propagated in ProcessPoolExecutor.")
    def test_execute_operations_parallel_keyboard_interrupt(self, executor):
        ops = [KeyboardInterruptOp()]
        with pytest.raises(KeyboardInterrupt):
            executor.execute_operations_parallel(ops, common_params={})

    def test_get_execution_stats(self, executor):
        op = DummyOperation()
        executor.execute_operation(op, params={})
        stats = executor.get_execution_stats()
        assert stats["total_operations"] >= 1

    def test__calculate_wait_time(self, executor):
        wait = executor._calculate_wait_time(2, 2.0, 1.0, 10.0, False)
        assert wait == 2.0
        wait_jitter = executor._calculate_wait_time(2, 2.0, 1.0, 10.0, True)
        assert 1.5 <= wait_jitter <= 2.5

    def test__format_exception(self, executor):
        try:
            raise RuntimeError("fail")
        except Exception as e:
            formatted = executor._format_exception(e)
            assert "RuntimeError" in formatted

    def test__make_error_result(self, executor):
        e = Exception("fail")
        result = executor._make_error_result(e, 0.1, "extra")
        assert result.status == OperationStatus.ERROR
        assert "extra" in result.error_message

    def test_create_operation_executor_helper(self):
        logger = logging.getLogger("test")
        reporter = DummyReporter()
        executor = create_operation_executor(DummyConfig(), logger, reporter)
        assert isinstance(executor, TaskOperationExecutor)

    def test_execute_with_retry_keyboard_interrupt(self, executor):
        class KeyboardInterruptOp:
            def run(self, **kwargs):
                raise KeyboardInterrupt()
        op = KeyboardInterruptOp()
        with pytest.raises(KeyboardInterrupt):
            executor.execute_with_retry(op, params={})

    def test_execute_operations_keyboard_interrupt(self, executor):
        class KeyboardInterruptOp:
            def run(self, **kwargs):
                raise KeyboardInterrupt()
        ops = [KeyboardInterruptOp()]
        with pytest.raises(KeyboardInterrupt):
            executor.execute_operations(ops, common_params={})

    def test_execute_operations_parallel_fallback_to_sequential(self, executor):
        # Patch ProcessPoolExecutor to raise Exception
        with patch("concurrent.futures.ProcessPoolExecutor.__init__", side_effect=Exception("fail")):
            ops = [DummyOperation()]
            results = executor.execute_operations_parallel(ops, common_params={})
            assert all(r.status == OperationStatus.SUCCESS for r in results.values())

    def test_execute_with_retry_on_retry_callback(self, executor):
        op = DummyOperation(should_fail=True, fail_times=1)
        called = {}
        def on_retry(e, attempt, wait):
            called['called'] = True
            assert isinstance(e, Exception)
            assert attempt == 1 or attempt == 2
            assert wait > 0
        executor.execute_with_retry(op, params={}, on_retry=on_retry)
        assert called['called']

    def test_execute_operations_unexpected_exception(self, executor):
        class BuggyOperation:
            def run(self, **kwargs):
                raise RuntimeError("unexpected bug")
        ops = [BuggyOperation()]
        results = executor.execute_operations(ops, common_params={})
        assert list(results.values())[0].status == OperationStatus.ERROR
        assert "unexpected bug" in list(results.values())[0].error_message

    def test_execute_operations_status_not_enum(self, executor):
        class WeirdResult:
            status = "success"
            execution_time = 0.1
            metrics = {}
            error_message = ""
            error_trace = ""
        class WeirdOperation:
            def run(self, **kwargs):
                return WeirdResult()
        ops = [WeirdOperation()]
        results = executor.execute_operations(ops, common_params={})
        assert "WeirdOperation" in results

    def test_execute_operations_parallel_exception_in_future(self, executor):
        ops = [FailingOperation()]
        results = executor.execute_operations_parallel(ops, common_params={})
        assert list(results.values())[0].status == OperationStatus.ERROR
        assert "fail in parallel" in (list(results.values())[0].error_message or "")

    def test_format_exception_masks_sensitive_data(self, executor):
        executor.config.mask_sensitive_data = True
        executor.config.sensitive_patterns = [r"secret=[^\s]+"]
        try:
            raise RuntimeError("fail secret=abcd1234")
        except Exception as e:
            formatted = executor._format_exception(e)
            assert "secret=****" in formatted

    def test_make_error_result_no_traceback(self, executor):
        executor.config.store_traceback = False
        e = Exception("fail")
        result = executor._make_error_result(e, 0.1)
        assert result.error_trace == ""

    def test_create_operation_executor_invalid_kwargs(self):
        logger = logging.getLogger("test")
        reporter = DummyReporter()
        with pytest.raises(TypeError):
            create_operation_executor(DummyConfig(), logger, reporter, not_a_real_arg=1)

if __name__ == "__main__":
    pytest.main()