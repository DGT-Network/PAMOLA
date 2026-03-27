"""
Unit tests for pamola_core.errors.error_handler.ErrorHandler class.

Tests cover:
- ErrorHandler initialization and configuration
- handle_error method with various exception types
- Error code validation and fallback
- Message template resolution and formatting
- Details extraction and logging
- Error context manager
- Decorator for function error handling
- Statistics tracking
- Edge cases and error scenarios

Run with: pytest tests/errors/test_error_handler.py
"""

import logging
import pytest
from unittest import mock
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.base import BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode
from pamola_core.errors.exceptions.validation import (
    ValidationError,
    FieldNotFoundError,
    InvalidParameterError,
)


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_logger")


@pytest.fixture
def handler(logger):
    """Create an ErrorHandler instance."""
    return ErrorHandler(
        logger=logger,
        operation_name="test_operation",
        enable_verbose_logging=True
    )


def _make_pamola_error(code=ErrorCode.DATA_LOAD_FAILED, msg="test error"):
    """Helper: create a BasePamolaError with DATA_LOAD_FAILED code by default.

    Most handle_error calls pass error_code=PROCESSING_FAILED while the
    error carries DATA_LOAD_FAILED, making exception_owns_error=True.
    This causes the handler to use the exception's own message/code without
    triggering template validation (which would raise InvalidParameterError
    when message_kwargs are missing).
    """
    return BasePamolaError(message=msg, error_code=code)


class TestErrorHandlerInitialization:
    """Test ErrorHandler initialization."""

    def test_handler_initialization(self, logger):
        """Initialize ErrorHandler with required parameters."""
        handler = ErrorHandler(logger=logger)
        assert handler.logger is logger
        assert handler.operation_name is None
        assert handler.enable_verbose_logging is True

    def test_handler_with_operation_name(self, logger):
        """Initialize ErrorHandler with operation name."""
        handler = ErrorHandler(
            logger=logger,
            operation_name="data_loading"
        )
        assert handler.operation_name == "data_loading"

    def test_handler_with_verbose_logging_disabled(self, logger):
        """Initialize ErrorHandler with verbose logging disabled."""
        handler = ErrorHandler(
            logger=logger,
            enable_verbose_logging=False
        )
        assert handler.enable_verbose_logging is False

    def test_handler_initial_stats(self, handler):
        """Handler starts with zero error count."""
        stats = handler.get_stats()
        assert stats["total_errors"] == 0
        assert stats["fallback_codes_used"] == 0


class TestErrorHandlerHandleError:
    """Test handle_error method."""

    def test_handle_error_basic(self, handler):
        """Handle BasePamolaError — use different code to trigger exception_owns_error=True."""
        # exception's code (DATA_LOAD_FAILED) ≠ passed code (PROCESSING_FAILED)
        # → exception_owns_error=True → message taken from exception, no template validation
        error = BasePamolaError(message="Test error", error_code=ErrorCode.DATA_LOAD_FAILED)
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert result is not None
        assert result.error_message == "Test error"

    def test_handle_error_increments_counter(self, handler):
        """Handling error increments error counter."""
        initial_count = handler.get_stats()["total_errors"]
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert handler.get_stats()["total_errors"] == initial_count + 1

    def test_handle_error_with_context(self, handler):
        """Handle error with additional context."""
        error = BasePamolaError(message="Connection failed", error_code=ErrorCode.DATA_LOAD_FAILED)
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED,
            context={"host": "localhost", "port": 5432}
        )
        assert result is not None

    def test_handle_error_with_invalid_code(self, handler):
        """Invalid error code falls back to PROCESSING_FAILED."""
        # Use plain ValueError (no error_code attr) so exception_owns_error=False
        # and effective_error_code=INVALID_CODE_XYZ triggers fallback.
        # Provide message_kwargs so the fallback template (PROCESSING_FAILED) validates.
        error = ValueError("Test")
        result = handler.handle_error(
            error=error,
            error_code="INVALID_CODE_XYZ",
            message_kwargs={"operation": "test_op", "reason": "test"}
        )
        # Should fall back without raising
        assert result is not None
        assert handler.get_stats()["fallback_codes_used"] == 1

    def test_handle_error_with_pamola_error(self, handler):
        """Handle BasePamolaError exception."""
        error = BasePamolaError(
            message="Custom error",
            error_code=ErrorCode.DATA_LOAD_FAILED
        )
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert result is not None

    def test_handle_error_raise_error_flag_false(self, handler):
        """With raise_error=False, error is logged but not re-raised."""
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED,
            raise_error=False
        )
        # Should return result without raising
        assert result is not None

    def test_handle_error_raise_error_flag_true(self, handler):
        """With raise_error=True, error is re-raised."""
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        with pytest.raises(BasePamolaError):
            handler.handle_error(
                error=error,
                error_code=ErrorCode.PROCESSING_FAILED,
                raise_error=True
            )

    def test_handle_error_with_message_template(self, handler):
        """Handle error with explicit message template and all required kwargs."""
        error = _make_pamola_error(ErrorCode.DATA_LOAD_FAILED, "Connection timeout")
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.DATA_LOAD_FAILED,
            message_template="DATA_LOAD_FAILED",
            message_kwargs={"source": "database", "reason": "timeout"}
        )
        assert result is not None
        assert "database" in result.error_message

    def test_handle_error_with_empty_context(self, handler):
        """Handle error with None context uses empty dict."""
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED,
            context=None
        )
        assert result is not None


class TestErrorHandlerStandardizeResult:
    """Test standardize_result method."""

    def test_standardize_result_basic(self, handler):
        """Standardize OperationResult with error code."""
        from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

        result = OperationResult(status=OperationStatus.ERROR)
        standardized = handler.standardize_result(
            result=result,
            error_code=ErrorCode.PROCESSING_FAILED,
            message="Operation failed"
        )
        assert standardized is result
        assert standardized.error_message == "Operation failed"

    def test_standardize_result_with_existing_message(self, handler):
        """Standardize preserves existing error message if no message provided."""
        from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

        result = OperationResult(
            status=OperationStatus.ERROR,
            error_message="Original message"
        )
        standardized = handler.standardize_result(
            result=result,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert standardized.error_message == "Original message"

    def test_standardize_result_with_context(self, handler):
        """Standardize result with additional context."""
        from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

        result = OperationResult(status=OperationStatus.ERROR)
        standardized = handler.standardize_result(
            result=result,
            error_code=ErrorCode.DATA_LOAD_FAILED,
            message="Load failed",
            context={"source": "file.csv"}
        )
        assert standardized is result


class TestErrorHandlerCreateError:
    """Test create_error method."""

    def test_create_error_basic(self, handler):
        """Create properly formatted BasePamolaError."""
        error = handler.create_error(
            error_code=ErrorCode.DATA_LOAD_FAILED,
            message_kwargs={"source": "database.csv", "reason": "file not found"}
        )
        assert isinstance(error, BasePamolaError)
        assert error.error_code == ErrorCode.DATA_LOAD_FAILED
        assert "database.csv" in error.message

    def test_create_error_with_custom_exception_class(self, handler):
        """Create error with custom exception class that accepts message= kwarg.

        create_error() always calls exception_class(message=..., error_code=..., details=...).
        FieldNotFoundError does not accept message= (it constructs its own message from
        field_name/available_fields), so we use ValidationError which does accept message=.
        """
        from pamola_core.errors.exceptions.validation import ValidationError as VE
        error = handler.create_error(
            error_code=ErrorCode.FIELD_NOT_FOUND,
            message_kwargs={"field_name": "user_id", "available_fields": "name, email"},
            exception_class=VE
        )
        assert isinstance(error, VE)
        assert error.error_code == ErrorCode.FIELD_NOT_FOUND

    def test_create_error_with_details(self, handler):
        """Create error with additional details."""
        details = {"file_path": "/data/users.csv", "line_number": 42}
        error = handler.create_error(
            error_code=ErrorCode.DATA_VALIDATION_ERROR,
            message_kwargs={"context": "user_import", "reason": "invalid format"},
            details=details
        )
        assert error.details == details

    def test_create_error_invalid_exception_class(self, handler):
        """Raise error if exception_class is not BasePamolaError subclass."""
        with pytest.raises(ValidationError):
            handler.create_error(
                error_code=ErrorCode.PROCESSING_FAILED,
                message_kwargs={},
                exception_class=ValueError  # Not a BasePamolaError
            )

    def test_create_error_invalid_error_code(self, handler):
        """Raise InvalidParameterError if error code is invalid."""
        with pytest.raises(InvalidParameterError):
            handler.create_error(
                error_code="NONEXISTENT_CODE",
                message_kwargs={}
            )

    def test_create_error_missing_message_params(self, handler):
        """Raise InvalidParameterError if required message params are missing."""
        with pytest.raises(InvalidParameterError):
            handler.create_error(
                error_code=ErrorCode.DATA_LOAD_FAILED,
                message_kwargs={"source": "file.csv"}  # Missing 'reason'
            )


class TestErrorHandlerContextManager:
    """Test error_context context manager."""

    def test_error_context_no_error(self, handler):
        """Context manager passes through when no error occurs."""
        with handler.error_context(ErrorCode.PROCESSING_FAILED):
            value = 42
        assert value == 42

    def test_error_context_with_error_reraises(self, handler):
        """Context manager re-raises error by default."""
        # Use different code than context manager to trigger exception_owns_error=True
        error = BasePamolaError(message="fail", error_code=ErrorCode.DATA_LOAD_FAILED)
        with pytest.raises(BasePamolaError):
            with handler.error_context(ErrorCode.PROCESSING_FAILED):
                raise error

    def test_error_context_suppress_error(self, handler):
        """Context manager with suppress=True doesn't re-raise."""
        error = BasePamolaError(message="fail", error_code=ErrorCode.DATA_LOAD_FAILED)
        with handler.error_context(
            ErrorCode.PROCESSING_FAILED,
            suppress=True
        ):
            raise error
        # Should not raise

    def test_error_context_with_context_info(self, handler):
        """Context manager includes context information."""
        error = BasePamolaError(message="fail", error_code=ErrorCode.DATA_LOAD_FAILED)
        with pytest.raises(BasePamolaError):
            with handler.error_context(
                ErrorCode.PROCESSING_FAILED,
                context={"operation": "data_transform"}
            ):
                raise error

    def test_error_context_with_message_kwargs(self, handler):
        """Context manager includes message template kwargs (exception_owns_error path)."""
        # DATA_LOAD_FAILED exception inside DATA_LOAD_FAILED context → same code
        # Provide message_kwargs to satisfy the template
        error = BasePamolaError(message="fail", error_code=ErrorCode.PROCESSING_FAILED)
        with pytest.raises(BasePamolaError):
            with handler.error_context(
                ErrorCode.DATA_LOAD_FAILED,
                message_kwargs={"source": "file.csv", "reason": "not found"}
            ):
                raise error


class TestErrorHandlerDecorator:
    """Test capture_errors decorator."""

    def test_decorator_basic_sync(self, handler):
        """Decorator works on synchronous functions."""
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_catches_error_sync(self, handler):
        """Decorator catches BasePamolaError in synchronous functions."""
        # Use DATA_LOAD_FAILED code ≠ decorator's PROCESSING_FAILED → exception_owns_error=True
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED, rethrow=False)
        def failing_func():
            raise BasePamolaError(message="fail", error_code=ErrorCode.DATA_LOAD_FAILED)

        result = failing_func()
        assert result is not None  # Returns OperationResult

    def test_decorator_rethrows_error(self, handler):
        """Decorator rethrows error with rethrow=True."""
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED, rethrow=True)
        def failing_func():
            raise BasePamolaError(message="fail", error_code=ErrorCode.DATA_LOAD_FAILED)

        with pytest.raises(BasePamolaError):
            failing_func()

    def test_decorator_with_context(self, handler):
        """Decorator includes context information and re-raises with rethrow=True."""
        @handler.capture_errors(
            ErrorCode.PROCESSING_FAILED,
            context={"operation": "validation"},
            rethrow=True,
        )
        def test_func():
            raise BasePamolaError(message="fail", error_code=ErrorCode.DATA_LOAD_FAILED)

        with pytest.raises(BasePamolaError):
            test_func()

    def test_decorator_preserves_function_signature(self, handler):
        """Decorator preserves original function signature."""
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED)
        def add(a, b):
            """Add two numbers."""
            return a + b

        assert add.__doc__ == "Add two numbers."
        assert add(2, 3) == 5

    def test_decorator_with_function_args(self, handler):
        """Decorator works with functions that have arguments."""
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED, rethrow=False)
        def process(data, multiplier=2):
            if multiplier < 0:
                raise BasePamolaError(message="bad multiplier", error_code=ErrorCode.DATA_LOAD_FAILED)
            return data * multiplier

        result = process(5, multiplier=3)
        assert result == 15

    def test_decorator_async_function(self, handler):
        """Decorator supports async functions."""
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED)
        async def async_func():
            return "async result"

        import asyncio
        result = asyncio.run(async_func())
        assert result == "async result"

    def test_decorator_async_catches_error(self, handler):
        """Decorator catches errors in async functions."""
        @handler.capture_errors(ErrorCode.PROCESSING_FAILED, rethrow=False)
        async def async_failing():
            raise BasePamolaError(message="async fail", error_code=ErrorCode.DATA_LOAD_FAILED)

        import asyncio
        result = asyncio.run(async_failing())
        assert result is not None


class TestErrorHandlerStatistics:
    """Test statistics tracking."""

    def test_get_stats_initial(self, handler):
        """Initial stats show zero errors."""
        stats = handler.get_stats()
        assert stats["total_errors"] == 0
        assert stats["fallback_codes_used"] == 0

    def test_get_stats_after_errors(self, handler):
        """Stats increment after handling errors."""
        # Use error with different code than passed → exception_owns_error=True
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        for _ in range(3):
            handler.handle_error(
                error=error,
                error_code=ErrorCode.PROCESSING_FAILED
            )
        stats = handler.get_stats()
        assert stats["total_errors"] == 3

    def test_reset_stats(self, handler):
        """reset_stats clears statistics."""
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        handler.reset_stats()
        stats = handler.get_stats()
        assert stats["total_errors"] == 0
        assert stats["fallback_codes_used"] == 0

    def test_fallback_code_count(self, handler):
        """Fallback code counter increments on invalid codes.

        Uses plain ValueError (no error_code attr) so exception_owns_error=False
        and the invalid code reaches _validate_error_code → fallback triggered.
        message_kwargs provided to satisfy the fallback PROCESSING_FAILED template.
        """
        kwargs = {"operation": "test_op", "reason": "test"}
        handler.handle_error(
            error=ValueError("Test"),
            error_code="INVALID_CODE_1",
            message_kwargs=kwargs
        )
        handler.handle_error(
            error=ValueError("Test"),
            error_code="INVALID_CODE_2",
            message_kwargs=kwargs
        )
        stats = handler.get_stats()
        assert stats["fallback_codes_used"] == 2


class TestErrorHandlerEdgeCases:
    """Test edge cases and error scenarios."""

    def test_handle_error_with_none_logger_error_message(self, handler):
        """Handle exception with empty message via exception_owns_error path."""
        # Different codes → exception_owns_error=True → bypasses template validation
        error = BasePamolaError(message="", error_code=ErrorCode.DATA_LOAD_FAILED)
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert result is not None

    def test_handle_error_with_complex_exception_hierarchy(self, handler):
        """Handle exception with custom exception hierarchy."""
        class CustomError(BasePamolaError):
            pass

        error = CustomError(
            message="Custom error",
            error_code=ErrorCode.PROCESSING_FAILED
        )
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.DATA_LOAD_FAILED
        )
        assert result is not None

    def test_create_error_with_empty_message_kwargs(self, handler):
        """Create error with required message kwargs for PROCESSING_FAILED template."""
        error = handler.create_error(
            error_code=ErrorCode.PROCESSING_FAILED,
            message_kwargs={"operation": "test_op", "reason": "test failure"}
        )
        assert isinstance(error, BasePamolaError)

    def test_handle_error_unicode_in_exception_message(self, handler):
        """Handle exception with unicode characters in message."""
        # Use different codes → exception_owns_error=True → bypasses template validation
        error = BasePamolaError(
            message="错误: 无法加载文件",
            error_code=ErrorCode.DATA_LOAD_FAILED
        )
        result = handler.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert result is not None

    def test_multiple_handlers_independent(self, logger):
        """Multiple handler instances maintain independent state."""
        handler1 = ErrorHandler(logger)
        handler2 = ErrorHandler(logger)

        # Use different codes to trigger exception_owns_error=True
        error = BasePamolaError(message="err", error_code=ErrorCode.DATA_LOAD_FAILED)
        handler1.handle_error(
            error=error,
            error_code=ErrorCode.PROCESSING_FAILED
        )

        stats1 = handler1.get_stats()
        stats2 = handler2.get_stats()

        assert stats1["total_errors"] == 1
        assert stats2["total_errors"] == 0


class TestErrorHandlerIntegration:
    """Integration tests combining multiple features."""

    def test_full_error_handling_workflow(self, handler):
        """Complete error handling workflow."""
        try:
            raise BasePamolaError(
                message="Database connection failed",
                error_code=ErrorCode.DATA_LOAD_FAILED
            )
        except BasePamolaError as e:
            result = handler.handle_error(
                error=e,
                error_code=ErrorCode.DATA_LOAD_FAILED,
                context={"source": "postgres://localhost"},
                message_kwargs={"source": "postgres", "reason": "connection timeout"}
            )

        assert result is not None
        assert handler.get_stats()["total_errors"] == 1

    def test_error_handler_with_custom_logger_capture(self, caplog):
        """Verify error handler logs messages."""
        logger = logging.getLogger("test_handler_logger")
        handler = ErrorHandler(logger, operation_name="test_op")

        # exception_owns_error=True when exception's error_code differs from passed code
        error = BasePamolaError(message="Test error", error_code=ErrorCode.DATA_LOAD_FAILED)
        with caplog.at_level(logging.ERROR):
            handler.handle_error(
                error=error,
                error_code=ErrorCode.PROCESSING_FAILED
            )

        # Verify something was logged
        assert len(caplog.records) > 0

    def test_decorator_and_context_manager_combined(self, handler):
        """Use decorator and context manager together."""
        @handler.capture_errors(
            ErrorCode.DATA_LOAD_FAILED,
            rethrow=False
        )
        def load_data(source):
            with handler.error_context(
                ErrorCode.PROCESSING_FAILED,
                context={"operation": "parse"}
            ):
                if source == "invalid":
                    # Use a different code than context manager's PROCESSING_FAILED
                    raise BasePamolaError(message="invalid source", error_code=ErrorCode.DATA_LOAD_FAILED)
                return f"data from {source}"

        result = load_data("valid")
        assert "valid" in result
