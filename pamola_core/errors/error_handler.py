# pamola_core/errors/error_handler.py
"""
Centralized error handling with structured logging and standardized messages.

This module ties error codes, messages, context suggestions, and OperationResult
into a single place to keep error reporting consistent across pamola_core.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
    Dict,
    Optional,
    Callable,
    TypeVar,
    cast,
    TYPE_CHECKING,
)
from functools import wraps
import logging
import asyncio
import traceback
from contextlib import contextmanager

from pamola_core.errors.base import BasePamolaError
from pamola_core.errors.context.suggestions import ErrorContext
from pamola_core.errors.codes.metadata import get_error_metadata
from pamola_core.errors.codes.registry import ErrorCode
from pamola_core.errors.codes.utils import validate_error_code_usage
from pamola_core.errors.messages.registry import ErrorMessages
from pamola_core.errors.exceptions.validation import (
    InvalidParameterError,
    ValidationError,
)

# Replace top-level import with TYPE_CHECKING guard
# → Imports only run when the type-checker (mypy/pyright) parses them; they do NOT run at runtime.
# → Break cycle: errors → utilities
if TYPE_CHECKING:
    from pamola_core.utils.ops.op_result import OperationResult

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorHandler:
    """
    Centralized error handling with structured logging.

    Features:
    - Validates error codes and message templates
    - Provides standardized technical messages
    - Adds recovery suggestions
    - Creates structured OperationResult
    - Logs with full context
    - Supports both sync and async functions

    Examples
    --------
        >>> handler = ErrorHandler(logger, operation_name="data_loading")
        >>> try:
        ...     load_data()
        ... except Exception as e:
        ...     result = handler.handle_error(
        ...         error=e,
        ...         error_code=ErrorCode.DATA_LOAD_FAILED,
        ...         context={"source": "file.csv"},
        ...         message_template="DATA_LOAD_FAILED",
        ...         message_kwargs={"source": "file.csv", "reason": str(e)}
        ...     )
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation_name: Optional[str] = None,
        enable_verbose_logging: bool = True,
    ):
        """
        Initialize error handler.

        Parameters
        ----------
            logger: Logger instance for error logging
            operation_name: Name of the operation being performed
            enable_verbose_logging: Enable detailed debug logging
        """
        self.logger = logger
        self.operation_name = operation_name
        self.enable_verbose_logging = enable_verbose_logging
        self._error_count = 0
        self._fallback_count = 0

    def handle_error(
        self,
        error: Exception,
        error_code: str,
        context: Optional[Dict[str, Any]] = None,
        raise_error: bool = False,
        message_template: Optional[str] = None,
        message_kwargs: Optional[Dict[str, Any]] = None,
    ) -> OperationResult:
        """
        Handle error with structured logging and optional recovery.

        Parameters
        ----------
        error : Exception
            The exception that occurred.
        error_code : str
            Standard error code from ErrorCode registry.
        context : dict, optional
            Additional context information to log.
        raise_error : bool, optional
            Whether to re-raise the error after logging. Default: False.
        message_template : str, optional
            Error message template name from ErrorMessages (e.g., "DATA_LOAD_FAILED").
        message_kwargs : dict, optional
            Parameters for formatting the message template.

        Returns
        -------
        OperationResult
            Structured result with error details and metrics.

        Raises
        ------
        ValueError
            If error_code is invalid or message_template doesn't exist
        Exception
            Re-raises the original error if raise_error=True
        """
        self._error_count += 1
        context = context or {}
        message_kwargs = message_kwargs or {}

        # Determine exception ownership and effective error code
        exception_owns_error, effective_error_code = self._resolve_error_ownership(
            error, error_code
        )

        # Validate and potentially fallback to default error code
        effective_error_code = self._validate_error_code(
            effective_error_code, exception_owns_error
        )

        # Determine message template with smart defaults
        message_template = self._resolve_message_template(
            message_template, effective_error_code, exception_owns_error
        )

        # Validate error code matches exception type (if applicable)
        self._validate_exception_type_match(
            error, effective_error_code, exception_owns_error
        )

        # Validate message template and parameters
        self._validate_message_template(
            message_template, message_kwargs, exception_owns_error
        )

        # Build the error message
        base_message = self._build_error_message(
            error, message_template, message_kwargs, exception_owns_error
        )

        # Get error metadata
        metadata = get_error_metadata(effective_error_code)

        # Build comprehensive error details
        error_details = self._build_error_details(
            error=error,
            effective_error_code=effective_error_code,
            base_message=base_message,
            context=context,
            metadata=metadata,
        )

        # Log with full context
        self._log_error(error_details, metadata, error)

        # Create and populate OperationResult
        result = self._create_operation_result(
            error=error,
            base_message=base_message,
            error_details=error_details,
            metadata=metadata,
        )

        if raise_error:
            raise error

        return result

    def standardize_result(
        self,
        result: OperationResult,
        error_code: str,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OperationResult:
        """
        Attach standardized error metadata to an existing OperationResult.

        This is useful when a function already created an OperationResult but
        did not include standardized error codes/metadata.

        Parameters
        ----------
            result: Existing OperationResult to enhance
            error_code: Standard error code to attach
            message: Optional override message
            context: Additional context information

        Returns
        -------
            Enhanced OperationResult with standardized metadata
        """
        context = context or {}
        message = message or result.error_message or "Unknown error"

        # Validate error code exists
        error_code = self._validate_error_code(error_code, exception_owns_error=False)

        metadata = get_error_metadata(error_code)
        error_details = {
            "error_type": "OperationError",
            "error_code": error_code,
            "error_message": message,
            "operation": self.operation_name,
            "context": context,
            "suggestions": ErrorContext.get_suggestions(error_code),
            "severity": metadata.get("severity"),
            "category": metadata.get("category"),
            "retry_allowed": metadata.get("retry_allowed"),
            "user_facing": metadata.get("user_facing"),
        }

        # Log standardized error info
        log_level = self._get_log_level(metadata.get("severity", "error"))
        self.logger.log(
            log_level,
            f"[{error_code}] {message}",
            extra={"error_details": error_details},
        )

        # Ensure error message is set
        result.error_message = message

        # Attach metrics idempotently
        self._attach_metrics_to_result(result, error_code, error_details, metadata)

        return result

    def create_error(
        self,
        error_code: str,
        message_kwargs: Dict[str, Any],
        exception_class: type = BasePamolaError,
        details: Optional[Dict[str, Any]] = None,
    ) -> BasePamolaError:
        """
        Create a properly formatted BasePamolaError with standardized message.

        Parameters
        ----------
            error_code: Error code from ErrorCode registry
            message_kwargs: Parameters for message formatting
            exception_class: Exception class to instantiate (must be BasePamolaError subclass)
            details: Additional details to attach to the exception

        Returns
        -------
            Formatted BasePamolaError instance

        Raises
        ------
            ValidationError: If exception_class is not BasePamolaError subclass
            InvalidParameterError: If error code is invalid, template not found, or missing parameters

        Examples
        --------
            >>> error = handler.create_error(
            ...     error_code=ErrorCode.FIELD_NOT_FOUND,
            ...     message_kwargs={"field_name": "age", "available_fields": "name, email"},
            ...     exception_class=FieldNotFoundError,
            ...     details={"dataset_columns": ["name", "email"]}
            ... )
            >>> raise error
        """
        # Validate exception class
        if not issubclass(exception_class, BasePamolaError):
            raise ValidationError(
                message=(
                    "exception_class must be a subclass of BasePamolaError, "
                    f"got {exception_class}"
                ),
                error_code=ErrorCode.PARAM_TYPE_ERROR,
            )

        # Validate error code exists
        try:
            ErrorCode.validate_code(error_code)
        except (ValidationError, ValueError) as e:
            raise InvalidParameterError(
                param_name="error_code",
                param_value=error_code,
                reason=str(e),
            )

        # Validate message template exists
        if not hasattr(ErrorMessages, error_code):
            raise InvalidParameterError(
                param_name="error_code",
                param_value=error_code,
                reason=f"No message template found for error code '{error_code}'",
            )

        # Validate template parameters
        valid, missing = ErrorMessages.validate_template_params(
            error_code, **message_kwargs
        )
        if not valid:
            raise InvalidParameterError(
                param_name="message_kwargs",
                param_value=list(message_kwargs.keys()),
                reason=(
                    f"Missing required parameters for template '{error_code}': "
                    f"{', '.join(missing)}"
                ),
            )

        # Get formatted message
        message = ErrorMessages.format(error_code, **message_kwargs)

        # Create exception
        return exception_class(
            message=message,
            error_code=error_code,
            details=details or {},
        )

    @contextmanager
    def error_context(
        self,
        error_code: str,
        context: Optional[Dict[str, Any]] = None,
        message_kwargs: Optional[Dict[str, Any]] = None,
        suppress: bool = False,
    ):
        """
        Context manager for automatic error handling.

        Parameters
        ----------
            error_code: Error code to use if exception occurs
            context: Additional context information
            message_kwargs: Message template parameters
            suppress: If True, suppress exceptions and return None

        Yields
        ------
            None

        Examples
        --------
            >>> with handler.error_context(ErrorCode.DATA_LOAD_FAILED, {"file": "data.csv"}):
            ...     load_data()
        """
        try:
            yield
        except Exception as e:
            result = self.handle_error(
                error=e,
                error_code=error_code,
                context=context,
                message_kwargs=message_kwargs,
                raise_error=not suppress,
            )
            if suppress:
                return result

    def capture_errors(
        self,
        error_code: str,
        rethrow: bool = False,
        context: Optional[Dict[str, Any]] = None,
        message_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to wrap functions with standardized error handling.

        Parameters
        ----------
            error_code: Error code to use for caught exceptions
            rethrow: Whether to re-raise after handling
            context: Additional context for error handling
            message_kwargs: Message template parameters

        Returns
        -------
            Decorated function

        Examples
        --------
            >>> @handler.capture_errors(error_code=ErrorCode.PROCESSING_FAILED)
            ... def process_data(data):
            ...     return data.transform()
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    result = self.handle_error(
                        error=exc,
                        error_code=error_code,
                        context=context or {},
                        message_kwargs=message_kwargs or {},
                        raise_error=rethrow,
                    )
                    if rethrow:
                        raise
                    return result

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    result = self.handle_error(
                        error=exc,
                        error_code=error_code,
                        context=context or {},
                        message_kwargs=message_kwargs or {},
                        raise_error=rethrow,
                    )
                    if rethrow:
                        raise
                    return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)

        return decorator

    def get_stats(self) -> Dict[str, int]:
        """Get error handling statistics."""
        return {
            "total_errors": self._error_count,
            "fallback_codes_used": self._fallback_count,
        }

    def reset_stats(self) -> None:
        """Reset error handling statistics."""
        self._error_count = 0
        self._fallback_count = 0

    # Private helper methods

    def _resolve_error_ownership(
        self, error: Exception, error_code: str
    ) -> tuple[bool, str]:
        """Determine if exception owns its error code and which code to use."""
        exception_error_code = getattr(error, "error_code", None)
        exception_owns_error = (
            exception_error_code is not None and exception_error_code != error_code
        )
        effective_error_code = (
            str(exception_error_code) if exception_owns_error else error_code
        )
        return exception_owns_error, effective_error_code

    def _validate_error_code(self, error_code: str, exception_owns_error: bool) -> str:
        """Validate error code and return fallback if invalid."""
        try:
            ErrorCode.validate_code(error_code)
            return error_code
        except (ValidationError, ValueError) as ve:
            self._fallback_count += 1
            self.logger.error(
                f"Invalid error code used: {ve}. Falling back to PROCESSING_FAILED"
            )
            return ErrorCode.PROCESSING_FAILED

    def _resolve_message_template(
        self,
        message_template: Optional[str],
        error_code: str,
        exception_owns_error: bool,
    ) -> Optional[str]:
        """Determine the appropriate message template to use."""
        if (
            not exception_owns_error
            and message_template is None
            and hasattr(ErrorMessages, error_code)
        ):
            return error_code
        return message_template

    def _validate_exception_type_match(
        self, error: Exception, error_code: str, exception_owns_error: bool
    ) -> None:
        """Validate that error code matches exception type."""
        if not exception_owns_error and isinstance(error, BasePamolaError):
            try:
                validate_error_code_usage(error_code, type(error))
            except (ValidationError, ValueError) as ve:
                self.logger.warning(f"Error code/exception mismatch: {ve}")

    def _validate_message_template(
        self,
        message_template: Optional[str],
        message_kwargs: Dict[str, Any],
        exception_owns_error: bool,
    ) -> None:
        """Validate message template exists and has required parameters."""
        if not exception_owns_error and message_template:
            if not hasattr(ErrorMessages, message_template):
                raise InvalidParameterError(
                    param_name="message_template",
                    param_value=message_template,
                    reason="template not found in ErrorMessages",
                )

            valid, missing = ErrorMessages.validate_template_params(
                message_template, **message_kwargs
            )
            if not valid:
                raise InvalidParameterError(
                    param_name="message_kwargs",
                    param_value=message_kwargs,
                    reason=(
                        f"missing parameters for template '{message_template}': {missing}"
                    ),
                )

    def _build_error_message(
        self,
        error: Exception,
        message_template: Optional[str],
        message_kwargs: Dict[str, Any],
        exception_owns_error: bool,
    ) -> str:
        """Build the final error message from various sources."""
        if exception_owns_error:
            if isinstance(error, BasePamolaError) and getattr(error, "message", None):
                return error.message
            return str(error)

        if message_template:
            return ErrorMessages.format(message_template, **message_kwargs)

        if isinstance(error, BasePamolaError) and getattr(error, "message", None):
            return error.message

        return str(error)

    def _build_error_details(
        self,
        error: Exception,
        effective_error_code: str,
        base_message: str,
        context: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build comprehensive error details dictionary."""
        return {
            "error_type": type(error).__name__,
            "error_code": effective_error_code,
            "error_message": base_message,
            "operation": self.operation_name,
            "context": context,
            "suggestions": ErrorContext.get_suggestions(effective_error_code),
            "severity": metadata.get("severity"),
            "category": metadata.get("category"),
            "retry_allowed": metadata.get("retry_allowed"),
            "user_facing": metadata.get("user_facing"),
        }

    def _log_error(
        self,
        error_details: Dict[str, Any],
        metadata: Dict[str, Any],
        error: Exception,
    ) -> None:
        """Log error with appropriate level and context."""
        log_level = self._get_log_level(metadata.get("severity", "error"))

        self.logger.log(
            log_level,
            f"[{error_details['error_code']}] {error_details['error_message']}",
            extra={"error_details": error_details},
            exc_info=True if self.enable_verbose_logging else False,
        )

        # Log recovery suggestions
        suggestions = error_details["suggestions"]
        if suggestions and self.enable_verbose_logging:
            self.logger.info("Recovery suggestions:")
            for idx, suggestion in enumerate(suggestions, 1):
                self.logger.info(f"  {idx}. {suggestion}")

    def _create_operation_result(
        self,
        error: Exception,
        base_message: str,
        error_details: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> OperationResult:
        """Create and populate OperationResult with error information."""
        # lazy import here — the only place to initialize OperationResult at runtime
        from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

        result = OperationResult(
            status=OperationStatus.ERROR,
            error_message=base_message,
            error_trace=self._get_formatted_traceback(error),
            exception=error,
        )

        # Attach metrics
        self._attach_metrics_to_result(
            result, error_details["error_code"], error_details, metadata
        )

        return result

    def _attach_metrics_to_result(
        self,
        result: OperationResult,
        error_code: str,
        error_details: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Attach error metrics to OperationResult (idempotent)."""
        if not hasattr(result, "add_metric"):
            return

        if result.metrics.get("error_code") is None:
            result.add_metric("error_code", error_code)
        if result.metrics.get("error_type") is None:
            result.add_metric("error_type", error_details["error_type"])
        if result.metrics.get("error_severity") is None:
            result.add_metric("error_severity", metadata.get("severity"))
        if result.metrics.get("error_category") is None:
            result.add_metric("error_category", metadata.get("category"))
        if result.metrics.get("retry_allowed") is None:
            result.add_metric("retry_allowed", metadata.get("retry_allowed"))

        result.add_nested_metric(
            "error_context", "details", error_details.get("context", {})
        )
        result.add_nested_metric(
            "error_context", "suggestions", error_details["suggestions"]
        )

    @staticmethod
    def _get_log_level(severity: str) -> int:
        """Convert severity to logging level."""
        severity_map = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }
        return severity_map.get(severity.lower(), logging.ERROR)

    @staticmethod
    def _get_formatted_traceback(error: Exception) -> str:
        """Get formatted traceback string."""
        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
