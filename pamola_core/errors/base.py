"""Base exception classes and decorator utilities for PAMOLA errors."""

from typing import Any, Callable, Dict, List, Optional, Type


class BasePamolaError(Exception):
    """
    Base exception for all PAMOLA errors.

    All PAMOLA exceptions should inherit from this class to ensure consistent
    error handling, logging, and telemetry across the framework.

    Attributes
    ----------
        message (str): Human-readable error message
        error_code (str): Standardized error code from ErrorCode registry
        details (Dict[str, Any]): Additional structured context for debugging

    Examples
    --------
        >>> raise BasePamolaError(
        ...     message="Operation failed",
        ...     error_code=ErrorCode.PROCESSING_FAILED,
        ...     details={"operation": "data_transform", "record_count": 1000}
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to structured dictionary for logging/telemetry.

        Returns
        -------
            Dictionary with error details and metadata from ErrorCode registry
        """
        from pamola_core.errors.codes.metadata import get_error_metadata

        metadata = get_error_metadata(self.error_code) if self.error_code else {}
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "severity": metadata.get("severity"),
            "category": metadata.get("category"),
            "retry_allowed": metadata.get("retry_allowed"),
            "user_facing": metadata.get("user_facing"),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message[:50]}...)"


def auto_exception(
    default_error_code: str,
    message_params: Optional[List[str]] = None,
    detail_params: Optional[List[str]] = None,
    custom_message_builder: Optional[Callable] = None,
    parent_class: Optional[Type[BasePamolaError]] = None,
):
    """
    Decorator to auto-generate exception __init__ method.

    This decorator eliminates boilerplate code by automatically:
    - Building error messages from ErrorMessages templates
    - Populating details dictionary with provided parameters
    - Handling None/default values gracefully

    Parameters
    ----------
        default_error_code: Default error code for this exception
        message_params: Parameters needed for ErrorMessages.format() template
        detail_params: Parameters to include in details dict (defaults to message_params)
        custom_message_builder: Custom function for complex message building
        parent_class: Parent exception class to call __init__ on (for subclasses)

    Examples
    --------
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"],
            detail_params=["source", "operation", "reason"]
        )
        class DataError(BasePamolaError):
            '''Errors related to data operations'''
            pass

        # Usage:
        raise DataError(source="database", reason="Connection timeout")
    """

    def decorator(cls: Type[BasePamolaError]) -> Type[BasePamolaError]:
        msg_params = message_params or []
        dtl_params = detail_params or msg_params
        parent = parent_class or BasePamolaError

        def __init__(
            self,
            message: Optional[str] = None,
            *,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            **params,
        ):
            from pamola_core.errors.messages.registry import ErrorMessages

            effective_error_code = error_code or default_error_code

            # Build message
            if message is None:
                if custom_message_builder:
                    message = custom_message_builder(effective_error_code, **params)
                elif msg_params:
                    template_args = {k: params.get(k, "<unknown>") for k in msg_params}
                    message = ErrorMessages.format(
                        effective_error_code, **template_args
                    )
                else:
                    message = params.get("reason", "Unknown error")

            # Build details
            details_payload = details.copy() if details else {}
            details_payload.update(
                {k: v for k, v in params.items() if k in dtl_params and v is not None}
            )

            # Call parent __init__
            parent.__init__(
                self,
                message=message,
                error_code=effective_error_code,
                details=details_payload,
            )

        cls.__init__ = __init__  # type: ignore[assignment,method-assign]
        return cls

    return decorator


def _format_field_list(fields: Optional[List[str]], max_show: int = 10) -> str:
    """Format field list for display with truncation."""
    if not fields:
        return "none"
    if len(fields) <= max_show:
        return ", ".join(fields)
    slice_fields = fields[:max_show]
    extra = len(fields) - max_show
    return f"{', '.join(slice_fields)} (+{extra} more)"
