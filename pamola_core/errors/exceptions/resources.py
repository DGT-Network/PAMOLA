"""Resource, datetime, privacy, mapping, and reporting exceptions."""

from typing import Optional

from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


@auto_exception(
    default_error_code=ErrorCode.DATETIME_PARSE_FAILED,
    message_params=["value", "format", "reason"],
    detail_params=["value", "format", "reason"],
)
class DateTimeParsingError(BasePamolaError):
    """Errors while parsing datetime values."""

    pass


@auto_exception(
    default_error_code=ErrorCode.DATETIME_GENERALIZATION_FAILED,
    message_params=["field_name", "reason"],
    detail_params=["field_name", "reason"],
)
class DateTimeGeneralizationError(BasePamolaError):
    """Errors during datetime generalization."""

    pass


@auto_exception(
    default_error_code=ErrorCode.PRIVACY_INSUFFICIENT,
    message_params=["required_level", "achieved_level"],
    detail_params=["required_level", "achieved_level", "reason"],
)
class InsufficientPrivacyError(BasePamolaError):
    """Raised when privacy constraints cannot be satisfied."""

    pass


# =============================================================================
# 9. RESOURCE MANAGEMENT
# =============================================================================


class ResourceError(BasePamolaError):
    """Resource-related errors (memory, disk, CPU, etc.)."""

    def __init__(
        self,
        message: Optional[str] = None,
        operation: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        if message is None:
            operation_display = operation or "<unknown>"
            reason_display = reason or "resource error"
            message = (
                f"Resource error for operation '{operation_display}': {reason_display}"
            )

        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_MEMORY_EXCEEDED,
            details={"operation": operation, "reason": reason},
        )


# =============================================================================
# 10. DATA MAPPING & GENERATION
# =============================================================================


@auto_exception(
    default_error_code=ErrorCode.MAPPING_ERROR,
    message_params=["context", "reason"],
    detail_params=["context", "reason"],
)
class MappingError(BasePamolaError):
    """Mapping errors within fake data or anonymization."""

    pass


@auto_exception(
    default_error_code=ErrorCode.MAPPING_STORAGE_ERROR,
    message_params=["reason"],
    detail_params=["reason"],
)
class MappingStorageError(BasePamolaError):
    """Errors for mapping storage operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.FAKE_DATA_GENERATION_FAILED,
    message_params=["field_name", "reason"],
    detail_params=["field_name", "reason"],
)
class FakeDataError(BasePamolaError):
    """Errors raised by fake data generators."""

    pass


# =============================================================================
# 11. REPORTING
# =============================================================================


@auto_exception(
    default_error_code=ErrorCode.REPORTING_ERROR,
    message_params=["report_name", "reason"],
    detail_params=["report_name", "reason"],
)
class ReportingError(BasePamolaError):
    """Errors in reporting pipeline."""

    pass
