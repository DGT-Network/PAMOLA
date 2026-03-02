"""Task execution and operations exceptions."""

from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


@auto_exception(
    default_error_code=ErrorCode.TASK_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class TaskError(BasePamolaError):
    """Base task error."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_INIT_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class TaskInitializationError(BasePamolaError):
    """Exception raised when task initialization fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_EXECUTION_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class TaskExecutionError(BasePamolaError):
    """Exception raised during task execution."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_FINALIZATION_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class TaskFinalizationError(BasePamolaError):
    """Exception raised during task finalization."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_DEPENDENCY_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class TaskDependencyError(BasePamolaError):
    """Exception raised when task dependencies are not satisfied."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_DEPENDENCY_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class DependencyError(BasePamolaError):
    """Generic dependency error."""

    pass


def _build_dependency_missing_message(error_code_val: str, **params) -> str:
    """
    Custom message builder for DependencyMissingError.

    Template: "Required dependency '{dependency}' is missing or unavailable: {reason}"
    Maps: dependency_name → dependency
    """
    from pamola_core.errors.messages.registry import ErrorMessages

    dependency_name = params.get("dependency_name", "<unknown>")
    reason = params.get("reason")
    required_by = params.get("required_by")

    if reason:
        reason_display = reason
    elif required_by:
        reason_display = f"required by {required_by}"
    else:
        reason_display = "missing or unavailable"

    # Template: "Required dependency '{dependency}' is missing or unavailable: {reason}"
    return ErrorMessages.format(
        error_code_val, dependency=dependency_name, reason=reason_display
    )


@auto_exception(
    default_error_code=ErrorCode.DEPENDENCY_MISSING,
    custom_message_builder=_build_dependency_missing_message,
    detail_params=["dependency_name", "required_by", "reason"],
)
class DependencyMissingError(BasePamolaError):
    """Exception raised when a required dependency is missing."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_DEPENDENCY_FAILED,
    message_params=["task_name", "dependency_name"],
    detail_params=["task_name", "dependency_name", "reason"],
)
class DependencyFailedError(BasePamolaError):
    """Exception raised when a dependency task has failed."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_REGISTRY_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class TaskRegistryError(BasePamolaError):
    """Errors when registering or resolving tasks."""

    pass


@auto_exception(
    default_error_code=ErrorCode.EXECUTION_ERROR,
    message_params=["operation", "reason"],
    detail_params=["operation", "reason"],
)
class ExecutionError(BasePamolaError):
    """Generic execution error for tasks."""

    pass


@auto_exception(
    default_error_code=ErrorCode.EXECUTION_LOG_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class ExecutionLogError(BasePamolaError):
    """Errors while handling execution logs."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CHECKPOINT_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class CheckpointError(BasePamolaError):
    """Exception raised for checkpoint-related errors."""

    pass


@auto_exception(
    default_error_code=ErrorCode.STATE_SERIALIZATION_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class StateSerializationError(BasePamolaError):
    """Exception raised when state serialization fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.STATE_RESTORATION_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class StateRestorationError(BasePamolaError):
    """Exception raised when state restoration fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.TASK_CONTEXT_ERROR,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"],
)
class ContextManagerError(BasePamolaError):
    """Errors in task context management."""

    pass


@auto_exception(
    default_error_code=ErrorCode.MAX_RETRIES_EXCEEDED,
    message_params=["operation", "max_retries"],
    detail_params=["operation", "max_retries", "reason"],
)
class MaxRetriesExceededError(BasePamolaError):
    """Raised when maximum retry attempts are reached."""

    pass


@auto_exception(
    default_error_code=ErrorCode.NON_RETRIABLE_ERROR,
    message_params=["operation", "reason"],
    detail_params=["operation", "reason"],
)
class NonRetriableError(BasePamolaError):
    """Raised for errors that should not be retried."""

    pass


@auto_exception(
    default_error_code=ErrorCode.OPERATION_ERROR,
    message_params=["operation", "reason"],
    detail_params=["operation", "reason"],
)
class OpsError(BasePamolaError):
    """Base error for operations (ops namespace)."""

    pass


class FeatureNotImplementedError(BasePamolaError):
    """Raised when a requested feature or code path is not implemented."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code=ErrorCode.FEATURE_NOT_IMPLEMENTED,
            details={"reason": message},
        )
