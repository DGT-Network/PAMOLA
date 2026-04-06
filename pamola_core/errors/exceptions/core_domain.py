"""Core domain exceptions (data, config, processing, cache, artifacts, visualization)."""

from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


def _build_data_error_message(error_code_val: str, **params) -> str:
    """
    Custom message builder for DataError with conditional logic.

    Maps params to appropriate templates:
    - source → DATA_LOAD_FAILED: "Failed to load data from source '{source}': {reason}"
    - operation → DATA_VALIDATION_ERROR: "Data validation failed for '{context}': {reason}"
    """
    from pamola_core.errors.messages.registry import ErrorMessages

    source = params.get("source")
    operation = params.get("operation")
    reason = params.get("reason", "unknown error")

    if source:
        # Template: "Failed to load data from source '{source}': {reason}"
        return ErrorMessages.format(
            ErrorCode.DATA_LOAD_FAILED, source=source, reason=reason
        )
    elif operation:
        # Template: "Data validation failed for '{context}': {reason}"
        return ErrorMessages.format(
            ErrorCode.DATA_VALIDATION_ERROR, context=operation, reason=reason
        )
    else:
        return reason


@auto_exception(
    default_error_code=ErrorCode.DATA_VALIDATION_ERROR,
    custom_message_builder=_build_data_error_message,
    detail_params=["source", "operation", "reason"],
)
class DataError(BasePamolaError):
    """Errors related to data operations (loading, validation, processing)."""

    pass


def _build_data_write_error_message(error_code_val: str, **params) -> str:
    """
    Custom message builder for DataWriteError.

    Template: "Failed to write data to '{destination}': {reason}"
    """
    from pamola_core.errors.messages.registry import ErrorMessages

    destination = params.get("destination") or params.get("file_path")
    reason = params.get("reason", "unknown error")

    if destination:
        # Template: "Failed to write data to '{destination}': {reason}"
        return ErrorMessages.format(
            ErrorCode.DATA_WRITE_FAILED, destination=destination, reason=reason
        )
    else:
        return reason or "Data write failed"


@auto_exception(
    default_error_code=ErrorCode.DATA_WRITE_FAILED,
    custom_message_builder=_build_data_write_error_message,
    detail_params=["destination", "file_path", "reason"],
)
class DataWriteError(BasePamolaError):
    """Errors writing data or artifacts to storage."""

    pass


@auto_exception(
    default_error_code=ErrorCode.DATA_FRAME_PROCESSING_ERROR,
    message_params=["operation", "reason"],
    detail_params=["operation", "reason"],
)
class DataFrameProcessingError(BasePamolaError):
    """Errors processing pandas or Dask DataFrames."""

    pass


def _build_config_error_message(error_code_val: str, **params) -> str:
    """
    Custom message builder for ConfigurationError.

    Maps to templates:
    - CONFIG_MISSING: "Missing required configuration: {key}"
    - CONFIG_INVALID: "Invalid configuration: {reason}"
    """
    from pamola_core.errors.messages.registry import ErrorMessages

    config_key = params.get("config_key")
    reason = params.get("reason")

    if config_key and not reason:
        # Template: "Missing required configuration: {key}"
        return ErrorMessages.format(ErrorCode.CONFIG_MISSING, key=config_key)
    else:
        reason_display = reason or "invalid configuration"
        if config_key:
            reason_display = f"{config_key}: {reason_display}"
        # Template: "Invalid configuration: {reason}"
        return ErrorMessages.format(ErrorCode.CONFIG_INVALID, reason=reason_display)


@auto_exception(
    default_error_code=ErrorCode.CONFIG_INVALID,
    custom_message_builder=_build_config_error_message,
    detail_params=["config_key", "reason"],
)
class ConfigurationError(BasePamolaError):
    """Errors related to configuration parsing or validation."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CONFIG_SAVE_FAILED,
    message_params=["path", "reason"],
    detail_params=["path", "reason"],
)
class ConfigSaveError(BasePamolaError):
    """Errors saving configuration to disk."""

    pass


@auto_exception(
    default_error_code=ErrorCode.PROCESSING_FAILED,
    message_params=["operation", "reason"],
    detail_params=["operation", "reason"],
)
class ProcessingError(BasePamolaError):
    """Errors during data processing operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.PROCESSING_BATCH_FAILED,
    message_params=["batch_index", "reason"],
    detail_params=["batch_index", "operation", "reason"],
)
class BatchProcessingError(ProcessingError):
    """Errors during batch processing operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.PROCESSING_CHUNK_FAILED,
    message_params=["chunk_index", "reason"],
    detail_params=["chunk_index", "operation", "reason"],
)
class ChunkProcessingError(ProcessingError):
    """Errors during chunk processing operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CACHE_READ_FAILED,
    message_params=["cache_key", "reason"],
    detail_params=["cache_key", "operation", "reason"],
)
class CacheError(BasePamolaError):
    """Errors related to caching operations (read, write, invalidation)."""

    pass


@auto_exception(
    default_error_code=ErrorCode.ARTIFACT_VALIDATION_FAILED,
    message_params=["path", "reason"],
    detail_params=["path", "operation", "reason"],
)
class ArtifactError(BasePamolaError):
    """Errors related to artifact management (validation, storage, retrieval)."""

    pass


@auto_exception(
    default_error_code=ErrorCode.VISUALIZATION_FAILED,
    message_params=["name", "reason"],
    detail_params=["name", "reason"],
)
class VisualizationError(BasePamolaError):
    """Errors related to visualization generation or rendering."""

    pass
