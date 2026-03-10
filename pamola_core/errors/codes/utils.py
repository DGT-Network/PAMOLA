"""
Utility functions for error code handling.

Provides validation, information retrieval, and formatting utilities.
"""

from typing import Any, Dict

from pamola_core.errors.codes.registry import ErrorCode
from pamola_core.errors.codes.metadata import get_error_metadata


def validate_error_code_usage(error_code: str, exception_type: type) -> None:
    """
    Validate that error code is used with appropriate exception type.

    Parameters
    ----------
        error_code: Error code to validate
        exception_type: Exception class being used

    Raises
    ------
        ValueError: When code/exception mismatch detected

    Examples
    --------
        >>> validate_error_code_usage("DATA_LOAD_FAILED", DataError)  # OK
        >>> validate_error_code_usage("DATA_LOAD_FAILED", CacheError)  # Raises ValueError
    """
    ErrorCode.validate_code(error_code)

    metadata = get_error_metadata(error_code)
    expected_type_name = metadata.get("exception_type")

    if not expected_type_name:
        return

    try:
        import pamola_core.errors.exceptions.core_domain as exc_core
        import pamola_core.errors.exceptions.validation as exc_validation
        import pamola_core.errors.exceptions.crypto as exc_crypto
        import pamola_core.errors.exceptions.tasks as exc_tasks
        import pamola_core.errors.exceptions.filesystem as exc_fs
        import pamola_core.errors.exceptions.nlp as exc_nlp
        import pamola_core.errors.exceptions.resources as exc_resources
        from pamola_core.errors.base import BasePamolaError
    except Exception:
        return

    exception_map = {
        "DataError": exc_core.DataError,
        "ValidationError": exc_validation.ValidationError,
        "ProcessingError": exc_core.ProcessingError,
        "CacheError": exc_core.CacheError,
        "ArtifactError": exc_core.ArtifactError,
        "VisualizationError": exc_core.VisualizationError,
        "ConfigurationError": exc_core.ConfigurationError,
        "EncryptionError": exc_crypto.EncryptionError,
        "DecryptionError": exc_crypto.DecryptionError,
        "FieldNotFoundError": exc_validation.FieldNotFoundError,
        "FieldTypeError": exc_validation.FieldTypeError,
        "FieldValueError": exc_validation.FieldValueError,
        "ColumnNotFoundError": exc_validation.ColumnNotFoundError,
        "InvalidParameterError": exc_validation.InvalidParameterError,
        "InvalidStrategyError": exc_validation.InvalidStrategyError,
        "ConditionalValidationError": exc_validation.ConditionalValidationError,
        "RangeValidationError": exc_validation.RangeValidationError,
        "InvalidDataFormatError": exc_validation.InvalidDataFormatError,
        "MarkerValidationError": exc_validation.MarkerValidationError,
        "FileValidationError": exc_validation.FileValidationError,
        "PamolaFileNotFoundError": exc_validation.PamolaFileNotFoundError,
        "InvalidFileFormatError": exc_validation.InvalidFileFormatError,
        "MultipleValidationErrors": exc_validation.MultipleValidationErrors,
        "DataWriteError": exc_core.DataWriteError,
        "DataFrameProcessingError": exc_core.DataFrameProcessingError,
        "ResourceError": exc_resources.ResourceError,
        "ResourceNotFoundError": exc_nlp.ResourceNotFoundError,
        "ConfigSaveError": exc_core.ConfigSaveError,
        "EncryptionInitializationError": exc_crypto.EncryptionInitializationError,
        "KeyGenerationError": exc_crypto.KeyGenerationError,
        "KeyLoadingError": exc_crypto.KeyLoadingError,
        "CryptoKeyError": exc_crypto.CryptoKeyError,
        "MasterKeyError": exc_crypto.MasterKeyError,
        "TaskKeyError": exc_crypto.TaskKeyError,
        "KeyStoreError": exc_crypto.KeyStoreError,
        "CryptoError": exc_crypto.CryptoError,
        "ProviderError": exc_crypto.ProviderError,
        "ModeError": exc_crypto.ModeError,
        "FormatError": exc_crypto.FormatError,
        "PseudonymizationError": exc_crypto.PseudonymizationError,
        "HashCollisionError": exc_crypto.HashCollisionError,
        "DataRedactionError": exc_crypto.DataRedactionError,
        "LegacyMigrationError": exc_crypto.LegacyMigrationError,
        "AgeToolError": exc_crypto.AgeToolError,
        "TaskError": exc_tasks.TaskError,
        "TaskInitializationError": exc_tasks.TaskInitializationError,
        "TaskExecutionError": exc_tasks.TaskExecutionError,
        "TaskFinalizationError": exc_tasks.TaskFinalizationError,
        "DependencyError": exc_tasks.DependencyError,
        "DependencyMissingError": exc_tasks.DependencyMissingError,
        "DependencyFailedError": exc_tasks.DependencyFailedError,
        "TaskRegistryError": exc_tasks.TaskRegistryError,
        "ContextManagerError": exc_tasks.ContextManagerError,
        "ExecutionError": exc_tasks.ExecutionError,
        "ExecutionLogError": exc_tasks.ExecutionLogError,
        "CheckpointError": exc_tasks.CheckpointError,
        "StateSerializationError": exc_tasks.StateSerializationError,
        "StateRestorationError": exc_tasks.StateRestorationError,
        "MaxRetriesExceededError": exc_tasks.MaxRetriesExceededError,
        "NonRetriableError": exc_tasks.NonRetriableError,
        "OpsError": exc_tasks.OpsError,
        "FeatureNotImplementedError": exc_tasks.FeatureNotImplementedError,
        "PathValidationError": exc_fs.PathValidationError,
        "PathSecurityError": exc_fs.PathSecurityError,
        "DirectoryCreationError": exc_fs.DirectoryCreationError,
        "DirectoryManagerError": exc_fs.DirectoryManagerError,
        "NLPError": exc_nlp.NLPError,
        "PromptValidationError": exc_nlp.PromptValidationError,
        "LLMError": exc_nlp.LLMError,
        "LLMConnectionError": exc_nlp.LLMConnectionError,
        "LLMGenerationError": exc_nlp.LLMGenerationError,
        "LLMResponseError": exc_nlp.LLMResponseError,
        "ModelNotAvailableError": exc_nlp.ModelNotAvailableError,
        "ModelLoadError": exc_nlp.ModelLoadError,
        "UnsupportedLanguageError": exc_nlp.UnsupportedLanguageError,
        "DateTimeParsingError": exc_resources.DateTimeParsingError,
        "DateTimeGeneralizationError": exc_resources.DateTimeGeneralizationError,
        "InsufficientPrivacyError": exc_resources.InsufficientPrivacyError,
        "MappingError": exc_resources.MappingError,
        "MappingStorageError": exc_resources.MappingStorageError,
        "FakeDataError": exc_resources.FakeDataError,
        "ReportingError": exc_resources.ReportingError,
        "BasePamolaError": BasePamolaError,
    }

    expected_type = exception_map.get(expected_type_name)
    if expected_type and not issubclass(exception_type, expected_type):
        from pamola_core.errors.exceptions.validation import InvalidParameterError

        raise InvalidParameterError(
            param_name="error_code",
            param_value=error_code,
            reason=(
                f"should be used with {expected_type.__name__}, "
                f"not {exception_type.__name__}"
            ),
        )


def get_error_info(error_code: str) -> Dict[str, Any]:
    """
    Get comprehensive error information including metadata and suggestions.

    Parameters
    ----------
        error_code: The error code to get information for

    Returns
    -------
        Dictionary with error metadata and recovery suggestions

    Examples
    --------
        >>> info = get_error_info("DATA_LOAD_FAILED")
        >>> print(info["metadata"]["severity"])
        error
        >>> print(info["recovery_suggestions"][0])
        Verify the file path exists and is accessible.
    """
    from pamola_core.errors.context.suggestions import ErrorContext

    metadata = get_error_metadata(error_code)
    suggestions = ErrorContext.get_suggestions(error_code)

    return {
        "code": error_code,
        "metadata": metadata,
        "recovery_suggestions": suggestions,
        "has_specific_suggestions": ErrorContext.has_suggestions(error_code),
    }


def format_error_help(error_code: str) -> str:
    """
    Format comprehensive help text for an error code.

    Parameters
    ----------
        error_code: The error code to format help for

    Returns
    -------
        Formatted help text with metadata and suggestions
    """
    from pamola_core.errors.context.suggestions import ErrorContext

    info = get_error_info(error_code)
    metadata = info["metadata"]

    lines = [
        f"Error Code: {error_code}",
        f"Category: {metadata['category']}",
        f"Severity: {metadata['severity']}",
        f"Retry Allowed: {metadata['retry_allowed']}",
        f"User Facing: {metadata['user_facing']}",
        "",
        ErrorContext.format_suggestions(error_code),
    ]

    return "\n".join(lines)
