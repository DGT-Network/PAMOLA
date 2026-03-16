"""
Error code metadata registry for PAMOLA Core.

Provides severity, category, retry settings, and exception type mappings
for all error codes.

Design note: Dict keys are raw string literals (not ErrorCode.XXX attributes)
to avoid a circular import with codes/registry.py. Both resolve to the same
values — ErrorCode class variables are themselves plain strings.
"""

from typing import Any, Dict


ERROR_CODE_METADATA: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Data Errors
    # =========================================================================
    "DATA_LOAD_FAILED": {
        "severity": "error",
        "category": "data",
        "retry_allowed": True,
        "user_facing": True,
        "exception_type": "DataError",
    },
    "DATA_VALIDATION_ERROR": {
        "severity": "error",
        "category": "data",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "DataError",
    },
    "DATA_EMPTY": {
        "severity": "warning",
        "category": "data",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "DataError",
    },
    "DATA_SOURCE_INVALID": {
        "severity": "error",
        "category": "data",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "DataError",
    },
    "DATA_WRITE_FAILED": {
        "severity": "error",
        "category": "data",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "DataWriteError",
    },
    "DATA_FRAME_PROCESSING_ERROR": {
        "severity": "error",
        "category": "data",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "DataFrameProcessingError",
    },
    # =========================================================================
    # Validation Errors - Fields
    # =========================================================================
    "FIELD_NOT_FOUND": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FieldNotFoundError",
    },
    "FIELD_TYPE_ERROR": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FieldTypeError",
    },
    "FIELD_VALUE_ERROR": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FieldValueError",
    },
    "COLUMN_NOT_FOUND": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ColumnNotFoundError",
    },
    # =========================================================================
    # Validation Errors - Parameters
    # =========================================================================
    "PARAM_MISSING": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InvalidParameterError",
    },
    "PARAM_INVALID": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InvalidParameterError",
    },
    "PARAM_TYPE_ERROR": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InvalidParameterError",
    },
    "STRATEGY_INVALID": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InvalidStrategyError",
    },
    "MODE_INVALID": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ValidationError",
    },
    "NULL_STRATEGY_INVALID": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ValidationError",
    },
    # =========================================================================
    # Validation Errors - Conditional & Range
    # =========================================================================
    "VALIDATION_CONDITIONAL_FAILED": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ConditionalValidationError",
    },
    "VALIDATION_RANGE_FAILED": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "RangeValidationError",
    },
    "VALIDATION_FORMAT_INVALID": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InvalidDataFormatError",
    },
    "VALIDATION_MARKER_FAILED": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "MarkerValidationError",
    },
    # =========================================================================
    # Validation Errors - Files
    # =========================================================================
    "FILE_ERROR": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FileValidationError",
    },
    "FILE_NOT_FOUND": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "PamolaFileNotFoundError",
    },
    "FILE_FORMAT_INVALID": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InvalidFileFormatError",
    },
    "FILE_ACCESS_DENIED": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FileValidationError",
    },
    "FILE_CORRUPTED": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FileValidationError",
    },
    # =========================================================================
    # Validation Errors - Multiple
    # =========================================================================
    "MULTIPLE_ERRORS": {
        "severity": "error",
        "category": "validation",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "MultipleValidationErrors",
    },
    # =========================================================================
    # Processing Errors
    # =========================================================================
    "PROCESSING_FAILED": {
        "severity": "error",
        "category": "processing",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ProcessingError",
    },
    "PROCESSING_BATCH_FAILED": {
        "severity": "error",
        "category": "processing",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ProcessingError",
    },
    "PROCESSING_CHUNK_FAILED": {
        "severity": "error",
        "category": "processing",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ProcessingError",
    },
    "PROCESSING_TIMEOUT": {
        "severity": "error",
        "category": "processing",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ProcessingError",
    },
    "PROCESSING_INTERRUPTED": {
        "severity": "warning",
        "category": "processing",
        "retry_allowed": True,
        "user_facing": True,
        "exception_type": "ProcessingError",
    },
    "FEATURE_NOT_IMPLEMENTED": {
        "severity": "error",
        "category": "processing",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "FeatureNotImplementedError",
    },
    # =========================================================================
    # Resource Errors
    # =========================================================================
    "RESOURCE_MEMORY_EXCEEDED": {
        "severity": "critical",
        "category": "resource",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ResourceError",
    },
    "RESOURCE_DISK_FULL": {
        "severity": "critical",
        "category": "resource",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ResourceError",
    },
    "RESOURCE_CPU_THROTTLED": {
        "severity": "warning",
        "category": "resource",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ResourceError",
    },
    "RESOURCE_NOT_FOUND": {
        "severity": "error",
        "category": "resource",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ResourceNotFoundError",
    },
    # =========================================================================
    # Cache Errors
    # =========================================================================
    "CACHE_READ_FAILED": {
        "severity": "error",
        "category": "cache",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "CacheError",
    },
    "CACHE_WRITE_FAILED": {
        "severity": "error",
        "category": "cache",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "CacheError",
    },
    "CACHE_KEY_INVALID": {
        "severity": "error",
        "category": "cache",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "CacheError",
    },
    "CACHE_EXPIRED": {
        "severity": "warning",
        "category": "cache",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "CacheError",
    },
    "CACHE_CORRUPTED": {
        "severity": "warning",
        "category": "cache",
        "retry_allowed": True,
        "user_facing": True,
        "exception_type": "CacheError",
    },
    # =========================================================================
    # Artifact Errors
    # =========================================================================
    "ARTIFACT_VALIDATION_FAILED": {
        "severity": "error",
        "category": "artifact",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ArtifactError",
    },
    "ARTIFACT_NOT_FOUND": {
        "severity": "error",
        "category": "artifact",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ArtifactError",
    },
    "ARTIFACT_WRITE_FAILED": {
        "severity": "error",
        "category": "artifact",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ArtifactError",
    },
    "ARTIFACT_CORRUPTED": {
        "severity": "error",
        "category": "artifact",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ArtifactError",
    },
    # =========================================================================
    # Visualization Errors
    # =========================================================================
    "VISUALIZATION_FAILED": {
        "severity": "error",
        "category": "visualization",
        "retry_allowed": True,
        "user_facing": True,
        "exception_type": "VisualizationError",
    },
    "VISUALIZATION_BACKEND_ERROR": {
        "severity": "error",
        "category": "visualization",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "VisualizationError",
    },
    "VISUALIZATION_TIMEOUT": {
        "severity": "error",
        "category": "visualization",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "VisualizationError",
    },
    # =========================================================================
    # Configuration Errors
    # =========================================================================
    "CONFIG_INVALID": {
        "severity": "error",
        "category": "configuration",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ConfigurationError",
    },
    "CONFIG_MISSING": {
        "severity": "error",
        "category": "configuration",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ConfigurationError",
    },
    "CONFIG_PARSE_ERROR": {
        "severity": "error",
        "category": "configuration",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ConfigurationError",
    },
    "CONFIG_SAVE_FAILED": {
        "severity": "error",
        "category": "configuration",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ConfigSaveError",
    },
    # =========================================================================
    # Cryptography & Security Errors
    # =========================================================================
    "ENCRYPTION_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "EncryptionError",
    },
    "ENCRYPTION_INIT_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "EncryptionInitializationError",
    },
    "DECRYPTION_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "DecryptionError",
    },
    "KEY_GENERATION_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "KeyGenerationError",
    },
    "KEY_LOADING_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "KeyLoadingError",
    },
    "KEY_INVALID": {
        "severity": "critical",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "CryptoKeyError",
    },
    "KEY_EXPIRED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "CryptoKeyError",
    },
    "KEY_MASTER_ERROR": {
        "severity": "critical",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "MasterKeyError",
    },
    "KEY_TASK_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "TaskKeyError",
    },
    "KEY_STORE_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "KeyStoreError",
    },
    "CRYPTO_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "CryptoError",
    },
    "CRYPTO_PROVIDER_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "ProviderError",
    },
    "CRYPTO_MODE_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "ModeError",
    },
    "CRYPTO_FORMAT_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "FormatError",
    },
    "CRYPTO_ALGORITHM_UNSUPPORTED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "CryptoError",
    },
    "PSEUDONYMIZATION_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "PseudonymizationError",
    },
    "HASH_COLLISION_DETECTED": {
        "severity": "warning",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "HashCollisionError",
    },
    "DATA_REDACTION_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "DataRedactionError",
    },
    "CRYPTO_MIGRATION_FAILED": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "LegacyMigrationError",
    },
    "CRYPTO_TOOL_ERROR": {
        "severity": "error",
        "category": "encryption",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "AgeToolError",
    },
    # =========================================================================
    # Task Execution & Operations
    # =========================================================================
    "TASK_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "TaskError",
    },
    "TASK_INIT_FAILED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "TaskInitializationError",
    },
    "TASK_EXECUTION_FAILED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "TaskExecutionError",
    },
    "TASK_FINALIZATION_FAILED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "TaskFinalizationError",
    },
    "TASK_DEPENDENCY_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "DependencyError",
    },
    "TASK_DEPENDENCY_MISSING": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "DependencyMissingError",
    },
    "TASK_DEPENDENCY_FAILED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "DependencyFailedError",
    },
    "DEPENDENCY_MISSING": {
        "severity": "error",
        "category": "dependency",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "DependencyMissingError",
    },
    "TASK_REGISTRY_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "TaskRegistryError",
    },
    "TASK_CONTEXT_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "ContextManagerError",
    },
    "EXECUTION_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ExecutionError",
    },
    "EXECUTION_LOG_ERROR": {
        "severity": "warning",
        "category": "task",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ExecutionLogError",
    },
    "CHECKPOINT_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "CheckpointError",
    },
    "STATE_SERIALIZATION_FAILED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "StateSerializationError",
    },
    "STATE_RESTORATION_FAILED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "StateRestorationError",
    },
    "MAX_RETRIES_EXCEEDED": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "MaxRetriesExceededError",
    },
    "NON_RETRIABLE_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "NonRetriableError",
    },
    "OPERATION_ERROR": {
        "severity": "error",
        "category": "task",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "OpsError",
    },
    # =========================================================================
    # File System & Path Errors
    # =========================================================================
    "PATH_INVALID": {
        "severity": "error",
        "category": "filesystem",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "PathValidationError",
    },
    "PATH_SECURITY_VIOLATION": {
        "severity": "critical",
        "category": "filesystem",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "PathSecurityError",
    },
    "DIRECTORY_CREATE_FAILED": {
        "severity": "error",
        "category": "filesystem",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "DirectoryCreationError",
    },
    "DIRECTORY_MANAGER_ERROR": {
        "severity": "error",
        "category": "filesystem",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "DirectoryManagerError",
    },
    # =========================================================================
    # NLP & Machine Learning Errors
    # =========================================================================
    "NLP_ERROR": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "NLPError",
    },
    "NLP_PROMPT_INVALID": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "PromptValidationError",
    },
    "LLM_ERROR": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "LLMError",
    },
    "LLM_CONNECTION_FAILED": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "LLMConnectionError",
    },
    "LLM_GENERATION_FAILED": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "LLMGenerationError",
    },
    "LLM_RESPONSE_INVALID": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "LLMResponseError",
    },
    "MODEL_NOT_AVAILABLE": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "ModelNotAvailableError",
    },
    "MODEL_LOAD_FAILED": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ModelLoadError",
    },
    "LANGUAGE_UNSUPPORTED": {
        "severity": "error",
        "category": "nlp",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "UnsupportedLanguageError",
    },
    # =========================================================================
    # Data Processing & Privacy
    # =========================================================================
    "DATETIME_PARSE_FAILED": {
        "severity": "error",
        "category": "data",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "DateTimeParsingError",
    },
    "DATETIME_GENERALIZATION_FAILED": {
        "severity": "error",
        "category": "data",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "DateTimeGeneralizationError",
    },
    "PRIVACY_INSUFFICIENT": {
        "severity": "error",
        "category": "data",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "InsufficientPrivacyError",
    },
    # =========================================================================
    # Data Mapping & Generation
    # =========================================================================
    "MAPPING_ERROR": {
        "severity": "error",
        "category": "data",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "MappingError",
    },
    "MAPPING_STORAGE_ERROR": {
        "severity": "error",
        "category": "data",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "MappingStorageError",
    },
    "FAKE_DATA_GENERATION_FAILED": {
        "severity": "error",
        "category": "data",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "FakeDataError",
    },
    # =========================================================================
    # Reporting
    # =========================================================================
    "REPORTING_ERROR": {
        "severity": "error",
        "category": "reporting",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "ReportingError",
    },
    # =========================================================================
    # Network Errors
    # =========================================================================
    "NETWORK_CONNECTION_FAILED": {
        "severity": "error",
        "category": "network",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "BasePamolaError",
    },
    "NETWORK_TIMEOUT": {
        "severity": "error",
        "category": "network",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "BasePamolaError",
    },
    "NETWORK_DNS_FAILED": {
        "severity": "error",
        "category": "network",
        "retry_allowed": True,
        "user_facing": False,
        "exception_type": "BasePamolaError",
    },
    "NETWORK_SSL_ERROR": {
        "severity": "error",
        "category": "network",
        "retry_allowed": False,
        "user_facing": False,
        "exception_type": "BasePamolaError",
    },
    # =========================================================================
    # Authentication & Authorization Errors
    # =========================================================================
    "AUTH_REQUIRED": {
        "severity": "error",
        "category": "authentication",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "BasePamolaError",
    },
    "AUTH_INVALID_CREDENTIALS": {
        "severity": "error",
        "category": "authentication",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "BasePamolaError",
    },
    "AUTH_TOKEN_EXPIRED": {
        "severity": "error",
        "category": "authentication",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "BasePamolaError",
    },
    "AUTH_PERMISSION_DENIED": {
        "severity": "error",
        "category": "authentication",
        "retry_allowed": False,
        "user_facing": True,
        "exception_type": "BasePamolaError",
    },
}


def get_error_metadata(error_code: str) -> Dict[str, Any]:
    """
    Return metadata for an error code; falls back to sensible defaults.

    Parameters
    ----------
        error_code: Error code string (e.g., "DATA_LOAD_FAILED")

    Returns
    -------
        Dictionary with severity, category, retry_allowed, user_facing,
        exception_type. Returns safe defaults for unknown codes.
    """
    return ERROR_CODE_METADATA.get(
        error_code,
        {
            "severity": "error",
            "category": "unknown",
            "retry_allowed": False,
            "user_facing": False,
            "exception_type": None,
        },
    )