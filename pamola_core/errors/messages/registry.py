"""
Error message registry for PAMOLA Core.

All message templates live in ErrorMessages for centralized management.
"""

from typing import ClassVar, Dict, List, Tuple


class ErrorMessages:
    """
    Technical error message templates for library consumers.

    Design Philosophy:
    - Messages are technical and detailed (for developers)
    - Include all relevant context (paths, values, constraints)
    - Actionable - help developers understand what went wrong
    - Library users can wrap/format these for end users if needed
    """

    # =========================================================================
    # DATA ERRORS (7 templates)
    # =========================================================================
    DATA_LOAD_FAILED: ClassVar[str] = (
        "Failed to load data from source '{source}': {reason}"
    )

    DATA_SOURCE_INVALID: ClassVar[str] = (
        "Data source '{source}' is invalid or not supported: {reason}"
    )

    DATA_VALIDATION_ERROR: ClassVar[str] = (
        "Data validation failed for '{context}': {reason}"
    )

    DATA_EMPTY: ClassVar[str] = (
        "Dataset is empty; cannot perform operation '{operation}'"
    )

    DATA_WRITE_FAILED: ClassVar[str] = (
        "Failed to write data to '{destination}': {reason}"
    )

    DATA_FRAME_PROCESSING_ERROR: ClassVar[str] = (
        "DataFrame processing failed for operation '{operation}': {reason}"
    )

    # =========================================================================
    # VALIDATION ERRORS - Fields (4 templates)
    # =========================================================================
    FIELD_NOT_FOUND: ClassVar[str] = (
        "Field '{field_name}' not found in data. "
        "Available fields: {available_fields}"
    )

    FIELD_TYPE_ERROR: ClassVar[str] = (
        "Field '{field_name}' has invalid type. "
        "Expected: {expected}, actual: {actual}"
    )

    FIELD_VALUE_ERROR: ClassVar[str] = (
        "Invalid value {value} for field '{field_name}': {reason}"
    )

    COLUMN_NOT_FOUND: ClassVar[str] = (
        "Column '{column_name}' not found in DataFrame. "
        "Available columns: {available_columns}"
    )

    # =========================================================================
    # VALIDATION ERRORS - Parameters (6 templates)
    # =========================================================================
    PARAM_MISSING: ClassVar[str] = (
        "Missing required parameter '{param_name}' for operation '{operation}'"
    )

    PARAM_INVALID: ClassVar[str] = (
        "Invalid value '{value}' for parameter '{param_name}'. "
        "Constraint: {constraint}"
    )

    PARAM_TYPE_ERROR: ClassVar[str] = (
        "Parameter '{param_name}' has wrong type. "
        "Expected: {expected_type}, actual: {actual_type}"
    )

    STRATEGY_INVALID: ClassVar[str] = (
        "Strategy '{strategy}' not supported for '{operation_type}'. "
        "Valid strategies: {valid_strategies}"
    )

    MODE_INVALID: ClassVar[str] = (
        "Operation mode '{mode}' not supported for '{operation_type}'"
    )

    NULL_STRATEGY_INVALID: ClassVar[str] = (
        "Invalid null handling strategy '{strategy}'. "
        "Valid strategies: {valid_strategies}"
    )

    # =========================================================================
    # VALIDATION ERRORS - Conditional & Range (4 templates)
    # =========================================================================
    VALIDATION_CONDITIONAL_FAILED: ClassVar[str] = (
        "Conditional validation failed for field '{field_name}': "
        "condition '{condition}' not satisfied. Reason: {reason}"
    )

    VALIDATION_RANGE_FAILED: ClassVar[str] = (
        "Value {value} for field '{field_name}' out of range. "
        "Expected: {min_value} to {max_value}"
    )

    VALIDATION_FORMAT_INVALID: ClassVar[str] = (
        "Field '{field_name}' has invalid format. "
        "Expected: {expected_format}, received: {actual_format}"
    )

    VALIDATION_MARKER_FAILED: ClassVar[str] = (
        "Marker validation failed for '{marker_name}': {reason}"
    )

    # =========================================================================
    # VALIDATION ERRORS - Files (5 templates)
    # =========================================================================
    FILE_ERROR: ClassVar[str] = "File operation failed for '{file_path}': {reason}"

    FILE_NOT_FOUND: ClassVar[str] = "File not found: '{file_path}'"

    FILE_FORMAT_INVALID: ClassVar[str] = (
        "File '{file_path}' has invalid format. "
        "Expected: {expected_format}, detected: {actual_format}"
    )

    FILE_ACCESS_DENIED: ClassVar[str] = "Access denied to file '{file_path}': {reason}"

    FILE_CORRUPTED: ClassVar[str] = (
        "File '{file_path}' appears to be corrupted: {reason}"
    )

    # =========================================================================
    # VALIDATION ERRORS - Multiple (1 template)
    # =========================================================================
    MULTIPLE_ERRORS: ClassVar[str] = (
        "Multiple validation errors ({error_count}). See details for complete list"
    )

    # =========================================================================
    # PROCESSING ERRORS (6 templates)
    # =========================================================================
    PROCESSING_FAILED: ClassVar[str] = (
        "Processing failed for operation '{operation}': {reason}"
    )

    PROCESSING_BATCH_FAILED: ClassVar[str] = (
        "Batch processing failed at batch #{batch_index}: {reason}"
    )

    PROCESSING_CHUNK_FAILED: ClassVar[str] = (
        "Chunk processing failed at position {chunk_index}: {reason}"
    )

    PROCESSING_TIMEOUT: ClassVar[str] = (
        "Operation '{operation}' exceeded timeout of {timeout}s"
    )

    PROCESSING_INTERRUPTED: ClassVar[str] = (
        "Processing interrupted for operation '{operation}': {reason}"
    )

    FEATURE_NOT_IMPLEMENTED: ClassVar[str] = (
        "Feature '{feature}' is not implemented: {reason}"
    )

    # =========================================================================
    # RESOURCE ERRORS (4 templates)
    # =========================================================================
    RESOURCE_MEMORY_EXCEEDED: ClassVar[str] = (
        "Insufficient memory for operation '{operation}'. "
        "Used: {memory_used}, Available: {memory_available}. "
        "Consider: reduce chunk_size, enable optimize_memory, or process in batches"
    )

    RESOURCE_DISK_FULL: ClassVar[str] = (
        "Insufficient disk space for operation '{operation}'. "
        "Required: {space_required}, Available: {space_available}"
    )

    RESOURCE_CPU_THROTTLED: ClassVar[str] = (
        "CPU throttled during operation '{operation}': {reason}"
    )

    RESOURCE_NOT_FOUND: ClassVar[str] = "Resource '{resource_name}' not found: {reason}"

    # =========================================================================
    # CACHE ERRORS (5 templates)
    # =========================================================================
    CACHE_READ_FAILED: ClassVar[str] = (
        "Failed to read cache with key '{cache_key}': {reason}"
    )

    CACHE_WRITE_FAILED: ClassVar[str] = (
        "Failed to write cache with key '{cache_key}': {reason}"
    )

    CACHE_KEY_INVALID: ClassVar[str] = "Cache key '{cache_key}' is invalid: {reason}"

    CACHE_EXPIRED: ClassVar[str] = "Cached data for key '{cache_key}' has expired"

    CACHE_CORRUPTED: ClassVar[str] = (
        "Cached data for key '{cache_key}' is corrupted or invalid"
    )

    # =========================================================================
    # ARTIFACT ERRORS (4 templates)
    # =========================================================================
    ARTIFACT_VALIDATION_FAILED: ClassVar[str] = (
        "Artifact validation failed at '{path}': {reason}"
    )

    ARTIFACT_NOT_FOUND: ClassVar[str] = "Artifact not found at path '{path}'"

    ARTIFACT_WRITE_FAILED: ClassVar[str] = (
        "Failed to write artifact at '{path}': {reason}"
    )

    ARTIFACT_CORRUPTED: ClassVar[str] = (
        "Artifact at '{path}' appears corrupted or has invalid checksum"
    )

    # =========================================================================
    # VISUALIZATION ERRORS (3 templates)
    # =========================================================================
    VISUALIZATION_FAILED: ClassVar[str] = (
        "Failed to generate visualization '{name}': {reason}"
    )

    VISUALIZATION_BACKEND_ERROR: ClassVar[str] = "Visualization backend error: {reason}"

    VISUALIZATION_TIMEOUT: ClassVar[str] = (
        "Visualization '{name}' exceeded timeout of {timeout}s"
    )

    # =========================================================================
    # CONFIGURATION ERRORS (4 templates)
    # =========================================================================
    CONFIG_INVALID: ClassVar[str] = "Invalid configuration: {reason}"

    CONFIG_MISSING: ClassVar[str] = "Missing required configuration: {key}"

    CONFIG_PARSE_ERROR: ClassVar[str] = (
        "Failed to parse configuration file '{path}': {reason}"
    )

    CONFIG_SAVE_FAILED: ClassVar[str] = (
        "Failed to save configuration to '{path}': {reason}"
    )

    # =========================================================================
    # CRYPTOGRAPHY & SECURITY ERRORS (18 templates)
    # =========================================================================
    ENCRYPTION_FAILED: ClassVar[str] = (
        "Encryption failed for resource '{name}': {reason}"
    )

    ENCRYPTION_INIT_FAILED: ClassVar[str] = "Encryption initialization failed: {reason}"

    DECRYPTION_FAILED: ClassVar[str] = (
        "Decryption failed for resource '{name}': {reason}"
    )

    KEY_GENERATION_FAILED: ClassVar[str] = "Key generation failed: {reason}"

    KEY_LOADING_FAILED: ClassVar[str] = "Failed to load key from '{path}': {reason}"

    KEY_INVALID: ClassVar[str] = "Encryption key is invalid or missing: {reason}"

    KEY_EXPIRED: ClassVar[str] = (
        "Encryption key has expired. Expiration: {expiration_date}"
    )

    KEY_MASTER_ERROR: ClassVar[str] = "Master key operation failed: {reason}"

    KEY_TASK_ERROR: ClassVar[str] = (
        "Task key operation failed for task '{task_id}': {reason}"
    )

    KEY_STORE_ERROR: ClassVar[str] = "Key store operation failed: {reason}"

    CRYPTO_ERROR: ClassVar[str] = "Cryptographic operation failed: {reason}"

    CRYPTO_PROVIDER_ERROR: ClassVar[str] = (
        "Crypto provider '{provider}' error: {reason}"
    )

    CRYPTO_MODE_ERROR: ClassVar[str] = "Crypto mode '{mode}' error: {reason}"

    CRYPTO_FORMAT_ERROR: ClassVar[str] = "Encrypted data format error: {reason}"

    CRYPTO_ALGORITHM_UNSUPPORTED: ClassVar[str] = (
        "Algorithm '{algorithm}' is not supported. "
        "Supported algorithms: {supported_algorithms}"
    )

    PSEUDONYMIZATION_FAILED: ClassVar[str] = (
        "Pseudonymization failed for field '{field_name}': {reason}"
    )

    HASH_COLLISION_DETECTED: ClassVar[str] = (
        "Hash collision detected for value '{value}' "
        "with existing hash '{existing_hash}'"
    )

    DATA_REDACTION_FAILED: ClassVar[str] = (
        "Data redaction failed for field '{field_name}': {reason}"
    )

    CRYPTO_MIGRATION_FAILED: ClassVar[str] = "Crypto migration failed: {reason}"

    CRYPTO_TOOL_ERROR: ClassVar[str] = "Crypto tool '{tool_name}' error: {reason}"

    # =========================================================================
    # TASK EXECUTION & OPERATIONS (17 templates)
    # =========================================================================
    TASK_ERROR: ClassVar[str] = "Task '{task_name}' failed: {reason}"

    TASK_INIT_FAILED: ClassVar[str] = (
        "Task '{task_name}' initialization failed: {reason}"
    )

    TASK_EXECUTION_FAILED: ClassVar[str] = (
        "Task '{task_name}' execution failed: {reason}"
    )

    TASK_FINALIZATION_FAILED: ClassVar[str] = (
        "Task '{task_name}' finalization failed: {reason}"
    )

    TASK_DEPENDENCY_ERROR: ClassVar[str] = (
        "Task '{task_name}' dependency error: {reason}"
    )

    TASK_DEPENDENCY_MISSING: ClassVar[str] = (
        "Task '{task_name}' missing required dependency: '{dependency_name}'"
    )

    TASK_DEPENDENCY_FAILED: ClassVar[str] = (
        "Task '{task_name}' failed because dependency '{dependency_name}' failed"
    )

    DEPENDENCY_MISSING: ClassVar[str] = (
        "Required dependency '{dependency}' is missing or unavailable: {reason}"
    )

    TASK_REGISTRY_ERROR: ClassVar[str] = (
        "Task registry error for task '{task_name}': {reason}"
    )

    TASK_CONTEXT_ERROR: ClassVar[str] = (
        "Task context error for task '{task_name}': {reason}"
    )

    EXECUTION_ERROR: ClassVar[str] = "Execution error for '{operation}': {reason}"

    EXECUTION_LOG_ERROR: ClassVar[str] = (
        "Execution log error for task '{task_name}': {reason}"
    )

    CHECKPOINT_ERROR: ClassVar[str] = (
        "Checkpoint error for task '{task_name}': {reason}"
    )

    STATE_SERIALIZATION_FAILED: ClassVar[str] = (
        "Failed to serialize state for task '{task_name}': {reason}"
    )

    STATE_RESTORATION_FAILED: ClassVar[str] = (
        "Failed to restore state for task '{task_name}': {reason}"
    )

    MAX_RETRIES_EXCEEDED: ClassVar[str] = (
        "Maximum retries ({max_retries}) exceeded for operation '{operation}'"
    )

    NON_RETRIABLE_ERROR: ClassVar[str] = (
        "Non-retriable error occurred in operation '{operation}': {reason}"
    )

    OPERATION_ERROR: ClassVar[str] = "Operation '{operation}' failed: {reason}"

    # =========================================================================
    # FILE SYSTEM & PATH ERRORS (4 templates)
    # =========================================================================
    PATH_INVALID: ClassVar[str] = "Path '{path}' is invalid: {reason}"

    PATH_SECURITY_VIOLATION: ClassVar[str] = (
        "Path '{path}' violates security constraints: {reason}"
    )

    DIRECTORY_CREATE_FAILED: ClassVar[str] = (
        "Failed to create directory '{path}': {reason}"
    )

    DIRECTORY_MANAGER_ERROR: ClassVar[str] = (
        "Directory manager error for '{path}': {reason}"
    )

    # =========================================================================
    # NLP & MACHINE LEARNING ERRORS (9 templates)
    # =========================================================================
    NLP_ERROR: ClassVar[str] = "NLP operation '{operation}' failed: {reason}"

    NLP_PROMPT_INVALID: ClassVar[str] = "Prompt validation failed: {reason}"

    LLM_ERROR: ClassVar[str] = "LLM operation failed: {reason}"

    LLM_CONNECTION_FAILED: ClassVar[str] = (
        "Failed to connect to LLM service '{service}': {reason}"
    )

    LLM_GENERATION_FAILED: ClassVar[str] = "LLM text generation failed: {reason}"

    LLM_RESPONSE_INVALID: ClassVar[str] = "LLM response is invalid: {reason}"

    MODEL_NOT_AVAILABLE: ClassVar[str] = (
        "Model '{model_name}' is not available: {reason}"
    )

    MODEL_LOAD_FAILED: ClassVar[str] = "Failed to load model '{model_name}': {reason}"

    LANGUAGE_UNSUPPORTED: ClassVar[str] = (
        "Language '{language}' is not supported. "
        "Supported languages: {supported_languages}"
    )

    # =========================================================================
    # DATA PROCESSING & PRIVACY (3 templates)
    # =========================================================================
    DATETIME_PARSE_FAILED: ClassVar[str] = (
        "Failed to parse datetime value '{value}' with format '{format}': {reason}"
    )

    DATETIME_GENERALIZATION_FAILED: ClassVar[str] = (
        "DateTime generalization failed for field '{field_name}': {reason}"
    )

    PRIVACY_INSUFFICIENT: ClassVar[str] = (
        "Privacy constraints not satisfied. "
        "Required: {required_level}, Achieved: {achieved_level}"
    )

    # =========================================================================
    # DATA MAPPING & GENERATION (3 templates)
    # =========================================================================
    MAPPING_ERROR: ClassVar[str] = "Mapping operation failed for '{context}': {reason}"

    MAPPING_STORAGE_ERROR: ClassVar[str] = "Mapping storage operation failed: {reason}"

    FAKE_DATA_GENERATION_FAILED: ClassVar[str] = (
        "Fake data generation failed for field '{field_name}': {reason}"
    )

    # =========================================================================
    # REPORTING (1 template)
    # =========================================================================
    REPORTING_ERROR: ClassVar[str] = (
        "Reporting operation failed for '{report_name}': {reason}"
    )

    # =========================================================================
    # NETWORK ERRORS (4 templates)
    # =========================================================================
    NETWORK_CONNECTION_FAILED: ClassVar[str] = (
        "Failed to connect to '{host}:{port}': {reason}"
    )

    NETWORK_TIMEOUT: ClassVar[str] = (
        "Network request to '{url}' timed out after {timeout}s"
    )

    NETWORK_DNS_FAILED: ClassVar[str] = (
        "DNS resolution failed for '{hostname}': {reason}"
    )

    NETWORK_SSL_ERROR: ClassVar[str] = "SSL error for '{url}': {reason}"

    # =========================================================================
    # AUTHENTICATION & AUTHORIZATION ERRORS (4 templates)
    # =========================================================================
    AUTH_REQUIRED: ClassVar[str] = "Authentication required for operation '{operation}'"

    AUTH_INVALID_CREDENTIALS: ClassVar[str] = (
        "Invalid credentials for user '{username}'"
    )

    AUTH_TOKEN_EXPIRED: ClassVar[str] = (
        "Authentication token expired. Expiration: {expiration}"
    )

    AUTH_PERMISSION_DENIED: ClassVar[str] = (
        "Permission denied for operation '{operation}'. "
        "Required permission: {required_permission}"
    )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def format(template_name: str, **kwargs) -> str:
        """
        Format a message template by name with given parameters.

        Parameters
        ----------
            template_name: Attribute name on ErrorMessages (e.g., "DATA_LOAD_FAILED")
            **kwargs: Template parameters

        Returns
        -------
            Formatted message string, or descriptive error if formatting fails
        """
        from pamola_core.errors.messages.utils import format_message

        template_str = getattr(ErrorMessages, template_name, None)
        if template_str is None or not isinstance(template_str, str):
            return f"Unknown error template: {template_name}"
        return format_message(template_str, **kwargs)

    @classmethod
    def get_all_templates(cls) -> Dict[str, str]:
        """
        Get all available message templates.

        Returns
        -------
            Dictionary of {template_name: template_string}
        """
        return {
            name: value
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, str) and name.isupper()
        }

    @classmethod
    def validate_template_params(
        cls, template: str, **kwargs
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all required parameters are provided for a template.

        Parameters
        ----------
            template: Template name (e.g., "DATA_LOAD_FAILED")
            **kwargs: Parameters to validate

        Returns
        -------
            Tuple of (is_valid, list_of_missing_params)
        """
        from pamola_core.errors.messages.utils import validate_template_params_str

        template_str = getattr(cls, template, None)
        if template_str is None or not isinstance(template_str, str):
            return False, [f"Template '{template}' not found"]
        return validate_template_params_str(template_str, **kwargs)