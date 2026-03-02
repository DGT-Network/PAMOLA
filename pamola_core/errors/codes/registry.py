"""
Error code registry for PAMOLA Core.

Centralized registry of all error codes with standardized naming convention:
<CATEGORY>_<SPECIFIC_ERROR> (e.g., DATA_LOAD_FAILED)
"""

from typing import ClassVar, List


class ErrorCode:
    """String-based error code registry with standardized naming."""

    # =========================================================================
    # DATA ERRORS (7 codes)
    # =========================================================================
    DATA_LOAD_FAILED: ClassVar[str] = "DATA_LOAD_FAILED"
    DATA_SOURCE_INVALID: ClassVar[str] = "DATA_SOURCE_INVALID"
    DATA_VALIDATION_ERROR: ClassVar[str] = "DATA_VALIDATION_ERROR"
    DATA_EMPTY: ClassVar[str] = "DATA_EMPTY"
    DATA_WRITE_FAILED: ClassVar[str] = "DATA_WRITE_FAILED"
    DATA_FRAME_PROCESSING_ERROR: ClassVar[str] = "DATA_FRAME_PROCESSING_ERROR"

    # =========================================================================
    # VALIDATION ERRORS - Field Validation (4 codes)
    # =========================================================================
    FIELD_NOT_FOUND: ClassVar[str] = "FIELD_NOT_FOUND"
    FIELD_TYPE_ERROR: ClassVar[str] = "FIELD_TYPE_ERROR"
    FIELD_VALUE_ERROR: ClassVar[str] = "FIELD_VALUE_ERROR"
    COLUMN_NOT_FOUND: ClassVar[str] = "COLUMN_NOT_FOUND"

    # =========================================================================
    # VALIDATION ERRORS - Parameters (6 codes)
    # =========================================================================
    PARAM_MISSING: ClassVar[str] = "PARAM_MISSING"
    PARAM_INVALID: ClassVar[str] = "PARAM_INVALID"
    PARAM_TYPE_ERROR: ClassVar[str] = "PARAM_TYPE_ERROR"
    STRATEGY_INVALID: ClassVar[str] = "STRATEGY_INVALID"
    MODE_INVALID: ClassVar[str] = "MODE_INVALID"
    NULL_STRATEGY_INVALID: ClassVar[str] = "NULL_STRATEGY_INVALID"

    # =========================================================================
    # VALIDATION ERRORS - Conditional & Range (4 codes)
    # =========================================================================
    VALIDATION_CONDITIONAL_FAILED: ClassVar[str] = "VALIDATION_CONDITIONAL_FAILED"
    VALIDATION_RANGE_FAILED: ClassVar[str] = "VALIDATION_RANGE_FAILED"
    VALIDATION_FORMAT_INVALID: ClassVar[str] = "VALIDATION_FORMAT_INVALID"
    VALIDATION_MARKER_FAILED: ClassVar[str] = "VALIDATION_MARKER_FAILED"

    # =========================================================================
    # VALIDATION ERRORS - Files (5 codes)
    # =========================================================================
    FILE_ERROR: ClassVar[str] = "FILE_ERROR"
    FILE_NOT_FOUND: ClassVar[str] = "FILE_NOT_FOUND"
    FILE_FORMAT_INVALID: ClassVar[str] = "FILE_FORMAT_INVALID"
    FILE_ACCESS_DENIED: ClassVar[str] = "FILE_ACCESS_DENIED"
    FILE_CORRUPTED: ClassVar[str] = "FILE_CORRUPTED"

    # =========================================================================
    # VALIDATION ERRORS - Multiple/Aggregated (1 code)
    # =========================================================================
    MULTIPLE_ERRORS: ClassVar[str] = "MULTIPLE_ERRORS"

    # =========================================================================
    # PROCESSING ERRORS (5 codes)
    # =========================================================================
    PROCESSING_FAILED: ClassVar[str] = "PROCESSING_FAILED"
    PROCESSING_BATCH_FAILED: ClassVar[str] = "PROCESSING_BATCH_FAILED"
    PROCESSING_CHUNK_FAILED: ClassVar[str] = "PROCESSING_CHUNK_FAILED"
    PROCESSING_TIMEOUT: ClassVar[str] = "PROCESSING_TIMEOUT"
    PROCESSING_INTERRUPTED: ClassVar[str] = "PROCESSING_INTERRUPTED"
    FEATURE_NOT_IMPLEMENTED: ClassVar[str] = "FEATURE_NOT_IMPLEMENTED"

    # =========================================================================
    # RESOURCE ERRORS (4 codes)
    # =========================================================================
    RESOURCE_MEMORY_EXCEEDED: ClassVar[str] = "RESOURCE_MEMORY_EXCEEDED"
    RESOURCE_DISK_FULL: ClassVar[str] = "RESOURCE_DISK_FULL"
    RESOURCE_CPU_THROTTLED: ClassVar[str] = "RESOURCE_CPU_THROTTLED"
    RESOURCE_NOT_FOUND: ClassVar[str] = "RESOURCE_NOT_FOUND"

    # =========================================================================
    # CACHE ERRORS (5 codes)
    # =========================================================================
    CACHE_READ_FAILED: ClassVar[str] = "CACHE_READ_FAILED"
    CACHE_WRITE_FAILED: ClassVar[str] = "CACHE_WRITE_FAILED"
    CACHE_KEY_INVALID: ClassVar[str] = "CACHE_KEY_INVALID"
    CACHE_EXPIRED: ClassVar[str] = "CACHE_EXPIRED"
    CACHE_CORRUPTED: ClassVar[str] = "CACHE_CORRUPTED"

    # =========================================================================
    # ARTIFACT ERRORS (4 codes)
    # =========================================================================
    ARTIFACT_VALIDATION_FAILED: ClassVar[str] = "ARTIFACT_VALIDATION_FAILED"
    ARTIFACT_NOT_FOUND: ClassVar[str] = "ARTIFACT_NOT_FOUND"
    ARTIFACT_WRITE_FAILED: ClassVar[str] = "ARTIFACT_WRITE_FAILED"
    ARTIFACT_CORRUPTED: ClassVar[str] = "ARTIFACT_CORRUPTED"

    # =========================================================================
    # VISUALIZATION ERRORS (3 codes)
    # =========================================================================
    VISUALIZATION_FAILED: ClassVar[str] = "VISUALIZATION_FAILED"
    VISUALIZATION_BACKEND_ERROR: ClassVar[str] = "VISUALIZATION_BACKEND_ERROR"
    VISUALIZATION_TIMEOUT: ClassVar[str] = "VISUALIZATION_TIMEOUT"

    # =========================================================================
    # CONFIGURATION ERRORS (4 codes)
    # =========================================================================
    CONFIG_INVALID: ClassVar[str] = "CONFIG_INVALID"
    CONFIG_MISSING: ClassVar[str] = "CONFIG_MISSING"
    CONFIG_PARSE_ERROR: ClassVar[str] = "CONFIG_PARSE_ERROR"
    CONFIG_SAVE_FAILED: ClassVar[str] = "CONFIG_SAVE_FAILED"

    # =========================================================================
    # CRYPTOGRAPHY & SECURITY ERRORS (18 codes)
    # =========================================================================
    # Encryption/Decryption (3 codes)
    ENCRYPTION_FAILED: ClassVar[str] = "ENCRYPTION_FAILED"
    ENCRYPTION_INIT_FAILED: ClassVar[str] = "ENCRYPTION_INIT_FAILED"
    DECRYPTION_FAILED: ClassVar[str] = "DECRYPTION_FAILED"

    # Key Management (7 codes)
    KEY_GENERATION_FAILED: ClassVar[str] = "KEY_GENERATION_FAILED"
    KEY_LOADING_FAILED: ClassVar[str] = "KEY_LOADING_FAILED"
    KEY_INVALID: ClassVar[str] = "KEY_INVALID"
    KEY_EXPIRED: ClassVar[str] = "KEY_EXPIRED"
    KEY_MASTER_ERROR: ClassVar[str] = "KEY_MASTER_ERROR"
    KEY_TASK_ERROR: ClassVar[str] = "KEY_TASK_ERROR"
    KEY_STORE_ERROR: ClassVar[str] = "KEY_STORE_ERROR"

    # Crypto Operations (5 codes)
    CRYPTO_ERROR: ClassVar[str] = "CRYPTO_ERROR"
    CRYPTO_PROVIDER_ERROR: ClassVar[str] = "CRYPTO_PROVIDER_ERROR"
    CRYPTO_MODE_ERROR: ClassVar[str] = "CRYPTO_MODE_ERROR"
    CRYPTO_FORMAT_ERROR: ClassVar[str] = "CRYPTO_FORMAT_ERROR"
    CRYPTO_ALGORITHM_UNSUPPORTED: ClassVar[str] = "CRYPTO_ALGORITHM_UNSUPPORTED"

    # Specialized Crypto (5 codes)
    PSEUDONYMIZATION_FAILED: ClassVar[str] = "PSEUDONYMIZATION_FAILED"
    HASH_COLLISION_DETECTED: ClassVar[str] = "HASH_COLLISION_DETECTED"
    DATA_REDACTION_FAILED: ClassVar[str] = "DATA_REDACTION_FAILED"
    CRYPTO_MIGRATION_FAILED: ClassVar[str] = "CRYPTO_MIGRATION_FAILED"
    CRYPTO_TOOL_ERROR: ClassVar[str] = "CRYPTO_TOOL_ERROR"

    # =========================================================================
    # TASK EXECUTION & OPERATIONS (17 codes)
    # =========================================================================
    # Task Lifecycle (4 codes)
    TASK_ERROR: ClassVar[str] = "TASK_ERROR"
    TASK_INIT_FAILED: ClassVar[str] = "TASK_INIT_FAILED"
    TASK_EXECUTION_FAILED: ClassVar[str] = "TASK_EXECUTION_FAILED"
    TASK_FINALIZATION_FAILED: ClassVar[str] = "TASK_FINALIZATION_FAILED"

    # Task Dependencies (3 codes)
    TASK_DEPENDENCY_ERROR: ClassVar[str] = "TASK_DEPENDENCY_ERROR"
    TASK_DEPENDENCY_MISSING: ClassVar[str] = "TASK_DEPENDENCY_MISSING"
    TASK_DEPENDENCY_FAILED: ClassVar[str] = "TASK_DEPENDENCY_FAILED"

    # =========================================================================
    # DEPENDENCY ERRORS (1 code)
    # =========================================================================
    DEPENDENCY_MISSING: ClassVar[str] = "DEPENDENCY_MISSING"

    # Task Management (2 codes)
    TASK_REGISTRY_ERROR: ClassVar[str] = "TASK_REGISTRY_ERROR"
    TASK_CONTEXT_ERROR: ClassVar[str] = "TASK_CONTEXT_ERROR"

    # Execution Management (2 codes)
    EXECUTION_ERROR: ClassVar[str] = "EXECUTION_ERROR"
    EXECUTION_LOG_ERROR: ClassVar[str] = "EXECUTION_LOG_ERROR"

    # State Management (3 codes)
    CHECKPOINT_ERROR: ClassVar[str] = "CHECKPOINT_ERROR"
    STATE_SERIALIZATION_FAILED: ClassVar[str] = "STATE_SERIALIZATION_FAILED"
    STATE_RESTORATION_FAILED: ClassVar[str] = "STATE_RESTORATION_FAILED"

    # Retry Management (2 codes)
    MAX_RETRIES_EXCEEDED: ClassVar[str] = "MAX_RETRIES_EXCEEDED"
    NON_RETRIABLE_ERROR: ClassVar[str] = "NON_RETRIABLE_ERROR"

    # Operations (1 code)
    OPERATION_ERROR: ClassVar[str] = "OPERATION_ERROR"

    # =========================================================================
    # FILE SYSTEM & PATH ERRORS (4 codes)
    # =========================================================================
    PATH_INVALID: ClassVar[str] = "PATH_INVALID"
    PATH_SECURITY_VIOLATION: ClassVar[str] = "PATH_SECURITY_VIOLATION"
    DIRECTORY_CREATE_FAILED: ClassVar[str] = "DIRECTORY_CREATE_FAILED"
    DIRECTORY_MANAGER_ERROR: ClassVar[str] = "DIRECTORY_MANAGER_ERROR"

    # =========================================================================
    # NLP & MACHINE LEARNING ERRORS (9 codes)
    # =========================================================================
    # NLP Operations (2 codes)
    NLP_ERROR: ClassVar[str] = "NLP_ERROR"
    NLP_PROMPT_INVALID: ClassVar[str] = "NLP_PROMPT_INVALID"

    # LLM Operations (4 codes)
    LLM_ERROR: ClassVar[str] = "LLM_ERROR"
    LLM_CONNECTION_FAILED: ClassVar[str] = "LLM_CONNECTION_FAILED"
    LLM_GENERATION_FAILED: ClassVar[str] = "LLM_GENERATION_FAILED"
    LLM_RESPONSE_INVALID: ClassVar[str] = "LLM_RESPONSE_INVALID"

    # Model Management (2 codes)
    MODEL_NOT_AVAILABLE: ClassVar[str] = "MODEL_NOT_AVAILABLE"
    MODEL_LOAD_FAILED: ClassVar[str] = "MODEL_LOAD_FAILED"

    # Language Support (1 code)
    LANGUAGE_UNSUPPORTED: ClassVar[str] = "LANGUAGE_UNSUPPORTED"

    # =========================================================================
    # DATA PROCESSING & PRIVACY (3 codes)
    # =========================================================================
    DATETIME_PARSE_FAILED: ClassVar[str] = "DATETIME_PARSE_FAILED"
    DATETIME_GENERALIZATION_FAILED: ClassVar[str] = "DATETIME_GENERALIZATION_FAILED"
    PRIVACY_INSUFFICIENT: ClassVar[str] = "PRIVACY_INSUFFICIENT"

    # =========================================================================
    # DATA MAPPING & GENERATION (3 codes)
    # =========================================================================
    MAPPING_ERROR: ClassVar[str] = "MAPPING_ERROR"
    MAPPING_STORAGE_ERROR: ClassVar[str] = "MAPPING_STORAGE_ERROR"
    FAKE_DATA_GENERATION_FAILED: ClassVar[str] = "FAKE_DATA_GENERATION_FAILED"

    # =========================================================================
    # REPORTING (1 code)
    # =========================================================================
    REPORTING_ERROR: ClassVar[str] = "REPORTING_ERROR"

    # =========================================================================
    # NETWORK ERRORS (4 codes)
    # =========================================================================
    NETWORK_CONNECTION_FAILED: ClassVar[str] = "NETWORK_CONNECTION_FAILED"
    NETWORK_TIMEOUT: ClassVar[str] = "NETWORK_TIMEOUT"
    NETWORK_DNS_FAILED: ClassVar[str] = "NETWORK_DNS_FAILED"
    NETWORK_SSL_ERROR: ClassVar[str] = "NETWORK_SSL_ERROR"

    # =========================================================================
    # AUTHENTICATION & AUTHORIZATION ERRORS (4 codes)
    # =========================================================================
    AUTH_REQUIRED: ClassVar[str] = "AUTH_REQUIRED"
    AUTH_INVALID_CREDENTIALS: ClassVar[str] = "AUTH_INVALID_CREDENTIALS"
    AUTH_TOKEN_EXPIRED: ClassVar[str] = "AUTH_TOKEN_EXPIRED"
    AUTH_PERMISSION_DENIED: ClassVar[str] = "AUTH_PERMISSION_DENIED"

    @classmethod
    def get_all_codes(cls) -> List[str]:
        """
        Get list of all error codes.

        Returns:
            Sorted list of all error code strings
        """
        codes = []
        for name, value in vars(cls).items():
            if not name.startswith("_") and isinstance(value, str) and value.isupper():
                codes.append(value)
        return sorted(list(set(codes)))

    @classmethod
    def get_codes_by_category(cls, category: str) -> List[str]:
        """
        Return codes whose prefix matches category.

        Args:
            category: Category name (case-insensitive)

        Returns:
            Sorted list of error codes in that category

        Example:
            >>> ErrorCode.get_codes_by_category("data")
            ['DATA_EMPTY', 'DATA_FRAME_PROCESSING_ERROR', 'DATA_LOAD_FAILED', ...]
        """
        prefix = f"{category.upper()}_"
        return sorted([code for code in cls.get_all_codes() if code.startswith(prefix)])

    @classmethod
    def is_valid_code(cls, code: str) -> bool:
        """
        Check if a code is defined.

        Args:
            code: Error code to validate

        Returns:
            True if code exists in registry
        """
        return code in cls.get_all_codes()

    @classmethod
    def validate_code(cls, code: str) -> None:
        """
        Raise InvalidParameterError if code is not defined.

        Args:
            code: Error code to validate

        Raises:
            InvalidParameterError: If code is not in registry
        """
        if not cls.is_valid_code(code):
            all_codes = cls.get_all_codes()
            sample = ", ".join(all_codes[:5])
            raise ValueError(
                f"Invalid error_code '{code}'. "
                f"Must be one of: {sample}... (total {len(all_codes)} defined)"
            )

    @classmethod
    def get_validation_codes(cls) -> List[str]:
        """
        Get all validation-related error codes.

        Returns:
            Sorted list of validation error codes

        Example:
            >>> codes = ErrorCode.get_validation_codes()
            >>> "FIELD_NOT_FOUND" in codes
            True
            >>> "DATA_LOAD_FAILED" in codes
            False
        """
        validation_prefixes = ["VALIDATION_", "FIELD_", "PARAM_", "FILE_", "COLUMN_"]
        all_codes = cls.get_all_codes()
        return sorted(
            [
                code
                for code in all_codes
                if any(code.startswith(prefix) for prefix in validation_prefixes)
            ]
        )

    @classmethod
    def get_retriable_codes(cls) -> List[str]:
        """
        Get all error codes that allow retry.

        Returns:
            Sorted list of retriable error codes

        Example:
            >>> codes = ErrorCode.get_retriable_codes()
            >>> "PROCESSING_TIMEOUT" in codes
            True
            >>> "PARAM_INVALID" in codes
            False
        """
        from pamola_core.errors.codes.metadata import get_error_metadata

        return sorted(
            [
                code
                for code in cls.get_all_codes()
                if get_error_metadata(code).get("retry_allowed", False)
            ]
        )

    @classmethod
    def get_user_facing_codes(cls) -> List[str]:
        """
        Get all error codes that should be shown to end users.

        Returns:
            Sorted list of user-facing error codes

        Example:
            >>> codes = ErrorCode.get_user_facing_codes()
            >>> "FILE_NOT_FOUND" in codes
            True
            >>> "ENCRYPTION_FAILED" in codes
            False
        """
        from pamola_core.errors.codes.metadata import get_error_metadata

        return sorted(
            [
                code
                for code in cls.get_all_codes()
                if get_error_metadata(code).get("user_facing", False)
            ]
        )

    @classmethod
    def is_user_facing(cls, code: str) -> bool:
        """
        Check if error code should be shown to end users.

        Args:
            code: Error code to check

        Returns:
            True if error should be shown to users, False otherwise

        Example:
            >>> ErrorCode.is_user_facing("FILE_NOT_FOUND")
            True
            >>> ErrorCode.is_user_facing("ENCRYPTION_FAILED")
            False
        """
        from pamola_core.errors.codes.metadata import get_error_metadata

        return get_error_metadata(code).get("user_facing", False)

    @classmethod
    def is_retriable(cls, code: str) -> bool:
        """
        Check if error code allows retry.

        Args:
            code: Error code to check

        Returns:
            True if retry is allowed for this error, False otherwise

        Example:
            >>> ErrorCode.is_retriable("NETWORK_TIMEOUT")
            True
            >>> ErrorCode.is_retriable("PARAM_INVALID")
            False
        """
        from pamola_core.errors.codes.metadata import get_error_metadata

        return get_error_metadata(code).get("retry_allowed", False)

    @classmethod
    def get_codes_by_severity(cls, severity: str) -> List[str]:
        """
        Get all error codes with specified severity level.

        Args:
            severity: Severity level (critical, error, warning, info, debug)

        Returns:
            Sorted list of error codes with that severity

        Example:
            >>> critical_codes = ErrorCode.get_codes_by_severity("critical")
            >>> "KEY_INVALID" in critical_codes
            True
        """
        from pamola_core.errors.codes.metadata import get_error_metadata

        return sorted(
            [
                code
                for code in cls.get_all_codes()
                if get_error_metadata(code).get("severity", "").lower()
                == severity.lower()
            ]
        )
