"""Cryptography and security exceptions."""

from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


@auto_exception(
    default_error_code=ErrorCode.ENCRYPTION_FAILED,
    message_params=["name", "reason"],
    detail_params=["name", "reason"],
)
class EncryptionError(BasePamolaError):
    """Errors related to encryption operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.DECRYPTION_FAILED,
    message_params=["name", "reason"],
    detail_params=["name", "reason"],
)
class DecryptionError(BasePamolaError):
    """Errors related to decryption operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.ENCRYPTION_INIT_FAILED,
    message_params=["reason"],
    detail_params=["reason"],
)
class EncryptionInitializationError(BasePamolaError):
    """Exception raised when encryption initialization fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.KEY_GENERATION_FAILED,
    message_params=["reason"],
    detail_params=["reason"],
)
class KeyGenerationError(BasePamolaError):
    """Exception raised when key generation fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.KEY_LOADING_FAILED,
    message_params=["path", "reason"],
    detail_params=["path", "reason"],
)
class KeyLoadingError(BasePamolaError):
    """Exception raised when key loading fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.KEY_MASTER_ERROR,
    message_params=["reason"],
    detail_params=["reason"],
)
class MasterKeyError(BasePamolaError):
    """Error related to master key operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.KEY_TASK_ERROR,
    message_params=["task_id", "reason"],
    detail_params=["task_id", "reason"],
)
class TaskKeyError(BasePamolaError):
    """Error related to task-specific key operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.KEY_INVALID,
    message_params=["reason"],
    detail_params=["reason"],
)
class CryptoKeyError(BasePamolaError):
    """Error related to cryptographic keys."""

    pass


@auto_exception(
    default_error_code=ErrorCode.KEY_STORE_ERROR,
    message_params=["reason"],
    detail_params=["reason"],
)
class KeyStoreError(BasePamolaError):
    """Errors interacting with keystore."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CRYPTO_ERROR,
    message_params=["reason"],
    detail_params=["reason"],
)
class CryptoError(BasePamolaError):
    """Errors related to cryptographic operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.PSEUDONYMIZATION_FAILED,
    message_params=["field_name", "reason"],
    detail_params=[
        "field_name",
        "reason",
        "pseudonym_type",
        "attempts",
        "max_attempts",
    ],
)
class PseudonymizationError(BasePamolaError):
    """Errors in pseudonymization workflow."""

    pass


@auto_exception(
    default_error_code=ErrorCode.HASH_COLLISION_DETECTED,
    message_params=["value", "existing_hash"],
    detail_params=["value", "existing_hash", "reason"],
)
class HashCollisionError(BasePamolaError):
    """Exception for hash collision detection."""

    pass


@auto_exception(
    default_error_code=ErrorCode.DATA_REDACTION_FAILED,
    message_params=["field_name", "reason"],
    detail_params=["field_name", "reason"],
)
class DataRedactionError(BasePamolaError):
    """Exception raised when data redaction fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CRYPTO_FORMAT_ERROR,
    message_params=["reason"],
    detail_params=["reason"],
)
class FormatError(BasePamolaError):
    """Error related to encrypted data format."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CRYPTO_PROVIDER_ERROR,
    message_params=["provider", "reason"],
    detail_params=["provider", "reason"],
)
class ProviderError(BasePamolaError):
    """Error related to crypto provider operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CRYPTO_MODE_ERROR,
    message_params=["mode", "reason"],
    detail_params=["mode", "reason"],
)
class ModeError(BasePamolaError):
    """Error related to crypto mode selection or detection."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CRYPTO_MIGRATION_FAILED,
    message_params=["reason"],
    detail_params=["reason"],
)
class LegacyMigrationError(BasePamolaError):
    """Error during migration of legacy encrypted data."""

    pass


@auto_exception(
    default_error_code=ErrorCode.CRYPTO_TOOL_ERROR,
    message_params=["tool_name", "reason"],
    detail_params=["tool_name", "reason"],
)
class AgeToolError(BasePamolaError):
    """Error when using the age CLI tool."""

    pass
