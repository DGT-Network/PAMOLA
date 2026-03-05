"""File system and path exceptions."""

from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


@auto_exception(
    default_error_code=ErrorCode.PATH_INVALID,
    message_params=["path", "reason"],
    detail_params=["path", "reason"],
)
class PathValidationError(BasePamolaError):
    """Exception raised when a path fails validation."""

    pass


@auto_exception(
    default_error_code=ErrorCode.PATH_SECURITY_VIOLATION,
    message_params=["path", "reason"],
    detail_params=["path", "reason"],
)
class PathSecurityError(BasePamolaError):
    """Path traversal or unsafe path detected."""

    pass


@auto_exception(
    default_error_code=ErrorCode.DIRECTORY_MANAGER_ERROR,
    message_params=["path", "reason"],
    detail_params=["path", "reason"],
)
class DirectoryManagerError(BasePamolaError):
    """Errors managing directories for tasks."""

    pass


@auto_exception(
    default_error_code=ErrorCode.DIRECTORY_CREATE_FAILED,
    message_params=["path", "reason"],
    detail_params=["path", "reason"],
)
class DirectoryCreationError(BasePamolaError):
    """Exception raised when directory creation fails."""

    pass
