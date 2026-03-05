"""NLP and machine learning exceptions."""

from typing import List, Optional

from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


@auto_exception(
    default_error_code=ErrorCode.NLP_ERROR,
    message_params=["operation", "reason"],
    detail_params=["operation", "reason"],
)
class NLPError(BasePamolaError):
    """Errors in NLP helper modules."""

    pass


@auto_exception(
    default_error_code=ErrorCode.NLP_PROMPT_INVALID,
    message_params=["reason"],
    detail_params=["reason"],
)
class PromptValidationError(BasePamolaError):
    """Prompt validation failures."""

    pass


@auto_exception(
    default_error_code=ErrorCode.LLM_ERROR,
    message_params=["reason"],
    detail_params=["reason"],
)
class LLMError(BasePamolaError):
    """Base exception for LLM operations."""

    pass


@auto_exception(
    default_error_code=ErrorCode.LLM_CONNECTION_FAILED,
    message_params=["service", "reason"],
    detail_params=["service", "reason"],
    parent_class=LLMError,
)
class LLMConnectionError(LLMError):
    """Exception raised when LLM connection fails."""

    pass


@auto_exception(
    default_error_code=ErrorCode.LLM_GENERATION_FAILED,
    message_params=["reason"],
    detail_params=["reason"],
    parent_class=LLMError,
)
class LLMGenerationError(LLMError):
    """Exception raised during text generation."""

    pass


@auto_exception(
    default_error_code=ErrorCode.LLM_RESPONSE_INVALID,
    message_params=["reason"],
    detail_params=["reason"],
    parent_class=LLMError,
)
class LLMResponseError(LLMError):
    """Exception raised when response is invalid."""

    pass


@auto_exception(
    default_error_code=ErrorCode.RESOURCE_NOT_FOUND,
    message_params=["resource_name", "reason"],
    detail_params=["resource_name", "reason"],
)
class ResourceNotFoundError(BasePamolaError):
    """Exception raised when a required resource is not found."""

    pass


@auto_exception(
    default_error_code=ErrorCode.MODEL_NOT_AVAILABLE,
    message_params=["model_name", "reason"],
    detail_params=["model_name", "reason"],
)
class ModelNotAvailableError(BasePamolaError):
    """Exception raised when a required model is not available."""

    pass


@auto_exception(
    default_error_code=ErrorCode.MODEL_LOAD_FAILED,
    message_params=["model_name", "reason"],
    detail_params=["model_name", "reason"],
)
class ModelLoadError(BasePamolaError):
    """Exception raised when a model fails to load."""

    pass


class UnsupportedLanguageError(BasePamolaError):
    """Exception raised when the requested language is not supported."""

    def __init__(
        self,
        language: Optional[str] = None,
        supported_languages: Optional[List[str]] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        language_display = language or "<unknown>"
        supported_display = (
            ", ".join(supported_languages) if supported_languages else "none"
        )

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.LANGUAGE_UNSUPPORTED,
                language=language_display,
                supported_languages=supported_display,
            ),
            error_code=ErrorCode.LANGUAGE_UNSUPPORTED,
            details={
                "language": language,
                "supported_languages": supported_languages,
            },
        )
