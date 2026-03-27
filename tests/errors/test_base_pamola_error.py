"""
Unit tests for pamola_core.errors.base.BasePamolaError class.

Tests cover:
- Exception creation with default and custom arguments
- Error code assignment and defaults
- Details dictionary population
- Exception raising and catching
- Structured output via to_dict()
- String representation
- Edge cases (empty messages, None values)

Run with: pytest tests/errors/test_base_pamola_error.py
"""

import pytest
from pamola_core.errors.base import BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode


class TestBasePamolaErrorCreation:
    """Test BasePamolaError creation with various argument combinations."""

    def test_create_with_message_only(self):
        """Create exception with message only."""
        error = BasePamolaError(message="Test error")
        assert error.message == "Test error"
        assert error.error_code == "BasePamolaError"  # Class name as default
        assert error.details == {}

    def test_create_with_message_and_error_code(self):
        """Create exception with message and error code."""
        error = BasePamolaError(
            message="Operation failed",
            error_code=ErrorCode.PROCESSING_FAILED
        )
        assert error.message == "Operation failed"
        assert error.error_code == ErrorCode.PROCESSING_FAILED
        assert error.details == {}

    def test_create_with_all_parameters(self):
        """Create exception with message, error code, and details."""
        details = {"operation": "data_transform", "record_count": 1000}
        error = BasePamolaError(
            message="Transform failed",
            error_code=ErrorCode.DATA_FRAME_PROCESSING_ERROR,
            details=details
        )
        assert error.message == "Transform failed"
        assert error.error_code == ErrorCode.DATA_FRAME_PROCESSING_ERROR
        assert error.details == details

    def test_create_with_empty_message(self):
        """Create exception with empty message string."""
        error = BasePamolaError(message="")
        assert error.message == ""
        assert error.error_code == "BasePamolaError"

    def test_create_with_none_error_code(self):
        """Create exception with None error code uses class name as default."""
        error = BasePamolaError(message="Test", error_code=None)
        assert error.error_code == "BasePamolaError"

    def test_create_with_none_details(self):
        """Create exception with None details creates empty dict."""
        error = BasePamolaError(message="Test", details=None)
        assert error.details == {}

    def test_create_with_empty_details_dict(self):
        """Create exception with empty details dict."""
        error = BasePamolaError(message="Test", details={})
        assert error.details == {}

    def test_custom_error_code_string(self):
        """Create exception with custom error code string."""
        custom_code = "CUSTOM_ERROR_CODE"
        error = BasePamolaError(
            message="Custom error",
            error_code=custom_code
        )
        assert error.error_code == custom_code


class TestBasePamolaErrorRaisingAndCatching:
    """Test raising and catching BasePamolaError exceptions."""

    def test_raise_and_catch_base_error(self):
        """Raise and catch BasePamolaError."""
        with pytest.raises(BasePamolaError) as exc_info:
            raise BasePamolaError(
                message="Test error",
                error_code=ErrorCode.PROCESSING_FAILED
            )
        assert exc_info.value.message == "Test error"
        assert exc_info.value.error_code == ErrorCode.PROCESSING_FAILED

    def test_catch_as_exception(self):
        """Catch BasePamolaError as standard Exception."""
        with pytest.raises(Exception) as exc_info:
            raise BasePamolaError(message="Generic error")
        assert isinstance(exc_info.value, BasePamolaError)

    def test_exception_str_representation(self):
        """String representation of exception contains message."""
        error = BasePamolaError(message="Test error message")
        assert str(error) == "Test error message"

    def test_raise_with_details(self):
        """Raise exception with details and verify details are preserved."""
        details = {"file": "data.csv", "reason": "not found"}
        error = BasePamolaError(
            message="File load failed",
            error_code=ErrorCode.FILE_NOT_FOUND,
            details=details
        )
        with pytest.raises(BasePamolaError) as exc_info:
            raise error
        assert exc_info.value.details == details

    def test_exception_is_instance_of_exception_class(self):
        """BasePamolaError is instance of Exception."""
        error = BasePamolaError(message="Test")
        assert isinstance(error, Exception)


class TestBasePamolaErrorToDict:
    """Test to_dict() method for structured output."""

    def test_to_dict_basic(self):
        """to_dict() returns structured dict with basic fields."""
        error = BasePamolaError(
            message="Test error",
            error_code=ErrorCode.DATA_LOAD_FAILED
        )
        result = error.to_dict()
        assert result["error_type"] == "BasePamolaError"
        assert result["message"] == "Test error"
        assert result["error_code"] == ErrorCode.DATA_LOAD_FAILED
        assert result["details"] == {}

    def test_to_dict_includes_metadata(self):
        """to_dict() includes metadata from ErrorCode registry."""
        error = BasePamolaError(
            message="Load failed",
            error_code=ErrorCode.DATA_LOAD_FAILED
        )
        result = error.to_dict()
        # Verify metadata fields are present
        assert "severity" in result
        assert "category" in result
        assert "retry_allowed" in result
        assert "user_facing" in result

    def test_to_dict_with_details(self):
        """to_dict() includes provided details."""
        details = {"source": "database", "reason": "connection timeout"}
        error = BasePamolaError(
            message="Connection failed",
            error_code=ErrorCode.DATA_LOAD_FAILED,
            details=details
        )
        result = error.to_dict()
        assert result["details"] == details

    def test_to_dict_unknown_error_code(self):
        """to_dict() handles unknown error code gracefully with default metadata."""
        error = BasePamolaError(
            message="Unknown error",
            error_code="NONEXISTENT_CODE"
        )
        result = error.to_dict()
        assert result["error_code"] == "NONEXISTENT_CODE"
        # get_error_metadata returns safe defaults for unknown codes ("error" severity)
        assert result.get("severity") == "error"

    def test_to_dict_with_complex_details(self):
        """to_dict() preserves complex nested details."""
        details = {
            "file": "data.csv",
            "context": {"line": 42, "column": 10},
            "suggestions": ["check file format", "verify encoding"]
        }
        error = BasePamolaError(
            message="Parse error",
            error_code=ErrorCode.DATA_VALIDATION_ERROR,
            details=details
        )
        result = error.to_dict()
        assert result["details"] == details


class TestBasePamolaErrorRepr:
    """Test __repr__ method for debugging."""

    def test_repr_format(self):
        """__repr__ returns formatted string with code and message preview."""
        error = BasePamolaError(
            message="This is a very long error message that exceeds fifty characters",
            error_code=ErrorCode.PROCESSING_FAILED
        )
        repr_str = repr(error)
        assert "BasePamolaError" in repr_str
        assert ErrorCode.PROCESSING_FAILED in repr_str
        assert "This is a very long error m" in repr_str  # Preview of message
        assert "..." in repr_str

    def test_repr_short_message(self):
        """__repr__ with short message doesn't truncate."""
        error = BasePamolaError(
            message="Short",
            error_code=ErrorCode.PARAM_INVALID
        )
        repr_str = repr(error)
        assert "BasePamolaError" in repr_str
        assert ErrorCode.PARAM_INVALID in repr_str
        assert "Short" in repr_str

    def test_repr_with_custom_code(self):
        """__repr__ works with custom error code."""
        error = BasePamolaError(
            message="Custom error",
            error_code="MY_CUSTOM_CODE"
        )
        repr_str = repr(error)
        assert "MY_CUSTOM_CODE" in repr_str


class TestBasePamolaErrorInheritance:
    """Test creating custom exception classes inheriting from BasePamolaError."""

    def test_custom_exception_class(self):
        """Create custom exception by inheriting from BasePamolaError."""
        class CustomError(BasePamolaError):
            """Custom error for testing."""
            pass

        error = CustomError(
            message="Custom error message",
            error_code="CUSTOM_CODE"
        )
        assert isinstance(error, BasePamolaError)
        assert isinstance(error, Exception)
        assert error.message == "Custom error message"

    def test_custom_exception_with_overridden_init(self):
        """Custom exception with custom __init__ method."""
        class DataError(BasePamolaError):
            """Error for data operations."""

            def __init__(self, source: str, reason: str):
                super().__init__(
                    message=f"Data error in {source}: {reason}",
                    error_code=ErrorCode.DATA_LOAD_FAILED,
                    details={"source": source, "reason": reason}
                )

        error = DataError(source="database", reason="timeout")
        assert error.message == "Data error in database: timeout"
        assert error.details["source"] == "database"


class TestBasePamolaErrorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_message(self):
        """Exception with very long message."""
        long_message = "x" * 10000
        error = BasePamolaError(message=long_message)
        assert error.message == long_message
        assert len(error.message) == 10000

    def test_special_characters_in_message(self):
        """Exception message with special characters."""
        special_message = "Error: \n\t\r\"'\\{}<>[]@#$%^&*()"
        error = BasePamolaError(message=special_message)
        assert error.message == special_message

    def test_unicode_in_message(self):
        """Exception message with unicode characters."""
        unicode_message = "错误: Erreur: エラー: Ошибка"
        error = BasePamolaError(message=unicode_message)
        assert error.message == unicode_message

    def test_details_with_none_values(self):
        """Details dict with None values."""
        details = {
            "key1": None,
            "key2": "value",
            "key3": None
        }
        error = BasePamolaError(message="Test", details=details)
        assert error.details == details
        assert error.details["key1"] is None

    def test_error_code_case_sensitivity(self):
        """Error code is case sensitive."""
        error1 = BasePamolaError(
            message="Test",
            error_code="ERROR_CODE"
        )
        error2 = BasePamolaError(
            message="Test",
            error_code="error_code"
        )
        assert error1.error_code != error2.error_code

    def test_details_modification_after_creation(self):
        """Modifying details after exception creation."""
        details = {"key": "value"}
        error = BasePamolaError(message="Test", details=details)
        error.details["new_key"] = "new_value"
        assert error.details["new_key"] == "new_value"

    def test_multiple_error_codes(self):
        """Multiple errors with different codes."""
        error1 = BasePamolaError(
            message="Error 1",
            error_code=ErrorCode.DATA_LOAD_FAILED
        )
        error2 = BasePamolaError(
            message="Error 2",
            error_code=ErrorCode.PARAM_INVALID
        )
        assert error1.error_code != error2.error_code
        assert error1.message != error2.message


class TestBasePamolaErrorWithAllErrorCodes:
    """Test BasePamolaError with various real error codes from ErrorCode registry."""

    @pytest.mark.parametrize("error_code", [
        ErrorCode.DATA_LOAD_FAILED,
        ErrorCode.PARAM_INVALID,
        ErrorCode.FILE_NOT_FOUND,
        ErrorCode.PROCESSING_FAILED,
        ErrorCode.VALIDATION_FORMAT_INVALID,
    ])
    def test_with_various_error_codes(self, error_code):
        """Create and verify exception with different error codes."""
        error = BasePamolaError(
            message=f"Error with {error_code}",
            error_code=error_code
        )
        assert error.error_code == error_code
        result = error.to_dict()
        assert result["error_code"] == error_code
        # Verify metadata is populated
        assert result.get("severity") is not None
        assert result.get("category") is not None
