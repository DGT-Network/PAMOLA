"""
Unit tests for pamola_core.errors.base.auto_exception decorator.

Tests cover:
- Decorator-generated __init__ methods
- Message parameter substitution from ErrorMessages
- Details dict population
- Custom message builders
- Parent class initialization
- Error code handling
- Edge cases and error scenarios

Run with: pytest tests/errors/test_auto_exception.py
"""

import pytest
from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode


class TestAutoExceptionBasic:
    """Test basic auto_exception decorator functionality."""

    def test_auto_exception_creates_init(self):
        """Decorator creates __init__ method on decorated class."""
        @auto_exception(default_error_code=ErrorCode.PROCESSING_FAILED)
        class TestError(BasePamolaError):
            """Test error class."""
            pass

        # Should be able to instantiate without errors
        error = TestError(message="Test error")
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.PROCESSING_FAILED

    def test_auto_exception_inherits_from_base(self):
        """Decorated class inherits from BasePamolaError."""
        @auto_exception(default_error_code=ErrorCode.DATA_LOAD_FAILED)
        class DataError(BasePamolaError):
            """Error for data loading."""
            pass

        error = DataError(message="Load failed")
        assert isinstance(error, BasePamolaError)
        assert isinstance(error, Exception)

    def test_auto_exception_with_default_error_code(self):
        """Decorator sets default error code for all instances."""
        @auto_exception(default_error_code=ErrorCode.PARAM_INVALID)
        class ParamError(BasePamolaError):
            """Parameter error."""
            pass

        error1 = ParamError(message="Error 1")
        error2 = ParamError(message="Error 2")
        assert error1.error_code == ErrorCode.PARAM_INVALID
        assert error2.error_code == ErrorCode.PARAM_INVALID

    def test_auto_exception_override_error_code(self):
        """Override default error code when creating instance."""
        @auto_exception(default_error_code=ErrorCode.PARAM_INVALID)
        class ParamError(BasePamolaError):
            """Parameter error."""
            pass

        error = ParamError(
            message="Custom error",
            error_code=ErrorCode.PARAM_MISSING
        )
        assert error.error_code == ErrorCode.PARAM_MISSING

    def test_auto_exception_with_explicit_message(self):
        """Decorator respects explicitly provided message."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            message_params=["operation"]
        )
        class ProcessError(BasePamolaError):
            """Processing error."""
            pass

        error = ProcessError(message="Custom message", operation="transform")
        assert error.message == "Custom message"


class TestAutoExceptionMessageParams:
    """Test auto_exception message parameter handling."""

    def test_auto_exception_with_message_params(self):
        """Decorator builds message from ErrorMessages template with params."""
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"]
        )
        class DataLoadError(BasePamolaError):
            """Data load error."""
            pass

        error = DataLoadError(source="database", reason="connection timeout")
        assert "database" in error.message
        assert "connection timeout" in error.message
        assert error.error_code == ErrorCode.DATA_LOAD_FAILED

    def test_auto_exception_message_params_with_missing_param(self):
        """Missing message param uses '<unknown>' placeholder."""
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"]
        )
        class DataLoadError(BasePamolaError):
            """Data load error."""
            pass

        error = DataLoadError(source="database")  # missing 'reason'
        assert "database" in error.message
        assert "<unknown>" in error.message

    def test_auto_exception_no_message_params_fallback(self):
        """With no message_params, uses 'reason' param or 'Unknown error'."""
        @auto_exception(default_error_code=ErrorCode.PROCESSING_FAILED)
        class SimpleError(BasePamolaError):
            """Simple error."""
            pass

        # With reason param
        error1 = SimpleError(reason="Something went wrong")
        assert error1.message == "Something went wrong"

        # Without reason param
        error2 = SimpleError()
        assert error2.message == "Unknown error"

    def test_auto_exception_empty_message_params_list(self):
        """Empty message_params list uses reason param."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            message_params=[]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error = Error(reason="Custom reason")
        assert error.message == "Custom reason"


class TestAutoExceptionDetailParams:
    """Test auto_exception details parameter handling."""

    def test_auto_exception_with_detail_params(self):
        """Decorator populates details dict from specified params."""
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"],
            detail_params=["source", "reason", "retry_count"]
        )
        class DataLoadError(BasePamolaError):
            """Data load error."""
            pass

        error = DataLoadError(
            source="database",
            reason="timeout",
            retry_count=3
        )
        assert error.details["source"] == "database"
        assert error.details["reason"] == "timeout"
        assert error.details["retry_count"] == 3

    def test_auto_exception_detail_params_defaults_to_message_params(self):
        """detail_params defaults to message_params if not specified."""
        @auto_exception(
            default_error_code=ErrorCode.PARAM_INVALID,
            message_params=["param_name", "value"]
        )
        class ParamError(BasePamolaError):
            """Parameter error."""
            pass

        error = ParamError(param_name="count", value="-1")
        # Both should be in details since detail_params defaults to message_params
        assert error.details["param_name"] == "count"
        assert error.details["value"] == "-1"

    def test_auto_exception_none_values_excluded_from_details(self):
        """None values are excluded from details dict."""
        @auto_exception(
            default_error_code=ErrorCode.FIELD_NOT_FOUND,
            message_params=["field_name"],
            detail_params=["field_name", "dataset_name"]
        )
        class FieldError(BasePamolaError):
            """Field error."""
            pass

        error = FieldError(field_name="user_id", dataset_name=None)
        assert error.details["field_name"] == "user_id"
        assert "dataset_name" not in error.details  # None excluded

    def test_auto_exception_details_dict_provided(self):
        """Explicitly provided details dict is merged with auto-generated."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            detail_params=["operation"]
        )
        class ProcessError(BasePamolaError):
            """Processing error."""
            pass

        error = ProcessError(
            message="Error occurred",
            operation="transform",
            details={"custom_key": "custom_value"}
        )
        assert error.details["operation"] == "transform"
        assert error.details["custom_key"] == "custom_value"


class TestAutoExceptionCustomMessageBuilder:
    """Test auto_exception custom_message_builder functionality."""

    def test_auto_exception_with_custom_message_builder(self):
        """custom_message_builder function creates formatted message."""
        def build_message(error_code, **kwargs):
            operation = kwargs.get("operation", "<unknown>")
            reason = kwargs.get("reason", "<unknown>")
            return f"Operation '{operation}' failed: {reason}"

        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            custom_message_builder=build_message
        )
        class CustomMsgError(BasePamolaError):
            """Error with custom message building."""
            pass

        error = CustomMsgError(operation="data_transform", reason="invalid format")
        assert error.message == "Operation 'data_transform' failed: invalid format"

    def test_custom_message_builder_receives_error_code(self):
        """custom_message_builder receives error_code as first argument."""
        received_codes = []

        def capture_code(error_code, **kwargs):
            received_codes.append(error_code)
            return "Message"

        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            custom_message_builder=capture_code
        )
        class TestError(BasePamolaError):
            """Test error."""
            pass

        error = TestError()
        assert ErrorCode.PROCESSING_FAILED in received_codes

    def test_custom_message_builder_with_overridden_error_code(self):
        """custom_message_builder receives overridden error code."""
        def build_message(error_code, **kwargs):
            return f"Code: {error_code}"

        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            custom_message_builder=build_message
        )
        class TestError(BasePamolaError):
            """Test error."""
            pass

        error = TestError(error_code=ErrorCode.PARAM_INVALID)
        assert "PARAM_INVALID" in error.message


class TestAutoExceptionParentClass:
    """Test auto_exception parent_class parameter."""

    def test_auto_exception_with_custom_parent_class(self):
        """parent_class parameter specifies parent __init__ to call.

        The generated __init__ calls parent.__init__ with only (message,
        error_code, details) — it does NOT forward arbitrary **params to the
        parent.  custom_attr therefore gets its default value (None) from
        CustomBase.__init__.
        """
        class CustomBase(BasePamolaError):
            """Custom base class with additional attributes."""

            def __init__(self, message, custom_attr=None, **kwargs):
                self.custom_attr = custom_attr
                super().__init__(message=message, **kwargs)

        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            parent_class=CustomBase
        )
        class DerivedError(CustomBase):
            """Error inheriting from custom base."""
            pass

        error = DerivedError(
            message="Test error",
            custom_attr="custom_value"
        )
        assert error.message == "Test error"
        # auto_exception only forwards (message, error_code, details) to parent,
        # so custom_attr is not passed through and remains at its default (None).
        assert error.custom_attr is None

    def test_auto_exception_parent_class_defaults_to_base_pamola_error(self):
        """parent_class defaults to BasePamolaError."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            parent_class=None
        )
        class SimpleError(BasePamolaError):
            """Simple error."""
            pass

        error = SimpleError(message="Test")
        # Should use BasePamolaError.__init__
        assert error.message == "Test"
        assert error.error_code == ErrorCode.PROCESSING_FAILED


class TestAutoExceptionMultipleLevels:
    """Test auto_exception with inheritance hierarchies."""

    def test_auto_exception_on_subclass_of_auto_exception(self):
        """Decorator can be applied to subclass of another decorated class."""
        @auto_exception(default_error_code=ErrorCode.DATA_LOAD_FAILED)
        class BaseDataError(BasePamolaError):
            """Base data error."""
            pass

        @auto_exception(default_error_code=ErrorCode.PROCESSING_FAILED)
        class DerivedError(BaseDataError):
            """Derived error."""
            pass

        error = DerivedError(message="Error")
        assert error.error_code == ErrorCode.PROCESSING_FAILED
        assert isinstance(error, BaseDataError)


class TestAutoExceptionEdgeCases:
    """Test auto_exception edge cases and error scenarios."""

    def test_auto_exception_with_empty_message_params(self):
        """Empty message_params list with no provided params."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            message_params=[]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error = Error()
        assert error.message == "Unknown error"
        assert error.error_code == ErrorCode.PROCESSING_FAILED

    def test_auto_exception_with_extra_kwargs(self):
        """Extra keyword arguments don't cause errors."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            message_params=["operation"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error = Error(
            operation="test",
            extra_param1="value1",
            extra_param2="value2"
        )
        assert error.message is not None
        assert error.error_code == ErrorCode.PROCESSING_FAILED

    def test_auto_exception_none_message_uses_template(self):
        """None message uses ErrorMessages template."""
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"]
        )
        class DataError(BasePamolaError):
            """Data error."""
            pass

        error = DataError(message=None, source="file.csv", reason="not found")
        assert error.message is not None
        assert "file.csv" in error.message
        assert "not found" in error.message

    def test_auto_exception_zero_values_in_details(self):
        """Zero values are included in details (falsy but valid).

        Note: 'error_code' cannot be used as a detail_param because the
        generated __init__ signature has ``error_code`` as a keyword-only
        parameter — it is consumed directly and never reaches **params.
        Use a different param name (e.g. 'code_value') instead.
        """
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            detail_params=["retry_count", "attempt_number"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error = Error(retry_count=0, attempt_number=0)
        # Zero values are not None, so they are included in details
        assert error.details["retry_count"] == 0
        assert error.details["attempt_number"] == 0

    def test_auto_exception_boolean_false_in_details(self):
        """Boolean False values are included in details."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            detail_params=["is_critical", "retry_allowed"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error = Error(is_critical=False, retry_allowed=False)
        # False is not None, so should be included
        assert error.details["is_critical"] is False
        assert error.details["retry_allowed"] is False

    def test_auto_exception_empty_string_in_details(self):
        """Empty string values are included in details."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            detail_params=["operation", "reason"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error = Error(operation="", reason="failed")
        # Empty string is not None, so should be included
        assert error.details["operation"] == ""
        assert error.details["reason"] == "failed"

    def test_auto_exception_special_characters_in_params(self):
        """Parameters with special characters work correctly.

        Use DATA_EMPTY which requires only 'operation' so the template
        formats successfully with no missing-param errors.
        """
        @auto_exception(
            default_error_code=ErrorCode.DATA_EMPTY,
            message_params=["operation"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        special_op = "operation\nwith\nnewlines\tand\ttabs"
        error = Error(operation=special_op)
        assert special_op in error.message

    def test_auto_exception_unicode_in_params(self):
        """Parameters with unicode characters work correctly.

        Use DATA_LOAD_FAILED with both required params so the template
        formats successfully and the unicode value appears in the message.
        """
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        unicode_source = "文件名.csv"
        error = Error(source=unicode_source, reason="not found")
        assert unicode_source in error.message


class TestAutoExceptionIntegration:
    """Integration tests with real error codes and scenarios."""

    def test_auto_exception_mimics_real_error_usage(self):
        """Decorator enables clean error class definitions."""
        @auto_exception(
            default_error_code=ErrorCode.DATA_LOAD_FAILED,
            message_params=["source", "reason"],
            detail_params=["source", "reason", "file_size", "retry_count"]
        )
        class DataLoadError(BasePamolaError):
            """Error loading data from source."""
            pass

        error = DataLoadError(
            source="s3://bucket/file.csv",
            reason="Connection timeout",
            file_size=1024000,
            retry_count=3
        )

        # Verify all expected attributes
        assert "s3://bucket/file.csv" in error.message
        assert "Connection timeout" in error.message
        assert error.error_code == ErrorCode.DATA_LOAD_FAILED
        assert error.details["source"] == "s3://bucket/file.csv"
        assert error.details["retry_count"] == 3

    def test_auto_exception_multiple_instances_independent(self):
        """Multiple instances don't interfere with each other."""
        @auto_exception(
            default_error_code=ErrorCode.PROCESSING_FAILED,
            detail_params=["operation"]
        )
        class Error(BasePamolaError):
            """Error class."""
            pass

        error1 = Error(operation="transform")
        error2 = Error(operation="validate")

        assert error1.details["operation"] == "transform"
        assert error2.details["operation"] == "validate"
        assert error1.details is not error2.details
