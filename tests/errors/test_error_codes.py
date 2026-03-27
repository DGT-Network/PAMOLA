"""
Unit tests for pamola_core.errors.codes.registry.ErrorCode class.

Tests cover:
- Error code registry validation
- Error code queries (by category, severity, etc.)
- Validation methods
- Classification methods
- Edge cases and comprehensive code coverage

Run with: pytest tests/errors/test_error_codes.py
"""

import pytest
from pamola_core.errors.codes.registry import ErrorCode
from pamola_core.errors.codes.metadata import get_error_metadata


class TestErrorCodeRegistry:
    """Test ErrorCode registry completeness and structure."""

    def test_all_codes_are_strings(self):
        """All error codes are string constants."""
        codes = ErrorCode.get_all_codes()
        assert len(codes) > 0
        for code in codes:
            assert isinstance(code, str)
            assert len(code) > 0
            assert code.isupper()

    def test_all_codes_have_underscores(self):
        """Error codes follow naming convention with underscores."""
        codes = ErrorCode.get_all_codes()
        for code in codes:
            # Should be CATEGORY_SPECIFIC format
            assert "_" in code, f"Code {code} doesn't have underscore"

    def test_all_codes_are_unique(self):
        """All error codes are unique."""
        codes = ErrorCode.get_all_codes()
        assert len(codes) == len(set(codes))

    def test_codes_are_sorted(self):
        """get_all_codes returns sorted list."""
        codes = ErrorCode.get_all_codes()
        assert codes == sorted(codes)

    def test_minimum_number_of_codes(self):
        """Registry contains substantial number of error codes."""
        codes = ErrorCode.get_all_codes()
        assert len(codes) > 50  # Should have many codes

    def test_class_attributes_match_get_all_codes(self):
        """ErrorCode class attributes match get_all_codes()."""
        codes = ErrorCode.get_all_codes()
        # Verify some known codes exist
        assert ErrorCode.DATA_LOAD_FAILED in codes
        assert ErrorCode.PROCESSING_FAILED in codes
        assert ErrorCode.PARAM_INVALID in codes


class TestErrorCodeValidation:
    """Test error code validation methods."""

    def test_is_valid_code_existing_code(self):
        """is_valid_code returns True for existing codes."""
        assert ErrorCode.is_valid_code(ErrorCode.DATA_LOAD_FAILED)
        assert ErrorCode.is_valid_code(ErrorCode.PROCESSING_FAILED)
        assert ErrorCode.is_valid_code(ErrorCode.PARAM_INVALID)

    def test_is_valid_code_nonexistent_code(self):
        """is_valid_code returns False for non-existent codes."""
        assert not ErrorCode.is_valid_code("NONEXISTENT_CODE")
        assert not ErrorCode.is_valid_code("INVALID_ERROR_CODE")

    def test_is_valid_code_case_sensitive(self):
        """is_valid_code is case sensitive."""
        assert ErrorCode.is_valid_code("DATA_LOAD_FAILED")
        assert not ErrorCode.is_valid_code("data_load_failed")
        assert not ErrorCode.is_valid_code("Data_Load_Failed")

    def test_validate_code_valid(self):
        """validate_code doesn't raise for valid codes."""
        # Should not raise
        ErrorCode.validate_code(ErrorCode.DATA_LOAD_FAILED)
        ErrorCode.validate_code(ErrorCode.PROCESSING_FAILED)

    def test_validate_code_invalid(self):
        """validate_code raises ValueError for invalid codes."""
        with pytest.raises(ValueError) as exc_info:
            ErrorCode.validate_code("INVALID_CODE")
        assert "INVALID_CODE" in str(exc_info.value)
        assert "Invalid error_code" in str(exc_info.value)

    def test_validate_code_error_message_contains_samples(self):
        """validate_code error includes sample valid codes."""
        with pytest.raises(ValueError) as exc_info:
            ErrorCode.validate_code("FAKE_CODE")
        error_msg = str(exc_info.value)
        # Should suggest some valid codes
        assert "Must be one of:" in error_msg


class TestErrorCodeByCategory:
    """Test error code filtering by category."""

    def test_get_codes_by_category_data(self):
        """Get all DATA category codes."""
        codes = ErrorCode.get_codes_by_category("data")
        assert len(codes) > 0
        for code in codes:
            assert code.startswith("DATA_")

    def test_get_codes_by_category_validation(self):
        """Get all VALIDATION category codes (prefix VALIDATION_).

        get_codes_by_category uses prefix matching, so "validation" returns
        only codes that start with "VALIDATION_" (not FIELD_, PARAM_, etc.).
        There are 4 such codes in the registry.
        """
        codes = ErrorCode.get_codes_by_category("validation")
        assert len(codes) > 0
        for code in codes:
            assert code.startswith("VALIDATION_")

    def test_get_codes_by_category_crypto(self):
        """Get all CRYPTO category codes (prefix-based: CRYPTO_)."""
        codes = ErrorCode.get_codes_by_category("crypto")
        assert len(codes) > 0
        # get_codes_by_category uses prefix matching: only returns codes starting with "CRYPTO_"
        for code in codes:
            assert code.startswith("CRYPTO_")

    def test_get_codes_by_category_case_insensitive(self):
        """get_codes_by_category is case insensitive."""
        codes_upper = ErrorCode.get_codes_by_category("DATA")
        codes_lower = ErrorCode.get_codes_by_category("data")
        codes_mixed = ErrorCode.get_codes_by_category("Data")
        assert codes_upper == codes_lower == codes_mixed

    def test_get_codes_by_category_nonexistent(self):
        """Nonexistent category returns empty list."""
        codes = ErrorCode.get_codes_by_category("nonexistent_category_xyz")
        assert codes == []

    def test_get_codes_by_category_sorted(self):
        """Codes returned by category are sorted."""
        codes = ErrorCode.get_codes_by_category("data")
        assert codes == sorted(codes)


class TestErrorCodeValidationMethods:
    """Test validation-specific error code methods."""

    def test_get_validation_codes(self):
        """Get all validation-related error codes."""
        codes = ErrorCode.get_validation_codes()
        assert len(codes) > 0
        # All should match validation prefixes
        validation_prefixes = ["VALIDATION_", "FIELD_", "PARAM_", "FILE_", "COLUMN_"]
        for code in codes:
            assert any(code.startswith(prefix) for prefix in validation_prefixes)

    def test_get_validation_codes_includes_expected(self):
        """get_validation_codes includes known validation codes."""
        codes = ErrorCode.get_validation_codes()
        assert ErrorCode.FIELD_NOT_FOUND in codes
        assert ErrorCode.PARAM_INVALID in codes
        assert ErrorCode.FILE_NOT_FOUND in codes

    def test_get_validation_codes_excludes_non_validation(self):
        """get_validation_codes excludes non-validation codes."""
        codes = ErrorCode.get_validation_codes()
        assert ErrorCode.DATA_LOAD_FAILED not in codes
        assert ErrorCode.ENCRYPTION_FAILED not in codes


class TestErrorCodeRetryBehavior:
    """Test retry-related error code methods."""

    def test_get_retriable_codes(self):
        """Get all error codes that allow retry."""
        codes = ErrorCode.get_retriable_codes()
        assert len(codes) > 0
        # Verify these codes actually have retry_allowed=True
        for code in codes:
            metadata = get_error_metadata(code)
            assert metadata.get("retry_allowed") is True

    def test_get_retriable_codes_sorted(self):
        """Retriable codes are sorted."""
        codes = ErrorCode.get_retriable_codes()
        assert codes == sorted(codes)

    def test_is_retriable_true(self):
        """is_retriable returns True for retriable codes."""
        # Should have at least one retriable code
        retriable = ErrorCode.get_retriable_codes()
        if retriable:
            assert ErrorCode.is_retriable(retriable[0])

    def test_is_retriable_false(self):
        """is_retriable returns False for non-retriable codes."""
        assert not ErrorCode.is_retriable(ErrorCode.PARAM_INVALID)
        assert not ErrorCode.is_retriable(ErrorCode.FIELD_NOT_FOUND)

    def test_is_retriable_all_codes_have_result(self):
        """is_retriable returns boolean for all codes."""
        codes = ErrorCode.get_all_codes()
        for code in codes:
            result = ErrorCode.is_retriable(code)
            assert isinstance(result, bool)


class TestErrorCodeUserFacing:
    """Test user-facing error code methods."""

    def test_get_user_facing_codes(self):
        """Get all user-facing error codes."""
        codes = ErrorCode.get_user_facing_codes()
        assert len(codes) > 0
        # Verify these are actually user-facing
        for code in codes:
            metadata = get_error_metadata(code)
            assert metadata.get("user_facing") is True

    def test_is_user_facing_true(self):
        """is_user_facing returns True for user-facing codes."""
        user_facing = ErrorCode.get_user_facing_codes()
        if user_facing:
            assert ErrorCode.is_user_facing(user_facing[0])

    def test_is_user_facing_false(self):
        """is_user_facing returns False for internal codes."""
        assert not ErrorCode.is_user_facing(ErrorCode.ENCRYPTION_FAILED)

    def test_is_user_facing_data_codes(self):
        """Data error codes are user-facing."""
        assert ErrorCode.is_user_facing(ErrorCode.DATA_LOAD_FAILED)
        assert ErrorCode.is_user_facing(ErrorCode.FILE_NOT_FOUND)


class TestErrorCodeSeverity:
    """Test error code severity-based queries."""

    def test_get_codes_by_severity_error(self):
        """Get all error codes with 'error' severity."""
        codes = ErrorCode.get_codes_by_severity("error")
        assert len(codes) > 0
        for code in codes:
            metadata = get_error_metadata(code)
            assert metadata.get("severity") == "error"

    def test_get_codes_by_severity_critical(self):
        """Get all error codes with 'critical' severity."""
        codes = ErrorCode.get_codes_by_severity("critical")
        assert len(codes) > 0
        for code in codes:
            metadata = get_error_metadata(code)
            assert metadata.get("severity") == "critical"

    def test_get_codes_by_severity_warning(self):
        """Get all error codes with 'warning' severity."""
        codes = ErrorCode.get_codes_by_severity("warning")
        # May or may not have warning codes
        for code in codes:
            metadata = get_error_metadata(code)
            assert metadata.get("severity") == "warning"

    def test_get_codes_by_severity_case_insensitive(self):
        """get_codes_by_severity is case insensitive."""
        codes_upper = ErrorCode.get_codes_by_severity("ERROR")
        codes_lower = ErrorCode.get_codes_by_severity("error")
        codes_mixed = ErrorCode.get_codes_by_severity("Error")
        assert codes_upper == codes_lower == codes_mixed

    def test_get_codes_by_severity_unknown(self):
        """Unknown severity returns empty list."""
        codes = ErrorCode.get_codes_by_severity("unknown_severity_xyz")
        assert codes == []


class TestErrorCodeConstants:
    """Test that all known error codes are defined."""

    def test_data_error_codes(self):
        """Data error codes are defined."""
        assert hasattr(ErrorCode, "DATA_LOAD_FAILED")
        assert hasattr(ErrorCode, "DATA_VALIDATION_ERROR")
        assert hasattr(ErrorCode, "DATA_EMPTY")

    def test_validation_error_codes(self):
        """Validation error codes are defined."""
        assert hasattr(ErrorCode, "FIELD_NOT_FOUND")
        assert hasattr(ErrorCode, "PARAM_INVALID")
        assert hasattr(ErrorCode, "COLUMN_NOT_FOUND")

    def test_file_error_codes(self):
        """File error codes are defined."""
        assert hasattr(ErrorCode, "FILE_NOT_FOUND")
        assert hasattr(ErrorCode, "FILE_FORMAT_INVALID")
        assert hasattr(ErrorCode, "FILE_ACCESS_DENIED")

    def test_processing_error_codes(self):
        """Processing error codes are defined."""
        assert hasattr(ErrorCode, "PROCESSING_FAILED")
        assert hasattr(ErrorCode, "PROCESSING_TIMEOUT")

    def test_crypto_error_codes(self):
        """Crypto error codes are defined."""
        assert hasattr(ErrorCode, "ENCRYPTION_FAILED")
        assert hasattr(ErrorCode, "DECRYPTION_FAILED")
        assert hasattr(ErrorCode, "KEY_INVALID")

    def test_task_error_codes(self):
        """Task error codes are defined."""
        assert hasattr(ErrorCode, "TASK_ERROR")
        assert hasattr(ErrorCode, "TASK_EXECUTION_FAILED")


class TestErrorCodeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_code_with_special_characters(self):
        """Error codes contain only uppercase letters and underscores."""
        codes = ErrorCode.get_all_codes()
        for code in codes:
            assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in code)

    def test_code_no_leading_underscore(self):
        """Error codes don't start with underscore."""
        codes = ErrorCode.get_all_codes()
        for code in codes:
            assert not code.startswith("_")

    def test_code_no_trailing_underscore(self):
        """Error codes don't end with underscore."""
        codes = ErrorCode.get_all_codes()
        for code in codes:
            assert not code.endswith("_")

    def test_category_extraction(self):
        """Categories can be extracted from code names."""
        codes = ErrorCode.get_all_codes()
        categories = set()
        for code in codes:
            category = code.split("_")[0]
            categories.add(category)
        assert len(categories) > 5  # Should have many categories

    def test_query_combinations(self):
        """Various query combinations work together."""
        # Get all data codes that are user-facing
        all_data = ErrorCode.get_codes_by_category("data")
        user_facing = ErrorCode.get_user_facing_codes()
        intersection = [c for c in all_data if c in user_facing]
        assert len(intersection) > 0

        # Get all error severity codes that are retriable
        all_error = ErrorCode.get_codes_by_severity("error")
        retriable = ErrorCode.get_retriable_codes()
        intersection = [c for c in all_error if c in retriable]
        assert len(intersection) > 0


class TestErrorCodeIntegration:
    """Integration tests for error code operations."""

    def test_typical_workflow(self):
        """Typical error code workflow."""
        # Validate an error code
        assert ErrorCode.is_valid_code(ErrorCode.DATA_LOAD_FAILED)

        # Check if it's retriable
        is_retriable = ErrorCode.is_retriable(ErrorCode.DATA_LOAD_FAILED)
        assert isinstance(is_retriable, bool)

        # Check if it's user-facing
        is_user_facing = ErrorCode.is_user_facing(ErrorCode.DATA_LOAD_FAILED)
        assert isinstance(is_user_facing, bool)

        # Get metadata
        metadata = get_error_metadata(ErrorCode.DATA_LOAD_FAILED)
        assert metadata is not None

    def test_error_categorization(self):
        """Error codes can be categorized in different ways."""
        all_codes = ErrorCode.get_all_codes()

        # Categorize by retry behavior
        retriable = ErrorCode.get_retriable_codes()
        non_retriable = [c for c in all_codes if c not in retriable]
        assert len(retriable) > 0
        assert len(non_retriable) > 0

        # Categorize by user-facing
        user_facing = ErrorCode.get_user_facing_codes()
        internal = [c for c in all_codes if c not in user_facing]
        assert len(user_facing) > 0
        assert len(internal) > 0

    def test_all_codes_have_metadata(self):
        """All registered codes have corresponding metadata."""
        codes = ErrorCode.get_all_codes()
        for code in codes:
            metadata = get_error_metadata(code)
            # Should have basic metadata fields
            assert "severity" in metadata
            assert "category" in metadata
            assert "retry_allowed" in metadata or "retry_allowed" in metadata
