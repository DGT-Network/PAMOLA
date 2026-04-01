"""
Comprehensive tests for validation_rules module.

Tests cover:
- RuleCode enum and has_value method
- ValidationResult dataclass
- All ValidationRule subclasses (Required, Unique, Datatype, Format, MinMax, ValidValues, Regex)
- Format validators (Email, Phone, URL, IP, CreditCard, PostalCode, SSN, UUID)
- Rule factory (create_rule_from_code)
- ValidationRuleRegistry and lazy registry
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from pamola_core.metrics.commons.validation_rules import (
    RuleCode,
    RuleType,
    ValidationResult,
    ValidationRule,
    RequiredRule,
    UniqueRule,
    DatatypeRule,
    FormatRule,
    MinMaxRule,
    ValidValuesRule,
    RegexRule,
    create_rule_from_code,
    ValidationRuleRegistry,
    get_rule_registry,
    EmailValidator,
    PhoneValidator,
    URLValidator,
    IPValidator,
    CreditCardValidator,
    PostalCodeValidator,
    SSNValidator,
    UUIDValidator,
)
from pamola_core.errors.exceptions import InvalidParameterError, ValidationError


class TestRuleCode:
    """Test RuleCode enum and its methods."""

    def test_rule_code_values(self):
        """Test that all RuleCode enum values are defined."""
        assert RuleCode.REQUIRED.value == "REQUIRED"
        assert RuleCode.UNIQUE.value == "UNIQUE"
        assert RuleCode.DATATYPE_INT.value == "DATATYPE_INT"
        assert RuleCode.FORMAT_EMAIL.value == "FORMAT_EMAIL"
        assert RuleCode.MIN_MAX.value == "MIN_MAX"

    def test_has_value_valid(self):
        """Test has_value with valid codes."""
        assert RuleCode.has_value("REQUIRED") is True
        assert RuleCode.has_value("UNIQUE") is True
        assert RuleCode.has_value("DATATYPE_INT") is True
        assert RuleCode.has_value("FORMAT_EMAIL") is True

    def test_has_value_invalid(self):
        """Test has_value with invalid codes."""
        assert RuleCode.has_value("INVALID_CODE") is False
        assert RuleCode.has_value("") is False
        assert RuleCode.has_value("RequiredRule") is False


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_init(self):
        """Test initialization of ValidationResult."""
        result = ValidationResult(is_valid=True, error_count=0)
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.error_indices == []
        assert result.error_messages == []

    def test_validation_result_post_init_sync(self):
        """Test that __post_init__ syncs error_count with error_indices length."""
        result = ValidationResult(
            is_valid=False,
            error_count=10,  # Will be overridden
            error_indices=[1, 2, 3],
            error_messages=["msg1", "msg2", "msg3"],
        )
        # error_count should be synced to match error_indices length
        assert result.error_count == 3


class TestEmailValidator:
    """Test EmailValidator format validator."""

    def test_valid_emails(self):
        """Test that valid email addresses pass validation."""
        validator = EmailValidator()
        assert validator.validate("user@example.com") is True
        assert validator.validate("john.doe+tag@company.co.uk") is True
        assert validator.validate("test_user.123@test-domain.org") is True

    def test_invalid_emails(self):
        """Test that invalid emails fail validation."""
        validator = EmailValidator()
        assert validator.validate("notanemail") is False
        assert validator.validate("@example.com") is False
        assert validator.validate("user@") is False
        assert validator.validate("user name@example.com") is False

    def test_get_pattern(self):
        """Test get_pattern method."""
        validator = EmailValidator()
        pattern = validator.get_pattern()
        assert isinstance(pattern, str)
        assert "@" in pattern

    def test_get_description(self):
        """Test get_description method."""
        validator = EmailValidator()
        desc = validator.get_description()
        assert "email" in desc.lower()


class TestPhoneValidator:
    """Test PhoneValidator format validator."""

    def test_valid_phones(self):
        """Test valid phone numbers."""
        validator = PhoneValidator()
        assert validator.validate("+1-234-567-8900") is True
        assert validator.validate("(123) 456-7890") is True
        assert validator.validate("+33 1 42 34 56 78") is True
        assert validator.validate("1234567") is True  # 7 digits minimum

    def test_invalid_phones(self):
        """Test invalid phone numbers."""
        validator = PhoneValidator()
        assert validator.validate("123") is False  # Too short
        assert validator.validate("abc-def-ghij") is False
        assert validator.validate("") is False


class TestURLValidator:
    """Test URLValidator format validator."""

    def test_valid_urls(self):
        """Test valid URLs."""
        validator = URLValidator()
        assert validator.validate("http://example.com") is True
        assert validator.validate("https://www.example.com") is True
        assert validator.validate("https://example.com:8080/path") is True
        assert validator.validate("https://example.com/path?query=value") is True

    def test_invalid_urls(self):
        """Test invalid URLs."""
        validator = URLValidator()
        assert validator.validate("not a url") is False
        assert validator.validate("ftp://example.com") is False  # Only http/https
        assert validator.validate("example.com") is False  # Missing scheme


class TestIPValidator:
    """Test IPValidator format validator."""

    def test_valid_ipv4(self):
        """Test valid IPv4 addresses."""
        validator = IPValidator()
        assert validator.validate("192.168.1.1") is True
        assert validator.validate("10.0.0.1") is True
        assert validator.validate("255.255.255.255") is True
        assert validator.validate("0.0.0.0") is True

    def test_invalid_ipv4(self):
        """Test invalid IPv4 addresses."""
        validator = IPValidator()
        assert validator.validate("256.1.1.1") is False
        assert validator.validate("1.1.1") is False
        assert validator.validate("1.1.1.1.1") is False

    def test_valid_ipv6(self):
        """Test valid IPv6 addresses."""
        validator = IPValidator()
        assert validator.validate("::1") is True
        assert validator.validate("::") is True


class TestCreditCardValidator:
    """Test CreditCardValidator format validator."""

    def test_valid_credit_cards(self):
        """Test valid credit card formats."""
        validator = CreditCardValidator()
        assert validator.validate("1234 5678 9012 3456") is True
        assert validator.validate("1234-5678-9012-3456") is True
        assert validator.validate("1234567890123456") is True

    def test_invalid_credit_cards(self):
        """Test invalid credit card formats."""
        validator = CreditCardValidator()
        assert validator.validate("123") is False  # Too short
        assert validator.validate("12345678901234567890") is False  # Too long
        assert validator.validate("abcd efgh ijkl mnop") is False


class TestPostalCodeValidator:
    """Test PostalCodeValidator format validator."""

    def test_valid_postal_codes(self):
        """Test valid US ZIP codes."""
        validator = PostalCodeValidator()
        assert validator.validate("12345") is True
        assert validator.validate("12345-6789") is True

    def test_invalid_postal_codes(self):
        """Test invalid postal codes."""
        validator = PostalCodeValidator()
        assert validator.validate("1234") is False  # Too short
        assert validator.validate("123456") is False  # Too long
        assert validator.validate("abcde") is False


class TestSSNValidator:
    """Test SSNValidator format validator."""

    def test_valid_ssn(self):
        """Test valid SSN format."""
        validator = SSNValidator()
        assert validator.validate("123-45-6789") is True

    def test_invalid_ssn(self):
        """Test invalid SSN format."""
        validator = SSNValidator()
        assert validator.validate("12345678") is False  # Missing dashes
        assert validator.validate("123-45-678") is False  # Short
        assert validator.validate("123-456-789") is False  # Wrong format


class TestUUIDValidator:
    """Test UUIDValidator format validator."""

    def test_valid_uuid(self):
        """Test valid UUID format."""
        validator = UUIDValidator()
        assert validator.validate("550e8400-e29b-41d4-a716-446655440000") is True
        assert validator.validate("12345678-1234-1234-1234-123456789012") is True

    def test_invalid_uuid(self):
        """Test invalid UUID format."""
        validator = UUIDValidator()
        assert validator.validate("550e8400-e29b-41d4-a716") is False  # Short
        assert validator.validate("12345678123412341234123456789012") is False  # No dashes
        assert validator.validate("not-a-uuid-here-1234-5678") is False


class TestRequiredRule:
    """Test RequiredRule validation."""

    def test_required_rule_valid(self):
        """Test series with no null values."""
        rule = RequiredRule()
        series = pd.Series(["a", "b", "c"])
        result = rule.validate(series)
        assert result.is_valid is True
        assert result.error_count == 0

    def test_required_rule_with_null(self):
        """Test series with null values."""
        rule = RequiredRule()
        series = pd.Series(["a", None, "c"])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 1
        assert 1 in result.error_indices

    def test_required_rule_with_empty_string(self):
        """Test series with empty strings."""
        rule = RequiredRule()
        series = pd.Series(["a", "", "c"], dtype="object")
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 1

    def test_required_rule_disabled(self):
        """Test disabled required rule."""
        rule = RequiredRule(enabled=False)
        series = pd.Series([None, None])
        result = rule.validate(series)
        assert result.is_valid is True
        assert result.error_count == 0

    def test_required_rule_get_description(self):
        """Test get_description method."""
        rule = RequiredRule()
        assert "non-null" in rule.get_description()


class TestUniqueRule:
    """Test UniqueRule validation."""

    def test_unique_rule_valid(self):
        """Test series with all unique values."""
        rule = UniqueRule()
        series = pd.Series([1, 2, 3, 4])
        result = rule.validate(series)
        assert result.is_valid is True
        assert result.error_count == 0

    def test_unique_rule_with_duplicates(self):
        """Test series with duplicate values."""
        rule = UniqueRule()
        series = pd.Series([1, 2, 2, 3, 3, 3])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count > 0  # Has duplicates
        assert len(result.error_messages) > 0

    def test_unique_rule_with_nulls(self):
        """Test that nulls are ignored in uniqueness check."""
        rule = UniqueRule()
        series = pd.Series([1, 2, None, None])
        result = rule.validate(series)
        assert result.is_valid is True
        assert result.error_count == 0

    def test_unique_rule_all_nulls(self):
        """Test series with all null values."""
        rule = UniqueRule()
        series = pd.Series([None, None])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_unique_rule_disabled(self):
        """Test disabled unique rule."""
        rule = UniqueRule(enabled=False)
        series = pd.Series([1, 1, 1])
        result = rule.validate(series)
        assert result.is_valid is True


class TestDatatypeRule:
    """Test DatatypeRule validation."""

    def test_datatype_int_valid(self):
        """Test valid integer values."""
        rule = DatatypeRule("int")
        series = pd.Series([1, 2, 3])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_datatype_int_with_floats(self):
        """Test integer rule with float values."""
        rule = DatatypeRule("int")
        series = pd.Series([1.5, 2.5])
        result = rule.validate(series)
        assert result.is_valid is False

    def test_datatype_float_valid(self):
        """Test valid float values."""
        rule = DatatypeRule("float")
        series = pd.Series([1.1, 2.2, 3.3])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_datatype_float_with_strings(self):
        """Test float rule with invalid strings."""
        rule = DatatypeRule("float")
        series = pd.Series(["abc", "def"])
        result = rule.validate(series)
        assert result.is_valid is False

    def test_datatype_bool_valid(self):
        """Test valid boolean values."""
        rule = DatatypeRule("bool")
        series = pd.Series([True, False, True])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_datatype_bool_string_formats(self):
        """Test boolean rule with string formats."""
        rule = DatatypeRule("bool")
        series = pd.Series(["true", "false", "yes", "no", "1", "0"])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_datatype_date_valid(self):
        """Test valid date values."""
        rule = DatatypeRule("date")
        series = pd.Series(pd.date_range("2020-01-01", periods=3))
        result = rule.validate(series)
        assert result.is_valid is True

    def test_datatype_date_string_formats(self):
        """Test date rule with string formats."""
        rule = DatatypeRule("date")
        series = pd.Series(["2020-01-01", "2020-01-02"])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_datatype_rule_with_nulls(self):
        """Test that null values are skipped."""
        rule = DatatypeRule("int")
        series = pd.Series([1, None, 3])
        result = rule.validate(series)
        assert result.is_valid is True
        assert result.error_count == 0

    def test_datatype_rule_disabled(self):
        """Test disabled datatype rule."""
        rule = DatatypeRule("int", enabled=False)
        series = pd.Series(["not", "int"])
        result = rule.validate(series)
        assert result.is_valid is True


class TestFormatRule:
    """Test FormatRule validation."""

    def test_format_email_valid(self):
        """Test format rule with valid emails."""
        rule = FormatRule("email")
        series = pd.Series(["user@example.com", "test@test.org"])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_format_email_invalid(self):
        """Test format rule with invalid emails."""
        rule = FormatRule("email")
        series = pd.Series(["notanemail", "test@"])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 2

    def test_format_email_with_nulls(self):
        """Test that nulls are skipped in format validation."""
        rule = FormatRule("email")
        series = pd.Series(["user@example.com", None])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_format_rule_unsupported_type(self):
        """Test that unsupported format types raise error."""
        with pytest.raises(InvalidParameterError):
            FormatRule("unsupported_format")

    def test_format_rule_get_supported_formats(self):
        """Test get_supported_formats class method."""
        formats = FormatRule.get_supported_formats()
        assert "email" in formats
        assert "phone" in formats
        assert "url" in formats
        assert len(formats) >= 8

    def test_format_rule_register_validator(self):
        """Test registering a custom format validator."""
        class CustomValidator:
            def validate(self, value):
                return value.startswith("CUSTOM_")

            def get_pattern(self):
                return "CUSTOM_.*"

            def get_description(self):
                return "Custom format"

        # Register the validator
        from pamola_core.metrics.commons.validation_rules import FormatValidator

        class ProperCustomValidator(FormatValidator):
            def validate(self, value):
                return value.startswith("CUSTOM_")

            def get_pattern(self):
                return "CUSTOM_.*"

            def get_description(self):
                return "Custom format"

        FormatRule.register_format_validator("custom", ProperCustomValidator)
        assert "custom" in FormatRule.get_supported_formats()

    def test_format_rule_unregister_validator(self):
        """Test unregistering a format validator."""
        # First register
        from pamola_core.metrics.commons.validation_rules import FormatValidator

        class TempValidator(FormatValidator):
            def validate(self, value):
                return True

            def get_pattern(self):
                return ".*"

            def get_description(self):
                return "Temp"

        FormatRule.register_format_validator("temp", TempValidator)
        assert "temp" in FormatRule.get_supported_formats()

        # Then unregister
        result = FormatRule.unregister_format_validator("temp")
        assert result is True
        assert "temp" not in FormatRule.get_supported_formats()

        # Unregister non-existent
        result = FormatRule.unregister_format_validator("nonexistent")
        assert result is False


class TestMinMaxRule:
    """Test MinMaxRule validation."""

    def test_min_max_rule_valid(self):
        """Test values within range."""
        rule = MinMaxRule(min_value=0, max_value=100)
        series = pd.Series([10, 50, 100])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_min_max_rule_below_min(self):
        """Test values below minimum."""
        rule = MinMaxRule(min_value=10)
        series = pd.Series([5, 15, 20])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 1
        assert 0 in result.error_indices

    def test_min_max_rule_above_max(self):
        """Test values above maximum."""
        rule = MinMaxRule(max_value=50)
        series = pd.Series([10, 50, 60])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 1
        assert 2 in result.error_indices

    def test_min_max_rule_only_min(self):
        """Test with only minimum specified."""
        rule = MinMaxRule(min_value=10)
        series = pd.Series([10, 100, 1000])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_min_max_rule_only_max(self):
        """Test with only maximum specified."""
        rule = MinMaxRule(max_value=100)
        series = pd.Series([1, 50, 100])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_min_max_rule_non_numeric(self):
        """Test with non-numeric values."""
        rule = MinMaxRule(min_value=10, max_value=100)
        series = pd.Series([10, "invalid", 50])
        result = rule.validate(series)
        assert result.is_valid is False
        assert 1 in result.error_indices

    def test_min_max_rule_with_nulls(self):
        """Test that nulls are skipped."""
        rule = MinMaxRule(min_value=10, max_value=100)
        series = pd.Series([10, None, 50])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_min_max_rule_get_description(self):
        """Test get_description method."""
        rule = MinMaxRule(min_value=10, max_value=100)
        desc = rule.get_description()
        assert "min: 10" in desc
        assert "max: 100" in desc


class TestValidValuesRule:
    """Test ValidValuesRule validation."""

    def test_valid_values_rule_valid(self):
        """Test values in allowed list."""
        rule = ValidValuesRule(valid_values=["A", "B", "C"])
        series = pd.Series(["A", "B", "C"])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_valid_values_rule_invalid(self):
        """Test values not in allowed list."""
        rule = ValidValuesRule(valid_values=["A", "B"])
        series = pd.Series(["A", "C", "D"])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 2

    def test_valid_values_rule_with_nulls(self):
        """Test that nulls are skipped."""
        rule = ValidValuesRule(valid_values=["A", "B"])
        series = pd.Series(["A", None, "B"])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_valid_values_rule_numeric(self):
        """Test with numeric values."""
        rule = ValidValuesRule(valid_values=[1, 2, 3])
        series = pd.Series([1, 2, 3])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_valid_values_rule_get_description(self):
        """Test get_description method."""
        rule = ValidValuesRule(valid_values=["A", "B"])
        desc = rule.get_description()
        assert "A" in desc
        assert "B" in desc


class TestRegexRule:
    """Test RegexRule validation."""

    def test_regex_rule_valid(self):
        """Test values matching pattern."""
        rule = RegexRule(pattern=r"^[A-Z]{3}$")
        series = pd.Series(["ABC", "XYZ"])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_regex_rule_invalid(self):
        """Test values not matching pattern."""
        rule = RegexRule(pattern=r"^[0-9]{3}$")
        series = pd.Series(["123", "ABC"])
        result = rule.validate(series)
        assert result.is_valid is False
        assert result.error_count == 1

    def test_regex_rule_with_nulls(self):
        """Test that nulls are skipped."""
        rule = RegexRule(pattern=r"^\d+$")
        series = pd.Series(["123", None])
        result = rule.validate(series)
        assert result.is_valid is True

    def test_regex_rule_complex_pattern(self):
        """Test with complex regex pattern."""
        rule = RegexRule(pattern=r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
        series = pd.Series(["user@example.com", "test@test.org"])
        result = rule.validate(series)
        assert result.is_valid is True


class TestCreateRuleFromCode:
    """Test create_rule_from_code factory function."""

    def test_create_required_rule(self):
        """Test creating a required rule from code."""
        rule = create_rule_from_code("REQUIRED", {})
        assert isinstance(rule, RequiredRule)

    def test_create_unique_rule(self):
        """Test creating a unique rule from code."""
        rule = create_rule_from_code("UNIQUE", {})
        assert isinstance(rule, UniqueRule)

    def test_create_datatype_rules(self):
        """Test creating datatype rules from codes."""
        rule_int = create_rule_from_code("DATATYPE_INT", {})
        assert isinstance(rule_int, DatatypeRule)
        assert rule_int.expected_type == "int"

        rule_float = create_rule_from_code("DATATYPE_FLOAT", {})
        assert rule_float.expected_type == "float"

        rule_date = create_rule_from_code("DATATYPE_DATE", {})
        assert rule_date.expected_type == "date"

        rule_bool = create_rule_from_code("DATATYPE_BOOL", {})
        assert rule_bool.expected_type == "bool"

    def test_create_format_rules(self):
        """Test creating format rules from codes."""
        rule_email = create_rule_from_code("FORMAT_EMAIL", {})
        assert isinstance(rule_email, FormatRule)
        assert rule_email.format_type == "email"

        rule_phone = create_rule_from_code("FORMAT_PHONE", {})
        assert rule_phone.format_type == "phone"

    def test_create_min_max_rule(self):
        """Test creating min/max rule with metadata."""
        rule = create_rule_from_code("MIN_MAX", {"min": 10, "max": 100})
        assert isinstance(rule, MinMaxRule)
        assert rule.min_value == 10
        assert rule.max_value == 100

    def test_create_valid_values_rule(self):
        """Test creating valid values rule with metadata."""
        rule = create_rule_from_code("VALID_VALUES", {"values": ["A", "B"]})
        assert isinstance(rule, ValidValuesRule)
        assert "A" in rule.valid_values

    def test_create_valid_values_rule_alt_key(self):
        """Test creating valid values rule with alternate key."""
        rule = create_rule_from_code("VALID_VALUES", {"valid_values": ["X", "Y"]})
        assert isinstance(rule, ValidValuesRule)

    def test_create_regex_rule(self):
        """Test creating regex rule with metadata."""
        rule = create_rule_from_code("REGEX", {"pattern": r"^\d+$"})
        assert isinstance(rule, RegexRule)
        assert rule.pattern == r"^\d+$"

    def test_create_regex_rule_no_pattern(self):
        """Test creating regex rule without pattern returns None."""
        rule = create_rule_from_code("REGEX", {})
        assert rule is None

    def test_create_rule_invalid_code(self):
        """Test creating rule with invalid code returns None."""
        rule = create_rule_from_code("INVALID_CODE", {})
        assert rule is None


class TestValidationRuleRegistry:
    """Test ValidationRuleRegistry."""

    def test_registry_register_rule(self):
        """Test registering a rule."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule, is_system_rule=True)
        assert registry.get_rule("Required") == rule

    def test_registry_get_rule(self):
        """Test retrieving a rule."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule)
        retrieved = registry.get_rule("Required")
        assert retrieved == rule

    def test_registry_get_rule_not_found(self):
        """Test retrieving non-existent rule."""
        registry = ValidationRuleRegistry()
        retrieved = registry.get_rule("NonExistent")
        assert retrieved is None

    def test_registry_get_rule_by_code(self):
        """Test retrieving rule by RuleCode."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule)
        retrieved = registry.get_rule_by_code(RuleCode.REQUIRED)
        assert retrieved == rule

    def test_registry_get_rule_by_code_string(self):
        """Test retrieving rule by code string."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule)
        retrieved = registry.get_rule_by_code("REQUIRED")
        assert retrieved == rule

    def test_registry_get_all_rules(self):
        """Test getting all registered rules."""
        registry = ValidationRuleRegistry()
        rule1 = RequiredRule()
        rule2 = UniqueRule()
        registry.register_rule(rule1)
        registry.register_rule(rule2)
        all_rules = registry.get_all_rules()
        assert len(all_rules) >= 2

    def test_registry_get_system_rules(self):
        """Test getting system rules."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule, is_system_rule=True)
        system_rules = registry.get_system_rules()
        assert rule.rule_name in system_rules

    def test_registry_get_custom_rules(self):
        """Test getting custom rules."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule, is_system_rule=False)
        custom_rules = registry.get_custom_rules()
        assert rule.rule_name in custom_rules

    def test_registry_remove_rule(self):
        """Test removing a rule."""
        registry = ValidationRuleRegistry()
        rule = RequiredRule()
        registry.register_rule(rule)
        result = registry.remove_rule("Required")
        assert result is True
        assert registry.get_rule("Required") is None

    def test_registry_remove_rule_not_found(self):
        """Test removing non-existent rule."""
        registry = ValidationRuleRegistry()
        result = registry.remove_rule("NonExistent")
        assert result is False


class TestGetRuleRegistry:
    """Test get_rule_registry function and lazy registry."""

    def test_get_rule_registry_singleton(self):
        """Test that get_rule_registry returns singleton."""
        registry1 = get_rule_registry()
        registry2 = get_rule_registry()
        assert registry1 is registry2

    def test_default_registry_has_system_rules(self):
        """Test that default registry is initialized with system rules."""
        registry = get_rule_registry()
        system_rules = registry.get_system_rules()
        # Should have many default system rules
        assert len(system_rules) > 0


class TestValidationRuleIntegration:
    """Integration tests for validation rules."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow with multiple rules."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, None],
                "email": ["user@example.com", "invalid", "test@test.org", None],
                "score": [10, 50, 100, 150],
            }
        )

        # Test required rule on id column
        required_rule = RequiredRule()
        result = required_rule.validate(df["id"])
        assert result.is_valid is False
        assert 3 in result.error_indices

        # Test format rule on email column
        email_rule = FormatRule("email")
        result = email_rule.validate(df["email"])
        assert result.is_valid is False
        assert 1 in result.error_indices

        # Test min/max rule on score column
        minmax_rule = MinMaxRule(min_value=0, max_value=100)
        result = minmax_rule.validate(df["score"])
        assert result.is_valid is False
        assert 3 in result.error_indices

    def test_rule_to_dict(self):
        """Test converting rule to dictionary."""
        rule = RequiredRule()
        rule_dict = rule.to_dict()
        assert "rule_name" in rule_dict and rule_dict["rule_name"] in ["Required", "RequiredRule"]
        assert "rule_type" in rule_dict and rule_dict["rule_type"] in ["required", "REQUIRED"]
        assert rule_dict["enabled"] is True
