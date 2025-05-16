"""
Tests for the pamola_core.fake_data.commons.validators module.

This module contains unit tests for the validation functions in the validators.py module.
"""

import unittest

# Import the module to test
from pamola_core.fake_data.commons.validators import (
    validate_name,
    validate_email,
    validate_phone,
    validate_format,
    validate_id_number
)


class TestValidateName(unittest.TestCase):
    """Tests for the validate_name function."""

    def test_valid_russian_names(self):
        """Test validation of valid Russian names."""
        valid_names = [
            "Иван",
            "Мария",
            "Анна-Мария",
            "Иван Петров",
            "Петров-Водкин Кузьма"
        ]

        for name in valid_names:
            result = validate_name(name, language="ru")
            self.assertTrue(result["valid"], f"Name '{name}' should be valid")
            self.assertEqual(len(result["errors"]), 0, f"No errors should be reported for '{name}'")
            self.assertIn("length", result["properties"])
            self.assertEqual(result["properties"]["length"], len(name))

    def test_valid_english_names(self):
        """Test validation of valid English names."""
        valid_names = [
            "John",
            "Mary",
            "John Smith",
            "Mary-Jane",
            "O'Connor"
        ]

        for name in valid_names:
            result = validate_name(name, language="en")
            self.assertTrue(result["valid"], f"Name '{name}' should be valid")
            self.assertEqual(len(result["errors"]), 0, f"No errors should be reported for '{name}'")

    def test_invalid_names(self):
        """Test validation of invalid names."""
        invalid_names = [
            "",  # Empty
            "a",  # Too short
            "Name123",  # Contains digits
            "Name!@#",  # Contains special characters
            "a" * 51,  # Too long
            "name"  # Starts with lowercase
        ]

        for name in invalid_names:
            result = validate_name(name)
            self.assertFalse(result["valid"], f"Name '{name}' should be invalid")
            self.assertGreater(len(result["errors"]), 0, f"Errors should be reported for '{name}'")

    def test_name_properties(self):
        """Test that properties are correctly detected for names."""
        # Test name with space
        result = validate_name("John Smith")
        self.assertTrue(result["properties"]["has_space"])

        # Test name with hyphen
        result = validate_name("Mary-Jane")
        self.assertTrue(result["properties"]["has_hyphen"])

        # Test name with apostrophe
        result = validate_name("O'Connor")
        self.assertTrue(result["properties"]["has_apostrophe"])

        # Test name without special characters
        result = validate_name("John")
        self.assertFalse(result["properties"]["has_space"])
        self.assertFalse(result["properties"]["has_hyphen"])
        self.assertFalse(result["properties"]["has_apostrophe"])


class TestValidateEmail(unittest.TestCase):
    """Tests for the validate_email function."""

    def test_valid_emails(self):
        """Test validation of valid email addresses."""
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user_name@example.co.uk",
            "123@example.com",
            "user@subdomain.example.com"
        ]

        for email in valid_emails:
            result = validate_email(email)
            self.assertTrue(result["valid"], f"Email '{email}' should be valid")
            self.assertEqual(len(result["errors"]), 0, f"No errors should be reported for '{email}'")
            self.assertIn("username", result["properties"])
            self.assertIn("domain", result["properties"])

    def test_invalid_emails(self):
        """Test validation of invalid email addresses."""
        invalid_emails = [
            "",  # Empty
            "notanemail",  # No @ symbol
            "@example.com",  # No username
            "user@",  # No domain
            "user@example",  # No TLD
            "user@.com",  # Missing domain part
            "user@exam?ple.com"  # Invalid characters
        ]

        for email in invalid_emails:
            result = validate_email(email)
            self.assertFalse(result["valid"], f"Email '{email}' should be invalid")
            self.assertGreater(len(result["errors"]), 0, f"Errors should be reported for '{email}'")

    def test_email_properties(self):
        """Test that properties are correctly detected for emails."""
        # Test email with plus tag
        result = validate_email("user+tag@example.com")
        self.assertIn("has_plus", result["properties"])
        self.assertTrue(result["properties"]["has_plus"])

        # Test email with dot in username
        result = validate_email("user.name@example.com")
        self.assertIn("has_dot", result["properties"])
        self.assertTrue(result["properties"]["has_dot"])

        # Test TLD extraction
        result = validate_email("user@example.com")
        self.assertIn("tld", result["properties"])
        self.assertEqual(result["properties"]["tld"], "com")

        # Test domain extraction
        result = validate_email("user@example.co.uk")
        self.assertIn("domain", result["properties"])
        self.assertEqual(result["properties"]["domain"], "example.co.uk")

        # Test username extraction
        result = validate_email("user.name+tag@example.com")
        self.assertIn("username", result["properties"])
        self.assertEqual(result["properties"]["username"], "user.name+tag")


class TestValidatePhone(unittest.TestCase):
    """Tests for the validate_phone function."""

    def test_valid_russian_phones(self):
        """Test validation of valid Russian phone numbers."""
        valid_phones = [
            "79261234567",  # Mobile
            "74951234567",  # Moscow landline
            "78121234567",  # St. Petersburg landline
            "+79261234567",  # With plus
            "89261234567"  # With 8 prefix
        ]

        for phone in valid_phones:
            result = validate_phone(phone, region="RU")
            self.assertTrue(result["valid"], f"Phone '{phone}' should be valid for RU")
            self.assertEqual(len(result["errors"]), 0, f"No errors should be reported for '{phone}'")

            # Check properties for proper formatting
            if len(result["properties"]) > 0:
                if "country_code" in result["properties"]:
                    self.assertIn(result["properties"]["country_code"], ["7", "8"])
                if "area_code" in result["properties"]:
                    self.assertEqual(len(result["properties"]["area_code"]), 3)

    def test_valid_us_phones(self):
        """Test validation of valid US phone numbers."""
        valid_phones = [
            "12025551234",  # With country code
            "2025551234",  # Without country code
            "+12025551234"  # With plus
        ]

        for phone in valid_phones:
            result = validate_phone(phone, region="US")
            self.assertTrue(result["valid"], f"Phone '{phone}' should be valid for US")
            self.assertEqual(len(result["errors"]), 0, f"No errors should be reported for '{phone}'")

            # Check properties for proper formatting
            if len(phone) == 10:
                self.assertIn("area_code", result["properties"])
                self.assertEqual(len(result["properties"]["area_code"]), 3)
            elif len(phone) == 11 or len(phone) == 12:  # 12 for +1
                self.assertIn("country_code", result["properties"])
                self.assertEqual(result["properties"]["country_code"], "1")

    def test_invalid_phones(self):
        """Test validation of invalid phone numbers."""
        # Russian invalid phones
        invalid_ru_phones = [
            "",  # Empty
            "123456",  # Too short
            "7926123456X",  # Non-digit character
            "99261234567",  # Invalid prefix
            "7926123456789"  # Too long
        ]

        for phone in invalid_ru_phones:
            result = validate_phone(phone, region="RU")
            self.assertFalse(result["valid"], f"Phone '{phone}' should be invalid for RU")
            self.assertGreater(len(result["errors"]), 0, f"Errors should be reported for '{phone}'")

        # US invalid phones
        invalid_us_phones = [
            "123456",  # Too short
            "202555123X",  # Non-digit character
            "22025551234",  # Invalid country code
            "20255512345678"  # Too long
        ]

        for phone in invalid_us_phones:
            result = validate_phone(phone, region="US")
            self.assertFalse(result["valid"], f"Phone '{phone}' should be invalid for US")
            self.assertGreater(len(result["errors"]), 0, f"Errors should be reported for '{phone}'")

    def test_phone_properties(self):
        """Test that properties are correctly detected for phone numbers."""
        # Test Russian mobile with 7 prefix
        result = validate_phone("79261234567", region="RU")
        self.assertIn("country_code", result["properties"])
        self.assertEqual(result["properties"]["country_code"], "7")
        self.assertIn("area_code", result["properties"])
        self.assertEqual(result["properties"]["area_code"], "926")

        # Test US number with country code
        result = validate_phone("12025551234", region="US")
        self.assertIn("country_code", result["properties"])
        self.assertEqual(result["properties"]["country_code"], "1")
        self.assertIn("area_code", result["properties"])
        self.assertEqual(result["properties"]["area_code"], "202")
        self.assertIn("prefix", result["properties"])
        self.assertEqual(result["properties"]["prefix"], "555")
        self.assertIn("line_number", result["properties"])
        self.assertEqual(result["properties"]["line_number"], "1234")


class TestValidateFormat(unittest.TestCase):
    """Tests for the validate_format function."""

    def test_valid_formats(self):
        """Test validation of strings against valid formats."""
        test_cases = [
            ("12345", r"^\d{5}$"),  # US ZIP code
            ("ABC-1234", r"^[A-Z]{3}-\d{4}$"),  # Custom format
            ("test@example.com", r"^[^@]+@[^@]+\.[^@]+$")  # Simple email pattern
        ]

        for value, pattern in test_cases:
            result = validate_format(value, pattern)
            self.assertTrue(result["valid"], f"Value '{value}' should match pattern '{pattern}'")
            self.assertEqual(len(result["errors"]), 0)

    def test_invalid_formats(self):
        """Test validation of strings against invalid formats."""
        test_cases = [
            ("1234", r"^\d{5}$"),  # Too short for ZIP
            ("ABC-123", r"^[A-Z]{3}-\d{4}$"),  # Too few digits
            ("testexample.com", r"^[^@]+@[^@]+\.[^@]+$")  # Missing @ for email
        ]

        for value, pattern in test_cases:
            result = validate_format(value, pattern)
            self.assertFalse(result["valid"], f"Value '{value}' should not match pattern '{pattern}'")
            self.assertGreater(len(result["errors"]), 0)

    def test_empty_value(self):
        """Test validation of an empty string."""
        result = validate_format("", r"^.*$")
        self.assertFalse(result["valid"])
        self.assertIn("Value is empty", result["errors"][0])


class TestValidateIdNumber(unittest.TestCase):
    """Tests for the validate_id_number function."""

    def test_valid_russian_passport(self):
        """Test validation of valid Russian passport numbers."""
        valid_passports = [
            "4509 123456",  # With space
            "4509123456"  # Without space
        ]

        for passport in valid_passports:
            result = validate_id_number(passport, id_type="passport", region="RU")
            self.assertTrue(result["valid"], f"Passport '{passport}' should be valid")
            self.assertEqual(len(result["errors"]), 0)

    def test_invalid_russian_passport(self):
        """Test validation of invalid Russian passport numbers."""
        invalid_passports = [
            "",  # Empty
            "450 123456",  # Wrong series format
            "45091234",  # Too short number
            "4509 12345X"  # Non-digit characters
        ]

        for passport in invalid_passports:
            result = validate_id_number(passport, id_type="passport", region="RU")
            self.assertFalse(result["valid"], f"Passport '{passport}' should be invalid")
            self.assertGreater(len(result["errors"]), 0)

    def test_valid_russian_inn(self):
        """Test validation of valid Russian INN (tax identification number)."""
        result = validate_id_number("123456789012", id_type="inn", region="RU")
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_invalid_russian_inn(self):
        """Test validation of invalid Russian INN."""
        invalid_inns = [
            "",  # Empty
            "1234567890",  # Too short
            "12345678901X"  # Non-digit characters
        ]

        for inn in invalid_inns:
            result = validate_id_number(inn, id_type="inn", region="RU")
            self.assertFalse(result["valid"], f"INN '{inn}' should be invalid")
            self.assertGreater(len(result["errors"]), 0)

    def test_valid_us_ssn(self):
        """Test validation of valid US Social Security Numbers."""
        valid_ssns = [
            "123-45-6789",  # With hyphens
            "123456789"  # Without hyphens
        ]

        for ssn in valid_ssns:
            result = validate_id_number(ssn, id_type="ssn", region="US")
            self.assertTrue(result["valid"], f"SSN '{ssn}' should be valid")
            self.assertEqual(len(result["errors"]), 0)

    def test_invalid_us_ssn(self):
        """Test validation of invalid US Social Security Numbers."""
        invalid_ssns = [
            "",  # Empty
            "123-45-678",  # Too short
            "123-45-678X"  # Non-digit characters
        ]

        for ssn in invalid_ssns:
            result = validate_id_number(ssn, id_type="ssn", region="US")
            self.assertFalse(result["valid"], f"SSN '{ssn}' should be invalid")
            self.assertGreater(len(result["errors"]), 0)

    def test_unsupported_id_type(self):
        """Test validation of an unsupported ID type."""
        # For unsupported ID types, the function should return invalid with empty properties
        result = validate_id_number("12345", id_type="unknown", region="RU")
        self.assertFalse(result["valid"])
        self.assertEqual(result["properties"], {})


if __name__ == "__main__":
    unittest.main()