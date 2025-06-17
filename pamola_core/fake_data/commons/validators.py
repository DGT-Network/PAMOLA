"""
Validators for various data types used in fake_data package.

This module provides validation functions for checking the correctness
of various data types: names, email addresses, phone numbers, etc.
"""

import re
from typing import Dict, Any



def validate_name(name: str, language: str = "ru") -> Dict[str, Any]:
    """
    Validates a personal name.

    Parameters:
    -----------
    name : str
        Name to validate
    language : str
        Language code (default: "ru")

    Returns:
    --------
    Dict[str, Any]
        Validation results with keys:
        - valid: bool - whether the name is valid
        - errors: list - list of errors if invalid
        - properties: dict - additional properties of the name
    """
    result = {
        "valid": False,
        "errors": [],
        "properties": {}
    }

    # Check for empty or None
    if not name:
        result["errors"].append("Name is empty")
        return result

    # Check length
    if len(name) < 2:
        result["errors"].append("Name is too short (min 2 characters)")
    elif len(name) > 50:
        result["errors"].append("Name is too long (max 50 characters)")

    # Check for special characters
    allowed_special = "-' "  # Hyphen, apostrophe, space
    if not all(c.isalpha() or c in allowed_special for c in name):
        result["errors"].append("Name contains invalid characters")

    # Check first letter is capitalized (for most languages)
    if name and not name[0].isupper():
        result["errors"].append("Name should start with a capital letter")

    # Set valid flag if no errors
    result["valid"] = len(result["errors"]) == 0

    # Set properties
    result["properties"] = {
        "length": len(name),
        "has_space": " " in name,
        "has_hyphen": "-" in name,
        "has_apostrophe": "'" in name,
    }

    return result




def validate_email(email: str) -> Dict[str, Any]:
    """
    Validates an email address.

    Parameters:
    -----------
    email : str
        Email address to validate

    Returns:
    --------
    Dict[str, Any]
        Validation results with keys:
        - valid: bool - whether the email is valid
        - errors: list - list of errors if invalid
        - properties: dict - additional properties of the email
    """
    result: Dict[str, Any] = {
        "valid": False,
        "errors": [],
        "properties": {}
    }

    # Check for empty or None
    if not email:
        result["errors"].append("Email is empty")
        return result

    # Basic email regex pattern (simplified for demonstration)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    # Check pattern
    if not re.match(email_pattern, email):
        result["errors"].append("Invalid email format")

    # Set valid flag if no errors
    result["valid"] = not result["errors"]

    # Extract properties if valid
    if result["valid"]:
        try:
            username, domain = email.split('@')
            tld = domain.split('.')[-1]
            result["properties"] = {
                "username": username,
                "domain": domain,
                "has_plus": "+" in username,
                "has_dot": "." in username,
                "tld": tld
            }
        except IndexError:
            result["valid"] = False
            result["errors"].append("Invalid email format (could not parse domain)")
        except ValueError:
            result["valid"] = False
            result["errors"].append("Invalid email format (missing '@' symbol)")

    return result




def validate_phone(phone: str, region: str = "RU") -> Dict[str, Any]:
    """
    Validates a phone number.

    Parameters:
    -----------
    phone : str
        Phone number to validate
    region : str
        Region code (default: "RU")

    Returns:
    --------
    Dict[str, Any]
        Validation results with keys:
        - valid: bool - whether the phone number is valid
        - errors: list - list of errors if invalid
        - properties: dict - additional properties of the phone number
    """
    result: Dict[str, Any] = {
        "valid": False,
        "errors": [],
        "properties": {}
    }

    # Check for empty or None
    if not phone:
        result["errors"].append("Phone number is empty")
        return result

    # Remove non-digit characters for analysis
    digits_only = ''.join(c for c in phone if c.isdigit())

    # Region-specific validation
    if region == "RU":
        # Russian phone number validation (with country code)
        if len(digits_only) != 11:
            result["errors"].append("Russian phone number should have 11 digits")
        elif not (digits_only.startswith('7') or digits_only.startswith('8')):
            result["errors"].append("Russian phone number should start with 7 or 8")

        # Extract properties
        if len(digits_only) == 11:
            properties_dict: Dict[str, str] = {
                "country_code": digits_only[0],
                "area_code": digits_only[1:4],
                "number": digits_only[4:11]
            }
            result["properties"] = properties_dict

    elif region == "US":
        # US phone number validation
        if len(digits_only) != 10 and len(digits_only) != 11:
            result["errors"].append("US phone number should have 10 or 11 digits")
        elif len(digits_only) == 11 and digits_only[0] != '1':
            result["errors"].append("US phone number with country code should start with 1")

        # Extract properties
        if len(digits_only) == 10:
            properties_dict: Dict[str, str] = {
                "area_code": digits_only[0:3],
                "prefix": digits_only[3:6],
                "line_number": digits_only[6:10]
            }
            result["properties"] = properties_dict

        elif len(digits_only) == 11:
            properties_dict: Dict[str, str] = {
                "country_code": digits_only[0],
                "area_code": digits_only[1:4],
                "prefix": digits_only[4:7],
                "line_number": digits_only[7:11]
            }
            result["properties"] = properties_dict

    # Set valid flag if no errors
    result["valid"] = not result["errors"]

    return result


def validate_format(value: str, format_pattern: str) -> Dict[str, Any]:
    """
    Validates a string against a specified format pattern.

    Parameters:
    -----------
    value : str
        String to validate
    format_pattern : str
        Regular expression pattern to match against

    Returns:
    --------
    Dict[str, Any]
        Validation results with keys:
        - valid: bool - whether the value matches the pattern
        - errors: list - list of errors if invalid
    """
    result = {
        "valid": False,
        "errors": []
    }

    # Check for empty or None
    if not value:
        result["errors"].append("Value is empty")
        return result

    # Check pattern match
    if not re.match(format_pattern, value):
        result["errors"].append(f"Value does not match the required format: {format_pattern}")

    # Set valid flag if no errors
    result["valid"] = len(result["errors"]) == 0

    return result


def validate_id_number(id_number: str, id_type: str, region: str = "RU") -> Dict[str, Any]:
    """
    Validates an identification number.

    Parameters:
    -----------
    id_number : str
        ID number to validate
    id_type : str
        Type of ID (e.g., "passport", "ssn", "inn")
    region : str
        Region code (default: "RU")

    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    result = {
        "valid": False,
        "errors": [],
        "properties": {}
    }

    # Check for empty or None
    if not id_number:
        result["errors"].append("ID number is empty")
        return result

    # Validation based on ID type and region
    if region == "RU":
        if id_type == "passport":
            # Russian passport number: 4 digits series, 6 digits number
            if not re.match(r'^\d{4}\s?\d{6}$', id_number):
                result["errors"].append("Russian passport should have format: 1234 567890")

        elif id_type == "inn":
            # Russian INN (Individual Taxpayer Number): 12 digits
            if not re.match(r'^\d{12}$', id_number):
                result["errors"].append("Russian INN should be 12 digits")

        else:
            result["errors"].append(f"Region: {region}. Unsupported id_type: {id_type}")

    elif region == "US":
        if id_type == "ssn":
            # US Social Security Number: 9 digits, often written as XXX-XX-XXXX
            if not re.match(r'^\d{3}-?\d{2}-?\d{4}$', id_number):
                result["errors"].append("US SSN should have format: 123-45-6789")

        else:
            result["errors"].append(f"Region: {region}. Unsupported id_type: {id_type}")

    # Set valid flag if no errors
    result["valid"] = len(result["errors"]) == 0

    return result