# fake_data.commons.validators Module Documentation

## Overview

The `validators.py` module is a critical component of the `fake_data` package, providing robust validation functionality for various data types encountered during anonymization processes. This module implements specialized validation logic for personal names, email addresses, phone numbers, and identification documents with region-specific rule enforcement. It ensures that generated synthetic data maintains the same structural integrity and format constraints as original data, thereby enhancing the realism and utility of anonymized datasets.

## Architecture

The module adopts a functional approach with validation functions organized by data type categories. Each validator follows a consistent pattern, returning detailed validation results with error messages and extracted data properties.

```
┌─────────────────────────────────────────────────────────┐
│                  validators.py                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ │
│  │ Personal Data   │  │ Contact Data    │  │   ID     │ │
│  │ Validators      │  │ Validators      │  │Validators│ │
│  │                 │  │                 │  │          │ │
│  │ • validate_name │  │ • validate_email│  │ • validate│ │
│  │                 │  │ • validate_phone│  │   _id_num │ │
│  └─────────────────┘  └─────────────────┘  └──────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Generic Validators                      ││
│  │                                                      ││
│  │             • validate_format                        ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

The module integrates with the broader fake_data package by:

1. Supporting the validation requirements of data generators
2. Providing format verification for mapping operations
3. Ensuring consistency across generated synthetic datasets
4. Enforcing region-specific format constraints

## Key Capabilities

The module provides the following validation capabilities:

1. **Personal Name Validation**: Checks for proper formatting, length constraints, and language-specific character sets
2. **Email Address Validation**: Verifies RFC 5322 compliance and structural integrity
3. **Phone Number Validation**: Region-specific phone number format validation with component extraction
4. **ID Number Validation**: Format verification for various national identification numbers
5. **Format Pattern Validation**: Generic validation against specified regular expression patterns

## Key Components

### Personal Data Validators

| Function | Purpose |
|----------|---------|
| `validate_name` | Validates personal names with language-specific rules |

### Contact Data Validators

| Function | Purpose |
|----------|---------|
| `validate_email` | Verifies email address format and extracts components |
| `validate_phone` | Validates phone numbers using region-specific rules |

### Identification Validators

| Function | Purpose |
|----------|---------|
| `validate_id_number` | Validates ID documents like passports, SSNs, etc. |

### Generic Validators

| Function | Purpose |
|----------|---------|
| `validate_format` | Validates strings against regular expression patterns |

## Usage Examples

### Name Validation

```python
from pamola_core.fake_data.commons.validators import validate_name

# Validate a Russian name
result = validate_name("Иван Петров", language="ru")
# result: {
#   "valid": True,
#   "errors": [],
#   "properties": {
#     "length": 11,
#     "has_space": True,
#     "has_hyphen": False,
#     "has_apostrophe": False
#   }
# }

# Validate an invalid name
result = validate_name("John123", language="en")
# result: {
#   "valid": False,
#   "errors": ["Name contains invalid characters"],
#   "properties": {
#     "length": 7,
#     "has_space": False,
#     "has_hyphen": False,
#     "has_apostrophe": False
#   }
# }
```

### Email Validation

```python
from pamola_core.fake_data.commons.validators import validate_email

# Validate a valid email
result = validate_email("user.name+tag@example.com")
# result: {
#   "valid": True,
#   "errors": [],
#   "properties": {
#     "username": "user.name+tag",
#     "domain": "example.com",
#     "has_plus": True,
#     "has_dot": True,
#     "tld": "com"
#   }
# }

# Validate an invalid email
result = validate_email("invalid-email@")
# result: {
#   "valid": False,
#   "errors": ["Invalid email format"],
#   "properties": {}
# }
```

### Phone Validation

```python
from pamola_core.fake_data.commons.validators import validate_phone

# Validate a Russian phone number
result = validate_phone("79261234567", region="RU")
# result: {
#   "valid": True,
#   "errors": [],
#   "properties": {
#     "country_code": "7",
#     "area_code": "926",
#     "number": "1234567"
#   }
# }

# Validate a US phone number
result = validate_phone("12025551234", region="US")
# result: {
#   "valid": True,
#   "errors": [],
#   "properties": {
#     "country_code": "1",
#     "area_code": "202",
#     "prefix": "555",
#     "line_number": "1234"
#   }
# }
```

### ID Number Validation

```python
from pamola_core.fake_data.commons.validators import validate_id_number

# Validate a Russian passport
result = validate_id_number("4509 123456", id_type="passport", region="RU")
# result: {
#   "valid": True,
#   "errors": [],
#   "properties": {}
# }

# Validate a US Social Security Number
result = validate_id_number("123-45-6789", id_type="ssn", region="US")
# result: {
#   "valid": True,
#   "errors": [],
#   "properties": {}
# }
```

### Format Pattern Validation

```python
from pamola_core.fake_data.commons.validators import validate_format

# Validate a specific pattern (ZIP code)
result = validate_format("12345", r"^\d{5}$")
# result: {
#   "valid": True,
#   "errors": []
# }
```

## Parameters and Return Values

### Name Validation

```python
def validate_name(
    name: str,            # Name to validate
    language: str = "ru"  # Language code for validation rules
) -> Dict[str, Any]:      # Validation results
    # Returns: {
    #   "valid": bool,     # Whether validation passed
    #   "errors": List[str], # Error messages if invalid
    #   "properties": Dict  # Properties of the name
    # }
```

### Email Validation

```python
def validate_email(
    email: str            # Email address to validate
) -> Dict[str, Any]:      # Validation results
    # Returns: {
    #   "valid": bool,     # Whether validation passed 
    #   "errors": List[str], # Error messages if invalid
    #   "properties": Dict  # Properties of the email
    # }
```

### Phone Validation

```python
def validate_phone(
    phone: str,           # Phone number to validate
    region: str = "RU"    # Region code for validation rules
) -> Dict[str, Any]:      # Validation results
    # Returns: {
    #   "valid": bool,     # Whether validation passed
    #   "errors": List[str], # Error messages if invalid
    #   "properties": Dict  # Extracted phone components
    # }
```

### ID Number Validation

```python
def validate_id_number(
    id_number: str,       # ID number to validate
    id_type: str,         # Type of ID (passport, ssn, etc.)
    region: str = "RU"    # Region code for validation rules
) -> Dict[str, Any]:      # Validation results
    # Returns: {
    #   "valid": bool,     # Whether validation passed
    #   "errors": List[str], # Error messages if invalid
    #   "properties": Dict  # Properties of the ID
    # }
```

### Format Validation

```python
def validate_format(
    value: str,           # String to validate
    format_pattern: str   # Regular expression pattern
) -> Dict[str, Any]:      # Validation results
    # Returns: {
    #   "valid": bool,     # Whether validation passed
    #   "errors": List[str]  # Error messages if invalid
    # }
```

## Integration with Fake Data Package

The `validators.py` module serves critical functions within the broader `fake_data` package:

1. **Pre-generation Validation**: Ensures input data meets format requirements before processing
2. **Post-generation Verification**: Confirms that generated synthetic data maintains proper formatting
3. **Quality Assurance**: Provides metrics on data validity for anonymization reports
4. **Consistency Enforcement**: Helps maintain internal consistency across related data fields

The validators are designed to be used at multiple stages of the data anonymization process:

```
┌───────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│ Original Data │───►│ Pre-Validation  │───►│ Synthetic    │───►│ Validation   │
│               │    │ (validators.py) │    │ Data         │    │ (validators) │
└───────────────┘    └─────────────────┘    │ Generation   │    └───────┬──────┘
                                            └──────────────┘            │
                                                   ▲                    │
                                                   │                    ▼
                                            ┌──────┴───────┐    ┌──────────────┐
                                            │ Format       │    │ Quality      │
                                            │ Adjustment   │◄───┤ Reports      │
                                            └──────────────┘    └──────────────┘
```

## Region-Specific Validation

The module implements region-specific validation rules for multiple data types:

### Russian (RU) Validation Specifics

- **Names**: Cyrillic character validation, patronymic format checking
- **Phone Numbers**: Validation for 11-digit format starting with 7 or 8
- **Passport Numbers**: 4-digit series followed by 6-digit number format validation
- **INN**: 12-digit taxpayer identification number validation

### US Validation Specifics

- **Names**: Latin character validation with apostrophe and hyphen support
- **Phone Numbers**: 10-digit format with optional country code (1)
- **SSN**: 9-digit Social Security Number with optional hyphen formatting

## Extension Mechanisms

The module is designed for extensibility, supporting:

1. **New Data Types**: Additional validator functions can be added following the established pattern
2. **New Regions**: Region-specific validation logic can be extended to support more countries
3. **Custom Patterns**: The generic validator can be used with custom regular expressions

To add support for a new region or data type:

1. Update the relevant validator function with region-specific logic
2. Add appropriate pattern matching or structural validation 
3. Ensure the return structure maintains consistency with existing validators

## Performance Considerations

The validator functions are optimized for performance:

1. **Early Termination**: Validation stops at the first critical error
2. **Efficient Regex**: Regular expressions are designed for optimal performance
3. **Minimal Dependencies**: Functions use standard library components where possible
4. **Reusable Components**: Common validation logic is shared between validators

## Error Handling

All validators follow a consistent error handling approach:

1. **Detailed Error Messages**: Specific, actionable error messages
2. **No Exceptions**: Errors are returned in the result dictionary rather than raising exceptions
3. **Multiple Error Reporting**: All validation errors are collected and returned

## Best Practices for Using Validators

When using the validators module:

1. **Check Valid Flag First**: Always check the "valid" flag before proceeding
2. **Handle Empty Results**: Check for empty values before validation
3. **Process Errors**: Use the detailed error messages for user feedback or logs
4. **Extract Properties**: Leverage the extracted properties for further processing

```python
def process_email(email):
    result = validate_email(email)
    
    if not result["valid"]:
        # Handle invalid email
        log_errors(result["errors"])
        return False
        
    # Use extracted properties
    domain = result["properties"]["domain"]
    username = result["properties"]["username"]
    
    # Continue processing...
    return True
```

## Conclusion

The `validators.py` module provides essential validation capabilities for the `fake_data` package, ensuring that synthetic data maintains the same format constraints and structural properties as original data. With comprehensive support for various data types and region-specific validation, it plays a critical role in producing realistic and consistent anonymized datasets.

The module's consistent return structure, detailed error reporting, and efficient implementation make it a valuable tool for data quality assurance throughout the anonymization process. As the package evolves, the validators module can be extended to support additional data types and regional formats, enhancing the system's capabilities for diverse anonymization scenarios.