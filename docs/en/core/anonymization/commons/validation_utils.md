# PAMOLA.CORE validation_utils.py Module Documentation

## Overview

The `validation_utils.py` module serves as a **facade** for the PAMOLA.CORE validation framework, providing backward compatibility while offering access to the new modular validation system. It acts as a bridge between legacy code and the refactored validation architecture, ensuring smooth migration paths for existing implementations.

**Version:** 3.1.0  
**Status:** Stable  
**Package:** `pamola_core.anonymization.commons`  
**Role:** Validation Facade & Backward Compatibility Layer  
**License:** BSD 3-Clause

## Architecture

### Module Purpose

The module serves three primary purposes:

1. **Backward Compatibility**: Maintains the original API for existing code that relies on the old monolithic validation functions
2. **Facade Pattern**: Provides a simplified interface to the new modular validation system located at `pamola_core.anonymization.commons.validation/`
3. **Migration Support**: Offers both legacy and modern approaches, allowing gradual migration to the new system

### Architectural Position

```
pamola_core/anonymization/commons/
├── validation_utils.py         # This module (Facade)
├── validation/                 # New modular validation system
│   ├── __init__.py
│   ├── base.py                # Core infrastructure
│   ├── decorators.py          # Validation decorators
│   ├── exceptions.py          # Custom exceptions
│   ├── field_validators.py    # Field type validators
│   ├── file_validators.py     # File/path validators
│   ├── strategy_validators.py # Strategy validators
│   └── type_validators.py     # Specialized validators
├── metric_utils.py
├── visualization_utils.py
├── privacy_metric_utils.py
└── data_utils.py
```

### Module Structure

```python
# 1. Imports from new modular system
from .validation import (
    # Core classes
    ValidationResult, BaseValidator, CompositeValidator,
    # Validators
    NumericFieldValidator, CategoricalFieldValidator, ...
    # Utilities
    check_field_exists, validate_strategy, ...
)

# 2. Legacy Support Layer
class LegacyValidationSupport:
    """Provides backward compatibility"""

# 3. Deprecated Functions (wrapped for compatibility)
def validate_field_exists(...)  # DEPRECATED
def validate_numeric_field(...)  # DEPRECATED
...

# 4. New Factory Functions
def create_validator(...)
def create_validation_pipeline(...)
def validate_dataframe_schema(...)
```

## Migration Guide

### Old vs New Approach

```python
# OLD (Deprecated but still works)
from pamola_core.anonymization.commons.validation_utils import validate_numeric_field
is_valid = validate_numeric_field(df, 'age', min_value=0, max_value=150)

# NEW (Recommended)
from pamola_core.anonymization.commons.validation import NumericFieldValidator
validator = NumericFieldValidator(min_value=0, max_value=150)
result = validator.validate(df['age'])
is_valid = result.is_valid
```

### Migration Benefits

1. **More Flexible**: Validators can be composed and reused
2. **Better Error Handling**: Structured `ValidationResult` objects
3. **Cacheable**: Results can be cached for performance
4. **Extensible**: Easy to create custom validators

## API Reference

### New Factory Functions (Recommended)

#### create_validator

```python
def create_validator(field_type: str, **params) -> BaseValidator
```

**Description**: Factory function for creating field validators with a simplified interface.

**Parameters**:
- `field_type`: Type of validator to create
  - Basic types: `'numeric'`, `'categorical'`, `'datetime'`, `'boolean'`, `'text'`
  - Specialized: `'network'`, `'geographic'`, `'temporal'`, `'financial'`
  - File types: `'file'`, `'directory'`, `'json'`, `'csv'`, `'hierarchy'`
- `**params`: Parameters specific to the validator type

**Example**:
```python
# Create numeric validator
num_validator = create_validator('numeric', min_value=0, max_value=100)

# Create file validator
file_validator = create_validator('file', must_exist=True, valid_extensions=['.csv'])
```

#### create_validation_pipeline

```python
def create_validation_pipeline(*validators: BaseValidator, 
                             stop_on_first_error: bool = False) -> CompositeValidator
```

**Description**: Creates a validation pipeline from multiple validators.

**Example**:
```python
email_pipeline = create_validation_pipeline(
    create_validator('text', pattern=r'^[^@]+@[^@]+\.[^@]+$'),
    create_validator('text', min_length=5, max_length=254),
    stop_on_first_error=True
)
```

#### validate_dataframe_schema

```python
def validate_dataframe_schema(df: pd.DataFrame,
                            schema: Dict[str, Dict[str, Any]],
                            strict: bool = False) -> ValidationResult
```

**Description**: Validates entire DataFrame against a schema definition.

**Example**:
```python
schema = {
    'age': {'type': 'numeric', 'min_value': 0, 'max_value': 150},
    'email': {'type': 'text', 'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
    'salary': {'type': 'numeric', 'min_value': 0, 'allow_null': False}
}
result = validate_dataframe_schema(df, schema, strict=True)
```

### Deprecated Functions (Legacy Support)

> **⚠️ DEPRECATION NOTICE**: The following functions are deprecated and maintained only for backward compatibility. Use the new validator classes or factory functions instead.

#### validate_field_exists (DEPRECATED)

```python
def validate_field_exists(df: pd.DataFrame, field_name: str,
                         logger_instance: Optional[logging.Logger] = None) -> bool
```

**Deprecated**: Use `check_field_exists()` or `@requires_field` decorator instead.

**Replacement**:
```python
# Old way (deprecated)
exists = validate_field_exists(df, 'field_name')

# New way
exists = check_field_exists(df, 'field_name')
```

#### validate_numeric_field (DEPRECATED)

```python
def validate_numeric_field(df: pd.DataFrame, field_name: str, 
                          allow_null: bool = True,
                          min_value: Optional[float] = None, 
                          max_value: Optional[float] = None,
                          logger_instance: Optional[logging.Logger] = None) -> bool
```

**Deprecated**: Use `NumericFieldValidator` for more features.

**Replacement**:
```python
# Old way (deprecated)
is_valid = validate_numeric_field(df, 'age', min_value=0, max_value=150)

# New way
validator = NumericFieldValidator(min_value=0, max_value=150)
result = validator.validate(df['age'])
is_valid = result.is_valid
```

#### validate_categorical_field (DEPRECATED)

```python
def validate_categorical_field(df: pd.DataFrame, field_name: str,
                             allow_null: bool = True,
                             min_categories: Optional[int] = None,
                             max_categories: Optional[int] = None,
                             valid_categories: Optional[List[str]] = None,
                             min_frequency_threshold: Optional[int] = None,
                             check_distribution: bool = False,
                             logger_instance: Optional[logging.Logger] = None
                             ) -> Tuple[bool, Dict[str, Any]]
```

**Deprecated**: Use `CategoricalFieldValidator` for better functionality.

#### validate_datetime_field (DEPRECATED)

```python
def validate_datetime_field(df: pd.DataFrame, field_name: str, 
                           allow_null: bool = True,
                           min_date: Optional[pd.Timestamp] = None,
                           max_date: Optional[pd.Timestamp] = None,
                           logger_instance: Optional[logging.Logger] = None) -> bool
```

**Deprecated**: Use `DateTimeFieldValidator` instead.

#### validate_generalization_strategy (DEPRECATED)

```python
def validate_generalization_strategy(strategy: str, valid_strategies: List[str],
                                   logger_instance: Optional[logging.Logger] = None) -> bool
```

**Deprecated**: Use `validate_strategy()` from the new system.

### Specialized Validation (Legacy Wrappers)

> **⚠️ DEPRECATED**: Use the corresponding validator classes directly.

- `validate_geographic_data()` → Use `GeographicValidator`
- `validate_temporal_sequence()` → Use `TemporalValidator`
- `validate_network_identifiers()` → Use `NetworkValidator`
- `validate_financial_data()` → Use `FinancialValidator`
- `validate_file_path()` → Use `FilePathValidator`
- `validate_directory_path()` → Use `DirectoryPathValidator`

## Re-exported Components

The module re-exports key components from the new validation system:

### Core Classes
- `ValidationResult`: Structured validation result
- `BaseValidator`: Abstract base for validators
- `CompositeValidator`: Compose multiple validators
- `ValidationContext`: Validation context management
- `ValidationCache`: TTL-based result caching

### Field Validators
- `NumericFieldValidator`
- `CategoricalFieldValidator`
- `DateTimeFieldValidator`
- `BooleanFieldValidator`
- `TextFieldValidator`

### File Validators
- `FilePathValidator`
- `DirectoryPathValidator`
- `HierarchyFileValidator`
- `JSONFileValidator`
- `CSVFileValidator`

### Specialized Validators
- `NetworkValidator`
- `GeographicValidator`
- `TemporalValidator`
- `FinancialValidator`

### Exceptions
- `ValidationError`
- `FieldNotFoundError`
- `FieldTypeError`
- `FieldValueError`

### Decorators
- `@validation_handler`: Convert exceptions to ValidationResult
- `@standard_validator`: Common validation patterns
- `@requires_field`: Ensure field exists

## Usage Examples

### Modern Approach (Recommended)

```python
from pamola_core.anonymization.commons.validation_utils import (
    create_validator,
    create_validation_pipeline,
    validate_dataframe_schema
)

# 1. Simple field validation
age_validator = create_validator('numeric', min_value=0, max_value=150)
result = age_validator.validate(df['age'])

# 2. Pipeline validation
email_validator = create_validation_pipeline(
    create_validator('text', pattern=r'^[^@]+@[^@]+\.[^@]+$'),
    create_validator('text', min_length=5, max_length=254)
)
result = email_validator.validate(df['email'])

# 3. Full schema validation
schema = {
    'age': {'type': 'numeric', 'min_value': 0, 'max_value': 150},
    'email': {'type': 'text', 'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
}
result = validate_dataframe_schema(df, schema)
```

### Legacy Approach (Still Supported)

```python
from pamola_core.anonymization.commons.validation_utils import (
    validate_field_exists,
    validate_numeric_field,
    validate_categorical_field
)

# Old-style validation (works but deprecated)
if validate_field_exists(df, 'age'):
    is_valid = validate_numeric_field(df, 'age', min_value=0)
```

## Best Practices

1. **Use Factory Functions**: Prefer `create_validator()` over deprecated functions
2. **Check ValidationResult**: Always examine the full result, not just boolean
3. **Handle Warnings**: Even when validation passes, check warnings
4. **Migrate Gradually**: Update critical paths first, legacy code can wait
5. **Use Type-Specific Validators**: More features than generic validation
6. **Leverage Decorators**: Reduce boilerplate in validation logic

## Version History

- **3.1.0** - Fixed import issues, improved error handling, enhanced documentation
- **3.0.0** - Complete refactoring into modular framework with facade pattern
- **2.1.0** - Enhanced categorical validation and hierarchy support
- **2.0.0** - Added conditional processing and specialized validators
- **1.0.0** - Initial monolithic implementation

