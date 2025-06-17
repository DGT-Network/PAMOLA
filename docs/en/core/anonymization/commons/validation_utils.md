# PAMOLA Core Documentation - Anonymization Validation Utilities

## Module Overview
`validation_utils.py` provides a set of validation functions for parameters used in anonymization operations. This module ensures data integrity, proper error handling, and parameter consistency across anonymization operations in the PAMOLA Core framework.

## Key Features
- Field existence and type validation (numeric, categorical, datetime)
- Strategy and parameter validation for different anonymization techniques
- Null-handling strategy validation
- Standardized error reporting with consistent messaging
- Type-safe validation functions with clear return values
- Support for logging validation failures at different severity levels

## Architecture
The validation utilities module sits within the commons package of the anonymization module and provides support functions for all anonymization operations:

```
pamola_core/anonymization/
├── commons/
│   ├── validation_utils.py   # This module
│   ├── metric_utils.py
│   ├── processing_utils.py
│   └── visualization_utils.py
├── generalization/
├── masking/
├── pseudonymization/
└── suppression/
```

The `validation_utils.py` module provides a collection of standalone functions that perform specific validation tasks. These functions are used by the operation classes to ensure parameters are valid before processing begins.

## Function Reference

### Field Validation Functions

#### `validate_field_exists`
Validates that a field exists in the DataFrame.

**Parameters:**
- `df`: `pd.DataFrame` - The DataFrame to check
- `field_name`: `str` - The name of the field to verify
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the field exists, False otherwise

**Logs:**
- ERROR if the field does not exist

#### `validate_numeric_field`
Validates that a field is numeric.

**Parameters:**
- `df`: `pd.DataFrame` - The DataFrame containing the field
- `field_name`: `str` - The name of the field to validate
- `allow_null`: `bool` - Whether to allow null values (default: True)
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the field is numeric and meets null criteria, False otherwise

**Logs:**
- ERROR if validation fails

#### `validate_categorical_field`
Validates that a field is categorical or string type.

**Parameters:**
- `df`: `pd.DataFrame` - The DataFrame containing the field
- `field_name`: `str` - The name of the field to validate
- `allow_null`: `bool` - Whether to allow null values (default: True)
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the field is categorical/string and meets null criteria, False otherwise

**Logs:**
- ERROR if validation fails

#### `validate_datetime_field`
Validates that a field is a datetime type.

**Parameters:**
- `df`: `pd.DataFrame` - The DataFrame containing the field
- `field_name`: `str` - The name of the field to validate
- `allow_null`: `bool` - Whether to allow null values (default: True)
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the field is datetime and meets null criteria, False otherwise

**Logs:**
- ERROR if validation fails

### Strategy Validation Functions

#### `validate_generalization_strategy`
Validates that a generalization strategy is supported.

**Parameters:**
- `strategy`: `str` - The strategy to validate
- `valid_strategies`: `List[str]` - List of valid strategies
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the strategy is valid, False otherwise

**Logs:**
- ERROR if validation fails

#### `validate_null_strategy`
Validates that a null handling strategy is supported.

**Parameters:**
- `strategy`: `str` - The strategy to validate
- `valid_strategies`: `List[str]` - List of valid strategies (default: None, will use ["PRESERVE", "EXCLUDE", "ERROR"])
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the strategy is valid, False otherwise

**Logs:**
- ERROR if validation fails

### Parameter Validation Functions

#### `validate_bin_count`
Validates that a bin count is valid.

**Parameters:**
- `bin_count`: `int` - The number of bins to validate
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the bin count is valid, False otherwise

**Logs:**
- ERROR if validation fails

#### `validate_precision`
Validates that a precision value is valid.

**Parameters:**
- `precision`: `int` - The precision value to validate
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the precision is valid, False otherwise

**Logs:**
- ERROR if validation fails

#### `validate_range_limits`
Validates that range limits are valid.

**Parameters:**
- `range_limits`: `Tuple[float, float]` - The (min, max) limits to validate
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the range limits are valid, False otherwise

**Logs:**
- ERROR if validation fails

#### `validate_output_field_name`
Validates output field name based on the mode.

**Parameters:**
- `df`: `pd.DataFrame` - DataFrame to check
- `output_field_name`: `str` - Output field name to validate
- `mode`: `str` - Mode of operation ("REPLACE" or "ENRICH")
- `logger_instance`: `Optional[logging.Logger]` - Logger instance to use for logging (default: module logger)

**Returns:**
- `bool` - True if the output field name is valid, False otherwise

**Logs:**
- ERROR if validation fails
- WARNING if output field already exists

### Error Handling Utility

#### `get_validation_error_result`
Creates a standardized validation error result.

**Parameters:**
- `error_message`: `str` - The error message
- `field_name`: `str` - The field name associated with the error (optional)

**Returns:**
- `Dict[str, Any]` - Validation error result with standardized structure

## Usage Examples

### Basic Field Validation

```python
import pandas as pd
from pamola_core.anonymization.commons.validation_utils import validate_field_exists, validate_numeric_field

# Create a sample DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'age': [25, 30, None, 40, 45],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
})

# Validate field existence
if not validate_field_exists(df, 'income'):
    print("Income field does not exist in the DataFrame")

# Validate numeric field
if validate_numeric_field(df, 'age', allow_null=True):
    print("Age is a valid numeric field with allowed nulls")
```

### Strategy and Parameter Validation

```python
from pamola_core.anonymization.commons.validation_utils import (
    validate_generalization_strategy, 
    validate_bin_count, 
    validate_range_limits
)

# Validate generalization strategy
valid_strategies = ["binning", "rounding", "range"]
if not validate_generalization_strategy("clustering", valid_strategies):
    print("Clustering is not a valid generalization strategy")

# Validate bin count
if validate_bin_count(10):
    print("Bin count is valid")

# Validate range limits
if validate_range_limits((0, 100)):
    print("Range limits are valid")
```

### Output Field Validation

```python
from pamola_core.anonymization.commons.validation_utils import validate_output_field_name

# Validate output field name for ENRICH mode
if not validate_output_field_name(df, "anon_age", "ENRICH"):
    print("Output field name is not valid for ENRICH mode")

# Validate output field name for REPLACE mode
if validate_output_field_name(df, None, "REPLACE"):
    print("Output field name is valid for REPLACE mode")
```

### Error Result Generation

```python
from pamola_core.anonymization.commons.validation_utils import get_validation_error_result

# Generate a validation error result
error = get_validation_error_result("Invalid bin count value", "age")
print(f"Error Type: {error['error_type']}")
print(f"Error Message: {error['error']}")
print(f"Field: {error['field']}")
```

## Limitations and Constraints

- The validation functions only validate parameters and do not modify the data
- Error messages are logged but exceptions are not automatically raised
- Validation is performed at a field level and does not consider dependencies between fields
- Datetime validation can be processor-intensive for large datasets when checking convertibility

## Integration with Other Modules

The validation utilities module is primarily used by:

1. `base_anonymization_op.py` - For validating common operation parameters
2. Specific anonymization operations - For validating operation-specific parameters
3. Task configuration validation - For validating configuration files before operation execution

## Best Practices

1. Always validate fields before processing to prevent runtime errors
2. Use the appropriate validation function for the field type
3. Handle validation failures gracefully
4. Log validation errors with context for easier debugging
5. Chain validations when multiple conditions need to be checked