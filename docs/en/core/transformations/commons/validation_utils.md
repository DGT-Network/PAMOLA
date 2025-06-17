# PAMOLA Core: `validation_utils` Module

---

## Overview

The `validation_utils` module is a core component of the PAMOLA Core framework, providing robust validation utilities for transformation operations. Its primary purpose is to ensure the integrity of parameters, schema correctness, and enforcement of data constraints throughout data processing pipelines. This module is designed to be used by transformation operators, data engineers, and pipeline developers to validate DataFrame structures, parameter types, and enforce business rules, thereby reducing runtime errors and improving data quality.

---

## Key Features

- **Field Existence Validation**: Check if required columns exist in a DataFrame.
- **Field Type Validation**: Ensure DataFrame columns have expected data types.
- **Parameter Validation**: Validate presence and types of operation parameters.
- **Constraint Enforcement**: Enforce constraints such as not-null, min/max values, allowed values, uniqueness, and regex patterns.
- **Aggregation and Grouping Validation**: Validate group-by and aggregation fields for transformation operations.
- **Join Type Validation**: Ensure only supported join types are used.
- **Comprehensive Logging**: All validation failures and warnings are logged for traceability.

---

## Dependencies

### Standard Library
- `logging`
- `typing` (Any, Callable, Dict, List, Optional, Tuple, Type, Union)

### External Libraries
- `pandas`

### Internal Modules
- None (self-contained, but designed to be used by other PAMOLA Core modules)

---

## Exception Classes

This module does not define custom exception classes, but raises standard Python exceptions with detailed error messages. The following exceptions may be raised:

- **TypeError**: Raised when input arguments are of incorrect types.
- **ValueError**: Raised when required fields or parameters are missing, or when invalid values are provided.

### Example: Handling Exceptions

```python
import pandas as pd
from pamola_core.transformations.commons import validation_utils

# Example DataFrame
sample_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

try:
    # Will raise ValueError if 'c' is missing
    validation_utils.validate_dataframe(sample_df, ['a', 'c'])
except ValueError as e:
    print(f"Validation failed: {e}")
```

**When exceptions are raised:**
- `TypeError` is raised if the input DataFrame or parameter list/dict is not of the expected type.
- `ValueError` is raised if required columns or parameters are missing, or if an invalid join type is specified.

---

## Main Functions

### 1. `validate_fields_exist`

```python
def validate_fields_exist(
    df: pd.DataFrame,
    required_fields: List[str]
) -> Tuple[bool, Optional[List[str]]]
```
**Parameters:**
- `df`: The DataFrame to check.
- `required_fields`: List of required field names.

**Returns:**
- `(True, None)` if all fields exist.
- `(False, [missing_fields])` if any fields are missing.

**Raises:**
- `TypeError` if arguments are of incorrect type.

---

### 2. `validate_field_types`

```python
def validate_field_types(
    df: pd.DataFrame,
    field_types: Dict[str, str]
) -> Tuple[bool, Optional[Dict[str, str]]]
```
**Parameters:**
- `df`: The DataFrame to check.
- `field_types`: Mapping of field names to expected type strings (e.g., 'numeric', 'datetime').

**Returns:**
- `(True, None)` if all types match.
- `(False, {field: "expected vs actual"})` if mismatches are found.

**Raises:**
- `TypeError`, `ValueError`

---

### 3. `validate_parameters`

```python
def validate_parameters(
    parameters: Dict[str, Any],
    required_params: List[str],
    param_types: Dict[str, Type]
) -> Tuple[bool, Optional[List[str]]]
```
**Parameters:**
- `parameters`: Dictionary of parameters to validate.
- `required_params`: List of required parameter names.
- `param_types`: Mapping of parameter names to expected types (can use `Union`).

**Returns:**
- `(True, None)` if all parameters are valid.
- `(False, list_of_error_messages)` otherwise.

**Raises:**
- `TypeError`

---

### 4. `validate_constraints`

```python
def validate_constraints(
    df: pd.DataFrame,
    constraints: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]
```
**Parameters:**
- `df`: The DataFrame to validate.
- `constraints`: Dictionary of field constraints (e.g., not_null, min, max, allowed_values, unique, regex).

**Returns:**
- Dictionary of violations: `{field: {constraint: {...}}}`

**Raises:**
- `TypeError`, `ValueError`

---

### 5. `validate_dataframe`

```python
def validate_dataframe(
    df: pd.DataFrame,
    columns: List[str]
) -> None
```
**Parameters:**
- `df`: The pandas DataFrame to validate.
- `columns`: List of column names to check for existence.

**Raises:**
- `ValueError` if columns are missing.

---

### 6. `validate_group_and_aggregation_fields`

```python
def validate_group_and_aggregation_fields(
    df: pd.DataFrame,
    group_by_fields: List[str],
    aggregations: Optional[Dict[str, List[str]]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
) -> None
```
**Parameters:**
- `df`: The DataFrame to validate.
- `group_by_fields`: Fields to group by.
- `aggregations`: Aggregation functions per field.
- `custom_aggregations`: Custom aggregation functions per field.

**Raises:**
- `ValueError` if required fields are missing.

---

### 7. `validate_join_type`

```python
def validate_join_type(
    join_type: str
) -> None
```
**Parameters:**
- `join_type`: The type of join to validate. Expected: "left", "right", "inner", or "outer".

**Raises:**
- `ValueError` if join type is invalid.

---

## Dependency Resolution and Completion Validation Logic

The module does not directly manage dependencies, but its validation functions are essential for ensuring that all required data and parameters are present before a transformation or operation is executed. This is critical for dependency resolution in data pipelines, as missing or invalid data can halt pipeline execution or produce incorrect results.

- **Field and Parameter Validation**: Ensures that all dependencies (columns, parameters) required by a transformation are available and valid.
- **Constraint Enforcement**: Validates that data meets all specified constraints before proceeding, preventing downstream errors.

---

## Usage Examples

### 1. Validating Required Fields

```python
import pandas as pd
from pamola_core.transformations.commons import validation_utils

# Example DataFrame
df = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})

# Validate required fields
ok, missing = validation_utils.validate_fields_exist(df, ['id', 'value'])
if not ok:
    print(f"Missing fields: {missing}")
```

### 2. Validating Field Types

```python
field_types = {'id': 'int64', 'value': 'numeric'}
ok, type_errors = validation_utils.validate_field_types(df, field_types)
if not ok:
    print(f"Type mismatches: {type_errors}")
```

### 3. Validating Parameters

```python
params = {'threshold': 0.5, 'method': 'mean'}
required = ['threshold', 'method']
types = {'threshold': float, 'method': str}
ok, errors = validation_utils.validate_parameters(params, required, types)
if not ok:
    print(f"Parameter errors: {errors}")
```

### 4. Enforcing Constraints

```python
constraints = {
    'value': {'min': 0, 'max': 100, 'not_null': True, 'unique': True}
}
violations = validation_utils.validate_constraints(df, constraints)
if violations:
    print(f"Constraint violations: {violations}")
```

### 5. Validating Group and Aggregation Fields

```python
# Group by 'id', aggregate 'value' with sum
group_by = ['id']
aggregations = {'value': ['sum']}
validation_utils.validate_group_and_aggregation_fields(df, group_by, aggregations)
```

### 6. Validating Join Type

```python
try:
    validation_utils.validate_join_type('cross')  # Invalid join type
except ValueError as e:
    print(f"Join type error: {e}")
```

---

## Integration Notes

- The validation utilities are designed to be used within transformation classes and pipeline tasks (e.g., with `BaseTask`).
- Use these functions before performing operations to ensure all prerequisites are met.
- Integrate with logging to capture validation failures for audit and debugging.

---

## Error Handling and Exception Hierarchy

- **TypeError**: Raised for incorrect argument types (e.g., passing a string instead of a DataFrame).
- **ValueError**: Raised for missing fields, parameters, or invalid values (e.g., unsupported join type).

**Example:**
```python
try:
    validation_utils.validate_dataframe(df, ['missing_col'])
except ValueError as e:
    # Handle missing columns
    print(e)
```

---

## Configuration Requirements

- No explicit configuration object is required for this module.
- For best results, ensure DataFrames and parameter dictionaries are well-formed and validated before use.

---

## Security Considerations and Best Practices

- **DataFrame Validation**: Always validate external or user-supplied DataFrames to prevent schema mismatches and injection of malicious data.
- **Parameter Validation**: Validate all parameters, especially if sourced from user input or configuration files.
- **Path Security**: This module does not handle file paths directly, but when integrating with other modules, avoid using untrusted absolute paths.

### Example: Security Failure and Handling

```python
# Security risk: trusting user-supplied DataFrame columns
user_df = pd.DataFrame({'user_input': [1, 2]})
try:
    # Fails if 'expected_col' is missing
    validation_utils.validate_dataframe(user_df, ['expected_col'])
except ValueError as e:
    # Securely handle the error
    print(f"Security check failed: {e}")
```

**Risks of Disabling Path Security:**
- If path validation is disabled in other modules, malicious users could access or overwrite sensitive files. Always validate and sanitize paths before use.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use DataFrame column names and task IDs for dependencies within the pipeline.
- **External Dependencies**: Use absolute paths only for data not produced by the pipeline. Always validate external data before use.

---

## Best Practices

1. **Validate Early**: Always validate DataFrames and parameters before performing operations.
2. **Use Logging**: Leverage the built-in logging for traceability and debugging.
3. **Handle Exceptions Gracefully**: Catch and handle exceptions to prevent pipeline crashes.
4. **Document Constraints**: Clearly specify and document all constraints for each transformation.
5. **Use Task IDs for Internal Dependencies**: Maintain logical connections within your project.
6. **Use Absolute Paths Judiciously**: Only for truly external data.
7. **Never Trust User Input**: Always validate user-supplied data and parameters.
