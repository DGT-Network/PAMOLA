# CustomFunctions Reference

**Module:** `pamola_core.common.enum.custom_functions`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

CustomFunctions defines all available custom function names for dynamic field updates and interactions in the PAMOLA.CORE form schema system. These functions enable context-aware, reactive form behavior based on user input and data changes.

## Members (11 Functions)

| Function | Value | Purpose |
|----------|-------|---------|
| UPDATE_FIELD_OPTIONS | `"update_field_options"` | Refresh available options for a field |
| UPDATE_CONDITION_OPERATOR | `"update_condition_operator"` | Update conditional logic operators |
| UPDATE_CONDITION_VALUES | `"update_condition_values"` | Update condition parameter values |
| UPDATE_QUASI_FIELD_OPTIONS | `"update_quasi_field_options"` | Refresh quasi-identifier field options |
| UPDATE_EXCLUSIVE_FIELD_OPTIONS | `"update_exclusive_field_options"` | Refresh mutually exclusive field options |
| UPDATE_INT64_FIELD_OPTIONS | `"update_int64_field_options"` | Refresh integer-typed field options |
| UPDATE_DEFAULT_COUNTRY_OPTIONS | `"update_default_country_options"` | Update country selection options |
| UPDATE_FAKE_PHONE_FORMAT | `"update_fake_phone_format"` | Update phone format patterns |
| INIT_FIELD_DOUBLE_SELECT | `"init_field_double_select"` | Initialize dual-list selector |
| INIT_DATA_TYPE_OPTIONS | `"init_data_type_options"` | Initialize data type options |
| INIT_UPLOAD | `"init_upload"` | Initialize file upload control |
| INIT_FIELD_STRATEGY_OPTIONS | `"init_field_strategy_options"` | Initialize field strategy options |

## Usage

### Access Functions

```python
from pamola_core.common.enum.custom_functions import CustomFunctions

# Access function name
func = CustomFunctions.UPDATE_FIELD_OPTIONS
print(func)  # Output: "update_field_options"
```

### Schema Definition with Custom Functions

```python
from pamola_core.common.enum.custom_functions import CustomFunctions

# Define form field with custom function trigger
schema = {
    "type": "object",
    "properties": {
        "operation_type": {
            "type": "string",
            "title": "Operation Type",
            "enum": ["anonymize", "encrypt"]
        },
        "fields": {
            "type": "array",
            "title": "Fields to Process",
            "x-custom-function": CustomFunctions.UPDATE_FIELD_OPTIONS,
            "description": "Available fields change based on operation type"
        }
    }
}
```

## Function Categories

### Field Option Updates

**Dynamic Option Refresh:**
- `UPDATE_FIELD_OPTIONS` - General field option updates
- `UPDATE_QUASI_FIELD_OPTIONS` - Quasi-identifier field options
- `UPDATE_EXCLUSIVE_FIELD_OPTIONS` - Mutually exclusive field options
- `UPDATE_INT64_FIELD_OPTIONS` - Integer field options
- `UPDATE_DEFAULT_COUNTRY_OPTIONS` - Geographic/country options

**Specialized Updates:**
- `UPDATE_FAKE_PHONE_FORMAT` - Phone number format patterns
- `UPDATE_FIELD_STRATEGY_OPTIONS` - Field processing strategy options

### Conditional Logic

- `UPDATE_CONDITION_OPERATOR` - Logical operators (AND, OR, NOT)
- `UPDATE_CONDITION_VALUES` - Condition operand values

### Initialization Functions

**Component Setup:**
- `INIT_FIELD_DOUBLE_SELECT` - Setup dual-list selector
- `INIT_DATA_TYPE_OPTIONS` - Setup data type options
- `INIT_UPLOAD` - Setup file upload component
- `INIT_FIELD_STRATEGY_OPTIONS` - Setup field strategy options

## Detailed Function Descriptions

### UPDATE_FIELD_OPTIONS
**Value:** `"update_field_options"`

Generic function to refresh available options for any field based on context or other field values.

**Triggered by:**
- Parent field value changes
- Data source modifications
- Dependent field updates

**Use cases:**
- Refresh column list after file upload
- Filter options based on selected operation type
- Update available metrics based on field type

### UPDATE_CONDITION_OPERATOR
**Value:** `"update_condition_operator"`

Updates the available conditional logic operators based on field type and context.

**Supported Operators:**
- Comparison: `=`, `!=`, `>`, `<`, `>=`, `<=`
- Logical: `AND`, `OR`, `NOT`
- Pattern: `CONTAINS`, `MATCHES`, `IN`

**Use cases:**
- Enable/disable operators for numeric vs. text fields
- Update operator list based on data type
- Filter operators by field characteristics

### UPDATE_CONDITION_VALUES
**Value:** `"update_condition_values"`

Updates the available values for condition operands based on operator and field selection.

**Example Flow:**
```
User selects: field="country"
User selects: operator="IN"
Function updates: values=["USA", "Canada", "Mexico", ...]
```

**Use cases:**
- Populate value list for dropdown conditions
- Update range bounds for numeric conditions
- Refresh enum values for categorical fields

### UPDATE_QUASI_FIELD_OPTIONS
**Value:** `"update_quasi_field_options"`

Specialized function to update quasi-identifier field options (fields used for privacy evaluation).

**Context:**
- Quasi-identifiers are attributes that combined may re-identify individuals
- Examples: age, zip code, gender

**Use cases:**
- Update available quasi-identifier fields
- Reflect changes in field selection
- Update k-anonymity analysis fields

### UPDATE_EXCLUSIVE_FIELD_OPTIONS
**Value:** `"update_exclusive_field_options"`

Updates fields that are mutually exclusive (cannot be selected together).

**Example:**
```
Cannot select both:
- "exact_value" and "value_range"
- "keep_original" and "anonymize"
```

**Use cases:**
- Ensure incompatible options aren't selected
- Manage conflicting configuration parameters
- Enforce business rules

### UPDATE_INT64_FIELD_OPTIONS
**Value:** `"update_int64_field_options"`

Filters available fields to only 64-bit integer types.

**Use cases:**
- Select integer ID fields for specific operations
- Filter numeric fields by data type
- Constraint operations to integer data

### UPDATE_DEFAULT_COUNTRY_OPTIONS
**Value:** `"update_default_country_options"`

Updates the available country/region options for geographic operations.

**Affected by:**
- Language selection
- Regional configurations
- Supported locales

**Use cases:**
- Localize country lists
- Update region options
- Filter by supported countries

### UPDATE_FAKE_PHONE_FORMAT
**Value:** `"update_fake_phone_format"`

Updates phone number format patterns based on country or region selection.

**Format Examples:**
```
US: +1 (xxx) xxx-xxxx
UK: +44 xxxx xxxxxx
Vietnam: +84 xxx xxxxxx
Russia: +7 (xxx) xxx-xx-xx
```

**Use cases:**
- Generate phone numbers in correct format
- Apply country-specific patterns
- Validate phone format

### INIT_FIELD_DOUBLE_SELECT
**Value:** `"init_field_double_select"`

Initializes a dual-list selector component with source and target lists.

**Initialization Parameters:**
- `sourceTitle`: Label for available items
- `targetTitle`: Label for selected items
- `sourceData`: Initial available items
- `targetData`: Initially selected items

**Use cases:**
- Select fields from available columns
- Arrange field order
- Assign roles to fields

### INIT_DATA_TYPE_OPTIONS
**Value:** `"init_data_type_options"`

Initializes available data type options based on dataset schema.

**Available Types:**
- `string` - Text/character data
- `integer` - Whole numbers
- `float` - Decimal numbers
- `date` - Temporal data
- `boolean` - True/false values
- `categorical` - Enumerated values

**Use cases:**
- Populate data type dropdown
- Detect and display column types
- Validate type compatibility

### INIT_UPLOAD
**Value:** `"init_upload"`

Initializes file upload control with configuration options.

**Configuration:**
- `accept`: File types (CSV, JSON, Excel, Parquet)
- `multiple`: Single or multiple files
- `maxSize`: Maximum file size
- `parseHeaders`: Extract column headers

**Use cases:**
- Configure file upload restrictions
- Set up data source ingestion
- Initialize file parsing

### INIT_FIELD_STRATEGY_OPTIONS
**Value:** `"init_field_strategy_options"`

Initializes available processing strategies for fields.

**Strategy Examples:**
- Anonymization: masking, generalization, encryption
- Handling: keep, remove, impute, synthesize
- Transformation: scale, normalize, derive

**Use cases:**
- Show applicable strategies for field type
- Filter strategies by operation
- Initialize strategy selector

## Common Patterns

### Event-Driven Form Updates

```python
from pamola_core.common.enum.custom_functions import CustomFunctions

def build_reactive_field(name: str, trigger_func: str) -> dict:
    """Build field with custom function trigger."""
    return {
        "name": name,
        "x-custom-function": trigger_func,
        "x-custom-function-props": {
            "trigger": "onChange",
            "debounce": 300
        }
    }

# Usage
field_selector = build_reactive_field(
    "column_selection",
    CustomFunctions.UPDATE_FIELD_OPTIONS
)
```

### Function Chaining

```python
from pamola_core.common.enum.custom_functions import CustomFunctions

# Chain functions for sequential updates
form_flow = {
    "file_upload": {
        "x-custom-function": CustomFunctions.INIT_UPLOAD,
        "triggers": [CustomFunctions.INIT_DATA_TYPE_OPTIONS]
    },
    "data_type": {
        "x-custom-function": CustomFunctions.INIT_DATA_TYPE_OPTIONS,
        "triggers": [CustomFunctions.UPDATE_FIELD_STRATEGY_OPTIONS]
    },
    "strategy": {
        "x-custom-function": CustomFunctions.UPDATE_FIELD_STRATEGY_OPTIONS,
        "triggers": []
    }
}
```

### Conditional Logic Building

```python
from pamola_core.common.enum.custom_functions import CustomFunctions

# Build conditional chain
condition = {
    "field": "operation_type",
    "operator": None,  # Updated by function
    "value": None,     # Updated by function
    "x-custom-functions": [
        CustomFunctions.UPDATE_CONDITION_OPERATOR,
        CustomFunctions.UPDATE_CONDITION_VALUES
    ]
}
```

## Best Practices

1. **Use Function Names Consistently**
   ```python
   # Good - function names as constants
   INIT_FUNC = CustomFunctions.INIT_UPLOAD

   schema["file_field"]["x-custom-function"] = INIT_FUNC
   ```

2. **Document Function Dependencies**
   ```python
   # Good - explains function dependencies
   "x-custom-function": CustomFunctions.UPDATE_FIELD_OPTIONS,
   "x-custom-function-depends-on": ["operation_type", "data_source"],
   "x-custom-function-description": "Refresh available fields when operation or source changes"
   ```

3. **Handle Function Errors**
   ```python
   def safe_function_call(func_name: str, context: dict):
       """Execute custom function with error handling."""
       try:
           return execute_custom_function(func_name, context)
       except Exception as e:
           logger.error(f"Function {func_name} failed: {e}")
           return None
   ```

4. **Test Function Order**
   ```python
   # Execute functions in correct dependency order
   execution_order = [
       CustomFunctions.INIT_UPLOAD,
       CustomFunctions.INIT_DATA_TYPE_OPTIONS,
       CustomFunctions.UPDATE_FIELD_STRATEGY_OPTIONS
   ]
   ```

## Related Components

- **CustomComponents:** UI components that trigger these functions
- **FormGroups:** Field organization that uses these functions
- **Schema Validation:** Validates function configurations

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Custom Components](./custom_components.md)
- [Form Groups](./form_groups.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- All 11 custom functions documented
- Detailed descriptions and use cases
- Common patterns and best practices
