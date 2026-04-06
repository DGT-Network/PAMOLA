# CustomComponents Reference

**Module:** `pamola_core.common.enum.custom_components`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

CustomComponents defines all available custom UI component names used in form schemas. These components enable dynamic, context-aware field rendering and updates in the PAMOLA.CORE UI framework.

## Members (23 Components)

| Component | Class Name | Purpose |
|-----------|-----------|---------|
| NUMERIC_RANGE_MODE | `NumericRangeMode` | Mode selection for numeric range inputs |
| DATE_FORMAT_ARRAY | `DateFormatArray` | Array input for date format patterns |
| DATE_PICKER_ARRAY | `DatePickerArray` | Array of date picker controls |
| NUMBER_ARRAY | `NumberArray` | Array of numeric input fields |
| RANGE_NUMBER | `RangeNumber` | Range slider for numeric values |
| STRING_ARRAY | `StringArray` | Array of text input fields |
| UPLOAD | `Upload` | File upload control |
| DEPEND_SELECT | `DependSelect` | Dependent/cascading select dropdown |
| DATE_FORMAT | `DateFormat` | Date format selector |
| FORMAT_PATTERNS | `FormatPatterns` | Format pattern input |
| VALUE_GROUP_ARRAY | `ValueGroupArray` | Array of value groups |
| VALUE_GROUP_ARRAY_AGGREGATIONS | `ValueGroupArrayAggregations` | Aggregation functions for value groups |
| CUSTOM_VALUE_GROUP_ARRAY_AGGREGATIONS | `CustomValueGroupArrayAggregations` | Custom aggregation config |
| DATA_SET_CONFIG | `DataSetConfig` | Dataset configuration input |
| FIELD_SELECT_UPLOAD_FILE_INPUT | `FieldSelectUploadFileInput` | Select fields from uploaded file |
| FIELD_MULTIPLE_SELECT_UPLOAD | `FieldMultipleSelectUpload` | Multiple field selection with upload |
| FIELD_DOUBLE_SELECT_INPUT | `FieldDoubleSelectInput` | Dual-list select component |
| FIELD_SELECT_UPLOAD_FILE_INPUT_FAKE_ORG | `FieldSelectUploadFileInputFakeOrg` | Field selection with fake organization |
| FIELD_MULTIPLE_SELECT_UPLOAD_FAKE_NAME | `FieldMultipleSelectUploadFakeName` | Field selection with fake name data |
| FIELD_DOUBLE_SELECT_INPUT_ADD_OR_MODIFY | `FieldDoubleSelectInputAddOrModify` | Dual select with add/modify capability |
| FIELD_SELECT_UPLOAD_FILE_INPUT_ADD_OR_MODIFY | `FieldSelectUploadFileInputAddOrModify` | File field select with add/modify |
| FORMAT_RATIO_SLIDER | `FormatRatioSlider` | Ratio/percentage slider input |
| FIELD_IMPUTE_STRATEGY | `FieldImputeStrategy` | Strategy selector for missing value imputation |
| FIELD_DOUBLE_SELECT_INPUT_CLEAN_INVALID | `FieldDoubleSelectInputCleanInvalid` | Dual select for data cleaning |
| FIELD_NULL_REPLACEMENT_INPUT | `FieldNullReplacementInput` | Null value replacement input |
| FIELD_SELECT_UPLOAD_FILE_INPUT_CLEAN_INVALID | `FieldSelectUploadFileInputCleanInvalid` | File field select for cleaning |
| SEPARATOR_OPTIONS | `SeparatorOptions` | CSV/text separator options |

## Usage

### Access Components

```python
from pamola_core.common.enum.custom_components import CustomComponents

# Access component name
component = CustomComponents.NUMERIC_RANGE_MODE
print(component)  # Output: "NumericRangeMode"
```

### Schema Definition with Components

```python
from pamola_core.common.enum.custom_components import CustomComponents

# Define UI schema with custom components
schema = {
    "type": "object",
    "properties": {
        "numeric_range": {
            "type": "string",
            "title": "Numeric Range",
            "x-component": CustomComponents.NUMERIC_RANGE_MODE
        },
        "date_formats": {
            "type": "array",
            "title": "Date Formats",
            "x-component": CustomComponents.DATE_FORMAT_ARRAY
        },
        "uploaded_file": {
            "type": "string",
            "title": "Upload File",
            "x-component": CustomComponents.UPLOAD
        },
        "field_selection": {
            "type": "object",
            "title": "Select Fields",
            "x-component": CustomComponents.FIELD_DOUBLE_SELECT_INPUT
        }
    }
}
```

## Component Categories

### Input Controls

**Basic Inputs:**
- `STRING_ARRAY` - Text field arrays
- `NUMBER_ARRAY` - Numeric field arrays
- `DATE_PICKER_ARRAY` - Date selection arrays

**Special Inputs:**
- `UPLOAD` - File upload functionality
- `RANGE_NUMBER` - Range slider (min-max)
- `FORMAT_RATIO_SLIDER` - Percentage/ratio slider

### Selectors

**Simple Selection:**
- `DEPEND_SELECT` - Dependent dropdown (cascading)
- `DATE_FORMAT` - Select date format
- `FIELD_IMPUTE_STRATEGY` - Choose imputation strategy
- `SEPARATOR_OPTIONS` - Pick delimiter

**Dual/Multi Selection:**
- `FIELD_DOUBLE_SELECT_INPUT` - Two-list selector
- `FIELD_DOUBLE_SELECT_INPUT_ADD_OR_MODIFY` - Dual select with actions
- `FIELD_MULTIPLE_SELECT_UPLOAD` - Multiple fields with upload

### Data Handling

**Configuration:**
- `DATA_SET_CONFIG` - Dataset configuration builder
- `VALUE_GROUP_ARRAY` - Define value groupings
- `VALUE_GROUP_ARRAY_AGGREGATIONS` - Aggregation settings
- `CUSTOM_VALUE_GROUP_ARRAY_AGGREGATIONS` - Custom aggregation config

**Data Operations:**
- `FORMAT_PATTERNS` - Pattern definition input
- `FIELD_NULL_REPLACEMENT_INPUT` - Null handling
- `FIELD_DOUBLE_SELECT_INPUT_CLEAN_INVALID` - Data quality control

### File & Field Handling

**File-Based Field Selection:**
- `FIELD_SELECT_UPLOAD_FILE_INPUT` - Basic file field select
- `FIELD_SELECT_UPLOAD_FILE_INPUT_ADD_OR_MODIFY` - With add/modify
- `FIELD_SELECT_UPLOAD_FILE_INPUT_CLEAN_INVALID` - For data cleaning
- `FIELD_SELECT_UPLOAD_FILE_INPUT_FAKE_ORG` - With fake organization data

**Field Arrays:**
- `FIELD_MULTIPLE_SELECT_UPLOAD` - Multiple fields
- `FIELD_MULTIPLE_SELECT_UPLOAD_FAKE_NAME` - With fake name data

### Mode Selectors

- `NUMERIC_RANGE_MODE` - Numeric range strategy
- `DATE_FORMAT_ARRAY` - Array of date formats
- `FORMAT_PATTERNS` - Pattern format selection

## Common Patterns

### Form Field Configuration

```python
from pamola_core.common.enum.custom_components import CustomComponents

def build_field_config(field_name: str, component: str) -> dict:
    """Build Formik field configuration."""
    return {
        "name": field_name,
        "label": field_name.replace("_", " ").title(),
        "x-component": component,
        "required": True
    }

# Usage
age_field = build_field_config("age", CustomComponents.NUMERIC_RANGE_MODE)
date_field = build_field_config("date", CustomComponents.DATE_PICKER_ARRAY)
```

### Dynamic Component Selection

```python
from pamola_core.common.enum.custom_components import CustomComponents

def select_component_for_field_type(field_type: str):
    """Choose component based on field type."""
    component_map = {
        "date": CustomComponents.DATE_PICKER_ARRAY,
        "numeric": CustomComponents.NUMERIC_RANGE_MODE,
        "file": CustomComponents.UPLOAD,
        "selection": CustomComponents.FIELD_DOUBLE_SELECT_INPUT,
        "text": CustomComponents.STRING_ARRAY,
    }
    return component_map.get(field_type, CustomComponents.STRING_ARRAY)
```

## Best Practices

1. **Use Component Names Consistently**
   ```python
   # Good - component names as constants
   UPLOAD_COMPONENT = CustomComponents.UPLOAD

   schema["file_field"]["x-component"] = UPLOAD_COMPONENT
   ```

2. **Validate Component Usage**
   ```python
   def validate_schema(schema: dict) -> bool:
       """Ensure components are valid."""
       for prop in schema["properties"].values():
           component = prop.get("x-component")
           if component and component not in [c for c in CustomComponents]:
               raise ValueError(f"Invalid component: {component}")
       return True
   ```

3. **Document Component Purpose**
   ```python
   # Good - explains component choice
   "x-component": CustomComponents.FIELD_DOUBLE_SELECT_INPUT,
   "x-component-props": {
       "title": "Available Fields",
       "leftTitle": "Source",
       "rightTitle": "Selected"
   }
   ```

4. **Group Related Components**
   ```python
   # Organization-related
   org_components = [
       CustomComponents.FIELD_SELECT_UPLOAD_FILE_INPUT_FAKE_ORG,
       CustomComponents.DATA_SET_CONFIG
   ]

   # Name-related
   name_components = [
       CustomComponents.FIELD_MULTIPLE_SELECT_UPLOAD_FAKE_NAME
   ]
   ```

## Related Components

- **CustomFunctions:** Event handlers for component interactions
- **FormGroups:** Organizational structure for field grouping
- **Form Generation:** Dynamic schema building system

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Custom Functions](./custom_functions.md)
- [Form Groups](./form_groups.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- All 23 custom component types documented
- Usage patterns and best practices
- Component categories for easy reference
