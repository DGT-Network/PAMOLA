# Form Groups Reference

**Module:** `pamola_core.common.enum.form_groups`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

Form Groups provides metadata for organizing operation configuration fields into logical UI groups. The GroupName enumeration and GROUP_TITLES dictionary enable hierarchical form organization, while OPERATION_CONFIG_GROUPS maps operations to their relevant field groups.

## Structure

### 1. GroupName Enumeration

Defines all available form field groups (60+ groups) for organizing configuration UI.

### 2. GROUP_TITLES Dictionary

Maps GroupName to human-readable display titles for UI labels.

### 3. OPERATION_CONFIG_GROUPS Dictionary

Maps operation configuration classes to their field groups in proper display order.

## Core Group Categories

### Generalization Strategy Groups

| Group | Title | Use |
|-------|-------|-----|
| `CORE_GENERALIZATION_STRATEGY` | Core Generalization Strategy | Primary anonymization method selection |
| `HIERARCHY_SETTINGS` | Hierarchy Settings | Hierarchical taxonomy definition |
| `FREQUENCY_GROUPING_SETTINGS` | Frequency & Grouping Settings | Frequency-based clustering options |
| `DATETIME_GENERALIZATION` | DateTime Generalization | Temporal anonymization methods |
| `NUMERIC_METHOD` | Numeric Generalization | Numeric value anonymization |

### Logic & Conditions

| Group | Title | Use |
|-------|-------|-----|
| `CONDITIONAL_LOGIC` | Conditional Logic | IF-THEN rule definition |
| `SIMPLE_CONDITIONAL_RULE` | Simple Conditional Rule | Basic condition setup |
| `ADVANCED_CONDITIONAL_RULES` | Advanced Conditional Rules | Complex rule chains |

### Masking & Substitution

| Group | Title | Use |
|-------|-------|-----|
| `CORE_MASKING_STRATEGY` | Core Masking Strategy | Masking approach selection |
| `CORE_MASKING_RULES` | Core Masking Rules | Masking configuration |
| `MASK_APPEARANCE` | Mask Appearance | Display format of masked data |
| `MASKING_RULES` | Masking Rules | Field-specific masking config |

### Data Generation

| Group | Title | Use |
|-------|-------|-----|
| `NAME_GENERATION_STYLE` | Name Generation Style | Synthetic name generation |
| `EMAIL_GENERATION_STYLE` | Email Generation Style | Synthetic email generation |
| `ORGANIZATION_GENERATION_STYLE` | Organization Generation Style | Synthetic org name generation |
| `GENDER_CONFIGURATION` | Gender Configuration | Gender attribute configuration |
| `REGIONAL_CONFIGURATION` | Regional Configuration | Region-specific generation |
| `GENERATION_LOGIC` | Generation Logic | Generation algorithm selection |

### Data Quality & Validation

| Group | Title | Use |
|-------|-------|-----|
| `VALIDATION_RANGE` | Validation Range | Min/max value constraints |
| `DATA_QUALITY_ANALYSIS` | Data Quality Analysis | Quality metrics and checks |
| `INVALID_VALUES_CONFIGURATION` | Invalid Values Configuration | Handling bad/invalid data |
| `NULL_REPLACEMENT_CONFIGURATION` | Null Replacement Configuration | Missing value handling |
| `WHITELIST_CONFIGURATION` | Whitelist Configuration | Allowed values list |
| `BLACKLIST_CONFIGURATION` | Blacklist Configuration | Forbidden values list |

### Aggregation & Grouping

| Group | Title | Use |
|-------|-------|-----|
| `AGGREGATION_SETUP` | Aggregation Setup | Aggregation function config |
| `CUSTOM_AGGREGATIONS` | Custom Aggregations | User-defined aggregations |
| `GROUP_CONFIGURATION` | Group Configuration | Grouping key selection |
| `GROUPING_SETTINGS` | Grouping Settings | Grouping algorithm options |
| `VALUE_GROUPS` | Value Groups | Value categorization |

### Data Operations

| Group | Title | Use |
|-------|-------|-----|
| `FIELD_REMOVAL` | Field Removal | Drop columns |
| `FIELD_OPERATIONS_CONFIGURATION` | Field Operations Configuration | Field transformations |
| `FIELD_STRATEGIES_CONFIGURATION` | Field Strategies Configuration | Per-field strategy selection |
| `FIELD_CONSTRAINTS_CONFIGURATION` | Field Constraints Configuration | Field-level constraints |
| `FIELD_WEIGHTS_CONFIGURATION` | Field Weights Configuration | Feature importance/weighting |

### Multi-Table Operations

| Group | Title | Use |
|-------|-------|-----|
| `INPUT_DATASETS` | Input Datasets | Source dataset selection |
| `JOIN_KEYS` | Join Keys | Join columns for merge |
| `SUFFIXES` | Suffixes | Column rename on merge |

### Privacy & Risk

| Group | Title | Use |
|-------|-------|-----|
| `RISK_BASED_PROCESSING_AND_PRIVACY` | Risk-Based Processing & Privacy | Privacy level thresholds |
| `RISK_BASED_FILTERING` | Risk-Based Filtering | Risk-based row filtering |
| `IDENTIFIER_CONFIGURATION` | Identifier Configuration | Quasi-identifier selection |

### Analysis & Metrics

| Group | Title | Use |
|-------|-------|-----|
| `ANALYSIS_PARAMETERS` | Analysis Parameters | Analysis method selection |
| `ANALYSIS_CONFIGURATION` | Analysis Configuration | Analysis settings |
| `DISTRIBUTION_AND_ANALYSIS_SETTINGS` | Distribution & Analysis Settings | Distribution analysis options |
| `DATA_SOURCES_FOR_GENERATION` | Data Sources for Generation | External data sources |
| `CONTEXT_AND_DATA_SOURCES` | Context & Data Sources | Contextual data references |

### Output & Formatting

| Group | Title | Use |
|-------|-------|-----|
| `FORMATTING_AND_STRUCTURE` | Formatting & Structure | Output format selection |
| `FORMATTING_AND_TIMEZONE` | Formatting & Timezone | Date/time formatting |
| `FORMATTING_RULES` | Formatting Rules | Custom formatting patterns |
| `OUTPUT_FORMATTING_CONSTRAINTS` | Output Formatting Constraints | Output value constraints |
| `OPERATION_BEHAVIOR_OUTPUT` | Operation Behavior & Output | General operation output config |

### Configuration

| Group | Title | Use |
|-------|-------|-----|
| `DICTIONARY_CONFIGURATION` | Dictionary Configuration | Lookup table definition |
| `LOOKUP_TABLE_CONFIGURATION` | Lookup Table Configuration | Reference table config |
| `CONSISTENCY_STRATEGY` | Consistency Strategy | Cross-field consistency rules |
| `TEXT_COMPARISON_SETTINGS` | Text Comparison Settings | String comparison options |
| `CURRENCY_PARSING_SETTINGS` | Currency Parsing Settings | Currency value parsing |
| `PARTITION_SETTINGS` | Partition Settings | Data partitioning |
| `INPUT_SETTINGS` | Input Settings | Input data configuration |
| `FIELD_GROUPS_CONFIGURATION` | Field Groups Configuration | Field grouping |
| `ID_FIELD` | ID Field | Identifier field selection |
| `FIELD_SETTINGS` | Field Settings | General field config |

## Usage

### Access Group Names

```python
from pamola_core.common.enum.form_groups import GroupName, GROUP_TITLES

# Access group
group = GroupName.CORE_GENERALIZATION_STRATEGY
print(group.value)  # Output: "core_generalization_strategy"

# Get display title
title = GROUP_TITLES[group]
print(title)  # Output: "Core Generalization Strategy"
```

### Get Groups for an Operation

```python
from pamola_core.common.enum.form_groups import OPERATION_CONFIG_GROUPS, GroupName, GROUP_TITLES

# Get groups for a specific operation
operation = "NumericGeneralizationConfig"
groups = OPERATION_CONFIG_GROUPS[operation]
# [GroupName.CORE_GENERALIZATION_STRATEGY, GroupName.CONDITIONAL_LOGIC, GroupName.OPERATION_BEHAVIOR_OUTPUT]

# Get titles for rendering
titles = [GROUP_TITLES[g] for g in groups]
# ["Core Generalization Strategy", "Conditional Logic", "Operation Behavior & Output"]
```

### Build Hierarchical Form

```python
from pamola_core.common.enum.form_groups import OPERATION_CONFIG_GROUPS, GROUP_TITLES

def build_form_sections(operation_class: str) -> list:
    """Build form sections for operation."""
    groups = OPERATION_CONFIG_GROUPS.get(operation_class, [])
    sections = []

    for group in groups:
        sections.append({
            "id": group.value,
            "title": GROUP_TITLES[group],
            "fields": get_fields_for_group(group)
        })

    return sections

# Usage
form = build_form_sections("NumericGeneralizationConfig")
```

## Common Patterns

### Render Grouped Form

```python
from pamola_core.common.enum.form_groups import OPERATION_CONFIG_GROUPS, GROUP_TITLES

def render_operation_form(operation: str, schema: dict) -> html:
    """Render form with grouped sections."""
    groups = OPERATION_CONFIG_GROUPS.get(operation, [])
    html = ""

    for group in groups:
        title = GROUP_TITLES[group]
        fields_in_group = [f for f in schema["properties"]
                           if f.get("group") == group.value]

        html += f"<fieldset><legend>{title}</legend>"
        for field in fields_in_group:
            html += render_field(field)
        html += "</fieldset>"

    return html
```

### Query Groups by Prefix

```python
from pamola_core.common.enum.form_groups import GroupName

# Find all data-related groups
data_groups = [g for g in GroupName if "data" in g.value.lower()]

# Find all configuration groups
config_groups = [g for g in GroupName if "configuration" in g.value.lower()]
```

### Operation Type to Groups Mapping

```python
from pamola_core.common.enum.form_groups import OPERATION_CONFIG_GROUPS

# Group operations by number of groups
from collections import defaultdict

operation_complexity = defaultdict(list)
for operation, groups in OPERATION_CONFIG_GROUPS.items():
    complexity_level = len(groups)
    operation_complexity[complexity_level].append(operation)

# Find operations with 5+ groups (complex)
complex_ops = operation_complexity[max(operation_complexity.keys())]
```

## Best Practices

1. **Use GroupName Enum for Type Safety**
   ```python
   # Good
   group = GroupName.CORE_GENERALIZATION_STRATEGY

   # Avoid
   group = "core_generalization_strategy"  # String less safe
   ```

2. **Always Look Up Titles from GROUP_TITLES**
   ```python
   # Good
   title = GROUP_TITLES[group]

   # Avoid
   title = group.value.replace("_", " ").title()  # May not match official title
   ```

3. **Preserve Group Order**
   ```python
   # Good - maintain order from OPERATION_CONFIG_GROUPS
   groups = OPERATION_CONFIG_GROUPS["OperationClass"]
   for group in groups:  # Iterate in order
       render_section(group)

   # Avoid - reordering groups
   groups = set(OPERATION_CONFIG_GROUPS["OperationClass"])
   ```

4. **Validate Group Assignments**
   ```python
   from pamola_core.common.enum.form_groups import GroupName

   def validate_field_group(field_group: str) -> bool:
       """Ensure field is assigned valid group."""
       return field_group in [g.value for g in GroupName]
   ```

## Related Operations

Operations that use form groups:
- Numeric Generalization
- Categorical Generalization
- DateTime Generalization
- Masking (Full & Partial)
- Noise Addition
- Suppression
- Fake Data Generation
- Data Aggregation
- And 20+ more operation types

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Custom Components](./custom_components.md)
- [Custom Functions](./custom_functions.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- 60+ form groups documented
- GROUP_TITLES mapping
- OPERATION_CONFIG_GROUPS mapping
- Usage patterns and best practices
