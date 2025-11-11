"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add or Modify Fields Config Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating add/modify fields operations in PAMOLA.CORE.
- Supports adding new fields or modifying existing ones using constants, lookups, conditions, or expressions
- Handles lookup table management for mapping-based transformations
- Provides multiple operation types for flexible field manipulation
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of add/modify fields config file
1.1.0 - 2025-11-11 - Updated with x-component, x-group, and enhanced descriptions
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class AddOrModifyFieldsOperationConfig(OperationConfig):
    """
    Configuration schema for AddOrModifyFieldsOperation.

    Extends BaseOperationConfig with field transformation parameters for adding new fields
    or modifying existing ones using various strategies (constants, lookups, conditions, expressions).
    """

    schema = {
        "type": "object",
        "title": "Add or Modify Fields Operation Configuration",
        "description": (
            "Configuration options for adding new fields or modifying existing fields. "
            "Supports constant values, lookup tables, conditional logic, and custom expressions."
        ),
        "allOf": [
            BaseOperationConfig.schema,  # merge base schema
            {
                "type": "object",
                "properties": {
                    # === Field operations ===
                    "field_operations": {
                        "type": ["object", "null"],
                        "title": "Field Operations",
                        "x-component": "Input",
                        "description": (
                            "Dictionary mapping field names to operation configurations.\n\n"
                            "Each field requires an operation_type and corresponding parameters.\n\n"
                            "Operation types for adding new fields:\n"
                            "- add_constant: Add field with constant value (requires: constant_value)\n"
                            "- add_from_lookup: Add field by mapping from another column (requires: base_on_column, lookup_table_name)\n"
                            "- add_conditional: Add field based on condition (requires: condition, value_if_true, value_if_false)\n\n"
                            "Operation types for modifying existing fields:\n"
                            "- modify_constant: Replace with constant value (requires: constant_value)\n"
                            "- modify_from_lookup: Replace by mapping from another column (requires: base_on_column, lookup_table_name)\n"
                            "- modify_conditional: Replace based on condition (requires: condition, value_if_true, value_if_false)\n"
                            "- modify_expression: Transform using custom expression (requires: base_on_column, expression_character, expression)\n\n"
                            "Example: {'tax': {'operation_type': 'add_constant', 'constant_value': 0.15}}"
                        ),
                        "x-group": GroupName.FIELD_OPERATIONS_CONFIGURATION,
                        "default": None,
                    },
                    # === Lookup tables ===
                    "lookup_tables": {
                        "type": ["object", "null"],
                        "title": "Lookup Tables",
                        "x-component": "Input",
                        "description": (
                            "Dictionary mapping table names to lookup data for field transformations.\n\n"
                            "Used by 'add_from_lookup' and 'modify_from_lookup' operations.\n\n"
                            "Format options:\n"
                            "- Inline dictionary: Direct key-value mapping\n"
                            "- File path: Path to JSON file containing mapping data\n\n"
                            "Example (inline): {'countries': {'US': 'United States', 'CA': 'Canada'}}\n"
                            "Example (file): {'countries': Path('lookup_tables/countries.json')}"
                        ),
                        "x-group": GroupName.LOOKUP_TABLE_CONFIGURATION,
                        "default": None,
                    },
                },
            },
        ],
    }
