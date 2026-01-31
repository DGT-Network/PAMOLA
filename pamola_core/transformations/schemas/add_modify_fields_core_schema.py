"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add or Modify Fields Core Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of add/modify fields configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines field transformation parameters for adding or modifying fields
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field operations dictionary for transformation configuration
- Lookup table management for mapping-based operations
- Support for multiple operation types (constant, lookup, conditional, expression)
- Flexible field manipulation strategies

Changelog:
1.0.0 - 2025-01-15 - Initial creation of add/modify fields core schema
1.1.0 - 2025-11-11 - Updated with enhanced operation type descriptions
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class AddOrModifyFieldsOperationConfig(OperationConfig):
    """
    Core configuration schema for AddOrModifyFieldsOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Add or Modify Fields Operation Core Configuration",
        "description": "Core schema for Add or Modify Fields operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_operations": {
                        "type": ["object", "null"],
                        "title": "Field Operations",
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
                        "default": None,
                    },
                    "lookup_tables": {
                        "type": ["object", "null"],
                        "title": "Lookup Tables",
                        "description": (
                            "Dictionary mapping table names to lookup data for field transformations.\n\n"
                            "Used by 'add_from_lookup' and 'modify_from_lookup' operations.\n\n"
                            "Format options:\n"
                            "- Inline dictionary: Direct key-value mapping\n"
                            "- File path: Path to JSON file containing mapping data\n\n"
                            "Example (inline): {'countries': {'US': 'United States', 'CA': 'Canada'}}\n"
                            "Example (file): {'countries': Path('lookup_tables/countries.json')}"
                        ),
                        "default": None,
                    },
                },
            },
        ],
    }
