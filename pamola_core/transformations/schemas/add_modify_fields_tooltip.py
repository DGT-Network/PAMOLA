"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add or Modify Fields Operation Tooltips
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Provides detailed tooltips for add or modify fields operation configuration fields in PAMOLA.CORE.
- Explains field operation types and lookup table configurations
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of field transformation operations

Changelog:
1.0.0 - 2025-11-11 - Initial creation of add or modify fields operation tooltip file
"""


class AddOrModifyFieldsOperationTooltip:
    field_operations = (
        "What it does: Defines operations to add new fields or modify existing ones.\n\n"
        "Each field requires an operation_type and corresponding parameters.\n\n"
        "Operation types for adding new fields:\n"
        "• add_constant: Add a field with a constant value\n"
        "  - constant_value: The value to set\n"
        "• add_from_lookup: Add a field by mapping values from another column\n"
        "  - base_on_column: Column to use for lookup\n"
        "  - lookup_table_name: Name of the lookup table\n"
        "• add_conditional: Add a field based on a condition\n"
        "  - condition: Python expression to evaluate\n"
        "  - value_if_true: Value when condition is true\n"
        "  - value_if_false: Value when condition is false\n\n"
        "Operation types for modifying existing fields:\n"
        "• modify_constant: Replace field with a constant value\n"
        "  - constant_value: The new value\n"
        "• modify_from_lookup: Replace field by mapping from another column\n"
        "  - base_on_column: Column to use for lookup\n"
        "  - lookup_table_name: Name of the lookup table\n"
        "• modify_conditional: Replace field based on a condition\n"
        "  - condition: Python expression to evaluate\n"
        "  - value_if_true: Value when condition is true\n"
        "  - value_if_false: Value when condition is false\n"
        "• modify_expression: Transform field using a custom expression\n"
        "  - base_on_column: Column to transform\n"
        "  - expression_character: Placeholder in expression (e.g., 'X')\n"
        "  - expression: Python expression (e.g., 'X * 2 + 1')\n\n"
        "Example:\n"
        "{\n"
        "  'tax': {'operation_type': 'add_constant', 'constant_value': 0.15},\n"
        "  'country_name': {'operation_type': 'add_from_lookup', 'base_on_column': 'country_code', 'lookup_table_name': 'countries'},\n"
        "  'price_doubled': {'operation_type': 'modify_expression', 'base_on_column': 'price', 'expression_character': 'X', 'expression': 'X * 2'}\n"
        "}"
    )

    lookup_tables = (
        "What it does: Provides mapping tables for lookup-based operations.\n\n"
        "Used by 'add_from_lookup' and 'modify_from_lookup' operations to map values from one column to another.\n\n"
        "Format: Dictionary mapping table names to either:\n"
        "• A dictionary of key-value pairs\n"
        "• A file path containing JSON mapping data\n\n"
        "Example (inline dictionary):\n"
        "{\n"
        "  'countries': {'US': 'United States', 'CA': 'Canada', 'MX': 'Mexico'},\n"
        "  'status_codes': {1: 'Active', 2: 'Inactive', 3: 'Pending'}\n"
        "}\n\n"
        "Example (file path):\n"
        "{\n"
        "  'countries': Path('lookup_tables/countries.json'),\n"
        "  'categories': Path('lookup_tables/categories.json')\n"
        "}\n\n"
        "Leave blank if no lookup operations are used."
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = "What it does: Forces the operation to recalculate results from scratch, ignoring any existing cached results. Useful when you want to ensure results reflect any subtle data changes"

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "field_operations": cls.field_operations,
            "lookup_tables": cls.lookup_tables,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }