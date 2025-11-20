"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split By ID Values UI Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of split by ID values configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Conditional field visibility using x-depend-on directives
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menus for ID field and partition method selection
- NumberPicker: Integer numeric inputs for partition count
- Input: Text input for value groups and invalid values

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split by ID values UI schema
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class SplitByIDValuesOperationUIConfig(OperationConfig):
    """
    UI configuration schema for SplitByIDValuesOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Split By ID Values Operation UI Configuration",
        "description": "UI schema for Split By ID Values operation configuration form.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "id_field": {
                        "x-component": "Select",
                        "x-group": GroupName.ID_FIELD,
                    },
                    "partition_method": {
                        "x-component": "Select",
                        "x-group": GroupName.PARTITION_SETTINGS,
                    },
                    "number_of_partitions": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.PARTITION_SETTINGS,
                        "x-depend-on": {"partition_method": "not_null"},
                    },
                    "value_groups": {
                        "x-component": "Object",
                        "x-group": GroupName.VALUE_GROUPS,
                    },
                    "invalid_values": {
                        "x-component": "Object",
                    },
                },
            },
        ],
    }
