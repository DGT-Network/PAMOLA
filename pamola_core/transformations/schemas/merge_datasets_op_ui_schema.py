"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Merge Datasets UI Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of merge datasets configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-custom-function)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Custom functions for dynamic field options
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menus for key selection, join type, and relationship type
- ArrayItems: Array input for suffix configuration
- Input: Text input for array items

Changelog:
1.0.0 - 2025-01-15 - Initial creation of merge datasets UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class MergeDatasetsOperationUIConfig(OperationConfig):
    """
    UI configuration schema for MergeDatasetsOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Merge Datasets Operation UI Configuration",
        "description": "UI schema for Merge Datasets operation configuration form.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "left_dataset_name": {
                        "x-component": "Input",
                    },
                    "right_dataset_name": {
                        "x-component": "Input",
                    },
                    "right_dataset_path": {
                        "x-component": "Input",
                    },
                    "left_key": {
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.JOIN_KEYS,
                    },
                    "right_key": {
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.JOIN_KEYS,
                    },
                    "join_type": {
                        "x-component": "Select",
                        "x-group": GroupName.INPUT_DATASETS,
                    },
                    "relationship_type": {
                        "x-component": "Select",
                        "x-group": GroupName.INPUT_DATASETS,
                    },
                    "suffixes": {
                        "x-component": "ArrayItems",
                        "x-group": GroupName.SUFFIXES,
                        "x-items": {
                            "x-component": "Input",
                            "x-items-title": [
                                "Left Column Suffix",
                                "Right Column Suffix",
                            ],
                            "x-item-params": ["left", "right"],
                        },
                    },
                },
            },
        ],
    }
