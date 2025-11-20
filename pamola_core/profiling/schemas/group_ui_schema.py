"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Group Analyzer UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of group analyzer configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on, x-required-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Conditional field visibility using x-depend-on directives
- No business logic validation - purely presentational metadata

UI Component Types:
- FloatPicker: Float numeric inputs with validation
- NumberPicker: Integer numeric inputs with validation
- Select: Dropdown menus for enums
- Object: Object input for field configuration mapping

Changelog:
1.0.0 - 2025-01-15 - Initial creation of group analyzer UI schema
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class GroupAnalyzerOperationUIConfig(OperationConfig):
    """
    UI configuration schema for GroupAnalyzerOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Group Analyzer UI Configuration",
        "description": "UI schema for group profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "variance_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.GROUP_CONFIGURATION,
                    },
                    "large_group_threshold": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.GROUP_CONFIGURATION,
                    },
                    "large_group_variance_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.GROUP_CONFIGURATION,
                    },
                    "text_length_threshold": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.TEXT_COMPARISON_SETTINGS,
                    },
                    "hash_algorithm": {
                        "x-component": "Select",
                        "x-group": GroupName.TEXT_COMPARISON_SETTINGS,
                    },
                    "minhash_similarity_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.TEXT_COMPARISON_SETTINGS,
                        "x-depend-on": {"hash_algorithm": "minhash"},
                        "x-required-on": {"hash_algorithm": "minhash"},
                    },
                    "fields_config": {
                        "x-component": "Object",
                        "x-group": GroupName.FIELD_WEIGHTS_CONFIGURATION,
                        "x-items": {
                            "x-component": "NumberPicker",
                        },
                    },
                }
            },
        ],
    }
