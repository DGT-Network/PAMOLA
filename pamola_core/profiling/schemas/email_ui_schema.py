"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Profiler UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of email profiling configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- NumberPicker: Numeric inputs with validation
- Checkbox: Boolean toggles

Changelog:
1.0.0 - 2025-01-15 - Initial creation of email profiler UI schema
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class EmailOperationUIConfig(OperationConfig):
    """
    UI configuration schema for EmailOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Email Operation UI Configuration",
        "description": "UI schema for email profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {"x-component": "Select"},
                    "top_n": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "min_frequency": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "profile_type": {"x-component": "Input"},
                    "analyze_privacy_risk": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                }
            },
        ],
    }
