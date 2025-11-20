"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Matrix UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of correlation matrix configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menus for field selection and enums
- Input: Text input for method mapping object
- NumberPicker: Numeric inputs with validation

Changelog:
1.0.0 - 2025-11-11 - Initial creation of correlation matrix UI schema
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class CorrelationMatrixOperationUIConfig(OperationConfig):
    """
    UI configuration schema for CorrelationMatrixOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Correlation Matrix Operation UI Configuration",
        "description": "UI schema for correlation matrix profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "fields": {
                        "x-component": "Select",
                    },
                    "methods": {
                        "x-component": "Input",
                        "x-group": GroupName.CORRELATION_CONFIGURATION,
                        "x-items": {
                            "x-component": "Select",
                        },
                    },
                    "min_threshold": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORRELATION_CONFIGURATION,
                    },
                    "null_handling": {
                        "x-component": "Select",
                        "x-group": GroupName.CORRELATION_CONFIGURATION,
                    },
                }
            },
        ],
    }
