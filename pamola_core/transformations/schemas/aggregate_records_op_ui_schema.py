"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Aggregate Records UI Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of aggregate records configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Multi-select dropdown for group by fields and aggregation functions
- Object: Object input for aggregation mappings
- Input: Text input for custom aggregation expressions

Changelog:
1.0.0 - 2025-01-15 - Initial creation of aggregate records UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class AggregateRecordsOperationUIConfig(OperationConfig):
    """
    UI configuration schema for AggregateRecordsOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Aggregate Records Operation UI Configuration",
        "description": "UI schema for aggregate records operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "group_by_fields": {
                        "x-component": "Select",
                        "x-group": GroupName.GROUPING_SETTINGS,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "aggregations": {
                        "x-component": CustomComponents.FIELD_GROUP_ARRAY,
                        "x-group": GroupName.AGGREGATION_SETUP,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "custom_aggregations": {
                        "x-component": CustomComponents.FIELD_GROUP_ARRAY,
                        "x-group": GroupName.CUSTOM_AGGREGATIONS,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                },
            },
        ],
    }
