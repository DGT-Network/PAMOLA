"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        K-Anonymity Profiler UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of k-anonymity profiling configurations in PAMOLA.CORE.
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
- Checkbox: Boolean toggles
- Select: Dropdown menus for enums/arrays
- Input: Text input fields
- NumberPicker: Numeric inputs with validation

Changelog:
1.0.0 - 2025-01-15 - Initial creation of k-anonymity profiler UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class KAnonymityProfilerOperationUIConfig(OperationConfig):
    """
    UI configuration schema for KAnonymityProfilerOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "K-Anonymity Profiler Operation UI Configuration",
        "description": "UI schema for k-anonymity profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "analysis_mode": {
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "quasi_identifiers": {
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_EXCLUSIVE_FIELD_OPTIONS
                        ],
                        "x-depend-on": {"id_fields": "not_null"},
                    },
                    "quasi_identifier_sets": {
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_EXCLUSIVE_FIELD_OPTIONS
                        ],
                        "x-depend-on": {"id_fields": "not_null"},
                    },
                    "threshold_k": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "max_combinations": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "id_fields": {
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_EXCLUSIVE_FIELD_OPTIONS
                        ],
                        "x-depend-on": {
                            "quasi_identifiers": "not_null",
                            "quasi_identifier_sets": "not_null",
                        },
                    },
                    "output_field_suffix": {
                        "x-component": "Input",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "export_metrics": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                }
            },
        ],
    }
