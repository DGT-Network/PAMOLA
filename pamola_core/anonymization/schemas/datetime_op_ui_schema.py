"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Datetime Generalization UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of datetime generalization configurations in PAMOLA.CORE.
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
- Select: Dropdown menus for enums/oneOf
- NumberPicker: Numeric inputs with validation
- DatePicker: Date selection component
- DatePickerArray: Array of date pickers
- DateFormatArray: Custom datetime format array input
- Input: Text input fields
- FloatPicker: Floating-point numeric inputs

Changelog:
1.0.0 - 2025-11-18 - Initial creation of datetime generalization UI schema
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class DateTimeGeneralizationUIConfig(OperationConfig):
    """
    UI configuration schema for DateTimeGeneralization form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "DateTime Generalization Configuration UI Schema",
        "description": "UI schema for datetime generalization configurations",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "rounding_unit": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "rounding"},
                        "x-required-on": {"strategy": "rounding"},
                    },
                    "bin_type": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "binning"},
                        "x-required-on": {"strategy": "binning"},
                    },
                    "interval_size": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"bin_type": ["hour_range", "day_range"]},
                        "x-required-on": {"bin_type": ["hour_range", "day_range"]},
                    },
                    "interval_unit": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"bin_type": ["hour_range", "day_range"]},
                        "x-required-on": {"bin_type": ["hour_range", "day_range"]},
                    },
                    "reference_date": {
                        "x-component": "DatePicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "relative"},
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                    },
                    "custom_bins": {
                        "x-component": "DatePickerArray",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"bin_type": "custom"},
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "getPopupContainer": "{{(node) => node?.parentElement || document.body}}",
                            "placeholder": "YYYY-MM-DD",
                        },
                    },
                    "keep_components": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "component"},
                    },
                    "strftime_output_format": {
                        "x-component": "Input",
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                    },
                    "timezone_handling": {
                        "x-component": "Select",
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                    },
                    "default_timezone": {
                        "x-component": "Input",
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                        "x-depend-on": {"timezone_handling": "utc"},
                    },
                    "input_formats": {
                        "x-component": "DateFormatArray",
                        "x-component-props": {
                            "formatActions": "{{ supportedFormatActions }}",
                            "placeholder": "Custom datetime pattern",
                        },
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                    },
                    "min_privacy_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "quasi_identifiers": {
                        "x-component": "Select",
                    },
                }
            },
        ],
    }
