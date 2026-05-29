"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Fidelity Metric UI Schema
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of fidelity metric operation
configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on)
- Defines component types and grouping for frontend forms
- Parallel structure to core schema but without validation rules
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class FidelityUIConfig(OperationConfig):
    """
    UI configuration schema for FidelityOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Fidelity Metric Operation UI Configuration",
        "description": "UI schema for fidelity metric operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    # --- Metric selection ---
                    "fidelity_metrics": {
                        "x-component": "Select",
                        "x-group": GroupName.METRIC_SELECTION,
                    },
                    "metric_params": {
                        "x-component": "Input",
                        "x-group": GroupName.METRIC_SELECTION,
                    },
                    # --- Column configuration ---
                    "columns": {
                        "x-component": "Select",
                        "x-group": GroupName.COLUMN_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "column_mapping": {
                        "x-component": "Input",
                        "x-group": GroupName.COLUMN_CONFIGURATION,
                    },
                    # --- Statistical settings ---
                    "normalize": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.STATISTICAL_SETTINGS,
                    },
                    "confidence_level": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.STATISTICAL_SETTINGS,
                    },
                    "sample_size": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.STATISTICAL_SETTINGS,
                    },
                },
            },
        ],
    }
