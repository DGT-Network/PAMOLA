"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL Linkage Operation UI schema
Package:       pamola_core.metrics.schemas
License:       BSD 3-Clause

UI metadata (x-component, x-group) for the CVPL linkage configuration form. No
validation rules here - see ``cvpl_op_core_schema``.
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class CVPLLinkageUIConfig(OperationConfig):
    """UI configuration schema for ``CVPLLinkageOperation`` form rendering."""

    schema = {
        "type": "object",
        "title": "CVPL Linkage Operation UI Configuration",
        "description": "UI schema for CVPL linkage operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "qids": {
                        "x-component": "Select",
                        "x-group": GroupName.COLUMN_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "auxiliary": {
                        "x-component": "Select",
                        "x-group": GroupName.COLUMN_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "blocking_ladder": {"x-component": "Input", "x-group": GroupName.METRIC_SELECTION},
                    "ground_truth": {"x-component": "Input", "x-group": GroupName.COLUMN_CONFIGURATION},
                    "tau": {"x-component": "FloatPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "fit_strategy": {"x-component": "Select", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "variance_retained": {"x-component": "FloatPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "min_components": {"x-component": "NumberPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "max_components": {"x-component": "NumberPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "max_categories": {"x-component": "NumberPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "qid_as_features": {"x-component": "Checkbox", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "null_policy": {"x-component": "Select", "x-group": GroupName.COLUMN_CONFIGURATION},
                    "wildcard_tokens": {"x-component": "Input", "x-group": GroupName.COLUMN_CONFIGURATION},
                    "seed": {"x-component": "NumberPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                    "max_comparisons": {"x-component": "NumberPicker", "x-group": GroupName.STATISTICAL_SETTINGS},
                },
            },
        ],
    }
