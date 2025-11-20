"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Base Operation UI Schema
Package:       pamola_core.utils.ops.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of base operation configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings (Checkbox, Select, Input, NumberPicker) for each field
- Logical grouping via x-group for organized form layout
- Conditional field visibility using x-depend-on directives
- No business logic validation - purely presentational metadata
- Ensures consistent UI/UX across all operation configuration forms

UI Component Types:
- Checkbox: Boolean toggles for feature flags
- Select: Dropdown menus for enum/oneOf choices
- Input: Text fields for string parameters
- NumberPicker: Numeric input with validation controls

Changelog:
1.0.0 - 2025-01-15 - Initial creation of base operation UI schema
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig


class BaseOperationUIConfig(OperationConfig):
    """
    UI configuration schema for BaseOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Base Operation UI Configuration",
        "description": "UI schema for base operation configuration.",
        "properties": {
            "name": {"x-component": "Input"},
            "description": {"x-component": "Input"},
            "scope": {},
            "config": {},
            "optimize_memory": {"x-component": "Checkbox"},
            "adaptive_chunk_size": {"x-component": "Checkbox"},
            "mode": {
                "x-component": "Select",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "output_field_name": {
                "x-component": "Input",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                "x-depend-on": {"mode": "ENRICH"},
            },
            "column_prefix": {
                "x-component": "Input",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                "x-depend-on": {"mode": "ENRICH", "output_field_name": "null"},
            },
            "null_strategy": {
                "x-component": "Select",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "engine": {"x-component": "Select"},
            "use_dask": {"x-component": "Checkbox"},
            "npartitions": {"x-component": "NumberPicker"},
            "dask_partition_size": {"x-component": "Input"},
            "use_vectorization": {"x-component": "Checkbox"},
            "parallel_processes": {"x-component": "NumberPicker"},
            "chunk_size": {"x-component": "NumberPicker"},
            "use_cache": {"x-component": "Checkbox"},
            "output_format": {
                "x-component": "Select",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "save_output": {
                "x-component": "Checkbox",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "visualization_theme": {"x-component": "Input"},
            "visualization_backend": {"x-component": "Select"},
            "visualization_strict": {"x-component": "Checkbox"},
            "visualization_timeout": {"x-component": "NumberPicker"},
            "use_encryption": {"x-component": "Checkbox"},
            "encryption_mode": {"x-component": "Select"},
            "encryption_key": {"x-component": "Input"},
            "force_recalculation": {
                "x-component": "Checkbox",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "generate_visualization": {
                "x-component": "Checkbox",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
        },
    }
