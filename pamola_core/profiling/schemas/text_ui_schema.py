"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Semantic Categorizer UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of text semantic categorizer configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-disabled-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Conditional field disabling using x-disabled-on
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menus for field and entity type selection
- Upload: File upload for dictionary
- NumberPicker: Integer numeric inputs with validation
- FloatPicker: Float numeric inputs with validation
- Checkbox: Boolean toggles

Changelog:
1.0.0 - 2025-01-15 - Initial creation of text semantic categorizer UI schema
1.1.0 - 2025-11-11 - Updated with enhanced UI controls
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class TextSemanticCategorizerOperationUIConfig(OperationConfig):
    """
    UI configuration schema for TextSemanticCategorizerOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Text Semantic Categorizer Operation UI Configuration",
        "description": "UI schema for text semantic categorizer operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "id_field": {
                        "x-component": "Select",
                    },
                    "entity_type": {
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "dictionary_path": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.DICTIONARY_CONFIGURATION,
                    },
                    "min_word_length": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "match_strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "clustering_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "use_ner": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                    "perform_categorization": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                    "perform_clustering": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-disabled-on": {"perform_categorization": False},
                    },
                }
            },
        ],
    }