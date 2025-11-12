"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Impute Missing Values Operation Tooltips
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Provides detailed tooltips for impute missing values operation configuration fields in PAMOLA.CORE.
- Explains field-specific imputation strategies and invalid value definitions
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of missing data imputation operations

Changelog:
1.0.0 - 2025-11-11 - Initial creation of impute missing values operation tooltip file
"""


class ImputeMissingValuesOperationTooltip:
    field_strategies = (
        "What it does: Defines the imputation method and configuration for each field with missing values.\n\n"
        "Each field requires:\n"
        "• data_type: Field type ('numeric', 'categorical', 'date', 'text')\n"
        "• imputation_strategy: Method to fill missing values\n"
        "• constant_value: Value to use (only for 'constant' strategies)\n\n"
        "Strategies by data type:\n"
        "• Numeric: 'constant', 'mean', 'median', 'mode', 'min', 'max', 'interpolation'\n"
        "• Categorical: 'constant', 'mode', 'most_frequent', 'random_sample'\n"
        "• Date: 'constant_date', 'mean_date', 'median_date', 'mode_date', 'previous_date', 'next_date'\n"
        "• Text: 'constant', 'most_frequent', 'random_sample'\n\n"
        "Example:\n"
        "{\n"
        "  'age': {'data_type': 'numeric', 'imputation_strategy': 'median'},\n"
        "  'status': {'data_type': 'categorical', 'imputation_strategy': 'mode'},\n"
        "  'name': {'data_type': 'text', 'imputation_strategy': 'constant', 'constant_value': 'Unknown'}\n"
        "}"
    )

    invalid_values = (
        "What it does: Specifies which values should be treated as missing for each field.\n\n"
        "These values will be replaced with NaN before imputation is applied.\n\n"
        "Common placeholders:\n"
        "• Numeric: -1, -999, 0, 99999\n"
        "• Text: 'N/A', 'Unknown', 'null', '', '--'\n\n"
        "Format: Dictionary mapping field names to lists of invalid values.\n\n"
        "Example: {'age': [-1, 0], 'name': ['N/A', 'Unknown', '']}"
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = "What it does: Forces the operation to recalculate results from scratch, ignoring any existing cached results. Useful when you want to ensure results reflect any subtle data changes"

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "field_strategies": cls.field_strategies,
            "invalid_values": cls.invalid_values,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
