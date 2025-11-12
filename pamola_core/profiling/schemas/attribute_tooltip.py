"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Data Attribute Profiler Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for data attribute profiler configuration fields in PAMOLA.CORE.
- Explains language settings, sampling, column limits, and dictionary options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of attribute profiling operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of data attribute profiler tooltip file
"""


class DataAttributeProfilerOperationTooltip:
    language = (
        "What it does: Sets the language for keyword matching and attribute role detection during processing.\n"
        "• Affects how the system interprets field names and values\n"
        "• Common codes: 'en' (English), 'vi' (Vietnamese), 'fr' (French), 'de' (German)\n"
        "• Default: 'en' (English)\n"
        "Example: Use 'vi' if your dataset contains Vietnamese column names or needs Vietnamese-specific processing rules."
    )

    sample_size = (
        "What it does: Specifies how many actual data values to extract and inspect from each column for pattern detection and validation\n"
        "Example: Sample size=10 extracts 10 random values from 'birth_date' column like ['1985-03-15', '1992-11-22', ...] to confirm it's a "
        "date field and detect format patterns"
    )

    max_columns = (
        "What it does: Limits the number of columns to analyze, processing only the first N columns in the dataset\n"
        "Example: Max columns=50 on a 200-column dataset analyzes only columns 1-50; leaving empty analyzes all 200 columns"
    )

    id_column = (
        "What it does: Specifies the name of the column that uniquely identifies each record/entity in the dataset for record-level analysis.\n"
        "Specifying ID column improves accuracy of quasi-identifier detection by ~15-20%. Without it, the operation relies only on statistical "
        "uniqueness which may miss semantic relationships.\n"
        "Example: Id column='user_id' tells the operation that 'user_id' uniquely identifies each person, helping detect if other columns like "
        "'email' are also unique per user"
    )

    dictionary_path = (
        "What it does: Path to a custom attribute dictionary file for role detection.\n"
        "• Leave empty to use the default built-in dictionary\n"
        "• Specify a file path to use custom attribute definitions for identifying field types and roles"
    )

    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value "
        "count distributions"
    )

    force_recalculation = (
        "What it does: Ignore saved results. Check this to force the operation to run again instead of using a cached result."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "language": cls.language,
            "sample_size": cls.sample_size,
            "max_columns": cls.max_columns,
            "id_column": cls.id_column,
            "dictionary_path": cls.dictionary_path,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }