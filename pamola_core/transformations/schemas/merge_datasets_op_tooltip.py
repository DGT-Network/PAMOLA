"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for numeric generalization configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for numeric anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric operation tooltip file
"""



class MergeDatasetsOperationTooltip:
    left_key = (
        "What it does: Specifies which column in the left dataset to use for matching records with the right dataset"
    )
    right_key = (
        "What it does: Specifies which column in the right dataset to use for matching records with the left dataset"
    )
    join_type = (
        "What it does: Determines which records from left and right datasets appear in the merged result\n"
        "•  Inner: keeps only matching records\n"
        "•  Left: keeps all left + matching right\n"
        "•  Right: keeps all right + matching left\n"
        "•  Outer: keeps all records from both"
    )
    relationship_type = (
        "What it does: Enforces or auto-detects the cardinality relationship between datasets based on key uniqueness\n"
        "•  Auto: detects based on key uniqueness\n"
        "•  One-to-one: requires both keys unique\n"
        "•  One-to-many: requires right key unique."
    )
    suffixes_0 = (
        'What it does: Appends this suffix to column names from the left dataset when both datasets have columns with identical names (excluding join keys)\n'
        'Example: Left column suffix="_main" with overlapping "price" column creates "price_main" from left dataset'
    )
    suffixes_1 = (
        'What it does: Appends this suffix to column names from the right dataset when both datasets have columns with identical names (excluding join keys) \n'
        'Example: Right column suffix="_lookup" with overlapping "category" column creates "category_lookup" from right dataset'
    )
    output_format = (
        "What it does: Specifies the file format for saving the merged dataset output "
    )
    save_output = (
        "What it does: Controls whether the merged dataset is written to a file in the output directory"
    )
    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"
    )
    force_recalculation = (
        "What it does: Forces the operation to recalculate results from scratch, ignoring any existing cached results. Useful when you want to ensure results reflect any subtle data changes"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "left_key": cls.left_key,
            "right_key": cls.right_key,
            "join_type": cls.join_type,
            "relationship_type": cls.relationship_type,
            "suffixes[0]": cls.suffixes_0,
            "suffixes[1]": cls.suffixes_1,
            "output_format": cls.output_format,
            "save_output": cls.save_output,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }

