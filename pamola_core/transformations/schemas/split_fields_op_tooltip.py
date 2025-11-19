"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split Fields Operation Tooltips
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for split fields configuration fields in PAMOLA.CORE.
- Explains field grouping, ID field inclusion, and output control options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of field splitting operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split fields tooltip file
"""


class SplitFieldsOperationTooltip:
    id_field = "What it does: Specifies which field in the dataset serves as the unique identifier for each record (row)"

    include_id_field = "What it does: Controls whether the ID field specified above is automatically added to every field group output"

    field_groups = "What it does: Allows you to search and select multiple fields from the dataset to add to a field group"

    output_format = "What it does: Specifies the file format used to save each partition as a separate file"

    save_output = (
        "What it does: Controls whether the generated partition files are actually saved to storage or just processed in memory\n"
        "• When checked, creates separate files for each partition that can be downloaded or accessed later.\n"
        "• When unchecked, partitions are created in memory only for immediate downstream processing"
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = (
        "What it does: Forces the operation to recalculate results from scratch, ignoring any existing cached results. Useful when you want "
        "to ensure results reflect any subtle data changes"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "id_field": cls.id_field,
            "include_id_field": cls.include_id_field,
            "field_groups": cls.field_groups,
            "output_format": cls.output_format,
            "save_output": cls.save_output,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
