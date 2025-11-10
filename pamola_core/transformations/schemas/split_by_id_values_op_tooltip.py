"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split By ID Values Operation Tooltips
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for split by ID values configuration fields in PAMOLA.CORE.
- Explains partitioning strategies, value groups, and output control options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of splitting operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split by ID values tooltip file
"""


class SplitByIDValuesOperationTooltip:
    number_of_partitions = (
        "What it does: Specifies the total number of separate datasets to create when using automatic partitioning mode\n"
        "Example: Number of partitions=4 with 1000 records creates 4 files with ~250 records each (exact distribution depends on partition method)"
    )

    partition_method = (
        "What it does: Determines the algorithm used to assign records to partitions when using automatic partitioning mode\n"
        "• Equal Size: aims for roughly equal record counts per partition\n"
        "• Random: randomly assigns each record to a partition\n"
        "• Modulo: uses mathematical modulo operation on ID values for deterministic assignment"
    )

    id_field = "What it does: Specifies which column contains the unique identifiers that will be used to split and partition the dataset"

    value_groups = (
        "What it does: Allows you to manually create named groups by specifying exactly which ID values should be included in each group\n"
        'Example: Create group "test_users" with ID values ["001", "045", "199"] and group "control_users" with values ["023", "067", "180"] - '
        "produces test_users.csv and control_users.csv files"
    )

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
            "partition_method": cls.partition_method,
            "number_of_partitions": cls.number_of_partitions,
            "id_field": cls.id_field,
            "value_groups": cls.value_groups,
            "output_format": cls.output_format,
            "save_output": cls.save_output,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
