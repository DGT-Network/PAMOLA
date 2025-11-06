"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Uniform Numeric Noise operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Uniform Numeric Noise anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Uniform Numeric Noise operation tooltip file
"""


class UniformNumericNoiseOpTooltip:

    noise_type = (
        "What it does: Determines how the random noise is combined with the original value.\n"
        "• Additive: Adds the noise value directly to the original number. `new_value = original + noise`. (Default)\n"
        "• Multiplicative: Multiplies the original number by a factor based on the noise. `new_value = original * (1 + noise)`.\n"
        "• Impact: 'Additive' is simpler and more common. 'Multiplicative' is useful when the desired noise level should be proportional "
        "to the magnitude of the original value (e.g., adding +/- 10% noise)."
    )

    noise_range = (
        "What it does: Defines the boundaries for the uniform random noise.\n"
        "• Symmetric: If you enter a single number (e.g., `10`), the noise will be in the range `[-10, 10]`.\n"
        "• Asymmetric: If you enter two numbers (e.g., `-5, 20`), the noise will be in that specific range.\n"
        "• Validation: This field is required. For a range, the first number must be smaller than the second."
    )

    scale_by_std = (
        "How it works: If enabled, the Noise Range is multiplied by the standard deviation of the data. "
        "This makes the noise adaptive to the spread of your data.\n"
        "• Example: If your data has a standard deviation of 100 and you set a Noise Range of `0.5`, "
        "the effective noise range will be `[-50, 50]`.\n"
        "• Impact: Helps maintain a consistent signal-to-noise ratio, preserving more utility in data with low variance.\n"
        "• Default: `False`."
    )

    scale_factor = (
        "What it does: Acts as a final adjustment to the noise level.\n"
        "• Example: A value of `2.0` will double the effective Noise Range. A value of `0.5` will halve it.\n"
        "• Validation: Must be a non-negative number.\n"
        "• Default: `1.0`."
    )

    output_min = (
        "What it does: Acts as a floor for the anonymized numbers. If adding noise results in a value below this floor, "
        "the value will be clipped to this minimum boundary.\n"
        "• Example: If your data is `[10, 20]` and noise is `-15`, the result `[-5, 5]` would be clipped to `[0, 5]` if the minimum is `0`.\n"
        "• Impact: Prevents the generation of unrealistic or invalid numbers (e.g., negative age or salary)."
    )

    output_max = (
        "What it does: Acts as a ceiling for the anonymized numbers. If adding noise results in a value above this ceiling, "
        "the value will be clipped to this maximum boundary.\n"
        "• Example: If you set a maximum of `100`, any noisy value greater than 100 will be set to `100`.\n"
        "• Impact: Prevents the generation of unrealistic or invalid numbers."
    )

    preserve_zero = (
        "How it works: If this is enabled, the operation will skip adding noise to any 0 values, leaving them unchanged.\n"
        "• Impact: Useful when zero has a special meaning in your dataset (e.g., 'not applicable' or 'no quantity') that should not be altered.\n"
        "• Default: `False`."
    )

    round_to_integer = (
        "How it works: If the original data type is an integer, this is enabled by default. "
        "You can override it to keep decimal results.\n"
        "• Example: A noisy value of `123.45` would become `123`. A value of `123.56` would become `124`.\n"
        "• Impact: Helps maintain the original data type and can be considered a form of light generalization."
    )

    condition_logic = (
        "What it does: Determines how to combine the rules set in Multi-field Conditions.\n"
        "• AND: All conditions must be true.\n"
        "• OR: Any of the conditions can be true.\n"
        "• Default: `AND`."
    )

    multi_conditions = (
        "What it does: Allows you to create more advanced filtering logic than the simple condition.\n"
        "• Example: You could apply noise only where `(department == 'Sales' AND salary > 100000)`.\n"
        "• Impact: Provides highly granular control and overrides the simple Condition Field settings if used."
    )

    condition_field = (
        "Which column determines if a row is anonymized?\n"
        "The generalization will only apply to rows that match the condition you set on this column.\n\n"
        "Example: Select the 'Country' column to apply this rule only to customers from a specific country."
    )

    condition_operator = (
        "How should the condition be evaluated?\n"
        "This operator compares the 'Condition Field' with the 'Condition Values'.\n\n"
        "• in: The value must be in your list of values.\n"
        "• not in: The value must *not* be in your list.\n"
        "• > (gt), < (lt), etc.: For Full Maskingal or date comparisons."
    )

    condition_values = (
        "Which values should trigger this rule?\n"
        "Provide a list of values to match against the 'Condition Field'.\n\n"
        "Example: If your 'Condition Field' is 'Country' and you enter `USA`, "
        "this rule will only apply to rows where the country is 'USA'. You can enter "
        "multiple values, separated by commas (e.g., `USA, Canada`)."
    )

    mode = (
        "How do you want the output?\n"
        "• REPLACE: Overwrites the original column.\n"
        "• ENRICH: Keeps the original and adds a new, anonymized column."
    )

    output_field_name = "Name the new column. If left blank, a name will be generated automatically (e.g., _age)."

    column_prefix = "Prefix for the new column name. The default is an underscore (_), which turns age into _age."

    null_strategy = (
        "What to do with empty cells?\n"
        "• PRESERVE: Leave them as they are (default).\n"
        "• EXCLUDE: Temporarily remove rows with empty values before processing.\n"
        "• ERROR: Stop the operation if any empty values are found."
    )

    force_recalculation = (
        "Ignore saved results. Check this box to force the operation to run again "
        "instead of using a cached result from a previous run with the same settings."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "noise_type": cls.noise_type,
            "noise_range": cls.noise_range,
            "scale_by_std": cls.scale_by_std,
            "scale_factor": cls.scale_factor,
            "output_min": cls.output_min,
            "output_max": cls.output_max,
            "preserve_zero": cls.preserve_zero,
            "round_to_integer": cls.round_to_integer,
            "condition_logic": cls.condition_logic,
            "multi_conditions": cls.multi_conditions,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
        }
