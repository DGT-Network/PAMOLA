"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Uniform Temporal Noise operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Uniform Temporal Noise anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Uniform Temporal Noise operation tooltip file
"""


class UniformTemporalNoiseOpTooltip:

    noise_range_days = (
        "What it does: Adds or subtracts a random number of days up to this value from each timestamp.\n"
        "• Example: A value of `7` will shift dates by a random amount within the range of -7 to +7 days "
        "(unless the Shift Direction is restricted).\n"
        "• Validation: Must be a positive number. At least one noise range (days, hours, etc.) must be set."
    )

    noise_range_hours = (
        "What it does: Adds or subtracts a random number of hours up to this value from each timestamp.\n"
        "• Example: A value of `12` will shift times by a random amount within the range of -12 to +12 hours.\n"
        "• Validation: Must be a positive number."
    )

    noise_range_minutes = (
        "What it does: Adds or subtracts a random number of minutes up to this value from each timestamp.\n"
        "• Example: A value of `30` will shift times by a random amount within the range of -30 to +30 minutes.\n"
        "• Validation: Must be a positive number."
    )

    noise_range_seconds = (
        "What it does: Adds or subtracts a random number of seconds up to this value from each timestamp.\n"
        "• Example: A value of `60` will shift times by a random amount within the range of -60 to +60 seconds.\n"
        "• Validation: Must be a positive number."
    )

    direction = (
        "What it does: Restricts the random time shift to a specific direction.\n"
        "• Both: The shift can be positive or negative (e.g., +/- 7 days). (Default)\n"
        "• Forward: The shift will only be positive, moving the date into the future.\n"
        "• Backward: The shift will only be negative, moving the date into the past."
    )

    preserve_special_dates = (
        "How it works: If a date in your data matches one of the dates listed in the Special Dates field, "
        "noise will not be applied to it, and the original value will be kept.\n"
        "• Example: You can use this to ensure that dates like `2025-01-01` or `2025-12-25` are never altered.\n"
        "• Default: `False`."
    )

    special_dates = (
        "What it does: Enter a list of dates to exclude from the noise operation.\n"
        "• Example: `2025-01-01, 2025-07-04, 2025-12-25`.\n"
        "• Note: The time component is ignored; only the date part is used for matching."
    )

    preserve_weekends = (
        "How it works: If a random shift moves a Saturday to a Monday, this setting will adjust the final date "
        "to the nearest Saturday to preserve its 'weekend' status. The adjustment will search up to 3 days forward or backward.\n"
        "• Impact: Maintains the day-of-week pattern in your data, which can be important for analysis, "
        "but slightly reduces the randomness of the noise.\n"
        "• Default: `False`."
    )

    preserve_time_of_day = (
        "How it works: This separates the date and time components. The random shift is applied only to the date, "
        "and then the original time is re-attached.\n"
        "• Example: `2025-06-15 14:30:00` with a +2 day shift becomes `2025-06-17 14:30:00`.\n"
        "• Impact: Useful for preserving patterns related to the time of day (e.g., business hours).\n"
        "• Default: `False`."
    )

    output_granularity = (
        "What it does: Truncates the final date/time to the selected unit, removing finer-grained details.\n"
        "• Example: If set to 'Day', a noisy timestamp of `2025-06-17 14:30:00` would become `2025-06-17 00:00:00`.\n"
        "• Options: `Day`, `Hour`, `Minute`, `Second`."
    )

    min_datetime = (
        "What it does: Acts as a floor for the anonymized dates. If a random shift results in a date earlier than this, "
        "the value will be clipped to this minimum boundary.\n"
        "• Example: If set to `2020-01-01`, no anonymized date will be earlier than that.\n"
        "• Impact: Prevents the generation of unrealistic dates (e.g., dates in the distant past)."
    )

    max_datetime = (
        "What it does: Acts as a ceiling for the anonymized dates. If a random shift results in a date later than this, "
        "the value will be clipped to this maximum boundary.\n"
        "• Example: If set to `2030-12-31`, no anonymized date will be later than that.\n"
        "• Impact: Prevents the generation of unrealistic dates (e.g., dates far in the future)."
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
            "noise_range_days": cls.noise_range_days,
            "noise_range_hours": cls.noise_range_hours,
            "noise_range_minutes": cls.noise_range_minutes,
            "noise_range_seconds": cls.noise_range_seconds,
            "direction": cls.direction,
            "preserve_special_dates": cls.preserve_special_dates,
            "special_dates": cls.special_dates,
            "preserve_weekends": cls.preserve_weekends,
            "preserve_time_of_day": cls.preserve_time_of_day,
            "output_granularity": cls.output_granularity,
            "min_datetime": cls.min_datetime,
            "max_datetime": cls.max_datetime,
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
