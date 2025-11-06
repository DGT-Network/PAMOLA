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


class DateTimeOpTooltip:
    strategy = (
        "What it does: Selects the main logic for making your timestamps less specific.\n"
        "• Rounding: Reduces precision to a larger time unit.\n"
        "• Binning: Groups timestamps into defined intervals.\n"
        "• Component: Keeps only specific parts of the date/time.\n"
        "• Relative: Converts timestamps into relative descriptions."
    )

    rounding_unit = (
        "How it works: All timestamps will be floored to the beginning of the selected time unit (e.g., year, month, day, hour).\n"
        "Impact: A larger unit (e.g., `year`) provides more privacy but loses more detail."
    )

    bin_type = (
        "What it does: Selects the method for grouping timestamps.\n"
        "• Hour/Day Range: Groups into fixed-size intervals.\n"
        '• Business Period: Groups into "Morning", "Afternoon", "Night".\n'
        '• Seasonal: Groups into "Winter", "Spring", "Summer", "Fall".\n'
        "• Custom: Define your own exact boundaries."
    )

    interval_size = (
        "How it works: Defines the duration of each bin, used with Interval Unit.\n"
        "Example: `7` with `days` creates weekly blocks."
    )

    interval_unit = "What it does: Sets the unit of measurement for the Interval Size."

    custom_bins = (
        "How it works: Provide a list of timestamps to serve as boundaries.\n"
        'Example: `["2023-01-01", "2023-07-01", "2024-01-01"]` creates two bins for the first and second half of 2023.'
    )

    reference_date = (
        'What it does: The "zero point" for relative time calculation. If left blank, the system uses the current date and time.\n'
        'Example: If this is `2024-01-01`, a value of `2023-12-15` might become "Days ago".'
    )

    keep_components = (
        "What it does: Creates a new value by combining only the selected date/time components.\n"
        'Example: Selecting `year` and `month` turns `2023-10-27 15:45:00` into `"2023-10"`.'
    )

    strftime_output_format = "How it works: Uses standard date formatting codes (e.g., %Y-%b) to create a custom string. Note: This will convert your date column into a text column."

    timezone_handling = (
        "What it does:\n"
        "• preserve: Keeps the original timezone (default).\n"
        "• utc: Converts all timestamps to UTC for consistency.\n"
        "• remove: Strips all timezone information."
    )

    default_timezone = "What it does: When converting to UTC, if a timestamp has no timezone info, it's assumed to be in this timezone first."

    input_formats = (
        "What it does: If your date is stored as text in a non-standard format, provide formatting codes to help parse it.\n"
        'Example: For "27/10/2023", provide `["%d/%m/%Y"]`.'
    )

    mode = (
        "How do you want the output?\n"
        "• REPLACE: Overwrites the original column.\n"
        "• ENRICH: Keeps the original and adds a new, anonymized column."
    )

    output_field_name = "Name the new column. If left blank, a name is auto-generated using the prefix below (e.g., _login_time)."

    column_prefix = "Prefix for the new column name. The default is an underscore (_), which turns login_time into _login_time."

    null_strategy = (
        "What to do with empty cells?\n"
        "• PRESERVE: Leave them as they are.\n"
        "• EXCLUDE: Remove rows with empty values.\n"
        "• ERROR: Stop if empty values are found.\n"
        '• ANONYMIZE: Replace with a "Not a Time" (`NaT`) placeholder.'
    )

    min_privacy_threshold = (
        "What it does: A safety check to ensure your settings provide enough privacy. It's a ratio from 0.0 to 1.0.\n"
        "Example: `0.3` means the number of unique dates must be reduced by at least 30%."
    )

    force_recalculation = "Ignore saved results. Check this to force the operation to run again instead of using a cached result."

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "strategy": cls.strategy,
            "rounding_unit": cls.rounding_unit,
            "bin_type": cls.bin_type,
            "interval_size": cls.interval_size,
            "interval_unit": cls.interval_unit,
            "custom_bins": cls.custom_bins,
            "reference_date": cls.reference_date,
            "keep_components": cls.keep_components,
            "strftime_output_format": cls.strftime_output_format,
            "timezone_handling": cls.timezone_handling,
            "default_timezone": cls.default_timezone,
            "input_formats": cls.input_formats,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "min_privacy_threshold": cls.min_privacy_threshold,
            "force_recalculation": cls.force_recalculation,
        }
