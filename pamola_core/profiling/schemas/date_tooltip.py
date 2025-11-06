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

class DateOpTooltip:
    min_year = (
        "What it does: Sets the earliest acceptable year for the date field.\n"
        "• Years < min_year are counted as anomalies and examples are saved to anomalies JSON and CSV."
    )

    max_year = (
        "What it does: Sets the latest acceptable year for the date field.\n"
        "• Years > max_year AND <= current year are counted as anomalies.\n"
        "• Future dates (year > current year) are tracked separately as 'future_dates' anomalies.\n"
        "• Does not filter data, only flags anomalies."
    )

    id_column = (
        "What it does: When provided, system identifies groups where the date field has multiple different values across records with the same ID.\n"
        "• Useful for detecting data quality issues where same entity has conflicting dates.\n"
        "• Example: 'resume_123 has birth dates: 1985-05-10, 1985-05-15'."
    )

    uid_column = (
        "What it does: When provided, system identifies UIDs (unique identifiers) where the same person has multiple different date values across their records.\n"
        "• Critical for detecting identity conflicts or data quality issues.\n"
        "• Example: 'person_456 has birth dates: 1990-03-15, 1990-03-20' - indicates data quality issue requiring investigation."
    )

    is_birth_date = (
        "What it does: When checked, system calculates age distribution (how many people in each age group), validates dates are biologically plausible for birth dates, and provides age-specific statistics.\n"
        "• Example: Checking this for 'birth_date' field adds age_distribution to output showing counts like: ages 20-30: 150 people, ages 30-40: 200 people, ages 40-50: 175 people."
    )

    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "min_year": cls.min_year,
            "max_year": cls.max_year,
            "id_column": cls.id_column,
            "uid_column": cls.uid_column,
            "is_birth_date": cls.is_birth_date,
            "generate_visualization": cls.generate_visualization,
        }
