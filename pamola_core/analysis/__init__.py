"""
Initialize the analysis package.
"""

from .dataset_summary import analyze_dataset_summary
from .correlation import analyze_correlation
from .descriptive_stats import analyze_descriptive_stats
from .distribution import visualize_distribution_df
from .field_analysis import analyze_field_level
from .privacy_risk import calculate_full_risk


# Make operations available at package level
__all__ = [
    "analyze_dataset_summary",
    "analyze_correlation",
    "analyze_descriptive_stats",
    "visualize_distribution_df",
    "analyze_field_level",
    "calculate_full_risk",
]