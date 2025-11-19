from enum import Enum


class AnalysisMode(Enum):
    """K-anonymity analysis modes."""

    ANALYZE = "ANALYZE"  # Generate metrics and reports
    ENRICH = "ENRICH"  # Add k-values to DataFrame
    BOTH = "BOTH"  # Both analyze and enrich
