"""
PAMOLA.CORE - Privacy Risk Assessment Module
------------------------------------------------
Module:        Privacy Risk Assessment
Package:       pamola_core.analysis
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Field-level analysis utilities for dataset privacy and quality assessment.
  Provides descriptive statistics, distribution visualizations and simple
  privacy-aware heuristics (uniqueness, missingness) intended for integration
  in larger anonymization and data governance workflows.

Key Features:
  - Field-level descriptive statistics and extra statistics integration
  - Distribution visualization support (histograms / bar charts)
  - Computes missingness and uniqueness indicators
  - Safe defaults, input validation and logging for diagnostics
  - Designed for easy extension with privacy metrics (k-anonymity, l-diversity)

Dependencies:
  - pandas - DataFrame operations
  - pathlib - Path handling for visualization outputs
  - typing - Type hints
  - pamola_core.analysis.descriptive_stats - Descriptive stats helper
  - pamola_core.analysis.distribution - Distribution visualization helper
  - pamola_core.utils.logging - Module logging helper
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from pamola_core.analysis.descriptive_stats import analyze_descriptive_stats
from pamola_core.analysis.distribution import visualize_distribution_df
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


def analyze_field_level(
    df: pd.DataFrame, field_name: str, viz_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze field level of data frame.

    Parameters:
    -----------
    df : pd.DataFrame
        Data frame for calculate.
    field_name : str
        Name of the field to analyze.
    viz_dir : Path
        Directory for saving visualizations

    Returns:
    --------
    Dict[str, Any]
        Field-Level analysis.
    """
    if pd.api.types.is_numeric_dtype(df[field_name]):
        describe_order = ["count", "mean", "std", "min", "max"]
        extra_statistics = ["unique", "median", "mode"]
    else:
        describe_order = ["count", "unique", "top", "freq"]
        extra_statistics = ["mode"]

    field_level_analysis = analyze_descriptive_stats(
        df=df,
        field_names=[field_name],
        describe_order=describe_order,
        extra_statistics=extra_statistics,
    )

    if viz_dir is None:
        viz_dir = Path.cwd()

    field_level_visualization = visualize_distribution_df(
        df=df,
        viz_dir=viz_dir,
        numeric_bar_charts=True,
        n_bins=10,
        field_names=[field_name],
    )

    return {
        "field_level_analysis": field_level_analysis[field_name],
        "field_level_visualization": field_level_visualization[field_name],
    }
