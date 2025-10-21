"""
PAMOLA.CORE - Distribution Visualization Module
------------------------------------------------
Module:        Distribution & Visualization Analyzer
Package:       pamola_core.analysis
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Utilities to generate distribution visualizations and summary data for pandas
  DataFrames. Provides histogram and bar-chart generation with configurable
  binning, numeric-to-categorical handling, timestamped output paths and
  integration with the pamola_core visualization helpers.

Key Features:
  - Generates histograms and bar charts for numeric and categorical fields
  - Supports numeric binning, frequency and normalized value counts
  - Produces timestamped output files and returns path map for downstream use
  - Uses pamola_core.utils.visualization for backend-agnostic plotting
  - Includes logging and safe handling of missing data

Dependencies:
  - pandas  - DataFrame operations
  - numpy   - Numeric operations and binning
  - typing  - Type hints
  - datetime/pathlib - File naming and paths
  - pamola_core.utils.visualization - Plotting helpers
  - pamola_core.utils.logging - Module logging helper
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pamola_core.utils.visualization import create_bar_plot, create_histogram

logger = logging.getLogger(__name__)


# Constants
DEFAULT_VIZ_FORMAT = "html"
DEFAULT_BACKEND = "plotly"
DEFAULT_N_BINS = 10


def visualize_distribution_df(
    df: pd.DataFrame,
    viz_dir: Path,
    numeric_bar_charts: bool = False,
    n_bins: int = DEFAULT_N_BINS,
    field_names: Optional[List[str]] = None,
    viz_format: str = DEFAULT_VIZ_FORMAT,
) -> Dict[str, Path]:
    """
    Visualize the distribution of fields in a DataFrame.

    Generates histograms for numeric fields and bar charts for categorical fields.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze and visualize
    viz_dir : Path
        Directory for saving visualizations
    numeric_bar_charts : bool, default=False
        If True, visualize numeric fields as binned bar charts instead of histograms
    n_bins : int, default=10
        Number of bins for numeric visualizations
    field_names : list, optional
        Specific field names to analyze. If None, analyzes all columns.
    viz_format : str, default='html'
        Output format for visualizations (e.g., 'html', 'png', 'svg')

    Returns:
    --------
    Dict[str, Path]
        Mapping of field names to their visualization file paths.
    """
    field_names = field_names or df.columns.tolist()
    viz_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    visualization_paths = {}

    for field in field_names:
        series = df[field].dropna()

        if series.empty:
            logger.warning(f"Skipping empty field: {field}")
            continue

        viz_path = _create_visualization(
            series=series,
            field_name=field,
            viz_dir=viz_dir,
            timestamp=timestamp,
            viz_format=viz_format,
            numeric_bar_charts=numeric_bar_charts,
            n_bins=n_bins,
        )

        if viz_path:
            visualization_paths[field] = viz_path

    return visualization_paths


def _create_visualization(
    series: pd.Series,
    field_name: str,
    viz_dir: Path,
    timestamp: str,
    viz_format: str,
    numeric_bar_charts: bool,
    n_bins: int,
) -> Optional[Path]:
    """Route field to appropriate visualization based on data type."""
    if pd.api.types.is_numeric_dtype(series):
        return _visualize_numeric_field(
            series,
            field_name,
            viz_dir,
            timestamp,
            viz_format,
            numeric_bar_charts,
            n_bins,
        )

    if pd.api.types.is_categorical_dtype(series):
        return _visualize_categorical_field(
            series, field_name, viz_dir, timestamp, viz_format
        )

    logger.warning(f"Skipping unsupported dtype for field: {field_name}")
    return None


def _visualize_numeric_field(
    series: pd.Series,
    field_name: str,
    viz_dir: Path,
    timestamp: str,
    viz_format: str,
    numeric_bar_charts: bool,
    n_bins: int,
) -> Optional[Path]:
    """Create histogram or binned bar chart for numeric data."""
    if numeric_bar_charts:
        return _create_numeric_bar_chart(
            series, field_name, viz_dir, timestamp, viz_format, n_bins
        )
    return _create_numeric_histogram(
        series, field_name, viz_dir, timestamp, viz_format, n_bins
    )


def _create_numeric_histogram(
    series: pd.Series,
    field_name: str,
    viz_dir: Path,
    timestamp: str,
    viz_format: str,
    n_bins: int,
) -> Optional[Path]:
    """Generate a histogram for numeric data."""
    filename = f"analysis_histogram_{field_name}_{timestamp}.{viz_format}"
    viz_path = viz_dir / filename

    viz_result = create_histogram(
        output_path=viz_path,
        data=series,
        title=f"Histogram: {field_name}",
        x_label=field_name,
        bins=n_bins,
        viz_format=viz_format,
        backend=DEFAULT_BACKEND,
    )

    return _handle_viz_result(viz_result, viz_path, field_name)


def _create_numeric_bar_chart(
    series: pd.Series,
    field_name: str,
    viz_dir: Path,
    timestamp: str,
    viz_format: str,
    n_bins: int,
) -> Optional[Path]:
    """Generate a bar chart with binned numeric data."""
    bins = np.linspace(series.min(), series.max(), n_bins + 1)
    bin_counts = pd.cut(series, bins=bins, include_lowest=True).value_counts(sort=False)
    viz_data = _format_bin_labels(series, bins, bin_counts)

    filename = f"analysis_bar_chart_{field_name}_{timestamp}.{viz_format}"
    viz_path = viz_dir / filename

    viz_result = create_bar_plot(
        output_path=viz_path,
        data=viz_data,
        title=f"Bar Chart: {field_name}",
        x_label=field_name,
        y_label="Count",
        sort_by="key",
        viz_format=viz_format,
        backend=DEFAULT_BACKEND,
    )

    return _handle_viz_result(viz_result, viz_path, field_name)


def _visualize_categorical_field(
    series: pd.Series,
    field_name: str,
    viz_dir: Path,
    timestamp: str,
    viz_format: str,
) -> Optional[Path]:
    """Create bar chart for categorical data."""
    viz_data = series.value_counts(normalize=True)
    filename = f"analysis_bar_chart_{field_name}_{timestamp}.{viz_format}"
    viz_path = viz_dir / filename

    viz_result = create_bar_plot(
        output_path=viz_path,
        data=viz_data,
        title=f"Bar Chart: {field_name}",
        x_label="Category",
        y_label="Frequency",
        sort_by="key",
        viz_format=viz_format,
        backend=DEFAULT_BACKEND,
    )

    return _handle_viz_result(viz_result, viz_path, field_name)


def _format_bin_labels(
    series: pd.Series, bins: np.ndarray, bin_counts: pd.Series
) -> Dict[str, int]:
    """
    Format bin labels based on data type (integer vs float).

    For integers with bin_width=1, show individual values.
    For other integers, show ranges with integer formatting.
    For floats, show ranges with decimal formatting.
    """
    if not pd.api.types.is_integer_dtype(series):
        # Float formatting
        return {
            f"{interval.left:.2f}–{interval.right:.2f}": count
            for interval, count in bin_counts.items()
        }

    bin_width = bins[1] - bins[0]
    if bin_width == 1:
        # Show individual integer values
        return {
            f"{int(interval.left)}": count for interval, count in bin_counts.items()
        }

    # Integer range formatting
    return {
        f"{int(interval.left)}–{int(interval.right)}": count
        for interval, count in bin_counts.items()
    }


def _handle_viz_result(
    viz_result: str, viz_path: Path, field_name: str
) -> Optional[Path]:
    """Check visualization result and log errors if needed."""
    if isinstance(viz_result, str) and viz_result.startswith("Error"):
        logger.error(f"Failed to create visualization for {field_name}: {viz_result}")
        return None
    return viz_path
