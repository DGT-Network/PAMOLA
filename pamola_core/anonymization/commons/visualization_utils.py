"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Anonymization Visualization Utilities
Description: Helper utilities for creating visualizations in anonymization operations
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides helper utilities for creating visualizations in anonymization
operations, simplifying interactions with the pamola_core visualization system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Union
from pamola_core.common.constants import Constants

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def generate_visualization_filename(
    field_name: str,
    operation_name: str,
    visualization_type: str,
    timestamp: Optional[str] = None,
    extension: str = "png",
) -> str:
    """
    Generate a standardized filename for a visualization.

    Parameters:
    -----------
    field_name : str
        Name of the field being visualized
    operation_name : str
        Name of the operation creating the visualization
    visualization_type : str
        Type of visualization (e.g., "histogram", "distribution")
    timestamp : str, optional
        Timestamp for file naming. If None, current timestamp is used.
    extension : str, optional
        File extension (default: "png")

    Returns:
    --------
    str
        Standardized filename
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{field_name}_{operation_name}_{visualization_type}_{timestamp}.{extension}"


def register_visualization_artifact(
    result: Any,
    reporter: Any,
    path: Path,
    field_name: str,
    visualization_type: str,
    description: Optional[str] = None,
) -> None:
    """
    Register a visualization artifact with the result and reporter.

    Parameters:
    -----------
    result : Any
        Operation result to add the artifact to
    reporter : Any
        Reporter to add the artifact to
    path : Path
        Path to the visualization file
    field_name : str
        Name of the field being visualized
    visualization_type : str
        Type of visualization
    description : str, optional
        Custom description of the visualization
    """
    if description is None:
        description = f"{field_name} {visualization_type} visualization"

    # Add to result
    if hasattr(result, "add_artifact"):
        result.add_artifact(
            artifact_type="png",
            path=path,
            description=description,
            category=Constants.Artifact_Category_Visualization,
        )

    # Add to reporter
    if reporter and hasattr(reporter, "add_artifact"):
        reporter.add_artifact("png", str(path), description)


def sample_large_dataset(
    data: Union[pd.Series, pd.DataFrame],
    max_samples: int = 10000,
    random_state: int = 42,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Sample a large Series or DataFrame to a manageable size for visualization.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        Original large dataset
    max_samples : int, optional
        Maximum number of samples to return (default: 10000)
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Sampled subset of the original data
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame")

    if len(data) <= max_samples:
        return data

    return data.sample(n=max_samples, random_state=random_state)


def prepare_comparison_data(
    original_data: pd.Series,
    anonymized_data: pd.Series,
    data_type: str = "auto",
    max_categories: int = 10,
) -> Tuple[Dict[str, Any], str]:
    """
    Prepare data for comparison visualizations based on data type.

    Parameters:
    -----------
    original_data : pd.Series
        Original data
    anonymized_data : pd.Series
        Anonymized data
    data_type : str, optional
        Force specific data type ('numeric', 'categorical') or 'auto' to detect
    max_categories : int, optional
        Maximum number of categories for categorical data

    Returns:
    --------
    Tuple[Dict[str, Any], str]
        Prepared data and detected data type
    """
    # Determine data type if auto
    if data_type == "auto":
        if pd.api.types.is_numeric_dtype(original_data):
            data_type = "numeric"
        else:
            data_type = "categorical"

    # Clean data
    original_clean = original_data.dropna()
    anonymized_clean = anonymized_data.dropna()

    # Prepare according to data type
    if data_type == "numeric":
        # For numeric data, simple conversion to list
        return {
            "Original": original_clean.tolist(),
            "Anonymized": anonymized_clean.tolist(),
        }, data_type

    elif data_type == "categorical":
        # For categorical data, get value counts
        orig_counts = original_data.value_counts().head(max_categories)
        anon_counts = anonymized_data.value_counts().head(max_categories)

        # Combine categories
        all_categories = set(orig_counts.index) | set(anon_counts.index)
        all_categories = list(all_categories)[
            :max_categories
        ]  # Limit to max_categories

        # Create aligned dictionaries
        orig_dict = {str(cat): int(orig_counts.get(cat, 0)) for cat in all_categories}
        anon_dict = {str(cat): int(anon_counts.get(cat, 0)) for cat in all_categories}

        return {"Original": orig_dict, "Anonymized": anon_dict}, data_type

    else:
        logger.warning(f"Unknown data type: {data_type}")
        return {}, "unknown"


def calculate_optimal_bins(
    data: pd.Series, min_bins: int = 5, max_bins: int = 30
) -> int:
    """
    Calculate optimal number of bins for histograms using square root rule.

    Parameters:
    -----------
    data : pd.Series
        Data to calculate bins for
    min_bins : int, optional
        Minimum number of bins
    max_bins : int, optional
        Maximum number of bins

    Returns:
    --------
    int
        Optimal number of bins
    """
    non_null_count = len(data.dropna())
    # Square root rule: bins ≈ √n
    optimal_bins = int(np.sqrt(non_null_count))
    return max(min_bins, min(optimal_bins, max_bins))


def create_visualization_path(task_dir: Path, filename: str) -> Path:
    """
    Create full path for visualization with directory creation if needed.

    Parameters:
    -----------
    task_dir : Path
        Base task directory
    filename : str
        Filename for the visualization

    Returns:
    --------
    Path
        Full path to the visualization file
    """
    # Create visualization directory if it doesn't exist
    viz_dir = task_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Return full path
    return viz_dir / filename
