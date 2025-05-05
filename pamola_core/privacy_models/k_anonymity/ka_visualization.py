"""
PAMOLA.CORE - k-Anonymity Visualization Utilities
-------------------------------------------------
This module provides specialized visualization functions for k-anonymity
anonymization model. It includes histograms, heatmaps, and other visualizations
specifically designed for understanding and assessing k-anonymity properties
of datasets.

These visualizations help in:
- Analyzing the distribution of k-values across records
- Identifying re-identification risk patterns
- Evaluating the effectiveness of k-anonymity transformations
- Communicating anonymization guarantees to stakeholders

Note: For visualizations related to other anonymization models (l-diversity,
t-closeness, etc.), please use their respective visualization modules.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import logging
import os
from datetime import datetime
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from pamola_core.utils.file_io import save_plot  # Use centralized function


# Configure logging
logger = logging.getLogger(__name__)

def visualize_k_distribution(data, k_column: str,
                             save_path: Optional[str] = None,
                             title: str = "Distribution of k-Anonymity Group Sizes",
                             figsize: Tuple[int, int] = (10, 6),
                             bins: int = 20,
                             save_format: str = "png") -> Tuple[plt.Figure, Optional[str]]:
    """
    Creates and optionally saves a visualization of k-anonymity distribution.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset with k-values.
    k_column : str
        Name of the column containing k-values.
    save_path : str, optional
        Path to save the visualization. If None, the figure is not saved.
        If a directory is provided, a timestamped filename will be generated.
    title : str, optional
        Title for the visualization (default: "Distribution of k-Anonymity Group Sizes").
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    bins : int, optional
        Number of bins for the histogram (default: 20).
    save_format : str, optional
        Format to save the figure (png, pdf, svg) (default: "png").

    Returns:
    --------
    tuple
        Figure object and path to saved figure (if saved, otherwise None).
    """
    # Validate input
    if data is None:
        raise ValueError("Input data cannot be None")
    if k_column not in data.columns:
        raise ValueError(f"Column {k_column} not found in dataset")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        sns.histplot(data[k_column].clip(upper=20), bins=bins, kde=True, ax=ax)

        # Add reference line for k threshold if present in dataset attributes
        if 'k' in data.attrs:
            k_threshold = data.attrs['k']
            ax.axvline(x=k_threshold, color='red', linestyle='--',
                       label=f'k-threshold = {k_threshold}')
            ax.legend()

        # Set labels and title
        ax.set_xlabel("k-Anonymity Group Size")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Save figure if a save path is provided
        saved_path = None
        if save_path:
            save_path = Path(save_path)
            if save_path.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = save_path / f"k_distribution_{timestamp}.{save_format}"

            saved_path = save_plot(fig, save_path, save_format)
            logger.info(f"k-distribution plot saved to {saved_path}")

        return fig, saved_path

    except Exception as e:
        logger.error(f"Error during k-distribution visualization: {e}", exc_info=True)
        raise



def visualize_risk_heatmap(data: pd.DataFrame,
                           risk_column: str,
                           feature_columns: List[str],
                           save_path: Optional[str] = None,
                           title: str = "Re-identification Risk Heatmap",
                           figsize: Tuple[int, int] = (12, 10),
                           save_format: str = "png") -> Tuple[plt.Figure, Optional[str]]:
    """
    Creates a heatmap visualizing re-identification risk across feature combinations
    based on k-anonymity values.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with risk values derived from k-anonymity analysis.
    risk_column : str
        Name of the column containing risk values (typically 1/k).
    feature_columns : list[str]
        Columns to include in the heatmap, often quasi-identifiers.
    save_path : str, optional
        Path to save the visualization.
    title : str, optional
        Title for the heatmap.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    save_format : str, optional
        Format to save the figure (png, pdf, svg) (default: "png").

    Returns:
    --------
    tuple
        Figure object and path to saved figure (if saved, otherwise None).
    """
    # Validate input
    if data is None:
        raise ValueError("Input data cannot be None")
    if risk_column not in data.columns:
        raise ValueError(f"Column {risk_column} not found in dataset")
    if feature_columns is None:
        raise ValueError("Feature columns list cannot be None")

    try:
        # Prepare data for heatmap
        risk_by_feature = {}

        for feature in feature_columns:
            if feature in data.columns:
                # Calculate average risk by feature value
                feature_risk = data.groupby(feature)[risk_column].mean().to_dict()
                risk_by_feature[feature] = feature_risk

        # Create matrix for heatmap
        matrix = pd.DataFrame(risk_by_feature).fillna(0)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(matrix, annot=True, cmap="YlOrRd", ax=ax)

        # Set labels and title
        ax.set_title(title)

        # Save figure if path provided
        saved_path = None
        if save_path:
            # Create directory if it doesn't exist
            if os.path.isdir(save_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"risk_heatmap_{timestamp}.{save_format}"
                full_path = os.path.join(save_path, filename)
            else:
                full_path = save_path

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)

            # Save figure
            plt.savefig(full_path, dpi=300, bbox_inches="tight", format=save_format)
            saved_path = full_path
            logger.info(f"Risk heatmap saved to {full_path}")

        return fig, saved_path

    except Exception as e:
        logger.error(f"Error during risk heatmap visualization: {e}")
        raise

def visualize_attribute_correlation(data, quasi_identifiers: List[str],
                                    sensitive_attributes: List[str],
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 10),
                                    save_format: str = "png") -> Tuple[plt.Figure, Optional[str]]:
    """
    Creates a correlation heatmap between quasi-identifiers and sensitive attributes
    to assess potential information leaks in k-anonymized data.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze, typically after k-anonymization.
    quasi_identifiers : list[str]
        List of column names used as quasi-identifiers.
    sensitive_attributes : list[str]
        List of column names with sensitive information.
    save_path : str, optional
        Path to save the visualization.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    save_format : str, optional
        Format to save the figure (png, pdf, svg) (default: "png").

    Returns:
    --------
    tuple
        Figure object and path to saved figure (if saved, otherwise None).
    """
    # Validate input
    if data is None:
        raise ValueError("Input data cannot be None")

    # Check if columns exist in dataframe
    missing_columns = [col for col in quasi_identifiers + sensitive_attributes if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in dataset: {missing_columns}")

    try:
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])

        # Filter columns
        valid_qi = [col for col in quasi_identifiers if col in numeric_data.columns]
        valid_sa = [col for col in sensitive_attributes if col in numeric_data.columns]

        if not valid_qi or not valid_sa:
            raise ValueError("Insufficient numeric columns for correlation analysis")

        # Calculate correlation matrix
        correlation = numeric_data[valid_qi + valid_sa].corr()

        # Extract QI-SA correlations only
        qi_sa_corr = correlation.loc[valid_qi, valid_sa]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(qi_sa_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)

        # Set labels and title
        ax.set_title("Correlation between Quasi-identifiers and Sensitive Attributes")

        # Save figure if path provided
        saved_path = None
        if save_path:
            save_path = Path(save_path)
            if save_path.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = save_path / f"correlation_{timestamp}.{save_format}"

            saved_path = save_plot(fig, save_path, save_format)
            logger.info(f"Correlation heatmap saved to {saved_path}")

        return fig, saved_path

    except Exception as e:
        logger.error(f"Error during correlation visualization: {e}", exc_info=True)
        raise
