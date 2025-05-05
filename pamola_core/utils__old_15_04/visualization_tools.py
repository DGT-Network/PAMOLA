"""
PAMOLA.CORE - UTILITY VISUALIZATION

Provides utility functions for l-diversity visualizations.
This module contains helper functions used by the main
visualization modules.

Key Features:
- Color palette generation
- Axis formatting
- Figure size calculation
- File path handling
- Shared plotting utilities

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Import file saving utility
from pamola_core.utils.file_io import save_plot

# Configure logging
logger = logging.getLogger(__name__)


# ==============================
# Color and Style Utilities
# ==============================

def get_compliant_color(compliant: bool, alpha: float = 1.0) -> str:
    """
    Get a color representing compliance status

    Parameters:
    -----------
    compliant : bool
        Whether the item is compliant
    alpha : float, optional
        Alpha value for transparency (0-1)

    Returns:
    --------
    str
        Color code with alpha
    """
    if compliant:
        return f"rgba(0, 128, 0, {alpha})"  # Green
    else:
        return f"rgba(255, 0, 0, {alpha})"  # Red


def get_risk_color_map() -> Dict[str, str]:
    """
    Get a color map for different risk levels

    Returns:
    --------
    Dict[str, str]
        Mapping from risk level to color
    """
    return {
        'VERY LOW': '#4CAF50',  # Green
        'LOW': '#8BC34A',  # Light Green
        'MODERATE': '#FFC107',  # Amber
        'HIGH': '#FF9800',  # Orange
        'VERY HIGH': '#F44336'  # Red
    }


def get_risk_color(risk_value: float) -> str:
    """
    Get a color corresponding to a risk level

    Parameters:
    -----------
    risk_value : float
        Risk percentage (0-100)

    Returns:
    --------
    str
        Color code
    """
    if risk_value < 15:
        return '#4CAF50'  # Green
    elif risk_value < 30:
        return '#8BC34A'  # Light Green
    elif risk_value < 50:
        return '#FFC107'  # Amber
    elif risk_value < 75:
        return '#FF9800'  # Orange
    else:
        return '#F44336'  # Red


def generate_custom_palette(n_colors: int, start_color: str = 'blue', end_color: str = 'red') -> List[str]:
    """
    Generate a custom color palette with n colors

    Parameters:
    -----------
    n_colors : int
        Number of colors to generate
    start_color : str, optional
        Starting color (default: 'blue')
    end_color : str, optional
        Ending color (default: 'red')

    Returns:
    --------
    List[str]
        List of color codes
    """
    return list(sns.color_palette(f"{start_color}_{end_color}", n_colors).as_hex())


def apply_style_to_figure(fig: plt.Figure, style: str = 'whitegrid') -> None:
    """
    Apply a style to a matplotlib figure

    Parameters:
    -----------
    fig : plt.Figure
        Figure to apply style to
    style : str, optional
        Name of the style to apply (default: 'whitegrid')
    """
    with plt.style.context(style):
        # Get current axes
        axes = fig.get_axes()

        # Apply style to each axis
        for ax in axes:
            # Set grid style
            ax.grid(True, linestyle='--', alpha=0.7)

            # Set spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Set background color
            ax.set_facecolor('#f9f9f9')


# ==============================
# Figure and Axis Utilities
# ==============================

def create_figure_with_size_adjustment(
        n_items: int,
        base_width: float = 10.0,
        base_height: float = 6.0,
        min_item_width: float = 0.5,
        max_width: float = 20.0
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with adjusted size based on number of items

    Parameters:
    -----------
    n_items : int
        Number of items to display
    base_width : float, optional
        Base figure width (default: 10.0)
    base_height : float, optional
        Base figure height (default: 6.0)
    min_item_width : float, optional
        Minimum width per item (default: 0.5)
    max_width : float, optional
        Maximum figure width (default: 20.0)

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    # Calculate width needed
    width = min(base_width + n_items * min_item_width, max_width)

    # Create figure
    fig, ax = plt.subplots(figsize=(width, base_height))

    return fig, ax


def format_percentage_axis(ax: plt.Axes, axis: str = 'y') -> None:
    """
    Format an axis to display percentages

    Parameters:
    -----------
    ax : plt.Axes
        Axes to format
    axis : str, optional
        Which axis to format ('x' or 'y') (default: 'y')
    """
    if axis == 'y':
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    else:
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))


def add_data_labels(
        ax: plt.Axes,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
        fmt: str = '{:.1f}',
        **kwargs
) -> None:
    """
    Add data labels to a plot

    Parameters:
    -----------
    ax : plt.Axes
        Axes to add labels to
    x : List or ndarray
        X-coordinates
    y : List or ndarray
        Y-coordinates
    fmt : str, optional
        Format string for labels (default: '{:.1f}')
    **kwargs : dict
        Additional parameters for text
    """
    for x_val, y_val in zip(x, y):
        ax.text(
            x_val, y_val,
            fmt.format(y_val),
            ha='center',
            va='bottom',
            **kwargs
        )


# ==============================
# File Path Utilities
# ==============================

def generate_unique_filename(
        base_name: str,
        suffix: str = '',
        file_format: str = 'png'
) -> str:
    """
    Generate a unique filename with timestamp

    Parameters:
    -----------
    base_name : str
        Base filename
    suffix : str, optional
        Suffix to add to filename
    file_format : str, optional
        File format extension (default: 'png')

    Returns:
    --------
    str
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if suffix:
        return f"{base_name}_{suffix}_{timestamp}.{file_format}"
    else:
        return f"{base_name}_{timestamp}.{file_format}"


def process_save_path(
        save_path: Optional[Union[str, Path]],
        default_filename: str
) -> Optional[Path]:
    """
    Process a save path, ensuring it's a valid file path

    Parameters:
    -----------
    save_path : str or Path, optional
        Path to save file to (directory or file)
    default_filename : str
        Default filename to use if save_path is a directory

    Returns:
    --------
    Path or None
        Processed save path or None if no save_path provided
    """
    if not save_path:
        return None

    save_path = Path(save_path)

    # If save_path is a directory, append the default filename
    if save_path.is_dir():
        return save_path / default_filename

    # Ensure parent directory exists
    os.makedirs(save_path.parent, exist_ok=True)

    return save_path


# ==============================
# Data Processing Utilities
# ==============================

def calculate_l_values(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        diversity_type: str = 'distinct'
) -> List[float]:
    """
    Calculate l-values for each group in the dataset

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attributes : List[str]
        Sensitive attribute columns
    diversity_type : str, optional
        Type of diversity to calculate (default: 'distinct')

    Returns:
    --------
    List[float]
        L-values for each group
    """
    # Group data by quasi-identifiers
    grouped = data.groupby(quasi_identifiers)
    l_values = []

    # Process each group
    for _, group_data in grouped:
        for sa in sensitive_attributes:
            # Skip if attribute not in dataset
            if sa not in group_data.columns:
                continue

            # Get unique values
            sa_values = group_data[sa].values

            if diversity_type == 'entropy':
                # Calculate entropy
                unique_values, counts = np.unique(sa_values, return_counts=True)
                probabilities = counts / len(sa_values)
                # Add small epsilon to avoid log(0)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                # Convert entropy to effective number of classes
                effective_l = np.exp(entropy) if entropy > 0 else 1
                l_values.append(effective_l)
            else:
                # Calculate distinct count
                distinct_values = len(np.unique(sa_values))
                l_values.append(distinct_values)

    return l_values


def extract_risk_metrics_from_processor(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        processor: Any,
        **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Extract risk metrics from a processor if available

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns
    processor : Any
        L-Diversity processor instance
    **kwargs : dict
        Additional parameters for risk calculation

    Returns:
    --------
    Dict[str, Any] or None
        Risk metrics if available, None otherwise
    """
    if not processor:
        return None

    # Try using evaluate_privacy method
    if hasattr(processor, 'evaluate_privacy'):
        try:
            return processor.evaluate_privacy(
                data, quasi_identifiers, sensitive_attributes, **kwargs
            )
        except Exception as e:
            logger.warning(f"Error using processor's evaluate_privacy: {e}")

    # Try using risk_assessor attribute
    if hasattr(processor, 'risk_assessor'):
        try:
            risk_assessor = processor.risk_assessor
            return risk_assessor.assess_privacy_risks(
                data, quasi_identifiers, sensitive_attributes, **kwargs
            )
        except Exception as e:
            logger.warning(f"Error using processor's risk_assessor: {e}")

    # If both methods fail, return None
    return None


# ==============================
# Composite Visualization Utilities
# ==============================

def create_multi_attribute_visualization(
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str],
        visualization_function: Callable,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Dict[str, Tuple[plt.Figure, Optional[str]]]:
    """
    Apply a visualization function to multiple attributes

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
    sensitive_attributes : List[str]
        Sensitive attribute columns
    visualization_function : Callable
        Function to apply to each attribute
    save_path : str or Path, optional
        Directory to save visualizations
    **kwargs : dict
        Additional parameters for visualization function

    Returns:
    --------
    Dict[str, Tuple[plt.Figure, Optional[str]]]
        Dictionary mapping attribute names to (figure, saved_path) tuples
    """
    results = {}

    for sa in sensitive_attributes:
        # Process save path
        attr_save_path = None
        if save_path:
            save_dir = Path(save_path)
            if save_dir.is_dir():
                attr_save_path = save_dir

        # Call visualization function
        fig, saved_path = visualization_function(
            data, quasi_identifiers, sa,
            save_path=attr_save_path,
            **kwargs
        )

        results[sa] = (fig, saved_path)

    return results


def create_dashboard(
        title: str,
        figures: List[Tuple[plt.Figure, str]],
        layout: Tuple[int, int] = None,
        figsize: Tuple[float, float] = (16, 10),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Create a dashboard with multiple figures

    Parameters:
    -----------
    title : str
        Dashboard title
    figures : List[Tuple[plt.Figure, str]]
        List of (figure, subtitle) tuples
    layout : Tuple[int, int], optional
        Grid layout (rows, cols) (auto-calculated if None)
    figsize : Tuple[float, float], optional
        Figure size (width, height) (default: (16, 10))
    save_path : str or Path, optional
        Path to save dashboard
    **kwargs : dict
        Additional parameters for save_plot

    Returns:
    --------
    Tuple[plt.Figure, Optional[str]]
        Dashboard figure and optional saved path
    """
    # Determine layout if not provided
    if not layout:
        n_figures = len(figures)
        cols = min(3, n_figures)
        rows = (n_figures + cols - 1) // cols  # Ceiling division
        layout = (rows, cols)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Add subfigures
    for i, (src_fig, subtitle) in enumerate(figures):
        if i >= layout[0] * layout[1]:
            logger.warning(f"Dashboard layout too small for {len(figures)} figures")
            break

        # Create subplot
        ax = fig.add_subplot(layout[0], layout[1], i + 1)

        # Copy content from source figure
        # This is complex and might not work perfectly for all plot types
        src_ax = src_fig.get_axes()[0]

        # Try to copy the main parts of the plot
        for collection in src_ax.collections:
            ax.add_collection(collection.copy())

        for line in src_ax.lines:
            ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())

        for patch in src_ax.patches:
            ax.add_patch(patch.copy())

        # Copy labels
        ax.set_xlabel(src_ax.get_xlabel())
        ax.set_ylabel(src_ax.get_ylabel())
        ax.set_title(subtitle)

    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save figure if path provided
    saved_path = None
    if save_path:
        # Process save path
        save_format = kwargs.get('save_format', 'png')
        dpi = kwargs.get('dpi', 300)

        save_path = process_save_path(
            save_path,
            generate_unique_filename('dashboard', file_format=save_format)
        )

        if save_path:
            # Use centralized save_plot utility
            saved_path = save_plot(
                fig,
                save_path,
                save_format=save_format,
                dpi=dpi,
                bbox_inches="tight"
            )

            if saved_path:
                logger.info(f"Dashboard saved to {saved_path}")
            else:
                logger.warning("Failed to save dashboard")

    return fig, saved_path