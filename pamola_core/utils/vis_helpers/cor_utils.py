"""
Utility functions for correlation visualizations.

This module provides utility functions for processing and transforming data
for correlation visualizations in the system.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def prepare_correlation_data(data: Union[pd.DataFrame, np.ndarray, Tuple]) -> Tuple[
    np.ndarray, List[str], List[str], Optional[Union[pd.DataFrame, np.ndarray]]]:
    """
    Prepare data for correlation matrix visualization.

    Parameters:
    -----------
    data : pd.DataFrame, np.ndarray, or Tuple
        Correlation matrix data or tuple of (correlations, methods)

    Returns:
    --------
    Tuple[np.ndarray, List[str], List[str], Optional[Union[pd.DataFrame, np.ndarray]]]
        Tuple containing (matrix, x_labels, y_labels, methods)

    Raises:
    -------
    TypeError
        If data type is not supported
    """
    # Check for extended correlation information (method data)
    methods = None

    # Handle tuple input (correlation data and methods)
    if isinstance(data, tuple) and len(data) == 2:
        correlation_data, methods = data
    else:
        correlation_data, methods = data, None

    # Process correlation data based on type
    if isinstance(correlation_data, pd.DataFrame):
        matrix = correlation_data.values
        x_labels = list(correlation_data.columns)
        y_labels = list(correlation_data.index)
    elif isinstance(correlation_data, np.ndarray):
        matrix = correlation_data
        x_labels = [str(i) for i in range(matrix.shape[1])]
        y_labels = [str(i) for i in range(matrix.shape[0])]
    else:
        raise TypeError(f"Unsupported data type for correlation matrix: {type(data)}")

    return matrix, x_labels, y_labels, methods


def create_correlation_mask(matrix: np.ndarray,
                            mask_upper: bool = False,
                            mask_diagonal: bool = False) -> np.ndarray:
    """
    Create a mask for correlation matrix visualization.

    Parameters:
    -----------
    matrix : np.ndarray
        Correlation matrix
    mask_upper : bool
        Whether to mask the upper triangle (above diagonal)
    mask_diagonal : bool
        Whether to mask the diagonal

    Returns:
    --------
    np.ndarray
        Boolean mask where True indicates values to keep
    """
    mask = np.ones_like(matrix, dtype=bool)

    if mask_upper and mask_diagonal:
        # Mask both upper triangle and diagonal
        mask = np.triu(mask, k=1)
    elif mask_upper:
        # Mask only upper triangle
        mask = np.triu(mask, k=0)
    elif mask_diagonal:
        # Mask only diagonal
        mask = ~np.eye(matrix.shape[0], dtype=bool)

    return mask


def apply_mask(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to a correlation matrix.

    Parameters:
    -----------
    matrix : np.ndarray
        Correlation matrix
    mask : np.ndarray
        Boolean mask where True indicates values to keep

    Returns:
    --------
    np.ndarray
        Masked correlation matrix with NaN in masked positions
    """
    return np.where(mask, matrix, np.nan)


def get_text_color(value: float) -> str:
    """
    Determine the appropriate text color for a correlation value.

    Parameters:
    -----------
    value : float
        Correlation value

    Returns:
    --------
    str
        Color code for the text
    """
    if np.isnan(value):
        return 'rgba(0,0,0,0)'  # Transparent for NaN values
    elif abs(value) > 0.5:  # Strong correlation
        return 'white'
    else:  # Weak correlation
        return 'black'


def create_text_colors_array(matrix: np.ndarray) -> np.ndarray:
    """
    Create an array of text colors based on correlation values.

    This method determines appropriate text colors for each cell in the correlation
    matrix to ensure optimal readability against the background color.

    Parameters:
    -----------
    matrix : np.ndarray
        Correlation matrix with float values between -1 and 1

    Returns:
    --------
    np.ndarray
        Array of text colors matching the dimensions of the input matrix
    """
    # Create an empty array with object dtype to store color strings
    text_colors = np.empty_like(matrix, dtype=object)

    # Iterate through each cell and determine appropriate text color
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Extract the cell value and convert to Python float
            value = float(matrix[i, j]) if not np.isnan(matrix[i, j]) else np.nan
            text_colors[i, j] = get_text_color(value)

    return text_colors


def create_significance_mask(matrix: np.ndarray,
                             masked_matrix: np.ndarray,
                             threshold: float) -> np.ndarray:
    """
    Create a mask highlighting significant correlations.

    Parameters:
    -----------
    matrix : np.ndarray
        Original correlation matrix
    masked_matrix : np.ndarray
        Masked correlation matrix
    threshold : float
        Threshold value to consider correlations significant

    Returns:
    --------
    np.ndarray
        Boolean mask where True indicates significant correlations
    """
    return np.logical_and(
        np.abs(matrix) >= threshold,
        ~np.isnan(masked_matrix)
    )


def prepare_hover_texts(matrix: np.ndarray,
                        x_labels: List[str],
                        y_labels: List[str],
                        methods: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                        method_labels: Optional[Dict[str, str]] = None) -> List[List[str]]:
    """
    Prepare hover texts for correlation matrix.

    Parameters:
    -----------
    matrix : np.ndarray
        Correlation matrix
    x_labels : List[str]
        Labels for x-axis
    y_labels : List[str]
        Labels for y-axis
    methods : pd.DataFrame or np.ndarray, optional
        Matrix of correlation methods
    method_labels : Dict[str, str], optional
        Dictionary mapping method codes to display labels

    Returns:
    --------
    List[List[str]]
        2D array of hover text strings
    """
    hover_texts = []
    for i in range(matrix.shape[0]):
        row_texts = []
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                row_texts.append("")
                continue

            text = f"x: {x_labels[j]}<br>y: {y_labels[i]}<br>correlation: {val:.4f}"

            if methods is not None and method_labels is not None:
                method_code = methods.iloc[i, j] if hasattr(methods, 'iloc') else methods[i, j]
                method_name = method_labels.get(method_code, method_code)
                text += f"<br>method: {method_name}"

            row_texts.append(text)
        hover_texts.append(row_texts)

    return hover_texts


def parse_annotation_format(annotation_format: str) -> str:
    """
    Parse annotation format string and convert to Plotly format.

    Parameters:
    -----------
    annotation_format : str
        Format string for annotations (e.g., ".2f")

    Returns:
    --------
    str
        Plotly format string for annotations
    """
    if annotation_format.startswith('.') and annotation_format.endswith('f'):
        # Parse decimal format like ".2f" to get number of decimal places
        decimal_places = int(annotation_format[1:-1])
        return f'%{{z:.{decimal_places}f}}'
    else:
        # Use provided format string
        return f'%{{z:{annotation_format}}}'


def calculate_symmetric_colorscale_range(matrix: np.ndarray) -> Tuple[float, float]:
    """
    Calculate symmetric range for colorscale centered at zero.

    Parameters:
    -----------
    matrix : np.ndarray
        Correlation matrix

    Returns:
    --------
    Tuple[float, float]
        (vmin, vmax) for colorscale
    """
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    abs_max = max(abs(vmin), abs(vmax))
    return -abs_max, abs_max


def calculate_correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Calculate correlation coefficient between two arrays.

    Parameters:
    -----------
    x : np.ndarray
        First array of values
    y : np.ndarray
        Second array of values

    Returns:
    --------
    Optional[float]
        Correlation coefficient or None if calculation fails
    """
    try:
        if len(x) <= 1 or len(y) <= 1:
            return None

        corr_matrix = np.corrcoef(x, y)
        if corr_matrix.size >= 4:
            return float(corr_matrix[0, 1])
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not calculate correlation: {e}")
        return None