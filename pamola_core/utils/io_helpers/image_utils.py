"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Image and Plot Utilities
Description: Helpers for managing, saving, and formatting visualizations and image outputs
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Unified interface for saving matplotlib figures and other plots using the pamola core I/O system
- Automatic format detection from file extensions
- Utilities for calculating optimal figure sizes for grid-based visualizations
- Preparation of customizable figure creation options (dpi, size, etc.)

"""


from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger("pamola_core.utils.io_helpers.image_utils")


def save_plot(plot_fig, file_path: Union[str, Path], dpi: int = 300, **kwargs) -> Path:
    """
    Saves a matplotlib figure to a file.

    Parameters:
    -----------
    plot_fig : matplotlib.figure.Figure
        The figure to save
    file_path : str or Path
        Path to save the image
    dpi : int
        Dots per inch for raster formats (default: 300)
    **kwargs
        Additional arguments to pass to fig.savefig

    Returns:
    --------
    Path
        Path to the saved file
    """
    # Use the IO module's save_visualization function
    from pamola_core.utils.io import save_visualization
    return save_visualization(plot_fig, file_path, dpi=dpi, **kwargs)


def get_figure_format(file_path: Union[str, Path]) -> str:
    """
    Determines figure format from file extension.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    str
        Figure format (e.g., 'png', 'pdf', 'svg')
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower().lstrip('.')

    # Map of file extensions to matplotlib formats
    format_map = {
        'png': 'png',
        'jpg': 'jpg',
        'jpeg': 'jpg',
        'pdf': 'pdf',
        'svg': 'svg',
        'eps': 'eps',
        'tif': 'tiff',
        'tiff': 'tiff'
    }

    if extension in format_map:
        return format_map[extension]
    else:
        logger.warning(f"Unknown figure format for extension: {extension}. Using 'png'")
        return 'png'


def prepare_figure_options(width: Optional[float] = None,
                           height: Optional[float] = None,
                           dpi: int = 300,
                           **kwargs) -> Dict[str, Any]:
    """
    Prepares options for figure creation.

    Parameters:
    -----------
    width : float, optional
        Figure width in inches
    height : float, optional
        Figure height in inches
    dpi : int
        Dots per inch (default: 300)
    **kwargs
        Additional figure options

    Returns:
    --------
    Dict[str, Any]
        Dictionary with figure options
    """
    options = {
        'dpi': dpi
    }

    # Set figsize if both width and height are provided
    if width is not None and height is not None:
        options['figsize'] = (width, height)

    # Add all other kwargs
    for key, value in kwargs.items():
        if key not in options:
            options[key] = value

    return options


def get_optimal_figure_size(num_rows: int,
                            num_cols: int,
                            base_size: Tuple[float, float] = (6.0, 4.0)) -> Tuple[float, float]:
    """
    Calculates optimal figure size based on grid dimensions.

    Parameters:
    -----------
    num_rows : int
        Number of rows in the plot grid
    num_cols : int
        Number of columns in the plot grid
    base_size : Tuple[float, float]
        Base figure size (width, height) for a single plot

    Returns:
    --------
    Tuple[float, float]
        Optimal figure size (width, height)
    """
    base_width, base_height = base_size

    # Scale base size by number of rows and columns
    width = base_width * num_cols
    height = base_height * num_rows

    # Add some padding
    width += 1.0
    height += 1.0

    return (width, height)