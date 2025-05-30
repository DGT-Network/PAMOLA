"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Visualization System
Description: Thread-safe visualization capabilities for data analysis and privacy metrics
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides a unified interface for creating various types of visualizations
primarily using Plotly. It coordinates with other modules in the vis_helpers package
to implement specific visualization types.

All functions accept data, create visualizations, save them as PNG files,
and return the path to the saved file or an error message.

The implementation uses contextvars to ensure that visualization state (themes, backends)
is properly isolated between concurrent execution contexts, eliminating state interference
when multiple visualization operations run in parallel.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from pamola_core.utils.vis_helpers.base import FigureFactory
from pamola_core.utils.vis_helpers.context import visualization_context, register_figure

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================


def _save_figure(
    fig: Union["plotly.graph_objects.Figure", "matplotlib.figure.Figure"],
    output_path: Union[str, Path], **kwargs
) -> str:
    """
    Saves a figure using the IO system.

    Parameters:
    -----------
    fig : Union[plotly.graph_objects.Figure, matplotlib.figure.Figure]
        Figure to save
    output_path : Union[str, Path]
        Path where the file should be saved

    Returns:
    --------
    str
        Path to the saved file or error message
    """
    try:
        # Convert to Path object
        output_path = Path(output_path)

        # Ensure the directory exists using IO module
        from pamola_core.utils.io import ensure_directory

        ensure_directory(output_path.parent)

        # Use the IO module's save_visualization function
        from pamola_core.utils.io import save_visualization

        use_encryption = kwargs.get('use_encryption', False)
        encryption_key= kwargs.get('encryption_key', None) if use_encryption else None
        saved_path = save_visualization(fig, output_path, encryption_key=encryption_key)

        # Close matplotlib figure if it's a matplotlib figure
        # This helps prevent memory leaks
        try:
            if hasattr(fig, "clf") and callable(fig.clf):
                fig.clf()
            if "matplotlib.figure" in str(type(fig)):
                import matplotlib.pyplot as plt

                plt.close(fig)
        except Exception as e:
            logger.debug(f"Non-critical error during figure cleanup: {e}")

        return str(saved_path)
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        return f"Error saving figure: {e}"


# ============================================================================
# Basic visualization functions
# ============================================================================


def create_bar_plot(
    data: Union[Dict[str, Any], pd.Series],
    output_path: Union[str, Path],
    title: str,
    orientation: str = "v",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    sort_by: str = "value",
    max_items: int = 15,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a bar plot visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, Any] or pd.Series
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    orientation : str
        Orientation of the bars: "v" for vertical, "h" for horizontal
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    sort_by : str
        How to sort the data: "value" (descending) or "key" (alphabetical)
    max_items : int
        Maximum number of items to show
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("bar")

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                orientation=orientation,
                x_label=x_label,
                y_label=y_label,
                sort_by=sort_by,
                max_items=max_items
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating bar plot: {e}")
            return f"Error creating bar plot: {e}"


def create_histogram(
    data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = "Count",
    bins: int = 20,
    kde: bool = True,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a histogram visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, int], pd.Series, np.ndarray, or List[float]
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    bins : int
        Number of bins for the histogram
    kde : bool
        Whether to show a kernel density estimate
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("histogram")

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                bins=bins,
                kde=kde,
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            return f"Error creating histogram: {e}"


def create_scatter_plot(
    x_data: Union[List[float], np.ndarray, pd.Series],
    y_data: Union[List[float], np.ndarray, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    add_trendline: bool = False,
    correlation: Optional[float] = None,
    method: Optional[str] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a scatter plot visualization and save it as PNG.

    Parameters:
    -----------
    x_data : List[float], np.ndarray, or pd.Series
        Data for the x-axis
    y_data : List[float], np.ndarray, or pd.Series
        Data for the y-axis
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    add_trendline : bool
        Whether to add a trendline to the plot
    correlation : float, optional
        Correlation coefficient to display on the plot
    method : str, optional
        Correlation method used (e.g., "pearson", "spearman")
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("scatter", backend=backend)

        try:
            # Create the figure
            fig = fig_creator.create(
                x_data=x_data,
                y_data=y_data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                add_trendline=add_trendline,
                correlation=correlation,
                method=method,
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return f"Error creating scatter plot: {e}"


def create_boxplot(
    data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = "Category",
    y_label: Optional[str] = "Value",
    orientation: str = "v",
    show_points: bool = True,
    notched: bool = False,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a box plot visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, List[float]], pd.DataFrame, or pd.Series
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis (categorical axis for vertical orientation)
    y_label : str, optional
        Label for the y-axis (value axis for vertical orientation)
    orientation : str
        Orientation of the boxes: "v" for vertical, "h" for horizontal
    show_points : bool
        Whether to show outlier points
    notched : bool
        Whether to show notched boxes
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("boxplot", backend=backend)

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                orientation=orientation,
                show_points=show_points,
                notched=notched,
                **kwargs,
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating boxplot: {e}")
            return f"Error creating boxplot: {e}"


def create_heatmap(
    data: Union[Dict[str, Dict[str, float]], pd.DataFrame, np.ndarray],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colorscale: Optional[str] = None,
    annotate: bool = True,
    annotation_format: str = ".2f",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    textfont_color: str = '#000000',
    **kwargs,
) -> str:
    """
    Create a heatmap visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, Dict[str, float]], pd.DataFrame, or np.ndarray
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    colorscale : str, optional
        Colorscale to use (default from theme if None)
    annotate : bool
        Whether to annotate the heatmap with values
    annotation_format : str
        Format string for annotations (e.g., ".2f" for 2 decimal places)
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("heatmap", backend=backend)

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                colorscale=colorscale,
                annotate=annotate,
                annotation_format=annotation_format,
                textfont={'color': textfont_color},
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return f"Error creating heatmap: {e}"


def create_line_plot(
    data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    add_markers: bool = True,
    add_area: bool = False,
    smooth: bool = False,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a line plot visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, List[float]], pd.DataFrame, or pd.Series
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_data : List, np.ndarray, or pd.Series, optional
        Data for the x-axis. If None, indices are used
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    add_markers : bool
        Whether to add markers at data points
    add_area : bool
        Whether to fill area under lines
    smooth : bool
        Whether to use spline interpolation for smoother lines
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("line", backend=backend)

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                x_data=x_data,
                x_label=x_label,
                y_label=y_label,
                add_markers=add_markers,
                add_area=add_area,
                smooth=smooth,
                **kwargs,
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating line plot: {e}")
            return f"Error creating line plot: {e}"


def create_correlation_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colorscale: Optional[str] = None,
    annotate: bool = True,
    annotation_format: str = ".2f",
    mask_upper: bool = False,
    mask_diagonal: bool = False,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a correlation matrix visualization and save it as PNG.

    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Correlation matrix data
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    colorscale : str, optional
        Colorscale to use (default from theme if None)
    annotate : bool
        Whether to annotate the matrix with correlation values
    annotation_format : str
        Format string for annotations (e.g., ".2f" for 2 decimal places)
    mask_upper : bool
        Whether to mask the upper triangle (above diagonal)
    mask_diagonal : bool
        Whether to mask the diagonal
    theme : str, optional
        Theme name to use (optional)
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("correlation_matrix", backend=backend)

        try:
            fig = fig_creator.create(
                data=data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                colorscale=colorscale,
                annotate=annotate,
                annotation_format=annotation_format,
                mask_upper=mask_upper,
                mask_diagonal=mask_diagonal,
                **kwargs,
            )

            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return f"Error creating correlation matrix: {e}"


def create_correlation_pair(
    x_data: Union[List[float], np.ndarray, pd.Series],
    y_data: Union[List[float], np.ndarray, pd.Series],
    output_path: Union[str, Path],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    correlation: Optional[float] = None,
    method: Optional[str] = "Pearson",
    add_trendline: bool = True,
    add_histogram: bool = True,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a correlation plot for a pair of variables and save it as PNG.

    Parameters:
    -----------
    x_data : List[float], np.ndarray, or pd.Series
        Data for the x-axis
    y_data : List[float], np.ndarray, or pd.Series
        Data for the y-axis
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    correlation : float, optional
        Correlation coefficient to display on the plot
    method : str, optional
        Correlation method name (for annotation)
    add_trendline : bool
        Whether to add a trendline
    add_histogram : bool
        Whether to add histograms for both variables
    theme : str, optional
        Theme name to use (optional)
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("correlation_pair", backend=backend)

        try:
            fig = fig_creator.create(
                x_data=x_data,
                y_data=y_data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                correlation=correlation,
                method=method,
                add_trendline=add_trendline,
                add_histogram=add_histogram,
                **kwargs,
            )

            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating correlation pair plot: {e}")
            return f"Error creating correlation pair plot: {e}"


def create_venn_diagram(
    set1: Union[set, list, pd.Series],
    set2: Union[set, list, pd.Series],
    output_path: Union[str, Path],
    set1_label: str = "Set 1",
    set2_label: str = "Set 2",
    title: str = "Venn Diagram",
    figsize: tuple = (5, 5),
    annotation: Optional[Dict[str, str]] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a Venn diagram visualization and save it as PNG.

    Parameters:
    -----------
    set1 : set, list, or pd.Series
        First set of elements.
    set2 : set, list, or pd.Series
        Second set of elements.
    output_path : str or Path
        Path where the PNG file should be saved.
    set1_label : str
        Label for the first set.
    set2_label : str
        Label for the second set.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size (width, height).
    annotation : dict, optional
        Additional annotation text for the diagram.
    theme : str, optional
        Theme name to use for this visualization.
    backend : str, optional
        Visualization backend ("matplotlib" recommended for venn).
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function.

    Returns:
    --------
    str
        Path to the saved PNG file or error message.
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("venn_diagram", backend=backend)

        try:
            fig = fig_creator.create(
                set1=set1,
                set2=set2,
                set1_label=set1_label,
                set2_label=set2_label,
                title=title,
                figsize=figsize,
                annotation=annotation,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating venn diagram: {e}")
            return f"Error creating venn diagram: {e}"


def create_spider_chart(
    data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
    output_path: Union[str, Path],
    title: str,
    categories: Optional[List[str]] = None,
    normalize_values: bool = True,
    fill_area: bool = True,
    show_gridlines: bool = True,
    angle_start: float = 90,
    show_legend: bool = True,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a spider/radar chart and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, Dict[str, float]] or pd.DataFrame
        Data to visualize. If dict, outer keys are series names, inner keys are categories.
        If DataFrame, columns are categories, index values are series names.
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    categories : List[str], optional
        List of categories to include (if None, all categories in data will be used)
    normalize_values : bool, optional
        Whether to normalize values to 0-1 range for each category
    fill_area : bool, optional
        Whether to fill the area under the radar lines
    show_gridlines : bool, optional
        Whether to show gridlines on the radar
    angle_start : float, optional
        Starting angle for the first axis in degrees (90 = top)
    show_legend : bool, optional
        Whether to show the legend
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("spider", backend=backend or "plotly")

        try:
            fig = fig_creator.create(
                data=data,
                title=title,
                categories=categories,
                normalize_values=normalize_values,
                fill_area=fill_area,
                show_gridlines=show_gridlines,
                angle_start=angle_start,
                show_legend=show_legend,
                **kwargs,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating spider chart: {e}")
            return f"Error creating spider chart: {e}"


def create_pie_chart(
    data: Union[Dict[str, float], pd.Series, List[float]],
    output_path: Union[str, Path],
    title: str,
    labels: Optional[List[str]] = None,
    hole: float = 0,  # 0 for pie chart, >0 for donut
    show_values: bool = True,
    show_percentages: bool = True,
    sort_values: bool = False,
    pull_largest: bool = False,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a pie chart visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, float], pd.Series, or List[float]
        Data to visualize. If dict or Series, keys are used as labels.
        If list, separate labels should be provided.
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    labels : List[str], optional
        List of labels for pie slices (not needed if data is dict or Series)
    hole : float, optional
        Size of the hole for a donut chart (0-1, default 0 for a normal pie)
    show_values : bool, optional
        Whether to show values on pie slices
    show_percentages : bool, optional
        Whether to show percentages on pie slices
    sort_values : bool, optional
        Whether to sort slices by value (descending)
    pull_largest : bool, optional
        Whether to pull out the largest slice
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("pie", backend=backend or "plotly")

        try:
            fig = fig_creator.create(
                data=data,
                title=title,
                labels=labels,
                hole=hole,
                show_values=show_values,
                show_percentages=show_percentages,
                sort_values=sort_values,
                pull_largest=pull_largest,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return f"Error creating pie chart: {e}"


def create_sunburst_chart(
    data: Union[Dict, pd.DataFrame],
    output_path: Union[str, Path],
    title: str,
    path_column: Optional[str] = None,
    values_column: Optional[str] = None,
    color_column: Optional[str] = None,
    maxdepth: Optional[int] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a sunburst chart visualization for hierarchical data and save it as PNG.

    Parameters:
    -----------
    data : Dict or pd.DataFrame
        Data to visualize. If DataFrame, it needs columns for path, values, and optionally colors.
        If Dict, it should be hierarchical with nested dictionaries.
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    path_column : str, optional
        For DataFrame data, the column containing hierarchical path
    values_column : str, optional
        For DataFrame data, the column containing values
    color_column : str, optional
        For DataFrame data, the column to use for coloring
    maxdepth : int, optional
        Maximum depth to display
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("sunburst", backend=backend or "plotly")

        try:
            fig = fig_creator.create(
                data=data,
                title=title,
                path_column=path_column,
                values_column=values_column,
                color_column=color_column,
                maxdepth=maxdepth,
                **kwargs,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating sunburst chart: {e}")
            return f"Error creating sunburst chart: {e}"


def create_combined_chart(
    primary_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    secondary_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    output_path: Union[str, Path],
    title: str,
    primary_type: str = "bar",  # "bar", "line", "scatter", "area"
    secondary_type: str = "line",  # "line", "scatter", "area", "bar"
    x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
    x_label: Optional[str] = None,
    primary_y_label: Optional[str] = None,
    secondary_y_label: Optional[str] = None,
    primary_color: Optional[str] = None,
    secondary_color: Optional[str] = None,
    primary_on_right: bool = False,
    vertical_alignment: bool = True,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a combined chart with dual Y-axes and save it as PNG.

    Parameters:
    -----------
    primary_data : Dict[str, Any], pd.Series, or pd.DataFrame
        Data for the primary Y-axis
    secondary_data : Dict[str, Any], pd.Series, or pd.DataFrame
        Data for the secondary Y-axis
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    primary_type : str, optional
        Type of visualization for primary data: "bar", "line", "scatter", "area"
    secondary_type : str, optional
        Type of visualization for secondary data: "line", "scatter", "area", "bar"
    x_data : List, np.ndarray, or pd.Series, optional
        Data for the x-axis. If None, indices are used.
    x_label : str, optional
        Label for the x-axis
    primary_y_label : str, optional
        Label for the primary Y-axis
    secondary_y_label : str, optional
        Label for the secondary Y-axis
    primary_color : str, optional
        Color for the primary series
    secondary_color : str, optional
        Color for the secondary series
    primary_on_right : bool, optional
        Whether to display primary Y-axis on the right side
    vertical_alignment : bool, optional
        Whether to align zero values across both axes
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("combined", backend=backend or "plotly")

        try:
            fig = fig_creator.create(
                primary_data=primary_data,
                secondary_data=secondary_data,
                title=title,
                primary_type=primary_type,
                secondary_type=secondary_type,
                x_data=x_data,
                x_label=x_label,
                primary_y_label=primary_y_label,
                secondary_y_label=secondary_y_label,
                primary_color=primary_color,
                secondary_color=secondary_color,
                primary_on_right=primary_on_right,
                vertical_alignment=vertical_alignment,
                **kwargs,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating combined chart: {e}")
            return f"Error creating combined chart: {e}"


# ============================================================================
# Specialized visualization functions for profiling
# ============================================================================


def plot_completeness(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Completeness Analysis",
    min_percent: float = 0,
    max_fields: int = 50,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize data completeness for each column in a DataFrame and save as PNG.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    min_percent : float
        Minimum completeness percentage to include in visualization
    max_fields : int
        Maximum number of fields to show
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        # Calculate completeness
        completeness = df.count() / len(df) * 100

        # Filter by minimum percentage if specified
        if min_percent > 0:
            completeness = completeness[completeness >= min_percent]

        # Limit number of fields if there are too many
        if len(completeness) > max_fields:
            completeness = completeness.sort_values().tail(max_fields)

        # Sort for better visualization
        completeness = completeness.sort_values()

        # Create a horizontal bar chart
        return create_bar_plot(
            data=completeness,
            output_path=output_path,
            title=title,
            orientation="h",
            x_label="Completeness (%)",
            y_label="Field",
            theme=theme,
            backend=backend,
            strict=strict,
            text=completeness.round(1).astype(str) + "%",
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating completeness plot: {e}")
        return f"Error creating completeness plot: {e}"


def plot_value_distribution(
    data: Dict[str, int],
    output_path: Union[str, Path],
    title: str,
    max_items: int = 15,
    sort_by: str = "value",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of values and save as PNG.

    Parameters:
    -----------
    data : Dict[str, int]
        Dictionary with values and their counts
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    max_items : int
        Maximum number of items to show
    sort_by : str
        How to sort the data: "value" (descending) or "key" (alphabetical)
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        # Convert to Series for easier sorting and slicing
        value_series = pd.Series(data)

        # Sort by value or key
        if sort_by == "value":
            value_series = value_series.sort_values(ascending=False)
        elif sort_by == "key":
            value_series = value_series.sort_index()
        else:
            logger.warning(f"Unknown sort_by: {sort_by}, defaulting to value.")
            value_series = value_series.sort_values(ascending=False)

        # Limit number of items
        if len(value_series) > max_items:
            value_series = value_series.head(max_items)

        # Create a horizontal bar chart
        return create_bar_plot(
            data=value_series,
            output_path=output_path,
            title=title,
            orientation="h",
            x_label="Count",
            y_label="Value",
            sort_by=sort_by,
            max_items=max_items,
            theme=theme,
            backend=backend,
            strict=strict,
            text=value_series.astype(str),
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating value distribution plot: {e}")
        return f"Error creating value distribution plot: {e}"


def plot_numeric_distribution(
    data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]],
    output_path: Union[str, Path],
    title: str,
    bins: int = 20,
    kde: bool = True,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of numeric values and save as PNG.

    Parameters:
    -----------
    data : Dict[str, int], pd.Series, np.ndarray, or List[float]
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    bins : int
        Number of bins for the histogram
    kde : bool
        Whether to show a kernel density estimate
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        return create_histogram(
            data=data,
            output_path=output_path,
            title=title,
            bins=bins,
            kde=kde,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating numeric distribution plot: {e}")
        return f"Error creating numeric distribution plot: {e}"


def plot_date_distribution(
    date_stats: Dict[str, Any],
    output_path: Union[str, Path],
    title: str,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of dates by year and save as PNG.

    Parameters:
    -----------
    date_stats : Dict[str, Any]
        Dictionary with date statistics, including 'year_distribution'
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        if "year_distribution" not in date_stats:
            factory = FigureFactory()
            fig_creator = factory.create_figure("base", backend="plotly")
            fig = fig_creator.create_empty_figure(
                title=title, message="No valid dates found"
            )
            return _save_figure(fig, output_path, **kwargs)

        # Extract year distribution data
        year_distribution = date_stats["year_distribution"]

        # Convert to series for better plotting
        years_series = pd.Series(year_distribution)

        # Sort by year for chronological presentation
        years_series = years_series.sort_index()

        # Create a bar chart
        return create_bar_plot(
            data=years_series,
            output_path=output_path,
            title=title,
            orientation="v",
            x_label="Year",
            y_label="Count",
            sort_by="key",  # Make sure years are sorted chronologically
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating date distribution plot: {e}")
        return f"Error creating date distribution plot: {e}"


def plot_email_domains(
    domains: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Email Domain Distribution",
    max_domains: int = 15,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of email domains and save as PNG.

    Parameters:
    -----------
    domains : Dict[str, int]
        Dictionary with domain names and their counts
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    max_domains : int
        Maximum number of domains to show
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        return plot_value_distribution(
            data=domains,
            output_path=output_path,
            title=title,
            max_items=max_domains,
            sort_by="value",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating email domain distribution plot: {e}")
        return f"Error creating email domain distribution plot: {e}"


def plot_phone_distribution(
    phone_data: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Phone Code Distribution",
    max_items: int = 15,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of phone components and save as PNG.

    Parameters:
    -----------
    phone_data : Dict[str, int]
        Dictionary with codes and their counts
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    max_items : int
        Maximum number of items to show
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        return plot_value_distribution(
            data=phone_data,
            output_path=output_path,
            title=title,
            max_items=max_items,
            sort_by="value",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating phone distribution plot: {e}")
        return f"Error creating phone distribution plot: {e}"


def plot_text_length_distribution(
    length_data: Dict[str, int],
    output_path: Union[str, Path],
    title: str = "Text Length Distribution",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of text lengths and save as PNG.

    Parameters:
    -----------
    length_data : Dict[str, int]
        Dictionary with length ranges and their counts
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        # Sort keys correctly (they're typically ranges like '0-10', '11-20', etc.)
        def sort_key(x):
            if "-" in x:
                return int(x.split("-")[0])
            if "+" in x:
                return int(x.replace("+", ""))
            return int(x) if x.isdigit() else float("inf")

        # Convert to series and sort
        length_series = pd.Series(length_data)
        length_series.index = pd.Categorical(
            length_series.index,
            categories=sorted(length_series.index, key=sort_key),
            ordered=True,
        )
        length_series = length_series.sort_index()

        # Create a vertical bar chart for better display of categories
        return create_bar_plot(
            data=length_series.to_dict(),
            output_path=output_path,
            title=title,
            orientation="v",
            x_label="Text Length Range",
            y_label="Count",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating text length distribution plot: {e}")
        return f"Error creating text length distribution plot: {e}"


def plot_group_variation_distribution(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Group Variation Distribution",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Visualize the distribution of group variation values and save as PNG.

    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary with group variation results (should include 'variation_distribution')
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        The title for the plot
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        if "variation_distribution" not in results:
            factory = FigureFactory()
            fig_creator = factory.create_figure("base", backend="plotly")
            fig = fig_creator.create_empty_figure(
                title=title, message="No variation distribution data found"
            )
            return _save_figure(fig, output_path, **kwargs)

        # Extract variation distribution data
        variation_distribution = results["variation_distribution"]

        # Convert to series for better plotting
        variation_series = pd.Series(variation_distribution)

        # Create a bar chart
        return create_bar_plot(
            data=variation_series,
            output_path=output_path,
            title=title,
            orientation="v",
            x_label="Variation Range",
            y_label="Number of Groups",
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error creating group variation distribution plot: {e}")
        return f"Error creating group variation distribution plot: {e}"


def plot_multiple_fields(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    fields: List[str],
    plot_type: str = "bar",
    title: str = "Field Comparison",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a comparison plot for multiple fields and save as PNG.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    output_path : str or Path
        Path where the PNG file should be saved
    fields : List[str]
        List of fields to compare
    plot_type : str
        Type of plot: "bar", "line", or "boxplot"
    title : str
        The title for the plot
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" or "matplotlib" (overrides global setting)
    strict : bool
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs:
        Additional arguments to pass to the underlying plotting function

    Returns:
    --------
    str
        Path to the saved PNG file or error message
    """
    try:
        # Check that all fields exist in the DataFrame
        missing_fields = [field for field in fields if field not in df.columns]
        if missing_fields:
            logger.warning(f"Fields not found in DataFrame: {missing_fields}")
            fields = [field for field in fields if field in df.columns]

        if not fields:
            factory = FigureFactory()
            fig_creator = factory.create_figure("base", backend="plotly")
            fig = fig_creator.create_empty_figure(
                title=title, message="No valid fields found in DataFrame"
            )
            return _save_figure(fig, output_path, **kwargs)

        # Create the plot according to the requested type
        if plot_type == "bar":
            # Calculate means for bar chart
            data = df[fields].mean()
            return create_bar_plot(
                data=data,
                output_path=output_path,
                title=title,
                orientation="v",
                x_label="Field",
                y_label="Average Value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs,
            )
        elif plot_type == "line":
            # Line plots typically need an x-axis
            return create_line_plot(
                data=df[fields],
                output_path=output_path,
                title=title,
                x_label="Index",
                y_label="Value",
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs,
            )
        elif plot_type == "boxplot":
            return create_boxplot(
                data=df[fields],
                output_path=output_path,
                title=title,
                theme=theme,
                backend=backend,
                strict=strict,
                **kwargs,
            )
        else:
            logger.error(f"Unknown plot type: {plot_type}")
            factory = FigureFactory()
            fig_creator = factory.create_figure("base", backend="plotly")
            fig = fig_creator.create_empty_figure(
                title=title, message=f"Unknown plot type: {plot_type}"
            )
            return _save_figure(fig, output_path, **kwargs)
    except Exception as e:
        logger.error(f"Error creating multiple fields plot: {e}")
        return f"Error creating multiple fields plot: {e}"
