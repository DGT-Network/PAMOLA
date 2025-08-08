"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Visualization System
Package:       pamola_core.utils.visualization
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
    This module provides a unified, thread-safe interface for generating a wide variety
    of data visualizations for analytical and privacy-preserving workflows. It integrates
    with modular vis_helpers to support specialized chart types including bar plots,
    histograms, boxplots, line and scatter plots, heatmaps, pie/donut, spider (radar),
    combined (dual axis), correlation, and word cloud visualizations.

Key Features:
    - Unified API for all major visualization types (tabular, distribution, categorical, correlation, text)
    - Thread/context-safe execution for use in parallel workflows (via contextvars)
    - Dynamic backend selection (Plotly, Matplotlib, and wordcloud where appropriate)
    - Extensible registry/factory for specialized and future chart types
    - Flexible customization via parameters and keyword arguments (colormaps, labels, annotations, etc.)
    - Error handling with strict/warning modes for robust production use
    - Full integration with IO subsystem for secure and reliable file output
    - Modular architecture for easy extension and integration

Framework:
    This module follows the PAMOLA.CORE visualization framework, with standardized
    interfaces for figure construction, backend/context management, and output routines.
    It works in conjunction with the vis_helpers package, and supports composable
    and extensible visualization workflows across the PAMOLA analytics and privacy pipeline.

Changelog:
    2.0.0 - Unified API for all visualization types, new modular architecture
          - Added thread/context isolation for parallel execution
          - Enhanced backend selection and extensibility
          - Expanded documentation and error handling
          - Added new chart types: spider, combined, correlation, word cloud
    1.0.0 - Initial implementation with basic bar, histogram, and boxplot support

Dependencies:
    - pandas       - DataFrame operations
    - numpy        - Numeric operations, arrays
    - matplotlib   - Matplotlib backend and figure handling
    - plotly       - Plotly backend for interactive figures
    - wordcloud    - Word cloud generation (text analytics)
    - PIL          - Image processing for text-based visualizations
    - contextvars  - Thread/context management for isolation
    - logging      - Error and event reporting
    - typing       - Type hints and validation
    - pathlib      - Path operations
    - pamola_core.utils.io - PAMOLA IO utilities for file management

TODO:
    - Add more interactive features for Jupyter and dashboard environments
    - Extend Matplotlib backend support for all chart types
    - Add real-time and streaming visualization support
    - Expand auto-theming and accessibility features
    - Integrate with distributed and cloud execution pipelines
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib
import numpy as np
import pandas as pd
import plotly
from pamola_core.utils.vis_helpers.base import FigureFactory
from pamola_core.utils.vis_helpers.context import visualization_context, register_figure

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================


def _save_figure(
    fig: Union["plotly.graph_objects.Figure", "matplotlib.figure.Figure"],
    output_path: Union[str, Path],
    **kwargs,
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

        use_encryption = kwargs.get("use_encryption", False)
        encryption_key = kwargs.get("encryption_key", None) if use_encryption else None
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


def _filter_kwargs(**kwargs):
    """
    Filter out unsupported parameters from kwargs.

    Parameters:
    -----------
    **kwargs : dict
        Keyword arguments passed to the visualization functions.

    Returns:
    --------
    dict
        Filtered kwargs containing only supported parameters.
    """
    # List of unsupported keys to exclude
    unsupported_keys = [
        "encryption_mode",
        "encryption_key",
        "use_encryption",
        "timestamp",
    ]

    # Filter kwargs to exclude unsupported keys
    custom_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_keys}

    return custom_kwargs


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
    show_values: bool = True,
    text: Optional[Union[List[str], pd.Series]] = None,
    colorscale: Optional[str] = None,
    color: Optional[Any] = None,
    figsize: Optional[Any] = None,
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
    show_values : bool, optional
        Whether to display value labels on bars (default True)
    text : list of str or pd.Series, optional
        Custom text labels for bars (overrides automatic value labels)
    colorscale : str, optional
        Plotly colorscale name to use for bars (e.g., "Viridis", "Blues")
    color : Any, optional
        Custom color or color array for bars (for Matplotlib or Plotly)
    figsize : Any, optional
        Figure size for Matplotlib backend (tuple of (width, height), e.g., (12, 8))
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

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)
        
        try:
            # Convert pandas Series to dict to avoid ambiguous truth value errors
            plot_data = data.to_dict() if isinstance(data, pd.Series) else data

            # Create the figure
            fig = fig_creator.create(
                data=plot_data,
                title=title,
                orientation=orientation,
                x_label=x_label,
                y_label=y_label,
                sort_by=sort_by,
                max_items=max_items,
                show_values=show_values,
                text=text,
                colorscale=colorscale,
                color=color,
                figsize=figsize,
                **custom_viz_kwargs,
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
    cumulative: bool = False,
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
    cumulative : bool
        Whether to show cumulative distribution
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

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                bins=bins,
                kde=kde,
                cumulative=cumulative,
                **custom_viz_kwargs,
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
    hover_text: Optional[List[str]] = None,
    marker_size: Optional[Union[List[float], float]] = None,
    color_scale: Optional[Union[List[str], str]] = None,
    color_values: Optional[List[float]] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a scatter plot visualization and save it as PNG.

    Parameters:
    -----------
    x_data : list, ndarray, or Series
        Data for the x-axis
    y_data : list, ndarray, or Series
        Data for the y-axis
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    add_trendline : bool, optional
        Whether to add a linear regression trendline (default False)
    correlation : float, optional
        Correlation coefficient to display
    method : str, optional
        Method used for correlation (e.g., "Pearson", "Spearman")
    hover_text : list of str, optional
        Custom hover text for each point
    marker_size : float or list of float, optional
        Marker size(s), either single value or per-point
    color_scale : str or list of str, optional
        Color scale for markers
    color_values : list of float, optional
        Values used for marker color mapping
    theme : str, optional
        Theme to use for the visualization
    backend : str, optional
        Backend to use: "plotly" (only supported)
    strict : bool, optional
        Strict mode for error handling
    **kwargs:
        Additional customization parameters passed to plotly.graph_objects.Scatter

    Returns:
    --------
    str
        Path to saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("scatter", backend=backend)

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                hover_text=hover_text,
                marker_size=marker_size,
                color_scale=color_scale,
                color_values=color_values,
                **custom_viz_kwargs,
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
    points: str = "outliers",
    box_width: float = 0.5,
    color: Optional[Any] = None,
    opacity: float = 0.7,
    boxmean: bool = True,
    figsize: Optional[Any] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    height: Optional[int] = 600,
    width: Optional[int] = 800,
    **kwargs,
) -> str:
    """
    Create a box plot visualization and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, List[float]], pd.DataFrame, or pd.Series
        Data to visualize. Each key/column represents a category.
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_label : str, optional
        Label for the x-axis (category axis for vertical orientation)
    y_label : str, optional
        Label for the y-axis (value axis for vertical orientation)
    orientation : str, optional
        Orientation of the boxes: "v" (vertical, default) or "h" (horizontal)
    show_points : bool, optional
        Whether to show outlier points (default True)
    notched : bool, optional
        Whether to show notched boxes (default False)
    points : str, optional
        How to show points: "outliers", "suspected", "all", or False
    box_width : float, optional
        Width of the boxes as a fraction of available space (default 0.5)
    color : Any, optional
        Color or palette for boxes (global or per column)
    opacity : float, optional
        Opacity for boxes (default 0.7)
    boxmean : bool, optional
        Whether to show the mean in boxes (default True)
    figsize : tuple or None, optional
        Figure size for Matplotlib
    height : int, optional
        Plot height (for Plotly)
    width : int, optional
        Plot width (for Plotly)
    theme : str, optional
        Theme name to use
    backend : str, optional
        Backend to use: "plotly" or "matplotlib"
    strict : bool, optional
        Strict mode for error handling
    **kwargs : additional parameters passed to plotting backend

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

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                points=points,
                box_width=box_width,
                color=color,
                opacity=opacity,
                boxmean=boxmean,
                figsize=figsize,
                height=height,
                width=width,
                **custom_viz_kwargs,
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
    cmap: Optional[str] = None,
    annotate: bool = True,
    annotation_format: str = ".2f",
    annotation_color_threshold: Optional[float] = 0.5,
    mask_values: Optional[np.ndarray] = None,
    colorbar_title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    figsize: Optional[Any] = (12, 10),
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
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
        Plotly colorscale to use (Plotly backend)
    cmap : str, optional
        Matplotlib colormap to use (Matplotlib backend)
    annotate : bool, optional
        Show values on the heatmap
    annotation_format : str, optional
        Value format for annotations (default ".2f")
    annotation_color_threshold : float, optional
        Threshold (0-1) for switching text color (white/black)
    mask_values : np.ndarray, optional
        Boolean mask to hide some cells (True=show, False=mask)
    colorbar_title : str, optional
        Title for colorbar (Plotly backend)
    colorbar_label : str, optional
        Label for colorbar (Matplotlib backend)
    figsize : tuple or None, optional
        Figure size for Matplotlib
    theme : str, optional
        Theme to use
    backend : str, optional
        Backend: "plotly" or "matplotlib"
    strict : bool, optional
        Strict mode for error handling
    **kwargs: additional arguments for plotly.graph_objects.Heatmap or ax.imshow

    Returns:
    --------
    str
        Path to saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("heatmap", backend=backend)

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

        try:
            # Create the figure
            fig = fig_creator.create(
                data=data,
                title=title,
                x_label=x_label,
                y_label=y_label,
                colorscale=colorscale,
                cmap=cmap,
                annotate=annotate,
                annotation_format=annotation_format,
                annotation_color_threshold=annotation_color_threshold,
                mask_values=mask_values,
                colorbar_title=colorbar_title,
                colorbar_label=colorbar_label,
                figsize=figsize,
                **custom_viz_kwargs,
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
        highlight_regions: Optional[List[Dict[str, Any]]] = None,
        line_width: float = 2.0,
        color: Optional[Any] = None,
        figsize: Optional[Any] = None,
        theme: Optional[str] = None,
        backend: Optional[str] = None,
        strict: bool = False,
        multi_x_data: bool = False,
        line_average: bool = False,
        **kwargs
) -> str:
    """
    Create a line plot visualization and save it as PNG.

    Parameters:
    -----------
    data : dict, DataFrame, or Series
        Data to visualize. Keys/columns are series, values are y-values.
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    x_data : list, ndarray, or Series, optional
        X-axis data (default: index)
    x_label : str, optional
        X-axis label
    y_label : str, optional
        Y-axis label
    add_markers : bool, optional
        Add markers at points (default True)
    add_area : bool, optional
        Fill area under line (default False)
    smooth : bool, optional
        Smooth (spline) lines (default False)
    highlight_regions : list of dict, optional
        Regions to highlight: each dict has 'start', 'end', 'color', 'label'
    line_width : float, optional
        Line width (default 2.0)
    color : any, optional
        Color for all series or per-series (see docs)
    figsize : any, optional
        (Reserved for future Matplotlib implementation)
    theme : str, optional
        Theme name
    backend : str, optional
        Backend name ("plotly" only for now)
    strict : bool, optional
        Strict mode for error handling
    **kwargs:
        Additional parameters for plotly.graph_objects.Scatter

    Returns:
    --------
    str
        Path to saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        # Get the appropriate figure creator
        factory = FigureFactory()
        fig_creator = factory.create_figure("line", backend=backend)

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                highlight_regions=highlight_regions,
                line_width=line_width,
                color=color,
                figsize=figsize,
                multi_x_data=multi_x_data,
                line_average=line_average,
                **custom_viz_kwargs,
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
    colorbar_title: Optional[str] = "Correlation",
    significant_threshold: Optional[float] = None,
    method_labels: Optional[Dict[str, str]] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a correlation matrix visualization and save it as PNG.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Correlation matrix data.
    output_path : str or Path
        Path where the PNG file should be saved.
    title : str
        Title for the plot.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    colorscale : str, optional
        Plotly colorscale (default: RdBu_r).
    annotate : bool, optional
        Show values on matrix (default True).
    annotation_format : str, optional
        Format for value annotation (default ".2f").
    mask_upper : bool, optional
        Mask upper triangle (default False).
    mask_diagonal : bool, optional
        Mask diagonal (default False).
    colorbar_title : str, optional
        Title for colorbar (default "Correlation").
    significant_threshold : float, optional
        Threshold to highlight significant correlations (draws rectangle).
    method_labels : dict, optional
        Mapping of method codes to labels (for mixed methods).
    theme : str, optional
        Visualization theme.
    backend : str, optional
        Backend ("plotly" only).
    strict : bool, optional
        Strict mode.
    **kwargs :
        Other Plotly go.Heatmap parameters.

    Returns
    -------
    str
        Path to saved PNG file or error message.
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("correlation_matrix", backend=backend)

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                colorbar_title=colorbar_title,
                significant_threshold=significant_threshold,
                method_labels=method_labels,
                **custom_viz_kwargs,
            )

            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return f"Error creating correlation matrix: {e}"


def create_correlation_pair_plot(
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
    color: Optional[str] = None,
    marker_size: int = 8,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a correlation pair plot (scatter + optional histograms and trendline) and save as PNG.

    Parameters
    ----------
    x_data : list, ndarray, or Series
        X data.
    y_data : list, ndarray, or Series
        Y data.
    output_path : str or Path
        Path to save file.
    title : str
        Plot title.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    correlation : float, optional
        Correlation value to display (calculated if not given).
    method : str, optional
        Correlation method name ("Pearson", etc).
    add_trendline : bool, optional
        Show trendline (default True).
    add_histogram : bool, optional
        Show marginal histograms (default True).
    color : str, optional
        Color for scatter and histograms.
    marker_size : int, optional
        Marker size (default 8).
    theme : str, optional
        Visualization theme.
    backend : str, optional
        Backend ("plotly" only).
    strict : bool, optional
        Strict mode.
    **kwargs :
        Other Plotly go.Scatter or layout parameters.

    Returns
    -------
    str
        Path to saved PNG file or error message.
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("correlation_pair", backend=backend)

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                color=color,
                marker_size=marker_size,
                **custom_viz_kwargs,
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

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

        try:
            fig = fig_creator.create(
                set1=set1,
                set2=set2,
                set1_label=set1_label,
                set2_label=set2_label,
                title=title,
                figsize=figsize,
                annotation=annotation,
                **custom_viz_kwargs,
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
    spider_type: str = "scatterpolar",
    max_value: Optional[float] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a spider (radar) chart visualization and save it as a PNG file.

    Parameters
    ----------
    data : Dict[str, Dict[str, float]] or pd.DataFrame
        Data to visualize. The outer keys (dict) or index (DataFrame) represent series (e.g., samples or groups),
        and the inner keys (dict) or columns (DataFrame) are the categories/axes.
    output_path : str or Path
        File path where the PNG file should be saved.
    title : str
        Chart title.
    categories : list of str, optional
        List of categories (axes). If None, all categories found in the data will be used.
    normalize_values : bool, optional
        Whether to normalize all values to [0, 1] range per category (default: True).
    fill_area : bool, optional
        If True, fills the area under the radar lines (default: True).
    show_gridlines : bool, optional
        Whether to show gridlines on the spider chart (default: True).
    angle_start : float, optional
        Starting angle in degrees for the first axis (default: 90, i.e., up).
    show_legend : bool, optional
        If True, shows the legend for series/groups (default: True).
    spider_type : str, optional
        Plotly trace type: "scatterpolar" (default, regular spider) or "barpolar" (polar bar chart).
    max_value : float, optional
        Explicit maximum value for the radius axis (if None, determined automatically).
    theme : str, optional
        Visualization theme to use for the plot (applies to Plotly layout/colors).
    backend : str, optional
        Backend to use ("plotly" only, required).
    strict : bool, optional
        If True, raises exceptions for invalid configuration; otherwise, logs warnings and continues.
    **kwargs :
        Additional keyword arguments passed to the underlying plotting backend
        (e.g., line_width, marker, color, height, width, opacity, etc.).

    Returns
    -------
    str
        Path to the saved PNG file or an error message if the visualization failed.
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("spider", backend=backend or "plotly")

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                spider_type=spider_type,
                max_value=max_value,
                **custom_viz_kwargs,
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
    hole: float = 0,
    show_values: bool = True,
    value_format: str = ".1f",
    show_percentages: bool = True,
    sort_values: bool = False,
    pull_largest: bool = False,
    pull_value: float = 0.1,
    clockwise: bool = True,
    start_angle: float = 90,
    textposition: str = "auto",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a pie or donut chart and save it as PNG.

    Parameters:
    -----------
    data : Dict[str, float], pd.Series, or list of float
        Data to visualize
    output_path : str or Path
        Path where the PNG file should be saved
    title : str
        Title for the plot
    labels : list of str, optional
        Labels for the slices (needed only if data is a list)
    hole : float, optional
        Size of the hole (0 for pie, >0 for donut)
    show_values : bool, optional
        Show raw values on slices
    value_format : str, optional
        Format for values (e.g., ".1f")
    show_percentages : bool, optional
        Show percentages on slices
    sort_values : bool, optional
        Sort slices by value descending
    pull_largest : bool, optional
        Explode the largest slice
    pull_value : float, optional
        Distance to pull (0–1)
    clockwise : bool, optional
        Draw slices clockwise
    start_angle : float, optional
        Start angle in degrees
    textposition : str, optional
        "inside", "outside", or "auto"
    theme : str, optional
        Visualization theme
    backend : str, optional
        Backend ("plotly" only)
    strict : bool, optional
        Strict mode
    **kwargs:
        Other plotly Pie parameters

    Returns:
    --------
    str
        Path to saved PNG or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("pie", backend=backend or "plotly")

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

        try:
            fig = fig_creator.create(
                data=data,
                title=title,
                labels=labels,
                hole=hole,
                show_values=show_values,
                value_format=value_format,
                show_percentages=show_percentages,
                sort_values=sort_values,
                pull_largest=pull_largest,
                pull_value=pull_value,
                clockwise=clockwise,
                start_angle=start_angle,
                textposition=textposition,
                **custom_viz_kwargs,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return f"Error creating pie chart: {e}"


def create_word_cloud(
    text_data: Union[str, List[str], Dict[str, float]],
    output_path: Union[str, Path],
    title: str,
    max_words: int = 200,
    background_color: str = "white",
    width: int = 800,
    height: int = 400,
    colormap: Optional[str] = "viridis",
    mask: Optional[np.ndarray] = None,
    contour_width: int = 1,
    contour_color: str = "steelblue",
    exclude_words: Optional[List[str]] = None,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a word cloud visualization and save it as a PNG file.

    Parameters
    ----------
    text_data : str, List[str], or Dict[str, float]
        Text data to visualize. If a string, interpreted as raw text; if a list, each element is a document;
        if a dict, it is interpreted as word-frequency pairs.
    output_path : str or Path
        File path where the PNG should be saved.
    title : str
        Title for the visualization.
    max_words : int, optional
        Maximum number of words to include in the word cloud (default 200).
    background_color : str, optional
        Background color for the word cloud (default "white").
    width : int, optional
        Width of the image (default 800).
    height : int, optional
        Height of the image (default 400).
    colormap : str, optional
        Matplotlib colormap for word colors (default "viridis").
    mask : np.ndarray, optional
        Image mask for the word cloud shape.
    contour_width : int, optional
        Width of the contour line around the cloud (default 1).
    contour_color : str, optional
        Color of the contour line (default 'steelblue').
    exclude_words : List[str], optional
        Words to exclude from the word cloud.
    theme : str, optional
        Visualization theme (currently not used, for future compatibility).
    backend : str, optional
        Backend to use (default only).
    strict : bool, optional
        If True, raise exceptions for invalid configuration; otherwise log warnings.
    **kwargs :
        Additional keyword arguments passed to the wordcloud.WordCloud constructor.

    Returns
    -------
    str
        Path to the saved PNG file or an error message.
    """
    with visualization_context(
        backend=backend, theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("wordcloud")

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

        try:
            # Create the word cloud result (dictionary with PIL image)
            result = fig_creator.create(
                text_data=text_data,
                title=title,
                max_words=max_words,
                background_color=background_color,
                width=width,
                height=height,
                colormap=colormap,
                mask=mask,
                contour_width=contour_width,
                contour_color=contour_color,
                exclude_words=exclude_words,
                **custom_viz_kwargs,
            )
            # Save as PNG (WordCloudGenerator returns {'image': PIL.Image...})
            image = result.get("image")
            if image is not None:
                from pamola_core.utils.vis_helpers.word_clouds import WordCloudGenerator

                saved_path = WordCloudGenerator.save_as_png(result, str(output_path))
                if saved_path:
                    return saved_path
                else:
                    return f"Error saving word cloud image to file: {output_path}"
            else:
                return f"Word cloud image not created: {result.get('message', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            return f"Error creating word cloud: {e}"


def create_sunburst_chart(
    data: Union[Dict, pd.DataFrame],
    output_path: Union[str, Path],
    title: str,
    path_column: Optional[str] = None,
    values_column: Optional[str] = None,
    color_column: Optional[str] = None,
    branchvalues: str = "total",
    maxdepth: Optional[int] = None,
    sort_siblings: bool = False,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a sunburst chart visualization for hierarchical data and save it as PNG.

    Parameters:
    -----------
    data : Union[Dict, pd.DataFrame]
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
    branchvalues : str, optional
        How to sum values: "total" (default) or "remainder"
    maxdepth : int, optional
        Maximum depth to display
    sort_siblings : bool, optional
        Whether to sort siblings by value
    theme : str, optional
        Theme name to use for this visualization
    backend : str, optional
        Backend to use: "plotly" (only supported)
    strict : bool, optional
        If True, raise exceptions for invalid configuration; otherwise log warnings
    **kwargs :
        Additional keyword arguments to pass to the plotting backend

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

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

        try:
            fig = fig_creator.create(
                data=data,
                title=title,
                path_column=path_column,
                values_column=values_column,
                color_column=color_column,
                branchvalues=branchvalues,
                maxdepth=maxdepth,
                sort_siblings=sort_siblings,
                **custom_viz_kwargs,
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
    primary_type: str = "bar",
    secondary_type: str = "line",
    x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
    x_label: Optional[str] = None,
    primary_y_label: Optional[str] = None,
    secondary_y_label: Optional[str] = None,
    primary_color: Optional[str] = None,
    secondary_color: Optional[str] = None,
    show_grid: bool = True,
    primary_on_right: bool = False,
    vertical_alignment: bool = True,
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a combined chart (bar+line, bar+area, etc.) with dual Y-axes and save as PNG.

    Parameters
    ----------
    primary_data : Dict[str, Any], pd.Series, or pd.DataFrame
        Data for the primary Y-axis
    secondary_data : Dict[str, Any], pd.Series, or pd.DataFrame
        Data for the secondary Y-axis
    output_path : str or Path
        File path to save the PNG
    title : str
        Title for the plot
    primary_type : str, optional
        Type for primary data: "bar", "line", "scatter", "area" (default "bar")
    secondary_type : str, optional
        Type for secondary data: "line", "scatter", "area", "bar" (default "line")
    x_data : list, ndarray, or Series, optional
        X-axis values (if None, uses indices from primary data)
    x_label : str, optional
        Label for x-axis
    primary_y_label : str, optional
        Label for primary Y-axis
    secondary_y_label : str, optional
        Label for secondary Y-axis
    primary_color : str, optional
        Color for the primary series
    secondary_color : str, optional
        Color for the secondary series
    show_grid : bool, optional
        Show grid lines (default True)
    primary_on_right : bool, optional
        Place primary Y-axis on the right (default False)
    vertical_alignment : bool, optional
        Align zero on both Y-axes (default True)
    theme : str, optional
        Visualization theme
    backend : str, optional
        Backend ("plotly" only)
    strict : bool, optional
        Strict mode for error handling
    **kwargs :
        Additional keyword arguments passed to Plotly traces/layout

    Returns
    -------
    str
        Path to saved PNG file or error message
    """
    # Use context manager to isolate theme and backend settings
    with visualization_context(
        backend=backend or "plotly", theme=theme, strict=strict
    ) as context_info:
        factory = FigureFactory()
        fig_creator = factory.create_figure("combined", backend=backend or "plotly")

        # Filter unsupported kwargs
        custom_viz_kwargs = _filter_kwargs(**kwargs)

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
                show_grid=show_grid,
                primary_on_right=primary_on_right,
                vertical_alignment=vertical_alignment,
                **custom_viz_kwargs,
            )
            register_figure(fig, context_info)
            return _save_figure(fig, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error creating combined chart: {e}")
            return f"Error creating combined chart: {e}"

def create_network_diagram(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    output_path: Union[str, Path],
    node_labels: Optional[Dict[str, str]] = None,
    title: str = "Network Diagram",
    theme: Optional[str] = None,
    backend: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a network diagram visualization and save it as PNG.

    Parameters:
    -----------
    nodes : List[str]
        List of node names.
    edges : List[Tuple[str, str]]
        List of edges as (source, target) tuples.
    output_path : Union[str, Path]
        Path where the PNG file should be saved.
    node_labels : Dict[str, str], optional
        Labels for nodes.
    title : str
        Title for the diagram.
    theme : str, optional
        Visualization theme to apply.
    backend : str, optional
        Visualization backend ("plotly" or "matplotlib").
    strict : bool
        If True, exceptions will be raised on errors; otherwise errors are logged.
    **kwargs:
        Additional arguments passed to the figure saving function.

    Returns:
    --------
    str
        Path to the saved PNG file or error message.
    """
    try:
        # Context manager for visualization settings
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            # Get the appropriate figure creator
            factory = FigureFactory()
            fig_creator = factory.create_figure("network_diagram", backend=backend)

            # Filter unsupported kwargs
            custom_viz_kwargs = _filter_kwargs(**kwargs)

            # Create the figure
            fig = fig_creator.create(
                nodes=nodes,
                edges=edges,
                node_labels=node_labels,
                title=title,
                **custom_viz_kwargs,
            )

            # Register figure for cleanup
            register_figure(fig, context_info)

            # Save the figure
            return _save_figure(fig, output_path, **kwargs)

    except Exception as e:
        logger.error(f"Error creating network diagram: {e}")
        return f"Error creating network diagram: {e}"
    
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


def plot_field_subset_network(
    output_data: Dict[str, pd.DataFrame],
    output_path: Path,
    title: str = "Field Distribution Across Subsets (Network Diagram)",
    theme: str = None,
    backend: str = None,
    strict: bool = False,
    **kwargs,
) -> str:
    """
    Create a network diagram showing how fields are distributed across subsets using Plotly or Matplotlib,
    and save it as a PNG file.

    Parameters:
    ----------- 
    output_data : Dict[str, pd.DataFrame]
        Dictionary mapping subset names to their corresponding DataFrames.
    output_path : Path
        Path where the PNG file will be saved.
    title : str
        Title of the network plot.
    theme : str, optional
        Visualization theme to apply.
    backend : str, optional
        Visualization backend (overrides global setting).
    strict : bool
        If True, exceptions will be raised on errors; otherwise errors are logged.
    **kwargs:
        Additional arguments passed to the figure saving function.

    Returns:
    --------
    str
        Path to the saved PNG file or error message.
    """
    try:
        # Prepare nodes, edges, and labels
        nodes = set()
        edges = []
        node_labels = {}

        for subset_name, df in output_data.items():
            nodes.add(subset_name)
            node_labels[subset_name] = "subset"
            for col in df.columns:
                nodes.add(col)
                node_labels[col] = "field"
                edges.append((subset_name, col))

        # Convert nodes to a list for consistent ordering
        nodes = list(nodes)

        # Call create_network_diagram
        return create_network_diagram(
            nodes=nodes,
            edges=edges,
            output_path=output_path,
            node_labels=node_labels,
            title=title,
            theme=theme,
            backend=backend,
            strict=strict,
            **kwargs,
        )
    except Exception as e:
        if strict:
            raise
        logger.error(f"Error creating field-subset network plot: {e}")
        return f"Error creating field-subset network plot: {e}"
