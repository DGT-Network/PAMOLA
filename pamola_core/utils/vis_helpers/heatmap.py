"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Heatmap Visualization Implementation
Description: Thread-safe heatmap visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for heatmaps using both
Plotly (primary) and Matplotlib (fallback) backends to create
standardized visualizations for matrix data such as correlation matrices.

Key features:
1. Context-isolated visualization state for thread-safe parallel execution
2. Support for multiple data formats (dict, DataFrame, ndarray)
3. Flexible annotation and masking capabilities
4. Customizable color scales for effective data representation
5. Thread-safe registry operations
6. Proper theme application using context-aware helpers
7. Comprehensive error handling with appropriate fallbacks
8. Memory-efficient operation with explicit resource cleanup

Implementation follows the PAMOLA.CORE framework with standardized interfaces for
visualization generation while ensuring concurrent operations do not interfere
with each other through proper context isolation.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from pamola_core.utils.vis_helpers.base import (
    PlotlyFigure,
    MatplotlibFigure,
    FigureRegistry,
)
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_plotly_figure,
    apply_theme_to_matplotlib_figure,
    get_matplotlib_colormap,
)
from pamola_core.utils.vis_helpers.context import visualization_context, register_figure

# Configure logger
logger = logging.getLogger(__name__)


def prepare_data_for_heatmap(
    data: Union[Dict[str, Dict[str, float]], pd.DataFrame, np.ndarray],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Prepare data for heatmap visualization.

    Parameters:
    -----------
    data : Dict[str, Dict[str, float]], pd.DataFrame, or np.ndarray
        Data to prepare

    Returns:
    --------
    Tuple[np.ndarray, List[str], List[str]]
        Tuple containing (matrix, x_labels, y_labels)
    """
    try:
        # Convert different input formats to matrix and labels
        if isinstance(data, dict):
            # Nested dictionary - convert to DataFrame first
            df = pd.DataFrame(data)
            matrix = df.values
            x_labels = list(df.columns)
            y_labels = list(df.index)
        elif isinstance(data, pd.DataFrame):
            # DataFrame - extract values and labels
            matrix = data.values
            x_labels = list(data.columns)
            y_labels = list(data.index)
        elif isinstance(data, np.ndarray):
            # Raw numpy array - use numbered labels
            matrix = data
            x_labels = [str(i) for i in range(matrix.shape[1])]
            y_labels = [str(i) for i in range(matrix.shape[0])]
        else:
            raise TypeError(f"Unsupported data type for heatmap: {type(data)}")

        return matrix, x_labels, y_labels
    except Exception as e:
        logger.error(f"Error preparing heatmap data: {e}")
        # Return empty but valid objects to avoid further errors
        return np.array([[0]]), [""], [""]


def prepare_text_colors(
    matrix: np.ndarray, annotation_color_threshold: Optional[float] = 0.5
) -> np.ndarray:
    """
    Prepare text colors for heatmap annotations based on cell values.

    Parameters:
    -----------
    matrix : np.ndarray
        The data matrix
    annotation_color_threshold : float, optional
        Threshold value (0-1) to switch annotation color from white to black

    Returns:
    --------
    np.ndarray
        Array of text colors for each cell
    """
    try:
        if annotation_color_threshold is not None:
            # Calculate the color range
            z_min, z_max = np.nanmin(matrix), np.nanmax(matrix)
            normalized_data = (
                (matrix - z_min) / (z_max - z_min)
                if z_max > z_min
                else np.ones_like(matrix)
            )

            # Create a 2D array of colors
            text_colors = np.where(
                normalized_data > annotation_color_threshold, "white", "black"
            )
            for i in range(text_colors.shape[0]):
                for j in range(text_colors.shape[1]):
                    if np.isnan(matrix[i, j]):
                        text_colors[i, j] = (
                            "rgba(0,0,0,0)"  # Make text transparent for NaN values
                        )
        else:
            # Use all black text
            text_colors = np.full_like(matrix, "black", dtype=object)
            for i in range(text_colors.shape[0]):
                for j in range(text_colors.shape[1]):
                    if np.isnan(matrix[i, j]):
                        text_colors[i, j] = (
                            "rgba(0,0,0,0)"  # Make text transparent for NaN values
                        )

        return text_colors
    except Exception as e:
        logger.error(f"Error preparing text colors: {e}")
        # Return a simple black text array as fallback
        default_shape = matrix.shape if hasattr(matrix, "shape") else (1, 1)
        return np.full(default_shape, "black", dtype=object)


def prepare_text_values(
    matrix: np.ndarray, annotation_format: str = ".2f"
) -> np.ndarray:
    """
    Prepare text values for heatmap annotations.

    Parameters:
    -----------
    matrix : np.ndarray
        The data matrix
    annotation_format : str
        Format string for annotations

    Returns:
    --------
    np.ndarray
        Formatted text values
    """
    try:
        if annotation_format.startswith(".") and annotation_format.endswith("f"):
            # Parse decimal format like ".2f" to get number of decimal places
            decimal_places = int(annotation_format[1:-1])
        else:
            decimal_places = 2

        return np.round(matrix, decimals=decimal_places)
    except Exception as e:
        logger.error(f"Error preparing text values: {e}")
        # Return the original matrix as fallback
        return matrix


def handle_mask_values(
    matrix: np.ndarray, mask_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply mask to the data matrix.

    Parameters:
    -----------
    matrix : np.ndarray
        The data matrix
    mask_values : np.ndarray, optional
        Boolean mask to hide certain values (True = visible, False = hidden)

    Returns:
    --------
    np.ndarray
        Masked matrix where hidden values are replaced with NaN
    """
    try:
        if mask_values is None:
            return matrix.copy()

        # Create a copy to avoid modifying the original matrix
        masked_matrix = matrix.copy()

        # Ensure mask is the right shape
        if mask_values.shape != matrix.shape:
            logger.warning(
                f"Mask shape {mask_values.shape} does not match data shape {matrix.shape}. Ignoring mask."
            )
            return masked_matrix

        # Make values that should be masked as NaN
        return np.where(mask_values, masked_matrix, np.nan)
    except Exception as e:
        logger.error(f"Error applying mask: {e}")
        # Return the original matrix as fallback
        return matrix.copy() if hasattr(matrix, "copy") else matrix


def add_colorbar_to_matplotlib(fig, im, ax, colorbar_label=None, pad=0.01):
    """
    Add a colorbar to a Matplotlib figure.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to
    im : matplotlib.image.AxesImage
        The image object from imshow
    ax : matplotlib.axes.Axes
        The axes object
    colorbar_label : str, optional
        Label for the colorbar
    pad : float
        Padding between the colorbar and the plot

    Returns:
    --------
    matplotlib.colorbar.Colorbar
        The colorbar object
    """
    try:
        import matplotlib.pyplot as plt

        cbar = plt.colorbar(im, ax=ax, pad=pad)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        return cbar
    except Exception as e:
        logger.error(f"Error adding colorbar: {e}")
        return None


def create_matplotlib_imshow(ax, masked_matrix, cmap_with_alpha, kwargs):
    """
    Create a matplotlib imshow plot with proper error handling for the aspect parameter.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    masked_matrix : np.ndarray
        The data matrix for visualization
    cmap_with_alpha : matplotlib.colors.Colormap
        Colormap with alpha channel for NaN values
    kwargs : dict
        Additional arguments for imshow

    Returns:
    --------
    matplotlib.image.AxesImage
        The image object from imshow
    """
    try:
        # Extract common parameters
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        # Handle the aspect parameter separately to avoid type errors
        aspect_value = kwargs.pop("aspect", "auto")

        try:
            # Try to create the heatmap with the provided aspect value
            im = ax.imshow(
                masked_matrix,
                cmap=cmap_with_alpha,
                vmin=vmin,
                vmax=vmax,
                aspect=aspect_value,
                origin="upper",
                **kwargs,
            )
        except TypeError:
            # If that fails, log a warning and use default aspect handling
            logger.warning(
                f"Invalid aspect value '{aspect_value}'. Using default instead."
            )
            im = ax.imshow(
                masked_matrix,
                cmap=cmap_with_alpha,
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                **kwargs,
            )

        return im
    except Exception as e:
        logger.error(f"Error creating matplotlib imshow: {e}")
        # Create a simple fallback plot
        return ax.imshow(masked_matrix, origin="upper")


class PlotlyHeatmap(PlotlyFigure):
    """Heatmap implementation using Plotly."""

    def create(
        self,
        data: Union[Dict[str, Dict[str, float]], pd.DataFrame, np.ndarray],
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colorscale: Optional[str] = None,
        annotate: bool = True,
        annotation_format: str = ".2f",
        annotation_color_threshold: Optional[float] = 0.5,
        mask_values: Optional[np.ndarray] = None,
        colorbar_title: Optional[str] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a heatmap using Plotly.

        Parameters:
        -----------
        data : Dict[str, Dict[str, float]], pd.DataFrame, or np.ndarray
            Data to visualize. If Dict, nested dictionary with row and column labels.
            If DataFrame, index and columns used as labels. If ndarray, raw values.
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
        annotation_color_threshold : float, optional
            Threshold value (0-1) to switch annotation color from white to black
        mask_values : np.ndarray, optional
            Boolean mask to hide certain values (True = visible, False = hidden)
        colorbar_title : str, optional
            Title for the colorbar
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional arguments to pass to go.Heatmap

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the heatmap
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:

                # Process input data to a standard format
                matrix, x_labels, y_labels = prepare_data_for_heatmap(data)

                # Handle masking if provided
                matrix = handle_mask_values(matrix, mask_values)

                # Create figure
                fig = go.Figure()

                # Register figure for cleanup
                register_figure(fig, context_info)

                # Get colorscale from theme if not provided
                theme_colorscale = kwargs.pop("theme_colorscale", colorscale)
                if theme_colorscale is None:
                    theme_colorscale = "Blues"  # Default to Blues colorscale

                # Create custom colorscale with white for NaN values
                # This ensures masked values appear as white/transparent
                custom_colorscale = kwargs.pop("custom_colorscale", None)

                # Add heatmap trace
                heatmap_args = {
                    "z": matrix,
                    "x": x_labels,
                    "y": y_labels,
                    "colorscale": theme_colorscale,
                    "hoverongaps": False,
                    "hovertemplate": "x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>",
                    "showscale": True,
                    "colorbar": {"title": colorbar_title} if colorbar_title else None,
                }

                fig.add_trace(go.Heatmap(**heatmap_args))

                # Add text annotations via scatter overlay if annotate is enabled
                if annotate:
                    # Create text template
                    if annotation_format.startswith(".") and annotation_format.endswith("f"):
                        decimal_places = int(annotation_format[1:-1])
                        text_template = f"%{{z:.{decimal_places}f}}"
                    else:
                        text_template = f"%{{z:{annotation_format}}}"

                    text_values = prepare_text_values(matrix, annotation_format)
                    text_colors = prepare_text_colors(matrix, annotation_color_threshold)

                    # Flatten for 2D loop
                    for i, y in enumerate(y_labels):
                        for j, x in enumerate(x_labels):
                            val = matrix[i][j]
                            if not np.isnan(val):
                                fig.add_trace(
                                    go.Scatter(
                                        x=[x],
                                        y=[y],
                                        text=[text_values[i][j]],
                                        mode="text",
                                        textfont=dict(color=text_colors[i][j]),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    )
                                )

                # Configure layout
                layout_args = {
                    "title": title,
                    "xaxis": {
                        "title": x_label,
                        "side": "bottom",
                        "tickangle": -45 if len(x_labels) > 5 else 0,
                    },
                    "yaxis": {
                        "title": y_label,
                        "autorange": "reversed",  # To have the first row at the top
                    },
                    "margin": {"t": 80, "b": 100, "l": 100, "r": 100},
                }

                fig.update_layout(**layout_args)

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except ImportError as e:
                logger.error(
                    f"Plotly is not available. Please install it with: pip install plotly. Error: {e}"
                )
                # Try to fall back to MatplotlibHeatmap
                try:
                    fallback = MatplotlibHeatmap()
                    logger.warning("Falling back to Matplotlib implementation")
                    return fallback.create(
                        data=data,
                        title=title,
                        x_label=x_label,
                        y_label=y_label,
                        cmap=colorscale,
                        annotate=annotate,
                        annotation_format=annotation_format,
                        annotation_color_threshold=annotation_color_threshold,
                        mask_values=mask_values,
                        colorbar_label=colorbar_title,
                        backend=backend,
                        theme=theme,
                        strict=strict,
                        **kwargs,
                    )
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to Matplotlib also failed: {fallback_error}"
                    )
                    return self.create_empty_figure(
                        title=title,
                        message="Error creating heatmap: Plotly not available and Matplotlib fallback failed",
                    )
            except Exception as e:
                logger.error(f"Error creating heatmap: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating heatmap: {str(e)}"
                )

    def update(
        self,
        fig: Any,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Update an existing Plotly heatmap.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Plotly figure to update
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Parameters to update

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated figure
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import plotly.graph_objects as go

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Ensure we have a Plotly figure
                if not isinstance(fig, go.Figure):
                    logger.warning("Cannot update non-Plotly figure with PlotlyHeatmap")
                    return fig

                # Update title if provided
                if "title" in kwargs:
                    fig.update_layout(title=kwargs["title"])

                # Update axis labels if provided
                if "x_label" in kwargs:
                    fig.update_layout(xaxis_title=kwargs["x_label"])
                if "y_label" in kwargs:
                    fig.update_layout(yaxis_title=kwargs["y_label"])

                # Update data if provided
                if "data" in kwargs:
                    data = kwargs["data"]

                    # Process new data
                    matrix, x_labels, y_labels = prepare_data_for_heatmap(data)

                    # Handle masking if provided
                    if "mask_values" in kwargs:
                        matrix = handle_mask_values(matrix, kwargs["mask_values"])

                    # Update colorscale if provided
                    if "colorscale" in kwargs:
                        fig.update_traces(
                            colorscale=kwargs["colorscale"],
                            selector=dict(type="heatmap"),
                        )

                    # Update annotations if needed
                    annotate = kwargs.get("annotate", None)
                    annotation_format = kwargs.get("annotation_format", ".2f")
                    annotation_color_threshold = kwargs.get(
                        "annotation_color_threshold", 0.5
                    )

                    if annotate is not None:
                        if annotate:
                            # Prepare text values and colors
                            text_values = prepare_text_values(matrix, annotation_format)
                            text_colors = prepare_text_colors(
                                matrix, annotation_color_threshold
                            )

                            # Create text template
                            if annotation_format.startswith(
                                "."
                            ) and annotation_format.endswith("f"):
                                # Parse decimal format like ".2f" to get number of decimal places
                                decimal_places = int(annotation_format[1:-1])
                                text_template = f"%{{z:.{decimal_places}f}}"
                            else:
                                # Use provided format string
                                text_template = f"%{{z:{annotation_format}}}"

                            # Update with annotations
                            fig.update_traces(
                                text=text_values,
                                texttemplate=text_template,
                                textfont={"color": text_colors.flatten()},
                                selector=dict(type="heatmap"),
                            )
                        else:
                            # Remove annotations
                            fig.update_traces(
                                text=None,
                                texttemplate=None,
                                selector=dict(type="heatmap"),
                            )

                    # Update colorbar title if provided
                    if "colorbar_title" in kwargs:
                        fig.update_traces(
                            colorbar={"title": kwargs["colorbar_title"]},
                            selector=dict(type="heatmap"),
                        )

                    # Update heatmap data
                    fig.update_traces(
                        z=matrix, x=x_labels, y=y_labels, selector=dict(type="heatmap")
                    )

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except Exception as e:
                logger.error(f"Error updating heatmap: {e}")
                return fig  # Return original figure as fallback


class MatplotlibHeatmap(MatplotlibFigure):
    """Heatmap implementation using Matplotlib."""

    def create(
        self,
        data: Union[Dict[str, Dict[str, float]], pd.DataFrame, np.ndarray],
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        cmap: Optional[str] = None,
        annotate: bool = True,
        annotation_format: str = ".2f",
        annotation_color_threshold: Optional[float] = 0.5,
        mask_values: Optional[np.ndarray] = None,
        colorbar_label: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a heatmap using Matplotlib.

        Parameters:
        -----------
        data : Dict[str, Dict[str, float]], pd.DataFrame, or np.ndarray
            Data to visualize. If Dict, nested dictionary with row and column labels.
            If DataFrame, index and columns used as labels. If ndarray, raw values.
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis
        y_label : str, optional
            Label for the y-axis
        cmap : str, optional
            Colormap to use (default from theme if None)
        annotate : bool
            Whether to annotate the heatmap with values
        annotation_format : str
            Format string for annotations (e.g., ".2f" for 2 decimal places)
        annotation_color_threshold : float, optional
            Threshold value (0-1) to switch annotation color from white to black
        mask_values : np.ndarray, optional
            Boolean mask to hide certain values (True = visible, False = hidden)
        colorbar_label : str, optional
            Label for the colorbar
        figsize : Tuple[int, int]
            Figure size (width, height)
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional arguments to pass to ax.imshow

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with the heatmap
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import matplotlib.pyplot as plt
                import matplotlib as mpl

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                # Process input data to a standard format
                matrix, x_labels, y_labels = prepare_data_for_heatmap(data)

                # Handle masking if provided
                masked_matrix = handle_mask_values(matrix, mask_values)

                # Create figure and axes
                fig, ax = plt.subplots(figsize=figsize)

                # Register figure for cleanup
                register_figure(fig, context_info)

                # Get colormap from theme or use provided cmap
                if cmap is None:
                    cmap = get_matplotlib_colormap()

                # Set the NaN color to light gray or white
                cmap_with_alpha = mpl.cm.get_cmap(cmap).copy()
                cmap_with_alpha.set_bad("white")

                # Create heatmap with proper error handling for aspect parameter
                im = create_matplotlib_imshow(
                    ax, masked_matrix, cmap_with_alpha, kwargs
                )

                # Add colorbar
                cbar = add_colorbar_to_matplotlib(fig, im, ax, colorbar_label, pad=0.01)

                # Add title and labels
                ax.set_title(title)
                if x_label:
                    ax.set_xlabel(x_label)
                if y_label:
                    ax.set_ylabel(y_label)

                # Set tick labels
                ax.set_xticks(np.arange(len(x_labels)))
                ax.set_yticks(np.arange(len(y_labels)))
                ax.set_xticklabels(x_labels)
                ax.set_yticklabels(y_labels)

                # Rotate the tick labels and set their alignment
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                # Add text annotations if requested
                if annotate:
                    # Define the threshold for color transition
                    threshold = (
                        annotation_color_threshold
                        if annotation_color_threshold is not None
                        else 0.5
                    )

                    # Get min/max values for normalization
                    z_min, z_max = np.nanmin(matrix), np.nanmax(matrix)

                    # Define a function to get text color based on background
                    def get_text_color(val, norm_val):
                        if np.isnan(val):
                            return "none"  # Transparent for NaN values
                        elif norm_val > threshold:
                            return "white"  # Light background, dark text
                        else:
                            return "black"  # Dark background, light text

                    # Add text annotations
                    for i in range(len(y_labels)):
                        for j in range(len(x_labels)):
                            val = matrix[i, j]

                            # Calculate normalized value for color determination
                            if annotation_color_threshold is not None:
                                norm_val = (
                                    (val - z_min) / (z_max - z_min)
                                    if z_max > z_min
                                    else 0.5
                                )
                            else:
                                norm_val = 0  # Default if not using threshold

                            # Get appropriate text color
                            text_color = get_text_color(val, norm_val)

                            # Add annotation if not NaN
                            if not np.isnan(val):
                                text = format(val, annotation_format)
                                ax.text(
                                    j,
                                    i,
                                    text,
                                    ha="center",
                                    va="center",
                                    color=text_color,
                                )

                # Adjust layout with tight margins
                plt.tight_layout()

                # Apply theme (this will handle background color, grid lines, etc.)
                fig = apply_theme_to_matplotlib_figure(fig)

                return fig
            except Exception as e:
                logger.error(f"Error creating matplotlib heatmap: {e}")
                return self.create_empty_figure(
                    title=title,
                    message=f"Error creating heatmap: {str(e)}",
                    figsize=figsize,
                )

    def update(
        self,
        fig: Any,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Update an existing Matplotlib heatmap.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to update
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Parameters to update

        Returns:
        --------
        matplotlib.figure.Figure
            Updated figure
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import matplotlib.pyplot as plt
                import matplotlib as mpl

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Ensure we have a Matplotlib figure
                if not isinstance(fig, plt.Figure):
                    logger.warning(
                        "Cannot update non-Matplotlib figure with MatplotlibHeatmap"
                    )
                    return fig

                # Get the axes (assuming single subplot)
                if len(fig.axes) == 0:
                    logger.warning("Figure has no axes to update")
                    return fig

                ax = fig.axes[0]

                # If we need to completely redraw the heatmap due to new data
                if "data" in kwargs:
                    # Clear axes for redrawing
                    ax.clear()

                    # Get updated parameters
                    data = kwargs["data"]
                    title = kwargs.get(
                        "title", fig.texts[0].get_text() if fig.texts else ""
                    )
                    x_label = kwargs.get("x_label", ax.get_xlabel())
                    y_label = kwargs.get("y_label", ax.get_ylabel())
                    cmap = kwargs.get("cmap", None)
                    annotate = kwargs.get("annotate", True)
                    annotation_format = kwargs.get("annotation_format", ".2f")
                    annotation_color_threshold = kwargs.get(
                        "annotation_color_threshold", 0.5
                    )
                    mask_values = kwargs.get("mask_values", None)
                    colorbar_label = kwargs.get("colorbar_label", None)

                    # Process new data
                    matrix, x_labels, y_labels = prepare_data_for_heatmap(data)

                    # Handle masking if provided
                    masked_matrix = handle_mask_values(matrix, mask_values)

                    # Get colormap from theme or use provided cmap
                    if cmap is None:
                        cmap = get_matplotlib_colormap()

                    # Set the NaN color to light gray or white
                    cmap_with_alpha = mpl.cm.get_cmap(cmap).copy()
                    cmap_with_alpha.set_bad("white")

                    # Create heatmap with proper error handling for aspect parameter
                    im = create_matplotlib_imshow(
                        ax, masked_matrix, cmap_with_alpha, kwargs
                    )

                    # Update colorbar
                    if len(fig.axes) > 1:
                        # Remove old colorbar
                        cbar_ax = fig.axes[1]
                        cbar_ax.remove()

                    # Add new colorbar
                    cbar = add_colorbar_to_matplotlib(
                        fig, im, ax, colorbar_label, pad=0.01
                    )

                    # Add title and labels
                    ax.set_title(title)
                    if x_label:
                        ax.set_xlabel(x_label)
                    if y_label:
                        ax.set_ylabel(y_label)

                    # Set tick labels
                    ax.set_xticks(np.arange(len(x_labels)))
                    ax.set_yticks(np.arange(len(y_labels)))
                    ax.set_xticklabels(x_labels)
                    ax.set_yticklabels(y_labels)

                    # Rotate the tick labels and set their alignment
                    plt.setp(
                        ax.get_xticklabels(),
                        rotation=45,
                        ha="right",
                        rotation_mode="anchor",
                    )

                    # Add text annotations if requested
                    if annotate:
                        # Define the threshold for color transition
                        threshold = (
                            annotation_color_threshold
                            if annotation_color_threshold is not None
                            else 0.5
                        )

                        # Get min/max values for normalization
                        z_min, z_max = np.nanmin(matrix), np.nanmax(matrix)

                        # Define a function to get text color based on background
                        def get_text_color(val, norm_val):
                            if np.isnan(val):
                                return "none"  # Transparent for NaN values
                            elif norm_val > threshold:
                                return "white"  # Light background, dark text
                            else:
                                return "black"  # Dark background, light text

                        # Add text annotations
                        for i in range(len(y_labels)):
                            for j in range(len(x_labels)):
                                val = matrix[i, j]

                                # Calculate normalized value for color determination
                                if annotation_color_threshold is not None:
                                    norm_val = (
                                        (val - z_min) / (z_max - z_min)
                                        if z_max > z_min
                                        else 0.5
                                    )
                                else:
                                    norm_val = 0  # Default if not using threshold

                                # Get appropriate text color
                                text_color = get_text_color(val, norm_val)

                                # Add annotation if not NaN
                                if not np.isnan(val):
                                    text = format(val, annotation_format)
                                    ax.text(
                                        j,
                                        i,
                                        text,
                                        ha="center",
                                        va="center",
                                        color=text_color,
                                    )
                else:
                    # Just update basic properties

                    # Update title if provided
                    if "title" in kwargs:
                        ax.set_title(kwargs["title"])

                    # Update axis labels if provided
                    if "x_label" in kwargs:
                        ax.set_xlabel(kwargs["x_label"])
                    if "y_label" in kwargs:
                        ax.set_ylabel(kwargs["y_label"])

                    # Update colorbar label if provided
                    if "colorbar_label" in kwargs and len(fig.axes) > 1:
                        cbar_ax = fig.axes[1]
                        cbar_ax.set_ylabel(kwargs["colorbar_label"])

                # Adjust layout
                plt.tight_layout()

                # Apply theme
                fig = apply_theme_to_matplotlib_figure(fig)

                return fig
            except Exception as e:
                logger.error(f"Error updating matplotlib heatmap: {e}")
                return fig  # Return original figure as fallback


# Register implementations
FigureRegistry.register("heatmap", "plotly", PlotlyHeatmap)
FigureRegistry.register("heatmap", "matplotlib", MatplotlibHeatmap)
