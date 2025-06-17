"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Scatter Plot Visualization Implementation
Description: Thread-safe scatter plot visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for scatter plots primarily using
the Plotly backend with context-isolated state management for concurrent
visualization operations.

Key features:
1. Context-isolated visualization state for thread-safe parallel execution
2. Support for multiple data formats with robust cleaning
3. Optional trendline calculation and visualization
4. Customizable marker size, color, and hover text
5. Thread-safe registry operations
6. Proper theme application using context-aware helpers
7. Comprehensive error handling with appropriate fallbacks
8. Memory-efficient operation with explicit resource cleanup

Implementation follows the PAMOLA.CORE framework with standardized interfaces for
visualization generation while ensuring concurrent operations do not interfere
with each other through proper context isolation.
"""

import logging
from typing import List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

from pamola_core.utils.vis_helpers.base import (
    MatplotlibFigure,
    PlotlyFigure,
    FigureRegistry,
)
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_matplotlib_figure,
    apply_theme_to_plotly_figure,
    get_theme_colors,
)
from pamola_core.utils.vis_helpers.context import visualization_context

# Configure logger
logger = logging.getLogger(__name__)


class BaseScatterPlot:
    """
    Base class for scatter plot implementations with shared utility methods.
    """

    @staticmethod
    def _prepare_scatter_data(
        x_data: Union[List[float], np.ndarray, pd.Series],
        y_data: Union[List[float], np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and clean scatter plot data.

        Parameters:
        -----------
        x_data : Union[List[float], np.ndarray, pd.Series]
            Data for x-axis
        y_data : Union[List[float], np.ndarray, pd.Series]
            Data for y-axis

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Cleaned x and y data arrays
        """
        try:
            # Convert data to arrays
            x = np.array(x_data)
            y = np.array(y_data)

            # Remove NaN values (points where either x or y is NaN)
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]

            return x, y
        except Exception as e:
            logger.error(f"Error preparing scatter data: {e}")
            # Return empty arrays as fallback
            return np.array([]), np.array([])

    @staticmethod
    def _calculate_trendline(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[
        Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Calculate linear regression trendline.

        Parameters:
        -----------
        x : np.ndarray
            X-axis data
        y : np.ndarray
            Y-axis data

        Returns:
        --------
        Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]
            Slope, intercept, x_range, y_range for trendline
        """
        try:
            # Ensure we have at least 2 points for regression
            if len(x) > 1:
                # Calculate linear regression
                slope, intercept = np.polyfit(x, y, 1)

                # Generate points for the line
                x_range = np.linspace(min(x), max(x), 100)
                y_range = slope * x_range + intercept

                return slope, intercept, x_range, y_range

            # Not enough points for regression
            return None, None, None, None

        except Exception as e:
            # Log any unexpected errors
            logger.warning(f"Error calculating trendline: {e}")
            return None, None, None, None


class PlotlyScatterPlot(PlotlyFigure, BaseScatterPlot):
    """Scatter plot implementation using Plotly."""

    def create(
        self,
        x_data: Union[List[float], np.ndarray, pd.Series],
        y_data: Union[List[float], np.ndarray, pd.Series],
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
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a scatter plot using Plotly with comprehensive customization.

        Parameters:
        -----------
        x_data : Union[List[float], np.ndarray, pd.Series]
            Data for x-axis
        y_data : Union[List[float], np.ndarray, pd.Series]
            Data for y-axis
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis
        y_label : str, optional
            Label for the y-axis
        add_trendline : bool
            Whether to add a linear regression trendline
        correlation : float, optional
            Correlation coefficient to display (if provided)
        method : str, optional
            Method used for correlation (e.g., "Pearson", "Spearman")
        hover_text : List[str], optional
            Custom hover text for each point
        marker_size : Union[List[float], float], optional
            Size(s) for markers, either single value or list of values
        color_scale : Union[List[str], str], optional
            Color scale for markers when using color_values
        color_values : List[float], optional
            Values used to color the markers
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional customization parameters

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly scatter plot figure
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import plotly.graph_objects as go

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                # Prepare data
                x, y = BaseScatterPlot._prepare_scatter_data(x_data, y_data)

                # Validate data
                if len(x) == 0 or len(y) == 0:
                    logger.warning("No valid data points for scatter plot")
                    return self.create_empty_figure(
                        title=title, message="No valid data points for scatter plot"
                    )

                # Create figure
                fig = go.Figure()

                # Register figure for cleanup
                register_figure(fig, context_info)

                # Prepare marker properties
                marker_props = {}

                # Handle marker size
                if marker_size is not None:
                    if isinstance(marker_size, (int, float)):
                        # Convert float to int if necessary to satisfy type requirements
                        marker_props["size"] = (
                            int(marker_size)
                            if isinstance(marker_size, float)
                            else marker_size
                        )
                    else:
                        # Convert to numpy array and ensure it matches data length
                        marker_size_array = np.array(
                            marker_size, dtype=np.int32
                        )  # Ensure int dtype
                        if len(marker_size_array) == len(x):
                            marker_props["size"] = marker_size_array.tolist()
                        else:
                            logger.warning(
                                "Marker size length doesn't match data length"
                            )
                            # Make sure we're using an integer value
                            if len(marker_size_array) > 0:
                                first_value = marker_size_array[0]
                                marker_props["size"] = (
                                    int(first_value)
                                    if isinstance(first_value, float)
                                    else first_value
                                )
                            else:
                                marker_props["size"] = 8  # Default size

                # Handle marker color
                marker_color = kwargs.pop("marker_color", get_theme_colors(1)[0])
                if color_values is not None:
                    # Convert color values to list for Plotly compatibility
                    color_values_array = np.array(color_values)

                    # Ensure color values match data length
                    if len(color_values_array) == len(x):
                        marker_props["color"] = color_values_array.tolist()

                        # Handle color scale
                        if color_scale is not None:
                            marker_props["colorscale"] = str(color_scale)

                        # Add colorbar with title
                        marker_props["colorbar"] = dict(
                            title=str(kwargs.pop("colorbar_title", ""))
                        )
                    else:
                        logger.warning("Color values length doesn't match data length")
                        marker_props["color"] = marker_color
                else:
                    marker_props["color"] = marker_color

                # Prepare hover text
                hovertext = None
                if hover_text is not None:
                    hover_text_array = np.array(hover_text, dtype=str)
                    if len(hover_text_array) == len(x):
                        hovertext = hover_text_array.tolist()

                # Add scatter trace
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=marker_props,
                        hovertext=hovertext,
                        **kwargs,
                    )
                )

                # Add trendline if requested
                if add_trendline:
                    slope, intercept, x_range, y_range = (
                        BaseScatterPlot._calculate_trendline(x, y)
                    )
                    if x_range is not None and y_range is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_range,
                                mode="lines",
                                name="Trendline",
                                line=dict(color="red", width=2, dash="dash"),
                            )
                        )

                # Set axis labels
                fig.update_layout(
                    xaxis_title=x_label or "X", yaxis_title=y_label or "Y"
                )

                # Set title
                fig.update_layout(title=title)

                # Add correlation annotation if provided
                if correlation is not None:
                    correlation_text = f"{method or 'Correlation'}: {correlation:.3f}"
                    fig.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=correlation_text,
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        font=dict(size=12),
                    )

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except ImportError as e:
                logger.error(
                    f"Plotly is not available. Please install it with: pip install plotly. Error: {e}"
                )
                return self.create_empty_figure(
                    title=title,
                    message="Plotly is not available. Please install it with: pip install plotly.",
                )
            except Exception as e:
                logger.error(f"Error creating scatter plot: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating scatter plot: {str(e)}"
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
        Update an existing Plotly scatter plot.

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

                # Validate figure
                if not isinstance(fig, go.Figure):
                    logger.warning(
                        "Cannot update non-Plotly figure with PlotlyScatterPlot"
                    )
                    return fig

                # Update title and axis labels if provided
                if "title" in kwargs:
                    fig.update_layout(title=kwargs["title"])
                if "x_label" in kwargs:
                    fig.update_layout(xaxis_title=kwargs["x_label"])
                if "y_label" in kwargs:
                    fig.update_layout(yaxis_title=kwargs["y_label"])

                # Update data if provided
                if "x_data" in kwargs and "y_data" in kwargs:
                    x_data = kwargs["x_data"]
                    y_data = kwargs["y_data"]

                    # Prepare data
                    x, y = BaseScatterPlot._prepare_scatter_data(x_data, y_data)

                    # Check if there are any valid data points
                    if len(x) == 0 or len(y) == 0:
                        logger.warning("No valid data points for updated scatter plot")
                        return fig

                    # Update scatter trace
                    fig.update_traces(x=x, y=y, selector=dict(mode="markers"))

                    # Update trendline if present and add_trendline is True
                    add_trendline = kwargs.get("add_trendline", None)
                    trendline_trace = next(
                        (
                            trace
                            for trace in fig.data
                            if getattr(trace, "name", "") == "Trendline"
                        ),
                        None,
                    )

                    if add_trendline is not None:
                        if add_trendline:
                            slope, intercept, x_range, y_range = (
                                BaseScatterPlot._calculate_trendline(x, y)
                            )
                            if x_range is not None and y_range is not None:
                                if trendline_trace is not None:
                                    # Update existing trendline
                                    fig.update_traces(
                                        x=x_range,
                                        y=y_range,
                                        selector=dict(name="Trendline"),
                                    )
                                else:
                                    # Add new trendline
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_range,
                                            y=y_range,
                                            mode="lines",
                                            name="Trendline",
                                            line=dict(
                                                color="red", width=2, dash="dash"
                                            ),
                                        )
                                    )
                        elif trendline_trace is not None:
                            # Remove trendline if add_trendline is False
                            fig.data = [
                                trace
                                for trace in fig.data
                                if getattr(trace, "name", "") != "Trendline"
                            ]

                # Update marker properties if provided
                marker_updates = {}

                if "marker_size" in kwargs:
                    marker_size = kwargs["marker_size"]
                    if isinstance(marker_size, (int, float)):
                        marker_updates["size"] = (
                            int(marker_size)
                            if isinstance(marker_size, float)
                            else marker_size
                        )
                    elif (
                        isinstance(marker_size, (list, np.ndarray))
                        and len(fig.data) > 0
                    ):
                        scatter_trace = fig.data[0]
                        x_data = getattr(scatter_trace, "x", [])
                        if len(marker_size) == len(x_data):
                            marker_updates["size"] = list(marker_size)

                if "marker_color" in kwargs:
                    marker_updates["color"] = kwargs["marker_color"]

                if marker_updates:
                    fig.update_traces(
                        marker=marker_updates, selector=dict(mode="markers")
                    )

                # Update correlation annotation if provided
                if "correlation" in kwargs:
                    correlation = kwargs["correlation"]
                    method = kwargs.get("method", "Correlation")

                    # Find and remove existing correlation annotation
                    annotations = list(fig.layout.annotations or [])
                    filtered_annotations = []

                    for a in annotations:
                        # Check if this is a correlation annotation (careful with attribute access)
                        if (
                            a.get("xref") == "paper"
                            and a.get("yref") == "paper"
                            and isinstance(a.get("text", ""), str)
                            and method in a.get("text", "")
                        ):
                            # Skip this annotation (it's a correlation annotation)
                            pass
                        else:
                            # Keep this annotation
                            filtered_annotations.append(a)

                    # Add new correlation annotation
                    correlation_text = f"{method}: {correlation:.3f}"

                    # Create the annotation directly rather than using a dictionary
                    new_annotation = go.layout.Annotation(
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=correlation_text,
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        font=dict(size=12),
                    )

                    filtered_annotations.append(new_annotation)
                    fig.update_layout(annotations=filtered_annotations)

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except ImportError as imp_error:
                # Define go as None to ensure it's defined in except block
                go = None
                logger.error(
                    f"Plotly is not available for updating the figure. Error: {imp_error}"
                )
                return fig
            except Exception as e:
                logger.error(f"Error updating scatter plot: {e}")
                return fig


class MatplotlibScatterPlot(MatplotlibFigure, BaseScatterPlot):
    """
    Scatter plot implementation using Matplotlib.
    """

    def create(
        self,
        x_data: Union[List[float], np.ndarray, pd.Series],
        y_data: Union[List[float], np.ndarray, pd.Series],
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
        figsize: Tuple[int, int] = (10, 7),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a scatter plot using Matplotlib.

        Parameters
        ----------
        x_data : Union[List[float], np.ndarray, pd.Series]
            Data for x-axis.
        y_data : Union[List[float], np.ndarray, pd.Series]
            Data for y-axis.
        title : str
            Title for the plot.
        x_label : str, optional
            Label for the x-axis.
        y_label : str, optional
            Label for the y-axis.
        add_trendline : bool, default False
            Whether to add a linear regression trendline.
        correlation : float, optional
            Correlation coefficient to display (if provided).
        method : str, optional
            Method used for correlation (e.g., "Pearson", "Spearman").
        hover_text : List[str], optional
            Custom hover text for each point (not supported in Matplotlib).
        marker_size : Union[List[float], float], optional
            Size(s) for markers, either single value or list of values.
        color_scale : Union[List[str], str], optional
            Color scale for markers when using color_values.
        color_values : List[float], optional
            Values used to color the markers.
        figsize : Tuple[int, int], default (10, 7)
            Figure size.
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting).
        theme : Optional[str]
            Theme to use for the visualization.
        strict : bool, default False
            If True, raise exceptions for invalid configuration; otherwise log warnings.
        **kwargs :
            Additional plotting parameters.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib scatter plot figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                x, y = self._prepare_scatter_data(x_data, y_data)
                if len(x) == 0 or len(y) == 0:
                    logger.warning("No valid data points for scatter plot")
                    return self.create_empty_figure(
                        title=title,
                        message="No valid data points for scatter plot",
                        figsize=figsize,
                    )

                fig, ax = plt.subplots(figsize=figsize)
                colors = get_theme_colors(1)
                marker_color = kwargs.get("marker_color", colors[0])

                # Handle marker size
                if marker_size is not None:
                    s = marker_size
                else:
                    s = 40

                # Handle color values
                if color_values is not None:
                    c = color_values
                    cmap = color_scale if color_scale is not None else "viridis"
                else:
                    c = marker_color
                    cmap = None

                sc = ax.scatter(
                    x, y, s=s, c=c, cmap=cmap, alpha=0.8, edgecolors="w", **kwargs
                )

                # Add colorbar if color_values is used
                if color_values is not None:
                    cb = fig.colorbar(sc, ax=ax)
                    cb.set_label(kwargs.get("colorbar_title", ""))

                # Add trendline if requested
                if add_trendline:
                    slope, intercept, x_range, y_range = self._calculate_trendline(x, y)
                    if x_range is not None and y_range is not None:
                        ax.plot(
                            x_range,
                            y_range,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="Trendline",
                        )

                # Add correlation annotation if provided
                if correlation is not None:
                    corr_text = f"{method or 'Correlation'}: {correlation:.3f}"
                    ax.annotate(
                        corr_text,
                        xy=(0.05, 0.95),
                        xycoords="axes fraction",
                        fontsize=12,
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", ec="black", lw=1
                        ),
                        ha="left",
                        va="top",
                    )

                ax.set_title(title)
                ax.set_xlabel(x_label or "X")
                ax.set_ylabel(y_label or "Y")
                ax.grid(True)
                ax.legend(loc="best", frameon=True)
                fig = apply_theme_to_matplotlib_figure(fig)
                plt.tight_layout()
                return fig

            except ImportError as imp_error:
                # Define plt as None to ensure it's defined in except block
                plt = None
                logger.error(
                    f"Matplotlib is not available. Please install it with: pip install matplotlib. Error: {imp_error}"
                )
                return None
            except Exception as e:
                logger.error(f"Error creating scatter plot with Matplotlib: {e}")
                try:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=figsize)
                    ax.text(
                        0.5,
                        0.5,
                        f"Error creating visualization: {str(e)}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.set_title(title)
                    ax.axis("off")
                    return fig
                except:
                    return None

    def update(
        self,
        fig: Any,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Update an existing Matplotlib scatter plot.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Existing Matplotlib figure to update.
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting).
        theme : Optional[str]
            Theme to use for the visualization.
        strict : bool, default False
            If True, raise exceptions for invalid configuration; otherwise log warnings.
        **kwargs :
            Update parameters (same as in create).

        Returns
        -------
        matplotlib.figure.Figure
            Updated Matplotlib figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                if not hasattr(fig, "axes") or len(fig.axes) == 0:
                    logger.warning(
                        "Cannot update non-Matplotlib figure with MatplotlibScatterPlot"
                    )
                    return fig
                ax = fig.axes[0]

                # Update title and labels
                if "title" in kwargs:
                    ax.set_title(kwargs["title"])
                if "x_label" in kwargs:
                    ax.set_xlabel(kwargs["x_label"])
                if "y_label" in kwargs:
                    ax.set_ylabel(kwargs["y_label"])

                # Update data if provided
                if "x_data" in kwargs and "y_data" in kwargs:
                    x_data = kwargs["x_data"]
                    y_data = kwargs["y_data"]
                    x, y = self._prepare_scatter_data(x_data, y_data)

                    # Remove all collections (scatter plots)
                    ax.cla()

                    # Redraw scatter
                    marker_size = kwargs.get("marker_size", 40)
                    color_values = kwargs.get("color_values", None)
                    color_scale = kwargs.get("color_scale", None)
                    marker_color = kwargs.get("marker_color", get_theme_colors(1)[0])

                    if color_values is not None:
                        c = color_values
                        cmap = color_scale if color_scale is not None else "viridis"
                    else:
                        c = marker_color
                        cmap = None

                    sc = ax.scatter(
                        x, y, s=marker_size, c=c, cmap=cmap, alpha=0.8, edgecolors="w"
                    )

                    if color_values is not None:
                        cb = fig.colorbar(sc, ax=ax)
                        cb.set_label(kwargs.get("colorbar_title", ""))

                    # Add trendline if requested
                    add_trendline = kwargs.get("add_trendline", False)
                    if add_trendline:
                        slope, intercept, x_range, y_range = self._calculate_trendline(
                            x, y
                        )
                        if x_range is not None and y_range is not None:
                            ax.plot(
                                x_range,
                                y_range,
                                color="red",
                                linestyle="--",
                                linewidth=2,
                                label="Trendline",
                            )

                    # Add correlation annotation if provided
                    correlation = kwargs.get("correlation", None)
                    method = kwargs.get("method", None)
                    if correlation is not None:
                        corr_text = f"{method or 'Correlation'}: {correlation:.3f}"
                        ax.annotate(
                            corr_text,
                            xy=(0.05, 0.95),
                            xycoords="axes fraction",
                            fontsize=12,
                            bbox=dict(
                                boxstyle="round,pad=0.3", fc="white", ec="black", lw=1
                            ),
                            ha="left",
                            va="top",
                        )

                    ax.set_xlabel(kwargs.get("x_label", "X"))
                    ax.set_ylabel(kwargs.get("y_label", "Y"))
                    ax.set_title(kwargs.get("title", ""))
                    ax.grid(True)
                    ax.legend(loc="best", frameon=True)

                fig = apply_theme_to_matplotlib_figure(fig)
                plt.tight_layout()
                return fig

            except ImportError as imp_error:
                # Define plt as None to ensure it's defined in except block
                plt = None
                logger.error(
                    f"Matplotlib is not available for updating the figure. Error: {imp_error}"
                )
                return fig
            except Exception as e:
                logger.error(f"Error updating scatter plot (Matplotlib): {e}")
                return fig


# Register plot implementations
FigureRegistry.register("scatter", "plotly", PlotlyScatterPlot)
FigureRegistry.register("scatter", "matplotlib", MatplotlibScatterPlot)
