"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Bar Plot Visualization Implementation
Description: Thread-safe bar plot visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for bar plots using both
Plotly and Matplotlib backends. Plotly is the primary implementation,
while Matplotlib serves as a fallback when needed.

The implementation uses contextvars via the visualization_context
to ensure thread-safe operation for concurrent execution contexts.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

from pamola_core.utils.vis_helpers.base import (
    PlotlyFigure,
    MatplotlibFigure,
    FigureRegistry,
    ensure_series,
    sort_series,
)
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_plotly_figure,
    apply_theme_to_matplotlib_figure,
    get_theme_colors,
)
from pamola_core.utils.vis_helpers.context import visualization_context

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyBarPlot(PlotlyFigure):
    """Bar plot implementation using Plotly (primary implementation)."""

    def create(
        self,
        data: Union[Dict[str, Any], pd.Series],
        title: str,
        orientation: str = "v",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        sort_by: str = "value",
        max_items: int = 15,
        show_values: bool = True,
        text: Optional[Union[List[str], pd.Series]] = None,
        colorscale: Optional[str] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a bar plot using Plotly.

        Parameters:
        -----------
        data : Union[Dict[str, Any], pd.Series]
            Data to visualize
        title : str
            Title for the plot
        orientation : str, optional
            Bar orientation ('v' for vertical, 'h' for horizontal)
        x_label : Optional[str]
            Label for x-axis
        y_label : Optional[str]
            Label for y-axis
        sort_by : str, optional
            Sort method ('value' or 'key')
        max_items : int, optional
            Maximum number of items to display
        show_values : bool, optional
            Whether to show values on bars
        text : Optional[Union[List[str], pd.Series]]
            Custom text for bars
        colorscale : Optional[str]
            Color scale for bars
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly bar plot figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Handle empty data
                if not data:
                    return self.create_empty_figure(
                        title=title, message="No data available for visualization"
                    )

                # Ensure data is a pandas Series
                try:
                    series = ensure_series(data)
                except TypeError as e:
                    logger.error(f"Error converting data to Series: {e}")
                    return self.create_empty_figure(
                        title=title, message=f"Error processing data: {str(e)}"
                    )

                # Handle single data point
                if len(series) == 1:
                    logger.info("Only one data point available for bar plot")

                # Sort the data
                sort_ascending = kwargs.get(
                    "sort_ascending", False if sort_by == "value" else True
                )
                try:
                    series = sort_series(
                        series,
                        sort_by=sort_by,
                        ascending=sort_ascending,
                        max_items=max_items,
                    )
                except Exception as e:
                    logger.warning(f"Error sorting data: {e}. Using unsorted data.")

                # Prepare bar text
                if text is not None:
                    # Ensure text matches series index
                    if isinstance(text, pd.Series):
                        text = text.loc[series.index]
                    bar_text = text
                elif show_values:
                    # Format values as text
                    bar_text = series.apply(
                        lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
                    )
                else:
                    bar_text = None

                # Create figure
                fig = go.Figure()

                # Add bar trace based on orientation
                trace_params: Dict[str, Any] = {
                    "text": bar_text,
                    "textposition": "auto",
                }

                # Apply color if specified
                if colorscale:
                    trace_params["marker"] = {
                        "color": series.values,
                        "colorscale": colorscale,
                    }
                elif "color" in kwargs:
                    trace_params["marker"] = {"color": kwargs.pop("color")}

                # Add any other kwargs to trace_params
                for key, value in kwargs.items():
                    if key not in ["sort_ascending", "figsize"]:
                        trace_params[key] = value

                if orientation == "v":
                    fig.add_trace(
                        go.Bar(x=series.index, y=series.values, **trace_params)
                    )
                    # Set vertical axis labels
                    fig.update_layout(
                        xaxis_title=x_label or "Category",
                        yaxis_title=y_label or "Value",
                    )
                else:  # horizontal
                    fig.add_trace(
                        go.Bar(
                            x=series.values,
                            y=series.index,
                            orientation="h",
                            **trace_params,
                        )
                    )
                    # Set horizontal axis labels
                    fig.update_layout(
                        xaxis_title=y_label or "Value",
                        yaxis_title=x_label or "Category",
                    )

                # Set title and apply theme
                fig.update_layout(
                    title=title,
                    height=kwargs.get("height", 600),
                    width=kwargs.get("width", 800),
                )
                fig = apply_theme_to_plotly_figure(fig)

                return fig

            except ImportError as imp_error:
                # Define go as None to ensure it's defined in except block
                go = None
                logger.error(
                    f"Plotly is not available. Please install it with: pip install plotly. Error: {imp_error}"
                )
                # Try to use matplotlib as fallback
                fallback = MatplotlibBarPlot()
                logger.warning("Falling back to Matplotlib implementation")
                return fallback.create(
                    data=data,
                    title=title,
                    orientation=orientation,
                    x_label=x_label,
                    y_label=y_label,
                    sort_by=sort_by,
                    max_items=max_items,
                    show_values=show_values,
                    backend=backend,
                    theme=theme,
                    strict=strict,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error creating bar plot with Plotly: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating visualization: {str(e)}"
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
        Update an existing Plotly bar plot.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Existing Plotly figure to update
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Update parameters

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated Plotly figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Validate figure type
                if not isinstance(fig, go.Figure):
                    logger.warning("Cannot update non-Plotly figure with PlotlyBarPlot")
                    return fig

                # Update layout elements if provided
                layout_updates = {}
                if "title" in kwargs:
                    layout_updates["title"] = kwargs["title"]
                if "x_label" in kwargs:
                    layout_updates["xaxis_title"] = kwargs["x_label"]
                if "y_label" in kwargs:
                    layout_updates["yaxis_title"] = kwargs["y_label"]
                if "height" in kwargs:
                    layout_updates["height"] = kwargs["height"]
                if "width" in kwargs:
                    layout_updates["width"] = kwargs["width"]

                if layout_updates:
                    fig.update_layout(**layout_updates)

                # Update data if provided
                if "data" in kwargs and len(fig.data) > 0:
                    data = kwargs["data"]

                    try:
                        series = ensure_series(data)

                        # Sort data if requested
                        if "sort_by" in kwargs:
                            sort_by = kwargs["sort_by"]
                            sort_ascending = kwargs.get(
                                "sort_ascending", False if sort_by == "value" else True
                            )
                            max_items = kwargs.get("max_items", len(series))
                            series = sort_series(
                                series,
                                sort_by=sort_by,
                                ascending=sort_ascending,
                                max_items=max_items,
                            )

                        # Determine orientation
                        orientation = kwargs.get("orientation", "v")

                        # Update trace data
                        if orientation == "v":
                            fig.update_traces(x=series.index, y=series.values)
                        else:
                            fig.update_traces(x=series.values, y=series.index)

                        # Update text if show_values is True
                        if kwargs.get("show_values", True):
                            bar_text = series.apply(
                                lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
                            )
                            fig.update_traces(text=bar_text, textposition="auto")

                    except Exception as e:
                        logger.error(f"Error updating bar plot data: {e}")

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
                logger.error(f"Error updating bar plot: {e}")
                return fig


class MatplotlibBarPlot(MatplotlibFigure):
    """
    Bar plot implementation using Matplotlib.

    Note: This is a fallback implementation. The primary implementation uses Plotly.
    """

    def create(
        self,
        data: Union[Dict[str, Any], pd.Series],
        title: str,
        orientation: str = "v",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        sort_by: str = "value",
        max_items: int = 15,
        show_values: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a bar plot using Matplotlib.

        Parameters:
        -----------
        data : Union[Dict[str, Any], pd.Series]
            Data to visualize
        title : str
            Plot title
        orientation : str, optional
            Bar orientation ('v' for vertical, 'h' for horizontal)
        x_label : Optional[str]
            X-axis label
        y_label : Optional[str]
            Y-axis label
        sort_by : str, optional
            Sort method ('value' or 'key')
        max_items : int, optional
            Maximum items to display
        show_values : bool, optional
            Whether to show bar values
        figsize : Tuple[int, int], optional
            Figure dimensions
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib bar plot figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Handle empty data
                if not data:
                    return self.create_empty_figure(
                        title=title,
                        message="No data available for visualization",
                        figsize=figsize,
                    )

                # Ensure data is a pandas Series
                try:
                    series = ensure_series(data)

                    # Filter out values that are not comparable (e.g., dict, list, set)
                    series = series[series.apply(lambda v: isinstance(v, (int, float, str, bool)))]

                    # Convert index to string if it contains unhashable types like dict
                    if any(isinstance(idx, dict) for idx in series.index):
                        series.index = series.index.map(str)

                except TypeError as e:
                    logger.error(f"Error converting data to Series: {e}")
                    return self.create_empty_figure(
                        title=title,
                        message=f"Error processing data: {str(e)}",
                        figsize=figsize,
                    )

                # Handle single data point
                if len(series) == 1:
                    logger.info("Only one data point available for bar plot")

                # Sort the data
                sort_ascending = kwargs.get(
                    "sort_ascending", False if sort_by == "value" else True
                )
                try:
                    series = sort_series(
                        series,
                        sort_by=sort_by,
                        ascending=sort_ascending,
                        max_items=max_items,
                    )
                except Exception as e:
                    logger.warning(f"Error sorting data: {e}. Using unsorted data.")

                # Create figure and axes
                fig, ax = plt.subplots(figsize=figsize)

                # Get color from theme or kwargs
                colors = get_theme_colors(1)
                color = kwargs.pop("color", colors[0])

                # Filter out kwargs that are specific to Plotly
                plt_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in ["text", "textposition", "colorscale", "width", "height"]
                }

                # Create bar plot based on orientation
                if orientation == "v":
                    bars = ax.bar(
                        series.index, series.values, color=color, **plt_kwargs
                    )
                    ax.set_xlabel(x_label or "Category")
                    ax.set_ylabel(y_label or "Value")
                    plt.xticks(rotation=45, ha="right")

                    # Add value labels
                    if show_values:
                        for bar in bars:
                            height = bar.get_height()
                            if not np.isnan(height):  # Skip NaN values
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01 * max(series.values),
                                    (
                                        f"{height:.1f}"
                                        if isinstance(height, float)
                                        else f"{height}"
                                    ),
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )
                else:  # horizontal
                    bars = ax.barh(
                        series.index, series.values, color=color, **plt_kwargs
                    )
                    ax.set_xlabel(y_label or "Value")
                    ax.set_ylabel(x_label or "Category")

                    # Add value labels
                    if show_values:
                        for bar in bars:
                            width = bar.get_width()
                            if not np.isnan(width):  # Skip NaN values
                                ax.text(
                                    width + 0.01 * max(series.values),
                                    bar.get_y() + bar.get_height() / 2.0,
                                    (
                                        f"{width:.1f}"
                                        if isinstance(width, float)
                                        else f"{width}"
                                    ),
                                    ha="left",
                                    va="center",
                                    fontsize=9,
                                )

                # Set title and finalize
                ax.set_title(title)
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
                logger.error(f"Error creating bar plot with Matplotlib: {e}")
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
        Update an existing Matplotlib bar plot.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Existing Matplotlib figure to update
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Update parameters

        Returns:
        --------
        matplotlib.figure.Figure
            Updated Matplotlib figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Validate figure type
                if not hasattr(fig, "axes"):
                    logger.warning(
                        "Cannot update non-Matplotlib figure with MatplotlibBarPlot"
                    )
                    return fig

                # Ensure figure has axes
                if len(fig.axes) == 0:
                    logger.warning("Figure has no axes to update")
                    return fig

                ax = fig.axes[0]

                # Update layout elements
                if "title" in kwargs:
                    ax.set_title(kwargs["title"])
                if "x_label" in kwargs:
                    ax.set_xlabel(kwargs["x_label"])
                if "y_label" in kwargs:
                    ax.set_ylabel(kwargs["y_label"])

                # Update data if provided
                if (
                    "data" in kwargs
                    and hasattr(ax, "containers")
                    and len(ax.containers) > 0
                ):
                    data = kwargs["data"]

                    try:
                        series = ensure_series(data)

                        # Sort if requested
                        if "sort_by" in kwargs:
                            sort_by = kwargs["sort_by"]
                            sort_ascending = kwargs.get(
                                "sort_ascending", False if sort_by == "value" else True
                            )
                            max_items = kwargs.get("max_items", len(series))
                            series = sort_series(
                                series,
                                sort_by=sort_by,
                                ascending=sort_ascending,
                                max_items=max_items,
                            )

                        # Determine orientation
                        orientation = kwargs.get("orientation", "v")

                        # Get bar container
                        bars = ax.containers[0]

                        # Update bars
                        if orientation == "v":
                            for i, (idx, val) in enumerate(series.items()):
                                if i < len(bars):
                                    bars[i].set_height(val)
                            ax.set_xticks(range(len(series)))
                            ax.set_xticklabels(series.index)
                        else:
                            for i, (idx, val) in enumerate(series.items()):
                                if i < len(bars):
                                    bars[i].set_width(val)
                            ax.set_yticks(range(len(series)))
                            ax.set_yticklabels(series.index)

                        # Update value annotations if requested
                        if kwargs.get("show_values", True):
                            # Remove existing annotations
                            for txt in ax.texts:
                                txt.remove()

                            # Add new annotations
                            if orientation == "v":
                                for i, (bar, (idx, val)) in enumerate(
                                    zip(bars, series.items())
                                ):
                                    ax.text(
                                        bar.get_x() + bar.get_width() / 2.0,
                                        val + 0.01 * max(series.values),
                                        (
                                            f"{val:.1f}"
                                            if isinstance(val, float)
                                            else f"{val}"
                                        ),
                                        ha="center",
                                        va="bottom",
                                        fontsize=9,
                                    )
                            else:
                                for i, (bar, (idx, val)) in enumerate(
                                    zip(bars, series.items())
                                ):
                                    ax.text(
                                        val + 0.01 * max(series.values),
                                        bar.get_y() + bar.get_height() / 2.0,
                                        (
                                            f"{val:.1f}"
                                            if isinstance(val, float)
                                            else f"{val}"
                                        ),
                                        ha="left",
                                        va="center",
                                        fontsize=9,
                                    )

                    except Exception as e:
                        logger.error(f"Error updating bar plot data: {e}")

                # Finalize
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
                logger.error(f"Error updating bar plot: {e}")
                return fig


# Register plot implementations
FigureRegistry.register("bar", "plotly", PlotlyBarPlot)
FigureRegistry.register("bar", "matplotlib", MatplotlibBarPlot)
