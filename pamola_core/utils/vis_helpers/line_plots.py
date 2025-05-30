"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Line Plot Visualization
Description: Thread-safe line plot visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for line plots primarily using the Plotly backend
with context-isolated state management for concurrent visualization operations.

Key features:
1. Context-isolated visualization state for thread-safe parallel execution
2. Support for multiple data formats (dict, DataFrame, Series)
3. Flexible styling options (markers, smooth lines, area fills)
4. Region highlighting capabilities for visual annotations
5. Proper theme application using context-aware helpers
6. Datetime index support for time series data
7. Comprehensive error handling with appropriate fallbacks
8. Memory-efficient operation with explicit resource cleanup

Implementation follows the PAMOLA.CORE framework with standardized interfaces for
visualization generation while ensuring concurrent operations do not interfere
with each other through proper context isolation.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pandas import DatetimeIndex

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


def prepare_data_for_lineplot(
    data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
    x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
) -> Tuple[pd.DataFrame, Union[List, np.ndarray, pd.Series], List[str]]:
    """
    Prepare data for line plot visualization.

    Parameters:
    -----------
    data : Dict[str, List[float]], pd.DataFrame, or pd.Series
        Data to prepare
    x_data : List, np.ndarray, or pd.Series, optional
        Data for the x-axis. If None, indices are used.

    Returns:
    --------
    Tuple[pd.DataFrame, Union[List, np.ndarray, pd.Series], List[str]]
        Tuple containing (processed_data, x_values, series_names)
    """
    try:
        # Convert different input formats to DataFrame
        if isinstance(data, dict):
            # Dictionary with lists - convert to DataFrame
            df = pd.DataFrame(data)
            series_names = list(df.columns)
        elif isinstance(data, pd.Series):
            # Series - convert to single-column DataFrame
            df = pd.DataFrame({data.name or "Value": data})
            series_names = [data.name or "Value"]
        elif isinstance(data, pd.DataFrame):
            # DataFrame - use directly
            df = data
            series_names = list(df.columns)
        else:
            raise TypeError(f"Unsupported data type for line plot: {type(data)}")

        # Determine x values
        if x_data is not None:
            # Use provided x values
            x = x_data
        else:
            # Use indices as x values
            idx = df.index

            # Convert to datetime if indices are DatetimeIndex
            if isinstance(idx, DatetimeIndex):
                x = idx.to_pydatetime()  # DatetimeIndex always has this method
            else:
                x = idx

        return df, x, series_names
    except Exception as e:
        logger.error(f"Error preparing data for line plot: {e}")
        # Return empty but valid objects to avoid further errors
        return pd.DataFrame(), [], []


class PlotlyLinePlot(PlotlyFigure):
    """Line plot implementation using Plotly."""

    def create(
        self,
        data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
        title: str,
        x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        add_markers: bool = True,
        add_area: bool = False,
        smooth: bool = False,
        highlight_regions: Optional[List[Dict[str, Any]]] = None,
        line_width: float = 2.0,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a line plot using Plotly.

        Parameters:
        -----------
        data : Dict[str, List[float]], pd.DataFrame, or pd.Series
            Data to visualize. If Dict, keys are series names and values are y values.
            If DataFrame, each column is a separate line. If Series, a single line.
        title : str
            Title for the plot
        x_data : List, np.ndarray, or pd.Series, optional
            Data for the x-axis. If None, indices are used.
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
        highlight_regions : List[Dict[str, Any]], optional
            List of regions to highlight. Each dict should have 'start', 'end', 'color', and 'label' keys.
        line_width : float
            Width of the lines
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional arguments to pass to go.Scatter

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the line plot
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import plotly.graph_objects as go

                # Process data to a standard format
                processed_data, x, series_names = prepare_data_for_lineplot(
                    data, x_data
                )

                # Create figure
                fig = go.Figure()

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Add traces for each series
                fig = self._add_data_traces(
                    fig,
                    processed_data,
                    x,
                    series_names,
                    add_markers,
                    add_area,
                    smooth,
                    line_width,
                    **kwargs,
                )

                # Add highlighted regions if requested
                if highlight_regions:
                    fig = self._add_highlight_regions(fig, highlight_regions)

                # Configure layout
                fig = self._configure_layout(
                    fig, title, x_label, y_label, len(processed_data.columns)
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
                logger.error(f"Error creating line plot: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating line plot: {str(e)}"
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
        Update an existing Plotly line plot.

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

                # Ensure we have a Plotly figure
                if not isinstance(fig, go.Figure):
                    logger.warning(
                        "Cannot update non-Plotly figure with PlotlyLinePlot"
                    )
                    return fig

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Update basic attributes (title, labels)
                if "title" in kwargs:
                    fig.update_layout(title=kwargs["title"])
                if "x_label" in kwargs:
                    fig.update_layout(xaxis_title=kwargs["x_label"])
                if "y_label" in kwargs:
                    fig.update_layout(yaxis_title=kwargs["y_label"])

                # Update data if provided
                if "data" in kwargs:
                    data = kwargs["data"]
                    x_data = kwargs.get("x_data", None)

                    # Process new data
                    processed_data, x, series_names = prepare_data_for_lineplot(
                        data, x_data
                    )

                    # Get the current trace names for comparison
                    # Use alternative access to trace name via ['name']
                    existing_trace_names = [
                        trace["name"] if "name" in trace else f"trace_{i}"
                        for i, trace in enumerate(fig.data)
                    ]

                    # Get style parameters
                    add_markers = kwargs.get("add_markers", True)
                    add_area = kwargs.get("add_area", False)
                    smooth = kwargs.get("smooth", False)
                    line_width = kwargs.get("line_width", 2.0)

                    # Clear existing traces (simpler than trying to update them)
                    fig.data = []

                    # Add updated traces
                    fig = self._add_data_traces(
                        fig,
                        processed_data,
                        x,
                        series_names,
                        add_markers,
                        add_area,
                        smooth,
                        line_width,
                        **kwargs,
                    )

                # Update highlighted regions if provided
                if "highlight_regions" in kwargs:
                    # Remove existing highlighted regions
                    shapes = []
                    annotations = []

                    # Keep shapes and annotations that aren't related to highlight regions
                    for shape in fig.layout.shapes or []:
                        if not (
                            shape.get("layer") == "below"
                            and shape.get("yref") == "paper"
                            and shape.get("type") == "rect"
                        ):
                            shapes.append(shape)

                    for annotation in fig.layout.annotations or []:
                        if not (
                            annotation.get("yref") == "paper"
                            and annotation.get("y") == 1.05
                        ):
                            annotations.append(annotation)

                    # Update layout without the highlight region shapes and annotations
                    fig.update_layout(shapes=shapes, annotations=annotations)

                    # Add new highlighted regions
                    highlight_regions = kwargs["highlight_regions"]
                    if highlight_regions:
                        fig = self._add_highlight_regions(fig, highlight_regions)

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except ImportError as e:
                logger.error(
                    f"Plotly is not available for updating the figure. Error: {e}"
                )
                return fig
            except Exception as e:
                logger.error(f"Error updating line plot: {e}")
                return fig

    def _add_data_traces(
        self,
        fig: Any,
        processed_data: pd.DataFrame,
        x: Union[List, np.ndarray, pd.Series],
        series_names: List[str],
        add_markers: bool,
        add_area: bool,
        smooth: bool,
        line_width: float,
        **kwargs,
    ) -> Any:
        """
        Add data traces to the figure.

        Parameters:
        -----------
        fig : go.Figure
            Figure to add traces to
        processed_data : pd.DataFrame
            Processed data with each column as a series
        x : Union[List, np.ndarray, pd.Series]
            X-axis data
        series_names : List[str]
            Names for each series
        add_markers : bool
            Whether to add markers to lines
        add_area : bool
            Whether to fill area under lines
        smooth : bool
            Whether to use smooth lines
        line_width : float
            Width of the lines
        **kwargs:
            Additional arguments

        Returns:
        --------
        go.Figure
            Figure with traces added
        """
        try:
            import plotly.graph_objects as go

            # Get colors from theme
            colors = get_theme_colors(len(processed_data.columns))

            # Add a trace for each series
            for i, column in enumerate(processed_data.columns):
                color = kwargs.get(
                    f"color_{column}", kwargs.get("color", colors[i % len(colors)])
                )

                # Determine mode based on markers parameter
                mode = "lines"
                if add_markers:
                    mode += "+markers"

                # Determine line shape based on smooth parameter
                line_shape = "spline" if smooth else "linear"

                # Create trace
                trace_args = {
                    "x": x,
                    "y": processed_data[column],
                    "mode": mode,
                    "name": series_names[i],
                    "line": {"width": line_width, "color": color, "shape": line_shape},
                    "connectgaps": kwargs.get(
                        "connectgaps", True
                    ),  # Connect gaps from missing values
                }

                # Add fill if requested
                if add_area:
                    trace_args["fill"] = "tozeroy"
                    # Create a more transparent version of the color for fill
                    if color.startswith("#"):
                        trace_args["fillcolor"] = f"{color}33"  # Add 20% opacity
                    else:
                        trace_args["fillcolor"] = (
                            f'rgba({color.replace("rgb(", "").replace(")", "")}, 0.2)'
                        )

                # Add custom hover text if provided
                if "hovertext" in kwargs:
                    if isinstance(kwargs["hovertext"], list) and len(
                        kwargs["hovertext"]
                    ) == len(x):
                        trace_args["hovertext"] = kwargs["hovertext"]
                        trace_args["hoverinfo"] = "text"

                # Add the trace to the figure
                fig.add_trace(go.Scatter(**trace_args))

            return fig
        except Exception as e:
            logger.error(f"Error adding data traces: {e}")
            return fig  # Return the figure even if incomplete

    def _add_highlight_regions(
        self, fig: Any, highlight_regions: List[Dict[str, Any]]
    ) -> Any:
        """
        Add highlighted regions to the figure.

        Parameters:
        -----------
        fig : go.Figure
            Figure to add regions to
        highlight_regions : List[Dict[str, Any]]
            List of regions to highlight. Each dict should have 'start', 'end', 'color', and 'label' keys.

        Returns:
        --------
        go.Figure
            Figure with highlight regions added
        """
        try:
            shapes = list(fig.layout.shapes or [])
            annotations = list(fig.layout.annotations or [])

            for region in highlight_regions:
                # Add shaded rectangle for the region
                start = region.get("start")
                end = region.get("end")
                color = region.get("color", "rgba(255, 0, 0, 0.2)")
                label = region.get("label", "")

                if start is not None and end is not None:
                    shapes.append(
                        {
                            "type": "rect",
                            "x0": start,
                            "x1": end,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "fillcolor": color,
                            "opacity": 0.3,
                            "layer": "below",
                            "line_width": 0,
                        }
                    )

                    # Add annotation for the region label
                    if label:
                        annotations.append(
                            {
                                "x": (start + end) / 2,
                                "y": 1.05,
                                "yref": "paper",
                                "text": label,
                                "showarrow": False,
                                "font": dict(size=10),
                            }
                        )

            fig.update_layout(shapes=shapes, annotations=annotations)
            return fig
        except Exception as e:
            logger.error(f"Error adding highlight regions: {e}")
            return fig  # Return the figure even if regions couldn't be added

    def _configure_layout(
        self,
        fig: Any,
        title: str,
        x_label: Optional[str],
        y_label: Optional[str],
        series_count: int,
    ) -> Any:
        """
        Configure figure layout.

        Parameters:
        -----------
        fig : go.Figure
            Figure to configure
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis
        y_label : str, optional
            Label for the y-axis
        series_count : int
            Number of data series

        Returns:
        --------
        go.Figure
            Figure with layout configured
        """
        try:
            # Configure layout
            layout_args = {
                "title": title,
                "xaxis": {"title": x_label, "showgrid": True},
                "yaxis": {"title": y_label, "showgrid": True},
                "legend": {
                    "orientation": "h" if series_count > 3 else "v",
                    "yanchor": "bottom" if series_count > 3 else "top",
                    "y": -0.2 if series_count > 3 else 1,
                    "xanchor": "center" if series_count > 3 else "left",
                    "x": 0.5 if series_count > 3 else 0,
                },
                "hovermode": "closest",
                "margin": dict(t=50, b=50, l=50, r=50),
            }

            fig.update_layout(**layout_args)
            return fig
        except Exception as e:
            logger.error(f"Error configuring layout: {e}")
            return fig  # Return the figure even if layout couldn't be fully configured


class MatplotlibLinePlot(MatplotlibFigure):
    """Line plot implementation using Matplotlib."""

    def create(
        self,
        data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
        title: str,
        x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        add_markers: bool = True,
        add_area: bool = False,
        smooth: bool = False,
        highlight_regions: Optional[List[Dict[str, Any]]] = None,
        line_width: float = 2.0,
        figsize: Tuple[int, int] = (12, 8),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a line plot using Matplotlib.

        Parameters
        ----------
        data : Dict[str, List[float]], pd.DataFrame, or pd.Series
            Data to visualize. If Dict, keys are series names and values are y values.
            If DataFrame, each column is a separate line. If Series, a single line.
        title : str
            Title for the plot.
        x_data : List, np.ndarray, or pd.Series, optional
            Data for the x-axis. If None, indices are used.
        x_label : str, optional
            Label for the x-axis.
        y_label : str, optional
            Label for the y-axis.
        add_markers : bool, default True
            Whether to add markers at data points.
        add_area : bool, default False
            Whether to fill area under lines.
        smooth : bool, default False
            Whether to use spline interpolation for smoother lines (requires scipy).
        highlight_regions : List[Dict[str, Any]], optional
            List of regions to highlight. Each dict should have 'start', 'end', 'color', and 'label' keys.
        line_width : float, default 2.0
            Width of the lines.
        figsize : Tuple[int, int], default (12, 8)
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
            Matplotlib line plot figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Use the shared data preparation logic
                df, x, series_names = prepare_data_for_lineplot(data, x_data)

                if df.empty or len(series_names) == 0:
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.set_title(title)
                    ax.text(
                        0.5,
                        0.5,
                        "No valid data points for line plot",
                        ha="center",
                        va="center",
                    )
                    return fig

                fig, ax = plt.subplots(figsize=figsize)
                colors = get_theme_colors(len(df.columns))

                for i, column in enumerate(df.columns):
                    y = df[column]
                    color = kwargs.get(
                        f"color_{column}", kwargs.get("color", colors[i % len(colors)])
                    )
                    plot_args = {
                        "label": series_names[i],
                        "color": color,
                        "linewidth": line_width,
                    }
                    if add_markers:
                        plot_args["marker"] = "o"
                    if smooth and len(y) > 3:
                        try:
                            from scipy.interpolate import make_interp_spline

                            x_num = np.arange(len(x))
                            spline = make_interp_spline(x_num, y, k=3)
                            x_smooth = np.linspace(x_num.min(), x_num.max(), 300)
                            y_smooth = spline(x_smooth)
                            x_plot = np.interp(x_smooth, x_num, x)
                            ax.plot(x_plot, y_smooth, **plot_args)
                            if add_area:
                                ax.fill_between(
                                    x_plot, y_smooth, color=color, alpha=0.2
                                )
                        except ImportError:
                            ax.plot(x, y, **plot_args)
                            if add_area:
                                ax.fill_between(x, y, color=color, alpha=0.2)
                    else:
                        ax.plot(x, y, **plot_args)
                        if add_area:
                            ax.fill_between(x, y, color=color, alpha=0.2)

                # Highlight regions
                if highlight_regions:
                    ylim = ax.get_ylim()
                    for region in highlight_regions:
                        start = region.get("start")
                        end = region.get("end")
                        color = region.get("color", "red")
                        label = region.get("label", "")
                        if start is not None and end is not None:
                            ax.axvspan(
                                start,
                                end,
                                color=color,
                                alpha=0.3,
                                label=label if label else None,
                            )
                            if label:
                                ax.text(
                                    (start + end) / 2,
                                    ylim[1],
                                    label,
                                    ha="center",
                                    va="bottom",
                                    fontsize=10,
                                )

                ax.set_title(title)
                ax.set_xlabel(x_label or "")
                ax.set_ylabel(y_label or "")
                ax.legend()
                fig = apply_theme_to_matplotlib_figure(fig)
                plt.tight_layout()
                return fig

            except ImportError as imp_error:
                plt = None
                logger.error(
                    f"Matplotlib is not available. Please install it with: pip install matplotlib. Error: {imp_error}"
                )
                return None
            except Exception as e:
                logger.error(f"Error creating line plot with Matplotlib: {e}")
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
        Update an existing Matplotlib line plot.

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
                        "Cannot update non-Matplotlib figure with MatplotlibLinePlot"
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
                if "data" in kwargs:
                    data = kwargs["data"]
                    x_data = kwargs.get("x_data", None)
                    # Use the shared data preparation logic
                    df, x, series_names = prepare_data_for_lineplot(data, x_data)

                    # Remove all lines
                    ax.cla()
                    colors = get_theme_colors(len(df.columns))
                    add_markers = kwargs.get("add_markers", True)
                    add_area = kwargs.get("add_area", False)
                    smooth = kwargs.get("smooth", False)
                    line_width = kwargs.get("line_width", 2.0)

                    for i, column in enumerate(df.columns):
                        y = df[column]
                        color = kwargs.get(
                            f"color_{column}",
                            kwargs.get("color", colors[i % len(colors)]),
                        )
                        plot_args = {
                            "label": series_names[i],
                            "color": color,
                            "linewidth": line_width,
                        }
                        if add_markers:
                            plot_args["marker"] = "o"
                        if smooth and len(y) > 3:
                            try:
                                from scipy.interpolate import make_interp_spline

                                x_num = np.arange(len(x))
                                spline = make_interp_spline(x_num, y, k=3)
                                x_smooth = np.linspace(x_num.min(), x_num.max(), 300)
                                y_smooth = spline(x_smooth)
                                x_plot = np.interp(x_smooth, x_num, x)
                                ax.plot(x_plot, y_smooth, **plot_args)
                                if add_area:
                                    ax.fill_between(
                                        x_plot, y_smooth, color=color, alpha=0.2
                                    )
                            except ImportError:
                                ax.plot(x, y, **plot_args)
                                if add_area:
                                    ax.fill_between(x, y, color=color, alpha=0.2)
                        else:
                            ax.plot(x, y, **plot_args)
                            if add_area:
                                ax.fill_between(x, y, color=color, alpha=0.2)

                    # Highlight regions
                    highlight_regions = kwargs.get("highlight_regions", None)
                    if highlight_regions:
                        ylim = ax.get_ylim()
                        for region in highlight_regions:
                            start = region.get("start")
                            end = region.get("end")
                            color = region.get("color", "red")
                            label = region.get("label", "")
                            if start is not None and end is not None:
                                ax.axvspan(
                                    start,
                                    end,
                                    color=color,
                                    alpha=0.3,
                                    label=label if label else None,
                                )
                                if label:
                                    ax.text(
                                        (start + end) / 2,
                                        ylim[1],
                                        label,
                                        ha="center",
                                        va="bottom",
                                        fontsize=10,
                                    )

                    ax.set_xlabel(kwargs.get("x_label", ""))
                    ax.set_ylabel(kwargs.get("y_label", ""))
                    ax.set_title(kwargs.get("title", ""))
                    ax.legend()

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
                logger.error(f"Error updating line plot: {e}")
                return fig


# Register only Plotly implementation
FigureRegistry.register("line", "plotly", PlotlyLinePlot)
FigureRegistry.register("line", "matplotlib", MatplotlibLinePlot)
