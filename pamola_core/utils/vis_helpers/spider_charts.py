"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Spider Chart Visualization Implementation
Description: Thread-safe spider/radar chart visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for spider/radar charts primarily using
the Plotly backend with context-isolated state management for concurrent
visualization operations.

Key features:
1. Context-isolated visualization state for thread-safe parallel execution
2. Support for multiple data formats (dict, DataFrame)
3. Customizable chart appearance with fill options and gridlines
4. Flexible angle configuration and normalization options
5. Thread-safe registry operations
6. Proper theme application using context-aware helpers
7. Comprehensive error handling with appropriate fallbacks
8. Memory-efficient operation with explicit resource cleanup

Implementation follows the PAMOLA.CORE framework with standardized interfaces for
visualization generation while ensuring concurrent operations do not interfere
with each other through proper context isolation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from pamola_core.utils.vis_helpers.base import MatplotlibFigure, PlotlyFigure, FigureRegistry
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_matplotlib_figure,
    apply_theme_to_plotly_figure,
    get_theme_colors,
)
from pamola_core.utils.vis_helpers.context import visualization_context

# Configure logger
logger = logging.getLogger(__name__)


class PlotlySpiderChart(PlotlyFigure):
    """Spider chart (radar chart) implementation using Plotly."""

    def create(
        self,
        data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
        title: str,
        categories: Optional[List[str]] = None,
        normalize_values: bool = True,
        fill_area: bool = True,
        show_gridlines: bool = True,
        angle_start: float = 90,
        show_legend: bool = True,
        spider_type: str = "scatterpolar",  # 'scatterpolar' or 'barpolar'
        max_value: Optional[float] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a spider/radar chart using Plotly.

        Parameters:
        -----------
        data : Union[Dict[str, Dict[str, float]], pd.DataFrame]
            Data to visualize.
            If dict, the outer dict keys are series names (for legend),
            inner dict keys are categories (radar axes).
            If DataFrame, columns are categories, index values are series names.
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
        spider_type : str, optional
            Type of spider chart: 'scatterpolar' (default) or 'barpolar'
        max_value : float, optional
            Maximum value for radial axis, if None will be determined from data
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional arguments to pass to the Plotly trace

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the spider chart
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import plotly.graph_objects as go

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                # Convert data to DataFrame if it's a dict
                if isinstance(data, dict):
                    df = pd.DataFrame(
                        {series: pd.Series(values) for series, values in data.items()}
                    ).T
                elif isinstance(data, pd.DataFrame):
                    df = data.copy()
                else:
                    raise TypeError(
                        f"Unsupported data type for spider chart: {type(data)}"
                    )

                # If no categories specified, use all columns
                if categories is None:
                    categories = list(df.columns)
                else:
                    # Filter to include only the specified categories
                    df = df[categories]

                # Get series names from DataFrame index
                series_names = list(df.index)

                # Normalize values if requested
                if normalize_values:
                    if max_value is None:
                        # Normalize each category to 0-1 range
                        for category in categories:
                            if df[category].max() > 0:  # Avoid division by zero
                                df[category] = df[category] / df[category].max()
                    else:
                        # Normalize using provided max value
                        df = df / max_value

                    # Set max radial value to 1 if normalized
                    radial_range = [0, 1]
                else:
                    # Determine max value for radial axis
                    if max_value is None:
                        max_val = df.max().max()
                        # Round up to next nice number for readability
                        max_val = np.ceil(max_val * 1.1)
                    else:
                        max_val = max_value

                    radial_range = [0, max_val]

                # Create figure
                fig = go.Figure()

                # Register figure for cleanup
                register_figure(fig, context_info)

                # Get colors from theme
                colors = get_theme_colors(len(series_names))

                # Create radar traces for each series
                for i, series in enumerate(series_names):
                    # Get values for this series
                    values = df.loc[series].tolist()

                    # Close the loop by repeating the first value
                    values_closed = values + [values[0]]
                    categories_closed = categories + [categories[0]]

                    # Determine color for this series
                    color = kwargs.get(
                        f"color_{series}", kwargs.get("color", colors[i % len(colors)])
                    )

                    # Set up trace parameters
                    trace_params = {
                        "r": values_closed,
                        "theta": categories_closed,
                        "name": series,
                        "line": dict(color=color, width=kwargs.get("line_width", 2)),
                    }

                    # Add fill if requested
                    if fill_area:
                        trace_params["fill"] = "toself"
                        r, g, b, _ = to_rgba(color)
                        trace_params["fillcolor"] = (
                            f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.2)"
                        )

                    # Add any additional parameters
                    for key, value in kwargs.items():
                        if key not in [
                            "title",
                            "color",
                            "line_width",
                            "height",
                            "width",
                        ]:
                            trace_params[key] = value

                    # Add the trace based on spider type
                    if spider_type.lower() == "barpolar":
                        fig.add_trace(go.Barpolar(**trace_params))
                    else:  # Default to scatterpolar
                        fig.add_trace(go.Scatterpolar(**trace_params))

                # Configure the layout
                fig.update_layout(
                    title=title,
                    polar=dict(
                        radialaxis=dict(visible=True, range=radial_range),
                        angularaxis=dict(direction="clockwise", rotation=angle_start),
                    ),
                    showlegend=show_legend,
                    height=kwargs.get("height", 600),
                    width=kwargs.get("width", 600),
                )

                # Configure grid visibility
                fig.update_polars(
                    radialaxis_showgrid=show_gridlines,
                    angularaxis_showgrid=show_gridlines,
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
                logger.error(f"Error creating spider chart: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating spider chart: {str(e)}"
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
        Update an existing Plotly spider chart.

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
        **kwargs:
            Parameters to update (same as create method)

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated Plotly figure
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import plotly.graph_objects as go

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Validate figure type
                if not isinstance(fig, go.Figure):
                    logger.warning(
                        "Cannot update non-Plotly figure with PlotlySpiderChart"
                    )
                    return fig

                # Update title if provided
                if "title" in kwargs:
                    fig.update_layout(title=kwargs["title"])

                # Update radial axis range if max_value provided
                if "max_value" in kwargs:
                    max_val = kwargs["max_value"]
                    fig.update_layout(polar=dict(radialaxis=dict(range=[0, max_val])))

                # Update gridlines if requested
                if "show_gridlines" in kwargs:
                    show_gridlines = kwargs["show_gridlines"]
                    fig.update_polars(
                        radialaxis_showgrid=show_gridlines,
                        angularaxis_showgrid=show_gridlines,
                    )

                # Update legend visibility if requested
                if "show_legend" in kwargs:
                    fig.update_layout(showlegend=kwargs["show_legend"])

                # Update dimensions if requested
                if "height" in kwargs or "width" in kwargs:
                    height = kwargs.get("height", fig.layout.height)
                    width = kwargs.get("width", fig.layout.width)
                    fig.update_layout(height=height, width=width)

                # Update data if provided
                if "data" in kwargs:
                    # This is more complex as we need to recreate traces
                    # Rather than trying to update in place, we'll create a new figure
                    # and copy over the relevant parts of the layout

                    # Get current layout settings to preserve
                    current_title = fig.layout.title.text if fig.layout.title else None
                    current_height = fig.layout.height
                    current_width = fig.layout.width
                    current_showlegend = fig.layout.showlegend

                    # Create new figure
                    new_kwargs = kwargs.copy()
                    if current_title and "title" not in new_kwargs:
                        new_kwargs["title"] = current_title
                    if current_height and "height" not in new_kwargs:
                        new_kwargs["height"] = current_height
                    if current_width and "width" not in new_kwargs:
                        new_kwargs["width"] = current_width
                    if (
                        current_showlegend is not None
                        and "show_legend" not in new_kwargs
                    ):
                        new_kwargs["show_legend"] = current_showlegend

                    return self.create(**new_kwargs)

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except Exception as e:
                logger.error(f"Error updating spider chart: {e}")
                return fig  # Return original figure as fallback


class MatplotlibSpiderChart(MatplotlibFigure):
    """
    Spider chart (radar chart) implementation using Matplotlib.
    """

    def create(
        self,
        data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
        title: str,
        categories: Optional[List[str]] = None,
        normalize_values: bool = True,
        fill_area: bool = True,
        show_gridlines: bool = True,
        angle_start: float = 90,
        show_legend: bool = True,
        max_value: Optional[float] = None,
        figsize: Tuple[int, int] = (8, 8),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a spider/radar chart using Matplotlib.

        Parameters
        ----------
        data : Union[Dict[str, Dict[str, float]], pd.DataFrame]
            Data to visualize.
            If dict, the outer dict keys are series names (for legend),
            inner dict keys are categories (radar axes).
            If DataFrame, columns are categories, index values are series names.
        title : str
            Title for the plot.
        categories : List[str], optional
            List of categories to include (if None, all categories in data will be used).
        normalize_values : bool, optional
            Whether to normalize values to 0-1 range for each category.
        fill_area : bool, optional
            Whether to fill the area under the radar lines.
        show_gridlines : bool, optional
            Whether to show gridlines on the radar.
        angle_start : float, optional
            Starting angle for the first axis in degrees (90 = top).
        show_legend : bool, optional
            Whether to show the legend.
        max_value : float, optional
            Maximum value for radial axis, if None will be determined from data.
        figsize : Tuple[int, int], optional
            Figure size.
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting).
        theme : Optional[str]
            Theme to use for the visualization.
        strict : bool, optional
            If True, raise exceptions for invalid configuration; otherwise log warnings.
        **kwargs :
            Additional plotting parameters.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib spider/radar chart figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Prepare data
                if isinstance(data, dict):
                    df = pd.DataFrame(data).T
                elif isinstance(data, pd.DataFrame):
                    df = data.copy()
                else:
                    raise TypeError(
                        f"Unsupported data type for spider chart: {type(data)}"
                    )

                if categories is None:
                    categories = list(df.columns)
                else:
                    df = df[categories]

                series_names = list(df.index)

                # Normalize if requested
                if normalize_values:
                    if max_value is None:
                        for cat in categories:
                            max_cat = df[cat].max()
                            if max_cat > 0:
                                df[cat] = df[cat] / max_cat
                        radial_max = 1
                    else:
                        df = df / max_value
                        radial_max = 1
                else:
                    if max_value is None:
                        radial_max = np.ceil(df.max().max() * 1.1)
                    else:
                        radial_max = max_value

                # Angles for axes
                num_vars = len(categories)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1]  # close the loop

                # Setup figure
                fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
                colors = get_theme_colors(len(series_names))

                # Draw gridlines
                if show_gridlines:
                    ax.yaxis.grid(True, color="gray", linestyle="dotted", linewidth=0.7)
                    ax.xaxis.grid(True, color="gray", linestyle="dotted", linewidth=0.7)
                else:
                    ax.yaxis.grid(False)
                    ax.xaxis.grid(False)

                # Draw each series
                for i, series in enumerate(series_names):
                    values = df.loc[series].tolist()
                    values += values[:1]  # close the loop
                    color = kwargs.get(
                        f"color_{series}", kwargs.get("color", colors[i % len(colors)])
                    )
                    line_width = kwargs.get("line_width", 2)
                    ax.plot(
                        angles, values, label=series, color=color, linewidth=line_width
                    )
                    if fill_area:
                        ax.fill(angles, values, color=color, alpha=0.2)

                # Set category labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_yticks(np.linspace(0, radial_max, 5))
                ax.set_ylim(0, radial_max)
                ax.set_title(title, y=1.08)
                if show_legend:
                    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
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
                logger.error(f"Error creating spider chart with Matplotlib: {e}")
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
        Update an existing Matplotlib spider chart.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Existing Matplotlib figure to update.
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting).
        theme : Optional[str]
            Theme to use for the visualization.
        strict : bool, optional
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
                        "Cannot update non-Matplotlib figure with MatplotlibSpiderChart"
                    )
                    return fig
                ax = fig.axes[0]

                # Update title if provided
                if "title" in kwargs:
                    ax.set_title(kwargs["title"])

                # Update data if provided
                if "data" in kwargs:
                    data = kwargs["data"]
                    categories = kwargs.get("categories", None)
                    # Remove old plot
                    ax.cla()
                    # Redraw with new data
                    self.create(
                        data=data,
                        title=kwargs.get("title", ""),
                        categories=categories,
                        normalize_values=kwargs.get("normalize_values", True),
                        fill_area=kwargs.get("fill_area", True),
                        show_gridlines=kwargs.get("show_gridlines", True),
                        angle_start=kwargs.get("angle_start", 90),
                        show_legend=kwargs.get("show_legend", True),
                        max_value=kwargs.get("max_value", None),
                        figsize=kwargs.get("figsize", (8, 8)),
                        backend=backend,
                        theme=theme,
                        strict=strict,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k
                            not in [
                                "data",
                                "categories",
                                "title",
                                "normalize_values",
                                "fill_area",
                                "show_gridlines",
                                "angle_start",
                                "show_legend",
                                "max_value",
                                "figsize",
                            ]
                        },
                    )
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
                logger.error(f"Error updating spider chart (Matplotlib): {e}")
                return fig


# Register the implementation
FigureRegistry.register("spider", "plotly", PlotlySpiderChart)
FigureRegistry.register("spider", "matplotlib", MatplotlibSpiderChart)
