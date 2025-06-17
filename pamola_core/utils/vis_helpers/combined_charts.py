"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
---------------------------------------------------
Module: Combined Chart Visualization Implementation

Description:
    Thread-safe combined chart visualization capabilities using both Plotly (primary) and Matplotlib (fallback) backends.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for charts combining multiple
visualization types (e.g., bar+line, bar+area, etc.) with support
for dual Y-axes. Plotly is the primary implementation,
while Matplotlib serves as a fallback when needed.

The implementation uses contextvars via the visualization_context
to ensure thread-safe operation for concurrent execution contexts.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from pamola_core.utils.vis_helpers.base import (
    MatplotlibFigure,
    PlotlyFigure,
    FigureRegistry,
)
from pamola_core.utils.vis_helpers.context import visualization_context
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_matplotlib_figure,
    apply_theme_to_plotly_figure,
    get_theme_colors,
)

# Configure logger
logger = logging.getLogger(__name__)


def prepare_data_for_combined_chart(
    data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
) -> pd.Series:
    """
    Prepare data for combined chart visualization.

    Parameters:
    -----------
    data : Dict[str, Any], pd.Series, or pd.DataFrame
        Data to process

    Returns:
    --------
    pd.Series
        Processed data as a pandas Series
    """
    if isinstance(data, dict):
        return pd.Series(data)
    elif isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            # Single column DataFrame
            return data.iloc[:, 0]
        else:
            # Multi-column DataFrame - use the first column with warning
            logger.warning(
                f"Multiple columns in DataFrame. Using the first column: {data.columns[0]}"
            )
            return data.iloc[:, 0]
    else:
        raise TypeError(f"Unsupported data type for combined chart: {type(data)}")


class PlotlyCombinedChart(PlotlyFigure):
    """Combined chart implementation using Plotly."""

    def create(
        self,
        primary_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        secondary_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        title: str,
        primary_type: str = "bar",  # "bar", "line", "scatter", "area"
        secondary_type: str = "line",  # "line", "scatter", "area", "bar"
        x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
        x_label: Optional[str] = None,
        primary_y_label: Optional[str] = None,
        secondary_y_label: Optional[str] = None,
        primary_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
        show_grid: bool = True,
        primary_on_right: bool = False,  # Whether to show primary y-axis on right
        vertical_alignment: bool = True,  # Whether to align zero across both axes
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a combined chart with dual Y-axes using Plotly.

        Parameters:
        -----------
        primary_data : Dict[str, Any], pd.Series, or pd.DataFrame
            Data for the primary Y-axis.
        secondary_data : Dict[str, Any], pd.Series, or pd.DataFrame
            Data for the secondary Y-axis.
        title : str
            Title for the plot.
        primary_type : str, optional
            Type of visualization for primary data: "bar", "line", "scatter", "area".
        secondary_type : str, optional
            Type of visualization for secondary data: "line", "scatter", "area", "bar".
        x_data : list, np.ndarray, or pd.Series, optional
            Data for the x-axis. If None, indices are used.
        x_label : str, optional
            Label for the x-axis.
        primary_y_label : str, optional
            Label for the primary Y-axis.
        secondary_y_label : str, optional
            Label for the secondary Y-axis.
        primary_color : str, optional
            Color for the primary series.
        secondary_color : str, optional
            Color for the secondary series.
        show_grid : bool, optional
            Whether to show gridlines on the plot.
        primary_on_right : bool, optional
            Whether to display the primary Y-axis on the right side.
        vertical_alignment : bool, optional
            Whether to align zero values across both axes.
        backend : str, optional
            Backend to use: "plotly" or "matplotlib" (overrides global setting).
        theme : str, optional
            Theme name to use for this visualization.
        strict : bool, optional
            If True, raise exceptions for invalid configuration; otherwise log warnings.
        **kwargs:
            Additional arguments to pass to the underlying plotting function.

        Returns:
        --------
        plotly.graph_objects.Figure
            The resulting Plotly figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Process primary data
                primary_series = prepare_data_for_combined_chart(primary_data)

                # Process secondary data
                secondary_series = prepare_data_for_combined_chart(secondary_data)

                # Determine x values
                if x_data is not None:
                    # Use provided x values
                    x = x_data
                else:
                    # Use indices from primary data as default
                    x = primary_series.index.tolist()

                    # If secondary data has different indices, use a union of both
                    if not primary_series.index.equals(secondary_series.index):
                        all_indices = pd.Index(
                            list(
                                set(primary_series.index) | set(secondary_series.index)
                            )
                        )
                        x = sorted(all_indices)
                        logger.warning(
                            "Primary and secondary data have different indices. Using union of both."
                        )

                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Get colors from theme if not provided
                theme_colors = get_theme_colors(2)
                if primary_color is None:
                    primary_color = theme_colors[0]
                if secondary_color is None:
                    secondary_color = theme_colors[1]

                # Determine which y-axis is on which side
                primary_y_axis = "y2" if primary_on_right else "y"
                secondary_y_axis = "y" if primary_on_right else "y2"

                # Add primary trace
                if primary_type == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=x,
                            y=primary_series,
                            name=kwargs.get(
                                "primary_name", primary_y_label or "Primary"
                            ),
                            marker=dict(
                                color=primary_color,
                                opacity=kwargs.get("primary_opacity", 0.8),
                            ),
                        ),
                        secondary_y=(primary_y_axis == "y2"),
                    )
                elif primary_type == "line":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=primary_series,
                            mode="lines",
                            name=kwargs.get(
                                "primary_name", primary_y_label or "Primary"
                            ),
                            line=dict(
                                color=primary_color,
                                width=kwargs.get("primary_line_width", 2),
                            ),
                        ),
                        secondary_y=(primary_y_axis == "y2"),
                    )
                elif primary_type == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=primary_series,
                            mode="markers",
                            name=kwargs.get(
                                "primary_name", primary_y_label or "Primary"
                            ),
                            marker=dict(
                                color=primary_color,
                                size=kwargs.get("primary_marker_size", 8),
                            ),
                        ),
                        secondary_y=(primary_y_axis == "y2"),
                    )
                elif primary_type == "area":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=primary_series,
                            mode="lines",
                            fill="tozeroy",
                            name=kwargs.get(
                                "primary_name", primary_y_label or "Primary"
                            ),
                            line=dict(
                                color=primary_color,
                                width=kwargs.get("primary_line_width", 2),
                            ),
                            fillcolor=f'rgba({primary_color.lstrip("rgb(").rstrip(")")}, 0.2)',
                        ),
                        secondary_y=(primary_y_axis == "y2"),
                    )
                else:
                    logger.warning(
                        f"Unsupported primary chart type: {primary_type}. Using line chart."
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=primary_series,
                            mode="lines",
                            name=kwargs.get(
                                "primary_name", primary_y_label or "Primary"
                            ),
                            line=dict(
                                color=primary_color,
                                width=kwargs.get("primary_line_width", 2),
                            ),
                        ),
                        secondary_y=(primary_y_axis == "y2"),
                    )

                # Add secondary trace
                if secondary_type == "line":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=secondary_series,
                            mode="lines",
                            name=kwargs.get(
                                "secondary_name", secondary_y_label or "Secondary"
                            ),
                            line=dict(
                                color=secondary_color,
                                width=kwargs.get("secondary_line_width", 2),
                            ),
                        ),
                        secondary_y=(secondary_y_axis == "y2"),
                    )
                elif secondary_type == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=secondary_series,
                            mode="markers",
                            name=kwargs.get(
                                "secondary_name", secondary_y_label or "Secondary"
                            ),
                            marker=dict(
                                color=secondary_color,
                                size=kwargs.get("secondary_marker_size", 8),
                            ),
                        ),
                        secondary_y=(secondary_y_axis == "y2"),
                    )
                elif secondary_type == "area":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=secondary_series,
                            mode="lines",
                            fill="tozeroy",
                            name=kwargs.get(
                                "secondary_name", secondary_y_label or "Secondary"
                            ),
                            line=dict(
                                color=secondary_color,
                                width=kwargs.get("secondary_line_width", 2),
                            ),
                            fillcolor=f'rgba({secondary_color.lstrip("rgb(").rstrip(")")}, 0.2)',
                        ),
                        secondary_y=(secondary_y_axis == "y2"),
                    )
                elif secondary_type == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=x,
                            y=secondary_series,
                            name=kwargs.get(
                                "secondary_name", secondary_y_label or "Secondary"
                            ),
                            marker=dict(
                                color=secondary_color,
                                opacity=kwargs.get("secondary_opacity", 0.8),
                            ),
                        ),
                        secondary_y=(secondary_y_axis == "y2"),
                    )
                else:
                    logger.warning(
                        f"Unsupported secondary chart type: {secondary_type}. Using line chart."
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=secondary_series,
                            mode="lines",
                            name=kwargs.get(
                                "secondary_name", secondary_y_label or "Secondary"
                            ),
                            line=dict(
                                color=secondary_color,
                                width=kwargs.get("secondary_line_width", 2),
                            ),
                        ),
                        secondary_y=(secondary_y_axis == "y2"),
                    )

                # Set axis titles
                fig.update_xaxes(title_text=x_label or "")

                # Set primary y-axis
                if primary_on_right:
                    fig.update_yaxes(title_text=primary_y_label or "", secondary_y=True)
                else:
                    fig.update_yaxes(
                        title_text=primary_y_label or "", secondary_y=False
                    )

                # Set secondary y-axis
                if primary_on_right:
                    fig.update_yaxes(
                        title_text=secondary_y_label or "", secondary_y=False
                    )
                else:
                    fig.update_yaxes(
                        title_text=secondary_y_label or "", secondary_y=True
                    )

                # Align axes if requested
                if vertical_alignment:
                    # Try to align zero on both axes when possible
                    fig.update_layout(
                        yaxis=dict(rangemode="tozero"), yaxis2=dict(rangemode="tozero")
                    )

                # Configure grid
                fig.update_xaxes(showgrid=show_grid)
                fig.update_yaxes(showgrid=show_grid)

                # Set title and dimensions
                fig.update_layout(
                    title=title,
                    height=kwargs.get("height", 600),
                    width=kwargs.get("width", 900),
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
                logger.error(f"Error creating combined chart: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating combined chart: {str(e)}"
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
        Update an existing Plotly combined chart.

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
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Validate figure type
                if not isinstance(fig, go.Figure):
                    logger.warning(
                        "Cannot update non-Plotly figure with PlotlyCombinedChart"
                    )
                    return fig

                # Check if the figure has at least two traces
                if len(fig.data) < 2:
                    logger.warning(
                        "Figure does not appear to be a combined chart (fewer than 2 traces)"
                    )
                    return fig

                # Initialize update_layout
                update_layout = {}

                # Update dimensions if provided
                if "height" in kwargs:
                    update_layout["height"] = kwargs["height"]
                if "width" in kwargs:
                    update_layout["width"] = kwargs["width"]

                # Update x-axis label if provided
                if "x_label" in kwargs:
                    fig.update_xaxes(title_text=kwargs["x_label"])

                # Update y-axis labels if provided
                if "primary_y_label" in kwargs or "secondary_y_label" in kwargs:
                    # Determine which axis is primary and which is secondary
                    # Use layout information instead of trace attributes
                    primary_is_secondary_y = fig.layout.yaxis2 is not None

                    if "primary_y_label" in kwargs:
                        if primary_is_secondary_y:
                            fig.update_yaxes(
                                title_text=kwargs["primary_y_label"], secondary_y=True
                            )
                        else:
                            fig.update_yaxes(
                                title_text=kwargs["primary_y_label"], secondary_y=False
                            )

                    if "secondary_y_label" in kwargs:
                        if primary_is_secondary_y:
                            fig.update_yaxes(
                                title_text=kwargs["secondary_y_label"],
                                secondary_y=False,
                            )
                        else:
                            fig.update_yaxes(
                                title_text=kwargs["secondary_y_label"], secondary_y=True
                            )

                # Apply layout updates
                if update_layout:
                    fig.update_layout(**update_layout)

                # Check if we need to update data
                data_update_needed = any(
                    key in kwargs
                    for key in [
                        "primary_data",
                        "secondary_data",
                        "primary_type",
                        "secondary_type",
                        "x_data",
                        "primary_on_right",
                        "vertical_alignment",
                    ]
                )

                if data_update_needed:
                    # For complex updates, it's better to recreate the chart
                    # Get current properties to preserve
                    current_title = fig.layout.title.text if fig.layout.title else None
                    current_height = fig.layout.height
                    current_width = fig.layout.width
                    current_x_label = (
                        fig.layout.xaxis.title.text
                        if fig.layout.xaxis and fig.layout.xaxis.title
                        else None
                    )

                    # Determine current primary/secondary y-axis labels
                    primary_is_secondary_y = fig.layout.yaxis2 is not None
                    if primary_is_secondary_y:
                        current_primary_y_label = (
                            fig.layout.yaxis2.title.text
                            if fig.layout.yaxis2 and fig.layout.yaxis2.title
                            else None
                        )
                        current_secondary_y_label = (
                            fig.layout.yaxis.title.text
                            if fig.layout.yaxis and fig.layout.yaxis.title
                            else None
                        )
                    else:
                        current_primary_y_label = (
                            fig.layout.yaxis.title.text
                            if fig.layout.yaxis and fig.layout.yaxis.title
                            else None
                        )
                        current_secondary_y_label = (
                            fig.layout.yaxis2.title.text
                            if fig.layout.yaxis2 and fig.layout.yaxis2.title
                            else None
                        )

                    # Check if we have the data needed for recreation
                    if "primary_data" not in kwargs and "secondary_data" not in kwargs:
                        logger.warning(
                            "Cannot update chart data without both primary_data and secondary_data"
                        )
                        return fig

                    # Create new kwargs for chart recreation
                    new_kwargs = kwargs.copy()

                    # Preserve current properties if not specified in kwargs
                    if current_title and "title" not in new_kwargs:
                        new_kwargs["title"] = current_title
                    if current_height and "height" not in new_kwargs:
                        new_kwargs["height"] = current_height
                    if current_width and "width" not in new_kwargs:
                        new_kwargs["width"] = current_width
                    if current_x_label and "x_label" not in new_kwargs:
                        new_kwargs["x_label"] = current_x_label
                    if current_primary_y_label and "primary_y_label" not in new_kwargs:
                        new_kwargs["primary_y_label"] = current_primary_y_label
                    if (
                        current_secondary_y_label
                        and "secondary_y_label" not in new_kwargs
                    ):
                        new_kwargs["secondary_y_label"] = current_secondary_y_label

                    # Recreate the chart
                    return self.create(**new_kwargs)

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
                logger.error(f"Error updating combined chart: {e}")
                return fig


class MatplotlibCombinedChart(MatplotlibFigure):
    """Combined chart implementation using Matplotlib."""

    def create(
        self,
        primary_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        secondary_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        title: str,
        primary_type: str = "bar",  # "bar", "line", "scatter", "area"
        secondary_type: str = "line",  # "line", "scatter", "area", "bar"
        x_data: Optional[Union[List, np.ndarray, pd.Series]] = None,
        x_label: Optional[str] = None,
        primary_y_label: Optional[str] = None,
        secondary_y_label: Optional[str] = None,
        primary_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
        show_grid: bool = True,
        primary_on_right: bool = False,
        vertical_alignment: bool = True,
        figsize: Tuple[int, int] = (9, 6),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a combined chart with dual Y-axes using Matplotlib.

        Parameters:
        -----------
        primary_data : Dict[str, Any], pd.Series, or pd.DataFrame
            Data for the primary Y-axis.
        secondary_data : Dict[str, Any], pd.Series, or pd.DataFrame
            Data for the secondary Y-axis.
        title : str
            Title for the plot.
        primary_type : str, optional
            Type of visualization for primary data: "bar", "line", "scatter", "area".
        secondary_type : str, optional
            Type of visualization for secondary data: "line", "scatter", "area", "bar".
        x_data : list, np.ndarray, or pd.Series, optional
            Data for the x-axis. If None, indices are used.
        x_label : str, optional
            Label for the x-axis.
        primary_y_label : str, optional
            Label for the primary Y-axis.
        secondary_y_label : str, optional
            Label for the secondary Y-axis.
        primary_color : str, optional
            Color for the primary series.
        secondary_color : str, optional
            Color for the secondary series.
        show_grid : bool, optional
            Whether to show gridlines on the plot.
        primary_on_right : bool, optional
            Whether to display the primary Y-axis on the right side.
        vertical_alignment : bool, optional
            Whether to align zero values across both axes.
        figsize : Tuple[int, int], default (9, 6)
            Figure size.
        backend : str, optional
            Backend to use: "plotly" or "matplotlib" (overrides global setting).
        theme : str, optional
            Theme name to use for this visualization.
        strict : bool, optional
            If True, raise exceptions for invalid configuration; otherwise log warnings.
        **kwargs:
            Additional arguments to pass to the underlying matplotlib function.
        Returns:
        --------
        plt.Figure: The resulting Matplotlib figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Prepare data
                primary_series = prepare_data_for_combined_chart(primary_data)
                secondary_series = prepare_data_for_combined_chart(secondary_data)

                # Determine x values
                if x_data is not None:
                    x = np.asarray(x_data)
                else:
                    x = primary_series.index.values

                fig, ax1 = plt.subplots(figsize=figsize)

                # Colors
                if primary_color is None:
                    primary_color = "#1f77b4"
                if secondary_color is None:
                    secondary_color = "#ff7f0e"

                # Plot primary
                if primary_type == "bar":
                    ax1.bar(
                        x,
                        primary_series.values,
                        color=primary_color,
                        label=primary_y_label or "Primary",
                        alpha=0.8,
                    )
                elif primary_type == "line":
                    ax1.plot(
                        x,
                        primary_series.values,
                        color=primary_color,
                        label=primary_y_label or "Primary",
                        linewidth=2,
                    )
                elif primary_type == "scatter":
                    ax1.scatter(
                        x,
                        primary_series.values,
                        color=primary_color,
                        label=primary_y_label or "Primary",
                    )
                elif primary_type == "area":
                    ax1.fill_between(
                        x,
                        primary_series.values,
                        color=primary_color,
                        alpha=0.3,
                        label=primary_y_label or "Primary",
                    )
                else:
                    ax1.plot(
                        x,
                        primary_series.values,
                        color=primary_color,
                        label=primary_y_label or "Primary",
                        linewidth=2,
                    )

                # Secondary axis
                if primary_on_right:
                    ax2 = ax1
                    ax1 = ax1.twinx()
                else:
                    ax2 = ax1.twinx()

                # Plot secondary
                if secondary_type == "bar":
                    ax2.bar(
                        x,
                        secondary_series.values,
                        color=secondary_color,
                        label=secondary_y_label or "Secondary",
                        alpha=0.5,
                    )
                elif secondary_type == "line":
                    ax2.plot(
                        x,
                        secondary_series.values,
                        color=secondary_color,
                        label=secondary_y_label or "Secondary",
                        linewidth=2,
                        linestyle="--",
                    )
                elif secondary_type == "scatter":
                    ax2.scatter(
                        x,
                        secondary_series.values,
                        color=secondary_color,
                        label=secondary_y_label or "Secondary",
                        marker="x",
                    )
                elif secondary_type == "area":
                    ax2.fill_between(
                        x,
                        secondary_series.values,
                        color=secondary_color,
                        alpha=0.2,
                        label=secondary_y_label or "Secondary",
                    )
                else:
                    ax2.plot(
                        x,
                        secondary_series.values,
                        color=secondary_color,
                        label=secondary_y_label or "Secondary",
                        linewidth=2,
                        linestyle="--",
                    )

                # Set labels and title
                ax1.set_xlabel(x_label or "")
                ax1.set_ylabel(primary_y_label or "", color=primary_color)
                ax2.set_ylabel(secondary_y_label or "", color=secondary_color)
                fig.suptitle(title)

                # Grid
                ax1.grid(show_grid)
                ax2.grid(False)

                # Align zero if requested
                if vertical_alignment:
                    ax1.set_ylim(bottom=min(0, np.nanmin(primary_series.values)))
                    ax2.set_ylim(bottom=min(0, np.nanmin(secondary_series.values)))

                # Legends
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                if handles1 or handles2:
                    fig.legend(
                        handles1 + handles2, labels1 + labels2, loc="upper right"
                    )

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
                logger.error(f"Error creating combined chart with Matplotlib: {e}")
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
        Update an existing Matplotlib combined chart.

        Parameters:
        -----------
        fig : plt.Figure
            Existing Matplotlib figure to update
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
        plt.Figure: Updated Matplotlib figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Only support updating data and labels for now
                # If new data is provided, re-create the chart
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

                if "primary_data" in kwargs and "secondary_data" in kwargs:
                    new_kwargs = kwargs.copy()
                    # Try to preserve title and labels if not provided
                    if "title" not in new_kwargs and fig._suptitle is not None:
                        new_kwargs["title"] = fig._suptitle.get_text()
                    axes = fig.get_axes()
                    if axes:
                        ax1 = axes[0]
                        if "x_label" not in new_kwargs:
                            new_kwargs["x_label"] = ax1.get_xlabel()
                        if "primary_y_label" not in new_kwargs:
                            new_kwargs["primary_y_label"] = ax1.get_ylabel()
                        if len(axes) > 1 and "secondary_y_label" not in new_kwargs:
                            new_kwargs["secondary_y_label"] = axes[1].get_ylabel()
                    return self.create(
                        new_kwargs["primary_data"],
                        new_kwargs["secondary_data"],
                        new_kwargs.get("title", ""),
                        primary_type=new_kwargs.get("primary_type", "bar"),
                        secondary_type=new_kwargs.get("secondary_type", "line"),
                        x_data=new_kwargs.get("x_data"),
                        x_label=new_kwargs.get("x_label"),
                        primary_y_label=new_kwargs.get("primary_y_label"),
                        secondary_y_label=new_kwargs.get("secondary_y_label"),
                        primary_color=new_kwargs.get("primary_color"),
                        secondary_color=new_kwargs.get("secondary_color"),
                        show_grid=new_kwargs.get("show_grid", True),
                        primary_on_right=new_kwargs.get("primary_on_right", False),
                        vertical_alignment=new_kwargs.get("vertical_alignment", True),
                        backend=backend,
                        theme=theme,
                        strict=strict,
                    )
                # Update title if provided
                if "title" in kwargs and fig._suptitle is not None:
                    fig._suptitle.set_text(kwargs["title"])
                # Update axis labels if provided
                axes = fig.get_axes()
                if axes:
                    ax1 = axes[0]
                    if "x_label" in kwargs:
                        ax1.set_xlabel(kwargs["x_label"])
                    if "primary_y_label" in kwargs:
                        ax1.set_ylabel(kwargs["primary_y_label"])
                    if len(axes) > 1 and "secondary_y_label" in kwargs:
                        axes[1].set_ylabel(kwargs["secondary_y_label"])

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
                logger.error(f"Error updating combined chart (Matplotlib): {e}")
                return fig


# Register the implementation
FigureRegistry.register("combined", "plotly", PlotlyCombinedChart)
FigureRegistry.register("combined", "matplotlib", MatplotlibCombinedChart)
