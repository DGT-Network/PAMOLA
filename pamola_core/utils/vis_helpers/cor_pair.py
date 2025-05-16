"""
Correlation pair plot visualization implementation.

This module provides implementation for correlation pair plot visualization
using Plotly backend.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pamola_core.utils.vis_helpers.base import PlotlyFigure, FigureRegistry
from pamola_core.utils.vis_helpers.cor_utils import calculate_correlation
from pamola_core.utils.vis_helpers.theme import apply_theme_to_plotly_figure, get_theme_colors

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyCorrelationPair(PlotlyFigure):
    """Correlation plot for a pair of variables implementation using Plotly."""

    def create(self,
               x_data: Union[List[float], np.ndarray, pd.Series],
               y_data: Union[List[float], np.ndarray, pd.Series],
               title: str,
               x_label: Optional[str] = None,
               y_label: Optional[str] = None,
               correlation: Optional[float] = None,
               method: Optional[str] = "Pearson",
               add_trendline: bool = True,
               add_histogram: bool = True,
               color: Optional[str] = None,
               **kwargs) -> go.Figure:
        """
        Create a correlation plot for a pair of variables using Plotly.

        Parameters:
        -----------
        x_data : List[float], np.ndarray, or pd.Series
            Data for the x-axis
        y_data : List[float], np.ndarray, or pd.Series
            Data for the y-axis
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
        color : str, optional
            Color for the scatter points
        **kwargs:
            Additional arguments to pass to go.Scatter

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the correlation plot

        Raises:
        -------
        ValueError
            If input data is invalid or insufficient for correlation
        """
        try:
            # Ensure data is in numpy array format for consistent handling
            x = np.asarray(x_data)
            y = np.asarray(y_data)

            # Remove entries with NaN values in either variable
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]

            # Handle empty data case
            if len(x) == 0 or len(y) == 0:
                return self.create_empty_figure(
                    title=title,
                    message="No valid data points for correlation plot"
                )

            # Calculate correlation if not provided
            if correlation is None and len(x) > 1:
                correlation = calculate_correlation(x, y)

            # Get color from theme if not provided
            if color is None:
                color = get_theme_colors(1)[0]

            # Create appropriate figure layout
            marker_size = kwargs.get('marker_size', 8)

            if add_histogram:
                # Create figure with 2x2 grid for scatter plot and histograms
                fig = make_subplots(
                    rows=2, cols=2,
                    column_widths=[0.8, 0.2],
                    row_heights=[0.2, 0.8],
                    vertical_spacing=0.05,
                    horizontal_spacing=0.05,
                    specs=[
                        [{"type": "histogram"}, None],
                        [{"type": "scatter"}, {"type": "histogram"}]
                    ]
                )

                # Add main scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        marker=dict(
                            color=color,
                            opacity=0.7,
                            size=marker_size
                        ),
                        name='Data Points'
                    ),
                    row=2, col=1
                )

                # Add trendline if requested and possible
                if add_trendline and len(x) > 1:
                    self._add_trendline(fig, x, y, color, row=2, col=1)

                # Add histograms
                fig.add_trace(
                    go.Histogram(
                        x=x,
                        marker=dict(color=color),
                        opacity=0.7,
                        name=x_label or 'X'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Histogram(
                        y=y,
                        marker=dict(color=color),
                        opacity=0.7,
                        name=y_label or 'Y'
                    ),
                    row=2, col=2
                )

                # Hide legends for histograms (just show for scatter)
                fig.update_traces(showlegend=False, selector=dict(type='histogram'))

                # Update layout
                fig.update_layout(
                    title=title,
                    showlegend=False,
                    # Apply theme colors and styling
                    margin=dict(l=80, r=50, t=80, b=80)
                )

                # Update axes
                fig.update_xaxes(title_text=x_label or "X", row=2, col=1)
                fig.update_yaxes(title_text=y_label or "Y", row=2, col=1)

                # Hide axis labels for histograms
                fig.update_xaxes(title_text="", showticklabels=False, row=1, col=1)
                fig.update_yaxes(title_text="", showticklabels=False, row=2, col=2)
            else:
                # Simple scatter plot without histograms
                fig = go.Figure()

                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        marker=dict(
                            color=color,
                            opacity=0.7,
                            size=marker_size
                        ),
                        name='Data Points'
                    )
                )

                # Add trendline if requested and possible
                if add_trendline and len(x) > 1:
                    self._add_trendline(fig, x, y, color)

                # Set axis labels
                fig.update_layout(
                    xaxis_title=x_label or "X",
                    yaxis_title=y_label or "Y",
                    title=title
                )

            # Add annotation for correlation coefficient if provided or calculated
            if correlation is not None:
                self._add_correlation_annotation(fig, correlation, method)

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error creating correlation pair plot: {e}")
            return self.create_empty_figure(
                title=title,
                message=f"Error creating correlation pair plot: {str(e)}"
            )

    def update(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Update an existing Plotly correlation pair plot.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Plotly figure to update
        **kwargs:
            Parameters to update. Supported parameters include:
            - x_data: New x-axis data
            - y_data: New y-axis data
            - title: New title for the plot
            - x_label: New x-axis label
            - y_label: New y-axis label
            - correlation: New correlation value to display
            - method: New correlation method name
            - add_trendline: Whether to add/update trendline (default: True)
            - color: New color for the scatter points
            - marker_size: New size for the markers

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated figure
        """
        try:
            # Validate input figure type
            if not isinstance(fig, go.Figure):
                logger.warning("Cannot update non-Plotly figure with PlotlyCorrelationPair")
                return fig

            # Update title if provided
            if 'title' in kwargs:
                fig.update_layout(title=kwargs['title'])

            # Update data if provided
            if 'x_data' in kwargs and 'y_data' in kwargs:
                try:
                    x_data = kwargs['x_data']
                    y_data = kwargs['y_data']

                    # Convert input data to arrays
                    x = np.asarray(x_data)
                    y = np.asarray(y_data)

                    # Remove entries with NaN values in either variable
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x = x[mask]
                    y = y[mask]

                    if len(x) == 0 or len(y) == 0:
                        logger.warning("No valid data points after filtering NaN values")
                        return fig

                    # Determine if this is a multi-plot layout (with histograms)
                    is_multi_plot = len(fig.data) > 2

                    # Get other styling parameters
                    color = kwargs.get('color', None)
                    marker_size = kwargs.get('marker_size', None)
                    add_trendline = kwargs.get('add_trendline', True)

                    # Update the scatter plot
                    self._update_scatter_plot(fig, x, y, color, marker_size, is_multi_plot)

                    # Update the trendline
                    self._update_trendline(fig, x, y, color, add_trendline, is_multi_plot)

                    # Update histograms if this is a multi-plot
                    if is_multi_plot:
                        self._update_histograms(fig, x, y, color)

                    # Update axis labels if provided
                    if 'x_label' in kwargs:
                        if is_multi_plot:
                            fig.update_xaxes(title_text=kwargs['x_label'], row=2, col=1)
                        else:
                            fig.update_layout(xaxis_title=kwargs['x_label'])

                    if 'y_label' in kwargs:
                        if is_multi_plot:
                            fig.update_yaxes(title_text=kwargs['y_label'], row=2, col=1)
                        else:
                            fig.update_layout(yaxis_title=kwargs['y_label'])

                    # Update correlation annotation
                    correlation = kwargs.get('correlation')
                    method = kwargs.get('method', 'Pearson')

                    if correlation is None and kwargs.get('calculate_correlation', False) and len(x) > 1:
                        correlation = calculate_correlation(x, y)

                    # Update correlation annotation if we have a valid correlation value
                    if correlation is not None:
                        # Ensure correlation is a float value
                        if isinstance(correlation, (np.ndarray, list, tuple)):
                            try:
                                correlation = float(correlation[0])
                            except (IndexError, TypeError, ValueError):
                                try:
                                    correlation = float(correlation)
                                except (TypeError, ValueError):
                                    logger.warning("Could not convert correlation to float")
                                    correlation = None

                        if correlation is not None:
                            self._update_correlation_annotation(fig, correlation, method)
                except Exception as e:
                    logger.error(f"Error updating correlation pair data: {e}")

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error updating correlation pair plot: {e}")
            return fig

    def _add_trendline(self,
                       fig: go.Figure,
                       x: np.ndarray,
                       y: np.ndarray,
                       color: str,
                       row: Optional[int] = None,
                       col: Optional[int] = None) -> None:
        """
        Add trendline to figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to add trendline to
        x : np.ndarray
            X-axis data
        y : np.ndarray
            Y-axis data
        color : str
            Base color for the plot
        row : int, optional
            Row index for subplot (if applicable)
        col : int, optional
            Column index for subplot (if applicable)
        """
        try:
            # Calculate linear regression
            slope, intercept = np.polyfit(x, y, 1)

            # Generate points for trendline
            x_trend = np.linspace(min(x), max(x), 100)
            y_trend = slope * x_trend + intercept

            # Create trendline trace
            trace = go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Trendline'
            )

            # Add trace appropriately based on whether subplot is specified
            if row is not None and col is not None:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)
        except Exception as e:
            logger.warning(f"Could not add trendline: {e}")

    def _update_trendline(self,
                          fig: go.Figure,
                          x: np.ndarray,
                          y: np.ndarray,
                          color: Optional[str],
                          add_trendline: bool,
                          is_multi_plot: bool) -> None:
        """
        Update or add trendline in figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to update
        x : np.ndarray
            X-axis data
        y : np.ndarray
            Y-axis data
        color : str, optional
            Color for the trendline (if not provided, red is used)
        add_trendline : bool
            Whether to add/keep trendline
        is_multi_plot : bool
            Whether this is a multi-plot layout with histograms
        """
        # Check if there's already a trendline
        has_trendline = False
        trendline_idx = None

        for i, trace in enumerate(fig.data):
            # Use getattr for safer attribute access
            trace_type = getattr(trace, 'type', None)
            trace_mode = getattr(trace, 'mode', None)

            if trace_type == 'scatter' and trace_mode == 'lines':
                has_trendline = True
                trendline_idx = i
                break

        # Handle trendline update based on request
        if add_trendline and len(x) > 1:
            try:
                # Calculate linear regression
                slope, intercept = np.polyfit(x, y, 1)

                # Generate points for trendline
                x_trend = np.linspace(min(x), max(x), 100)
                y_trend = slope * x_trend + intercept

                if has_trendline and trendline_idx is not None:
                    # Update existing trendline
                    # Using update_traces is safer than directly modifying data
                    fig.update_traces(
                        x=x_trend,
                        y=y_trend,
                        selector=dict(mode='lines')
                    )
                else:
                    # Add new trendline
                    trendline_color = 'red'  # Default color
                    if color is not None:
                        # We could use a derivative of the main color for the trendline
                        trendline_color = color

                    trendline = go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        line=dict(color=trendline_color, width=2, dash='dash'),
                        name='Trendline'
                    )

                    if is_multi_plot:
                        # For multi-plot, add to the main scatter subplot (row=2, col=1)
                        fig.add_trace(trendline, row=2, col=1)
                    else:
                        fig.add_trace(trendline)
            except Exception as e:
                logger.warning(f"Could not update trendline: {e}")
        elif has_trendline and not add_trendline and trendline_idx is not None:
            # Remove trendline if it exists but is not wanted
            updated_data = []
            for i, trace in enumerate(fig.data):
                if i != trendline_idx:  # Keep all traces except the trendline
                    updated_data.append(trace)
            fig.data = updated_data  # Replace with filtered list

    def _update_scatter_plot(self,
                             fig: go.Figure,
                             x: np.ndarray,
                             y: np.ndarray,
                             color: Optional[str],
                             marker_size: Optional[int],
                             is_multi_plot: bool) -> None:
        """
        Update scatter plot in figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to update
        x : np.ndarray
            New x-axis data
        y : np.ndarray
            New y-axis data
        color : str, optional
            New color for scatter points
        marker_size : int, optional
            New size for markers
        is_multi_plot : bool
            Whether this is a multi-plot layout with histograms
        """
        # Find main scatter plot trace
        for i, trace in enumerate(fig.data):
            # Use getattr for safer attribute access
            trace_type = getattr(trace, 'type', None)
            trace_mode = getattr(trace, 'mode', None)

            # Check if this is a scatter trace with markers mode
            if trace_type == 'scatter' and trace_mode == 'markers':
                # Create update dictionary
                update_dict = {'x': x, 'y': y}

                # Update marker properties if provided
                if color is not None or marker_size is not None:
                    marker_update = {}
                    if color is not None:
                        marker_update['color'] = color
                    if marker_size is not None:
                        marker_update['size'] = marker_size

                    if marker_update:
                        update_dict['marker'] = marker_update

                # Update trace
                if is_multi_plot:
                    fig.update_traces(
                        **update_dict,
                        selector=dict(type='scatter', mode='markers'),
                        row=2, col=1  # Main scatter is in second row, first column
                    )
                else:
                    fig.update_traces(
                        **update_dict,
                        selector=dict(type='scatter', mode='markers')
                    )
                break

    def _update_histograms(self,
                           fig: go.Figure,
                           x: np.ndarray,
                           y: np.ndarray,
                           color: Optional[str] = None) -> None:
        """
        Update histograms in figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to update
        x : np.ndarray
            New x-axis data
        y : np.ndarray
            New y-axis data
        color : str, optional
            New color for histograms
        """
        # Update x histogram (top row)
        fig.update_traces(
            x=x,
            selector=dict(type='histogram'),
            row=1, col=1
        )

        # Update y histogram (right column)
        fig.update_traces(
            y=y,
            selector=dict(type='histogram'),
            row=2, col=2
        )

        # Update color if provided
        if color is not None:
            fig.update_traces(
                marker=dict(color=color),
                selector=dict(type='histogram')
            )

    def _add_correlation_annotation(self,
                                    fig: go.Figure,
                                    correlation: float,
                                    method: Optional[str] = "Pearson") -> None:
        """
        Add correlation annotation to figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to add annotation to
        correlation : float
            Correlation coefficient
        method : str, optional
            Correlation method name
        """
        annotation_text = f"{method} correlation: {correlation:.3f}"
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=4,
            align="left"
        )

    def _update_correlation_annotation(self,
                                       fig: go.Figure,
                                       correlation: float,
                                       method: str) -> None:
        """
        Update correlation annotation in figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to update
        correlation : float
            New correlation value
        method : str
            Correlation method name
        """
        # Check for existing correlation annotation
        found = False
        for i, annotation in enumerate(fig.layout.annotations or []):
            if "correlation:" in getattr(annotation, 'text', ''):
                # Update existing annotation
                fig.layout.annotations[i].text = f"{method} correlation: {correlation:.3f}"
                found = True
                break

        if not found:
            # Add new correlation annotation
            self._add_correlation_annotation(fig, correlation, method)


# Register this figure type with the registry
FigureRegistry.register("correlation_pair", "plotly", PlotlyCorrelationPair)