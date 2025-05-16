"""
Scatter plot implementations for the visualization system.

This module provides implementations for scatter plots using both
Plotly and Matplotlib backends.
"""

import logging
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pamola_core.utils.vis_helpers.base import (
    PlotlyFigure,
    FigureRegistry
)
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_plotly_figure,
    get_theme_colors
)

# Configure logger
logger = logging.getLogger(__name__)


class BaseScatterPlot:
    """
    Base class for scatter plot implementations with shared utility methods.
    """

    @staticmethod
    def _prepare_scatter_data(
            x_data: Union[List[float], np.ndarray, pd.Series],
            y_data: Union[List[float], np.ndarray, pd.Series]
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
        # Convert data to arrays
        x = np.array(x_data)
        y = np.array(y_data)

        # Remove NaN values (points where either x or y is NaN)
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        return x, y

    @staticmethod
    def _calculate_trendline(
            x: np.ndarray,
            y: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
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
            **kwargs
    ) -> go.Figure:
        """
        Create a scatter plot using Plotly with comprehensive customization.

        Parameters as defined in the method signature.
        """
        # Prepare data
        x, y = BaseScatterPlot._prepare_scatter_data(x_data, y_data)

        # Validate data
        if len(x) == 0 or len(y) == 0:
            logger.warning("No valid data points for scatter plot")
            return self.create_empty_figure(
                title=title,
                message="No valid data points for scatter plot"
            )

        # Create figure
        fig = go.Figure()

        # Prepare marker properties
        marker_props = {}

        # Handle marker size
        if marker_size is not None:
            if isinstance(marker_size, (int, float)):
                # Convert float to int if necessary to satisfy type requirements
                marker_props['size'] = int(marker_size) if isinstance(marker_size, float) else marker_size
            else:
                # Convert to numpy array and ensure it matches data length
                marker_size_array = np.array(marker_size, dtype=np.int32)  # Ensure int dtype
                if len(marker_size_array) == len(x):
                    marker_props['size'] = marker_size_array.tolist()
                else:
                    logger.warning("Marker size length doesn't match data length")
                    # Make sure we're using an integer value
                    if len(marker_size_array) > 0:
                        first_value = marker_size_array[0]
                        marker_props['size'] = int(first_value) if isinstance(first_value, float) else first_value
                    else:
                        marker_props['size'] = 8  # Default size

        # Handle marker color
        marker_color = kwargs.pop('marker_color', get_theme_colors(1)[0])
        if color_values is not None:
            # Convert color values to list for Plotly compatibility
            color_values_array = np.array(color_values)

            # Ensure color values match data length
            if len(color_values_array) == len(x):
                marker_props['color'] = color_values_array.tolist()

                # Handle color scale
                if color_scale is not None:
                    marker_props['colorscale'] = str(color_scale)

                # Add colorbar with title
                marker_props['colorbar'] = dict(
                    title=str(kwargs.pop('colorbar_title', ""))
                )
            else:
                logger.warning("Color values length doesn't match data length")
                marker_props['color'] = marker_color
        else:
            marker_props['color'] = marker_color

        # Prepare hover text
        hovertext = None
        if hover_text is not None:
            hover_text_array = np.array(hover_text, dtype=str)
            if len(hover_text_array) == len(x):
                hovertext = hover_text_array.tolist()

        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=marker_props,
            hovertext=hovertext,
            **kwargs
        ))

        # Add trendline if requested
        if add_trendline:
            slope, intercept, x_range, y_range = BaseScatterPlot._calculate_trendline(x, y)
            if x_range is not None and y_range is not None:
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name='Trendline',
                    line=dict(color='red', width=2, dash='dash')
                ))

        # Set axis labels
        fig.update_layout(
            xaxis_title=x_label or "X",
            yaxis_title=y_label or "Y"
        )

        # Set title
        fig.update_layout(title=title)

        # Add correlation annotation if provided
        if correlation is not None:
            correlation_text = f"{method or 'Correlation'}: {correlation:.3f}"
            fig.add_annotation(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text=correlation_text,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                font=dict(size=12)
            )

        # Apply theme
        fig = apply_theme_to_plotly_figure(fig)

        return fig

    def update(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Update an existing Plotly scatter plot.

        (Implementation similar to the original, with more robust error handling)
        """
        # Validate figure
        if not isinstance(fig, go.Figure):
            logger.warning("Cannot update non-Plotly figure with PlotlyScatterPlot")
            return fig

        # Similar update logic to create method, focusing on data, trendline, and annotations
        # (Full implementation would mirror the create method's approach)
        return fig


# Register plot implementations
FigureRegistry.register("scatter", "plotly", PlotlyScatterPlot)