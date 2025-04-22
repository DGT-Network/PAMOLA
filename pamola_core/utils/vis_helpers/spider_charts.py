"""
Spider chart (radar chart) implementations for the visualization system.

This module provides implementations for spider/radar charts using
Plotly as the primary backend, with support for multiple metrics
across different categories.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pamola_core.utils.vis_helpers.base import PlotlyFigure, FigureRegistry
from pamola_core.utils.vis_helpers.theme import apply_theme_to_plotly_figure, get_theme_colors

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
            **kwargs
    ) -> go.Figure:
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
        **kwargs:
            Additional arguments to pass to the Plotly trace

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the spider chart
        """
        try:
            # Convert data to DataFrame if it's a dict
            if isinstance(data, dict):
                df = pd.DataFrame({series: pd.Series(values) for series, values in data.items()}).T
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise TypeError(f"Unsupported data type for spider chart: {type(data)}")

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
                color = kwargs.get(f'color_{series}', kwargs.get('color', colors[i % len(colors)]))

                # Set up trace parameters
                trace_params = {
                    'r': values_closed,
                    'theta': categories_closed,
                    'name': series,
                    'line': dict(color=color, width=kwargs.get('line_width', 2))
                }

                # Add fill if requested
                if fill_area:
                    trace_params['fill'] = 'toself'
                    trace_params['fillcolor'] = f'rgba({color.lstrip("rgb(").rstrip(")")}, 0.2)'

                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in ['title', 'color', 'line_width', 'height', 'width']:
                        trace_params[key] = value

                # Add the trace based on spider type
                if spider_type.lower() == 'barpolar':
                    fig.add_trace(go.Barpolar(**trace_params))
                else:  # Default to scatterpolar
                    fig.add_trace(go.Scatterpolar(**trace_params))

            # Configure the layout
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=radial_range
                    ),
                    angularaxis=dict(
                        direction="clockwise",
                        rotation=angle_start
                    )
                ),
                showlegend=show_legend,
                height=kwargs.get('height', 600),
                width=kwargs.get('width', 600)
            )

            # Configure grid visibility
            fig.update_polars(radialaxis_showgrid=show_gridlines, angularaxis_showgrid=show_gridlines)

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error creating spider chart: {e}")
            return self.create_empty_figure(
                title=title,
                message=f"Error creating spider chart: {str(e)}"
            )

    def update(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Update an existing Plotly spider chart.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Existing Plotly figure to update
        **kwargs:
            Parameters to update (same as create method)

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated Plotly figure
        """
        try:
            # Validate figure type
            if not isinstance(fig, go.Figure):
                logger.warning("Cannot update non-Plotly figure with PlotlySpiderChart")
                return fig

            # Update title if provided
            if 'title' in kwargs:
                fig.update_layout(title=kwargs['title'])

            # Update radial axis range if max_value provided
            if 'max_value' in kwargs:
                max_val = kwargs['max_value']
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, max_val])))

            # Update gridlines if requested
            if 'show_gridlines' in kwargs:
                show_gridlines = kwargs['show_gridlines']
                fig.update_polars(radialaxis_showgrid=show_gridlines, angularaxis_showgrid=show_gridlines)

            # Update legend visibility if requested
            if 'show_legend' in kwargs:
                fig.update_layout(showlegend=kwargs['show_legend'])

            # Update dimensions if requested
            if 'height' in kwargs or 'width' in kwargs:
                height = kwargs.get('height', fig.layout.height)
                width = kwargs.get('width', fig.layout.width)
                fig.update_layout(height=height, width=width)

            # Update data if provided
            if 'data' in kwargs:
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
                if current_title and 'title' not in new_kwargs:
                    new_kwargs['title'] = current_title
                if current_height and 'height' not in new_kwargs:
                    new_kwargs['height'] = current_height
                if current_width and 'width' not in new_kwargs:
                    new_kwargs['width'] = current_width
                if current_showlegend is not None and 'show_legend' not in new_kwargs:
                    new_kwargs['show_legend'] = current_showlegend

                return self.create(**new_kwargs)

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error updating spider chart: {e}")
            return fig


# Register the implementation
FigureRegistry.register("spider", "plotly", PlotlySpiderChart)