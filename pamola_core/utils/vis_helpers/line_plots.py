"""
Line plot implementations for the visualization system.

This module provides the implementation for line plots using Plotly backend.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pamola_core.utils.vis_helpers.base import PlotlyFigure, FigureRegistry
from pamola_core.utils.vis_helpers.theme import apply_theme_to_plotly_figure, get_theme_colors

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyLinePlot(PlotlyFigure):
    """Line plot implementation using Plotly."""

    def create(self,
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
               **kwargs) -> 'go.Figure':
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
        **kwargs:
            Additional arguments to pass to go.Scatter

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the line plot
        """
        # Process data to a standard format
        processed_data, x, series_names = self._prepare_data_for_lineplot(data, x_data)

        # Create figure
        fig = go.Figure()

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
            **kwargs
        )

        # Add highlighted regions if requested
        if highlight_regions:
            fig = self._add_highlight_regions(fig, highlight_regions)

        # Configure layout
        fig = self._configure_layout(
            fig,
            title,
            x_label,
            y_label,
            len(processed_data.columns)
        )

        # Apply theme
        fig = apply_theme_to_plotly_figure(fig)

        return fig

    def update(self, fig: 'go.Figure', **kwargs) -> 'go.Figure':
        """
        Update an existing Plotly line plot.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Plotly figure to update
        **kwargs:
            Parameters to update

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated figure
        """
        # Ensure we have a Plotly figure
        if not isinstance(fig, go.Figure):
            logger.warning("Cannot update non-Plotly figure with PlotlyLinePlot")
            return fig

        # Update basic attributes (title, labels)
        if 'title' in kwargs:
            fig.update_layout(title=kwargs['title'])
        if 'x_label' in kwargs:
            fig.update_layout(xaxis_title=kwargs['x_label'])
        if 'y_label' in kwargs:
            fig.update_layout(yaxis_title=kwargs['y_label'])

        # Update data if provided
        if 'data' in kwargs:
            data = kwargs['data']
            x_data = kwargs.get('x_data', None)

            # Process new data
            processed_data, x, series_names = self._prepare_data_for_lineplot(data, x_data)

            # Get the current trace names for comparison
            # Используем альтернативный способ доступа к имени трассы через ['name']
            existing_trace_names = [trace['name'] if 'name' in trace else f'trace_{i}' for i, trace in
                                    enumerate(fig.data)]

            # Get style parameters
            add_markers = kwargs.get('add_markers', True)
            add_area = kwargs.get('add_area', False)
            smooth = kwargs.get('smooth', False)
            line_width = kwargs.get('line_width', 2.0)

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
                **kwargs
            )

        # Update highlighted regions if provided
        if 'highlight_regions' in kwargs:
            # Remove existing highlighted regions
            shapes = []
            annotations = []

            # Keep shapes and annotations that aren't related to highlight regions
            for shape in fig.layout.shapes or []:
                if not (shape.get('layer') == 'below' and shape.get('yref') == 'paper' and shape.get('type') == 'rect'):
                    shapes.append(shape)

            for annotation in fig.layout.annotations or []:
                if not (annotation.get('yref') == 'paper' and annotation.get('y') == 1.05):
                    annotations.append(annotation)

            # Update layout without the highlight region shapes and annotations
            fig.update_layout(shapes=shapes, annotations=annotations)

            # Add new highlighted regions
            highlight_regions = kwargs['highlight_regions']
            if highlight_regions:
                fig = self._add_highlight_regions(fig, highlight_regions)

        # Apply theme
        fig = apply_theme_to_plotly_figure(fig)

        return fig

    def _prepare_data_for_lineplot(self,
                                   data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
                                   x_data: Optional[Union[List, np.ndarray, pd.Series]] = None) -> Tuple[
        pd.DataFrame, Union[List, np.ndarray, pd.Series], List[str]]:
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
        # Convert different input formats to DataFrame
        if isinstance(data, dict):
            # Dictionary with lists - convert to DataFrame
            df = pd.DataFrame(data)
            series_names = list(df.columns)
        elif isinstance(data, pd.Series):
            # Series - convert to single-column DataFrame
            df = pd.DataFrame({data.name or 'Value': data})
            series_names = [data.name or 'Value']
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
            x = df.index

            # Convert to datetime if indices are DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                x = df.index.to_pydatetime()

        return df, x, series_names

    def _add_data_traces(self,
                         fig: 'go.Figure',
                         processed_data: pd.DataFrame,
                         x: Union[List, np.ndarray, pd.Series],
                         series_names: List[str],
                         add_markers: bool,
                         add_area: bool,
                         smooth: bool,
                         line_width: float,
                         **kwargs) -> 'go.Figure':
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
        # Get colors from theme
        colors = get_theme_colors(len(processed_data.columns))

        # Add a trace for each series
        for i, column in enumerate(processed_data.columns):
            color = kwargs.get(f'color_{column}', kwargs.get('color', colors[i % len(colors)]))

            # Determine mode based on markers parameter
            mode = 'lines'
            if add_markers:
                mode += '+markers'

            # Determine line shape based on smooth parameter
            line_shape = 'spline' if smooth else 'linear'

            # Create trace
            trace_args = {
                'x': x,
                'y': processed_data[column],
                'mode': mode,
                'name': series_names[i],
                'line': {
                    'width': line_width,
                    'color': color,
                    'shape': line_shape
                },
                'connectgaps': kwargs.get('connectgaps', True)  # Connect gaps from missing values
            }

            # Add fill if requested
            if add_area:
                trace_args['fill'] = 'tozeroy'
                # Create a more transparent version of the color for fill
                if color.startswith('#'):
                    trace_args['fillcolor'] = f'{color}33'  # Add 20% opacity
                else:
                    trace_args['fillcolor'] = f'rgba({color.replace("rgb(", "").replace(")", "")}, 0.2)'

            # Add custom hover text if provided
            if 'hovertext' in kwargs:
                if isinstance(kwargs['hovertext'], list) and len(kwargs['hovertext']) == len(x):
                    trace_args['hovertext'] = kwargs['hovertext']
                    trace_args['hoverinfo'] = 'text'

            # Add the trace to the figure
            fig.add_trace(go.Scatter(**trace_args))

        return fig

    def _add_highlight_regions(self,
                               fig: 'go.Figure',
                               highlight_regions: List[Dict[str, Any]]) -> 'go.Figure':
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
        shapes = list(fig.layout.shapes or [])
        annotations = list(fig.layout.annotations or [])

        for region in highlight_regions:
            # Add shaded rectangle for the region
            start = region.get('start')
            end = region.get('end')
            color = region.get('color', 'rgba(255, 0, 0, 0.2)')
            label = region.get('label', '')

            if start is not None and end is not None:
                shapes.append({
                    'type': 'rect',
                    'x0': start,
                    'x1': end,
                    'y0': 0,
                    'y1': 1,
                    'yref': 'paper',
                    'fillcolor': color,
                    'opacity': 0.3,
                    'layer': 'below',
                    'line_width': 0,
                })

                # Add annotation for the region label
                if label:
                    annotations.append({
                        'x': (start + end) / 2,
                        'y': 1.05,
                        'yref': 'paper',
                        'text': label,
                        'showarrow': False,
                        'font': dict(size=10)
                    })

        fig.update_layout(shapes=shapes, annotations=annotations)
        return fig

    def _configure_layout(self,
                          fig: 'go.Figure',
                          title: str,
                          x_label: Optional[str],
                          y_label: Optional[str],
                          series_count: int) -> 'go.Figure':
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
        # Configure layout
        layout_args = {
            'title': title,
            'xaxis': {
                'title': x_label,
                'showgrid': True
            },
            'yaxis': {
                'title': y_label,
                'showgrid': True
            },
            'legend': {
                'orientation': 'h' if series_count > 3 else 'v',
                'yanchor': 'bottom' if series_count > 3 else 'top',
                'y': -0.2 if series_count > 3 else 1,
                'xanchor': 'center' if series_count > 3 else 'left',
                'x': 0.5 if series_count > 3 else 0
            },
            'hovermode': 'closest',
            'margin': dict(t=50, b=50, l=50, r=50)
        }

        fig.update_layout(**layout_args)
        return fig


# Register only Plotly implementation
FigureRegistry.register("line", "plotly", PlotlyLinePlot)