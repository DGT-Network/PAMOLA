"""
Pie chart implementations for the visualization system.

This module provides implementations for pie charts using Plotly as
the primary backend, with support for donut charts and sunburst charts
for hierarchical data visualization.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import plotly.graph_objects as go

from pamola_core.utils.vis_helpers.base import PlotlyFigure, FigureRegistry
from pamola_core.utils.vis_helpers.theme import apply_theme_to_plotly_figure, get_theme_colors

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyPieChart(PlotlyFigure):
    """Pie chart implementation using Plotly."""

    def create(
            self,
            data: Union[Dict[str, float], pd.Series, List[float]],
            title: str,
            labels: Optional[List[str]] = None,
            hole: float = 0,  # 0 for pie chart, >0 for donut (e.g., 0.4)
            show_values: bool = True,
            value_format: str = ".1f",
            show_percentages: bool = True,
            sort_values: bool = False,
            pull_largest: bool = False,
            pull_value: float = 0.1,
            clockwise: bool = True,
            start_angle: float = 90,
            textposition: str = "auto",
            **kwargs
    ) -> go.Figure:
        """
        Create a pie chart using Plotly.

        Parameters:
        -----------
        data : Union[Dict[str, float], pd.Series, List[float]]
            Data to visualize. If dict or Series, keys are used as labels.
            If list, separate labels should be provided.
        title : str
            Title for the plot
        labels : List[str], optional
            List of labels for pie slices (not needed if data is dict or Series)
        hole : float, optional
            Size of the hole for a donut chart (0-1, default 0 for a normal pie)
        show_values : bool, optional
            Whether to show values on pie slices
        value_format : str, optional
            Format string for values (e.g., ".1f" for 1 decimal place)
        show_percentages : bool, optional
            Whether to show percentages on pie slices
        sort_values : bool, optional
            Whether to sort slices by value (descending)
        pull_largest : bool, optional
            Whether to pull out the largest slice
        pull_value : float, optional
            How far to pull the slice (0-1)
        clockwise : bool, optional
            Whether slices go clockwise (True) or counterclockwise (False)
        start_angle : float, optional
            Starting angle for the first slice in degrees
        textposition : str, optional
            Position of text labels ("inside", "outside", "auto")
        **kwargs:
            Additional arguments to pass to the Plotly trace

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the pie chart
        """
        try:
            # Convert data to values and labels
            if isinstance(data, dict):
                labels_list = list(data.keys())
                values_list = list(data.values())
            elif isinstance(data, pd.Series):
                labels_list = list(data.index)
                values_list = data.tolist()
            elif isinstance(data, list):
                if labels is None:
                    # Generate default labels if not provided
                    labels_list = [f"Item {i + 1}" for i in range(len(data))]
                    logger.warning("No labels provided for pie chart. Using default labels.")
                else:
                    if len(labels) != len(data):
                        logger.warning(f"Number of labels ({len(labels)}) does not match "
                                       f"number of values ({len(data)}). Using available labels.")
                    labels_list = labels[:len(data)]
                values_list = data
            else:
                raise TypeError(f"Unsupported data type for pie chart: {type(data)}")

            # Handle empty data
            if not values_list or all(v == 0 for v in values_list):
                return self.create_empty_figure(
                    title=title,
                    message="No non-zero data available for visualization"
                )

            # Sort by value if requested
            if sort_values:
                sorted_data = sorted(zip(labels_list, values_list), key=lambda x: x[1], reverse=True)
                labels_list, values_list = zip(*sorted_data)

            # Generate colors
            colors = get_theme_colors(len(values_list))

            # Create pull array for exploding slices
            pull = None
            if pull_largest:
                max_index = values_list.index(max(values_list))
                pull = [pull_value if i == max_index else 0 for i in range(len(values_list))]

            # Determine text format based on options
            if show_values and show_percentages:
                texttemplate = f'%{{label}}<br>%{{value:{value_format}}}<br>(%{{percent}})'
            elif show_values:
                texttemplate = f'%{{label}}<br>%{{value:{value_format}}}'
            elif show_percentages:
                texttemplate = '%{label}<br>(%{percent})'
            else:
                texttemplate = '%{label}'

            # Create figure
            fig = go.Figure()

            # Add pie trace
            pie_params = {
                'labels': labels_list,
                'values': values_list,
                'hole': hole,
                'marker': {'colors': colors},
                'textposition': textposition,
                'texttemplate': texttemplate,
                'direction': 'clockwise' if clockwise else 'counterclockwise',
                'rotation': start_angle,
                'pull': pull
            }

            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in ['title', 'height', 'width']:
                    pie_params[key] = value

            fig.add_trace(go.Pie(**pie_params))

            # Configure layout
            fig.update_layout(
                title=title,
                height=kwargs.get('height', 500),
                width=kwargs.get('width', 700)
            )

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return self.create_empty_figure(
                title=title,
                message=f"Error creating pie chart: {str(e)}"
            )

    def update(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Update an existing Plotly pie chart.

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
                logger.warning("Cannot update non-Plotly figure with PlotlyPieChart")
                return fig

            # Check if there are pie traces
            has_pie_trace = any(isinstance(trace, go.Pie) for trace in fig.data)
            if not has_pie_trace:
                logger.warning("No pie traces found in figure")
                return fig

            # Update title if provided
            if 'title' in kwargs:
                fig.update_layout(title=kwargs['title'])

            # Update dimensions if requested
            if 'height' in kwargs or 'width' in kwargs:
                height = kwargs.get('height', fig.layout.height)
                width = kwargs.get('width', fig.layout.width)
                fig.update_layout(height=height, width=width)

            # Update data if provided
            if 'data' in kwargs:
                # For data updates, it's simpler to recreate the pie chart
                # while preserving the layout settings

                # Get current layout settings to preserve
                current_title = fig.layout.title.text if fig.layout.title else None
                current_height = fig.layout.height
                current_width = fig.layout.width

                # Create new figure
                new_kwargs = kwargs.copy()
                if current_title and 'title' not in new_kwargs:
                    new_kwargs['title'] = current_title
                if current_height and 'height' not in new_kwargs:
                    new_kwargs['height'] = current_height
                if current_width and 'width' not in new_kwargs:
                    new_kwargs['width'] = current_width

                return self.create(**new_kwargs)

            # Update hole size if provided
            if 'hole' in kwargs:
                for trace in fig.data:
                    if isinstance(trace, go.Pie):
                        trace.hole = kwargs['hole']

            # Update text options if provided
            text_updates = {}
            if 'show_values' in kwargs or 'show_percentages' in kwargs or 'value_format' in kwargs:
                show_values = kwargs.get('show_values')
                show_percentages = kwargs.get('show_percentages')
                value_format = kwargs.get('value_format', '.1f')

                # Need to determine the current settings if not all are provided
                if show_values is None or show_percentages is None:
                    # Initialize defaults without trying to access potentially missing attributes
                    current_show_values = False
                    current_show_percentages = False

                    # Try to analyze the first trace properties safely
                    try:
                        # Get trace properties as a dict to avoid attribute access issues
                        trace_props = fig.data[0].to_plotly_json() if hasattr(fig.data[0], 'to_plotly_json') else {}

                        # Check if texttemplate exists in the properties
                        if 'texttemplate' in trace_props:
                            template_text = str(trace_props['texttemplate'])
                            current_show_values = '%{value' in template_text
                            current_show_percentages = '%{percent' in template_text
                    except Exception as e:
                        logger.warning(f"Error determining current text template settings: {e}")

                    # Use provided values or defaults
                    show_values = show_values if show_values is not None else current_show_values
                    show_percentages = show_percentages if show_percentages is not None else current_show_percentages

                # Determine new text format
                if show_values and show_percentages:
                    text_updates['texttemplate'] = f'%{{label}}<br>%{{value:{value_format}}}<br>(%{{percent}})'
                elif show_values:
                    text_updates['texttemplate'] = f'%{{label}}<br>%{{value:{value_format}}}'
                elif show_percentages:
                    text_updates['texttemplate'] = '%{label}<br>(%{percent})'
                else:
                    text_updates['texttemplate'] = '%{label}'

            # Update text position if provided
            if 'textposition' in kwargs:
                text_updates['textposition'] = kwargs['textposition']

            # Apply text updates if needed
            if text_updates:
                for trace in fig.data:
                    if isinstance(trace, go.Pie):
                        for key, value in text_updates.items():
                            setattr(trace, key, value)

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error updating pie chart: {e}")
            return fig


class PlotlySunburstChart(PlotlyFigure):
    """Sunburst chart implementation using Plotly for hierarchical data."""

    def create(
            self,
            data: Union[Dict, pd.DataFrame],
            title: str,
            path_column: Optional[str] = None,
            values_column: Optional[str] = None,
            color_column: Optional[str] = None,
            branchvalues: str = "total",  # "total" or "remainder"
            maxdepth: Optional[int] = None,
            sort_siblings: bool = False,
            **kwargs
    ) -> go.Figure:
        """
        Create a sunburst chart using Plotly.

        Parameters:
        -----------
        data : Union[Dict, pd.DataFrame]
            Data to visualize.
            If DataFrame, it should contain columns for the hierarchical path,
            values, and optionally colors.
            If Dict, it should be a hierarchical structure with nested dictionaries
            where leaf nodes have values.
        title : str
            Title for the plot
        path_column : str, optional
            For DataFrame data, the column containing the hierarchical path
            Can be a list of columns if path is split across multiple columns
        values_column : str, optional
            For DataFrame data, the column containing values
        color_column : str, optional
            For DataFrame data, the column to use for coloring
        branchvalues : str, optional
            How to sum values: "total" (default) or "remainder"
        maxdepth : int, optional
            Maximum depth to display
        sort_siblings : bool, optional
            Whether to sort siblings by value
        **kwargs:
            Additional arguments to pass to the Plotly trace

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the sunburst chart
        """
        try:
            # Create figure
            fig = go.Figure()

            # Process based on input data type
            if isinstance(data, pd.DataFrame):
                # For DataFrame input, we need columns for path, values, and optionally colors
                if path_column is None:
                    raise ValueError("path_column must be provided for DataFrame input")

                # Prepare paths - can be a single column or list of columns
                if isinstance(path_column, (list, tuple)):
                    # Path is split across multiple columns
                    paths = data[path_column].values.tolist()
                else:
                    # Path is in a single column, might need splitting
                    paths = data[path_column].tolist()
                    # Check if paths contain separators (like '/')
                    if isinstance(paths[0], str) and ('/' in paths[0] or '\\' in paths[0] or '.' in paths[0]):
                        # Split paths by common separators
                        split_char = '/' if '/' in paths[0] else ('\\' if '\\' in paths[0] else '.')
                        paths = [path.split(split_char) for path in paths]

                # Prepare values
                values = None
                if values_column:
                    values = data[values_column].tolist()

                # Prepare colors
                colors = None
                if color_column:
                    colors = data[color_column].tolist()

                # Create sunburst trace
                sunburst_params = {
                    'labels': [item for sublist in paths for item in
                               (sublist if isinstance(sublist, list) else [sublist])],
                    'parents': [''] + [item for sublist in paths[:-1] for item in
                                       (sublist if isinstance(sublist, list) else [sublist])],
                    'branchvalues': branchvalues
                }

                if values:
                    sunburst_params['values'] = values

                if colors:
                    sunburst_params['marker'] = {'colors': colors}

                if maxdepth:
                    sunburst_params['maxdepth'] = maxdepth

                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in ['title', 'height', 'width']:
                        sunburst_params[key] = value

            elif isinstance(data, dict):
                # For dictionary input (hierarchical data)
                labels, parents, values = self._process_hierarchical_dict(data, "")

                # Create sunburst trace
                sunburst_params = {
                    'labels': labels,
                    'parents': parents,
                    'values': values,
                    'branchvalues': branchvalues
                }

                if maxdepth:
                    sunburst_params['maxdepth'] = maxdepth

                # Add any additional parameters
                for key, value in kwargs.items():
                    if key not in ['title', 'height', 'width']:
                        sunburst_params[key] = value

            else:
                raise TypeError(f"Unsupported data type for sunburst chart: {type(data)}")

            # Add the trace
            fig.add_trace(go.Sunburst(**sunburst_params))

            # Configure layout
            fig.update_layout(
                title=title,
                height=kwargs.get('height', 700),
                width=kwargs.get('width', 700)
            )

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error creating sunburst chart: {e}")
            return self.create_empty_figure(
                title=title,
                message=f"Error creating sunburst chart: {str(e)}"
            )

    def _process_hierarchical_dict(self, data: Dict, parent: str = "") -> Tuple[List[str], List[str], List[float]]:
        """
        Process hierarchical dictionary data for sunburst chart.

        Parameters:
        -----------
        data : Dict
            Hierarchical dictionary where leaf nodes have values
        parent : str
            Parent node name

        Returns:
        --------
        Tuple[List[str], List[str], List[float]]
            Tuple of (labels, parents, values) for the sunburst chart
        """
        labels = []
        parents = []
        values = []

        for key, value in data.items():
            # Add this node
            labels.append(key)
            parents.append(parent)

            if isinstance(value, dict):
                # Recursive case: child is another dictionary
                child_labels, child_parents, child_values = self._process_hierarchical_dict(value, key)
                labels.extend(child_labels)
                parents.extend(child_parents)
                values.extend(child_values)

                # Set the parent node's value to the sum of its children
                values.append(sum(child_values))
            else:
                # Base case: child is a value
                values.append(value)

        return labels, parents, values

    def update(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Update an existing Plotly sunburst chart.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Existing Plotly figure to update
        **kwargs:
            Parameters to update

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated Plotly figure
        """
        try:
            # Validate figure type
            if not isinstance(fig, go.Figure):
                logger.warning("Cannot update non-Plotly figure with PlotlySunburstChart")
                return fig

            # Check if there are sunburst traces
            has_sunburst_trace = any(isinstance(trace, go.Sunburst) for trace in fig.data)
            if not has_sunburst_trace:
                logger.warning("No sunburst traces found in figure")
                return fig

            # Update title if provided
            if 'title' in kwargs:
                fig.update_layout(title=kwargs['title'])

            # Update dimensions if requested
            if 'height' in kwargs or 'width' in kwargs:
                height = kwargs.get('height', fig.layout.height)
                width = kwargs.get('width', fig.layout.width)
                fig.update_layout(height=height, width=width)

            # Update data if provided
            if 'data' in kwargs:
                # For data updates, it's simpler to recreate the chart
                # while preserving the layout settings

                # Get current layout settings to preserve
                current_title = fig.layout.title.text if fig.layout.title else None
                current_height = fig.layout.height
                current_width = fig.layout.width

                # Create new figure
                new_kwargs = kwargs.copy()
                if current_title and 'title' not in new_kwargs:
                    new_kwargs['title'] = current_title
                if current_height and 'height' not in new_kwargs:
                    new_kwargs['height'] = current_height
                if current_width and 'width' not in new_kwargs:
                    new_kwargs['width'] = current_width

                return self.create(**new_kwargs)

            # Update maxdepth if provided
            if 'maxdepth' in kwargs:
                for trace in fig.data:
                    if isinstance(trace, go.Sunburst):
                        trace.maxdepth = kwargs['maxdepth']

            # Update branchvalues if provided
            if 'branchvalues' in kwargs:
                for trace in fig.data:
                    if isinstance(trace, go.Sunburst):
                        trace.branchvalues = kwargs['branchvalues']

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error updating sunburst chart: {e}")
            return fig


# Register the implementations
FigureRegistry.register("pie", "plotly", PlotlyPieChart)
FigureRegistry.register("sunburst", "plotly", PlotlySunburstChart)