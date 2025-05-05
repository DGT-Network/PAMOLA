"""
Correlation matrix visualization implementation.

This module provides implementation for correlation matrix visualization
using Plotly backend.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pamola_core.utils.vis_helpers.base import PlotlyFigure, FigureRegistry
from pamola_core.utils.vis_helpers.cor_utils import (
    prepare_correlation_data, create_correlation_mask, apply_mask,
    create_text_colors_array, create_significance_mask, prepare_hover_texts,
    parse_annotation_format, calculate_symmetric_colorscale_range
)
from pamola_core.utils.vis_helpers.theme import apply_theme_to_plotly_figure

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyCorrelationMatrix(PlotlyFigure):
    """Correlation matrix implementation using Plotly."""

    def create(self,
               data: Union[pd.DataFrame, np.ndarray],
               title: str,
               x_label: Optional[str] = None,
               y_label: Optional[str] = None,
               colorscale: Optional[str] = None,
               annotate: bool = True,
               annotation_format: str = ".2f",
               mask_upper: bool = False,
               mask_diagonal: bool = False,
               colorbar_title: Optional[str] = "Correlation",
               significant_threshold: Optional[float] = None,
               method_labels: Optional[Dict[str, str]] = None,
               **kwargs) -> go.Figure:
        """
        Create a correlation matrix visualization using Plotly.

        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            Correlation matrix data. If DataFrame, index and columns used as labels.
            If ndarray, numeric indices are used.
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis
        y_label : str, optional
            Label for the y-axis
        colorscale : str, optional
            Colorscale to use (default from theme if None)
        annotate : bool
            Whether to annotate the heatmap with correlation values
        annotation_format : str
            Format string for annotations (e.g., ".2f" for 2 decimal places)
        mask_upper : bool
            Whether to mask the upper triangle (above diagonal)
        mask_diagonal : bool
            Whether to mask the diagonal
        colorbar_title : str, optional
            Title for the colorbar
        significant_threshold : float, optional
            Threshold to highlight significant correlations
        method_labels : Dict[str, str], optional
            Dictionary mapping method codes to display labels (for mixed correlation methods)
        **kwargs:
            Additional arguments to pass to go.Heatmap

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the correlation matrix

        Raises:
        -------
        ValueError
            If the input data is empty or invalid
        """
        try:
            # Process input data to a standard format
            matrix, x_labels, y_labels, methods = prepare_correlation_data(data)

            # Check for empty data
            if matrix.size == 0:
                return self.create_empty_figure(
                    title=title,
                    message="No data to display in correlation matrix"
                )

            # Create mask based on parameters
            mask = create_correlation_mask(matrix, mask_upper, mask_diagonal)

            # Apply mask (set masked values to NaN)
            masked_matrix = apply_mask(matrix, mask)

            # Create figure
            fig = go.Figure()

            # Get colorscale from theme if not provided
            theme_colorscale = kwargs.pop('theme_colorscale', colorscale)
            if theme_colorscale is None:
                theme_colorscale = 'RdBu_r'  # Default for correlation (blue-white-red)

            # Handle hover information
            hover_info = 'z'
            hover_text = None

            if methods is not None and method_labels is not None:
                hover_text = prepare_hover_texts(
                    matrix, x_labels, y_labels, methods, method_labels
                )
                hover_info = 'text'

            # Handle text annotations
            text_template = None
            text_colors = None

            if annotate:
                text_template = parse_annotation_format(annotation_format)
                text_colors = create_text_colors_array(matrix)

            # Define colorscale range
            vmin, vmax = None, None
            if kwargs.pop('custom_colorscale', None) is None and theme_colorscale == 'RdBu_r':
                vmin, vmax = calculate_symmetric_colorscale_range(matrix)
            else:
                vmin = kwargs.pop('zmin', None)
                vmax = kwargs.pop('zmax', None)

            # Build heatmap arguments
            heatmap_args = {
                'z': masked_matrix,
                'x': x_labels,
                'y': y_labels,
                'colorscale': theme_colorscale,
                'zmid': 0,  # Center colorscale at zero correlation
                'zmin': vmin,
                'zmax': vmax,
                'hoverongaps': False,
                'showscale': True,
                'colorbar': {'title': colorbar_title} if colorbar_title else None,
            }

            # Add text-related arguments if annotations are requested
            if annotate:
                heatmap_args.update({
                    'texttemplate': text_template,
                    'textfont': {'color': text_colors.flatten()},
                })

            # Add hover information
            if hover_text is not None:
                heatmap_args.update({
                    'hovertext': hover_text,
                    'hoverinfo': hover_info,
                })

            # Update with any additional keyword arguments
            heatmap_args.update(kwargs)

            # Add heatmap trace
            fig.add_trace(go.Heatmap(**heatmap_args))

            # Highlight significant correlations if threshold is provided
            if significant_threshold is not None:
                # Create a mask for significant correlations
                significant_mask = create_significance_mask(
                    matrix, masked_matrix, significant_threshold
                )

                if np.any(significant_mask):
                    # Add markers for significant correlations
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            if significant_mask[i, j]:
                                fig.add_shape(
                                    type="rect",
                                    x0=j - 0.5,
                                    y0=i - 0.5,
                                    x1=j + 0.5,
                                    y1=i + 0.5,
                                    line=dict(color="black", width=2),
                                    fillcolor="rgba(0,0,0,0)",
                                )

            # Configure layout
            layout_args = {
                'title': title,
                'xaxis': {
                    'title': x_label,
                    'side': 'bottom',
                    'tickangle': -45 if len(x_labels) > 5 else 0
                },
                'yaxis': {
                    'title': y_label,
                    'autorange': 'reversed'  # To have the first row at the top
                },
                'margin': {'t': 80, 'b': 100, 'l': 100, 'r': 100}
            }

            fig.update_layout(**layout_args)

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return self.create_empty_figure(
                title=title,
                message=f"Error creating correlation matrix: {str(e)}"
            )

    def update(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Update an existing Plotly correlation matrix.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Plotly figure to update
        **kwargs:
            Parameters to update. Supported parameters include:
            - data: New correlation matrix data
            - title: New title for the plot
            - x_label/y_label: New axis labels
            - colorscale: New colorscale
            - annotate: Whether to show annotations
            - annotation_format: Format for annotation values
            - mask_upper/mask_diagonal: Masking options
            - colorbar_title: Title for the colorbar
            - significant_threshold: Threshold for highlighting significant correlations

        Returns:
        --------
        plotly.graph_objects.Figure
            Updated figure
        """
        try:
            # Validate input figure type
            if not isinstance(fig, go.Figure):
                logger.warning("Cannot update non-Plotly figure with PlotlyCorrelationMatrix")
                return fig

            # Update title if provided
            if 'title' in kwargs:
                fig.update_layout(title=kwargs['title'])

            # Update axis labels if provided
            if 'x_label' in kwargs:
                fig.update_layout(xaxis_title=kwargs['x_label'])
            if 'y_label' in kwargs:
                fig.update_layout(yaxis_title=kwargs['y_label'])

            # Update data if provided
            if 'data' in kwargs:
                data = kwargs['data']

                try:
                    # Process new data
                    matrix, x_labels, y_labels, methods = prepare_correlation_data(data)

                    # Create mask based on parameters
                    mask_upper = kwargs.get('mask_upper', False)
                    mask_diagonal = kwargs.get('mask_diagonal', False)
                    mask = create_correlation_mask(matrix, mask_upper, mask_diagonal)

                    # Apply mask (set masked values to NaN)
                    masked_matrix = apply_mask(matrix, mask)

                    # Update heatmap data
                    fig.update_traces(
                        z=masked_matrix,
                        x=x_labels,
                        y=y_labels,
                        selector=dict(type='heatmap')
                    )

                    # Update colorscale if provided
                    if 'colorscale' in kwargs:
                        fig.update_traces(colorscale=kwargs['colorscale'], selector=dict(type='heatmap'))

                    # Update annotations if needed
                    annotate = kwargs.get('annotate', None)
                    if annotate is not None:
                        annotation_format = kwargs.get('annotation_format', '.2f')

                        if annotate:
                            # Prepare text formatting
                            text_template = parse_annotation_format(annotation_format)
                            text_colors = create_text_colors_array(matrix)

                            # Update with annotations
                            fig.update_traces(
                                texttemplate=text_template,
                                textfont={'color': text_colors.flatten()},
                                selector=dict(type='heatmap')
                            )
                        else:
                            # Remove annotations
                            fig.update_traces(
                                texttemplate=None,
                                selector=dict(type='heatmap')
                            )

                    # Update colorbar title if provided
                    if 'colorbar_title' in kwargs:
                        fig.update_traces(
                            colorbar={'title': kwargs['colorbar_title']},
                            selector=dict(type='heatmap')
                        )

                    # Handle significant correlations highlighting
                    significant_threshold = kwargs.get('significant_threshold')
                    if significant_threshold is not None:
                        # Remove existing significant correlation shapes
                        shapes = []
                        for shape in fig.layout.shapes or []:
                            # Keep shapes that are not our correlation markers
                            # Our correlation markers have specific attributes
                            if not (shape.line and
                                    shape.line.color == "black" and
                                    shape.line.width == 2 and
                                    shape.fillcolor == "rgba(0,0,0,0)"):
                                shapes.append(shape)

                        # Create a mask for significant correlations using a valid numeric threshold
                        significant_mask = create_significance_mask(
                            matrix, masked_matrix, float(significant_threshold)
                        )

                        # Add new significant correlation shapes
                        if np.any(significant_mask):
                            for i in range(matrix.shape[0]):
                                for j in range(matrix.shape[1]):
                                    if significant_mask[i, j]:
                                        shapes.append(dict(
                                            type="rect",
                                            x0=j - 0.5,
                                            y0=i - 0.5,
                                            x1=j + 0.5,
                                            y1=i + 0.5,
                                            line=dict(color="black", width=2),
                                            fillcolor="rgba(0,0,0,0)",
                                        ))

                        # Update shapes
                        fig.update_layout(shapes=shapes)
                except Exception as e:
                    logger.error(f"Error updating correlation matrix data: {e}")

            # Apply theme
            fig = apply_theme_to_plotly_figure(fig)

            return fig

        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
            return fig


# Register this figure type with the registry
FigureRegistry.register("correlation_matrix", "plotly", PlotlyCorrelationMatrix)