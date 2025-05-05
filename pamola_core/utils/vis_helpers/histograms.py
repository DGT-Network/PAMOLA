"""
Histogram implementations for the visualization system.

This module provides the implementations for histograms using both
Plotly and Matplotlib backends.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from pamola_core.utils.vis_helpers.base import PlotlyFigure, MatplotlibFigure, FigureRegistry
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_plotly_figure, apply_theme_to_matplotlib_figure,
    get_theme_colors
)

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyHistogram(PlotlyFigure):
    """Histogram implementation using Plotly."""

    def _prepare_data(self, data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        """
        Convert input data to numpy array and remove NaN values.

        Parameters:
        -----------
        data : Dict[str, int], pd.Series, np.ndarray, or List[float]
            Input data to prepare

        Returns:
        --------
        np.ndarray
            Prepared data without NaN values
        """
        if isinstance(data, dict):
            values = np.array(list(data.values()))
        elif isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.array(data)

        # Remove NaN values
        return values[~np.isnan(values)]

    def _calculate_kde(self, values: np.ndarray, bins: int = 20, histnorm: str = '') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Kernel Density Estimation (KDE) for histogram.

        Parameters:
        -----------
        values : np.ndarray
            Input data values
        bins : int, optional
            Number of histogram bins
        histnorm : str, optional
            Histogram normalization method

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            KDE x and y values
        """
        from scipy import stats

        # Calculate KDE
        kde_x = np.linspace(min(values), max(values), 1000)
        kde_y = stats.gaussian_kde(values)(kde_x)

        # Scale KDE to match histogram height if not normalized
        if not histnorm:
            hist, _ = np.histogram(values, bins=bins)
            max_count = np.max(hist)
            scaling_factor = max_count / np.max(kde_y)
            kde_y = kde_y * scaling_factor

        return kde_x, kde_y

    def create(
            self,
            data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]],
            title: str,
            x_label: Optional[str] = None,
            y_label: Optional[str] = "Count",
            bins: int = 20,
            kde: bool = True,
            histnorm: str = "",
            cumulative: bool = False,
            **kwargs
    ) -> 'go.Figure':
        """
        Create a histogram using Plotly.

        Parameters and returns are the same as in the original method.
        """
        import plotly.graph_objects as go

        # Prepare data
        values = self._prepare_data(data)

        if len(values) == 0:
            logger.warning("No valid data points for histogram")
            return self.create_empty_figure(
                title=title,
                message="No valid data points for histogram"
            )

        # Create figure
        fig = go.Figure()

        # Add histogram trace
        histogram_color = kwargs.get('color', get_theme_colors(1)[0])

        # Extract and handle 'marker' from kwargs
        marker_style = kwargs.pop('marker', None)
        if isinstance(marker_style, dict):
            marker = marker_style
        elif marker_style is not None:
            logger.warning(f"Invalid marker type: {type(marker_style)}. Using default marker style.")
            marker = {'color': histogram_color}
        else:
            marker = {'color': histogram_color}  # Default marker

        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=bins,
            histnorm=histnorm,
            cumulative=dict(enabled=cumulative),
            marker=marker,
            opacity=0.7,
            name='Histogram',
            **kwargs
        ))

        # Add KDE trace if requested
        if kde:
            # Calculate KDE
            kde_x, kde_y = self._calculate_kde(values, bins, histnorm)

            # Add KDE trace
            kde_color = kwargs.get('kde_color', 'rgba(255, 0, 0, 0.6)')

            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color=kde_color, width=2),
            ))

        # Set axis labels
        fig.update_layout(
            xaxis_title=x_label or "Value",
            yaxis_title=y_label or "Count"
        )

        # Set title
        fig.update_layout(title=title)

        # Apply theme
        fig = apply_theme_to_plotly_figure(fig)

        return fig

    def update(self, fig: 'go.Figure', **kwargs) -> 'go.Figure':
        """
        Update an existing Plotly histogram.

        Parameters and returns are the same as in the original method.
        """
        import plotly.graph_objects as go

        # Ensure we have a Plotly figure
        if not isinstance(fig, go.Figure):
            logger.warning("Cannot update non-Plotly figure with PlotlyHistogram")
            return fig

        # Update title and axis labels if provided
        if 'title' in kwargs:
            fig.update_layout(title=kwargs['title'])

        if 'x_label' in kwargs:
            fig.update_layout(xaxis_title=kwargs['x_label'])

        if 'y_label' in kwargs:
            fig.update_layout(yaxis_title=kwargs['y_label'])

        # Process data if provided
        if 'data' in kwargs:
            data = kwargs['data']
            values = self._prepare_data(data)

            # Prepare histogram update parameters
            hist_update_params = {
                'x': values,
                'nbinsx': kwargs.get('bins', None),
                'histnorm': kwargs.get('histnorm', None),
                'cumulative': dict(enabled=kwargs.get('cumulative', None)),
                'selector': dict(type='histogram')
            }
            # Remove None values
            hist_update_params = {k: v for k, v in hist_update_params.items() if v is not None}

            # Update histogram trace
            fig.update_traces(**hist_update_params)

            # Handle KDE (Kernel Density Estimation) trace
            kde_trace = next((trace for trace in fig.data
                              if getattr(trace, 'name', '') == 'KDE' and
                              getattr(trace, 'type', '') == 'scatter'),
                             None)

            # Determine KDE behavior based on kwargs
            kde_requested = kwargs.get('kde', True)

            if kde_requested:
                # Calculate KDE values
                kde_x, kde_y = self._calculate_kde(
                    values,
                    bins=kwargs.get('bins', 20),
                    histnorm=kwargs.get('histnorm', '')
                )

                if kde_trace is None:
                    # Add new KDE trace if it doesn't exist
                    kde_color = kwargs.get('kde_color', 'rgba(255, 0, 0, 0.6)')
                    fig.add_trace(go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode='lines',
                        name='KDE',
                        line=dict(color=kde_color, width=2),
                    ))
                else:
                    # Update existing KDE trace
                    fig.update_traces(
                        x=kde_x,
                        y=kde_y,
                        selector=dict(mode='lines', name='KDE')
                    )
            elif kde_trace is not None:
                # Remove KDE trace if it exists and kde is set to False
                fig.data = [
                    trace for trace in fig.data
                    if not (
                            getattr(trace, 'name', '') == 'KDE' and
                            getattr(trace, 'type', '') == 'scatter'
                    )
                ]

        return fig


class MatplotlibHistogram(MatplotlibFigure):
    """Histogram implementation using Matplotlib."""

    def create(self,
               data: Union[Dict[str, int], pd.Series, np.ndarray, List[float]],
               title: str,
               x_label: Optional[str] = None,
               y_label: Optional[str] = "Count",
               bins: int = 20,
               kde: bool = True,
               figsize: Tuple[int, int] = (12, 8),
               density: bool = False,
               cumulative: bool = False,
               **kwargs) -> 'plt.Figure':
        """
        Create a histogram using Matplotlib.

        Parameters:
        -----------
        data : Dict[str, int], pd.Series, np.ndarray, or List[float]
            Data to visualize
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis
        y_label : str, optional
            Label for the y-axis
        bins : int
            Number of bins for the histogram
        kde : bool
            Whether to show a kernel density estimate
        figsize : Tuple[int, int]
            Figure size (width, height)
        density : bool
            Whether to normalize the histogram
        cumulative : bool
            Whether to show cumulative distribution
        **kwargs:
            Additional arguments to pass to plt.hist

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with the histogram
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert data to array
        if isinstance(data, dict):
            values = np.array(list(data.values()))
        elif isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.array(data)

        # Remove NaN values
        values = values[~np.isnan(values)]

        if len(values) == 0:
            logger.warning("No valid data points for histogram")
            return self.create_empty_figure(
                title=title,
                message="No valid data points for histogram",
                figsize=figsize
            )

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Get colors from theme
        hist_color = kwargs.pop('color', get_theme_colors(1)[0])

        # Create histogram
        if kde:
            # Use Seaborn's distplot for combined histogram and KDE
            try:
                sns.histplot(
                    values,
                    bins=bins,
                    kde=True,
                    stat="density" if density else "count",
                    cumulative=cumulative,
                    color=hist_color,
                    ax=ax,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Error using seaborn.histplot: {e}")
                logger.warning("Falling back to matplotlib histogram")

                # Fall back to matplotlib
                ax.hist(
                    values,
                    bins=bins,
                    density=density,
                    cumulative=cumulative,
                    color=hist_color,
                    alpha=0.7,
                    **kwargs
                )

                # Add KDE if requested
                if kde:
                    try:
                        from scipy import stats

                        # Calculate KDE
                        kde_x = np.linspace(min(values), max(values), 1000)
                        kde_y = stats.gaussian_kde(values)(kde_x)

                        # Scale KDE to match histogram height if not normalized
                        if not density:
                            hist, bin_edges = np.histogram(values, bins=bins)
                            bin_width = bin_edges[1] - bin_edges[0]
                            max_count = np.max(hist)
                            scaling_factor = max_count / (np.max(kde_y) * bin_width * len(values))
                            kde_y = kde_y * scaling_factor * len(values) * bin_width

                        # Plot KDE
                        ax.plot(kde_x, kde_y, 'r-', lw=2, label='KDE')
                        ax.legend()
                    except Exception as e:
                        logger.warning(f"Error calculating KDE: {e}")
        else:
            # Create regular histogram without KDE
            ax.hist(
                values,
                bins=bins,
                density=density,
                cumulative=cumulative,
                color=hist_color,
                alpha=0.7,
                **kwargs
            )

        # Set axis labels
        ax.set_xlabel(x_label or "Value")
        ax.set_ylabel(y_label or "Count")

        # Set title
        ax.set_title(title)

        # Apply theme
        fig = apply_theme_to_matplotlib_figure(fig)

        # Adjust layout
        plt.tight_layout()

        return fig

    def update(self, fig: 'plt.Figure', **kwargs) -> 'plt.Figure':
        """
        Update an existing Matplotlib histogram.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to update
        **kwargs:
            Parameters to update

        Returns:
        --------
        matplotlib.figure.Figure
            Updated figure
        """
        import matplotlib.pyplot as plt

        # Ensure we have a Matplotlib figure
        if not isinstance(fig, plt.Figure):
            logger.warning("Cannot update non-Matplotlib figure with MatplotlibHistogram")
            return fig

        # Get the axes (assuming single subplot)
        if len(fig.axes) == 0:
            logger.warning("Figure has no axes to update")
            return fig

        ax = fig.axes[0]

        # Update title if provided
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])

        # Update axis labels if provided
        if 'x_label' in kwargs:
            ax.set_xlabel(kwargs['x_label'])
        if 'y_label' in kwargs:
            ax.set_ylabel(kwargs['y_label'])

        # Update data if provided
        if 'data' in kwargs:
            data = kwargs['data']

            # Convert data to array
            if isinstance(data, dict):
                values = np.array(list(data.values()))
            elif isinstance(data, pd.Series):
                values = data.values
            else:
                values = np.array(data)

            # Remove NaN values
            values = values[~np.isnan(values)]

            # Clear axes
            ax.clear()

            # Get histogram parameters
            bins = kwargs.get('bins', 20)
            kde = kwargs.get('kde', True)
            density = kwargs.get('density', False)
            cumulative = kwargs.get('cumulative', False)
            hist_color = kwargs.get('color', get_theme_colors(1)[0])

            # Recreate histogram
            if kde:
                # Use Seaborn's distplot for combined histogram and KDE
                try:
                    import seaborn as sns
                    sns.histplot(
                        values,
                        bins=bins,
                        kde=True,
                        stat="density" if density else "count",
                        cumulative=cumulative,
                        color=hist_color,
                        ax=ax
                    )
                except Exception as e:
                    logger.warning(f"Error using seaborn.histplot: {e}")
                    logger.warning("Falling back to matplotlib histogram")

                    # Fall back to matplotlib
                    ax.hist(
                        values,
                        bins=bins,
                        density=density,
                        cumulative=cumulative,
                        color=hist_color,
                        alpha=0.7
                    )

                    # Add KDE
                    try:
                        from scipy import stats

                        # Calculate KDE
                        kde_x = np.linspace(min(values), max(values), 1000)
                        kde_y = stats.gaussian_kde(values)(kde_x)

                        # Scale KDE to match histogram height if not normalized
                        if not density:
                            hist, bin_edges = np.histogram(values, bins=bins)
                            bin_width = bin_edges[1] - bin_edges[0]
                            max_count = np.max(hist)
                            scaling_factor = max_count / (np.max(kde_y) * bin_width * len(values))
                            kde_y = kde_y * scaling_factor * len(values) * bin_width

                        # Plot KDE
                        ax.plot(kde_x, kde_y, 'r-', lw=2, label='KDE')
                        ax.legend()
                    except Exception as e:
                        logger.warning(f"Error calculating KDE: {e}")
            else:
                # Create regular histogram without KDE
                ax.hist(
                    values,
                    bins=bins,
                    density=density,
                    cumulative=cumulative,
                    color=hist_color,
                    alpha=0.7
                )

            # Reset axis labels and title
            ax.set_xlabel(kwargs.get('x_label', 'Value'))
            ax.set_ylabel(kwargs.get('y_label', 'Count'))
            ax.set_title(kwargs.get('title', ''))

        # Apply theme
        fig = apply_theme_to_matplotlib_figure(fig)

        # Adjust layout
        plt.tight_layout()

        return fig


# Register implementations
FigureRegistry.register("histogram", "plotly", PlotlyHistogram)
FigureRegistry.register("histogram", "matplotlib", MatplotlibHistogram)