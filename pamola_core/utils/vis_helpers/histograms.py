"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Histogram Visualization Implementation
Description: Thread-safe histogram visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for histograms using both
Plotly (primary) and Matplotlib (fallback) backends to create
standardized visualizations for data distribution analysis.

Key features:
1. Context-isolated visualization state for thread-safe parallel execution
2. Support for multiple data formats (dict, DataFrame, ndarray, list)
3. Support for comparison histograms with multiple data series
4. Optional kernel density estimation (KDE) overlays
5. Customizable binning and formatting options
6. Thread-safe registry operations
7. Proper theme application using context-aware helpers
8. Comprehensive error handling with appropriate fallbacks
9. Memory-efficient operation with explicit resource cleanup
10. Robust type checking for numeric data validation

Implementation follows the PAMOLA.CORE framework with standardized interfaces for
visualization generation while ensuring concurrent operations do not interfere
with each other through proper context isolation.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

from pamola_core.utils.vis_helpers.base import (
    PlotlyFigure,
    MatplotlibFigure,
    FigureRegistry,
)
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_plotly_figure,
    apply_theme_to_matplotlib_figure,
    get_theme_colors,
)
from pamola_core.utils.vis_helpers.context import visualization_context

# Configure logger
logger = logging.getLogger(__name__)


class PlotlyHistogram(PlotlyFigure):
    """Histogram implementation using Plotly."""

    def _is_numeric_data(self, values: np.ndarray) -> bool:
        """
        Check if data contains numeric values.

        Parameters:
        -----------
        values : np.ndarray
            Array of values to check

        Returns:
        --------
        bool
            True if data is numeric, False otherwise
        """
        try:
            # Try to convert to float array
            float_values = values.astype(float)
            # Check if conversion was successful (no NaN from string conversion)
            return not np.all(np.isnan(float_values))
        except (ValueError, TypeError):
            return False

    def _prepare_data(
        self, data: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Convert input data to numpy array(s) and remove NaN values.

        If input is a dictionary with string keys, it's treated as multiple series for comparison.

        Parameters:
        -----------
        data : Dict[str, Any], pd.Series, np.ndarray, or List[float]
            Input data to prepare

        Returns:
        --------
        Union[np.ndarray, Dict[str, np.ndarray]]
            Prepared data without NaN values, or dict of prepared arrays for comparison
        """
        try:
            if isinstance(data, dict):
                # Check if dictionary keys are strings (indicating multiple series)
                if all(isinstance(k, str) for k in data.keys()):
                    # This is a comparison histogram with multiple series
                    prepared_dict = {}
                    for key, values in data.items():
                        if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                            arr = (
                                np.array(values)
                                if not isinstance(values, np.ndarray)
                                else values
                            )
                            # Only include numeric data
                            if self._is_numeric_data(arr):
                                # Remove NaN values
                                clean_arr = arr[~np.isnan(arr.astype(float))]
                                if len(clean_arr) > 0:
                                    prepared_dict[key] = clean_arr
                            else:
                                logger.warning(
                                    f"Series '{key}' contains non-numeric data, skipping"
                                )
                    return prepared_dict if prepared_dict else np.array([0])
                else:
                    # Regular dict with numeric keys - convert values only
                    values = np.array(list(data.values()))
            elif isinstance(data, pd.Series):
                values = data.values
            else:
                values = np.array(data)

            # Check if data is numeric
            if not self._is_numeric_data(values):
                logger.error("Data contains non-numeric values")
                return np.array([0])

            # Convert to float and remove NaN values
            float_values = values.astype(float)
            return float_values[~np.isnan(float_values)]

        except Exception as e:
            logger.error(f"Error preparing histogram data: {e}")
            # Return a simple array with a zero to avoid further errors
            return np.array([0])

    def _calculate_kde(
        self, values: np.ndarray, bins: int = 20, histnorm: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        try:
            # Check if we have enough data points for KDE
            if len(values) < 2:
                logger.warning("Not enough data points for KDE calculation")
                return np.array([]), np.array([])

            from scipy import stats

            # Calculate KDE
            kde_x = np.linspace(min(values), max(values), 1000)
            kde_y = stats.gaussian_kde(values)(kde_x)

            # Scale KDE to match histogram height if not normalized
            if not histnorm:
                hist, _ = np.histogram(values, bins=bins)
                if np.max(hist) > 0 and np.max(kde_y) > 0:
                    max_count = np.max(hist)
                    scaling_factor = max_count / np.max(kde_y)
                    kde_y = kde_y * scaling_factor

            return kde_x, kde_y
        except Exception as e:
            logger.error(f"Error calculating KDE: {e}")
            # Return empty arrays to avoid further errors
            return np.array([]), np.array([])

    def create(
        self,
        data: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]],
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = "Count",
        bins: int = 20,
        kde: bool = True,
        histnorm: str = "",
        cumulative: bool = False,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a histogram using Plotly.

        Supports comparison histograms when data is a dict with string keys.

        Parameters:
        -----------
        data : Dict[str, Any], pd.Series, np.ndarray, or List[float]
            Data to visualize. If dict with string keys, creates comparison histogram
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
        histnorm : str
            Normalization method (e.g., "", "percent", "probability", "density", "probability density")
        cumulative : bool
            Whether to show cumulative distribution
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional arguments to pass to go.Histogram

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with the histogram
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import plotly.graph_objects as go

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                # Prepare data
                prepared_data = self._prepare_data(data)

                # Check if we have comparison data
                is_comparison = isinstance(prepared_data, dict)

                if is_comparison:
                    # Multiple series comparison
                    if not prepared_data:
                        logger.warning("No valid data series for comparison histogram")
                        return self.create_empty_figure(
                            title=title,
                            message="No valid data series for comparison histogram",
                        )
                else:
                    # Single series
                    if len(prepared_data) == 0:
                        logger.warning("No valid data points for histogram")
                        return self.create_empty_figure(
                            title=title, message="No valid data points for histogram"
                        )

                # Create figure
                fig = go.Figure()

                # Register figure for cleanup
                register_figure(fig, context_info)

                # Get colors for multiple series
                if is_comparison:
                    colors = get_theme_colors(len(prepared_data))
                else:
                    colors = [kwargs.get("color", get_theme_colors(1)[0])]

                # Handle comparison histogram
                if is_comparison:
                    for idx, (series_name, values) in enumerate(prepared_data.items()):
                        # Add histogram trace for each series
                        fig.add_trace(
                            go.Histogram(
                                x=values,
                                nbinsx=bins,
                                histnorm=histnorm,
                                cumulative=dict(enabled=cumulative),
                                marker=dict(color=colors[idx % len(colors)]),
                                opacity=0.7,
                                name=series_name,
                                **{
                                    k: v
                                    for k, v in kwargs.items()
                                    if k not in ["color", "marker"]
                                },
                            )
                        )

                        # Add KDE trace if requested
                        if kde and len(values) >= 2:
                            kde_x, kde_y = self._calculate_kde(values, bins, histnorm)
                            if len(kde_x) > 0:
                                # Use slightly different color for KDE
                                fig.add_trace(
                                    go.Scatter(
                                        x=kde_x,
                                        y=kde_y,
                                        mode="lines",
                                        name=f"{series_name} KDE",
                                        line=dict(
                                            color=colors[idx % len(colors)],
                                            width=2,
                                            dash="dash",
                                        ),
                                        showlegend=True,
                                    )
                                )

                    # Update layout for better comparison
                    fig.update_layout(barmode="overlay")

                else:
                    # Single series histogram
                    values = prepared_data

                    # Extract and handle 'marker' from kwargs
                    marker_style = kwargs.pop("marker", None)
                    if isinstance(marker_style, dict):
                        marker = marker_style
                    else:
                        marker = {"color": colors[0]}

                    fig.add_trace(
                        go.Histogram(
                            x=values,
                            nbinsx=bins,
                            histnorm=histnorm,
                            cumulative=dict(enabled=cumulative),
                            marker=marker,
                            opacity=0.7,
                            name="Histogram",
                            **kwargs,
                        )
                    )

                    # Add KDE trace if requested
                    if kde and len(values) >= 2:
                        kde_x, kde_y = self._calculate_kde(values, bins, histnorm)
                        if len(kde_x) > 0:
                            kde_color = kwargs.get("kde_color", "rgba(255, 0, 0, 0.6)")
                            fig.add_trace(
                                go.Scatter(
                                    x=kde_x,
                                    y=kde_y,
                                    mode="lines",
                                    name="KDE",
                                    line=dict(color=kde_color, width=2),
                                )
                            )

                # Set axis labels
                fig.update_layout(
                    xaxis_title=x_label or "Value", yaxis_title=y_label or "Count"
                )

                # Set title
                fig.update_layout(title=title)

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig

            except ImportError as e:
                logger.error(
                    f"Plotly is not available. Please install it with: pip install plotly. Error: {e}"
                )
                # Try to fall back to MatplotlibHistogram
                try:
                    fallback = MatplotlibHistogram()
                    logger.warning("Falling back to Matplotlib implementation")
                    return fallback.create(
                        data=data,
                        title=title,
                        x_label=x_label,
                        y_label=y_label,
                        bins=bins,
                        kde=kde,
                        density=histnorm != "",
                        cumulative=cumulative,
                        backend=backend,
                        theme=theme,
                        strict=strict,
                        **kwargs,
                    )
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to Matplotlib also failed: {fallback_error}"
                    )
                    return self.create_empty_figure(
                        title=title,
                        message="Error creating histogram: Plotly not available and Matplotlib fallback failed",
                    )
            except Exception as e:
                logger.error(f"Error creating histogram: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating histogram: {str(e)}"
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
        Update an existing Plotly histogram.

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

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Ensure we have a Plotly figure
                if not isinstance(fig, go.Figure):
                    logger.warning(
                        "Cannot update non-Plotly figure with PlotlyHistogram"
                    )
                    return fig

                # Update title and axis labels if provided
                if "title" in kwargs:
                    fig.update_layout(title=kwargs["title"])

                if "x_label" in kwargs:
                    fig.update_layout(xaxis_title=kwargs["x_label"])

                if "y_label" in kwargs:
                    fig.update_layout(yaxis_title=kwargs["y_label"])

                # Process data if provided
                if "data" in kwargs:
                    data = kwargs["data"]
                    prepared_data = self._prepare_data(data)

                    # Check if we have comparison data
                    is_comparison = isinstance(prepared_data, dict)

                    if is_comparison:
                        # Clear existing traces for comparison update
                        fig.data = []

                        # Get colors for multiple series
                        colors = get_theme_colors(len(prepared_data))

                        # Add traces for each series
                        for idx, (series_name, values) in enumerate(
                            prepared_data.items()
                        ):
                            fig.add_trace(
                                go.Histogram(
                                    x=values,
                                    nbinsx=kwargs.get("bins", 20),
                                    histnorm=kwargs.get("histnorm", ""),
                                    cumulative=dict(
                                        enabled=kwargs.get("cumulative", False)
                                    ),
                                    marker=dict(color=colors[idx % len(colors)]),
                                    opacity=0.7,
                                    name=series_name,
                                )
                            )

                            # Add KDE if requested
                            if kwargs.get("kde", True) and len(values) >= 2:
                                kde_x, kde_y = self._calculate_kde(
                                    values,
                                    bins=kwargs.get("bins", 20),
                                    histnorm=kwargs.get("histnorm", ""),
                                )
                                if len(kde_x) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=kde_x,
                                            y=kde_y,
                                            mode="lines",
                                            name=f"{series_name} KDE",
                                            line=dict(
                                                color=colors[idx % len(colors)],
                                                width=2,
                                                dash="dash",
                                            ),
                                        )
                                    )
                    else:
                        # Single series update
                        values = prepared_data

                        # Prepare histogram update parameters
                        hist_update_params = {
                            "x": values,
                            "nbinsx": kwargs.get("bins", None),
                            "histnorm": kwargs.get("histnorm", None),
                            "cumulative": dict(enabled=kwargs.get("cumulative", None)),
                            "selector": dict(type="histogram"),
                        }
                        # Remove None values
                        hist_update_params = {
                            k: v for k, v in hist_update_params.items() if v is not None
                        }

                        # Update histogram trace
                        fig.update_traces(**hist_update_params)

                        # Handle KDE trace update
                        self._update_kde_trace(fig, values, kwargs)

                # Apply theme
                fig = apply_theme_to_plotly_figure(fig)

                return fig
            except Exception as e:
                logger.error(f"Error updating histogram: {e}")
                return fig  # Return original figure as fallback

    def _update_kde_trace(
        self, fig: Any, values: np.ndarray, kwargs: Dict[str, Any]
    ) -> None:
        """
        Update or add KDE trace to figure.

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Figure to update
        values : np.ndarray
            Data values for KDE calculation
        kwargs : Dict[str, Any]
            Update parameters
        """
        try:
            # Find existing KDE trace
            kde_trace = next(
                (
                    trace
                    for trace in fig.data
                    if getattr(trace, "name", "") == "KDE"
                    and getattr(trace, "type", "") == "scatter"
                ),
                None,
            )

            # Determine KDE behavior based on kwargs
            kde_requested = kwargs.get("kde", True)

            if kde_requested and len(values) >= 2:
                # Calculate KDE values
                kde_x, kde_y = self._calculate_kde(
                    values,
                    bins=kwargs.get("bins", 20),
                    histnorm=kwargs.get("histnorm", ""),
                )

                if len(kde_x) > 0:
                    if kde_trace is None:
                        # Add new KDE trace if it doesn't exist
                        import plotly.graph_objects as go

                        kde_color = kwargs.get("kde_color", "rgba(255, 0, 0, 0.6)")
                        fig.add_trace(
                            go.Scatter(
                                x=kde_x,
                                y=kde_y,
                                mode="lines",
                                name="KDE",
                                line=dict(color=kde_color, width=2),
                            )
                        )
                    else:
                        # Update existing KDE trace
                        fig.update_traces(
                            x=kde_x, y=kde_y, selector=dict(mode="lines", name="KDE")
                        )
            elif kde_trace is not None:
                # Remove KDE trace if it exists and kde is set to False
                fig.data = [
                    trace
                    for trace in fig.data
                    if not (
                        getattr(trace, "name", "") == "KDE"
                        and getattr(trace, "type", "") == "scatter"
                    )
                ]
        except Exception as e:
            logger.error(f"Error updating KDE trace: {e}")


class MatplotlibHistogram(MatplotlibFigure):
    """Histogram implementation using Matplotlib."""

    def _is_numeric_data(self, values: np.ndarray) -> bool:
        """
        Check if data contains numeric values.

        Parameters:
        -----------
        values : np.ndarray
            Array of values to check

        Returns:
        --------
        bool
            True if data is numeric, False otherwise
        """
        try:
            # Try to convert to float array
            float_values = values.astype(float)
            # Check if conversion was successful
            return not np.all(np.isnan(float_values))
        except (ValueError, TypeError):
            return False

    def create(
        self,
        data: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]],
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = "Count",
        bins: int = 20,
        kde: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        density: bool = False,
        cumulative: bool = False,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a histogram using Matplotlib.

        Supports comparison histograms when data is a dict with string keys.

        Parameters:
        -----------
        data : Dict[str, Any], pd.Series, np.ndarray, or List[float]
            Data to visualize. If dict with string keys, creates comparison histogram
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
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs:
            Additional arguments to pass to plt.hist

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with the histogram
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import matplotlib.pyplot as plt

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                # Prepare data
                is_comparison = isinstance(data, dict) and all(
                    isinstance(k, str) for k in data.keys()
                )

                if is_comparison:
                    # Multiple series comparison
                    prepared_dict = {}
                    for key, values in data.items():
                        if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                            arr = (
                                np.array(values)
                                if not isinstance(values, np.ndarray)
                                else values
                            )
                            # Only include numeric data
                            if self._is_numeric_data(arr):
                                # Remove NaN values
                                clean_arr = arr[~np.isnan(arr.astype(float))]
                                if len(clean_arr) > 0:
                                    prepared_dict[key] = clean_arr
                            else:
                                logger.warning(
                                    f"Series '{key}' contains non-numeric data, skipping"
                                )

                    if not prepared_dict:
                        logger.warning("No valid data series for comparison histogram")
                        return self.create_empty_figure(
                            title=title,
                            message="No valid data series for comparison histogram",
                            figsize=figsize,
                        )
                else:
                    # Single series
                    if isinstance(data, dict):
                        values = np.array(list(data.values()))
                    elif isinstance(data, pd.Series):
                        values = data.values
                    else:
                        values = np.array(data)

                    # Check if data is numeric
                    if not self._is_numeric_data(values):
                        logger.error("Data contains non-numeric values")
                        return self.create_empty_figure(
                            title=title,
                            message="Data contains non-numeric values",
                            figsize=figsize,
                        )

                    # Remove NaN values
                    values = values.astype(float)
                    values = values[~np.isnan(values)]

                    if len(values) == 0:
                        logger.warning("No valid data points for histogram")
                        return self.create_empty_figure(
                            title=title,
                            message="No valid data points for histogram",
                            figsize=figsize,
                        )

                # Create figure and axes
                fig, ax = plt.subplots(figsize=figsize)

                # Register figure for cleanup
                register_figure(fig, context_info)

                # Get colors
                if is_comparison:
                    colors = get_theme_colors(len(prepared_dict))
                else:
                    colors = [kwargs.pop("color", get_theme_colors(1)[0])]

                # Create histogram(s)
                if is_comparison:
                    # Multiple series comparison
                    for idx, (series_name, values) in enumerate(prepared_dict.items()):
                        ax.hist(
                            values,
                            bins=bins,
                            density=density,
                            cumulative=cumulative,
                            color=colors[idx % len(colors)],
                            alpha=0.5,
                            label=series_name,
                            **kwargs,
                        )

                        # Add KDE if requested
                        if kde and len(values) >= 2:
                            try:
                                from scipy import stats

                                kde_x = np.linspace(min(values), max(values), 1000)
                                kde_y = stats.gaussian_kde(values)(kde_x)

                                # Scale KDE if needed
                                if not density:
                                    hist, bin_edges = np.histogram(values, bins=bins)
                                    if np.max(hist) > 0 and np.max(kde_y) > 0:
                                        bin_width = bin_edges[1] - bin_edges[0]
                                        max_count = np.max(hist)
                                        scaling_factor = max_count / (
                                            np.max(kde_y) * bin_width * len(values)
                                        )
                                        kde_y = (
                                            kde_y
                                            * scaling_factor
                                            * len(values)
                                            * bin_width
                                        )

                                ax.plot(
                                    kde_x,
                                    kde_y,
                                    "-",
                                    lw=2,
                                    color=colors[idx % len(colors)],
                                    label=f"{series_name} KDE",
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error calculating KDE for {series_name}: {e}"
                                )

                    ax.legend()
                else:
                    # Single series histogram
                    if kde:
                        # Use Seaborn's histplot for combined histogram and KDE
                        try:
                            import seaborn as sns

                            sns.histplot(
                                values,
                                bins=bins,
                                kde=True,
                                stat="density" if density else "count",
                                cumulative=cumulative,
                                color=colors[0],
                                ax=ax,
                                **kwargs,
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
                                color=colors[0],
                                alpha=0.7,
                                **kwargs,
                            )

                            # Add KDE manually
                            if kde and len(values) >= 2:
                                try:
                                    from scipy import stats

                                    kde_x = np.linspace(min(values), max(values), 1000)
                                    kde_y = stats.gaussian_kde(values)(kde_x)

                                    # Scale KDE if needed
                                    if not density:
                                        hist, bin_edges = np.histogram(
                                            values, bins=bins
                                        )
                                        if np.max(hist) > 0 and np.max(kde_y) > 0:
                                            bin_width = bin_edges[1] - bin_edges[0]
                                            max_count = np.max(hist)
                                            scaling_factor = max_count / (
                                                np.max(kde_y) * bin_width * len(values)
                                            )
                                            kde_y = (
                                                kde_y
                                                * scaling_factor
                                                * len(values)
                                                * bin_width
                                            )

                                    ax.plot(kde_x, kde_y, "r-", lw=2, label="KDE")
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
                            color=colors[0],
                            alpha=0.7,
                            **kwargs,
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
            except Exception as e:
                logger.error(f"Error creating matplotlib histogram: {e}")
                return self.create_empty_figure(
                    title=title,
                    message=f"Error creating histogram: {str(e)}",
                    figsize=figsize,
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
        Update an existing Matplotlib histogram.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to update
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
        matplotlib.figure.Figure
            Updated figure
        """
        with visualization_context(
            backend=backend, theme=theme, strict=strict
        ) as context_info:
            try:
                import matplotlib.pyplot as plt

                # Register figure for cleanup
                from pamola_core.utils.vis_helpers.context import register_figure

                register_figure(fig, context_info)

                # Ensure we have a Matplotlib figure
                if not isinstance(fig, plt.Figure):
                    logger.warning(
                        "Cannot update non-Matplotlib figure with MatplotlibHistogram"
                    )
                    return fig

                # Get the axes (assuming single subplot)
                if len(fig.axes) == 0:
                    logger.warning("Figure has no axes to update")
                    return fig

                ax = fig.axes[0]

                # Update title if provided
                if "title" in kwargs:
                    ax.set_title(kwargs["title"])

                # Update axis labels if provided
                if "x_label" in kwargs:
                    ax.set_xlabel(kwargs["x_label"])
                if "y_label" in kwargs:
                    ax.set_ylabel(kwargs["y_label"])

                # Update data if provided
                if "data" in kwargs:
                    data = kwargs["data"]

                    # Clear axes for redraw
                    ax.clear()

                    # Recreate histogram with new data
                    # (Reuse create logic for consistency)
                    is_comparison = isinstance(data, dict) and all(
                        isinstance(k, str) for k in data.keys()
                    )

                    bins = kwargs.get("bins", 20)
                    kde = kwargs.get("kde", True)
                    density = kwargs.get("density", False)
                    cumulative = kwargs.get("cumulative", False)

                    if is_comparison:
                        # Handle comparison update
                        colors = get_theme_colors(len(data))
                        for idx, (series_name, values) in enumerate(data.items()):
                            if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                                arr = (
                                    np.array(values)
                                    if not isinstance(values, np.ndarray)
                                    else values
                                )
                                if self._is_numeric_data(arr):
                                    clean_arr = arr[~np.isnan(arr.astype(float))]
                                    if len(clean_arr) > 0:
                                        ax.hist(
                                            clean_arr,
                                            bins=bins,
                                            density=density,
                                            cumulative=cumulative,
                                            color=colors[idx % len(colors)],
                                            alpha=0.5,
                                            label=series_name,
                                        )
                        ax.legend()
                    else:
                        # Handle single series update
                        hist_color = kwargs.get("color", get_theme_colors(1)[0])

                        # Convert data to array
                        if isinstance(data, dict):
                            values = np.array(list(data.values()))
                        elif isinstance(data, pd.Series):
                            values = data.values
                        else:
                            values = np.array(data)

                        # Check and clean data
                        if self._is_numeric_data(values):
                            values = values.astype(float)
                            values = values[~np.isnan(values)]

                            if len(values) > 0:
                                ax.hist(
                                    values,
                                    bins=bins,
                                    density=density,
                                    cumulative=cumulative,
                                    color=hist_color,
                                    alpha=0.7,
                                )

                    # Reset axis labels and title
                    ax.set_xlabel(kwargs.get("x_label", "Value"))
                    ax.set_ylabel(kwargs.get("y_label", "Count"))
                    ax.set_title(kwargs.get("title", ""))

                # Apply theme
                fig = apply_theme_to_matplotlib_figure(fig)

                # Adjust layout
                plt.tight_layout()

                return fig
            except Exception as e:
                logger.error(f"Error updating matplotlib histogram: {e}")
                return fig  # Return original figure as fallback


# Register implementations
FigureRegistry.register("histogram", "plotly", PlotlyHistogram)
FigureRegistry.register("histogram", "matplotlib", MatplotlibHistogram)
