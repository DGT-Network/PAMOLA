"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Boxplot Visualization Implementation
Description: Thread-safe boxplot visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for boxplots using both
Plotly (primary) and Matplotlib (fallback) backends to create
standardized visualizations for statistical distributions.

The implementation uses contextvars via the visualization_context
to ensure thread-safe operation for concurrent execution contexts.
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


class PlotlyBoxPlot(PlotlyFigure):
    """Box plot implementation using Plotly (primary implementation)."""

    @staticmethod
    def _prepare_data_for_boxplot(
        data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
    ) -> pd.DataFrame:
        """
        Convert input data to a standardized DataFrame format.

        Parameters:
        -----------
        data : Union[Dict[str, List[float]], pd.DataFrame, pd.Series]
            Input data in various formats

        Returns:
        --------
        pd.DataFrame
            Standardized DataFrame for box plot
        """
        try:
            # Convert different input formats to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.Series):
                df = pd.DataFrame({data.name or "Value": data})
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise TypeError(f"Unsupported data type for boxplot: {type(data)}")

            # Check for empty data
            if df.empty:
                raise ValueError("Empty dataset provided")

            return df
        except Exception as e:
            logger.error(f"Error preparing data for boxplot: {e}")
            raise

    @staticmethod
    def _prepare_box_trace_args(
        column: str,
        values: Union[np.ndarray, List[float], pd.Series],
        orientation: str,
        color: str,
        kwargs: dict,
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive arguments for a box trace.

        Parameters:
        -----------
        column : str
            Column/category name
        values : Union[np.ndarray, List[float], pd.Series]
            Numerical data values
        orientation : str
            Plot orientation ('v' for vertical, 'h' for horizontal)
        color : str
            Trace color
        kwargs : dict
            Additional customization parameters

        Returns:
        --------
        Dict[str, Any]
            Prepared box trace arguments
        """
        try:
            # Convert values to numpy array, handling different input types
            if isinstance(values, pd.Series):
                values_array = values.dropna().values
            elif isinstance(values, list):
                values_array = np.array(
                    [v for v in values if v is not None and not np.isnan(v)]
                )
            elif isinstance(values, np.ndarray):
                values_array = values[~np.isnan(values)]
            else:
                raise TypeError(f"Unsupported type for values: {type(values)}")

            # Handle single data point
            if len(values_array) == 1:
                logger.warning(
                    f"Column '{column}' has only one data point. Box plot may not display properly."
                )

            # Handle empty data
            if len(values_array) == 0:
                logger.warning(f"Column '{column}' has no valid data points.")
                values_array = np.array([0])  # Use placeholder

            # Prepare base box arguments with explicit type annotation
            box_args: Dict[str, Any] = {
                "name": str(column),
                "marker_color": color,
                "line": {"width": 1},
                "opacity": kwargs.get("opacity", 0.7),
                "boxmean": kwargs.get("boxmean", True),
                "jitter": kwargs.get("jitter", 0.3),
                "pointpos": kwargs.get("pointpos", -1.5),
            }

            # Set data and orientation
            if orientation == "v":
                box_args['y'] = values_array
                box_args['x'] = [column] * len(values_array)
            else:
                box_args['x'] = values_array
                box_args['y'] = [column] * len(values_array)
                box_args['orientation'] = 'h'

            # Handle optional parameters
            box_args["boxpoints"] = (
                kwargs.get("points", "outliers")
                if kwargs.get("show_points", True)
                else False
            )
            box_args["notched"] = kwargs.get("notched", False)
            box_args["width"] = kwargs.get("box_width", 0.5)

            return box_args
        except Exception as e:
            logger.error(
                f"[_prepare_box_trace_args] column={column!r}, "
                f"orientation={orientation!r}, color={color!r}: {e}"
            )
            raise

    def create(
        self,
        data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
        title: str,
        x_label: Optional[str] = "Category",
        y_label: Optional[str] = "Value",
        orientation: str = "v",
        show_points: bool = True,
        notched: bool = False,
        points: str = "outliers",
        box_width: float = 0.5,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a box plot using Plotly.

        Parameters:
        -----------
        data : Union[Dict[str, List[float]], pd.DataFrame, pd.Series]
            Data to visualize
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis (category axis for vertical orientation)
        y_label : str, optional
            Label for the y-axis (value axis for vertical orientation)
        orientation : str
            Orientation of the boxes: "v" for vertical, "h" for horizontal
        show_points : bool
            Whether to show outlier points
        notched : bool
            Whether to show notched boxes (confidence interval around median)
        points : str
            How to show points: "outliers", "suspected", "all", or False
        box_width : float
            Width of the boxes as a fraction of the available space
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Additional parameters for customization

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly box plot figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Create figure
                fig = go.Figure()

                # Handle empty or invalid data
                if not data:
                    return self.create_empty_figure(
                        title=title, message="No data available for visualization"
                    )

                try:
                    # Prepare data for visualization
                    processed_data = self._prepare_data_for_boxplot(data)
                except Exception as e:
                    return self.create_empty_figure(
                        title=title, message=f"Error processing data: {str(e)}"
                    )

                # Generate color palette
                colors = get_theme_colors(len(processed_data.columns))

                # Create box traces
                for i, column in enumerate(processed_data.columns):
                    try:
                        # Determine color for the trace
                        color = kwargs.get(
                            f"color_{column}",
                            kwargs.get("color", colors[i % len(colors)]),
                        )

                        # Prepare box trace arguments
                        box_args = self._prepare_box_trace_args(
                            column,
                            processed_data[column].values,
                            orientation,
                            color,
                            {
                                **kwargs,
                                "show_points": show_points,
                                "notched": notched,
                                "points": points,
                                "box_width": box_width,
                            },
                        )

                        # Add box trace to the figure
                        fig.add_trace(go.Box(**box_args))
                    except Exception as e:
                        logger.warning(
                            f"Could not create box trace for column '{column}': {e}"
                        )
                        continue

                # Check if any traces were created
                if len(fig.data) == 0:
                    return self.create_empty_figure(
                        title=title, message="Could not create any valid box traces"
                    )

                # Set axis labels based on orientation
                if orientation == "v":
                    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                else:
                    fig.update_layout(xaxis_title=y_label, yaxis_title=x_label)

                # Set title, dimensions and apply theme
                fig.update_layout(
                    title=title,
                    height=kwargs.get("height", 600),
                    width=kwargs.get("width", 800),
                )
                fig = apply_theme_to_plotly_figure(fig)

                return fig

            except ImportError as imp_error:
                # Define go as None to ensure it's defined in except block
                go = None
                logger.error(
                    f"Plotly is not available. Please install it with: pip install plotly. Error: {imp_error}"
                )
                # Try to use matplotlib as fallback
                fallback = MatplotlibBoxPlot()
                logger.warning("Falling back to Matplotlib implementation")
                return fallback.create(
                    data=data,
                    title=title,
                    orientation=orientation,
                    x_label=x_label,
                    y_label=y_label,
                    show_points=show_points,
                    notched=notched,
                    backend=backend,
                    theme=theme,
                    strict=strict,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error creating box plot with Plotly: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating visualization: {str(e)}"
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
        Update an existing Plotly box plot.

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
        **kwargs :
            Update parameters (same as create method)

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
                    logger.warning("Cannot update non-Plotly figure with PlotlyBoxPlot")
                    return fig

                # Update layout elements
                layout_updates = {}
                if "title" in kwargs:
                    layout_updates["title"] = kwargs["title"]
                if "x_label" in kwargs:
                    layout_updates["xaxis_title"] = kwargs["x_label"]
                if "y_label" in kwargs:
                    layout_updates["yaxis_title"] = kwargs["y_label"]
                if "height" in kwargs:
                    layout_updates["height"] = kwargs["height"]
                if "width" in kwargs:
                    layout_updates["width"] = kwargs["width"]

                if layout_updates:
                    fig.update_layout(**layout_updates)

                # Process data update
                if "data" in kwargs:
                    try:
                        # Prepare data and get orientation
                        processed_data = self._prepare_data_for_boxplot(kwargs["data"])
                        orientation = kwargs.get("orientation", "v")

                        # Get existing trace names
                        existing_names = [
                            getattr(trace, "name", "") for trace in fig.data
                        ]

                        # Determine if we need to rebuild traces
                        rebuild_traces = set(existing_names) != set(
                            processed_data.columns
                        ) or len(existing_names) != len(processed_data.columns)

                        # Colors for traces
                        colors = get_theme_colors(len(processed_data.columns))

                        if rebuild_traces:
                            # Clear and recreate all traces
                            fig.data = []

                            for i, column in enumerate(processed_data.columns):
                                try:
                                    # Determine color
                                    color = kwargs.get(
                                        f"color_{column}",
                                        kwargs.get("color", colors[i % len(colors)]),
                                    )

                                    # Prepare box trace arguments
                                    box_args = self._prepare_box_trace_args(
                                        column,
                                        processed_data[column].values,
                                        orientation,
                                        color,
                                        kwargs,
                                    )

                                    # Add trace
                                    fig.add_trace(go.Box(**box_args))
                                except Exception as e:
                                    logger.warning(
                                        f"Could not create box trace for column '{column}': {e}"
                                    )
                                    continue
                        else:
                            # Update existing traces
                            for i, column in enumerate(processed_data.columns):
                                try:
                                    if column in existing_names:
                                        trace_idx = existing_names.index(column)

                                        # Update data safely
                                        if orientation == "v":
                                            fig.data[trace_idx]["y"] = (
                                                processed_data[column].dropna().values
                                            )
                                        else:
                                            fig.data[trace_idx]["x"] = (
                                                processed_data[column].dropna().values
                                            )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not update trace for column '{column}': {e}"
                                    )
                                    continue
                    except Exception as e:
                        logger.error(f"Error updating box plot data: {e}")

                # Reapply theme to ensure consistency
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
                logger.error(f"Error updating box plot: {e}")
                return fig


class MatplotlibBoxPlot(MatplotlibFigure):
    """
    Box plot implementation using Matplotlib.

    Note: This is a fallback implementation. The primary implementation uses Plotly.
    """

    def create(
        self,
        data: Union[Dict[str, List[float]], pd.DataFrame, pd.Series],
        title: str,
        x_label: Optional[str] = "Category",
        y_label: Optional[str] = "Value",
        orientation: str = "v",
        show_points: bool = True,
        notched: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a box plot using Matplotlib.

        Parameters:
        -----------
        data : Union[Dict[str, List[float]], pd.DataFrame, pd.Series]
            Data to visualize
        title : str
            Title for the plot
        x_label : str, optional
            Label for the x-axis (category axis for vertical orientation)
        y_label : str, optional
            Label for the y-axis (value axis for vertical orientation)
        orientation : str
            Orientation of the boxes: "v" for vertical, "h" for horizontal
        show_points : bool
            Whether to show outlier points
        notched : bool
            Whether to show notched boxes (confidence interval around median)
        figsize : Tuple[int, int], optional
            Figure dimensions
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Additional parameters for customization

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib box plot figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Handle empty or invalid data
                if not data:
                    return self.create_empty_figure(
                        title=title,
                        message="No data available for visualization",
                        figsize=figsize,
                    )

                # Prepare data for visualization
                try:
                    if isinstance(data, dict):
                        df = pd.DataFrame(data)
                    elif isinstance(data, pd.Series):
                        df = pd.DataFrame({data.name or "Value": data})
                    elif isinstance(data, pd.DataFrame):
                        df = data
                    else:
                        raise TypeError(
                            f"Unsupported data type for boxplot: {type(data)}"
                        )

                    # Check for empty data
                    if df.empty:
                        return self.create_empty_figure(
                            title=title,
                            message="Empty dataset provided",
                            figsize=figsize,
                        )
                except Exception as e:
                    return self.create_empty_figure(
                        title=title,
                        message=f"Error processing data: {str(e)}",
                        figsize=figsize,
                    )

                # Create figure and axes
                fig, ax = plt.subplots(figsize=figsize)

                # Get colors from theme
                colors = get_theme_colors(len(df.columns))

                # Prepare boxplot parameters
                boxplot_kwargs = {
                    "notch": notched,
                    "patch_artist": True,
                    "showfliers": show_points,
                    "showmeans": kwargs.get("boxmean", True),
                    "widths": kwargs.get("box_width", 0.5),
                }

                # Create box plot based on orientation
                if orientation == "v":
                    # For vertical orientation
                    bp = ax.boxplot(
                        [df[col].dropna().values for col in df.columns],
                        vert=True,
                        **boxplot_kwargs,
                    )
                    # Set labels after creating the boxplot
                    ax.set_xticklabels(df.columns)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    plt.xticks(rotation=45, ha="right")
                else:
                    # For horizontal orientation
                    bp = ax.boxplot(
                        [df[col].dropna().values for col in df.columns],
                        vert=False,
                        **boxplot_kwargs,
                    )
                    # Set labels after creating the boxplot
                    ax.set_yticklabels(df.columns)
                    ax.set_xlabel(y_label)
                    ax.set_ylabel(x_label)

                # Color the boxes
                for i, box in enumerate(bp["boxes"]):
                    box_color = kwargs.get(
                        f"color_{df.columns[i]}",
                        kwargs.get("color", colors[i % len(colors)]),
                    )
                    box.set(facecolor=box_color, alpha=kwargs.get("opacity", 0.7))

                # Set title and apply theme
                ax.set_title(title)
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
                logger.error(f"Error creating box plot with Matplotlib: {e}")
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
        Update an existing Matplotlib box plot.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Existing Matplotlib figure to update
        backend : Optional[str]
            Backend to use: "plotly" or "matplotlib" (overrides global setting)
        theme : Optional[str]
            Theme to use for the visualization
        strict : bool
            If True, raise exceptions for invalid configuration; otherwise log warnings
        **kwargs :
            Update parameters (same as create method)

        Returns:
        --------
        matplotlib.figure.Figure
            Updated Matplotlib figure
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                # Validate figure type
                if not hasattr(fig, "axes"):
                    logger.warning(
                        "Cannot update non-Matplotlib figure with MatplotlibBoxPlot"
                    )
                    return fig

                # Ensure figure has axes
                if len(fig.axes) == 0:
                    logger.warning("Figure has no axes to update")
                    return fig

                ax = fig.axes[0]

                # Update simple properties
                if "title" in kwargs:
                    ax.set_title(kwargs["title"])
                if "x_label" in kwargs:
                    ax.set_xlabel(kwargs["x_label"])
                if "y_label" in kwargs:
                    ax.set_ylabel(kwargs["y_label"])

                # For data updates, it's usually simpler to recreate the boxplot
                # rather than trying to update the existing one
                if "data" in kwargs:
                    # Clear the axes
                    ax.clear()

                    # Get the new data
                    data = kwargs["data"]

                    try:
                        # Process data
                        if isinstance(data, dict):
                            df = pd.DataFrame(data)
                        elif isinstance(data, pd.Series):
                            df = pd.DataFrame({data.name or "Value": data})
                        elif isinstance(data, pd.DataFrame):
                            df = data
                        else:
                            raise TypeError(f"Unsupported data type: {type(data)}")

                        # Get orientation
                        orientation = kwargs.get("orientation", "v")

                        # Get colors
                        colors = get_theme_colors(len(df.columns))

                        # Prepare boxplot parameters
                        boxplot_kwargs = {
                            "notch": kwargs.get("notched", False),
                            "patch_artist": True,
                            "showfliers": kwargs.get("show_points", True),
                            "showmeans": kwargs.get("boxmean", True),
                            "widths": kwargs.get("box_width", 0.5),
                        }

                        # Create new boxplot
                        if orientation == "v":
                            bp = ax.boxplot(
                                [df[col].dropna().values for col in df.columns],
                                vert=True,
                                **boxplot_kwargs,
                            )
                            # Set labels after creating the boxplot
                            ax.set_xticklabels(df.columns)
                            ax.set_xlabel(kwargs.get("x_label", "Category"))
                            ax.set_ylabel(kwargs.get("y_label", "Value"))
                            plt.xticks(rotation=45, ha="right")
                        else:
                            bp = ax.boxplot(
                                [df[col].dropna().values for col in df.columns],
                                vert=False,
                                **boxplot_kwargs,
                            )
                            # Set labels after creating the boxplot
                            ax.set_yticklabels(df.columns)
                            ax.set_xlabel(kwargs.get("y_label", "Value"))
                            ax.set_ylabel(kwargs.get("x_label", "Category"))

                        # Color the boxes
                        for i, box in enumerate(bp["boxes"]):
                            box_color = kwargs.get(
                                f"color_{df.columns[i]}",
                                kwargs.get("color", colors[i % len(colors)]),
                            )
                            box.set(
                                facecolor=box_color, alpha=kwargs.get("opacity", 0.7)
                            )

                    except Exception as e:
                        logger.error(f"Error updating box plot data: {e}")

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
                logger.error(f"Error updating box plot: {e}")
                return fig


# Register plot implementations
FigureRegistry.register("boxplot", "plotly", PlotlyBoxPlot)
FigureRegistry.register("boxplot", "matplotlib", MatplotlibBoxPlot)
