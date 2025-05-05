"""
Base classes and utilities for the visualization subsystem.

This module provides the foundation for the visualization system, including:
- Abstract base classes for figure creation
- Factory methods for obtaining the appropriate figure implementations
- Backend detection and management
- Common utilities used across visualization types
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Type

import pandas as pd
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Global backend setting
_CURRENT_BACKEND = "plotly"  # Default to plotly
_AVAILABLE_BACKENDS = ["plotly", "matplotlib"]


def set_backend(backend: str) -> None:
    """
    Set the global visualization backend.

    Parameters:
    -----------
    backend : str
        Backend to use: "plotly" or "matplotlib"
    """
    global _CURRENT_BACKEND
    if backend.lower() not in _AVAILABLE_BACKENDS:
        logger.warning(f"Unsupported backend: {backend}. Falling back to plotly.")
        _CURRENT_BACKEND = "plotly"
    else:
        _CURRENT_BACKEND = backend.lower()


def get_backend() -> str:
    """
    Get the current visualization backend.

    Returns:
    --------
    str
        Current backend name
    """
    return _CURRENT_BACKEND


class BaseFigure(ABC):
    """Base abstract class for all visualization figures."""

    @abstractmethod
    def create(self, **kwargs) -> Any:
        """
        Create and return a figure.

        Returns:
        --------
        Figure object
            Plotly or Matplotlib figure
        """
        pass

    @abstractmethod
    def update(self, fig: Any, **kwargs) -> Any:
        """
        Update an existing figure.

        Parameters:
        -----------
        fig : Any
            Figure to update

        Returns:
        --------
        Figure object
            Updated figure
        """
        pass

    @abstractmethod
    def create_empty_figure(self, title: str, message: str,
                            figsize: Tuple[int, int] = (8, 6)) -> Any:
        """
        Create an empty figure with an error or info message.

        Parameters:
        -----------
        title : str
            Figure title
        message : str
            Message to display
        figsize : Tuple[int, int]
            Figure size

        Returns:
        --------
        Figure object
            Plotly or Matplotlib figure with message
        """
        pass


class PlotlyFigure(BaseFigure):
    """Base class for Plotly figures."""

    def create_empty_figure(self, title: str, message: str,
                            figsize: Tuple[int, int] = (8, 6)) -> Any:
        """
        Create an empty Plotly figure with a message.

        Parameters:
        -----------
        title : str
            Figure title
        message : str
            Message to display
        figsize : Tuple[int, int]
            Figure size (for compatibility, not used directly in Plotly)

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure with message
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title=title,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )

            return fig
        except ImportError:
            logger.error("Plotly is not available. Please install it with: pip install plotly")
            raise


class MatplotlibFigure(BaseFigure):
    """Base class for Matplotlib figures."""

    def create_empty_figure(self, title: str, message: str,
                            figsize: Tuple[int, int] = (8, 6)) -> Any:
        """
        Create an empty Matplotlib figure with a message.

        Parameters:
        -----------
        title : str
            Figure title
        message : str
            Message to display
        figsize : Tuple[int, int]
            Figure size

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with message
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
            ax.set_title(title)
            ax.axis('off')

            return fig
        except ImportError:
            logger.error("Matplotlib is not available. Please install it with: pip install matplotlib")
            raise


class FigureRegistry:
    """Registry for figure implementations."""

    _registry = {
        'plotly': {},
        'matplotlib': {}
    }

    @classmethod
    def register(cls, figure_type: str, backend: str, implementation: Type[BaseFigure]) -> None:
        """
        Register a figure implementation.

        Parameters:
        -----------
        figure_type : str
            Type of figure (e.g., "bar", "histogram")
        backend : str
            Backend name ("plotly" or "matplotlib")
        implementation : Type[BaseFigure]
            Class that implements the figure type
        """
        if backend not in cls._registry:
            logger.warning(f"Unknown backend: {backend}. Registration skipped.")
            return

        cls._registry[backend][figure_type] = implementation

    @classmethod
    def get(cls, figure_type: str, backend: str) -> Type[BaseFigure]:
        """
        Get figure implementation for a specific type and backend.

        Parameters:
        -----------
        figure_type : str
            Type of figure
        backend : str
            Backend name

        Returns:
        --------
        Type[BaseFigure]
            Figure implementation class
        """
        # Handle special case for base figure
        if figure_type == 'base':
            return PlotlyFigure if backend == 'plotly' else MatplotlibFigure

        # Check if the requested backend is available
        if backend not in cls._registry:
            logger.warning(f"Unknown backend: {backend}. Falling back to plotly.")
            backend = 'plotly'

        # Check if the figure type is implemented for the backend
        if figure_type not in cls._registry[backend]:
            # First try the other backend if available
            other_backend = 'matplotlib' if backend == 'plotly' else 'plotly'
            if figure_type in cls._registry[other_backend]:
                logger.warning(f"Figure type '{figure_type}' not found for backend '{backend}'. "
                               f"Falling back to '{other_backend}'.")
                return cls._registry[other_backend][figure_type]

            # If neither backend has the implementation, use a base implementation
            logger.warning(f"Figure type '{figure_type}' not implemented for any backend. "
                           f"Using base implementation.")
            return PlotlyFigure if backend == 'plotly' else MatplotlibFigure

        return cls._registry[backend][figure_type]


class FigureFactory:
    """Factory for creating figure instances."""

    def create_figure(self, figure_type: str, backend: Optional[str] = None) -> BaseFigure:
        """
        Create a figure instance of the specified type.

        Parameters:
        -----------
        figure_type : str
            Type of figure to create
        backend : str, optional
            Backend to use (defaults to global setting)

        Returns:
        --------
        BaseFigure
            Figure instance
        """
        # Use specified backend or fall back to global setting
        backend = backend or get_backend()

        try:
            # Get the implementation class
            implementation_class = FigureRegistry.get(figure_type, backend)

            # Create and return an instance
            return implementation_class()
        except Exception as e:
            logger.error(f"Error creating figure: {e}")
            # Fallback to Plotly base figure if possible
            return PlotlyFigure()


# Utility functions for data preparation
def ensure_series(data: Union[Dict[str, Any], pd.Series, List, np.ndarray]) -> pd.Series:
    """
    Ensure data is a pandas Series.

    Parameters:
    -----------
    data : Union[Dict[str, Any], pd.Series, List, np.ndarray]
        Input data

    Returns:
    --------
    pd.Series
        Data as a pandas Series
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, dict):
        return pd.Series(data)
    elif isinstance(data, (list, np.ndarray)):
        return pd.Series(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to pandas Series")


def sort_series(series: pd.Series, sort_by: str = "value", ascending: bool = False,
                max_items: Optional[int] = None) -> pd.Series:
    """
    Sort a pandas Series.

    Parameters:
    -----------
    series : pd.Series
        Series to sort
    sort_by : str
        Sort by "value" or "key"
    ascending : bool
        Whether to sort in ascending order
    max_items : int, optional
        Maximum number of items to include after sorting

    Returns:
    --------
    pd.Series
        Sorted series
    """
    if sort_by == "value":
        sorted_series = series.sort_values(ascending=ascending)
    elif sort_by == "key":
        sorted_series = series.sort_index(ascending=ascending)
    else:
        logger.warning(f"Unknown sort_by value: {sort_by}. Using 'value'.")
        sorted_series = series.sort_values(ascending=ascending)

    # Limit to max_items if specified
    if max_items is not None and len(sorted_series) > max_items:
        sorted_series = sorted_series.iloc[:max_items]

    return sorted_series


def prepare_dataframe(data: Union[Dict[str, List], pd.DataFrame],
                      orient: str = "dict") -> pd.DataFrame:
    """
    Prepare a DataFrame from various input formats.

    Parameters:
    -----------
    data : Union[Dict[str, List], pd.DataFrame]
        Input data
    orient : str
        How to interpret the dictionary: "dict" or "records"

    Returns:
    --------
    pd.DataFrame
        DataFrame prepared from input
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        if orient == "dict":
            return pd.DataFrame(data)
        elif orient == "records":
            return pd.DataFrame.from_records(data)
        else:
            logger.warning(f"Unknown orient value: {orient}. Using 'dict'.")
            return pd.DataFrame(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to pandas DataFrame")