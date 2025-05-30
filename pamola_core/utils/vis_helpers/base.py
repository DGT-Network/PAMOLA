"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Visualization Base System
Description: Thread-safe foundation for visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides the foundation for the visualization system, including:
- Abstract base classes for figure creation
- Factory methods for obtaining appropriate figure implementations
- Thread-safe backend detection and management
- Common utilities used across visualization types

The implementation uses contextvars to ensure that backend settings are properly
isolated between concurrent execution contexts, eliminating state interference
when multiple visualization operations run in parallel.
"""

import logging
import contextvars
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Type

import pandas as pd
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Define a context variable for backend storage
# This replaces the global _CURRENT_BACKEND variable with a context-isolated version
_backend_context = contextvars.ContextVar("current_backend", default="plotly")

# Available backends and backend aliases
_AVAILABLE_BACKENDS = ["plotly", "matplotlib"]
_BACKEND_ALIASES = {
    "default": "plotly",  # Map 'default' to plotly since it's our primary backend
    "mpl": "matplotlib",
    "matplot": "matplotlib",
}


def set_backend(backend: str, strict: bool = False) -> None:
    """
    Set the visualization backend for the current execution context.

    This function uses context variables to ensure that backend settings
    are isolated between concurrent execution contexts, preventing
    interference when multiple visualization operations run in parallel.

    Parameters:
    -----------
    backend : str
        Backend to use: "plotly" or "matplotlib"
    strict : bool
        If True, raise exceptions for invalid backends; otherwise log warnings

    Raises:
    -------
    ValueError
        If strict=True and backend is not supported
    """
    # Normalize backend name
    backend_lower = backend.lower()

    # Check if it's an alias and map it to a supported backend
    if backend_lower in _BACKEND_ALIASES:
        actual_backend = _BACKEND_ALIASES[backend_lower]
        logger.debug(f"Mapping backend alias '{backend}' to '{actual_backend}'")
        backend_lower = actual_backend

    # Set the backend if supported
    if backend_lower not in _AVAILABLE_BACKENDS:
        error_msg = f"Unsupported backend: {backend}. Supported backends are: {', '.join(_AVAILABLE_BACKENDS)}"
        if strict:
            raise ValueError(error_msg)
        else:
            logger.warning(f"{error_msg} Falling back to plotly.")
            _backend_context.set("plotly")
    else:
        _backend_context.set(backend_lower)
        logger.debug(f"Backend set to '{backend_lower}' in current context")


def get_backend() -> str:
    """
    Get the current visualization backend for the current execution context.

    Returns:
    --------
    str
        Current backend name
    """
    return _backend_context.get()


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
    def create_empty_figure(
        self, title: str, message: str, figsize: Tuple[int, int] = (8, 6)
    ) -> Any:
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

    def create_empty_figure(
        self, title: str, message: str, figsize: Tuple[int, int] = (8, 6)
    ) -> Any:
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
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )
            fig.update_layout(
                title=title,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )

            return fig
        except ImportError:
            logger.error(
                "Plotly is not available. Please install it with: pip install plotly"
            )
            raise
        except Exception as e:
            logger.error(f"Error creating empty Plotly figure: {e}")
            # Create a minimal fallback figure
            import plotly.graph_objects as go

            return go.Figure()


class MatplotlibFigure(BaseFigure):
    """Base class for Matplotlib figures."""

    def create_empty_figure(
        self, title: str, message: str, figsize: Tuple[int, int] = (8, 6)
    ) -> Any:
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
            ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
            ax.set_title(title)
            ax.axis("off")

            return fig
        except ImportError:
            logger.error(
                "Matplotlib is not available. Please install it with: pip install matplotlib"
            )
            raise
        except Exception as e:
            logger.error(f"Error creating empty Matplotlib figure: {e}")
            # Try to create a minimal fallback figure
            try:
                import matplotlib.pyplot as plt

                return plt.figure()
            except:
                raise


class FigureRegistry:
    """Thread-safe registry for figure implementations."""

    # Class-level registry which is shared across all instances
    # but individual accesses are isolated by backend context
    _registry = {"plotly": {}, "matplotlib": {}}

    # Add a thread lock for registry operations
    _lock = threading.RLock()

    @classmethod
    def register(
        cls, figure_type: str, backend: str, implementation: Type[BaseFigure]
    ) -> None:
        """
        Register a figure implementation in a thread-safe manner.

        Parameters:
        -----------
        figure_type : str
            Type of figure (e.g., "bar", "histogram")
        backend : str
            Backend name ("plotly" or "matplotlib")
        implementation : Type[BaseFigure]
            Class that implements the figure type
        """
        # Use lock to ensure thread-safe registry operations
        with cls._lock:
            # Normalize backend name and handle aliases
            backend_lower = backend.lower()
            if backend_lower in _BACKEND_ALIASES:
                backend_lower = _BACKEND_ALIASES[backend_lower]
                logger.debug(f"Mapped backend alias '{backend}' to '{backend_lower}'")

            # Check if the backend is supported
            if backend_lower not in _AVAILABLE_BACKENDS:
                # Don't log warning for known aliases to reduce noise
                if backend.lower() not in _BACKEND_ALIASES:
                    logger.warning(
                        f"Cannot register for unknown backend: {backend}. Registration skipped."
                    )
                else:
                    logger.debug(
                        f"Mapping alias '{backend}' to '{backend_lower}' for figure type '{figure_type}'"
                    )
                return

            cls._registry[backend_lower][figure_type] = implementation
            logger.debug(
                f"Registered {figure_type} implementation for {backend_lower} backend"
            )

    @classmethod
    def get(cls, figure_type: str, backend: str) -> Type[BaseFigure]:
        """
        Get figure implementation for a specific type and backend in a thread-safe manner.

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
        # Use lock to ensure thread-safe registry access
        with cls._lock:
            # Handle special case for base figure
            if figure_type == "base":
                return PlotlyFigure if backend == "plotly" else MatplotlibFigure

            # Normalize backend name and handle aliases
            backend_lower = backend.lower()
            if backend_lower in _BACKEND_ALIASES:
                backend_lower = _BACKEND_ALIASES[backend_lower]
                logger.debug(f"Using '{backend_lower}' backend for alias '{backend}'")

            # Check if the requested backend is available
            if backend_lower not in cls._registry:
                logger.warning(f"Unknown backend: {backend}. Falling back to plotly.")
                backend_lower = "plotly"

            # Check if the figure type is implemented for the backend
            if figure_type not in cls._registry[backend_lower]:
                # First try the other backend if available
                other_backend = "matplotlib" if backend_lower == "plotly" else "plotly"
                if figure_type in cls._registry[other_backend]:
                    logger.warning(
                        f"Figure type '{figure_type}' not found for backend '{backend_lower}'. "
                        f"Falling back to '{other_backend}'."
                    )
                    return cls._registry[other_backend][figure_type]

                # If neither backend has the implementation, use a base implementation
                logger.warning(
                    f"Figure type '{figure_type}' not implemented for any backend. "
                    f"Using base implementation."
                )
                return PlotlyFigure if backend_lower == "plotly" else MatplotlibFigure

            return cls._registry[backend_lower][figure_type]


class FigureFactory:
    """
    Thread-safe factory for creating figure instances.

    This factory uses the current backend from the execution context
    to create figures, ensuring isolation between concurrent operations.
    """

    def create_figure(
        self, figure_type: str, backend: Optional[str] = None
    ) -> BaseFigure:
        """
        Create a figure instance of the specified type.

        Parameters:
        -----------
        figure_type : str
            Type of figure to create
        backend : str, optional
            Backend to use (defaults to current context's backend)

        Returns:
        --------
        BaseFigure
            Figure instance
        """
        # Use specified backend or fall back to context's setting
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
def ensure_series(
    data: Union[Dict[str, Any], pd.Series, List, np.ndarray],
) -> pd.Series:
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


def sort_series(
    series: pd.Series,
    sort_by: str = "value",
    ascending: bool = False,
    max_items: Optional[int] = None,
) -> pd.Series:
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


def prepare_dataframe(
    data: Union[Dict[str, List], pd.DataFrame], orient: str = "dict"
) -> pd.DataFrame:
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
