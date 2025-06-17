"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Visualization Context Management
Description: Thread-safe context management for visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides context management for the visualization system, including:
- Context managers for theme and backend isolation
- Resource cleanup utilities
- Common context-aware helper functions

The implementation uses contextvars to ensure that visualization state is properly
isolated between concurrent execution contexts, eliminating state interference
when multiple visualization operations run in parallel.
"""

import contextlib
import logging
from typing import Optional, Generator, Any, Dict, Union, Tuple, Callable

# Configure logger
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def visualization_context(
    backend: Optional[str] = None,
    theme: Optional[str] = None,
    auto_close: bool = True,
    strict: bool = False,
    headless: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for visualization operations.

    Provides isolated context for visualization operations, ensuring that theme
    and backend settings don't interfere between concurrent visualizations.
    Automatically applies headless mode for matplotlib if requested.

    Parameters:
    -----------
    backend : Optional[str]
        Backend to use for this context
    theme : Optional[str]
        Theme to use for this context
    auto_close : bool
        Whether to automatically close matplotlib figures when exiting context
    strict : bool
        If True, raise exceptions for invalid configuration; if False, log warnings
    headless : bool
        Whether to use headless (Agg) backend for matplotlib operations

    Yields:
    -------
    Dict[str, Any]
        Context information dictionary
    """
    from pamola_core.utils.vis_helpers.base import get_backend, set_backend
    from pamola_core.utils.vis_helpers.theme import get_current_theme_name, set_theme

    # Store original values to restore later
    original_backend = get_backend()
    original_theme = get_current_theme_name()

    # Keep track of created figures for cleanup
    context_info = {
        "original_backend": original_backend,
        "original_theme": original_theme,
        "figures": [],
        "strict": strict,
    }

    # Apply headless mode for matplotlib if requested
    matplotlib_context = matplotlib_agg_context() if headless else null_context()

    try:
        # Enter matplotlib context first (if headless mode is enabled)
        with matplotlib_context:
            # Set new values if provided
            if backend is not None:
                try:
                    set_backend(backend, strict=strict)
                    context_info["current_backend"] = backend
                except ValueError as e:
                    if strict:
                        raise
                    else:
                        logger.warning(f"Non-critical backend setting error: {e}")
                        context_info["current_backend"] = get_backend()
            else:
                context_info["current_backend"] = original_backend

            if theme is not None:
                try:
                    set_theme(theme, strict=strict)
                    context_info["current_theme"] = theme
                except ValueError as e:
                    if strict:
                        raise
                    else:
                        logger.warning(f"Non-critical theme setting error: {e}")
                        context_info["current_theme"] = get_current_theme_name()
            else:
                context_info["current_theme"] = original_theme

            # Yield context info back to the caller
            yield context_info

    finally:
        # Clean up resources if requested
        if auto_close:
            _cleanup_figures(context_info.get("figures", []))

        # Restore original values if they were changed
        if backend is not None:
            try:
                set_backend(original_backend, strict=False)  # Never strict in cleanup
            except Exception as e:
                logger.debug(f"Non-critical error restoring backend: {e}")

        if theme is not None:
            try:
                set_theme(original_theme, strict=False)  # Never strict in cleanup
            except Exception as e:
                logger.debug(f"Non-critical error restoring theme: {e}")


def register_figure(fig: Any, context_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a figure with the current context for cleanup.

    Parameters:
    -----------
    fig : Any
        Figure object to register
    context_info : Optional[Dict[str, Any]]
        Context information from visualization_context
    """
    if context_info is not None and isinstance(context_info, dict):
        if "figures" in context_info:
            context_info["figures"].append(fig)


def _cleanup_figures(figures: list) -> None:
    """
    Clean up matplotlib figures to prevent memory leaks.

    Parameters:
    -----------
    figures : list
        List of figure objects to clean up
    """
    for fig in figures:
        try:
            # Check if it's a matplotlib figure
            if hasattr(fig, "clf") and callable(fig.clf):
                fig.clf()
            if "matplotlib.figure" in str(type(fig)):
                import matplotlib.pyplot as plt

                plt.close(fig)
        except Exception as e:
            logger.debug(f"Non-critical error during figure cleanup: {e}")


@contextlib.contextmanager
def matplotlib_agg_context() -> Generator[None, None, None]:
    """
    Temporarily switch Matplotlib to 'Agg' backend for headless rendering.
    Restores original backend on exit.
    """
    try:
        import matplotlib
    except ImportError:
        # No matplotlib — just skip the whole context
        logger.warning("Matplotlib is not installed; skipping Agg context.")
        yield
        return

    # In the thread below - guaranteed to have the module
    original_backend = matplotlib.get_backend()
    switched = False

    if original_backend.lower() != "agg":
        try:
            matplotlib.use("Agg")
            switched = True
            logger.debug(f"Switched Matplotlib backend from {original_backend} to Agg")
        except Exception as e:
            logger.error(f"Failed to switch to Agg backend: {e}")

    try:
        yield
    finally:
        # We will restore it only if it was actually switched
        if switched:
            try:
                matplotlib.use(original_backend)
                logger.debug(f"Restored Matplotlib backend to {original_backend}")
            except Exception as e:
                logger.error(f"Failed to restore original backend: {e}")


@contextlib.contextmanager
def null_context() -> Generator[None, None, None]:
    """
    A no-op context manager that does nothing.

    Used as a placeholder when certain context managers are conditionally applied.

    Yields:
    -------
    None
    """
    yield


def get_figure_size(
    size: Optional[Union[str, Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    """
    Get standardized figure size based on named presets or custom dimensions.

    Parameters:
    -----------
    size : Optional[Union[str, Tuple[int, int]]]
        Size specification. Can be a preset name ('small', 'medium', 'large',
        'wide', 'tall') or a tuple of (width, height) in inches.

    Returns:
    --------
    Tuple[int, int]
        Figure size as (width, height) in inches
    """
    # Standard size presets
    presets = {
        "small": (6, 4),
        "medium": (8, 6),
        "large": (12, 8),
        "wide": (12, 6),
        "tall": (8, 10),
        "square": (8, 8),
        "slide": (10, 5.6),  # 16:9 ratio
        "poster": (16, 12),
        "thumbnail": (4, 3),
    }

    # If size is None, return medium size
    if size is None:
        return presets["medium"]

    # If size is a preset name, return that preset
    if isinstance(size, str):
        size_lower = size.lower()
        if size_lower in presets:
            return presets[size_lower]
        else:
            logger.warning(f"Unknown size preset: {size}. Using 'medium' instead.")
            return presets["medium"]

    # If size is a tuple, return it
    if isinstance(size, tuple) and len(size) == 2:
        # Validate size values
        width, height = size
        if width <= 0 or height <= 0:
            logger.warning(
                f"Invalid figure dimensions: {size}. Using 'medium' instead."
            )
            return presets["medium"]
        return size

    # For any other case, return medium size
    logger.warning(f"Unrecognized size specification: {size}. Using 'medium' instead.")
    return presets["medium"]


def auto_visualization_context(func: Callable) -> Callable:
    """
    Decorator to automatically wrap a function with a visualization_context.

    This decorator helps ensure that functions that directly use visualization
    backends (e.g., matplotlib.pyplot) are properly wrapped with the necessary
    context management even if they bypass the standard visualization API.

    Parameters:
    -----------
    func : Callable
        Function to wrap

    Returns:
    --------
    Callable
        Wrapped function
    """

    def wrapper(*args, **kwargs):
        # Extract visualization context args from kwargs if present
        vis_kwargs = {
            "backend": kwargs.pop("backend", None),
            "theme": kwargs.pop("theme", None),
            "auto_close": kwargs.pop("auto_close", True),
            "strict": kwargs.pop("strict", False),
            "headless": kwargs.pop("headless", True),
        }

        with visualization_context(**vis_kwargs) as context_info:
            result = func(*args, **kwargs)
            # If result is a figure object, register it for cleanup
            if (hasattr(result, "clf") and callable(result.clf)) or (
                "matplotlib.figure" in str(type(result))
            ):
                register_figure(result, context_info)
            return result

    return wrapper
