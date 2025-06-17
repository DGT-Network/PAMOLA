"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Visualization Theme System
Description: Thread-safe theme management for visualization capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides theming capabilities for the visualization system, including:
- Definition of color schemes and other visual properties
- Thread-safe context-isolated management of current theme
- Theme application to figures
- Utilities for obtaining theme-consistent colors

The implementation uses contextvars to ensure that theme settings are properly
isolated between concurrent execution contexts, eliminating state interference
when multiple visualization operations run in parallel.
"""

import contextvars
import logging
from typing import List, Union
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import plotly
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

# core/utils/vis_helpers/colormap.py


# Configure logger
logger = logging.getLogger(__name__)

# Define a context variable for theme storage
# This replaces the global _CURRENT_THEME variable with a context-isolated version
_theme_context = contextvars.ContextVar("current_theme", default="default")

# Default themes
THEMES = {
    "default": {
        "colors": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "colorscale": "Blues",
        "background_color": "#ffffff",
        "grid_color": "#e0e0e0",
        "text_color": "#000000",
        "font_family": "Arial, Helvetica, sans-serif",
        "font_size": 12,
        "title_font_size": 16,
        "margin": {"l": 80, "r": 40, "t": 60, "b": 80},
        "showlegend": True,
        "legend_position": {"x": 1.02, "y": 1},
        "bar_mode": "group",
        "opacity": 0.8,
        "template": "plotly_white",
    },
    "dark": {
        "colors": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "colorscale": "Viridis",
        "background_color": "#121212",
        "grid_color": "#333333",
        "text_color": "#ffffff",
        "font_family": "Arial, Helvetica, sans-serif",
        "font_size": 12,
        "title_font_size": 16,
        "margin": {"l": 80, "r": 40, "t": 60, "b": 80},
        "showlegend": True,
        "legend_position": {"x": 1.02, "y": 1},
        "bar_mode": "group",
        "opacity": 0.9,
        "template": "plotly_dark",
    },
    "pastel": {
        "colors": [
            "#a1c9f4",
            "#ffb482",
            "#8de5a1",
            "#ff9f9b",
            "#d0bbff",
            "#debb9b",
            "#fab0e4",
            "#cfcfcf",
            "#fffea3",
            "#b9f2f0",
        ],
        "colorscale": "Pastel",
        "background_color": "#f8f9fa",
        "grid_color": "#e9ecef",
        "text_color": "#343a40",
        "font_family": "Verdana, Geneva, sans-serif",
        "font_size": 11,
        "title_font_size": 15,
        "margin": {"l": 80, "r": 40, "t": 60, "b": 80},
        "showlegend": True,
        "legend_position": {"x": 1.02, "y": 1},
        "bar_mode": "group",
        "opacity": 0.7,
        "template": "simple_white",
    },
    "professional": {
        "colors": [
            "#4c78a8",
            "#f58518",
            "#54a24b",
            "#e45756",
            "#72b7b2",
            "#eeca3b",
            "#b279a2",
            "#ff9da6",
            "#9d755d",
            "#bab0ac",
        ],
        "colorscale": "RdBu",
        "background_color": "#ffffff",
        "grid_color": "#d9d9d9",
        "text_color": "#333333",
        "font_family": "Helvetica, Arial, sans-serif",
        "font_size": 12,
        "title_font_size": 16,
        "margin": {"l": 80, "r": 40, "t": 60, "b": 80},
        "showlegend": True,
        "legend_position": {"x": 1.02, "y": 1},
        "bar_mode": "group",
        "opacity": 0.85,
        "template": "plotly_white",
    },
}


def set_theme(theme_name: str, strict: bool = False) -> None:
    """
    Set the current theme in the current execution context.

    This function uses context variables to ensure that theme settings
    are isolated between concurrent execution contexts, preventing
    interference when multiple visualization operations run in parallel.

    Parameters:
    -----------
    theme_name : str
        Name of the theme to use
    strict : bool
        If True, raise exceptions for invalid themes; otherwise log warnings

    Raises:
    -------
    ValueError
        If strict=True and theme is not found
    """
    if theme_name not in THEMES:
        error_msg = f"Theme '{theme_name}' not found. Available themes: {', '.join(THEMES.keys())}"
        if strict:
            raise ValueError(error_msg)
        else:
            logger.warning(f"{error_msg} Using default theme.")
            _theme_context.set("default")
    else:
        _theme_context.set(theme_name)
        logger.debug(f"Theme set to '{theme_name}' in current context")


def get_current_theme_name() -> str:
    """
    Get the name of the current theme for the current execution context.

    Returns:
    --------
    str
        Current theme name
    """
    return _theme_context.get()


def get_current_theme() -> Dict[str, Any]:
    """
    Get the current theme configuration for the current execution context.

    Returns:
    --------
    Dict[str, Any]
        Current theme configuration
    """
    theme_name = _theme_context.get()
    return THEMES[theme_name]


def create_custom_theme(
    name: str, config: Dict[str, Any], strict: bool = False
) -> None:
    """
    Create a custom theme.

    Parameters:
    -----------
    name : str
        Name for the custom theme
    config : Dict[str, Any]
        Theme configuration
    strict : bool
        If True, raise exceptions for missing required keys; otherwise fill with defaults

    Raises:
    -------
    ValueError
        If strict=True and required keys are missing
    """
    # Check for required keys
    required_keys = ["colors", "colorscale", "background_color", "text_color"]
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        error_msg = f"Custom theme '{name}' is missing required keys: {missing_keys}"
        if strict:
            raise ValueError(f"{error_msg}. Theme not created.")
        else:
            logger.warning(f"{error_msg}. Using defaults for these keys.")

            # Fill in missing keys from default theme
            for key in missing_keys:
                config[key] = THEMES["default"][key]

    # Add theme to themes dictionary
    THEMES[name] = config
    logger.info(f"Custom theme '{name}' created")


def get_theme_colors(n_colors: int = 10) -> List[str]:
    """
    Get a list of colors from the current theme in the current execution context.

    Parameters:
    -----------
    n_colors : int
        Number of colors to return

    Returns:
    --------
    List[str]
        List of hex color codes
    """
    theme = get_current_theme()
    colors = theme["colors"]

    # If we need more colors than available, cycle through the list
    if n_colors <= len(colors):
        return colors[:n_colors]
    else:
        return [colors[i % len(colors)] for i in range(n_colors)]


def apply_theme_to_plotly_figure(
    fig, theme: Optional[Dict[str, Any]] = None
) -> "plotly.graph_objects.Figure":
    """
    Apply theme settings to a Plotly figure.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to apply theme to
    theme : Dict[str, Any], optional
        Theme configuration (uses current theme if None)

    Returns:
    --------
    plotly.graph_objects.Figure
        Themed figure
    """
    try:
        import plotly.graph_objects as go

        if not isinstance(fig, go.Figure):
            logger.warning("Cannot apply Plotly theme to non-Plotly figure")
            return fig

        # Use specified theme or current theme
        theme_config = theme or get_current_theme()

        # Apply theme settings
        fig.update_layout(
            font=dict(
                family=theme_config["font_family"],
                size=theme_config["font_size"],
                color=theme_config["text_color"],
            ),
            paper_bgcolor=theme_config["background_color"],
            plot_bgcolor=theme_config["background_color"],
            margin=theme_config["margin"],
            showlegend=theme_config["showlegend"],
            legend=dict(
                x=theme_config["legend_position"]["x"],
                y=theme_config["legend_position"]["y"],
            ),
            template=theme_config.get("template", "plotly_white"),
        )

        # Update title font size
        if fig.layout.title:
            fig.update_layout(
                title=dict(font=dict(size=theme_config["title_font_size"]))
            )

        # Update grid color
        fig.update_xaxes(gridcolor=theme_config["grid_color"])
        fig.update_yaxes(gridcolor=theme_config["grid_color"])

        return fig
    except ImportError:
        logger.warning("Plotly is not available for theming")
        return fig
    except Exception as e:
        logger.error(f"Error applying Plotly theme: {e}")
        return fig


def apply_theme_to_matplotlib_figure(
    fig: Figure, theme: Optional[Dict[str, Any]] = None
) -> Figure:
    """
    Apply theme settings to a Matplotlib figure.

    Parameters:
    -----------
    fig : Figure
        Matplotlib figure to apply theme to
    theme : dict, optional
        Theme configuration (uses current theme if None)

    Returns:
    --------
    Figure
        Themed figure
    """
    try:
        # Ensure it's a Matplotlib figure
        if not isinstance(fig, Figure):
            logger.warning("Cannot apply Matplotlib theme to non-Matplotlib figure")
            return fig

        # Load theme config
        theme_config = theme or get_current_theme()

        # Figure background
        fig.set_facecolor(theme_config["background_color"])

        # Apply to each Axes
        for ax in fig.get_axes():
            ax.set_facecolor(theme_config["background_color"])
            for spine in ax.spines.values():
                spine.set_color(theme_config["grid_color"])

            ax.tick_params(
                colors=theme_config["text_color"], labelsize=theme_config["font_size"]
            )

            ax.grid(
                color=theme_config["grid_color"],
                linestyle="-",
                linewidth=0.5,
                alpha=0.5,
            )

            # Text elements
            ax.xaxis.label.set_color(theme_config["text_color"])
            ax.yaxis.label.set_color(theme_config["text_color"])
            ax.title.set_color(theme_config["text_color"])

            ax.xaxis.label.set_fontsize(theme_config["font_size"])
            ax.yaxis.label.set_fontsize(theme_config["font_size"])
            ax.title.set_fontsize(theme_config["title_font_size"])

            # Legend styling
            leg = ax.get_legend()
            if leg:
                leg.set_frame_on(True)
                leg.get_frame().set_facecolor(theme_config["background_color"])
                leg.get_frame().set_edgecolor(theme_config["grid_color"])
                for text in leg.get_texts():
                    text.set_color(theme_config["text_color"])

        fig.tight_layout()
        return fig

    except ImportError:
        logger.warning("Matplotlib is not available for theming")
        return fig
    except Exception as e:
        logger.error(f"Error applying Matplotlib theme: {e}")
        return fig


def get_colorscale(
    theme: Optional[Dict[str, Any]] = None,
) -> List[List[Union[float, str]]]:
    """
    Get a colorscale from the current theme for the current execution context.

    Parameters:
    -----------
    theme : Dict[str, Any], optional
        Theme configuration (uses current theme if None)

    Returns:
    --------
    List[List[Union[float, str]]]
        Colorscale in Plotly format
    """
    try:
        import plotly.colors as pc

        # Use specified theme or current theme
        theme_config = theme or get_current_theme()
        colorscale_name = theme_config["colorscale"]

        # Get colorscale from Plotly
        if hasattr(pc.sequential, colorscale_name):
            return getattr(pc.sequential, colorscale_name)
        elif hasattr(pc.diverging, colorscale_name):
            return getattr(pc.diverging, colorscale_name)
        else:
            logger.warning(f"Colorscale '{colorscale_name}' not found. Using Blues.")
            return pc.sequential.Blues
    except ImportError:
        # If Plotly is not available, return a default blue scale
        blues = [
            [0.0, "#f7fbff"],
            [0.1, "#deebf7"],
            [0.2, "#c6dbef"],
            [0.3, "#9ecae1"],
            [0.4, "#6baed6"],
            [0.5, "#4292c6"],
            [0.6, "#2171b5"],
            [0.7, "#08519c"],
            [0.8, "#08306b"],
            [1.0, "#041836"],
        ]
        return blues
    except Exception as e:
        logger.error(f"Error getting colorscale: {e}")
        # Return default blue scale on error
        blues = [
            [0.0, "#f7fbff"],
            [0.1, "#deebf7"],
            [0.2, "#c6dbef"],
            [0.3, "#9ecae1"],
            [0.4, "#6baed6"],
            [0.5, "#4292c6"],
            [0.6, "#2171b5"],
            [0.7, "#08519c"],
            [0.8, "#08306b"],
            [1.0, "#041836"],
        ]
        return blues


def get_matplotlib_colormap(
    theme: Optional[Dict[str, Any]] = None,
) -> Optional[Colormap]:
    """
    Get a Matplotlib colormap from the current theme for the current execution context.

    Parameters:
    -----------
    theme : dict, optional
        Theme configuration (uses current theme if None)

    Returns:
    --------
    Optional[Colormap]
        Matplotlib colormap, or None if Matplotlib is unavailable
    """
    try:
        # Choosing a theme config
        theme_config = theme or get_current_theme()
        colorscale_name = theme_config.get("colorscale", "")

        # Plotly → Matplotlib mapping
        colorscale_map = {
            "Blues": "Blues",
            "Reds": "Reds",
            "Greens": "Greens",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Magma": "magma",
            "Cividis": "cividis",
            "RdBu": "RdBu",
            "BrBG": "BrBG",
            "Pastel": "Pastel1",
        }

        cmap_name = colorscale_map.get(colorscale_name, "Blues")

        try:
            # plt.get_cmap returns a Colormap
            return plt.get_cmap(cmap_name)
        except Exception:
            logger.warning(
                f"Colormap '{cmap_name}' not found. Falling back to 'Blues'."
            )
            return plt.get_cmap("Blues")

    except ImportError:
        logger.warning("Matplotlib is not available for getting a colormap")
        return None
    except Exception as e:
        logger.error(f"Error getting matplotlib colormap: {e}")
        return None
