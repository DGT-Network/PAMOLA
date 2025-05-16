"""
Theme management for visualizations.

This module provides theming capabilities for the visualization system, including:
- Definition of color schemes and other visual properties
- Management of current theme
- Theme application to figures
- Utilities for obtaining theme-consistent colors
"""

import logging
from typing import Dict, Any, List, Optional, Union

import plotly.graph_objects as go
from matplotlib import pyplot as plt


# Configure logger
logger = logging.getLogger(__name__)

# Default themes
THEMES = {
    "default": {
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
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
        "template": "plotly_white"
    },
    "dark": {
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
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
        "template": "plotly_dark"
    },
    "pastel": {
        "colors": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff",
                   "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"],
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
        "template": "simple_white"
    },
    "professional": {
        "colors": ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
                   "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"],
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
        "template": "plotly_white"
    }
}

# Current theme
_CURRENT_THEME = "default"


def set_theme(theme_name: str) -> None:
    """
    Set the current theme.

    Parameters:
    -----------
    theme_name : str
        Name of the theme to use
    """
    global _CURRENT_THEME
    if theme_name not in THEMES:
        logger.warning(f"Theme '{theme_name}' not found. Using default theme.")
        _CURRENT_THEME = "default"
    else:
        _CURRENT_THEME = theme_name
        logger.debug(f"Theme set to '{theme_name}'")


def get_current_theme() -> Dict[str, Any]:
    """
    Get the current theme configuration.

    Returns:
    --------
    Dict[str, Any]
        Current theme configuration
    """
    return THEMES[_CURRENT_THEME]


def create_custom_theme(name: str, config: Dict[str, Any]) -> None:
    """
    Create a custom theme.

    Parameters:
    -----------
    name : str
        Name for the custom theme
    config : Dict[str, Any]
        Theme configuration
    """
    # Check for required keys
    required_keys = ["colors", "colorscale", "background_color", "text_color"]
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        logger.warning(f"Custom theme '{name}' is missing required keys: {missing_keys}. "
                       f"Using defaults for these keys.")

        # Fill in missing keys from default theme
        for key in missing_keys:
            config[key] = THEMES["default"][key]

    # Add theme to themes dictionary
    THEMES[name] = config
    logger.info(f"Custom theme '{name}' created")


def get_theme_colors(n_colors: int = 10) -> List[str]:
    """
    Get a list of colors from the current theme.

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


def apply_theme_to_plotly_figure(fig, theme: Optional[Dict[str, Any]] = None) -> 'go.Figure':
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
            color=theme_config["text_color"]
        ),
        paper_bgcolor=theme_config["background_color"],
        plot_bgcolor=theme_config["background_color"],
        margin=theme_config["margin"],
        showlegend=theme_config["showlegend"],
        legend=dict(
            x=theme_config["legend_position"]["x"],
            y=theme_config["legend_position"]["y"]
        ),
        template=theme_config.get("template", "plotly_white")
    )

    # Update title font size
    if fig.layout.title:
        fig.update_layout(
            title=dict(
                font=dict(
                    size=theme_config["title_font_size"]
                )
            )
        )

    # Update grid color
    fig.update_xaxes(gridcolor=theme_config["grid_color"])
    fig.update_yaxes(gridcolor=theme_config["grid_color"])

    return fig


def apply_theme_to_matplotlib_figure(fig, theme: Optional[Dict[str, Any]] = None) -> 'plt.Figure':
    """
    Apply theme settings to a Matplotlib figure.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Matplotlib figure to apply theme to
    theme : Dict[str, Any], optional
        Theme configuration (uses current theme if None)

    Returns:
    --------
    matplotlib.figure.Figure
        Themed figure
    """
    import matplotlib.pyplot as plt

    if not isinstance(fig, plt.Figure):
        logger.warning("Cannot apply Matplotlib theme to non-Matplotlib figure")
        return fig

    # Use specified theme or current theme
    theme_config = theme or get_current_theme()

    # Apply theme settings to figure
    fig.set_facecolor(theme_config["background_color"])

    # Apply to all axes
    for ax in fig.get_axes():
        ax.set_facecolor(theme_config["background_color"])
        ax.spines['bottom'].set_color(theme_config["grid_color"])
        ax.spines['top'].set_color(theme_config["grid_color"])
        ax.spines['left'].set_color(theme_config["grid_color"])
        ax.spines['right'].set_color(theme_config["grid_color"])

        ax.tick_params(colors=theme_config["text_color"])

        # Set grid color
        ax.grid(color=theme_config["grid_color"], linestyle='-', linewidth=0.5, alpha=0.5)

        # Update text colors
        ax.xaxis.label.set_color(theme_config["text_color"])
        ax.yaxis.label.set_color(theme_config["text_color"])
        ax.title.set_color(theme_config["text_color"])

        # Update font sizes
        ax.xaxis.label.set_size(theme_config["font_size"])
        ax.yaxis.label.set_size(theme_config["font_size"])
        ax.title.set_size(theme_config["title_font_size"])

        # Update tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=theme_config["font_size"])

        # Show legend if applicable and update its properties
        if ax.get_legend():
            ax.legend(frameon=True, facecolor=theme_config["background_color"],
                      edgecolor=theme_config["grid_color"],
                      labelcolor=theme_config["text_color"])

    # Adjust layout
    fig.tight_layout()

    return fig


def get_colorscale(theme: Optional[Dict[str, Any]] = None) -> List[List[Union[float, str]]]:
    """
    Get a colorscale from the current theme.

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
            [1.0, "#041836"]
        ]
        return blues


def get_matplotlib_colormap(theme: Optional[Dict[str, Any]] = None) -> 'plt.cm':
    """
    Get a matplotlib colormap from the current theme.

    Parameters:
    -----------
    theme : Dict[str, Any], optional
        Theme configuration (uses current theme if None)

    Returns:
    --------
    matplotlib.cm
        Matplotlib colormap
    """
    import matplotlib.pyplot as plt

    # Use specified theme or current theme
    theme_config = theme or get_current_theme()
    colorscale_name = theme_config["colorscale"]

    # Map common Plotly colorscales to Matplotlib
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

    # Get the appropriate colormap
    cmap_name = colorscale_map.get(colorscale_name, "Blues")

    try:
        return plt.cm.get_cmap(cmap_name)
    except:
        logger.warning(f"Colormap '{cmap_name}' not found. Using 'Blues'.")
        return plt.cm.get_cmap("Blues")