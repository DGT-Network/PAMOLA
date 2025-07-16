"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Network Diagram Visualization Implementation

Description:
    Thread-safe network diagram visualization capabilities using both Plotly (primary) and Matplotlib (fallback) backends.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for network diagrams using both Plotly and Matplotlib backends.
Plotly is the primary implementation, while Matplotlib serves as a fallback when needed.

The implementation uses contextvars via the visualization_context
to ensure thread-safe operation for concurrent execution contexts.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pamola_core.utils.vis_helpers.base import (
    MatplotlibFigure,
    FigureRegistry,
    PlotlyFigure,
)
from pamola_core.utils.vis_helpers.context import visualization_context
from pamola_core.utils.vis_helpers.theme import (
    apply_theme_to_matplotlib_figure,
    apply_theme_to_plotly_figure,
)

logger = logging.getLogger(__name__)


class PlotlyNetworkDiagram(PlotlyFigure):
    """Network diagram implementation using Plotly."""

    def create(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        node_labels: Optional[Dict[str, str]] = None,
        title: str = "Network Diagram",
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a network diagram using Plotly.

        Parameters:
            nodes (List[str]): List of node names.
            edges (List[Tuple[str, str]]): List of edges as (source, target) tuples.
            node_labels (Dict[str, str], optional): Labels for nodes.
            title (str): Title for the diagram.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Additional arguments for customization.

        Returns:
            go.Figure: Plotly figure containing the network diagram.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go


                # Create a lookup dictionary of node types (default to 'field')
                node_types = {n: node_labels.get(n, "field") for n in nodes}

                # Separate nodes by type
                subset_nodes = [n for n in nodes if node_types[n] == "subset"]
                field_nodes = [n for n in nodes if node_types[n] == "field"]

                # Assign positions for each type
                positions = {n: (0, i * -1.5) for i, n in enumerate(subset_nodes)}
                positions.update({n: (3, i * -1.2) for i, n in enumerate(field_nodes)})

                # Draw edge lines
                edge_x, edge_y = [], []
                for src, tgt in edges:
                    x0, y0 = positions[src]
                    x1, y1 = positions[tgt]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(
                        width=kwargs.get("edge_width", 1),
                        color=kwargs.get("edge_color", "gray"),
                    ),
                    hoverinfo="none",
                    mode="lines",
                )

                # Draw nodes
                node_x, node_y, node_text, node_colors = [], [], [], []
                for node in nodes:
                    x, y = positions[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_colors.append("skyblue" if node_types[node] == "subset" else "lightgreen")

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    marker=dict(
                        size=kwargs.get("node_size", 20),
                        color=node_colors,
                        line=dict(
                            width=kwargs.get("node_border_width", 1),
                            color=kwargs.get("node_border_color", "black"),
                        ),
                    ),
                    text=node_text,
                    textposition=kwargs.get("text_position", "top center"),
                    hoverinfo="text",
                )

                # Combine into a single figure
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text=title,
                            font=dict(size=kwargs.get("title_font_size", 16)),
                        ),
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(l=40, r=40, t=80, b=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor=kwargs.get("background_color", "white"),
                    ),
                )

                fig = apply_theme_to_plotly_figure(fig)
                return fig

            except Exception as e:
                logger.error(f"Error creating network diagram with Plotly: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating network diagram: {str(e)}"
                )

    def update(
        self,
        fig: Any,
        nodes: Optional[List[str]] = None,
        edges: Optional[List[Tuple[str, str]]] = None,
        node_labels: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Update an existing Plotly network diagram.

        Parameters:
            fig (go.Figure): Existing Plotly figure to update.
            nodes (List[str], optional): Updated list of nodes.
            edges (List[Tuple[str, str]], optional): Updated list of edges.
            node_labels (Dict[str, str], optional): Updated labels for nodes.
            title (str, optional): Updated title for the diagram.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Additional arguments for customization.

        Returns:
            go.Figure: Updated Plotly figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                if nodes and edges:
                    return self.create(
                        nodes=nodes,
                        edges=edges,
                        node_labels=node_labels,
                        title=title or fig.layout.title.text,
                        backend=backend,
                        theme=theme,
                        strict=strict,
                        **kwargs,
                    )

                if title:
                    fig.update_layout(
                        title=dict(
                            text=title,
                            font=dict(size=kwargs.get("title_font_size", 16)),
                        )
                    )

                fig = apply_theme_to_plotly_figure(fig)
                return fig
            except Exception as e:
                logger.error(f"Error updating network diagram with Plotly: {e}")
                return fig


class MatplotlibNetworkDiagram(MatplotlibFigure):
    """Network diagram implementation using Matplotlib."""

    def create(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        node_labels: Optional[Dict[str, str]] = None,
        title: str = "Network Diagram",
        figsize: Tuple[int, int] = (8, 6),
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a network diagram using Matplotlib.

        Parameters:
        ----------
        nodes : List[str]
            List of node identifiers (subset and field names).
        edges : List[Tuple[str, str]]
            List of (source, target) tuples representing edges between nodes.
        node_labels : Dict[str, str], optional
            Dictionary specifying node types. Each key is a node, and value should be "subset" or "field".
        title : str
            Title for the diagram.
        figsize : Tuple[int, int]
            Size of the Matplotlib figure.
        backend : str, optional
            Visualization backend (passed to context).
        theme : str, optional
            Visualization theme (light, dark, etc.).
        strict : bool
            If True, raise errors instead of logging them.
        **kwargs :
            Additional visualization customization options:
                - node_size (int): Size of the nodes.
                - node_border_color (str): Color of node borders.
                - node_border_width (int): Width of node borders.
                - edge_color (str): Color of the edges.
                - edge_width (float): Width of the edges.
                - font_size (int): Font size of the node labels.
                - title_font_size (int): Font size of the plot title.

        Returns:
        -------
        matplotlib.figure.Figure or None
            The generated network diagram figure or None if an error occurs.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt
                import networkx as nx

                node_labels = node_labels or {}

                # Precompute node types to avoid repeated lookups
                node_types = {n: node_labels.get(n, "field") for n in nodes}

                # Separate nodes by type
                subset_nodes = [n for n in nodes if node_types[n] == "subset"]
                field_nodes = [n for n in nodes if node_types[n] == "field"]

                # Assign fixed positions
                positions = {}
                for i, node in enumerate(subset_nodes):
                    positions[node] = (0, -i * 1.5)
                for i, node in enumerate(field_nodes):
                    positions[node] = (3, -i * 1.2)

                # Assign colors based on type
                node_colors = [
                    "skyblue" if node_types[node] == "subset" else "lightgreen"
                    for node in nodes
                ]

                fig, ax = plt.subplots(figsize=figsize)

                # Draw edges
                for src, tgt in edges:
                    if src in positions and tgt in positions:
                        x0, y0 = positions[src]
                        x1, y1 = positions[tgt]
                        ax.plot(
                            [x0, x1], [y0, y1],
                            color=kwargs.get("edge_color", "gray"),
                            linewidth=kwargs.get("edge_width", 1),
                            zorder=1
                        )

                # Draw nodes and labels
                for i, node in enumerate(nodes):
                    x, y = positions[node]
                    ax.scatter(
                        x, y,
                        s=kwargs.get("node_size", 360),
                        color=node_colors[i],
                        edgecolors=kwargs.get("node_border_color", "black"),
                        linewidths=kwargs.get("node_border_width", 1),
                        zorder=2
                    )
                    ax.text(
                        x, y + 0.2,
                        node,
                        ha="center",
                        fontsize=kwargs.get("font_size", 10),
                        zorder=3
                    )

                ax.set_title(title, fontsize=kwargs.get("title_font_size", 16))
                ax.axis("off")
                plt.tight_layout()
                fig = apply_theme_to_matplotlib_figure(fig)
                return fig

            except Exception as e:
                logger.error(f"Error creating network diagram with Matplotlib: {e}")
                return None

    def update(
        self,
        fig: plt.Figure,
        nodes: Optional[List[str]] = None,
        edges: Optional[List[Tuple[str, str]]] = None,
        node_labels: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Update an existing Matplotlib network diagram.

        Parameters:
            fig (plt.Figure): Existing figure to update.
            nodes (List[str], optional): Updated list of nodes.
            edges (List[Tuple[str, str]], optional): Updated list of edges.
            node_labels (Dict[str, str], optional): Updated labels for nodes.
            title (str, optional): Updated title for the diagram.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Additional arguments for customization.

        Returns:
            plt.Figure: Updated Matplotlib figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt
                import networkx as nx

                if nodes and edges:
                    return self.create(
                        nodes=nodes,
                        edges=edges,
                        node_labels=node_labels,
                        title=title or fig.axes[0].get_title(),
                        backend=backend,
                        theme=theme,
                        strict=strict,
                        **kwargs,
                    )

                if title:
                    fig.axes[0].set_title(
                        title, fontsize=kwargs.get("title_font_size", 16)
                    )

                fig = apply_theme_to_matplotlib_figure(fig)
                plt.tight_layout()
                return fig
            except Exception as e:
                logger.error(f"Error updating network diagram with Matplotlib: {e}")
                return fig


# Register with visualization system
FigureRegistry.register("network_diagram", "plotly", PlotlyNetworkDiagram)
FigureRegistry.register("network_diagram", "matplotlib", MatplotlibNetworkDiagram)
