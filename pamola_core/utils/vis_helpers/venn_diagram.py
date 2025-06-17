"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Line Plot Visualization Implementation

Description:
    Thread-safe line plot visualization capabilities using both Plotly (primary) and Matplotlib (fallback) backends.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides implementations for line plots using both Plotly and Matplotlib backends.
Plotly is the primary implementation, while Matplotlib serves as a fallback when needed.

The implementation uses contextvars via the visualization_context
to ensure thread-safe operation for concurrent execution contexts.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Set

import numpy as np
import pandas as pd
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

try:
    from matplotlib_venn import venn2
except ImportError:
    venn2 = None

logger = logging.getLogger(__name__)


class PlotlyVennDiagram(PlotlyFigure):
    """Venn diagram implementation using Plotly."""

    def create(
        self,
        set1: Union[Set, List, pd.Series],
        set2: Union[Set, List, pd.Series],
        set1_label: str = "Set 1",
        set2_label: str = "Set 2",
        title: str = "Venn Diagram",
        annotation: Optional[Dict[str, str]] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a Venn diagram using Plotly.

        Parameters:
            set1 (Set | List | pd.Series): First set of elements.
            set2 (Set | List | pd.Series): Second set of elements.
            set1_label (str): Label for the first set.
            set2_label (str): Label for the second set.
            title (str): Title for the plot.
            annotation (Dict[str, str], optional): Custom annotation text.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Extra arguments passed to go shapes or traces.

        Returns:
            go.Figure: Plotly figure containing the Venn diagram.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Convert to sets
                set1 = set(set1)
                set2 = set(set2)
                n1 = len(set1)
                n2 = len(set2)
                n12 = len(set1 & set2)

                # Calculate circle positions and sizes
                # These are heuristics for visual clarity
                r1 = max(0.5, n1**0.5 / 6)
                r2 = max(0.5, n2**0.5 / 6)
                d = 1.2 * max(r1, r2)
                x1, y1 = 0, 0
                x2, y2 = d, 0

                fig = go.Figure()

                # Draw circles using shapes
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=x1 - r1,
                    y0=y1 - r1,
                    x1=x1 + r1,
                    y1=y1 + r1,
                    line_color="blue",
                    fillcolor="rgba(0,0,255,0.2)",
                    layer="below",
                )
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=x2 - r2,
                    y0=y2 - r2,
                    x1=x2 + r2,
                    y1=y2 + r2,
                    line_color="orange",
                    fillcolor="rgba(255,165,0,0.2)",
                    layer="below",
                )

                # Add set labels
                fig.add_annotation(
                    x=x1 - r1 * 0.7,
                    y=y1 + r1,
                    text=f"{set1_label}<br>n={n1}",
                    showarrow=False,
                    font=dict(size=14, color="blue"),
                    xanchor="right",
                )
                fig.add_annotation(
                    x=x2 + r2 * 0.7,
                    y=y2 + r2,
                    text=f"{set2_label}<br>n={n2}",
                    showarrow=False,
                    font=dict(size=14, color="orange"),
                    xanchor="left",
                )

                # Add intersection label
                fig.add_annotation(
                    x=(x1 + x2) / 2,
                    y=y1,
                    text=f"n={n12}",
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    xanchor="center",
                )

                # Add custom annotation if provided
                if annotation:
                    for key, value in annotation.items():
                        fig.add_annotation(
                            x=0.5 * (x1 + x2),
                            y=min(y1, y2) - max(r1, r2) * 0.7,
                            text=value,
                            showarrow=False,
                            font=dict(size=12),
                            xanchor="center",
                        )

                fig.update_layout(
                    title=title,
                    xaxis=dict(visible=False, range=[x1 - r1 * 1.5, x2 + r2 * 1.5]),
                    yaxis=dict(
                        visible=False, range=[-max(r1, r2) * 1.5, max(r1, r2) * 1.5]
                    ),
                    margin=dict(l=40, r=40, t=80, b=40),
                    plot_bgcolor="white",
                    showlegend=False,
                )

                fig = apply_theme_to_plotly_figure(fig)
                return fig
            except ImportError as e:
                logger.error(
                    f"Plotly is not available. Please install it with: pip install plotly. Error: {e}"
                )
                return self.create_empty_figure(
                    title=title,
                    message="Plotly is not available. Please install it with: pip install plotly.",
                )
            except Exception as e:
                logger.error(f"Error creating Venn diagram: {e}")
                return self.create_empty_figure(
                    title=title, message=f"Error creating Venn diagram: {str(e)}"
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
        Update an existing Plotly Venn diagram.

        Parameters:
            fig (go.Figure): Existing Plotly figure to update.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Parameters to update such as set1, set2, set1_label, set2_label, title.

        Returns:
            go.Figure: Updated Plotly figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import plotly.graph_objects as go

                # Validate figure type
                if not isinstance(fig, go.Figure):
                    logger.warning("Cannot update non-Plotly figure with PlotlyBarPlot")
                    return fig

                # Only support updating set data and labels for now
                set1 = kwargs.get("set1")
                set2 = kwargs.get("set2")
                set1_label = kwargs.get("set1_label", "Set 1")
                set2_label = kwargs.get("set2_label", "Set 2")
                title = kwargs.get("title", None)
                annotation = kwargs.get("annotation", None)

                if set1 is not None and set2 is not None:
                    # Re-create the diagram with new sets and labels
                    return self.create(
                        set1=set1,
                        set2=set2,
                        set1_label=set1_label,
                        set2_label=set2_label,
                        title=(
                            title or fig.layout.title.text
                            if fig.layout.title
                            else "Venn Diagram"
                        ),
                        annotation=annotation,
                        backend=backend,
                        theme=theme,
                        strict=strict,
                    )

                # Update title if provided
                if title:
                    fig.update_layout(title=title)

                # Update annotations if provided
                if annotation:
                    for ann in fig.layout.annotations:
                        if ann.text in annotation.values():
                            ann.text = annotation.get(ann.text, ann.text)

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
                logger.error(f"Error updating Venn diagram: {e}")
                return fig


class MatplotlibVennDiagram(MatplotlibFigure):
    """Venn diagram implementation using Matplotlib."""

    def create(
        self,
        set1: Union[Set, List, pd.Series],
        set2: Union[Set, List, pd.Series],
        set1_label: str = "Set 1",
        set2_label: str = "Set 2",
        title: str = "Venn Diagram",
        figsize: Tuple[int, int] = (5, 5),
        annotation: Optional[Dict[str, str]] = None,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a Venn diagram using Matplotlib.

        Parameters:
            set1 (Set | List | pd.Series): First set of elements.
            set2 (Set | List | pd.Series): Second set of elements.
            set1_label (str): Label for the first set.
            set2_label (str): Label for the second set.
            title (str): Title for the plot.
            figsize (Tuple[int, int]): Figure size.
            annotation (Dict[str, str], optional): Custom annotation text.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Extra arguments passed to venn2.

        Returns:
            plt.Figure: Matplotlib figure containing the Venn diagram.
        """
        if venn2 is None:
            raise ImportError(
                "matplotlib_venn is not installed. Run `pip install matplotlib-venn`."
            )

        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=figsize)

                # Ensure input is converted to sets
                set1 = set(set1)
                set2 = set(set2)

                venn_kwargs = {
                    "set_labels": (set1_label, set2_label),
                }

                if "set_colors" in kwargs:
                    venn_kwargs["set_colors"] = kwargs.pop("set_colors")

                venn2([set1, set2], ax=ax, **venn_kwargs, **kwargs)
                ax.set_title(title)

                if annotation:
                    for key, value in annotation.items():
                        ax.annotate(
                            value,
                            xy=(0.5, -0.1),
                            xycoords="axes fraction",
                            ha="center",
                            fontsize=10,
                        )

                # Finalize
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
                logger.error(f"Error creating Venn diagram with Matplotlib: {e}")
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
        fig: plt.Figure,
        backend: Optional[str] = None,
        theme: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> Any:
        """
        Update an existing Matplotlib Venn diagram.

        Parameters:
            fig (plt.Figure): Existing figure to update.
            backend (str, optional): Visualization backend.
            theme (str, optional): Visualization theme.
            strict (bool, optional): Strict mode for errors.
            **kwargs: Parameters to update such as title, set1_label, set2_label.

        Returns:
            plt.Figure: Updated Matplotlib figure.
        """
        with visualization_context(backend=backend, theme=theme, strict=strict):
            try:
                import matplotlib.pyplot as plt

                if not isinstance(fig, plt.Figure):
                    logger.warning("Provided object is not a Matplotlib figure.")
                    return fig

                if not fig.axes:
                    logger.warning("Figure has no axes to update.")
                    return fig

                ax = fig.axes[0]

                # Update title if provided
                if title := kwargs.get("title"):
                    ax.set_title(title)

                # Update labels (this assumes text[0] is set1_label, text[1] is set2_label)
                set1_label = kwargs.get("set1_label")
                set2_label = kwargs.get("set2_label")

                if set1_label or set2_label:
                    text_elements = ax.texts
                    if len(text_elements) >= 2:
                        if set1_label:
                            text_elements[0].set_text(set1_label)
                        if set2_label:
                            text_elements[1].set_text(set2_label)

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
                logger.error(f"Error updating Venn diagram: {e}")
                return fig


# Register with visualization system
FigureRegistry.register("venn_diagram", "plotly", PlotlyVennDiagram)
FigureRegistry.register("venn_diagram", "matplotlib", MatplotlibVennDiagram)
