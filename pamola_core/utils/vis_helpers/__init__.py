"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Visualization Helpers Package
Description: Thread-safe visualization capabilities for data analysis and privacy metrics
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This package provides implementation details for the visualization system,
including various plotting engines, themes, and specialized visualizations.
The implementation uses contextvars to ensure that theme and backend settings
are properly isolated between concurrent execution contexts, eliminating state
interference when multiple visualization operations run in parallel.
"""

# Import pamola_core functionality
from pamola_core.utils.vis_helpers.base import (
    BaseFigure,
    PlotlyFigure,
    MatplotlibFigure,
    FigureFactory,
    FigureRegistry,
    set_backend,
    get_backend,
    ensure_series,
    sort_series,
    prepare_dataframe,
)

# Import theme management
from pamola_core.utils.vis_helpers.theme import (
    set_theme,
    get_current_theme,
    get_current_theme_name,
    create_custom_theme,
    get_theme_colors,
    apply_theme_to_plotly_figure,
    apply_theme_to_matplotlib_figure,
    get_colorscale,
    get_matplotlib_colormap,
)

# Import context management
from pamola_core.utils.vis_helpers.context import (
    visualization_context,
    matplotlib_agg_context,
    null_context,
    register_figure,
    get_figure_size,
    auto_visualization_context,
)

# Import plot implementations
from pamola_core.utils.vis_helpers.bar_plots import PlotlyBarPlot, MatplotlibBarPlot
from pamola_core.utils.vis_helpers.boxplot import PlotlyBoxPlot, MatplotlibBoxPlot
from pamola_core.utils.vis_helpers.combined_charts import (
    PlotlyCombinedChart,
    MatplotlibCombinedChart,
)
from pamola_core.utils.vis_helpers.cor_matrix import (
    PlotlyCorrelationMatrix,
    MatplotlibCorrelationMatrix,
)
from pamola_core.utils.vis_helpers.cor_pair import (
    PlotlyCorrelationPair,
    MatplotlibCorrelationPair,
)
from pamola_core.utils.vis_helpers.heatmap import PlotlyHeatmap, MatplotlibHeatmap
from pamola_core.utils.vis_helpers.histograms import (
    PlotlyHistogram,
    MatplotlibHistogram,
)
from pamola_core.utils.vis_helpers.line_plots import PlotlyLinePlot, MatplotlibLinePlot
from pamola_core.utils.vis_helpers.pie_charts import (
    PlotlyPieChart,
    PlotlySunburstChart,
    MatplotlibPieChart,
)
from pamola_core.utils.vis_helpers.scatter_plots import (
    PlotlyScatterPlot,
    MatplotlibScatterPlot,
)
from pamola_core.utils.vis_helpers.spider_charts import (
    PlotlySpiderChart,
    MatplotlibSpiderChart,
)
from pamola_core.utils.vis_helpers.word_clouds import WordCloudGenerator
from pamola_core.utils.vis_helpers.venn_diagram import (
    PlotlyVennDiagram,
    MatplotlibVennDiagram,
)

# Import utility functions
from pamola_core.utils.vis_helpers.cor_utils import (
    prepare_correlation_data,
    create_correlation_mask,
    apply_mask,
    create_text_colors_array,
    create_significance_mask,
    prepare_hover_texts,
    parse_annotation_format,
    calculate_symmetric_colorscale_range,
    calculate_correlation,
)

# Define version
__version__ = "1.1.1"  # Updated version to reflect thread-safe changes with additional context improvements

# Define public API
__all__ = [
    # Base classes and factories
    "BaseFigure",
    "PlotlyFigure",
    "MatplotlibFigure",
    "FigureFactory",
    "FigureRegistry",
    # Pamola Core functions
    "set_backend",
    "get_backend",
    "ensure_series",
    "sort_series",
    "prepare_dataframe",
    # Theme management
    "set_theme",
    "get_current_theme",
    "get_current_theme_name",
    "create_custom_theme",
    "get_theme_colors",
    "apply_theme_to_plotly_figure",
    "apply_theme_to_matplotlib_figure",
    "get_colorscale",
    "get_matplotlib_colormap",
    # Context management
    "visualization_context",
    "matplotlib_agg_context",
    "null_context",
    "register_figure",
    "get_figure_size",
    "auto_visualization_context",
    # Plot implementations
    "PlotlyBarPlot",
    "PlotlyCombinedChart",
    "PlotlyPieChart",
    "PlotlySunburstChart",
    "PlotlySpiderChart",
    "MatplotlibBarPlot",
    "PlotlyHistogram",
    "MatplotlibHistogram",
    "PlotlyScatterPlot",
    "PlotlyBoxPlot",
    "MatplotlibBoxPlot",
    "PlotlyHeatmap",
    "MatplotlibHeatmap",
    "PlotlyLinePlot",
    "PlotlyCorrelationMatrix",
    "PlotlyCorrelationPair",
    "WordCloudGenerator",
    "MatplotlibVennDiagram",
    "PlotlyVennDiagram",
    "MatplotlibCombinedChart",
    "MatplotlibCorrelationMatrix",
    "MatplotlibCorrelationPair",
    "MatplotlibLinePlot",
    "MatplotlibScatterPlot",
    "MatplotlibPieChart",
    "MatplotlibSpiderChart",
    # Utility functions
    "prepare_correlation_data",
    "create_correlation_mask",
    "apply_mask",
    "create_text_colors_array",
    "create_significance_mask",
    "prepare_hover_texts",
    "parse_annotation_format",
    "calculate_symmetric_colorscale_range",
    "calculate_correlation",
]
