"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Visualization Helpers Package
Package:       pamola_core.utils.vis_helpers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
    This package implements a unified, thread-safe visualization framework supporting
    all major analytical and privacy-focused chart types required by PAMOLA data workflows.
    The architecture provides seamless integration of multiple plotting backends (Plotly,
    Matplotlib, WordCloud), advanced theme management, contextual backend switching, and
    composable chart construction for complex analytics and reporting.

    All operations are isolated via contextvars to guarantee thread and parallel
    safety, eliminating state leakage and supporting high-concurrency environments.

Key Features:
    - Unified public API for bar, histogram, boxplot, line, scatter, heatmap, pie, sunburst,
      spider/radar, combined (dual axis), correlation, and word cloud visualizations
    - Modular base and registry system for extensibility and pluggable new chart types
    - Advanced theme and color management (user, system, and custom themes)
    - Support for context-based backend switching (Plotly/Matplotlib/WordCloud)
    - Utility functions for correlation analysis, masking, color/annotation formatting
    - Complete integration with IO and pipeline modules for secure, consistent file output
    - All components are thread/context safe for parallel and asynchronous workflows

Changelog:
    2.0.0 - Unified all chart types, added combined, spider, pie, sunburst, and advanced correlation plots
          - Upgraded registry, theme, and context isolation
          - Enhanced type hints and modularity for all public APIs
    1.1.1 - Added thread/context safety for backend and theme management
    1.0.0 - Initial implementation with base plotting classes

Dependencies:
    - pandas       - DataFrame operations
    - numpy        - Numeric operations, arrays
    - matplotlib   - Matplotlib backend
    - plotly       - Plotly backend
    - wordcloud    - Word cloud visualizations
    - PIL          - Image operations for text analytics
    - contextvars  - Thread/context management
    - typing       - Type hints and validation
    - pathlib      - Path operations

"""

# Import pamola_core base classes and factories
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

# Theme management
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

# Context management
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
from pamola_core.utils.vis_helpers.network_diagram import (
    PlotlyNetworkDiagram,
    MatplotlibNetworkDiagram,
)

# Correlation and annotation utilities
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

# Version string (update with each major release)
__version__ = "2.0.0"

# Unified public API for import *
__all__ = [
    # Base classes and factories
    "BaseFigure",
    "PlotlyFigure",
    "MatplotlibFigure",
    "FigureFactory",
    "FigureRegistry",
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
    "PlotlyNetworkDiagram",
    "MatplotlibNetworkDiagram",
    # Correlation/annotation utilities
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
