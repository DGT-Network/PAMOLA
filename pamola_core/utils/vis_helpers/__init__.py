"""
Visualization Helpers Package for HHR Project.

This package provides implementation details for the visualization system,
including various plotting engines, themes, and specialized visualizations.
"""

# Import core functionality
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
    prepare_dataframe
)

# Import theme management
from pamola_core.utils.vis_helpers.theme import (
    set_theme,
    get_current_theme,
    create_custom_theme,
    get_theme_colors,
    apply_theme_to_plotly_figure,
    apply_theme_to_matplotlib_figure,
    get_colorscale,
    get_matplotlib_colormap
)

# Import plot implementations
from pamola_core.utils.vis_helpers.bar_plots import PlotlyBarPlot, MatplotlibBarPlot
from pamola_core.utils.vis_helpers.boxplot import PlotlyBoxPlot, MatplotlibBoxPlot
from pamola_core.utils.vis_helpers.combined_charts import PlotlyCombinedChart
from pamola_core.utils.vis_helpers.cor_matrix import PlotlyCorrelationMatrix
from pamola_core.utils.vis_helpers.cor_pair import PlotlyCorrelationPair
from pamola_core.utils.vis_helpers.heatmap import PlotlyHeatmap, MatplotlibHeatmap
from pamola_core.utils.vis_helpers.histograms import PlotlyHistogram, MatplotlibHistogram
from pamola_core.utils.vis_helpers.line_plots import PlotlyLinePlot
from pamola_core.utils.vis_helpers.pie_charts import PlotlyPieChart, PlotlySunburstChart
from pamola_core.utils.vis_helpers.scatter_plots import PlotlyScatterPlot
from pamola_core.utils.vis_helpers.spider_charts import PlotlySpiderChart
from pamola_core.utils.vis_helpers.word_clouds import WordCloudGenerator

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
    calculate_correlation
)

# Define version
__version__ = '1.0.0'

# Define public API
__all__ = [
    # Base classes and factories
    'BaseFigure',
    'PlotlyFigure',
    'MatplotlibFigure',
    'FigureFactory',
    'FigureRegistry',

    # Core functions
    'set_backend',
    'get_backend',
    'ensure_series',
    'sort_series',
    'prepare_dataframe',

    # Theme management
    'set_theme',
    'get_current_theme',
    'create_custom_theme',
    'get_theme_colors',
    'apply_theme_to_plotly_figure',
    'apply_theme_to_matplotlib_figure',
    'get_colorscale',
    'get_matplotlib_colormap',

    # Plot implementations
    'PlotlyBarPlot',
    'PlotlyCombinedChart',
    'PlotlyPieChart',
    'PlotlySunburstChart',
    'PlotlySpiderChart',
    'MatplotlibBarPlot',
    'PlotlyHistogram',
    'MatplotlibHistogram',
    'PlotlyScatterPlot',
    'PlotlyBoxPlot',
    'MatplotlibBoxPlot',
    'PlotlyHeatmap',
    'MatplotlibHeatmap',
    'PlotlyLinePlot',
    'PlotlyCorrelationMatrix',
    'PlotlyCorrelationPair',
    'WordCloudGenerator',

    # Utility functions
    'prepare_correlation_data',
    'create_correlation_mask',
    'apply_mask',
    'create_text_colors_array',
    'create_significance_mask',
    'prepare_hover_texts',
    'parse_annotation_format',
    'calculate_symmetric_colorscale_range',
    'calculate_correlation'
]