"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Package: pamola_core.utils.vis_helpers
Type: Internal (Non-Public API)

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from pamola_core.utils.vis_helpers.registry import register_builtin_figures

from pamola_core.utils.vis_helpers.base import BaseFigure
from pamola_core.utils.vis_helpers.base import FigureFactory
from pamola_core.utils.vis_helpers.base import FigureRegistry
from pamola_core.utils.vis_helpers.base import MatplotlibFigure
from pamola_core.utils.vis_helpers.base import PlotlyFigure
from pamola_core.utils.vis_helpers.base import ensure_series
from pamola_core.utils.vis_helpers.base import get_backend
from pamola_core.utils.vis_helpers.base import prepare_dataframe
from pamola_core.utils.vis_helpers.base import set_backend
from pamola_core.utils.vis_helpers.base import sort_series

from pamola_core.utils.vis_helpers.bar_plots import MatplotlibBarPlot
from pamola_core.utils.vis_helpers.bar_plots import PlotlyBarPlot

from pamola_core.utils.vis_helpers.boxplot import MatplotlibBoxPlot
from pamola_core.utils.vis_helpers.boxplot import PlotlyBoxPlot

from pamola_core.utils.vis_helpers.combined_charts import MatplotlibCombinedChart
from pamola_core.utils.vis_helpers.combined_charts import PlotlyCombinedChart

from pamola_core.utils.vis_helpers.cor_matrix import MatplotlibCorrelationMatrix
from pamola_core.utils.vis_helpers.cor_matrix import PlotlyCorrelationMatrix

from pamola_core.utils.vis_helpers.cor_pair import MatplotlibCorrelationPair
from pamola_core.utils.vis_helpers.cor_pair import PlotlyCorrelationPair

from pamola_core.utils.vis_helpers.heatmap import MatplotlibHeatmap
from pamola_core.utils.vis_helpers.heatmap import PlotlyHeatmap

from pamola_core.utils.vis_helpers.histograms import MatplotlibHistogram
from pamola_core.utils.vis_helpers.histograms import PlotlyHistogram

from pamola_core.utils.vis_helpers.line_plots import MatplotlibLinePlot
from pamola_core.utils.vis_helpers.line_plots import PlotlyLinePlot

from pamola_core.utils.vis_helpers.network_diagram import MatplotlibNetworkDiagram
from pamola_core.utils.vis_helpers.network_diagram import PlotlyNetworkDiagram

from pamola_core.utils.vis_helpers.pie_charts import MatplotlibPieChart
from pamola_core.utils.vis_helpers.pie_charts import PlotlyPieChart
from pamola_core.utils.vis_helpers.pie_charts import PlotlySunburstChart

from pamola_core.utils.vis_helpers.scatter_plots import MatplotlibScatterPlot
from pamola_core.utils.vis_helpers.scatter_plots import PlotlyScatterPlot

from pamola_core.utils.vis_helpers.spider_charts import MatplotlibSpiderChart
from pamola_core.utils.vis_helpers.spider_charts import PlotlySpiderChart

from pamola_core.utils.vis_helpers.venn_diagram import MatplotlibVennDiagram
from pamola_core.utils.vis_helpers.venn_diagram import PlotlyVennDiagram

from pamola_core.utils.vis_helpers.word_clouds import WordCloudGenerator

from pamola_core.utils.vis_helpers.cor_utils import apply_mask
from pamola_core.utils.vis_helpers.cor_utils import calculate_correlation
from pamola_core.utils.vis_helpers.cor_utils import calculate_symmetric_colorscale_range
from pamola_core.utils.vis_helpers.cor_utils import create_correlation_mask
from pamola_core.utils.vis_helpers.cor_utils import create_significance_mask
from pamola_core.utils.vis_helpers.cor_utils import create_text_colors_array
from pamola_core.utils.vis_helpers.cor_utils import parse_annotation_format
from pamola_core.utils.vis_helpers.cor_utils import prepare_correlation_data
from pamola_core.utils.vis_helpers.cor_utils import prepare_hover_texts

from pamola_core.utils.vis_helpers.theme import apply_theme_to_matplotlib_figure
from pamola_core.utils.vis_helpers.theme import apply_theme_to_plotly_figure
from pamola_core.utils.vis_helpers.theme import create_custom_theme
from pamola_core.utils.vis_helpers.theme import get_colorscale
from pamola_core.utils.vis_helpers.theme import get_current_theme
from pamola_core.utils.vis_helpers.theme import get_current_theme_name
from pamola_core.utils.vis_helpers.theme import get_matplotlib_colormap
from pamola_core.utils.vis_helpers.theme import get_theme_colors
from pamola_core.utils.vis_helpers.theme import set_theme

from pamola_core.utils.vis_helpers.context import auto_visualization_context
from pamola_core.utils.vis_helpers.context import get_figure_size
from pamola_core.utils.vis_helpers.context import matplotlib_agg_context
from pamola_core.utils.vis_helpers.context import null_context
from pamola_core.utils.vis_helpers.context import register_figure
from pamola_core.utils.vis_helpers.context import visualization_context

__all__ = [
    'BaseFigure',
    'PlotlyFigure',
    'MatplotlibFigure',
    'FigureFactory',
    'FigureRegistry',
    'set_backend',
    'get_backend',
    'ensure_series',
    'sort_series',
    'prepare_dataframe',
    'set_theme',
    'get_current_theme',
    'get_current_theme_name',
    'create_custom_theme',
    'get_theme_colors',
    'apply_theme_to_plotly_figure',
    'apply_theme_to_matplotlib_figure',
    'get_colorscale',
    'get_matplotlib_colormap',
    'visualization_context',
    'matplotlib_agg_context',
    'null_context',
    'register_figure',
    'get_figure_size',
    'auto_visualization_context',
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
    'MatplotlibVennDiagram',
    'PlotlyVennDiagram',
    'MatplotlibCombinedChart',
    'MatplotlibCorrelationMatrix',
    'MatplotlibCorrelationPair',
    'MatplotlibLinePlot',
    'MatplotlibScatterPlot',
    'MatplotlibPieChart',
    'MatplotlibSpiderChart',
    'PlotlyNetworkDiagram',
    'MatplotlibNetworkDiagram',
    'prepare_correlation_data',
    'create_correlation_mask',
    'apply_mask',
    'create_text_colors_array',
    'create_significance_mask',
    'prepare_hover_texts',
    'parse_annotation_format',
    'calculate_symmetric_colorscale_range',
    'calculate_correlation',
    'register_builtin_figures',
]

