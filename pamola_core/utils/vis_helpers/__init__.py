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

import importlib
from typing import Dict
from pamola_core.utils.vis_helpers.registry import register_builtin_figures

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

_LAZY_IMPORTS: Dict[str, str] = {
    'BaseFigure': 'pamola_core.utils.vis_helpers.base',
    'FigureFactory': 'pamola_core.utils.vis_helpers.base',
    'FigureRegistry': 'pamola_core.utils.vis_helpers.base',
    'MatplotlibBarPlot': 'pamola_core.utils.vis_helpers.bar_plots',
    'MatplotlibBoxPlot': 'pamola_core.utils.vis_helpers.boxplot',
    'MatplotlibCombinedChart': 'pamola_core.utils.vis_helpers.combined_charts',
    'MatplotlibCorrelationMatrix': 'pamola_core.utils.vis_helpers.cor_matrix',
    'MatplotlibCorrelationPair': 'pamola_core.utils.vis_helpers.cor_pair',
    'MatplotlibFigure': 'pamola_core.utils.vis_helpers.base',
    'MatplotlibHeatmap': 'pamola_core.utils.vis_helpers.heatmap',
    'MatplotlibHistogram': 'pamola_core.utils.vis_helpers.histograms',
    'MatplotlibLinePlot': 'pamola_core.utils.vis_helpers.line_plots',
    'MatplotlibNetworkDiagram': 'pamola_core.utils.vis_helpers.network_diagram',
    'MatplotlibPieChart': 'pamola_core.utils.vis_helpers.pie_charts',
    'MatplotlibScatterPlot': 'pamola_core.utils.vis_helpers.scatter_plots',
    'MatplotlibSpiderChart': 'pamola_core.utils.vis_helpers.spider_charts',
    'MatplotlibVennDiagram': 'pamola_core.utils.vis_helpers.venn_diagram',
    'PlotlyBarPlot': 'pamola_core.utils.vis_helpers.bar_plots',
    'PlotlyBoxPlot': 'pamola_core.utils.vis_helpers.boxplot',
    'PlotlyCombinedChart': 'pamola_core.utils.vis_helpers.combined_charts',
    'PlotlyCorrelationMatrix': 'pamola_core.utils.vis_helpers.cor_matrix',
    'PlotlyCorrelationPair': 'pamola_core.utils.vis_helpers.cor_pair',
    'PlotlyFigure': 'pamola_core.utils.vis_helpers.base',
    'PlotlyHeatmap': 'pamola_core.utils.vis_helpers.heatmap',
    'PlotlyHistogram': 'pamola_core.utils.vis_helpers.histograms',
    'PlotlyLinePlot': 'pamola_core.utils.vis_helpers.line_plots',
    'PlotlyNetworkDiagram': 'pamola_core.utils.vis_helpers.network_diagram',
    'PlotlyPieChart': 'pamola_core.utils.vis_helpers.pie_charts',
    'PlotlyScatterPlot': 'pamola_core.utils.vis_helpers.scatter_plots',
    'PlotlySpiderChart': 'pamola_core.utils.vis_helpers.spider_charts',
    'PlotlySunburstChart': 'pamola_core.utils.vis_helpers.pie_charts',
    'PlotlyVennDiagram': 'pamola_core.utils.vis_helpers.venn_diagram',
    'WordCloudGenerator': 'pamola_core.utils.vis_helpers.word_clouds',
    'apply_mask': 'pamola_core.utils.vis_helpers.cor_utils',
    'apply_theme_to_matplotlib_figure': 'pamola_core.utils.vis_helpers.theme',
    'apply_theme_to_plotly_figure': 'pamola_core.utils.vis_helpers.theme',
    'auto_visualization_context': 'pamola_core.utils.vis_helpers.context',
    'calculate_correlation': 'pamola_core.utils.vis_helpers.cor_utils',
    'calculate_symmetric_colorscale_range': 'pamola_core.utils.vis_helpers.cor_utils',
    'create_correlation_mask': 'pamola_core.utils.vis_helpers.cor_utils',
    'create_custom_theme': 'pamola_core.utils.vis_helpers.theme',
    'create_significance_mask': 'pamola_core.utils.vis_helpers.cor_utils',
    'create_text_colors_array': 'pamola_core.utils.vis_helpers.cor_utils',
    'ensure_series': 'pamola_core.utils.vis_helpers.base',
    'get_backend': 'pamola_core.utils.vis_helpers.base',
    'get_colorscale': 'pamola_core.utils.vis_helpers.theme',
    'get_current_theme': 'pamola_core.utils.vis_helpers.theme',
    'get_current_theme_name': 'pamola_core.utils.vis_helpers.theme',
    'get_figure_size': 'pamola_core.utils.vis_helpers.context',
    'get_matplotlib_colormap': 'pamola_core.utils.vis_helpers.theme',
    'get_theme_colors': 'pamola_core.utils.vis_helpers.theme',
    'matplotlib_agg_context': 'pamola_core.utils.vis_helpers.context',
    'null_context': 'pamola_core.utils.vis_helpers.context',
    'parse_annotation_format': 'pamola_core.utils.vis_helpers.cor_utils',
    'prepare_correlation_data': 'pamola_core.utils.vis_helpers.cor_utils',
    'prepare_dataframe': 'pamola_core.utils.vis_helpers.base',
    'prepare_hover_texts': 'pamola_core.utils.vis_helpers.cor_utils',
    'register_figure': 'pamola_core.utils.vis_helpers.context',
    'set_backend': 'pamola_core.utils.vis_helpers.base',
    'set_theme': 'pamola_core.utils.vis_helpers.theme',
    'sort_series': 'pamola_core.utils.vis_helpers.base',
    'visualization_context': 'pamola_core.utils.vis_helpers.context',
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
