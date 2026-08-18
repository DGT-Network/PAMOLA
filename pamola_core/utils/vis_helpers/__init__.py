"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
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

# Lazy re-exports (TD-PC-21 / viz import boundary).
#
# This module used to import all 63 names eagerly. Importing *any* submodule of
# this package runs this file, and `pamola_core.utils.visualization` imports
# `vis_helpers.base` — so `import pamola_core` pulled in every chart backend,
# and with them matplotlib, plotly, wordcloud and matplotlib-venn. That is the
# single reason the base install could not do without the plotting stack.
#
# The eager imports were never needed for correctness: figure classes register
# themselves through `registry.register_builtin_figures()`, which
# `FigureFactory.create_figure()` already calls on demand. They were convenience
# re-exports, and nothing inside this repository imports them from the package
# root — but an external caller might, so the names stay available through
# PEP 562 module `__getattr__` instead of disappearing.
#
# Cost after the first access is a dict lookup: the resolved attribute is cached
# in the module namespace.

import importlib
from typing import Any

_NAME_TO_MODULE = {
    "BaseFigure": "base",
    "FigureFactory": "base",
    "FigureRegistry": "base",
    "MatplotlibBarPlot": "bar_plots",
    "MatplotlibBoxPlot": "boxplot",
    "MatplotlibCombinedChart": "combined_charts",
    "MatplotlibCorrelationMatrix": "cor_matrix",
    "MatplotlibCorrelationPair": "cor_pair",
    "MatplotlibFigure": "base",
    "MatplotlibHeatmap": "heatmap",
    "MatplotlibHistogram": "histograms",
    "MatplotlibLinePlot": "line_plots",
    "MatplotlibNetworkDiagram": "network_diagram",
    "MatplotlibPieChart": "pie_charts",
    "MatplotlibScatterPlot": "scatter_plots",
    "MatplotlibSpiderChart": "spider_charts",
    "MatplotlibVennDiagram": "venn_diagram",
    "PlotlyBarPlot": "bar_plots",
    "PlotlyBoxPlot": "boxplot",
    "PlotlyCombinedChart": "combined_charts",
    "PlotlyCorrelationMatrix": "cor_matrix",
    "PlotlyCorrelationPair": "cor_pair",
    "PlotlyFigure": "base",
    "PlotlyHeatmap": "heatmap",
    "PlotlyHistogram": "histograms",
    "PlotlyLinePlot": "line_plots",
    "PlotlyNetworkDiagram": "network_diagram",
    "PlotlyPieChart": "pie_charts",
    "PlotlyScatterPlot": "scatter_plots",
    "PlotlySpiderChart": "spider_charts",
    "PlotlySunburstChart": "pie_charts",
    "PlotlyVennDiagram": "venn_diagram",
    "WordCloudGenerator": "word_clouds",
    "apply_mask": "cor_utils",
    "apply_theme_to_matplotlib_figure": "theme",
    "apply_theme_to_plotly_figure": "theme",
    "auto_visualization_context": "context",
    "calculate_correlation": "cor_utils",
    "calculate_symmetric_colorscale_range": "cor_utils",
    "create_correlation_mask": "cor_utils",
    "create_custom_theme": "theme",
    "create_significance_mask": "cor_utils",
    "create_text_colors_array": "cor_utils",
    "ensure_series": "base",
    "get_backend": "base",
    "get_colorscale": "theme",
    "get_current_theme": "theme",
    "get_current_theme_name": "theme",
    "get_figure_size": "context",
    "get_matplotlib_colormap": "theme",
    "get_theme_colors": "theme",
    "matplotlib_agg_context": "context",
    "null_context": "context",
    "parse_annotation_format": "cor_utils",
    "prepare_correlation_data": "cor_utils",
    "prepare_dataframe": "base",
    "prepare_hover_texts": "cor_utils",
    "register_builtin_figures": "registry",
    "register_figure": "context",
    "set_backend": "base",
    "set_theme": "theme",
    "sort_series": "base",
    "visualization_context": "context",
}


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name by importing its submodule on first use."""
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache; __getattr__ is not consulted again
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_NAME_TO_MODULE))


__all__ = [
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
    "set_theme",
    "get_current_theme",
    "get_current_theme_name",
    "create_custom_theme",
    "get_theme_colors",
    "apply_theme_to_plotly_figure",
    "apply_theme_to_matplotlib_figure",
    "get_colorscale",
    "get_matplotlib_colormap",
    "visualization_context",
    "matplotlib_agg_context",
    "null_context",
    "register_figure",
    "get_figure_size",
    "auto_visualization_context",
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
    "prepare_correlation_data",
    "create_correlation_mask",
    "apply_mask",
    "create_text_colors_array",
    "create_significance_mask",
    "prepare_hover_texts",
    "parse_annotation_format",
    "calculate_symmetric_colorscale_range",
    "calculate_correlation",
    "register_builtin_figures",
]
