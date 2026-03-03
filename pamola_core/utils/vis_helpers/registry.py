"""
Registry bootstrap for visualization helpers.

Imports figure modules to register built-in implementations.
"""

_REGISTERED = False


def register_builtin_figures() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    # Import modules to trigger FigureRegistry registration
    import pamola_core.utils.vis_helpers.bar_plots  # noqa: F401
    import pamola_core.utils.vis_helpers.boxplot  # noqa: F401
    import pamola_core.utils.vis_helpers.combined_charts  # noqa: F401
    import pamola_core.utils.vis_helpers.cor_matrix  # noqa: F401
    import pamola_core.utils.vis_helpers.cor_pair  # noqa: F401
    import pamola_core.utils.vis_helpers.heatmap  # noqa: F401
    import pamola_core.utils.vis_helpers.histograms  # noqa: F401
    import pamola_core.utils.vis_helpers.line_plots  # noqa: F401
    import pamola_core.utils.vis_helpers.network_diagram  # noqa: F401
    import pamola_core.utils.vis_helpers.pie_charts  # noqa: F401
    import pamola_core.utils.vis_helpers.scatter_plots  # noqa: F401
    import pamola_core.utils.vis_helpers.spider_charts  # noqa: F401
    import pamola_core.utils.vis_helpers.venn_diagram  # noqa: F401
    import pamola_core.utils.vis_helpers.word_clouds  # noqa: F401

    _REGISTERED = True
