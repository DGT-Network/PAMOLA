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
    from pamola_core.utils.vis_helpers import bar_plots  # noqa: F401
    from pamola_core.utils.vis_helpers import boxplot  # noqa: F401
    from pamola_core.utils.vis_helpers import combined_charts  # noqa: F401
    from pamola_core.utils.vis_helpers import cor_matrix  # noqa: F401
    from pamola_core.utils.vis_helpers import cor_pair  # noqa: F401
    from pamola_core.utils.vis_helpers import heatmap  # noqa: F401
    from pamola_core.utils.vis_helpers import histograms  # noqa: F401
    from pamola_core.utils.vis_helpers import line_plots  # noqa: F401
    from pamola_core.utils.vis_helpers import pie_charts  # noqa: F401
    from pamola_core.utils.vis_helpers import scatter_plots  # noqa: F401
    from pamola_core.utils.vis_helpers import spider_charts  # noqa: F401
    from pamola_core.utils.vis_helpers import venn_diagram  # noqa: F401
    from pamola_core.utils.vis_helpers import word_clouds  # noqa: F401
    from pamola_core.utils.vis_helpers import network_diagram  # noqa: F401

    _REGISTERED = True
