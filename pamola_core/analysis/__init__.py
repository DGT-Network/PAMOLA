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

Package: pamola_core.analysis
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    # dataset_summary.py
    "analyze_dataset_summary",
    # privacy_risk.py
    "calculate_full_risk",
    # descriptive_stats.py
    "analyze_descriptive_stats",
    # distribution.py
    "visualize_distribution_df",
    # correlation.py
    "analyze_correlation",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "analyze_dataset_summary": "pamola_core.analysis.dataset_summary",
    "calculate_full_risk": "pamola_core.analysis.privacy_risk",
    "analyze_descriptive_stats": "pamola_core.analysis.descriptive_stats",
    "visualize_distribution_df": "pamola_core.analysis.distribution",
    "analyze_correlation": "pamola_core.analysis.correlation",
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        target = _LAZY_IMPORTS[name]
        if isinstance(target, tuple):
            module_name, attr_name = target
        else:
            module_name = target
            attr_name = name
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
