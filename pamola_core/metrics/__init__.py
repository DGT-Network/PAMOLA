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

Package: pamola_core.metrics
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    # operations/
    "FidelityOperation",
    "PrivacyMetricOperation",
    "UtilityMetricOperation",
    # commons/
    "RuleCode",
    "SchemaManager",
    "calculate_quality_with_rules",
    "calculate_provisional_risk",
    "calculate_predicted_utility",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "FidelityOperation": "pamola_core.metrics.operations.fidelity_ops",
    "PrivacyMetricOperation": "pamola_core.metrics.operations.privacy_ops",
    "UtilityMetricOperation": "pamola_core.metrics.operations.utility_ops",
    "RuleCode": "pamola_core.metrics.commons",
    "SchemaManager": "pamola_core.metrics.commons",
    "calculate_quality_with_rules": "pamola_core.metrics.commons",
    "calculate_provisional_risk": "pamola_core.metrics.commons",
    "calculate_predicted_utility": "pamola_core.metrics.commons",
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
