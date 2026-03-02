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

Package: pamola_core.attacks
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    # Base
    "AttackInitialization",
    "BaseAttack",
    # Core attacks
    "LinkageAttack",
    "MembershipInference",
    "MembershipInferenceAttack",
    "AttributeInference",
    "AttributeInferenceAttack",
    "DistanceToClosestRecord",
    "DistanceToClosestRecordAttack",
    "NearestNeighborDistanceRatio",
    "NearestNeighborDistanceRatioAttack",
    # Metrics
    "AttackMetrics",
]

_LAZY_IMPORTS: Dict[str, tuple[str, str] | str] = {
    "AttackInitialization": "pamola_core.attacks.base",
    "BaseAttack": ("pamola_core.attacks.base", "AttackInitialization"),
    "LinkageAttack": "pamola_core.attacks.linkage_attack",
    "MembershipInference": "pamola_core.attacks.membership_inference",
    "MembershipInferenceAttack": ("pamola_core.attacks.membership_inference", "MembershipInference"),
    "AttributeInference": "pamola_core.attacks.attribute_inference",
    "AttributeInferenceAttack": ("pamola_core.attacks.attribute_inference", "AttributeInference"),
    "DistanceToClosestRecord": "pamola_core.attacks.distance_to_closest_record",
    "DistanceToClosestRecordAttack": ("pamola_core.attacks.distance_to_closest_record", "DistanceToClosestRecord"),
    "NearestNeighborDistanceRatio": "pamola_core.attacks.nearest_neighbor_distance_ratio",
    "NearestNeighborDistanceRatioAttack": ("pamola_core.attacks.nearest_neighbor_distance_ratio", "NearestNeighborDistanceRatio"),
    "AttackMetrics": "pamola_core.attacks.attack_metrics",
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
