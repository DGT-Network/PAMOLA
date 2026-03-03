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

from pamola_core.attacks.base import AttackInitialization
from pamola_core.attacks.base import AttackInitialization as BaseAttack

from pamola_core.attacks.linkage_attack import LinkageAttack

from pamola_core.attacks.membership_inference import MembershipInference
from pamola_core.attacks.membership_inference import MembershipInference as MembershipInferenceAttack

from pamola_core.attacks.attribute_inference import AttributeInference
from pamola_core.attacks.attribute_inference import AttributeInference as AttributeInferenceAttack

from pamola_core.attacks.distance_to_closest_record import DistanceToClosestRecord
from pamola_core.attacks.distance_to_closest_record import DistanceToClosestRecord as DistanceToClosestRecordAttack

from pamola_core.attacks.nearest_neighbor_distance_ratio import NearestNeighborDistanceRatio
from pamola_core.attacks.nearest_neighbor_distance_ratio import NearestNeighborDistanceRatio as NearestNeighborDistanceRatioAttack

from pamola_core.attacks.attack_metrics import AttackMetrics

