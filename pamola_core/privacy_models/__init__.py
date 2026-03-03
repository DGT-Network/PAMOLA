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

Package: pamola_core.privacy_models
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

__all__ = [
    # Base
    "BasePrivacyModelProcessor",
    "PrivacyModel",
    # Core models
    "KAnonymityProcessor",
    "KAnonymityModel",
    "LDiversityCalculator",
    "LDiversityModel",
    "TCloseness",
    "TClosenessModel",
    "DifferentialPrivacyProcessor",
    "DifferentialPrivacyModel",
    # Reports
    "KAnonymityReport",
    "LDiversityReport",
    # Metrics / risk
    "LDiversityMetricsCalculator",
    "LDiversityPrivacyRiskAssessor",
    "AttributeDisclosureRiskAnalyzer",
    "RiskInterpreter",
    # Strategies / config
    "AnonymizationStrategy",
    "SuppressionStrategy",
    "FullMaskingStrategy",
    "PartialMaskingStrategy",
    "LDiversityModelApplicator",
]

from pamola_core.privacy_models.base import BasePrivacyModelProcessor
from pamola_core.privacy_models.base import BasePrivacyModelProcessor as PrivacyModel

from pamola_core.privacy_models.k_anonymity.calculation import KAnonymityProcessor
from pamola_core.privacy_models.k_anonymity.calculation import KAnonymityProcessor as KAnonymityModel

from pamola_core.privacy_models.l_diversity.calculation import LDiversityCalculator
from pamola_core.privacy_models.l_diversity.calculation import LDiversityCalculator as LDiversityModel

from pamola_core.privacy_models.t_closeness.calculation import TCloseness
from pamola_core.privacy_models.t_closeness.calculation import TCloseness as TClosenessModel

from pamola_core.privacy_models.differential_privacy.calculation import DifferentialPrivacyProcessor
from pamola_core.privacy_models.differential_privacy.calculation import DifferentialPrivacyProcessor as DifferentialPrivacyModel

from pamola_core.privacy_models.k_anonymity.ka_reporting import KAnonymityReport

from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport

from pamola_core.privacy_models.l_diversity.metrics import LDiversityMetricsCalculator

from pamola_core.privacy_models.l_diversity.privacy import LDiversityPrivacyRiskAssessor

from pamola_core.privacy_models.l_diversity.attribute_risk import AttributeDisclosureRiskAnalyzer

from pamola_core.privacy_models.l_diversity.interpretation import RiskInterpreter

from pamola_core.privacy_models.l_diversity.apply_model import AnonymizationStrategy
from pamola_core.privacy_models.l_diversity.apply_model import SuppressionStrategy
from pamola_core.privacy_models.l_diversity.apply_model import FullMaskingStrategy
from pamola_core.privacy_models.l_diversity.apply_model import PartialMaskingStrategy
from pamola_core.privacy_models.l_diversity.apply_model import LDiversityModelApplicator

