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

import importlib
from typing import Dict

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

_LAZY_IMPORTS: Dict[str, tuple[str, str] | str] = {
    "BasePrivacyModelProcessor": "pamola_core.privacy_models.base",
    "PrivacyModel": ("pamola_core.privacy_models.base", "BasePrivacyModelProcessor"),
    "KAnonymityProcessor": "pamola_core.privacy_models.k_anonymity.calculation",
    "KAnonymityModel": ("pamola_core.privacy_models.k_anonymity.calculation", "KAnonymityProcessor"),
    "LDiversityCalculator": "pamola_core.privacy_models.l_diversity.calculation",
    "LDiversityModel": ("pamola_core.privacy_models.l_diversity.calculation", "LDiversityCalculator"),
    "TCloseness": "pamola_core.privacy_models.t_closeness.calculation",
    "TClosenessModel": ("pamola_core.privacy_models.t_closeness.calculation", "TCloseness"),
    "DifferentialPrivacyProcessor": "pamola_core.privacy_models.differential_privacy.calculation",
    "DifferentialPrivacyModel": ("pamola_core.privacy_models.differential_privacy.calculation", "DifferentialPrivacyProcessor"),
    "KAnonymityReport": "pamola_core.privacy_models.k_anonymity.ka_reporting",
    "LDiversityReport": "pamola_core.privacy_models.l_diversity.reporting",
    "LDiversityMetricsCalculator": "pamola_core.privacy_models.l_diversity.metrics",
    "LDiversityPrivacyRiskAssessor": "pamola_core.privacy_models.l_diversity.privacy",
    "AttributeDisclosureRiskAnalyzer": "pamola_core.privacy_models.l_diversity.attribute_risk",
    "RiskInterpreter": "pamola_core.privacy_models.l_diversity.interpretation",
    "AnonymizationStrategy": "pamola_core.privacy_models.l_diversity.apply_model",
    "SuppressionStrategy": "pamola_core.privacy_models.l_diversity.apply_model",
    "FullMaskingStrategy": "pamola_core.privacy_models.l_diversity.apply_model",
    "PartialMaskingStrategy": "pamola_core.privacy_models.l_diversity.apply_model",
    "LDiversityModelApplicator": "pamola_core.privacy_models.l_diversity.apply_model",
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
