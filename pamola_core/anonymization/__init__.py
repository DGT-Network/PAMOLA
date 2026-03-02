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

Package: pamola_core.anonymization
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    "AttributeSuppressionOperation",
    "CategoricalGeneralizationOperation",
    "CellSuppressionOperation",
    "DateTimeGeneralizationOperation",
    "FullMaskingOperation",
    "NumericGeneralizationOperation",
    "PartialMaskingOperation",
    "RecordSuppressionOperation",
    "UniformNumericNoiseOperation",
    "UniformTemporalNoiseOperation",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "AttributeSuppressionOperation": "pamola_core.anonymization.suppression.attribute_op",
    "CategoricalGeneralizationOperation": "pamola_core.anonymization.generalization.categorical_op",
    "CellSuppressionOperation": "pamola_core.anonymization.suppression.cell_op",
    "DateTimeGeneralizationOperation": "pamola_core.anonymization.generalization.datetime_op",
    "FullMaskingOperation": "pamola_core.anonymization.masking.full_masking_op",
    "NumericGeneralizationOperation": "pamola_core.anonymization.generalization.numeric_op",
    "PartialMaskingOperation": "pamola_core.anonymization.masking.partial_masking_op",
    "RecordSuppressionOperation": "pamola_core.anonymization.suppression.record_op",
    "UniformNumericNoiseOperation": "pamola_core.anonymization.noise.uniform_numeric_op",
    "UniformTemporalNoiseOperation": "pamola_core.anonymization.noise.uniform_temporal_op",
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
