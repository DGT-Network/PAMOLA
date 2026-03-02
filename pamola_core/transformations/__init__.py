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

Package: pamola_core.transformations
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    "AddOrModifyFieldsOperation",
    "AggregateRecordsOperation",
    "CleanInvalidValuesOperation",
    "ImputeMissingValuesOperation",
    "MergeDatasetsOperation",
    "RemoveFieldsOperation",
    "SplitByIDValuesOperation",
    "SplitFieldsOperation",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "AddOrModifyFieldsOperation": "pamola_core.transformations.field_ops.add_modify_fields",
    "AggregateRecordsOperation": "pamola_core.transformations.grouping.aggregate_records_op",
    "CleanInvalidValuesOperation": "pamola_core.transformations.cleaning.clean_invalid_values",
    "ImputeMissingValuesOperation": "pamola_core.transformations.imputation.impute_missing_values",
    "MergeDatasetsOperation": "pamola_core.transformations.merging.merge_datasets_op",
    "RemoveFieldsOperation": "pamola_core.transformations.field_ops.remove_fields",
    "SplitByIDValuesOperation": "pamola_core.transformations.splitting.split_by_id_values_op",
    "SplitFieldsOperation": "pamola_core.transformations.splitting.split_fields_op",
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
