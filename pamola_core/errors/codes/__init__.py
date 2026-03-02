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

Package: pamola_core.errors.codes
Type: Internal (Non-Public API)

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    # registry.py
    "ErrorCode",
    # metadata.py
    "ERROR_CODE_METADATA",
    "get_error_metadata",
    # utils.py
    "validate_error_code_usage",
    "get_error_info",
    "format_error_help",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "ErrorCode": "pamola_core.errors.codes.registry",
    "ERROR_CODE_METADATA": "pamola_core.errors.codes.metadata",
    "get_error_metadata": "pamola_core.errors.codes.metadata",
    "validate_error_code_usage": "pamola_core.errors.codes.utils",
    "get_error_info": "pamola_core.errors.codes.utils",
    "format_error_help": "pamola_core.errors.codes.utils",
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
