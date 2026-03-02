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

Package: pamola_core.utils
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import importlib
from typing import Dict

__all__ = [
    # io_helpers/
    "decrypt_file",
    "decrypt_data",
    "encrypt_file",
    "safe_remove_temp_file",
    "detect_encryption_mode",
    # crypto_helpers/
    "get_key_for_task",
    # tasks/
    "TaskConfig",
    "BaseTask",
    "TaskInitializationError",
    # io/
    "load_data_operation",
    "load_settings_operation",
    "optimize_dataframe_memory",
    # ops/
    "BaseOperation",
    "DataSource",
    "OperationStatus",
]

_LAZY_IMPORTS: Dict[str, str] = {
    # io_helpers/
    "decrypt_file": "pamola_core.utils.io_helpers",
    "decrypt_data": "pamola_core.utils.io_helpers",
    "encrypt_file": "pamola_core.utils.io_helpers",
    "safe_remove_temp_file": "pamola_core.utils.io_helpers",
    "detect_encryption_mode": "pamola_core.utils.io_helpers",
    # crypto_helpers/
    "get_key_for_task": "pamola_core.utils.crypto_helpers",
    # tasks/
    "TaskConfig": "pamola_core.utils.tasks",
    "BaseTask": "pamola_core.utils.tasks",
    "TaskInitializationError": "pamola_core.utils.tasks",
    # io/
    "load_data_operation": "pamola_core.utils.io",
    "load_settings_operation": "pamola_core.utils.io",
    "optimize_dataframe_memory": "pamola_core.utils.io",
    # ops/
    "BaseOperation": "pamola_core.utils.ops",
    "DataSource": "pamola_core.utils.ops",
    "OperationStatus": "pamola_core.utils.ops",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
