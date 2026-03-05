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

Package: pamola_core.errors
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

__all__ = [
    # base.py
    "BasePamolaError",
    "auto_exception",
    "_format_field_list",
    # tasks.py
    "TaskInitializationError",
]

from pamola_core.errors.base import BasePamolaError
from pamola_core.errors.base import auto_exception
from pamola_core.errors.base import _format_field_list

from pamola_core.errors.exceptions.tasks import TaskInitializationError

