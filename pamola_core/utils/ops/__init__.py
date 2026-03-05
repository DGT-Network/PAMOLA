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

Package: pamola_core.utils.ops
Type: Internal (Non-Public API)

Author: Realm Inveo Inc. & DGT Network Inc.
"""

__all__ = [
    # op_base.py
    "BaseOperation",
    # op_data_source.py
    "DataSource",
    # op_result.py
    "OperationStatus",
]

from pamola_core.utils.ops.op_base import BaseOperation

from pamola_core.utils.ops.op_data_source import DataSource

from pamola_core.utils.ops.op_result import OperationStatus

