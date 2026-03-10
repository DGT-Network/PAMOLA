"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
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

from pamola_core.errors.codes.registry import ErrorCode

from pamola_core.errors.codes.metadata import ERROR_CODE_METADATA
from pamola_core.errors.codes.metadata import get_error_metadata

from pamola_core.errors.codes.utils import validate_error_code_usage
from pamola_core.errors.codes.utils import get_error_info
from pamola_core.errors.codes.utils import format_error_help

