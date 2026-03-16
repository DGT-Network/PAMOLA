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

Package: pamola_core.io
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

__all__ = [
    # csv/
    "read_csv",
    # json/
    "read_json",
    # excel/
    "read_excel",
    # parquet/
    "read_parquet",
]

from pamola_core.io.csv import read_csv

from pamola_core.io.json import read_json

from pamola_core.io.excel import read_excel

from pamola_core.io.parquet import read_parquet

