"""
    PAMOLA Core - IO Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all IO operations in the PAMOLA Core
    library. It defines the standard contracts that all format-specific handlers must implement.

    Key features:
    - DataReader protocol for standardized data reading
    - DataWriter protocol for standardized data writing
    - Common parameter definitions and defaults
    - Error type definitions for IO operations
    - Configuration integration points

    This module serves as the foundation for all IO operations in PAMOLA Core, ensuring
    consistency across different data formats and storage methods.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:
        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

from .paths import PathManager
from .csv import read_csv, write_csv
from .excel import read_excel, write_excel
from .json import read_json, write_json
from .parquet import read_parquet, write_parquet
