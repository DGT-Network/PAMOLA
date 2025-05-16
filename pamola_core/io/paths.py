"""
    PAMOLA Core - IO Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all IO operations in the PAMOLA Core
    library. It defines the standard contracts that all format-specific handlers must implement.

    Key features:
    - PathManager implementation for standardized path resolution

    This module serves as the foundation for all IO operations in PAMOLA Core, ensuring
    consistency across different data formats and storage methods.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:
        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

from pathlib import Path

class PathManager:
    def __init__(self, root_dir = None):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()

    def __getitem__(self, key):
        return getattr(self, key) if hasattr(self, key) else None

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, Path(value))

    def get_raw_path(self, *args):
        return Path(self.root_dir, *args)
