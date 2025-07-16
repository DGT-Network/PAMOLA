"""
PAMOLA.CORE - Type aliases Module
---------------------------------------------------------
This module defines global type aliases used throughout the project to improve
maintainability and reduce hardcoded values in the codebase.

Features:
 - Centralized global type aliases to ensure consistency across modules.
 - Prevent hardcoded strings and facilitate easy updates.

This module is useful for logging, data transformations, and privacy-preserving
operations where global type aliases are required.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from pathlib import Path
from typing import Dict, List, Union, Optional

import dask.dataframe as dd
import pandas as pd
import numpy as np

# Define type variables for better type hints
PathLike = Union[str, Path]
PathLikeOrList = Union[PathLike, List[PathLike]]
DataFrameType = Union[pd.DataFrame, dd.DataFrame]


class CryptoConfig:
    """Class-based configuration for crypto."""

    # Class attributes
    mode: Optional[str] = None
    algorithm: Optional[str] = None
    key: Optional[str] = None
    key_path: Optional[PathLike] = None

    def __init__(
            self,
            mode: Optional[str] = None,
            algorithm: Optional[str] = None,
            key: Optional[str] = None,
            key_path: Optional[PathLike] = None
    ):
        """Initialize the configuration."""
        self.mode = mode
        self.algorithm = algorithm
        self.key = key
        self.key_path = key_path

        # Validate the configuration
        self._validate()

    def __repr__(
            self
    ):
        return f"CryptoConfig(mode={self.mode}, algorithm={self.algorithm}, key={self.key}, key_path={self.key_path})"

    def to_dict(
            self
    ) -> Dict:
        """Convert the instance to a dictionary."""
        return {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "key": self.key,
            "key_path": str(self.key_path) if self.key_path else None
        }

    @classmethod
    def from_dict(
            cls,
            data: Optional[dict] = None
    ) -> 'CryptoConfig':
        """Create an instance from a dictionary."""
        if data is None:
            data = {}

        return cls(
            mode=data.get("mode", cls.mode),
            algorithm=data.get("algorithm", cls.algorithm),
            key=data.get("key", cls.key),
            key_path=data.get("key_path", cls.key_path)
        )

    def _validate(
            self
    ):
        """Validate the configuration."""
        allowed_modes = []
        allowed_algorithms = []

        if allowed_modes and self.mode and self.mode not in allowed_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Allowed modes are {allowed_modes}")

        if allowed_algorithms and self.algorithm and self.algorithm not in allowed_algorithms:
            raise ValueError(f"Invalid algorithm: {self.algorithm}. Allowed algorithms are {allowed_algorithms}")


class FileCryptoConfig:
    """Class for file configuration with associated crypto settings."""

    # Class attributes
    file_paths: Optional[PathLikeOrList] = None
    crypto_config: Optional[CryptoConfig] = None

    def __init__(
            self,
            file_paths: Optional[PathLikeOrList] = None,
            crypto_config: Optional[CryptoConfig] = None
    ):
        """Initialize the configuration."""
        self.file_paths = file_paths
        self.crypto_config = crypto_config

    def __repr__(
            self
    ):
        return f"FileCryptoConfig(file_paths={self.file_paths}, crypto_config={self.crypto_config})"

    def to_dict(
            self
    ) -> Dict:
        """Convert the instance to a dictionary."""
        file_paths = None
        if self.file_paths and isinstance(self.file_paths, list):
            file_paths = [str(file_path) if file_path else None for file_path in self.file_paths]
        else:
            file_paths = str(self.file_paths) if self.file_paths else None

        return {
            "file_paths": file_paths,
            "crypto_config": self.crypto_config.to_dict()  # Convert the nested CryptoConfig to a dictionary
        }

    @classmethod
    def from_dict(
            cls,
            data: Optional[dict] = None
    ) -> 'FileCryptoConfig':
        """Create an instance from a dictionary."""
        if data is None:
            data = {}

        # Convert the nested dictionary back to CryptoConfig
        crypto_config = CryptoConfig.from_dict(data=data.get("crypto_config", cls.crypto_config))

        return cls(
            file_paths=data.get("file_paths", cls.file_paths),
            crypto_config=crypto_config
        )


def convert_to_flatten_dict(
        data: dict,
        prefix: str = "",
        separator: str = "_",
        append_key: bool = True
) -> dict:
    """Function to flatten a plain dictionary"""
    flat_dict = {}

    for key, value in data.items():
        # Generate the new key by appending the prefix
        new_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # If the value is a dictionary, flatten it further with append separator to the key
            flat_dict.update(
                convert_to_flatten_dict(
                    data=value,
                    prefix=f"{new_key}{separator}" if append_key else "",
                    separator=separator,
                    append_key=append_key
                )
            )
        else:
            # Otherwise, just add the value to the flat dictionary
            flat_dict[new_key] = value

    return flat_dict
