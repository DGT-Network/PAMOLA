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

Module: Hashing Pseudonymization Processor
-------------------------------------------
This module provides an implementation of hashing-based pseudonymization
for anonymization-preserving data transformations. It extends the
BasePseudonymizationProcessor and allows controlled hashing of
identifiers using different cryptographic hash functions.

Hashing is a one-way transformation that replaces sensitive data
with irreversible values, preventing direct re-identification.

Common use cases include:
- Masking personally identifiable information (PII).
- Anonymizing structured identifiers while preserving uniqueness.
- Ensuring data integrity in anonymization-sensitive applications.

Supported hashing algorithms:
- **SHA-256**: Strong cryptographic hashing with high security.
- **SHA-512**: A more robust version of SHA-256.
- **bcrypt**: Adaptive hashing for increased security.

Additional features:
- **Salted hashing**: Introduces unique randomness per value.
- **Peppered hashing**: Adds system-wide randomness for added security.
- **Exporting hash mappings**: Stores original-hash mappings in a secure file.

NOTE: This module requires `pandas`, `hashlib`, `bcrypt`, and `os` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import hashlib
import bcrypt
import os
import json
from abc import ABC
from pamola.pamola_core.anonymization.pseudonymization.base import BasePseudonymizationProcessor

class HashingPseudonymizationProcessor(BasePseudonymizationProcessor, ABC):
    """
    Hashing Pseudonymization Processor for anonymizing structured identifiers.
    This class extends BasePseudonymizationProcessor and applies cryptographic
    hashing techniques for one-way pseudonymization.

    Methods:
    --------
    - hash_values(): Applies hashing to numerical or string-based identifiers.
    - generate_salt(): Creates a unique salt value per record.
    - apply_pepper(): Adds a system-wide pepper value for additional security.
    - export_hash_mappings(): Saves original-to-hash mappings in a secure file.
    """

    def __init__(self, hash_algorithm: str = "sha256", use_salt: bool = True, use_pepper: bool = False, pepper_value: str = None):
        """
        Initializes the hashing pseudonymization processor.

        Parameters:
        -----------
        hash_algorithm : str, optional
            The hashing algorithm to use ("sha256", "sha512", "bcrypt") (default: "sha256").
        use_salt : bool, optional
            Whether to add a unique salt for each record (default: True).
        use_pepper : bool, optional
            Whether to add a global pepper value to all hashes (default: False).
        pepper_value : str, optional
            A fixed pepper value (if not provided, a random one will be generated).
        """
        self.hash_algorithm = hash_algorithm
        self.use_salt = use_salt
        self.use_pepper = use_pepper
        self.pepper = pepper_value if pepper_value else os.urandom(16).hex()

    def pseudonymize(self, data: pd.DataFrame, columns: list[str], salt_export_path: str = None) -> pd.DataFrame:
        """
        Apply hashing pseudonymization to specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing data to be pseudonymized.
        columns : list[str]
            List of column names that should be hashed.
        salt_export_path : str, optional
            Path to export salt values for record re-identification (default: None).

        Returns:
        --------
        pd.DataFrame
            The dataset with hashed values.
        """
        salt_mappings = {}
        for column in columns:
            data[column], salts = self.hash_values(data[column])
            if self.use_salt:
                salt_mappings[column] = salts

        if salt_export_path and self.use_salt:
            self.export_hash_mappings(salt_export_path, salt_mappings)

        return data

    def hash_values(self, series: pd.Series) -> tuple[pd.Series, dict]:
        """
        Apply hashing with optional salt and pepper.

        Parameters:
        -----------
        series : pd.Series
            The data series to be hashed.

        Returns:
        --------
        tuple[pd.Series, dict]
            The hashed series and corresponding salt values (if applicable).
        """
        salts = {}
        def hash_function(value):
            if pd.isna(value):
                return value

            str_value = str(value)
            salt = self.generate_salt() if self.use_salt else ""
            if self.use_salt:
                salts[str_value] = salt

            input_value = str_value + salt + (self.pepper if self.use_pepper else "")

            if self.hash_algorithm == "sha256":
                return hashlib.sha256(input_value.encode()).hexdigest()
            elif self.hash_algorithm == "sha512":
                return hashlib.sha512(input_value.encode()).hexdigest()
            elif self.hash_algorithm == "bcrypt":
                return bcrypt.hashpw(input_value.encode(), bcrypt.gensalt()).decode()
            else:
                raise ValueError(f"Unsupported hashing algorithm: {self.hash_algorithm}")

        return series.apply(hash_function), salts

    def generate_salt(self) -> str:
        """
        Generate a unique salt value.

        Returns:
        --------
        str
            A random salt string.
        """
        return os.urandom(8).hex()

    def export_hash_mappings(self, file_path: str, mappings: dict):
        """
        Export original-to-hash mappings to a JSON file.

        Parameters:
        -----------
        file_path : str
            The file path to store the mapping data.

        Returns:
        --------
        None
        """
        with open(file_path, "w") as f:
            json.dump(mappings, f, indent=4)

