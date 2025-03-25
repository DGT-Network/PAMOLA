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

Module: Tokenization Pseudonymization Processor
------------------------------------------------
This module provides an implementation of tokenization-based pseudonymization
for anonymization-preserving data transformations. It extends the
BasePseudonymizationProcessor and allows mapping sensitive data to
unique, non-reversible tokens.

Tokenization is a process of replacing sensitive data with unique tokens
that can be mapped back to their original values in a secure manner.

Common use cases include:
- Protecting personally identifiable information (PII).
- Reducing data exposure in compliance with GDPR/CCPA.
- Maintaining data utility while ensuring anonymization.

Features:
- **Unique token generation**: Assigns unique tokens per value.
- **Mapping storage**: Stores mappings in a secure JSON file.
- **Reversible transformation**: Enables authorized re-identification.

NOTE: This module requires `pandas`, `uuid`, `json`, and `os` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import uuid
import json
import os
from abc import ABC
from pamola.pamola_core.anonymization.pseudonymization.base import BasePseudonymizationProcessor

class TokenizationPseudonymizationProcessor(BasePseudonymizationProcessor, ABC):
    """
    Tokenization Pseudonymization Processor for anonymizing structured identifiers.
    This class extends BasePseudonymizationProcessor and applies tokenization
    techniques for reversible pseudonymization.

    Methods:
    --------
    - generate_tokens(): Creates a tokenized mapping for sensitive values.
    - tokenize_values(): Replaces values with unique tokens.
    - detokenize_values(): Restores original values from stored mappings.
    - export_token_mappings(): Saves token mappings to a secure file.
    - load_token_mappings(): Loads token mappings from a file.
    """

    def __init__(self, mapping_file: str = "token_mappings.json"):
        """
        Initializes the tokenization pseudonymization processor.

        Parameters:
        -----------
        mapping_file : str, optional
            The file path where token mappings will be stored (default: "token_mappings.json").
        """
        self.mapping_file = mapping_file
        self.token_map = self.load_token_mappings()

    def pseudonymize(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply tokenization to specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing data to be tokenized.
        columns : list[str]
            List of column names that should be tokenized.

        Returns:
        --------
        pd.DataFrame
            The dataset with tokenized values.
        """
        for column in columns:
            data[column] = self.tokenize_values(data[column])

        self.export_token_mappings()
        return data

    def tokenize_values(self, series: pd.Series) -> pd.Series:
        """
        Replace values with unique tokens.

        Parameters:
        -----------
        series : pd.Series
            The data series to be tokenized.

        Returns:
        --------
        pd.Series
            The tokenized series.
        """
        def get_or_generate_token(value):
            if pd.isna(value):
                return value

            str_value = str(value)
            if str_value not in self.token_map:
                self.token_map[str_value] = str(uuid.uuid4())  # Generate unique UUID token
            return self.token_map[str_value]

        return series.apply(get_or_generate_token)

    def detokenize_values(self, series: pd.Series) -> pd.Series:
        """
        Restore original values from stored token mappings.

        Parameters:
        -----------
        series : pd.Series
            The data series containing tokens.

        Returns:
        --------
        pd.Series
            The detokenized series.
        """
        reverse_map = {v: k for k, v in self.token_map.items()}

        def get_original_value(token):
            return reverse_map.get(token, token)  # Return token if not found

        return series.apply(get_original_value)

    def export_token_mappings(self):
        """
        Save token mappings to a JSON file.

        Returns:
        --------
        None
        """
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.token_map, f, indent=4)

    def load_token_mappings(self) -> dict:
        """
        Load token mappings from a JSON file.

        Returns:
        --------
        dict
            A dictionary containing the original-to-token mappings.
        """
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
