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

Module: Replacement Pseudonymization Processor
----------------------------------------------
This module provides an implementation of replacement-based pseudonymization
for anonymization-preserving data transformations. It extends the
BasePseudonymizationProcessor and allows mapping sensitive data to
predefined pseudonyms based on rules or lookup tables.

Replacement pseudonymization substitutes identifiable values with
alternative values while maintaining data consistency.

Common use cases include:
- Substituting real names with synthetic names.
- Replacing addresses or contact details with fictitious equivalents.
- Ensuring consistency in pseudonyms across datasets.

Features:
- **Rule-based replacement**: Defines substitution rules.
- **Lookup-table replacement**: Uses preloaded mappings from a file.
- **Configurable consistency**: Ensures stable pseudonyms per record.

NOTE: This module requires `pandas`, `json`, and `os` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import json
import os
from abc import ABC
from pamola_core.anonymization__old_15_04.pseudonymization.base import BasePseudonymizationProcessor

class ReplacementPseudonymizationProcessor(BasePseudonymizationProcessor, ABC):
    """
    Replacement Pseudonymization Processor for structured identifiers.
    This class extends BasePseudonymizationProcessor and applies rule-based
    or lookup-table-based replacement techniques.

    Methods:
    --------
    - replace_values(): Substitutes values based on predefined rules.
    - load_replacement_table(): Loads mapping rules from a JSON file.
    - export_replacement_table(): Saves replacement mappings for consistency.
    """

    def __init__(self, mapping_file: str = "replacement_mappings.json"):
        """
        Initializes the replacement pseudonymization processor.

        Parameters:
        -----------
        mapping_file : str, optional
            The file path where replacement mappings will be stored (default: "replacement_mappings.json").
        """
        self.mapping_file = mapping_file
        self.replacement_map = self.load_replacement_table()

    def pseudonymize(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply replacement pseudonymization to specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing data to be pseudonymized.
        columns : list[str]
            List of column names that should be replaced.

        Returns:
        --------
        pd.DataFrame
            The dataset with replaced values.
        """
        for column in columns:
            data[column] = self.replace_values(data[column])

        self.export_replacement_table()
        return data

    def replace_values(self, series: pd.Series) -> pd.Series:
        """
        Replace values with predefined pseudonyms.

        Parameters:
        -----------
        series : pd.Series
            The data series to be replaced.

        Returns:
        --------
        pd.Series
            The series with pseudonymized values.
        """
        def get_replacement(value):
            if pd.isna(value):
                return value

            str_value = str(value)
            if str_value not in self.replacement_map:
                self.replacement_map[str_value] = f"User{len(self.replacement_map) + 1000}"  # Example pseudonymization
            return self.replacement_map[str_value]

        return series.apply(get_replacement)

    def load_replacement_table(self) -> dict:
        """
        Load replacement mappings from a JSON file.

        Returns:
        --------
        dict
            A dictionary containing original-to-replacement mappings.
        """
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def export_replacement_table(self):
        """
        Save replacement mappings to a JSON file.

        Returns:
        --------
        None
        """
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.replacement_map, f, indent=4)
