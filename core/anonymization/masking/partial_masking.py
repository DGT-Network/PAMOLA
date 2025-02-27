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

Module: Partial Masking Processor
----------------------------------
This module provides methods for partially masking sensitive attributes
to enhance anonymization while preserving part of the original data.
It extends the BaseMaskingProcessor and implements techniques such as
fixed-length masking, regular expression-based masking, and
context-based masking.

Partial masking is often used to:
- Preserve data utility while reducing re-identification risk.
- Hide parts of structured identifiers (e.g., Social Security Numbers,
  credit card numbers, phone numbers).
- Maintain readability while obscuring confidential information.

Common approaches include:
- **Fixed-Length Masking**: Hiding a predefined number of characters
  while keeping the format (`123-45-XXXX` for SSNs).
- **Regex-Based Masking**: Applying pattern-based masking (e.g.,
  hiding middle digits in phone numbers).
- **Context-Based Masking**: Retaining leading/trailing characters
  for usability (`****6789` for account numbers).

NOTE: This module requires `pandas` and `re` (regular expressions) as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import re
from abc import ABC
from core.anonymization.masking.base import BaseMaskingProcessor

class PartialMaskingProcessor(BaseMaskingProcessor, ABC):
    """
    Partial Masking Processor for anonymizing sensitive attributes.
    This class extends BaseMaskingProcessor and provides techniques
    for partially masking data while maintaining some usability.

    Methods:
    --------
    - mask_fixed_length(): Masks a fixed part of the value.
    - mask_with_regex(): Masks values based on a regular expression.
    - mask_keep_prefix_suffix(): Retains leading and/or trailing characters
      while hiding the rest.
    """

    def __init__(self, mask_char: str = "X"):
        """
        Initializes the partial masking processor.

        Parameters:
        -----------
        mask_char : str, optional
            The character to use for masking (default: "X").
        """
        self.mask_char = mask_char

    def mask(self, data: pd.DataFrame, columns: list, method: str = "fixed_length", **kwargs) -> pd.DataFrame:
        """
        Apply partial masking to specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be partially masked.
        method : str, optional
            The masking method ("fixed_length", "regex", "prefix_suffix").

        Returns:
        --------
        pd.DataFrame
            The dataset with partially masked values.
        """
        if method == "fixed_length":
            return self.mask_fixed_length(data, columns, **kwargs)
        elif method == "regex":
            return self.mask_with_regex(data, columns, **kwargs)
        elif method == "prefix_suffix":
            return self.mask_keep_prefix_suffix(data, columns, **kwargs)
        else:
            raise ValueError(f"Invalid masking method: {method}")

    def mask_fixed_length(self, data: pd.DataFrame, columns: list, mask_length: int = 4) -> pd.DataFrame:
        """
        Apply fixed-length masking, replacing part of the value with mask characters.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be masked.
        mask_length : int, optional
            Number of characters to mask from the end (default: 4).

        Returns:
        --------
        pd.DataFrame
            The dataset with fixed-length masked values.
        """
        data[columns] = data[columns].astype(str).applymap(lambda x: x[:-mask_length] + self.mask_char * mask_length if len(x) > mask_length else self.mask_char * len(x))
        return data

    def mask_with_regex(self, data: pd.DataFrame, columns: list, pattern: str, replacement: str = "XXX") -> pd.DataFrame:
        """
        Apply regex-based masking to values matching a pattern.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be masked.
        pattern : str
            Regular expression pattern to match sensitive parts.
        replacement : str, optional
            Replacement string for matched portions (default: "XXX").

        Returns:
        --------
        pd.DataFrame
            The dataset with regex-masked values.
        """
        data[columns] = data[columns].astype(str).applymap(lambda x: re.sub(pattern, replacement, x))
        return data

    def mask_keep_prefix_suffix(self, data: pd.DataFrame, columns: list, keep_prefix: int = 2, keep_suffix: int = 2) -> pd.DataFrame:
        """
        Mask a string while keeping a specified number of leading and trailing characters.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be masked.
        keep_prefix : int, optional
            Number of characters to keep at the beginning (default: 2).
        keep_suffix : int, optional
            Number of characters to keep at the end (default: 2).

        Returns:
        --------
        pd.DataFrame
            The dataset with partially masked values.
        """
        def mask_value(value):
            if len(value) > (keep_prefix + keep_suffix):
                return value[:keep_prefix] + self.mask_char * (len(value) - keep_prefix - keep_suffix) + value[-keep_suffix:]
            return self.mask_char * len(value)

        data[columns] = data[columns].astype(str).applymap(mask_value)
        return data
