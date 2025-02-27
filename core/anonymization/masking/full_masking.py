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

Module: Full Masking Processor
------------------------------
This module provides methods for fully masking sensitive attributes
to enhance anonymization. It extends the BaseMaskingProcessor and implements
techniques such as replacing values with standard placeholders,
randomized masks, and format-preserving masking.

Full masking completely removes the original information while
preserving the data structure.

Common approaches include:
- **Fixed Masking**: Replacing values with `"MASKED"` or `"REDACTED"`.
- **Symbol Masking**: Replacing values with predefined symbols (`XXXXXX`, `******`).
- **Format-Preserving Masking**: Retaining the original format while hiding content
  (e.g., `+1-XXX-XXX-XXXX` for phone numbers).

NOTE: This module requires `pandas` and `re` (regular expressions) as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import re
from abc import ABC
from core.anonymization.masking.base import BaseMaskingProcessor

class FullMaskingProcessor(BaseMaskingProcessor, ABC):
    """
    Full Masking Processor for anonymizing sensitive attributes.
    This class extends BaseMaskingProcessor and provides techniques
    for fully masking data while preserving format when required.

    Methods:
    --------
    - apply_fixed_mask(): Replaces values with a predefined mask (`"MASKED"`).
    - apply_symbol_mask(): Replaces values with symbols (`"XXXXXX"`, `"*****"`).
    - apply_format_preserving_mask(): Preserves the format while hiding content.
    """

    def __init__(self, mask_char: str = "X", fixed_mask: str = "MASKED"):
        """
        Initializes the full masking processor.

        Parameters:
        -----------
        mask_char : str, optional
            The character to use for symbolic masking (default: "X").
        fixed_mask : str, optional
            The fixed replacement string for full masking (default: "MASKED").
        """
        self.mask_char = mask_char
        self.fixed_mask = fixed_mask

    def mask(self, data: pd.DataFrame, columns: list, method: str = "fixed") -> pd.DataFrame:
        """
        Apply full masking to specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be fully masked.
        method : str, optional
            The masking method ("fixed", "symbolic", "format_preserving").

        Returns:
        --------
        pd.DataFrame
            The dataset with masked values.
        """
        if method == "fixed":
            return self.apply_fixed_mask(data, columns)
        elif method == "symbolic":
            return self.apply_symbol_mask(data, columns)
        elif method == "format_preserving":
            return self.apply_format_preserving_mask(data, columns)
        else:
            raise ValueError(f"Invalid masking method: {method}")

    def apply_fixed_mask(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Replace all values in specified columns with a fixed mask.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be fully masked.

        Returns:
        --------
        pd.DataFrame
            The dataset with fully masked values.
        """
        data[columns] = self.fixed_mask
        return data

    def apply_symbol_mask(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Replace all values in specified columns with a symbolic mask.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be fully masked.

        Returns:
        --------
        pd.DataFrame
            The dataset with symbolically masked values.
        """
        data[columns] = data[columns].astype(str).applymap(lambda x: self.mask_char * len(x))
        return data

    def apply_format_preserving_mask(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Preserve original format while applying masking.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive data.
        columns : list
            List of column names to be masked.

        Returns:
        --------
        pd.DataFrame
            The dataset with format-preserving masked values.
        """
        def format_preserving_mask(value):
            if isinstance(value, str):
                if re.match(r"\+\d{1,3}-\d{3}-\d{3}-\d{4}", value):  # Phone number format
                    return "+X-XXX-XXX-XXXX"
                elif re.match(r"\d{16}", value):  # Credit card format
                    return "XXXX-XXXX-XXXX-XXXX"
                elif re.match(r"\d{3}-\d{2}-\d{4}", value):  # Social Security Number
                    return "XXX-XX-XXXX"
                else:
                    return self.mask_char * len(value)
            return self.fixed_mask

        data[columns] = data[columns].astype(str).applymap(format_preserving_mask)
        return data
