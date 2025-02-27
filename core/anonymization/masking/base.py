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

Module: Masking Base Processor
------------------------------
This module defines an abstract base class for masking techniques
in PAMOLA.CORE. It extends the generic BaseProcessor and provides
a structured interface for all masking methods.

Masking is a anonymization-enhancing technique that replaces sensitive
data with obfuscated values to prevent direct identification while
preserving dataset usability.

Common masking techniques include:
- **Full Masking**: Replacing the entire value with a placeholder
  (`"MASKED"`, `"REDACTED"`).
- **Partial Masking**: Redacting part of the value while keeping some
  information intact (e.g., `"123-45-XXXX"` for SSNs).
- **Format-Preserving Masking**: Retaining the structure of data while
  hiding its real content (`+X-XXX-XXX-XXXX` for phone numbers).

This class serves as a template for implementing specific masking strategies.

NOTE: This module requires `pandas` as a dependency.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import ABC, abstractmethod
import pandas as pd
from core.base_processor import BaseProcessor

class BaseMaskingProcessor(BaseProcessor, ABC):
    """
    Abstract base class for masking processors in PAMOLA.CORE.
    This class extends BaseProcessor and defines methods specific to
    data masking techniques.

    Masking is used to obscure sensitive data by replacing identifiable
    information with symbols, hashes, or other neutral values while
    maintaining the dataset structure.
    """

    @abstractmethod
    def mask(self, data: pd.DataFrame, columns: list, **kwargs) -> pd.DataFrame:
        """
        Apply masking techniques to specified columns in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset to be masked.
        columns : list
            List of column names that should be masked.
        kwargs : dict
            Additional parameters for masking, such as mask character,
            format preservation settings, or regex patterns.

        Returns:
        --------
        pd.DataFrame
            The dataset with masked values.
        """
        pass
