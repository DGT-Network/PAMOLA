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

Module: Suppression Base Processor
-----------------------------------
This module defines an abstract base class for suppression techniques
in PAMOLA.CORE. It extends the generic BaseProcessor and provides a structured
interface for suppression methods.

Suppression is a anonymization-preserving technique that removes or redacts sensitive
data entirely when anonymization methods are insufficient. It ensures compliance
with anonymization regulations by eliminating high-risk attributes.

Common methods include:
- **Cell suppression** (removal of specific cell values)
- **Record suppression** (removal of entire rows based on criteria)
- **Attribute suppression** (removal of high-risk columns)

NOTE: This is a template and will be updated as development progresses.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import ABC, abstractmethod
import pandas as pd
from pamola.pamola_core.base_processor import BaseProcessor

class BaseSuppressionProcessor(BaseProcessor, ABC):
    """
    Abstract base class for suppression processors in PAMOLA.CORE.
    This class extends BaseProcessor and defines methods specific to
    suppression techniques.

    Suppression removes sensitive data that cannot be anonymized
    effectively while preserving dataset structure.
    """

    @abstractmethod
    def suppress(self, data: pd.DataFrame, columns: list[str] = None, condition: callable = None, rule_based: dict = None, **kwargs) -> pd.DataFrame:
        """
        Apply suppression techniques to specified columns or rows in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset where suppression will be applied.
        columns : list[str], optional
            List of column names that should be suppressed (default: None).
        condition : callable, optional
            A function that determines which records to suppress (default: None).
        rule_based : dict, optional
            A dictionary defining custom suppression rules per column (default: None).
        kwargs : dict
            Additional parameters for suppression methods.

        Returns:
        --------
        pd.DataFrame
            The dataset with suppressed values.
        """
        pass
