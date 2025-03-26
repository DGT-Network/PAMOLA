"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
---------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for privacy-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Data Profiling Base Processor
-------------------------------------
This module defines an abstract base class for data profiling techniques
in PAMOLA.CORE. It extends the generic BaseProcessor and provides
a structured interface for analyzing datasets to identify unique,
quasi-unique, and sensitive columns.

Data profiling is a fundamental step in data anonymization and privacy
preservation. It involves examining datasets to detect:
- **Unique columns**: Columns where each value is distinct.
- **Quasi-unique columns**: Columns with a high proportion of unique values.
- **Sensitive columns**: Columns containing sensitive information based on
  predefined keywords.

NOTE: This is a template and will be updated as development progresses.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import ABC, abstractmethod
from typing import Tuple, List
import pandas as pd
from core.base_processor import BaseProcessor

class BaseDataProfilingProcessor(BaseProcessor, ABC):
    """
    Abstract base class for data profiling processors in PAMOLA.CORE.
    This class extends BaseProcessor and defines methods specific to
    analyzing datasets to identify unique, quasi-unique, and sensitive
    columns.

    Data profiling is essential for understanding the structure and
    sensitivity of data, aiding in effective anonymization and privacy
    preservation strategies.
    """
    
    @abstractmethod
    def process(self, data):
        """
        Process the input data.

        Parameters:
        -----------
        data : Any
            The input data to be processed.

        Returns:
        --------
        Processed data, transformed according to the specific processor logic.
        """
        pass

    @abstractmethod
    def data_profiling(
        self,
        df: pd.DataFrame,
        unique_threshold: float = 0.9,
        sensitive_keywords: List[str] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Analyze the dataset to identify unique, quasi-unique, and sensitive
        columns.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataset to be analyzed.
        unique_threshold : float, optional
            Threshold proportion to consider a column as quasi-unique
            (default: 0.9).
        sensitive_keywords : list of str, optional
            List of keywords to identify sensitive columns (default: None).

        Returns:
        --------
        Tuple[List[str], List[str], List[str]]
            A tuple containing three lists:
            - direct identifiers.
            - quasi-identifiers.
            - sensitive attributes.
        """
        pass
