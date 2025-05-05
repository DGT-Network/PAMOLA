"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
---------------------------------------------------
This module is part of the PAMOLA ecosystem, an open-source suite
for privacy-preserving data processing and anonymization.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

Licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Base Processor for Data Profiling & Analysis
----------------------------------------------------
This module defines an abstract base class for various profiling
techniques in PAMOLA.CORE. It provides a standardized framework
for analyzing datasets and supports multiple profiling categories,
including:

- **Data Field Profiling**: Alyzing data fields.

This base processor ensures a modular approach, allowing different
profiling techniques to inherit and implement specific analysis logic.

⚠ NOTE: This is a foundational component and may evolve as
development progresses.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseDataFieldProfilingProcessor(ABC):
    """
    Abstract base class for dataset profiling processors in PAMOLA.CORE.
    
    This class serves as a foundation for implementing profiling mechanisms 
    that analyze datasets across multiple dimensions, helping to assess 
    structure, consistency, and sensitivity. Profiling plays a crucial role 
    in enabling effective anonymization and privacy-preserving strategies.
    """

    @abstractmethod
    def execute(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Performs dataset profiling and returns relevant insights.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input dataset to be analyzed.
        **kwargs : dict
            Additional parameters for customization.

        Returns:
        --------
        Any
            The profiling results, which may include:
            - A tuple of lists: (direct identifiers, quasi-identifiers, sensitive attributes, indirect identifiers, and non-sensitive attributes)
            - A dictionary containing detailed profiling metrics
            - A Pandas DataFrame with structured profiling data
            - A single list or value if the result is minimal
        """
        pass