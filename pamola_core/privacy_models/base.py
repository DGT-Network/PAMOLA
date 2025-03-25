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

Module: Privacy Models Base Processor
--------------------------------------
This module defines an abstract base class for **anonymization models**
used in anonymization-preserving data processing. It provides a
structured interface for evaluating anonymization risks and applying
anonymization transformations.

Common anonymization models include:
- **k-Anonymity**: Guarantees that each individual is indistinguishable
  from at least `k-1` others in the dataset.
- **l-Diversity**: Ensures diversity of sensitive attributes within
  k-anonymous groups.
- **t-Closeness**: Maintains statistical similarity between groups
  and the original distribution.
- **k-Map**: A mapping-based anonymization model that aligns with real-world
  external datasets.

This base class enforces a **common structure** for all anonymization models,
ensuring consistency and flexibility in implementation.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import ABC, abstractmethod
import pandas as pd
from pamola.pamola_core.base_processor import BaseProcessor

class BasePrivacyModelProcessor(BaseProcessor, ABC):
    """
    Abstract base class for anonymization model processors in PAMOLA.CORE.
    This class enforces a standardized interface for anonymization evaluation
    and model application.

    Privacy models define the mathematical principles for ensuring
    data anonymization while preserving statistical utility.

    Methods:
    --------
    - evaluate_privacy(): Assesses anonymization risks and model compliance.
    - apply_model(): Applies the anonymization model to a dataset.
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
    def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: list[str], **kwargs) -> dict:
        """
        Evaluate anonymization risks and compliance of the dataset based on
        the specific anonymization model.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to be evaluated.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        kwargs : dict
            Additional parameters for model evaluation.

        Returns:
        --------
        dict
            A dictionary containing anonymization metrics and evaluation results.
        """
        pass

    @abstractmethod
    def apply_model(self, data: pd.DataFrame, quasi_identifiers: list[str], suppression: bool = True, **kwargs) -> pd.DataFrame:
        """
        Apply the anonymization model to transform the dataset while maintaining
        compliance with anonymization constraints.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset to be transformed.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        suppression : bool, optional
            Whether to suppress non-compliant records (default: True).
        kwargs : dict
            Additional parameters for model application.

        Returns:
        --------
        pd.DataFrame
            The transformed dataset with anonymization guarantees applied.
        """
        pass
