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

Module: Attribute Suppression Processor
----------------------------------------
This module provides an implementation of attribute (column) suppression
for anonymization-preserving data transformations. It extends the
BaseSuppressionProcessor and allows removing entire columns containing
sensitive information.

Attribute suppression is a fundamental anonymization technique used to
eliminate high-risk attributes when anonymization is insufficient.

Common use cases include:
- Removing direct identifiers (e.g., name, SSN, email).
- Eliminating attributes with high uniqueness to reduce re-identification risk.
- Enforcing compliance with data minimization principles (GDPR, HIPAA).

Features:
- **Direct attribute removal**: Deletes specified columns.
- **Conditional suppression**: Suppresses attributes based on sensitivity scores.
- **Configurable policies**: Allows defining rules for automatic suppression.

NOTE: This module requires `pandas` as a dependency.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
from abc import ABC
from core.anonymization.suppression.base import BaseSuppressionProcessor

class AttributeSuppressionProcessor(BaseSuppressionProcessor, ABC):
    """
    Attribute Suppression Processor for removing entire columns
    containing sensitive information.

    This class extends BaseSuppressionProcessor and provides
    flexible suppression methods for anonymization compliance.

    Methods:
    --------
    - suppress(): Removes specified attributes or those meeting suppression criteria.
    """

    def __init__(self, sensitive_attributes: list[str] = None, sensitivity_threshold: float = None):
        """
        Initializes the attribute suppression processor.

        Parameters:
        -----------
        sensitive_attributes : list[str], optional
            List of column names that should always be removed (default: None).
        sensitivity_threshold : float, optional
            A threshold score for automatic suppression (default: None).
        """
        self.sensitive_attributes = sensitive_attributes if sensitive_attributes else []
        self.sensitivity_threshold = sensitivity_threshold

    def suppress(self, data: pd.DataFrame, columns: list[str] = None, sensitivity_scores: dict = None, **kwargs) -> pd.DataFrame:
        """
        Apply attribute suppression to specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing sensitive attributes.
        columns : list[str], optional
            List of column names that should be suppressed (default: None).
        sensitivity_scores : dict, optional
            A dictionary mapping column names to sensitivity scores
            for automatic suppression (default: None).

        Returns:
        --------
        pd.DataFrame
            The dataset with specified attributes removed.
        """
        columns_to_remove = set(self.sensitive_attributes)  # Always suppress these

        if columns:
            columns_to_remove.update(columns)

        if sensitivity_scores and self.sensitivity_threshold:
            for col, score in sensitivity_scores.items():
                if score >= self.sensitivity_threshold:
                    columns_to_remove.add(col)

        # Convert the set to a list before calling drop()
        return data.drop(columns=list(columns_to_remove), errors="ignore")
