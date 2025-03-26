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

Module: Data Profiling Processor
--------------------------------
This module provides an implementation of data profiling techniques
for privacy-preserving data transformations. It extends the
BaseDataProfilingProcessor and enables the analysis of datasets to
identify direct identifiers, quasi-identifiers, and sensitive attributes
based on predefined rules.

Data profiling is essential in understanding the structure and
sensitivity of data, aiding in effective anonymization and privacy
preservation strategies.

Common use cases include:
- Identifying columns with unique or quasi-unique values that may
  pose re-identification risks.
- Detecting columns containing sensitive information based on
  specific keywords.
- Informing subsequent data anonymization processes by providing
  insights into data characteristics.

Features:
- **Unique and quasi-unique detection**: Identifies columns with
  unique or high proportion of unique values.
- **Sensitive data detection**: Recognizes columns containing
  sensitive information based on predefined keywords.
- **Customizable thresholds**: Allows setting thresholds for
  uniqueness and specifying sensitive keywords.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import re
from typing import List, Tuple
import numpy as np
import pandas as pd
from abc import ABC

from core.data_profiling.base import BaseDataProfilingProcessor


class DetectProfilingProcessor(BaseDataProfilingProcessor, ABC):
    """
    Processor for detecting direct identifiers, quasi-identifiers, and sensitive attributes in datasets.

    This class extends BaseDataProfilingProcessor and provides methods to analyze datasets
    for attributes that may pose privacy risks, facilitating anonymization processes.

    Attributes:
        unique_threshold (float): Threshold for cardinality to classify columns as direct identifiers.
        sensitive_keywords (List[str]): Keywords used to identify sensitive attributes based on column names.
    """

    def __init__(self, unique_threshold: float = 0.9, sensitive_keywords: List[str] = None):
        """
        Initializes the DetectProfilingProcessor with specified parameters.

        Args:
            unique_threshold (float, optional): Threshold for cardinality to classify columns as direct identifiers. Defaults to 0.9.
            sensitive_keywords (List[str], optional): Keywords used to identify sensitive attributes based on column names. Defaults to a predefined set of common sensitive terms.
        """
        self.unique_threshold = unique_threshold
        self.sensitive_keywords = sensitive_keywords or [
            "disease", "income", "salary", "condition", "health", "ssn", "medical",
            "insurance", "credit", "bank", "tax", "password", "social security",
            "ethnicity", "religion", "gender", "disability"
        ]
        
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

    def data_profiling(
        self,
        df: pd.DataFrame,
        unique_threshold: float = None,
        sensitive_keywords: List[str] = None,
        sample_size: int = 100,
        **kwargs
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Detects direct identifiers, quasi-identifiers, and sensitive attributes in a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataset to be analyzed.
        unique_threshold : float, optional
            Threshold proportion to consider a column as a direct identifier
            (default: 0.9).
        sensitive_keywords : list of str, optional
            List of keywords to identify sensitive columns (default: None).
        sample_size: int, optional
            Number of samples to consider for pattern matching in object-type columns. Defaults to 100.
        kwargs : dict
            Additional parameters for profiling logic.

        Returns:
        --------
        Tuple[List[str], List[str], List[str]]
            A tuple containing three lists:
            - direct_identifiers.
            - quasi_identifiers.
            - sensitive_attributes.
        """
        unique_threshold = unique_threshold or self.unique_threshold
        sensitive_keywords = sensitive_keywords or self.sensitive_keywords

        categorized_columns = {
            "direct_identifiers": [],
            "quasi_identifiers": [],
            "sensitive_attributes": [],
        }

        # Compile regex patterns for common direct identifiers
        patterns = {
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "phone": re.compile(r"^\+?(\d[\d\-. ]+)?(\([\d\-. ]+\))?[\d\-. ]+\d$"),
            "credit_card": re.compile(r"^\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}$"),
        }

        row_count = len(df)

        for col in df.columns:
            is_sensitive = any(keyword in col.lower() for keyword in sensitive_keywords)
            unique_values = df[col].nunique(dropna=True)
            cardinality_ratio = unique_values / row_count

            # Check for direct identifiers based on cardinality
            if cardinality_ratio > unique_threshold and not is_sensitive:
                categorized_columns["direct_identifiers"].append(col)
                continue

            # Check for sensitive attributes based on column name
            if is_sensitive:
                categorized_columns["sensitive_attributes"].append(col)
                continue

            # Analyze object-type columns for direct identifier patterns
            if pd.api.types.is_object_dtype(df[col].dtype):
                non_null_values = df[col].dropna().astype(str)
                sample_data = non_null_values.sample(min(sample_size, len(non_null_values)), random_state=42)

                # Check for patterns indicating direct identifiers
                if any(sample_data.str.match(pattern).any() for pattern in patterns.values()):
                    categorized_columns["direct_identifiers"].append(col)
                else:
                    categorized_columns["quasi_identifiers"].append(col)
            # Analyze numerical columns for quasi-identifiers based on cardinality
            elif np.issubdtype(df[col].dtype, np.number) and 0.1 < cardinality_ratio < unique_threshold:
                categorized_columns["quasi_identifiers"].append(col)

        return (
            categorized_columns["direct_identifiers"],
            categorized_columns["quasi_identifiers"],
            categorized_columns["sensitive_attributes"],
        )