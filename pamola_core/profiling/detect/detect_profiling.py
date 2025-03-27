"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
---------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for privacy-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

Licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Data Profiling Processor
--------------------------------
Detects direct identifiers, quasi-identifiers, sensitive attributes,
indirect identifiers, and non-sensitive attributes to support privacy-preserving data transformations.

NOTE: Requires `pandas` and `numpy`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import re
from typing import List, Tuple
import numpy as np
import pandas as pd
from abc import ABC

from pamola_core.profiling.base import BaseProfilingProcessor


class DetectProfilingProcessor(BaseProfilingProcessor, ABC):
    """
    Processor for detecting direct identifiers, quasi-identifiers, sensitive attributes,
    indirect identifiers, and non-sensitive attributes in datasets.

    Attributes:
        unique_threshold (float): Threshold to classify columns as direct identifiers.
        sensitive_keywords (List[str]): Keywords used to identify sensitive attributes.
        indirect_keywords (List[str]): Keywords used to identify indirect identifiers.
    """

    def __init__(
        self,
        unique_threshold: float = 0.9,
        sample_size: int = 100,
        sensitive_keywords: List[str] = None,
        indirect_keywords: List[str] = None,
    ):
        """
        Initializes DetectProfilingProcessor.

        Args:
            unique_threshold (float): Threshold for direct identifier detection. Default is 0.9.
            sample_size (int, optional): Number of samples for pattern matching. Default is 100.
            sensitive_keywords (List[str], optional): Keywords to identify sensitive attributes.
            indirect_keywords (List[str], optional): Keywords to identify indirect identifiers.
        """
        super().__init__() 
        self.unique_threshold = unique_threshold
        self.sample_size = sample_size
        self.sensitive_keywords = sensitive_keywords or [
            "disease",
            "income",
            "salary",
            "condition",
            "health",
            "ssn",
            "medical",
            "insurance",
            "credit",
            "bank",
            "tax",
            "password",
            "social security",
            "ethnicity",
            "religion",
            "gender",
            "disability",
        ]
        self.indirect_keywords = indirect_keywords or [
            "job",
            "role",
            "title",
            "department",
            "organization",
            "employer",
            "company",
        ]

    def execute(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Detects direct identifiers, quasi-identifiers, sensitive attributes, indirect identifiers, and non-sensitive attributes.

        Parameters:
            df (pd.DataFrame): The dataset to analyze.

        Kwargs:
            unique_threshold (float, optional): Threshold for identifying direct identifiers. Defaults to self.unique_threshold.
            sensitive_keywords (List[str], optional): Keywords to detect sensitive attributes. Defaults to self.sensitive_keywords.
            indirect_keywords (List[str], optional): Keywords to detect indirect identifiers. Defaults to self.indirect_keywords.
            sample_size (int, optional): Number of samples for pattern matching. Defaults to self.sample_size.

        Returns:
            Tuple[List[str], List[str], List[str], List[str], List[str]]: Lists of
            direct identifiers, quasi-identifiers, sensitive attributes, indirect identifiers, and non-sensitive attributes.
        """
        unique_threshold = kwargs.get("unique_threshold", self.unique_threshold)
        sensitive_keywords = kwargs.get("sensitive_keywords", self.sensitive_keywords)
        indirect_keywords = kwargs.get("indirect_keywords", self.indirect_keywords)
        sample_size = kwargs.get("sample_size", self.sample_size)

        direct_identifiers, quasi_identifiers, sensitive_attributes = [], [], []
        indirect_identifiers, non_sensitive_attributes = [], []

        # Regex patterns for common identifiers
        patterns = {
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "phone": re.compile(r"^\+?(\d[\d\-. ]+)?(\([\d\-. ]+\))?[\d\-. ]+\d$"),
            "credit_card": re.compile(r"^\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}$"),
        }

        row_count = len(df)

        for col in df.columns:
            unique_values = df[col].nunique(dropna=True)
            cardinality_ratio = unique_values / row_count
            col_lower = col.lower()

            # 1. Detect **Sensitive Attributes**
            if any(keyword in col_lower for keyword in sensitive_keywords):
                sensitive_attributes.append(col)
                continue

            # 2. Detect **Direct Identifiers**
            if cardinality_ratio > unique_threshold:
                direct_identifiers.append(col)
                continue

            # Object-type columns: Check regex patterns for direct identifiers
            if pd.api.types.is_object_dtype(df[col].dtype):
                non_null_values = df[col].dropna().astype(str)
                if len(non_null_values) > 0:
                    sample_data = non_null_values.sample(
                        min(sample_size, len(non_null_values)), random_state=42
                    )

                    if any(
                        sample_data.str.contains(pattern, regex=True).any()
                        for pattern in patterns.values()
                    ):
                        direct_identifiers.append(col)
                        continue

            # 3. Detect **Indirect Identifiers**
            if any(keyword in col_lower for keyword in indirect_keywords):
                indirect_identifiers.append(col)
                continue

            # 4. Detect **Quasi-Identifiers**
            if 0.1 < cardinality_ratio < unique_threshold:
                quasi_identifiers.append(col)
                continue

            # 5. Everything else → **Non-Sensitive Attributes**
            non_sensitive_attributes.append(col)

        return (
            direct_identifiers,
            quasi_identifiers,
            sensitive_attributes,
            indirect_identifiers,
            non_sensitive_attributes,
        )
