"""
PAMOLA.CORE - L-Diversity Metrics Calculator

This module provides comprehensive metrics calculation
for l-diversity anonymization techniques.

Key Features:
- Detailed performance metrics
- Information loss estimation
- Diversity measurement

(C) 2024 Realm Inveo Inc. and DGT Network Inc.
Licensed under BSD 3-Clause License
"""

import logging
import math
from typing import Dict, List, Any, Optional
from functools import lru_cache

import pandas as pd
import numpy as np


class LDiversityMetricsCalculator:
    """
    Calculates metrics for l-diversity datasets

    Provides comprehensive metrics calculation with
    caching and performance optimization.
    """

    def __init__(self, processor=None):
        """
        Initialize Metrics Calculator

        Parameters:
        -----------
        processor : object, optional
            L-Diversity processor instance for advanced calculations
        """
        self.processor = processor
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=128)
    def calculate_metrics(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive metrics calculation with caching

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns
        **kwargs : dict
            Additional metrics calculation parameters

        Returns:
        --------
        Dict[str, Any]
            Comprehensive metrics dictionary
        """
        try:
            # Calculate group diversity
            if self.processor:
                group_diversity = self.processor.calculate_group_diversity(
                    data, quasi_identifiers, sensitive_attributes
                )
            else:
                group_diversity = data.groupby(quasi_identifiers)[sensitive_attributes].agg(
                    distinct_count=('nunique')
                )

            # Calculate detailed metrics
            metrics = {
                'group_metrics': self._calculate_group_metrics(group_diversity, sensitive_attributes),
                'information_loss': self._calculate_information_loss(
                    data, quasi_identifiers, sensitive_attributes
                ),
                'diversity_metrics': self._calculate_diversity_metrics(
                    group_diversity, sensitive_attributes
                )
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Metrics calculation error: {e}")
            raise

    def _calculate_group_metrics(
            self,
            group_diversity: pd.DataFrame,
            sensitive_attributes: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate group-level metrics

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Grouped diversity information
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Group-level metrics
        """
        metrics = {
            'total_groups': len(group_diversity),
            'unique_values': {}
        }

        for sa in sensitive_attributes:
            metrics['unique_values'][sa] = {
                'mean': group_diversity[f'{sa}_distinct'].mean(),
                'median': group_diversity[f'{sa}_distinct'].median(),
                'min': group_diversity[f'{sa}_distinct'].min(),
                'max': group_diversity[f'{sa}_distinct'].max()
            }

        return metrics

    def _calculate_information_loss(
            self,
            data: pd.DataFrame,
            quasi_identifiers: List[str],
            sensitive_attributes: List[str]
    ) -> Dict[str, float]:
        """
        Calculate information loss metrics

        Parameters:
        -----------
        data : pd.DataFrame
            Original dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, float]
            Information loss metrics
        """
        # Placeholder for more complex information loss calculation
        loss_metrics = {}

        for sa in sensitive_attributes:
            # Simple information loss estimation
            original_entropy = self._calculate_entropy(data[sa])
            grouped_entropy = data.groupby(quasi_identifiers)[sa].apply(
                lambda x: self._calculate_entropy(x)
            ).mean()

            loss_metrics[sa] = {
                'original_entropy': original_entropy,
                'grouped_entropy': grouped_entropy,
                'entropy_loss_percentage': (
                    (original_entropy - grouped_entropy) / original_entropy * 100
                    if original_entropy > 0 else 0
                )
            }

        return loss_metrics

    def _calculate_diversity_metrics(
            self,
            group_diversity: pd.DataFrame,
            sensitive_attributes: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate diversity-specific metrics

        Parameters:
        -----------
        group_diversity : pd.DataFrame
            Grouped diversity information
        sensitive_attributes : List[str]
            Sensitive attribute columns

        Returns:
        --------
        Dict[str, Any]
            Diversity-specific metrics
        """
        diversity_metrics = {}

        for sa in sensitive_attributes:
            diversity_metrics[sa] = {
                'effective_distinct_values': group_diversity[f'{sa}_distinct'].mean(),
                'entropy': self._calculate_entropy(group_diversity[f'{sa}_distinct']),
                'concentration_index': group_diversity[f'{sa}_distinct'].std() /
                                       group_diversity[f'{sa}_distinct'].mean()
            }

        return diversity_metrics

    @staticmethod
    def _calculate_entropy(series: pd.Series) -> float:
        """
        Calculate Shannon entropy for a series

        Parameters:
        -----------
        series : pd.Series
            Input series of values

        Returns:
        --------
        float
            Shannon entropy value
        """
        # Normalize value counts
        value_counts = series.value_counts(normalize=True)

        # Calculate entropy
        entropy = -sum(p * math.log(p) for p in value_counts if p > 0)

        return entropy


# Utility functions for external use
def calculate_information_loss(
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attributes: List[str]
) -> Dict[str, Any]:
    """
    Calculate information loss between original and anonymized datasets

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original dataset
    anonymized_data : pd.DataFrame
        Anonymized dataset
    quasi_identifiers : List[str]
        Columns used as quasi-identifiers
    sensitive_attributes : List[str]
        Sensitive attribute columns

    Returns:
    --------
    Dict[str, Any]
        Information loss metrics
    """
    calculator = LDiversityMetricsCalculator()

    metrics = {
        'original_metrics': calculator.calculate_metrics(
            original_data, quasi_identifiers, sensitive_attributes
        ),
        'anonymized_metrics': calculator.calculate_metrics(
            anonymized_data, quasi_identifiers, sensitive_attributes
        )
    }

    return metrics