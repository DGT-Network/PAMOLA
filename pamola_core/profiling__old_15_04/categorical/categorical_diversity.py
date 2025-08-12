"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor` for analyzing
diversity metrics of categorical fields, calculating entropy, Gini impurity, and other
measures of distribution diversity.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Categorical Field Profiling Operations 
--------------------------------   
It includes the following capabilities:  
- Shannon entropy and normalized entropy.
- Gini impurity (probability of misclassifying a randomly chosen element).
- Simpson's diversity index.
- Berger-Parker dominance index.
- Hill numbers (effective number of categories).
- Diversity classification (very_low, low, medium, high).
- Distribution visualization (Lorenz curve).

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import entropy

from pamola_core.profiling.base import BaseProfilingProcessor

class CategoricalDiversityProfilingProcessor(BaseProfilingProcessor, ABC):
    """
    Processor for analyzing the cardinality characteristics of categorical fields, including
    uniqueness assessment and cardinality classification.
    """
    
    def __init__(
        self,
        exclude_nulls: bool = True,
        max_categories: int = 100,
        min_frequency_threshold: float = 0.01,
        top_n: float = 0.8,
        calculate_frequency_stats: bool = True,
    ):
        """
        Initializes the Categorical Cardinality Profiling Processor with configurable options.

        Parameters:
        -----------
        exclude_nulls: bool, optional (default=True)
            Whether to exclude null values from analysis.
        max_categories: int, optional (default=100)
            Maximum number of categories to analyze in detail.
        min_frequency_threshold: float, optional (default=0.01)
            Minimum frequency threshold for including in reports.
        top_n: float, optional (default=0.8)
            Number of top values to highlight in results.
        calculate_frequency_stats: bool, optional (default=True)
            Whether to calculate advanced frequency statistics.
        """
        super().__init__()  # Ensure proper inheritance from BaseProfilingProcessor
        self.exclude_nulls = exclude_nulls
        self.max_categories = max_categories
        self.min_frequency_threshold = min_frequency_threshold
        self.top_n = top_n
        self.calculate_frequency_stats = calculate_frequency_stats
    
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

    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform categorical field analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing numeric columns to analyze.
        
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all numeric columns will be automatically selected.

        **kwargs : dict
            Optional parameters to override class-level settings dynamically:
            - `exclude_nulls` (bool, default=self.exclude_nulls): 
                Whether to exclude null values from analysis.
            - `max_categories` (int, default=self.max_categories): 
                Maximum number of categories to analyze in detail.
            - `min_frequency_threshold` (float, default=self.min_frequency_threshold): 
                Minimum frequency threshold for including in reports.
            - `top_n` (int, default=self.top_n): 
                Number of top values to highlight in results.
            - `calculate_frequency_stats` (bool, default=self.calculate_frequency_stats): 
                Whether to calculate advanced frequency statistics.

        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their numeric analysis results.
        """

        exclude_nulls = kwargs.get("exclude_nulls", self.exclude_nulls)
        max_categories = kwargs.get("max_categories", self.max_categories)
        min_frequency_threshold = kwargs.get("min_frequency_threshold", self.min_frequency_threshold)
        top_n = kwargs.get("top_n", self.top_n)
        calculate_frequency_stats = kwargs.get("calculate_frequency_stats", self.calculate_frequency_stats)

        results = {}
        columns = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            series = df[col]
            if exclude_nulls:
                series = series.dropna()
            
            value_counts = series.value_counts(normalize=True)
            
            # Apply thresholds
            if min_frequency_threshold > 0:
                value_counts = value_counts[value_counts >= min_frequency_threshold]
            
            if max_categories and len(value_counts) > max_categories:
                value_counts = value_counts[:max_categories]
            
            # Shannon entropy
            shannon_entropy = entropy(value_counts, base=2)
            normalized_entropy = shannon_entropy / np.log2(len(value_counts)) if len(value_counts) > 1 else 0

            # Gini impurity
            gini_impurity = 1 - np.sum(value_counts**2)

            # Simpson's diversity index
            simpson_index = 1 - np.sum(value_counts**2)

            # Berger-Parker dominance index
            berger_parker_index = value_counts.max()

            # Hill numbers (effective number of categories)
            hill_number_q0 = self._hill_number(series, q=0)
            hill_number_q1 = self._hill_number(series, q=1)
            hill_number_q2 = self._hill_number(series, q=2)
            hill_number_q3 = self._hill_number(series, q=3)
            hill_number_q05 = self._hill_number(series, q=0.5)

            # Diversity classification
            diversity_class = self._classify_diversity(normalized_entropy)

            # Top-N categories
            top_n_count = min(len(value_counts), max(1, int(top_n * len(value_counts))))
            top_categories = value_counts.head(top_n_count).to_dict() if calculate_frequency_stats else None

            results[col] = {
                "shannon_entropy": shannon_entropy,
                "normalized_entropy": normalized_entropy,
                "gini_impurity": gini_impurity,
                "simpson_index": simpson_index,
                "berger_parker_index": berger_parker_index,
                "hill_number": {
                    "hill_number_q0": hill_number_q0,
                    "hill_number_q1": hill_number_q1,
                    "hill_number_q2": hill_number_q2,
                    "hill_number_q3": hill_number_q3,
                    "hill_number_q05": hill_number_q05,
                },
                "diversity_class": diversity_class,
                "top_categories": top_categories,
            }
        
        return results

    def _classify_diversity(self, normalized_entropy: float) -> str:
        """
        Classifies the diversity level based on normalized entropy.
        
        Parameters:
        - normalized_entropy (float): A value between 0 and 1, representing the normalized entropy.
        
        Returns:
        - str: A string classification of diversity level: "very_low", "low", "medium", or "high".
        
        Classification thresholds:
        - < 0.3  → "very_low"  (Highly dominated by a few categories)
        - < 0.5  → "low"       (Moderate dominance by some categories)
        - < 0.7  → "medium"    (Balanced diversity)
        - ≥ 0.7  → "high"      (Highly diverse distribution)
        """

        if normalized_entropy < 0.3:
            return "very_low"
        elif normalized_entropy < 0.5:
            return "low"
        elif normalized_entropy < 0.7:
            return "medium"
        else:
            return "high"
    
    def _hill_number(self, series: pd.Series, q: float = 1):
        """
        Calculate Hill Numbers (Effective Number of Categories) for a given q.
        
        Parameters:
        - series (pd.Series): A Pandas Series containing categorical data.
        - q (float): The diversity order parameter. 
            - q = 0: Counts the actual number of unique categories (Species Richness).
            - q = 1: Based on Shannon entropy (measures general diversity).
            - q = 2: Based on Simpson’s Index (emphasizes dominant categories).
            - q > 2 or q < 1: Uses the generalized Hill Number formula.

        Returns:
        - float: Hill number value, representing the effective number of categories.
        """
        value_counts = series.value_counts(normalize=True)

        if q == 0:
            # Simply count the number of unique categories (Species Richness)
            return series.nunique()
        elif q == 1:
            # Based on Shannon entropy, measures overall diversity
            return np.exp(entropy(value_counts, base=2))  # Exp(Shannon entropy)
        elif q == 2:
            # Based on Simpson’s Index, emphasizes dominant categories
            return 1 / np.sum(value_counts ** 2)  # Inverse of Simpson's index
        else:
            # Generalized Hill Number formula
            return (np.sum(value_counts ** q)) ** (1 / (1 - q))