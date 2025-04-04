"""
PAMOLA.CORE - Data Field Analysis Processor
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor` for analyzing
frequency distributions of categorical fields, focusing on how often each value
appears and identifying dominant values.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Categorical Field Profiling Operations
--------------------------------   
It includes the following capabilities:  
- Most common values and their frequencies
- Frequency distribution characterization (dominated, concentrated, diverse, uniform)
- Gini coefficient for frequency inequality
- Coefficient of variation
- Top-N concentration ratio
- Mode, average, and median frequencies
- Visualizations of value distributio

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from scipy.stats import entropy
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from pamola_core.profiling.categorical.base import BaseCategoricalFieldProfilingProcessor

class CategoricalFrequencyProfilingProcessor(BaseCategoricalFieldProfilingProcessor):
    """
    Processor for analyzing frequency distributions of categorical fields, focusing on how often 
    each value appears and identifying dominant values.
    """
    
    def __init__(
        self,
        exclude_nulls=True, 
        max_categories=100, 
        min_frequency_threshold=0.01, 
        top_n=10, 
        calculate_frequency_stats=True, 
    ):
        """
        Initializes the Categorical Frequency Profiling Processor.

        Parameters:
        -----------
        exclude_nulls : bool, optional
            Whether to exclude null values from analysis (default=true).
        max_categories : int, optional
            Maximum number of categories to analyze in detail (default=100).
        min_frequency_threshold : float, optional
            Minimum frequency threshold for including in reports (default=0.01).
        top_n : int, optional
            Number of top values to highlight in results (default=10).
        calculate_frequency_stats : bool, optional
            Whether to calculate advanced frequency statistics (default=True).
        """
        super().__init__()
        self.exclude_nulls = exclude_nulls
        self.max_categories = max_categories
        self.min_frequency_threshold = min_frequency_threshold
        self.top_n = top_n
        self.calculate_frequency_stats = calculate_frequency_stats
    
    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform categorical validation analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all categorical columns will be selected.
        **kwargs : dict
            Dynamic parameter overrides:
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
            A dictionary mapping column names to their pattern analysis results.
        """

        exclude_nulls = kwargs.get("exclude_nulls", self.exclude_nulls)
        max_categories = kwargs.get("max_categories", self.max_categories)
        min_frequency_threshold = kwargs.get("min_frequency_threshold", self.min_frequency_threshold)
        top_n = kwargs.get("top_n", self.top_n)
        calculate_frequency_stats = kwargs.get("calculate_frequency_stats", self.calculate_frequency_stats)

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        results = {}

        for col in columns:
            series = df[col]
            if exclude_nulls:
                series = series.dropna()

            value_counts = series.value_counts(normalize=True)

            filtered_counts = value_counts[value_counts >= min_frequency_threshold]

            if len(filtered_counts) > max_categories:
                filtered_counts = filtered_counts.head(max_categories)

            stats = {}
            if calculate_frequency_stats:
                stats = self._calculate_frequency_characterization(value_counts, top_n)

            results[col] = {
                'top_values': filtered_counts.to_dict(),
                'stats': stats
            }
        
        return results
    
    def _gini_coefficient(self, frequencies: List[float]) -> float:
        """
        Calculate the Gini coefficient, a measure of frequency inequality.

        Parameters:
        -----------
        frequencies : List[float]
            A list of frequencies representing the distribution of categorical values.

        Returns:
        --------
        float
            The Gini coefficient, ranging from 0 (perfect equality) to 1 (maximum inequality).
        """
        
        if len(frequencies) == 0:
            return 0.0  # Avoid division by zero

        sorted_freqs = np.sort(frequencies)  # Sort frequencies in ascending order
        n = len(sorted_freqs)
        mean_freq = np.mean(sorted_freqs)

        if mean_freq == 0:
            return 0.0  # Avoid division by zero

        # Compute Gini coefficient
        cumulative_diffs = np.sum(np.abs(sorted_freqs[:, None] - sorted_freqs))
        gini = cumulative_diffs / (2 * n**2 * mean_freq)

        return gini
    
    def _calculate_frequency_characterization(self, value_counts: pd.Series, top_n: int):
        """
        Determine distribution type (dominated, concentrated, diverse, uniform)
        based on statistical measures of frequency inequality.

        Parameters:
        -----------
        value_counts : pd.Series
            A Pandas Series containing value frequencies of a categorical field.
        top_n : int
            Number of top values to consider for concentration ratio calculation.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing statistical measures of frequency distribution, including:
            - mode (most frequent value)
            - mean, median, standard deviation of frequencies
            - coefficient of variation
            - Gini coefficient (inequality measure)
            - Berger-Parker Index (dominance measure)
            - Normalized entropy (spread measure)
            - Top-N concentration ratio
            - Distribution type classification
        """
        stats = {}
        if value_counts is None:
            return None
        frequencies = value_counts.values
        
        # Mode - Most frequent value
        stats['mode'] = value_counts.idxmax()
        
        # Mean & Median Frequency
        stats['mean_frequency'] = np.mean(frequencies)
        stats['median_frequency'] = np.median(frequencies)

        # Standard deviation of frequencies
        stats['std_frequency'] = np.std(frequencies)

        # Coefficient of Variation (standard deviation / mean)
        stats['coefficient_of_variation'] = (
            stats['std_frequency'] / stats['mean_frequency'] if stats['mean_frequency'] != 0 else 0
        )
        
        # Gini coefficient - Measures frequency inequality
        stats['gini_coefficient'] = self._gini_coefficient(frequencies)

        sum_frequencies = sum(frequencies)
        
        # Berger-Parker Index - Measures dominance of most frequent category
        stats['berger_parker_index'] = frequencies[0] / sum_frequencies

        # Entropy - Measures spread of values
        h = entropy(frequencies, base=2)
        max_entropy = np.log2(len(frequencies)) if len(frequencies) > 1 else 1
        stats['normalized_entropy'] = h / max_entropy  # Normalize from 0 to 1

        # Top-N Concentration Ratio - Sum of top N frequencies relative to total
        stats['top_n_concentration_ratio'] = sum(frequencies[:top_n]) / sum_frequencies
        
        # Classify distribution type based on statistical measures
        if stats['gini_coefficient'] > 0.6 and stats['berger_parker_index'] > 0.7:
            stats['distribution_type'] = "Dominated"
        elif stats['gini_coefficient'] > 0.4 and stats['berger_parker_index'] > 0.4:
            stats['distribution_type'] = "Concentrated"
        elif stats['normalized_entropy'] > 0.6:
            stats['distribution_type'] = "Diverse"
        else:
            stats['distribution_type'] = "Uniform"
        
        return stats