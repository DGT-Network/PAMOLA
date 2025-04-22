"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of `BaseProfilingProcessor` for analyzing
patterns in categorical fields, including validation against allowed values, regex
pattern detection, and multi-value field handling.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Categorical Field Profiling Operations 
--------------------------------   
It includes the following capabilities:  
- Matching/non-matching values analysis
- Pattern detection and classification
- Multi-value field analysis (if separator provided)
- Value distribution statistics
- List of non-matching values with frequency
- Pattern frequency distribution
- Value dictionary for referenc

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


import json
import re
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pamola_core.profiling.base import BaseProfilingProcessor

class CategoricalPatternProfilingProcessor(BaseProfilingProcessor):
    """
    Processor for analyzing patterns in categorical fields, including validation
    against allowed values, regex pattern detection, and multi-value field handling.
    """
    
    def __init__(
        self,
        allowed_values=None, 
        multi_value_separator=None, 
        case_sensitive=False, 
        allow_nulls=True, 
        detect_regex_patterns=True, 
        min_pattern_coverage=0.7, 
        common_patterns=None, 
        save_non_matching=False, 
        max_non_matching_examples=1000
    ):
        """
        Initializes the Categorical Pattern Profiling Processor.

        Parameters:
        -----------
        allowed_values : list or str, optional
            List of allowed values or file path containing them (e.g., ["Male", "Female", "Other"], "allowed_values.json").
        multi_value_separator : str, optional
            Separator for multi-value fields (e.g., ",", ";", "|").
        case_sensitive : bool, default=False
            Whether validation should be case-sensitive.
        allow_nulls : bool, default=True
            Whether null values are allowed.
        detect_regex_patterns : bool, default=True
            Whether to detect regex patterns in data.
        min_pattern_coverage : float, default=0.7
            Minimum coverage for a pattern to be reported (e.g., 0.5, 0.8).
        common_patterns : list, optional
            List of common patterns to check (e.g., [{"name": "email", "regex": "..."}]).
        save_non_matching : bool, default=False
            Whether to save non-matching records to a file.
        max_non_matching_examples : int, default=1000
            Maximum non-matching examples to include (e.g., 100, 5000).
        """
        super().__init__()
        self.allowed_values = allowed_values
        self.multi_value_separator = multi_value_separator
        self.case_sensitive = case_sensitive
        self.allow_nulls = allow_nulls
        self.detect_regex_patterns = detect_regex_patterns
        self.min_pattern_coverage = min_pattern_coverage
        self.common_patterns = common_patterns
        self.save_non_matching = save_non_matching
        self.max_non_matching_examples = max_non_matching_examples
    
    def execute(self, df: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform categorical pattern analysis on the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        columns : Optional[List[str]], default=None
            A list of column names to analyze. If None, all categorical columns will be selected.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - `allowed_values` (list or str, default=self.allowed_values):
                List of allowed values, or file path containing them.
            - `multi_value_separator` (str, default=self.multi_value_separator):
                Separator for multi-value fields.
            - `case_sensitive` (bool, default=self.case_sensitive):
                Whether validation should be case sensitive.
            - `allow_nulls` (int, default=self.allow_nulls):
                Whether null values are allowed.
            - `detect_regex_patterns` (int, default=self.detect_regex_patterns):
                Whether to detect regex patterns in data.
            - `min_pattern_coverage` (int, default=self.min_pattern_coverage):
                Minimum coverage for a pattern to be reported.
            - `common_patterns` (int, default=self.common_patterns):
                List of common patterns to check.
            - `save_non_matching` (int, default=self.save_non_matching):
                Whether to save non-matching records to a file.
            - `max_non_matching_examples` (int, default=self.max_non_matching_examples):
                Maximum non-matching examples to include.

        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their pattern analysis results.
        """

        allowed_values = kwargs.get("allowed_values", self.allowed_values)
        multi_value_separator = kwargs.get("multi_value_separator", self.multi_value_separator)
        case_sensitive = kwargs.get("case_sensitive", self.case_sensitive)
        allow_nulls = kwargs.get("allow_nulls", self.allow_nulls)
        detect_regex_patterns = kwargs.get("detect_regex_patterns", self.detect_regex_patterns)
        min_pattern_coverage = kwargs.get("min_pattern_coverage", self.min_pattern_coverage)
        common_patterns = kwargs.get("common_patterns", self.common_patterns)
        save_non_matching = kwargs.get("save_non_matching", self.save_non_matching)
        max_non_matching_examples = kwargs.get("max_non_matching_examples", self.max_non_matching_examples)

        results = {}

        # If no columns are specified, analyze all categorical columns
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            series = df[col]

            # Drop null values if not allowed
            if not allow_nulls:
                series = series.dropna()

            # Convert to lowercase if case-insensitive comparison is enabled
            if not case_sensitive:
                series = series.str.lower()

            # If multi-value separator is provided, split the values and expand
            if multi_value_separator:
                series = series.str.split(multi_value_separator).explode()

            # Compute value counts
            value_counts = series.value_counts()
            total_values = len(series)

            # Matching / Non-matching values
            matching_values = value_counts.index
            non_matching_counts = {}
            if allowed_values:
                matching_values = value_counts.index.intersection(allowed_values)
                non_matching_values = value_counts.index.difference(allowed_values)
                non_matching_counts = value_counts.reindex(non_matching_values, fill_value=0).nlargest(max_non_matching_examples).to_dict()

            # Only run regex pattern detection if `detect_regex_patterns=True`
            pattern_distribution = {}
            if detect_regex_patterns and common_patterns:
                pattern_counts = self._detect_patterns_pandas(series, common_patterns)
                pattern_distribution = {p: c for p, c in pattern_counts.items() if c / total_values >= min_pattern_coverage}

            # Save non-matching values to a JSON file if `save_non_matching=True`
            if save_non_matching and non_matching_counts:
                filename = f"non_matching_{col}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(non_matching_counts, f, ensure_ascii=False, indent=4)

            results[col] = {
                "total_values": total_values,
                "matching_values": len(matching_values),
                "non_matching_values": len(non_matching_values),
                "non_matching_counts": non_matching_counts,
                "pattern_distribution": pattern_distribution,
                "value_distribution": value_counts.to_dict(),
                "value_dictionary": list(matching_values),
            }
        
        return results
    
    def _detect_patterns_pandas(self, series: pd.Series, common_patterns: list) -> Dict[str, int]:
        """
        Detects patterns in a given Pandas Series using regex matching.

        Parameters:
        -----------
        series : pd.Series
            The Pandas Series containing categorical data to analyze.
        common_patterns : list
            A list of dictionaries, where each dictionary represents a pattern to check.
            Example:
            [
                {"name": "email", "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
                {"name": "phone_number", "regex": r"^\+?[0-9]{10,15}$"}
            ]

        Returns:
        --------
        Dict[str, int]
            A dictionary mapping pattern names to the count of matching values in the series.
        """

        # Dictionary to store pattern match counts
        pattern_counts = {}

        # Iterate through each pattern definition
        for pattern in common_patterns:
            matches = series.str.extract(f'({pattern["regex"]})', expand=False)
            pattern_counts[pattern["name"]] = matches.notna().sum()

        # Return a dictionary mapping pattern names to the count of matching records
        return pattern_counts