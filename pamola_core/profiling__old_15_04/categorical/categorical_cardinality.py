"""
PAMOLA.CORE - Data Field Analysis Processor  
---------------------------------------------------  
This module provides an implementation of BaseProfilingProcessor or validating categorical
fields against predefined value sets and identifying invalid or unexpected values.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.  
For details, see the LICENSE file or visit:  

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Categorical Field Profiling Operations 
--------------------------------   
It includes the following capabilities:  
- Valid/invalid value counts and percentages
- Detailed breakdown of invalid values
- Null and empty string analysis
- Pass/fail validation status
- Invalid value examples
- Visualizations of validity distribution

NOTE: Requires pandas.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from typing import Any, Dict, List, Optional
import pandas as pd

from pamola_core.profiling.categorical.base import BaseCategoricalFieldProfilingProcessor

class CategoricalValidationProfilingProcessor(BaseCategoricalFieldProfilingProcessor):
    """
    Processor or analyzing the cardinality characteristics of categorical fields, including uniqueness
    assessment and cardinality classification.
    """
    
    def __init__(
        self,
        allowed_values, 
        case_sensitive=False,
        allow_nulls=True,
        treat_empty_as_null=True, 
        save_invalid_records=False, 
        max_invalid_examples=100
    ):
        """
        Initializes the Categorical Validation Profiling Processor.

        Parameters:
        -----------
        allowed_values : List[str], required
            List of allowed values for the field (default=None).
        case_sensitive : bool, optional
            Whether validation should be case sensitive (default=False).
        allow_nulls : bool, optional
            Whether null values are allowed (default=default=True).
        treat_empty_as_null : bool, optional
            Whether to treat empty strings as nulls (default=True).
        save_invalid_records : bool, optional
            Whether to save invalid records to a file (default=False).
        max_invalid_examples: int, optional
            Maximum invalid examples to include in report (default=100).
        """
        super().__init__()
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive
        self.allow_nulls = allow_nulls
        self.treat_empty_as_null = treat_empty_as_null
        self.save_invalid_records = save_invalid_records
        self.max_invalid_examples = max_invalid_examples
    
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
            
            - allowed_values (List[str], default=self.allowed_values):
                List of allowed values for the field.
            - case_sensitive (bool, default=self.case_sensitive):
                Whether validation should be case sensitive.
            - allow_nulls (bool, default=self.allow_nulls):
                Whether null values are allowed.
            - treat_empty_as_null (bool, default=self.treat_empty_as_null):
                Whether to treat empty strings as nulls.
            - save_invalid_records (bool, default=self.save_invalid_records):
                Whether to save invalid records to a file.
            - max_invalid_examples (int, default=self.max_invalid_examples):
                Maximum invalid examples to include in report.
        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping column names to their pattern analysis results.
        """

        allowed_values = kwargs.get("allowed_values", self.allowed_values)
        case_sensitive = kwargs.get("case_sensitive", self.case_sensitive)
        allow_nulls = kwargs.get("allow_nulls", self.allow_nulls)
        treat_empty_as_null = kwargs.get("treat_empty_as_null", self.treat_empty_as_null)
        save_invalid_records = kwargs.get("save_invalid_records", self.save_invalid_records)
        max_invalid_examples = kwargs.get("max_invalid_examples", self.max_invalid_examples)

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        results = {}

        for col in columns:
            series = df[col]

            if allow_nulls:
                series = df[col].dropna()
            
            allowed_values_set = allowed_values
            if not case_sensitive:
                series = series.astype(str).str.lower()
                allowed_values_set = {v.lower() for v in allowed_values} if allowed_values else None
            
            null_mask = series.isna()
            empty_mask = series.str.strip() == ""
            if treat_empty_as_null:
                null_mask |= empty_mask
            
            total_count = len(series)
            null_count = null_mask.sum()
            valid_mask = series.notna()

            if allowed_values_set is not None:
                valid_mask &= series.isin(allowed_values_set)
            
            valid_count = valid_mask.sum()
            invalid_count = total_count - valid_count - null_count

            invalid_values = series[~valid_mask].value_counts().head(max_invalid_examples).to_dict()
            
            # Store results
            results[col] = {
                "total_count": total_count,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "null_count": null_count,
                "valid_percentage": round(valid_count / total_count * 100, 2) if total_count > 0 else 0,
                "invalid_percentage": round(invalid_count / total_count * 100, 2) if total_count > 0 else 0,
                "null_percentage": round(null_count / total_count * 100, 2) if total_count > 0 else 0,
                "pass_fail": "PASS" if invalid_count == 0 else "FAIL",
                "invalid_value_examples": invalid_values,
            }

        return results