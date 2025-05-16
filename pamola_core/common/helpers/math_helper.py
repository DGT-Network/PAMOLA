"""
PAMOLA.CORE - Utility Functions for Data Processing
---------------------------------------------------------
This module provides a set of utility functions for handling numerical operations,
data cleaning, and DataFrame manipulations using NumPy and Pandas.

Features:
 - Replace special float values (NaN, Inf) with None.
 - Compute sum, min, max of lists, NumPy arrays, and DataFrame columns.
 - Perform group-by operations on Pandas DataFrames.
 - Generate descriptive statistics for a DataFrame.

This class is useful for data profiling, feature selection, and privacy analysis.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""


import math
import numpy as np
import pandas as pd

def replace_special_floats(obj):
    """
    Recursively replace NaN and Inf values with None in dictionaries, lists, and float values.

    Args:
        obj (dict, list, float, or any type): The input object to process.

    Returns:
        dict, list, float, or any type: The processed object with NaN and Inf replaced by None.
    """
    if isinstance(obj, dict):
        # If the object is a dictionary, apply the function to each key-value pair
        return {k: replace_special_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # If the object is a list, apply the function to each element
        return [replace_special_floats(i) for i in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        # If the object is a float and is either NaN (Not a Number) or Inf (Infinity), replace it with None
        return None
    # Return the object as-is if it does not match any of the above cases
    return obj


def calculate_sum(values):
    """Return the sum of all elements in a list or NumPy array."""
    return np.sum(values)

def calculate_min(values):
    """Return the minimum value from a list or NumPy array."""
    return np.min(values)

def calculate_max(values):
    """Return the maximum value from a list or NumPy array."""
    return np.max(values)

def sum_column(df, col_name):
    """
    Calculate the sum of values in a column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Name of the column to sum.

    Returns:
        float: Sum of the column values.
    """
    return df[col_name].sum()

def min_column(df, col_name):
    """
    Find the minimum value in a column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Name of the column to find the minimum value.

    Returns:
        float: The minimum value in the column.
    """
    return df[col_name].min()

def max_column(df, col_name):
    """
    Find the maximum value in a column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Name of the column to find the maximum value.

    Returns:
        float: The maximum value in the column.
    """
    return df[col_name].max()

def std_column(df, col_name):
    """Calculate the standard deviation of a column in the DataFrame."""
    return df[col_name].std()

def group_by_dataframe(df: pd.DataFrame, group_by_columns: list):
    """
    Perform group-by operation on a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_by_columns (list): List of columns to group by.

    Returns:
        pd.pamola_core.groupby.DataFrameGroupBy: Grouped DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return ValueError("Input must be a Pandas DataFrame.")
    if not group_by_columns:
        return ValueError("group_by_columns cannot be empty.")
    
    return df.groupby(group_by_columns)

def describe_dataframe(df, include="all"):
    """
    Generate descriptive statistics for a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        include (str or list, optional): 
            - "number": Describes only numeric columns.
            - "all": Describes both numeric and non-numeric columns.
            - A list of data types (e.g., ["object", "category"]) to describe specific column types.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics.
    """
    return df.describe(include=include)
