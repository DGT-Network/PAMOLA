"""
PAMOLA.CORE - A utility class for data type detection and range processing.
---------------------------------------------------------
This module provides various utility functions for data type detection, data validation,
transformation, and generalization. It supports checking numeric types, detecting malformed
values, converting range strings, binning numeric data, and applying privacy transformations.

Key features:
 - Data Type Detection: Identifies numeric, boolean, integer, and float values, and determines if a column mostly contains integers.
 - Range and String Handling: Detects and processes numeric ranges, converting them into numeric values and extracting bounds.
 - Data Quality and Validation: Checks for misformatted numeric, boolean, date, and categorical values, ensuring data consistency.
 - Range Calculation: Computes minimum and maximum values for numeric and date columns.
 - Generalization & Privacy Preservation: Applies generalization, binning, and k-anonymity techniques to protect privacy in data.

This class is useful for data preprocessing, validation, generalization, and privacy-preserving transformations in machine learning
and data analytics workflows.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""


import datetime
import re
from typing import List
import numpy as np
import pandas as pd

class DataHelper:
    """
    A helper class providing utility functions for data type detection
    and range string conversion.
    """

    @staticmethod
    def is_non_numeric(column):
        """
        Check if a column is non-numeric.

        Parameters:
            column (pd.Series): The column to check.

        Returns:
            bool: True if the column is non-numeric, False otherwise.
        """
        return pd.api.types.is_object_dtype(column.dtype) or not np.issubdtype(
            column.dtype, np.number
        )

    @staticmethod
    def is_bool(column):
        """
        Check if a column contains boolean data.

        Parameters:
            column (pd.Series): The column to check.

        Returns:
            bool: True if the column contains boolean data, False otherwise.
        """
        return column.dtype == "bool" or set(column.dropna().unique()).issubset(
            {True, False, "yes", "no"}
        )

    @staticmethod
    def is_integer(value: any) -> bool:
        """
        Checks if a given value is an integer, allowing `.0` but rejecting `.00`, `.000`, etc.

        Parameters:
        - value (any): The value to check (can be int, float, or str).

        Returns:
        - bool: True if the value is an integer (including ".0" format), False otherwise.
        """
        if pd.isna(value):  # Handle NaN values
            return False
        return bool(re.match(r"^-?\d+(\.0)?$", str(value)))

    @staticmethod
    def is_float(value: any) -> bool:
        """
        Checks if a given value is a float with a valid decimal part (rejects `.0` but allows `.01`, `.10`, etc.).

        Parameters:
        - value (any): The value to check (can be int, float, or str).

        Returns:
        - bool: True if the value is a float with a valid decimal part, False otherwise.
        """
        if pd.isna(value):  # Handle NaN values
            return False
        return bool(re.match(r"^-?\d+\.\d*[1-9]\d*$", str(value)))

    @staticmethod
    def determine_mostly_integer(df: pd.DataFrame, column: str, sample_size: int = 1000) -> bool:
        """
        Determines if a column contains mostly integer-like values.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column (str): The target column to analyze.
        - sample_size (int, optional): The maximum number of samples to analyze (default: 1000).

        Returns:
        - bool: True if the majority of values are integers, False otherwise.
        """
        non_na_values = df[column].dropna()
        if non_na_values.empty:
            return False

        # Sample up to `sample_size` rows for better performance
        sample_data = non_na_values.sample(n=min(sample_size, len(non_na_values)), random_state=42)

        # Count integer and float-like values
        integer_count = sample_data.apply(DataHelper.is_integer).sum()
        float_count = sample_data.apply(DataHelper.is_float).sum()

        return integer_count > float_count

    @staticmethod
    def is_range_string(value):
        """
        Check if a value is a range string, like '18-35', '-0.38--0.13', or '0.38--0.13'.

        Parameters:
            value (str): The value to check.

        Returns:
            bool: True if the value is a range string, False otherwise.
        """
        return isinstance(value, str) and bool(
            re.match(r"^-?\d+(\.\d+)?-(-?\d+(\.\d+)?)$", value)
        )

    @staticmethod
    def convert_range_to_numeric(range_str):
        """
        Convert a range string like '-0.38--0.13', '-0.38-0.13', or '0.38--0.13' into its numeric midpoint.

        Parameters:
            range_str (str): A string representing a range (e.g., '1-5' or '-0.38--0.13').

        Returns:
            float or int: The midpoint of the range.
        """
        try:
            # Use regex to split the range string correctly
            match = re.match(r"^(-?\d+(\.\d+)?)-(-?\d+(\.\d+)?)$", range_str)
            if not match:
                raise ValueError(f"Invalid range format: {range_str}")

            # Extract start and end values
            start, end = match.group(1), match.group(3)

            # Determine if the range contains float values
            is_float = "." in start or "." in end

            # Convert start and end to numeric types (float or int)
            start = float(start) if is_float else int(start)
            end = float(end) if is_float else int(end)

            # Compute the midpoint
            midpoint = (start + end) / 2

            # Return as int if no decimal point, otherwise as float
            return int(midpoint) if not is_float else midpoint

        except ValueError as e:
            print(f"Error processing range: {e}")
            pass
            return None

    @staticmethod
    def calculate_misformatted_count_numeric(col):
        """
        Identify misformatted numeric values in a column.

        Parameters:
            col (pd.Series): The column to check.

        Returns:
            int: The count of misformatted values.
        """
        misformatted_count = col.apply(lambda x: not pd.api.types.is_number(x)).sum()
        return misformatted_count

    @staticmethod
    def calculate_misformatted_count_bool(col):
        """
        Calculate the number of misformatted values in a boolean column.

        Parameters:
            col (pd.Series): The column to check.

        Returns:
            int: The count of misformatted boolean values.
        """
        valid_bool_values = {True, False}

        misformatted_count = (
            col.dropna().apply(lambda x: x not in valid_bool_values).sum()
        )

        return misformatted_count

    @staticmethod
    def calculate_misformatted_count_date(col):
        """
        Identify misformatted date values in a column.

        Parameters:
            col (pd.Series): The column to check.

        Returns:
            int: The count of misformatted date values.
        """
        misformatted_count = (
            col.apply(lambda x: pd.to_datetime(x, errors="coerce")).isna().sum()
        )
        return misformatted_count

    @staticmethod
    def calculate_misformatted_count_category(col, allowed):
        """
        Identify misformatted category values in a column.

        Parameters:
            col (pd.Series): The column to check.
            allowed (list): The allowed category values.

        Returns:
            int: The count of misformatted category values.
        """
        misformatted_count = col.apply(lambda x: x not in allowed).sum()
        return misformatted_count

    @staticmethod
    def calculate_range_values(col, date_format: str = "%Y-%m-%d") -> str:
        """
        Calculate the range of values as a 'min-max' string.

        Parameters:
        - col (pd.Series): The data column to calculate the range for.
        - date_format (str, optional): The date format when processing datetime data. Default is "%Y-%m-%d".

        Returns:
        - str: A string representing the value range as 'min-max'. Returns None if the data is invalid or unsupported.
        """
        if col.empty or col.isna().all():
            return None  # Return None if the column is empty or all values are NaN

        # Process numerical data
        if pd.api.types.is_numeric_dtype(col):
            min_val, max_val = col.min(), col.max()
            return (
                f"{min_val}-{max_val}"
                if pd.notna(min_val) and pd.notna(max_val)
                else None
            )

        # Process datetime data
        elif (
            pd.api.types.is_datetime64_any_dtype(col)
            or pd.to_datetime(col, errors="coerce", utc=True).notna().all()
        ):
            min_val, max_val = col.min(), col.max()
            return (
                f"{min_val.strftime(date_format)} - {max_val.strftime(date_format)}"
                if pd.notna(min_val) and pd.notna(max_val)
                else None
            )

        return None  # Not applicable for boolean, categorical, or object data

    @staticmethod
    def get_allowed_categories(col):
        """
        Extract allowed categories from a column, filtering out non-string and numeric values.

        Parameters:
            col (pd.Series): The column to check.

        Returns:
            list: The list of allowed categories.
        """
        allowed_categories = (
            col.dropna()
            .loc[(col != "") & (~col.apply(lambda x: isinstance(x, (int, float))))]
            .unique()
            .tolist()
        )
        return allowed_categories

    @staticmethod
    def is_malformed(value, col_dtype, date_range=(1900, 2100)):
        """
        Check if a value is malformed based on its column's data type.

        Parameters:
        value: The value to check.
        col_dtype: The data type of the column.
        date_range: A tuple specifying the valid year range for datetime values.

        Returns:
        bool: True if the value is malformed, False otherwise.
        """
        if pd.isna(value):  # Handle NaN or None
            return True

        if pd.api.types.is_numeric_dtype(col_dtype):
            if isinstance(value, (int, float, np.number)):
                return np.isinf(value) or np.isnan(
                    value
                )  # Check for infinity and NaN explicitly
            return True  # If it's not numeric, it's malformed

        if pd.api.types.is_string_dtype(col_dtype):
            return (
                isinstance(value, str) and value.strip() == ""
            )  # Empty or whitespace-only strings

        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            if not isinstance(value, (pd.Timestamp, datetime)):
                return True
            try:
                return not (date_range[0] <= value.year <= date_range[1])
            except AttributeError:
                return True  # Malformed if it doesn't have a year attribute

        if pd.api.types.is_bool_dtype(col_dtype):
            return not isinstance(
                value, (bool, np.bool_)
            )  # Only allow True/False, not 1/0 or "true"/"false"

        if pd.api.types.is_categorical_dtype(col_dtype):
            if col_dtype.categories is None:
                return True
            return value not in col_dtype.categories

        return False

    @staticmethod
    def extract_range(value):
        """
        Extracts the lower and upper bounds from a range string like '31-32'.
        Returns (lower_bound, upper_bound) or (None, None) if not a valid range.
        """
        # Use regex to split the range string correctly
        match = re.match(r"^(-?\d+(\.\d+)?)-(-?\d+(\.\d+)?)$", value)
        if not match:
            raise ValueError(f"Invalid range format: {value}")

        # Extract lower_bound and upper_bound values
        lower_bound, upper_bound = match.group(1), match.group(3)

        # Determine if the range contains float values
        is_float = "." in lower_bound or "." in upper_bound

        # Convert lower_bound and upper_bound to numeric types (float or int)
        lower_bound = float(lower_bound) if is_float else int(lower_bound)
        upper_bound = float(upper_bound) if is_float else int(upper_bound)
        if match:
            return lower_bound, upper_bound
        return None, None

    @staticmethod
    def bin_numeric(series: pd.Series, num_bins: int) -> pd.Series:
        """Bins numeric values into equal-width ranges with text labels."""
        min_val, max_val = series.min(), series.max()
        bins = np.linspace(min_val, max_val, num_bins + 1)
        bins = np.unique(bins)  # Ensure unique bin edges

        if len(bins) < 2:
            raise ValueError("Insufficient unique bin edges for numeric binning.")

        # If the series is of integer type, adjust bins
        if np.issubdtype(series.dtype, np.integer):
            bins = bins.astype(int)

        # Generate labels
        labels = [f" {bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

        return pd.cut(
            series, bins=bins, labels=labels, include_lowest=True, duplicates="drop"
        )

    @staticmethod
    def generalize_categorical_or_range(value):
        """Generalizes categorical and range-based values."""
        if isinstance(value, str) and "-" in value:
            return DataHelper.generalize_range(value)
        elif isinstance(value, str) and len(value) > 1:
            return value[:-1] + "*"  # Prefix generalization
        return "*"  # Default generalization for single-character values or unknowns

    @staticmethod
    def generalize_range(value: str) -> str:
        """Generalizes a numerical range (e.g., '31-32' → '30-34')."""
        predefined_ranges = {
            (30, 34): "30-34",
            (35, 39): "35-39",
            (40, 49): "40-49",
        }

        try:
            lower, upper = map(int, value.split("-"))
            for (low, high), label in predefined_ranges.items():
                if low <= lower <= high and low <= upper <= high:
                    return label
            return f"{lower}-{upper}"  # Keep original if no match
        except ValueError:
            return value  # Return as-is if not a valid range

    @staticmethod
    def transform_for_privacy(
        df: pd.DataFrame, column: str, num_bins: int = 4
    ) -> pd.DataFrame:
        """
        Generalizes a column based on its data type:
        - **Numerical**: Bins numeric values into ranges.
        - **Ranges**: Merges predefined ranges into broader categories.
        - **Categorical**: Applies prefix-based generalization.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            column (str): Column to generalize.
            num_bins (int): Number of bins for numeric values.

        Returns:
            pd.DataFrame: Dataframe with generalized column.
        """
        df_copy = df.copy()

        if column not in df_copy.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = DataHelper.bin_numeric(df_copy[column], num_bins)
        elif df_copy[column].dtype == object:
            df_copy[column] = df_copy[column].apply(
                DataHelper.generalize_categorical_or_range
            )
        else:
            raise ValueError(f"Unsupported data type for column '{column}'")

        return df_copy

    @staticmethod
    def suppress_non_k_anonymous(
        df: pd.DataFrame, quasi_identifiers: List[str], k: int
    ) -> pd.DataFrame:
        """
        Suppresses (removes) rows that do not meet the k-anonymity requirement.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            quasi_identifiers (List[str]): List of quasi-identifiers.
            k (int): Required k-anonymity level.

        Returns:
            pd.DataFrame: DataFrame with suppressed rows.
        """
        equivalence_classes = df.groupby(quasi_identifiers).size()
        valid_classes = equivalence_classes[equivalence_classes >= k].index

        return df[
            df.set_index(quasi_identifiers).index.isin(valid_classes)
        ].reset_index(drop=True)

    @staticmethod
    def adjust_categorical_column(
        df: pd.DataFrame, column: str, quasi_identifiers: List[str]
    ) -> pd.Series:
        df_copy = df.copy()

        if column not in df_copy.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        # Compute global distribution
        global_distribution = df_copy[column].value_counts(normalize=True)
        if not np.isclose(global_distribution.sum(), 1.0):
            global_distribution /= global_distribution.sum()

        categories = global_distribution.index.to_numpy()
        probabilities = global_distribution.values

        # Get group indices
        grouped_indices = df_copy.groupby(quasi_identifiers, observed=False).indices

        # Use NumPy array for efficiency
        column_values = df_copy[column].values  # Directly access NumPy array

        for group, indices in grouped_indices.items():
            indices_array = np.array(indices)  # Ensure indices is a NumPy array
            column_values[indices_array] = np.random.choice(
                categories, size=len(indices_array), p=probabilities
            )

        return pd.Series(
            column_values, index=df.index
        )  # Return only the adjusted column
