import re
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
        return column.dtype == 'object' or not np.issubdtype(column.dtype, np.number)

    @staticmethod
    def is_bool(column):
        """
        Check if a column contains boolean data.

        Parameters:
            column (pd.Series): The column to check.

        Returns:
            bool: True if the column contains boolean data, False otherwise.
        """
        return column.dtype == 'bool' or set(column.dropna().unique()).issubset({True, False, 'yes', 'no'})

    @staticmethod
    def is_range_string(value):
        """
        Check if a value is a range string, like '18-35', '-0.38--0.13', or '0.38--0.13'.

        Parameters:
            value (str): The value to check.

        Returns:
            bool: True if the value is a range string, False otherwise.
        """
        return isinstance(value, str) and bool(re.match(r"^-?\d+(\.\d+)?-(-?\d+(\.\d+)?)$", value))
        
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
            is_float = '.' in start or '.' in end

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

        misformatted_count = col.dropna().apply(lambda x: x not in valid_bool_values).sum()

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
        misformatted_count = col.apply(lambda x: pd.to_datetime(x, errors='coerce')).isna().sum()
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
    def get_allowed_categories(col):
        """
        Extract allowed categories from a column, filtering out non-string and numeric values.

        Parameters:
            col (pd.Series): The column to check.

        Returns:
            list: The list of allowed categories.
        """
        allowed_categories = col.dropna().loc[
            (col != "") & (~col.apply(lambda x: isinstance(x, (int, float))))
        ].unique().tolist()
        return allowed_categories
    
    @staticmethod
    def is_numerical(column_type) -> bool:
        """
        Check if the column type is numerical (integer or float).
        
        Args:
        column_type: The data type of the column (e.g., int, float, etc.)
        
        Returns:
        bool: True if the column type is either integer or float, otherwise False.
        """
        return pd.api.types.is_integer_dtype(column_type) or pd.api.types.is_float_dtype(column_type)

    @staticmethod
    def is_categorical(column_type) -> bool:
        """
        Check if the column type is categorical (object or boolean).
        
        Args:
        column_type: The data type of the column (e.g., object, bool, etc.)
        
        Returns:
        bool: True if the column type is either object (categorical) or boolean, otherwise False.
        """
        return pd.api.types.is_object_dtype(column_type) or pd.api.types.is_bool_dtype(column_type)

    @staticmethod
    def is_datetime(column_type) -> bool:
        """
        Check if the column type is datetime.
        
        Args:
        column_type: The data type of the column (e.g., datetime64, etc.)
        
        Returns:
        bool: True if the column type is datetime, otherwise False.
        """
        return pd.api.types.is_datetime64_any_dtype(column_type)
    
    @staticmethod
    def get_column_dtype(column: pd.Series) -> str:
        """
        Get the data type of a Pandas Series as a string.

        Parameters:
        column (pd.Series): The column for which to get the data type.

        Returns:
        str: The string representation of the column's data type.
        """
        return str(column.dtype)
    