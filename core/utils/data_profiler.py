# Standard Library Imports
import pandas as pd
import logging
from scipy.stats import entropy

logger = logging.getLogger(__name__)

class DataProfiler:

    @staticmethod
    def compute_shannon_entropy(column: pd.Series, base: int = 2) -> float:
        """
        Compute the Shannon entropy of a Pandas Series.

        Parameters:
        -----------
        column : pd.Series
            The input data column for entropy calculation.
        base : int, optional (default=2)
            The logarithm base used for entropy computation (e.g., 2 for bits, e for nats).

        Returns:
        --------
        float
            The calculated Shannon entropy value. Returns 0.0 if the input column is empty.
        """
        if not isinstance(column, pd.Series):
            raise TypeError("Input must be a Pandas Series.")

        # Remove NaN values
        column_clean = column.dropna()

        if column_clean.empty:
            return 0.0

        # Compute probability distribution
        value_counts = column_clean.value_counts(normalize=True)

        # Compute entropy
        entropy_value = entropy(value_counts, base=base)

        return float(entropy_value)
    
    @staticmethod
    def unique_value_percentage(column: pd.Series) -> float:
        """
        Calculate the percentage of unique values in a given Pandas Series.

        Parameters:
        column (pd.Series): The column for which to calculate the uniqueness percentage.

        Returns:
        float: The percentage of unique values in the column.
        """
        total_values = len(column)
        unique_values = column.nunique()
        
        if total_values == 0:
            return 0.0
        
        return (unique_values / total_values) * 100

    @staticmethod
    def count_missing_values(column: pd.Series) -> int:
        """
        Count the number of missing (NaN) values in a given Pandas Series.

        Parameters:
        column (pd.Series): The column for which to count missing values.

        Returns:
        int: The number of missing values.
        """
        return column.isna().sum()
    
    @staticmethod
    def get_row_count(data: pd.DataFrame | pd.Series) -> int:
        """
        Get the number of rows in a Pandas DataFrame or Series.

        Parameters:
        data (pd.DataFrame | pd.Series): The DataFrame or Series for which to count rows.

        Returns:
        int: The number of rows.
        """
        return data.shape[0]
 