# THE CLASS SUPPORTS VARIOUS WAYS TO CALCULATE THE METRICS K-ANONYMITY, L-DIVERSITY, AND T-CLOSENESS
import numpy as np
import pandas as pd
import time
from metrics.divergence_metrics import calculate_emd, calculate_group_distribution, calculate_overall_distribution
from scipy.stats import wasserstein_distance as calculate_emd
from utils.field_name_utils import generate_field_name
from datetime import datetime


class AnonymityMetrics:

    # [1] CALCULATE K-ANONYMITY AND ADD NEW COLUMN
    @staticmethod
    def calculate_k_anonymity(df, quasi_identifiers_str, new_attribute_name="", threshold=3, print_flag=False):
        """
        Calculates k-anonymity for a given DataFrame and a set of quasi-identifiers.
        It adds a new column with the calculated k-anonymity value for each row.

        Args:
            df (DataFrame): The pandas DataFrame to calculate k-anonymity on.
            quasi_identifiers_str (str): String of quasi-identifiers separated by semicolons.
            new_attribute_name (str): Name of new column to be added to the dataframe
            threshold (int): The threshold value for k-anonymity.
            print_flag (bool): Whether to print the calculated k-anonymity and time

        Returns:
            DataFrame: Original DataFrame with an additional column for k-anonymity values.
            int: The minimum k-anonymity value across the DataFrame.
            int: Number of rows with k-anonymity below the threshold.
            float: Percentage of rows with k-anonymity below the threshold.
            :param df:
            :param print_flag:
            :param threshold:
            :param quasi_identifiers_str:
            :param new_attribute_name:
        """
        quasi_identifiers = quasi_identifiers_str.split(';')
        # Check new attribute name and calculate it if necessary
        if not new_attribute_name:
            new_attribute_name = generate_field_name(quasi_identifiers_str, prefix="k_", letters_count=1)

        # Calculation itself
        df[new_attribute_name] = df.groupby(quasi_identifiers)[quasi_identifiers[0]].transform('count')

        # k_min below is inverse re-identification probability
        k_min = df[new_attribute_name].min()
        count_below_threshold = (df[new_attribute_name] < threshold).sum()
        percent_below_threshold = (count_below_threshold / len(df)) * 100

        if print_flag:
            print(f"Minimum k-anonymity value: {k_min}")
            print(f"Number of rows with k-anonymity below {threshold}: {count_below_threshold}")
            print(f"Percentage of rows with k-anonymity below {threshold}: {percent_below_threshold:.2f}%")

        return df, k_min, count_below_threshold, percent_below_threshold

    # [2] GLOBAL K-ANONYMITY
    @staticmethod
    def calculate_global_k_anonymity(df, quasi_identifiers_str, stat_type='min'):
        """
        Calculates global k-anonymity values (min, mean, median) for the entire dataset based on the
        provided quasi-identifiers.

        Args:
            df (DataFrame): The pandas DataFrame to calculate k-anonymity on.
            quasi_identifiers_str (str): String of quasi-identifiers separated by semicolons.
            stat_type (str): Type of statistic to return. Options are 'min', 'mean', 'median'.
                - 'min': Minimum k-anonymity value, indicating the vulnerability of data to identification.
                         Lower values indicate higher identification risks.
                - 'mean': Mean k-anonymity value, indicating the average diversity of data across groups.
                          Higher values indicate greater data diversity.
                - 'median': Median k-anonymity value, indicating the typicality of data within the dataset.
                            Higher values indicate more similarity across the data.

        Returns:
            int or float: The calculated k-anonymity statistic for the dataset based on the chosen stat_type.
    """
        # Convert string of quasi-identifiers to a list
        quasi_identifiers = quasi_identifiers_str.split(';')

        # Group by quasi-identifiers and count the occurrences
        grouped = df.groupby(quasi_identifiers).size()

        # Calculate the requested statistic
        if stat_type == 'min':
            return grouped.min()
        elif stat_type == 'mean':
            return grouped.mean()
        elif stat_type == 'median':
            return grouped.median()
        else:
            raise ValueError("Invalid stat_type provided. Choose 'min', 'mean', or 'median'.")

    # [3] CALCULATE L-DIVERSITY AND ADD NEW COLUMN
    @staticmethod
    def calculate_l_diversity(df, quasi_identifiers_str, sensitive_columns_str, new_attribute_name="", threshold=2,
                              print_flag=False):
        """
        Calculate L-Diversity for a given DataFrame, returning numeric L-Diversity values for each group and adding
        them to a new column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be analyzed.
        - quasi_identifiers_str (str): Semicolon-separated string of column names to be used as quasi-identifiers.
        - sensitive_columns_str (str): Semicolon-separated string of column names considered sensitive.
        - new_attribute_name (str): The name for the new attribute where L-Diversity values will be stored. If empty,
          a name will be generated.
        - threshold (int): The minimum number of unique sensitive values required for L-Diversity.
        - print_flag (bool): If True, prints the time taken for execution and statistics below threshold.

        Returns:
        - pd.DataFrame: The original DataFrame with an additional column indicating the L-Diversity value for each row.
        - int: The count of groups not meeting the L-Diversity threshold.
        - float: The percentage of groups not meeting the L-Diversity threshold.
        """
        start_time = time.time()  # Start timing

        # Convert input strings to lists
        quasi_identifiers = quasi_identifiers_str.split(';')
        sensitive_columns = sensitive_columns_str.split(';')

        # Generate new attribute name if not provided
        if not new_attribute_name:
            new_attribute_name = generate_field_name(quasi_identifiers_str, prefix="l_", letters_count=1)

        # Group by quasi-identifiers and calculate the number of unique sensitive values for each group
        l_diversity_by_group = df.groupby(quasi_identifiers)[sensitive_columns].nunique().min(axis=1).rename(
            new_attribute_name)

        # Merge this numeric L-Diversity information back to the original DataFrame
        df = df.merge(l_diversity_by_group, left_on=quasi_identifiers, right_index=True, how='left')

        # Calculate statistics
        count_below_threshold = (df[new_attribute_name] < threshold).sum()
        percent_below_threshold = 100 * count_below_threshold / len(df)

        # Print execution time and statistics if requested
        if print_flag:
            execution_time = time.time() - start_time  # Calculate execution time
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Number of groups not meeting the L-Diversity threshold of {threshold}: {count_below_threshold}")
            print(f"Percentage of groups not meeting the L-Diversity threshold: {percent_below_threshold:.2f}%")

        return df, count_below_threshold, percent_below_threshold

    # [4] CALCULATION OF GLOBAL L-DIVERSITY
    @staticmethod
    def calculate_global_l_diversity(df, quasi_identifiers_str, sensitive_columns_str, statistic='min',
                                     print_flag=False):
        """
        Calculate global L-Diversity for a given DataFrame based on the specified statistic.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be analyzed.
        - quasi_identifiers_str (str): Semicolon-separated string of column names to be used as quasi-identifiers.
        - sensitive_columns_str (str): Semicolon-separated string of column names considered sensitive.
        - statistic (str): The statistic to use for calculating global L-Diversity ('min', 'mean', or 'median').
        - print_flag (bool): If True, prints the time taken for execution and the global L-Diversity value.

        Returns:
        - float: The global L-Diversity value for the dataset based on the chosen statistic.
        """

        start_time = time.time()  # Start timing

        # Convert input strings to lists
        quasi_identifiers = quasi_identifiers_str.split(';')
        sensitive_columns = sensitive_columns_str.split(';')

        # Group by quasi-identifiers and calculate the number of unique sensitive values for each group
        l_diversity_by_group = df.groupby(quasi_identifiers)[sensitive_columns].nunique().min(axis=1)

        # Calculate global L-Diversity based on the chosen statistic
        if statistic == 'min':
            global_l_diversity = l_diversity_by_group.min()
        elif statistic == 'mean':
            global_l_diversity = l_diversity_by_group.mean()
        elif statistic == 'median':
            global_l_diversity = l_diversity_by_group.median()
        else:
            raise ValueError("Invalid statistic. Choose 'min', 'mean', or 'median'.")

        # Print execution time and global L-Diversity if requested
        if print_flag:
            execution_time = time.time() - start_time  # Calculate execution time
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Global L-Diversity ({statistic}): {global_l_diversity}")

        return global_l_diversity

    # [5] CALCULATE T-CLOSENESS
    @staticmethod
    def calculate_t_closeness(df, quasi_identifiers_str, sensitive_column, threshold, new_column_name=None,
                              print_flag=False):
        """
        Calculate t-Closeness for a DataFrame and add a new column with the t-Closeness value for each group.
        Additionally, count and return the number of values below a specified threshold. Print progress for each
        group if print_flag is True.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be analyzed.
        - quasi_identifiers_str (str): Semicolon-separated string of column names to be used as quasi-identifiers.
        - sensitive_column (str): The name of the single sensitive column to analyze.
        - threshold (float): The threshold value to compare t-Closeness values against.
        - new_column_name (str, optional): The name of the new column to store t-Closeness values. If None, a name will
          be generated.
        - print_flag (bool): If True, prints execution time, progress, and stats about values below the threshold.

        Returns:
        - pd.DataFrame: The updated DataFrame with a new column for t-Closeness values.
        - int: The count of t-Closeness values below the threshold.
        - float: The percentage of t-Closeness values below the threshold.
        """

        start_time = datetime.now()

        # Adjust split method to use the correct delimiter
        quasi_identifiers = quasi_identifiers_str.split(';')  # Changed from '; ' to ';'
        # Optionally, strip whitespace around quasi-identifiers to ensure clean column names
        quasi_identifiers = [qid.strip() for qid in quasi_identifiers]

        if not new_column_name:
            new_column_name = 't_closeness'

        if print_flag:
            print("CALCULATE OVERALL DISTRIBUTION")
        overall_distribution = calculate_overall_distribution(df, sensitive_column)

        if print_flag:
            print("CALCULATE GROUP DISTRIBUTION")
        group_distribution = calculate_group_distribution(df, quasi_identifiers, sensitive_column)

        # Initialize the new column with default values
        df[new_column_name] = np.nan

        values_below_threshold = 0

        if print_flag:
            print("CALCULATE ROWS")

        for group, row in group_distribution.iterrows():
            group_emd = calculate_emd(row, overall_distribution)
            # Ensure that the dataframe is correctly indexed for the isin check
            if isinstance(group, (tuple, list)):
                group_index = pd.MultiIndex.from_tuples([group], names=quasi_identifiers)
            else:
                group_index = pd.Index([group], name=quasi_identifiers[0])

            df.loc[df.set_index(quasi_identifiers).index.isin(group_index), new_column_name] = group_emd

            if group_emd < threshold:
                values_below_threshold += len(df[df.set_index(quasi_identifiers).index.isin(group_index)])

            if print_flag:
                print(f'Group: {group}, EMD: {group_emd}')

        percent_below_threshold = (values_below_threshold / len(df)) * 100

        if print_flag:
            print(f'Execution time: {datetime.now() - start_time}')
            print(f'Values below threshold: {values_below_threshold}')
            print(f'Percent below threshold: {percent_below_threshold}%')

        return df, values_below_threshold, percent_below_threshold

