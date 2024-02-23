# CALCULATE ENTROPY METRIC FOR DATASET
import pandas as pd
import numpy as np
import time


# CALCULATE SHANON ENTROPY
def calculate_entropy(df, column_names_str):
    """
    Calculate the entropy for a given column or a group of columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - column_names_str (str): The name of the column or a group of columns separated by ";" for which to calculate the entropy.

    Returns:
    - float: The entropy value for the specified column or group of columns.
    """
    # Splitting the column names and creating a "virtual" column for group values if there are multiple columns
    column_names = column_names_str.split(';')
    if len(column_names) > 1:
        # Concatenate values from the specified columns to treat them as a single entity
        group_values = df[column_names].astype(str).agg(';'.join, axis=1)
    else:
        group_values = df[column_names[0]]

    # Calculate the probabilities of each unique value
    probabilities = group_values.value_counts(normalize=True).values

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


# CALCULATE RENYI ENTROPY
def calculate_renyi_entropy(df, column_names_str, alpha=2):
    """
    Calculate the Rényi entropy of a given column or a group of columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - column_names_str (str): The name of the column or a group of columns separated by ";" for which to calculate the entropy.
    - alpha (float): The order of Rényi entropy. Alpha > 0, alpha != 1. For alpha=2, it's called the collision entropy.

    Returns:
    - float: The Rényi entropy value for the specified column or group of columns.

    Note:
    - When alpha approaches 1, Rényi entropy converges to Shannon entropy.
    """
    # Ensure alpha is valid
    if alpha <= 0 or alpha == 1:
        raise ValueError("Alpha must be > 0 and != 1.")

    # Splitting the column names and creating a "virtual" column for group values if there are multiple columns
    column_names = column_names_str.split(';')
    if len(column_names) > 1:
        # Concatenate values from the specified columns to treat them as a single entity
        group_values = df[column_names].astype(str).agg(';'.join, axis=1)
    else:
        group_values = df[column_names[0]]

    # Calculate the probabilities of each unique value
    probabilities = group_values.value_counts(normalize=True).values

    # Calculate Rényi entropy
    if alpha > 1:
        renyi_entropy = 1 / (1 - alpha) * np.log2(np.sum(probabilities ** alpha))
    else:  # Handling the limit as alpha approaches 1 (Shannon entropy)
        renyi_entropy = -np.sum(probabilities * np.log2(probabilities))

    return renyi_entropy


# CALCULATE CONDITIONAL ENTROPY
def calculate_conditional_entropy(df, primary_fields_str, conditional_fields_str, print_flag=False):
    """
    Calculate the conditional entropy of one group of fields given another group of fields in a DataFrame using an
    iterative approach.

    This function computes quasi-identifiers for primary and conditional fields by concatenating their values. Then, it
    iteratively calculates the joint distribution of these quasi-identifiers and uses it to compute the conditional
    entropy. If print_flag is set, it prints intermediate values and execution time.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - primary_fields_str (str): The name(s) of the primary column(s), separated by ";", for which to calculate the entropy.
    - conditional_fields_str (str): The name(s) of the conditional column(s), separated by ";", used for the condition.
    - print_flag (bool): If True, prints execution time and intermediate values for groups.

    Returns:
    - float: The conditional entropy value.
    """
    start_time = time.time()

    # Combine and convert the specified columns into a single "virtual" column for each group
    primary_keys = df[primary_fields_str.split(';')].astype(str).agg(';'.join, axis=1)
    conditional_keys = df[conditional_fields_str.split(';')].astype(str).agg(';'.join, axis=1)

    # Create a DataFrame for joint probabilities
    joint_df = pd.DataFrame({'Primary': primary_keys, 'Conditional': conditional_keys})
    joint_probs = joint_df.groupby(['Primary', 'Conditional']).size() / len(df)

    # Marginal probabilities for conditional values
    marginal_probs_conditional = conditional_keys.value_counts(normalize=True)

    # Calculate conditional entropy iteratively
    conditional_entropy = 0

    for tuple_ in joint_probs.reset_index().itertuples():
        # Extract Index, Quasi-identifiers, conditional fields, probability
        index = tuple_[0]
        primary = tuple_[1]
        conditional = tuple_[2]
        prob = tuple_[3]

        # Conditional probability P(Primary|Conditional)
        conditional_prob = prob / marginal_probs_conditional[conditional]
        entropy_contribution = -prob * np.log2(conditional_prob)
        conditional_entropy += entropy_contribution

        if print_flag:
            print(
                f"Processed group {index}: Primary={primary}, Conditional={conditional}, "
                f"Contribution={entropy_contribution:.4f}")

    if print_flag:
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")

    return conditional_entropy

