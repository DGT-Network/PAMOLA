# A DIVERGENCE METRIC IS A WAY OF MEASURING THE SIMILARITY OR DISSIMILARITY BETWEEN TWO PROBABILITY DISTRIBUTIONS OR
# OTHER OBJECTS. UNLIKE A REGULAR METRIC, A DIVERGENCE METRIC DOES NOT HAVE TO BE SYMMETRIC OR SATISFY THE TRIANGLE
# INEQUALITY. A DIVERGENCE METRIC CAN BE USEFUL FOR COMPARING DIFFERENT MODELS, FEATURES, POLICIES, OR DATA SETS IN
# VARIOUS FIELDS OF DATA ANALYSIS AND MACHINE LEARNING. SOME EXAMPLES OF DIVERGENCE METRICS ARE: EMD, Kullback-Leibler
# divergence, Jensen-Shannon divergence, Wasserstein distance
import numpy as np
import pandas as pd
from tqdm import tqdm


# VECTORIZED EARTH MOVER'S DISTANCE (EMD) CALCULATION
# The function calculates the Earth Mover’s Distance (EMD) between the two distributions, which is a measure of
# dissimilarity between them. The EMD is computed by summing the absolute differences between the corresponding
# elements of the two arrays, and dividing the result by 2. The EMD is also known as the Wasserstein metric or
# the Kantorovich-Rubinstein metric. The two distributions that you want to compare using the EMD function should have
# the same size, meaning that they should have the same number of elements. This is because the EMD function calculates
# the absolute difference between the corresponding elements of the two arrays.
def calculate_emd(distribution1, distribution2):
    """
    Calculate the Earth Mover's Distance (EMD) between two distributions.
    """
    return np.sum(np.abs(distribution1 - distribution2)) / 2


# CALCULATION OF GROUP DISTRIBUTION
def calculate_group_distribution(df, group_columns, sensitive_column):
    """
    Calculate the distribution of the sensitive attribute within each quasi-identifier group.
    """
    # Ensure sensitive_column is in a numeric format for efficient processing
    if np.issubdtype(df[sensitive_column].dtype, np.datetime64):
        df[sensitive_column] = df[sensitive_column].apply(lambda x: x.toordinal())

    # Initialize an empty DataFrame to store distributions
    distributions = pd.DataFrame()

    # Use tqdm to show progress over the unique combinations of quasi-identifiers
    unique_groups = df[group_columns].drop_duplicates()
    for _, group_values in tqdm(unique_groups.iterrows(), total=unique_groups.shape[0]):
        # Filter df for the current group
        mask = np.ones(len(df), dtype=bool)
        for col in group_columns:
            mask &= df[col] == group_values[col]
        group_df = df[mask]

        # Calculate distribution for the current group
        value_counts = group_df[sensitive_column].value_counts(normalize=True)
        value_counts.name = tuple(group_values)  # Using group values as index name

        # Append the distribution
        distributions = distributions.append(value_counts)

    # Reformat distributions DataFrame for output
    distribution_output = distributions.unstack(level=-1, fill_value=0).T
    return distribution_output


def calculate_overall_distribution(df, sensitive_column):
    """
    Calculate the overall distribution of the sensitive attribute across the entire dataset.
    """
    if np.issubdtype(df[sensitive_column].dtype, np.datetime64):
        df[sensitive_column] = df[sensitive_column].apply(lambda x: x.toordinal())

    overall_distribution = df[sensitive_column].value_counts(normalize=True)
    return overall_distribution
