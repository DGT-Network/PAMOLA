# DATA PERTURBATION IS A DATA DE-IDENTIFICATION TOOL THAT MODIFIES THE INITIAL DATASET MARGINALLY BY APPLYING
# ROUND-NUMBERING METHODS AND ADDING RANDOM NOISE2. THE SET OF VALUES MUST BE PROPORTIONAL TO THE DISTURBANCE. A SMALL
# BASE CAN CONTRIBUTE TO POOR ANONYMIZATION, WHILE A BROAD BASE CAN REDUCE A DATASET’S UTILITY

import pandas as pd
import numpy as np
from tqdm import tqdm
import time


class Perturbation:
    def __init__(self):
        pass

    @staticmethod
    def check_data_type_consistency(column, expected_type):
        for value in column[:3]:
            if not isinstance(value, expected_type):
                return False
        return True

    def numeric_randomization(self, df, column_name, data_type, blur_radius, blur_type='percentage', print_flag=False,
                              lower_bound=None, upper_bound=None):
        global expected_type
        start_time = time.time()

        # Determine the correct Python data type based on the input data type
        if data_type in ['integer', 'floating', 'money']:
            if data_type == 'integer':
                expected_type = (int, np.integer)
            elif data_type in ['floating', 'money']:
                expected_type = (float, np.floating)
        else:
            raise ValueError("Unsupported data type")

        # Check data type consistency
        if not self.check_data_type_consistency(df[column_name], expected_type):
            raise ValueError("Data type inconsistency found in the first three rows")

        original_mean = df[column_name].mean()
        original_std = df[column_name].std()

        def apply_randomization(x):
            if pd.isnull(x) or (lower_bound is not None and x < lower_bound) or (
                    upper_bound is not None and x > upper_bound):
                base_value = np.random.uniform(lower_bound,
                                               upper_bound) if lower_bound is not None and upper_bound is not None else \
                df[column_name].mean()
            else:
                base_value = x

            if blur_type == 'absolute':
                noise = np.random.uniform(-blur_radius, blur_radius)
            else:  # percentage
                noise = np.random.uniform(-base_value * blur_radius, base_value * blur_radius)

            randomized_value = base_value + noise
            if lower_bound is not None and randomized_value < lower_bound:
                randomized_value = lower_bound + abs(noise)
            if upper_bound is not None and randomized_value > upper_bound:
                randomized_value = upper_bound - abs(noise)
            return randomized_value

        if print_flag:
            tqdm.pandas(desc="Numeric Randomization")
            df[column_name] = df[column_name].progress_apply(apply_randomization)
        else:
            df[column_name] = df[column_name].apply(apply_randomization)

        new_mean = df[column_name].mean()
        new_std = df[column_name].std()

        # Calculate deltas
        mean_delta = new_mean - original_mean
        std_delta = new_std - original_std

        # Print information if print_flag is true
        if print_flag:
            end_time = time.time()
            print(f"Processing completed. Time taken: {end_time - start_time:.2f} seconds")
            print(f"Records processed: {len(df)}")
            print(f"Mean change: {mean_delta}, Standard deviation change: {std_delta}")

        return df, mean_delta, std_delta
