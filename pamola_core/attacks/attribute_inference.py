"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Attack Simulation
-----------------------
This module provides an abstract base class for attack simulation feature
in PAMOLA.CORE. It defines the general structure and required methods for
implementing specific attack simulation

NOTE: This module requires 'numpy' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
import pandas as pd
from pamola_core.attacks.preprocess_data import PreprocessData


class AttributeInference(PreprocessData):
    """
    AttributeInference class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods attribute inference attack for attack simulation.
    """

    def __init__(self):
        pass

    def attribute_inference_attack(self, data_train, data_test, target_attribute):
        """
        The function performs an Attribute Inference Attack
        This is an attack that attempts to infer sensitive attributes of an individual (data point)

        Parameters:
        -----------
        data_train: The training dataset, containing attributes of the entities, includes the target_column attribute whose value is known.
        data_test: The test dataset, containing attributes of the entities, includes target_column attribute whose value is unknown and needs to be inferred
        target_column: attribute of the data_test dataset needs to be inferred

        Returns:
        -----------
        List contains the predicted values of target_attribute
        """

        # Validate inputs
        if data_train.empty or data_test.empty:
            return pd.Series(dtype=object)

        if target_attribute not in data_train.columns:
            raise KeyError(
                f"Target attribute '{target_attribute}' not found in training data."
            )

        features = [col for col in data_train.columns if col != target_attribute]
        if not features:
            raise ValueError("No features available for inference.")

        # Compute entropy for each candidate feature
        entropy_values = {}
        for column in features:
            counts = data_train[column].value_counts(dropna=True)
            probs = counts / counts.sum()
            entropy_values[column] = -(probs * np.log2(probs)).sum()

        # Select the feature with the lowest entropy
        best_feature = min(entropy_values, key=entropy_values.get)

        # Build mapping: feature_value -> majority target
        mapping = data_train.groupby(best_feature)[target_attribute].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.dropna().iloc[0]
        )

        # Infer on test
        predictions = data_test[best_feature].map(mapping)
        global_mode = data_train[target_attribute].mode().iloc[0]
        predictions = predictions.fillna(global_mode)

        return predictions
