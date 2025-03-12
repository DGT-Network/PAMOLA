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
from pamola.core.attack_simulation.preprocess_data import PreprocessData



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

        # Check that the datasets are valid
        if data_train is None or data_test is None:
            raise ValueError("Input datasets cannot be None.")
        
        entropy_values = {}
        for column in data_train.columns:
            if column != target_attribute:
                _, counts = np.unique(data_train[column], return_counts=True)
                probs = counts / len(data_train[column])
                # Ensure we avoid log2(0) which is undefined
                entropy_values[column] = -np.sum([p * np.log2(p) for p in probs if p > 0])

        # Find the best feature with minimum entropy
        best_feature = min(entropy_values, key=entropy_values.get)

        # Inference based on the best feature
        prediction_values = data_train.groupby(best_feature)[target_attribute].apply(
            lambda x: x.mode()[0] if not x.mode().empty else np.random.choice(x))

        # Predict values based on inferred values from the best feature
        prediction_values = data_test[best_feature].map(prediction_values).fillna(data_train[target_attribute].mode()[0])

        # Return only the predicted values
        return prediction_values