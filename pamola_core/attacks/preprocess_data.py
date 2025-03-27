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

NOTE: This module requires 'numpy' and 'scikit-learn' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pamola_core.attacks.base import AttackInitialization



class PreprocessData(AttackInitialization):
    """
    Base class for attack simulation in PAMOLA.CORE.
    This class extends AttackInitialization and define methods used for attack simulation.
    """

    def __init__(self):
        pass


    def preprocess_data(self, data1, data2):
        """
        Data preprocessing: Use TF-ID to convert all string elements of data1 and data2 to numbers

        Parameters:
        -----------
        data1: First dataset
        data2: Second dataset

        Returns:
        -----------
        data1_final: The dataset contains numeric values corresponding to data1
        data2_final: The dataset contains numeric values corresponding to data2
        """

        # Get all string columns from both datasets
        all_cat_cols = list(set(data1.select_dtypes(exclude=[np.number]).columns) |
                            set(data2.select_dtypes(exclude=[np.number]).columns))

        # Process strings using TF-IDF
        data1_text = data1.reindex(columns=all_cat_cols, fill_value="").astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
        data2_text = data2.reindex(columns=all_cat_cols, fill_value="").astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(data1_text + data2_text)

        data1_tfidf = vectorizer.transform(data1_text).toarray()
        data2_tfidf = vectorizer.transform(data2_text).toarray()

        # Get all numeric columns from both datasets
        all_num_cols = list(set(data1.select_dtypes(include=[np.number]).columns) |
                            set(data2.select_dtypes(include=[np.number]).columns))

        data1_num = data1.reindex(columns=all_num_cols, fill_value=0).to_numpy()
        data2_num = data2.reindex(columns=all_num_cols, fill_value=0).to_numpy()

        # Check if there is numeric data or not
        if data1_num.shape[1] == 0:
            data1_final = data1_tfidf
        else:
            data1_final = np.hstack((data1_tfidf, data1_num))

        if data2_num.shape[1] == 0:
            data2_final = data2_tfidf
        else:
            data2_final = np.hstack((data2_tfidf, data2_num))

        return data1_final, data2_final


    def process(self, data):
        """
        Process the input data.

        Parameters:
        -----------
        data : Any
            The input data to be processed.

        Returns:
        --------
        Processed data, transformed according to the specific processor logic.
        """
        
        print("class PreprocessData: Override the abstract process() method of the parent class BaseProcessor here")