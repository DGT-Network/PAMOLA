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

NOTE: This module requires 'numpy', 'pandas', 'recordlinkage' and 'scikit-learn' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
import pandas as pd
import recordlinkage
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from pamola_core.attacks.preprocess_data import PreprocessData



class LinkageAttack(PreprocessData):
    """
    LinkageAttack class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods linkage attack for attack simulation.
    """

    def __init__(self, fs_threshold = None, n_components = 2):
        """
        The constructor of the LinkageAttack class

        Parameters:
        -----------
        fs_threshold: Threshold used for Fellegi-Sunter model (probabilistic_linkage_attack)
        n_components: Dimensionality reduction using PCA (cluster_vector_linkage_attack)
        """

        self.fs_threshold = fs_threshold
        self.n_components = n_components


    def record_linkage_attack(self, data1, data2, keys):
        """
        The record linkage attack function uses directly compare common columns of 2 datasets to find pairs of match records

        Parameters:
        -----------
        data1: First dataset
        data2: Second dataset
        keys: List of properties to compare

        Returns:
        -----------
        DataFrame contains pairs of records that matching between two datasets
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
    
        data1_reset = data1.reset_index()
        data2_reset = data2.reset_index()

        # If keys = None -> find common columns between 2 datasets
        if keys is None:
            keys = list(set(data1_reset.columns) & set(data2_reset.columns))

        if not keys:
            return pd.DataFrame()
    
        matches = data1_reset.merge(data2_reset, on=keys, how='inner')

        return matches


    def probabilistic_linkage_attack(self, data1, data2, keys = None):
        """
        The probabilistic linkage attack function uses the Fellegi-Sunter model to compare and link records from two datasets

        Parameters:
        -----------
        data1: First dataset
        data2: Second dataset
        keys: List of properties use to compare

        Returns:
        -----------
        - DataFrame containing pairs of records matching the Fellegi-Sunter score between two datasets
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        
        # Ensure changes on data1 and data2 only occur on the copy. Converting all data to string and replace all null values to empty
        data1 = data1.copy().astype(str).fillna("")
        data2 = data2.copy().astype(str).fillna("")

        # Initialize Indexer to create comparison pairs
        indexer = recordlinkage.Index()

        # Compare all possible pairs
        # indexer.full()

        # Compare only records with the same value on the keys
        for key in keys:
            indexer.block(key)

        candidate_links = indexer.index(data1, data2)

        # Compare attributes using Fellegi-Sunter algorithm
        compare = recordlinkage.Compare()
        for key in keys:
            compare.string(key, key, method='jarowinkler', label=key)

        features = compare.compute(candidate_links, data1, data2)

        # Calculate the sum of the Fellegi-Sunter scores (Score) for each pair
        features['Score'] = features.sum(axis=1)

        # If fs_threshold is None, determine fs_threshold based on len(keys) * 0.85
        if self.fs_threshold is None:
            self.fs_threshold = len(keys) * 0.85

        # Filter pairs with score greater than or equal to fs_threshold
        matches = features[features['Score'] >= self.fs_threshold].reset_index()

        return matches


    def cluster_vector_linkage_attack(self, data1, data2):
        """
        Cluster-Vector Linkage Attack (CVPL) function using PCA & Cosine Similarity to compare and link records from two datasets

        Parameters:
        -----------
        data1: First dataset
        data2: Second dataset

        Returns:
        -----------
        DataFrame containing pairs of records matching PCA score & Cosine Similarity between two datasets
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        
        data1_transform, data2_transform = self.preprocess_data(data1, data2)

        # Limit n_components to avoid PCA errors
        self.n_components = min(self.n_components, data1_transform.shape[1], data2_transform.shape[1])

        # Normalize data by column (axis=0 -> column, 1e-8 -> avoid error when std = 0)
        data1_transform = (data1_transform - data1_transform.mean(axis=0)) / (data1_transform.std(axis=0) + 1e-8)
        data2_transform = (data2_transform - data2_transform.mean(axis=0)) / (data2_transform.std(axis=0) + 1e-8)

        # Apply PCA to reduce data dimensionality
        pca = PCA(n_components=self.n_components)
        data1_pca = pca.fit_transform(data1_transform)
        data2_pca = pca.transform(data2_transform)

        # Calculate Cosine Similarity for two datasets data1 and data2
        similarity_matrix = cosine_similarity(data1_pca, data2_pca)

        # For each sample in data2, find the sample in data1 with the highest similarity
        best_matches = np.argmax(similarity_matrix, axis=0)

        # Retrieve original IDs directly from data1 and data2 index
        id_df1 = data1.index[best_matches].tolist()
        id_df2 = data2.index.tolist()

        scores = similarity_matrix[best_matches, np.arange(len(data2))]

        matches = pd.DataFrame({
            'ID_DF1': id_df1,
            'ID_DF2': id_df2,
            'Score': scores.round(6)
        })

        return matches