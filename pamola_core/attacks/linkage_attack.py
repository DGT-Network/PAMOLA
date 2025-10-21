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

from typing import Optional
import numpy as np
import pandas as pd
import recordlinkage
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from pamola_core.attacks.preprocess_data import PreprocessData
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


class LinkageAttack(PreprocessData):
    """
    LinkageAttack class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods linkage attack for attack simulation.
    """

    def __init__(
        self, fs_threshold=None, n_components=2, logger: Optional[logging.Logger] = None
    ):
        """
        The constructor of the LinkageAttack class

        Parameters:
        -----------
        fs_threshold: Threshold used for Fellegi-Sunter model (probabilistic_linkage_attack)
        n_components: Dimensionality reduction using PCA (cluster_vector_linkage_attack)
        """

        self.fs_threshold = fs_threshold
        self.n_components = n_components
        self.logger = logger or logging.get_logger(__name__)

    def record_linkage_attack(self, data1, data2, linkage_keys):
        """
        The record linkage attack function uses directly compare common columns of 2 datasets to find pairs of match records

        Parameters:
        -----------
        data1: First dataset
        data2: Second dataset
        linkage_keys: List of properties to compare

        Returns:
        -----------
        DataFrame contains pairs of records that matching between two datasets
        """

        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")

        # Reset index cleanly to avoid index column collision
        df1 = data1.reset_index(drop=True)
        df2 = data2.reset_index(drop=True)

        # Determine linkage keys if not provided
        if linkage_keys is None:
            linkage_keys = list(set(df1.columns) & set(df2.columns))

        if not linkage_keys:
            self.logger.warning("No common linkage keys found between datasets.")
            return pd.DataFrame()

        # Merge on linkage keys
        linked_records = df1.merge(
            df2, on=linkage_keys, how="inner", suffixes=("_data1", "_data2")
        )

        # Optional: warn about many-to-many matches
        if linked_records.duplicated(subset=linkage_keys, keep=False).any():
            self.logger.warning(
                "Multiple matches detected for some linkage keys (1-n or n-m). "
                "Review linkage quality."
            )

        return linked_records

    def probabilistic_linkage_attack(self, data1, data2, keys=None):
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
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")

        # Determine keys if not provided
        if keys is None:
            keys = list(set(data1.columns) & set(data2.columns))
        if not keys:
            return pd.DataFrame()

        # Keep only relevant columns
        data1 = data1[keys].copy().astype(str).fillna("")
        data2 = data2[keys].copy().astype(str).fillna("")

        # Remove rows where all key values are empty
        mask1 = data1.replace("", np.nan).notna().any(axis=1)
        mask2 = data2.replace("", np.nan).notna().any(axis=1)
        data1 = data1[mask1]
        data2 = data2[mask2]

        # Create candidate pairs via blocking
        indexer = recordlinkage.Index()
        for key in keys:
            indexer.block(key)
        candidate_links = indexer.index(data1, data2)

        # Fellegi–Sunter comparison
        compare = recordlinkage.Compare()
        for key in keys:
            compare.string(key, key, method="jarowinkler", label=key)
        features = compare.compute(candidate_links, data1, data2)

        # Compute normalized similarity score
        features["similarity_score"] = features.sum(axis=1) / len(keys)

        # Determine threshold if not provided
        if self.fs_threshold is None:
            self.fs_threshold = 0.85

        # Filter matches
        matches = features[
            features["similarity_score"] >= self.fs_threshold
        ].reset_index()

        # Optional logging
        self.logger.info(
            f"Probabilistic linkage: {len(candidate_links)} candidate pairs, {len(matches)} matches found"
        )

        return matches

    def cluster_vector_linkage_attack(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        similarity_threshold: float = 0.8,
    ) -> pd.DataFrame:
        """
        Cluster-Vector Linkage Attack (CVPLA) using TruncatedSVD & Cosine Similarity
        to link records between two datasets based on their latent vector representations.

        Parameters
        ----------
        data1 : pd.DataFrame
            First dataset (e.g., target dataset)
        data2 : pd.DataFrame
            Second dataset (e.g., external dataset)
        similarity_threshold : float, optional (default=0.8)
            Minimum cosine similarity score to consider a match.

        Returns
        -------
        matches : pd.DataFrame
            DataFrame with columns:
            - ID_DF1: index in data1
            - ID_DF2: index in data2
            - Score: cosine similarity score
        """
        # --- 1. Validate input ---
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        if data1.empty or data2.empty:
            return pd.DataFrame(columns=["ID_DF1", "ID_DF2", "Score"])

        # --- 2. Preprocess data (TF-IDF + numeric) ---
        data1_vec, data2_vec = self.preprocess_data(data1, data2)

        # --- 3. Normalize ---
        def normalize(X):
            return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        data1_vec = normalize(data1_vec)
        data2_vec = normalize(data2_vec)

        # --- 4. Dimensionality reduction ---
        n_components = min(self.n_components, data1_vec.shape[1], data2_vec.shape[1])
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        data1_reduced = reducer.fit_transform(data1_vec)
        data2_reduced = reducer.transform(data2_vec)

        # --- 5. Cosine similarity ---
        similarity_matrix = cosine_similarity(data1_reduced, data2_reduced)

        # --- 6. Find best matches for each record in data2 ---
        best_idx_in_data1 = np.argmax(similarity_matrix, axis=0)
        best_scores = similarity_matrix[best_idx_in_data1, np.arange(len(data2))]

        # --- 7. Build matches DataFrame ---
        matches = pd.DataFrame(
            {
                "ID_DF1": data1.index[best_idx_in_data1],
                "ID_DF2": data2.index,
                "Score": best_scores,
            }
        )

        # --- 8. Filter by similarity threshold ---
        matches = matches[matches["Score"] >= similarity_threshold]

        # --- 9. Resolve duplicate ID_DF1 by keeping highest score ---
        matches = matches.sort_values("Score", ascending=False)
        matches = matches.drop_duplicates(subset=["ID_DF1"], keep="first").reset_index(
            drop=True
        )

        return matches
