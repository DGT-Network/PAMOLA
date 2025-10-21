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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pamola_core.attacks.base import AttackInitialization


class PreprocessData(AttackInitialization):
    """
    Data preprocessing utilities for attack simulation modules in PAMOLA.CORE.
    Converts categorical and numeric columns into numeric feature matrices
    using TF-IDF and StandardScaler.
    """

    def __init__(self):
        pass

    def preprocess_data(
        self, data1: pd.DataFrame, data2: pd.DataFrame, max_features: int = 5000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert categorical + numeric columns of two DataFrames into numeric arrays.

        Parameters
        ----------
        data1 : pd.DataFrame
            First dataset (typically used as the "reference" or "train" set).
            - Used to fit the TF-IDF vocabulary (categorical)
            - Used to fit the StandardScaler (numeric)
        data2 : pd.DataFrame
            Second dataset (typically used as the "target" or "test" set).
            - Transformed using the vocabulary/scaler learned from `data1`.
        max_features : int, optional
            Maximum number of features for the TF-IDF vectorizer (default = 5000).
            Higher values capture more unique categorical tokens, but may increase
            computation and memory cost.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - data1_final : np.ndarray
                Feature matrix representing `data1` after TF-IDF + scaling.
            - data2_final : np.ndarray
                Feature matrix representing `data2` after TF-IDF + scaling.
        """
        data1_tfidf, data2_tfidf = self._vectorize_categorical(
            data1, data2, max_features
        )
        data1_num, data2_num = self._scale_numeric(data1, data2)

        # Combine categorical (TF-IDF) + numeric (scaled) feature matrices
        data1_final = np.hstack([data1_tfidf, data1_num])
        data2_final = np.hstack([data2_tfidf, data2_num])
        return data1_final, data2_final

    def _vectorize_categorical(
        self, data1: pd.DataFrame, data2: pd.DataFrame, max_features: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert categorical columns of two DataFrames into TF-IDF vectors.

        Parameters
        ----------
        data1 : pd.DataFrame
            First dataset (used to fit the TF-IDF vocabulary).
        data2 : pd.DataFrame
            Second dataset (transformed using the vocabulary from `data1`).
        max_features : int
            Maximum size of the TF-IDF vocabulary.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - TF-IDF matrix for `data1`
            - TF-IDF matrix for `data2`
        """
        cat_cols = list(
            set(data1.select_dtypes(exclude=[np.number]).columns)
            | set(data2.select_dtypes(exclude=[np.number]).columns)
        )

        def row_to_text(row):
            # Represent each row as "col:value col:value ..." to reduce collisions
            return " ".join(f"{col}:{val}" for col, val in row.items())

        if cat_cols:
            # Ensure both datasets have the same categorical columns (fill missing with "")
            data1_cat = data1.reindex(columns=cat_cols, fill_value="").astype(str)
            data2_cat = data2.reindex(columns=cat_cols, fill_value="").astype(str)

            data1_text = data1_cat.apply(row_to_text, axis=1).tolist()
            data2_text = data2_cat.apply(row_to_text, axis=1).tolist()

            vectorizer = TfidfVectorizer(max_features=max_features)
            vectorizer.fit(data1_text)  # Fit only on data1 (reference)

            return (
                vectorizer.transform(data1_text).toarray(),
                vectorizer.transform(data2_text).toarray(),
            )
        else:
            # No categorical columns
            return np.zeros((len(data1), 0)), np.zeros((len(data2), 0))

    def _scale_numeric(
        self, data1: pd.DataFrame, data2: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Standardize numeric columns of two DataFrames.

        Parameters
        ----------
        data1 : pd.DataFrame
            First dataset (used to fit the StandardScaler).
        data2 : pd.DataFrame
            Second dataset (transformed using the scaler from `data1`).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - Scaled numeric matrix for `data1`
            - Scaled numeric matrix for `data2`
        """
        num_cols = list(
            set(data1.select_dtypes(include=[np.number]).columns)
            | set(data2.select_dtypes(include=[np.number]).columns)
        )

        if num_cols:
            data1_num = data1.reindex(columns=num_cols, fill_value=0).to_numpy()
            data2_num = data2.reindex(columns=num_cols, fill_value=0).to_numpy()

            scaler = StandardScaler().fit(data1_num)
            return scaler.transform(data1_num), scaler.transform(data2_num)
        else:
            return np.zeros((len(data1), 0)), np.zeros((len(data2), 0))

    def process(self, data):
        """
        Placeholder for compatibility with the PAMOLA pipeline.

        Parameters
        ----------
        data : Any
            The input data to be processed.

        Returns
        -------
        Any
            This method currently only prints a message; use `preprocess_data()`
            for actual data transformations.
        """
        print(
            "[PreprocessData] No direct process implemented; use preprocess_data() instead."
        )
