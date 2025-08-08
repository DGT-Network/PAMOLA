"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Privacy Metric Operation - NearestNeighborDistanceRatio
Package:       pamola_core.metrics.privacy
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-07-17
License:       BSD 3-Clause

Description:
    This module implements the Nearest Neighbor Distance Ratio (NNDR) privacy metric,
    which quantifies the ratio of distances from synthetic records to their nearest
    and second-nearest real records. Lower NNDR values indicate higher privacy risk,
    as synthetic data points are much closer to a single real record than to others.

Key Features:
    - Custom NNDR calculation using sklearn.neighbors
    - Multiple distance metrics: Euclidean, Manhattan, Cosine
    - Feature normalization and configurable neighbor count
    - Threshold-based risk assessment and privacy classification
    - Human-readable interpretation and recommendations
    - Seamless integration into PAMOLA.CORE metric pipelines
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pamola_core.metrics.commons.preprocessing import prepare_data_for_distance_metrics


class NearestNeighborDistanceRatio:
    """
    NNDR measures ratio of distances to nearest and second-nearest neighbors.

    Parameters:
    - distance_metric: str ('euclidean', 'manhattan', 'cosine')
    - n_neighbors: int (default: 2, for NNDR)
    - normalize_features: bool
    - threshold: float (NNDR threshold for privacy risk)
    - columns: list of columns to use (optional)

    Formula:
    NNDR_i = dist(x_i^s, x_1^r) / dist(x_i^s, x_2^r)
    where x_1^r and x_2^r are first and second nearest neighbors

    Example:
        nndr = NearestNeighborDistanceRatio(distance_metric='euclidean', normalize_features=True, threshold=0.5)
        results = nndr.calculate_metric(original_df, transformed_df)
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        n_neighbors: int = 2,
        normalize_features: bool = False,
        threshold: float = 0.5,
        realistic_threshold: float = 0.8,
        at_risk_threshold: float = 0.5,
    ):
        """
        Initialize NearestNeighborDistanceRatio metric parameters.

        Args:
            distance_metric (str): Distance metric ('euclidean', 'manhattan', 'cosine').
            n_neighbors (int): Number of neighbors to consider (default: 2).
            normalize_features (bool): Whether to normalize features before calculation.
            threshold (float): NNDR threshold for high privacy risk.
            realistic_threshold (float): Threshold for realistic privacy classification.
            at_risk_threshold (float): Threshold for at-risk privacy classification.
        """
        self.distance_metric = distance_metric
        self.n_neighbors = n_neighbors
        self.normalize_features = normalize_features
        self.threshold = threshold
        self.realistic_threshold = realistic_threshold
        self.at_risk_threshold = at_risk_threshold

    def calculate_metric(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate the Nearest Neighbor Distance Ratio (NNDR) privacy metric between two datasets.

        Args:
            original_df (pd.DataFrame): Original/real dataset.
            transformed_df (pd.DataFrame): Transformed/synthetic dataset.
        Returns:
            Dict[str, Any]: NNDR statistics, risk assessment, privacy classification, and interpretation.
        """

        # Preprocess: convert all features to numeric using OrdinalEncoder for object/category columns
        orig_processed = prepare_data_for_distance_metrics(original_df)
        trans_processed = prepare_data_for_distance_metrics(transformed_df)

        # Normalize features if requested, otherwise use raw numeric values
        # StandardScaler ensures all features have mean=0 and std=1, improving distance calculations
        if self.normalize_features:
            scaler = StandardScaler().fit(orig_processed)
            orig_data = scaler.transform(orig_processed)
            trans_data = scaler.transform(trans_processed)
        else:
            orig_data = orig_processed.values
            trans_data = trans_processed.values

        # Fit nearest neighbors model
        nn_model = NearestNeighbors(
            n_neighbors=2, metric=self.distance_metric, algorithm="auto"
        ).fit(orig_data)

        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(trans_data)

        # Calculate NNDR
        nndr_values = distances[:, 0] / (
            distances[:, 1] + 1e-10
        )  # Avoid division by zero

        # Statistics
        nndr_stats = {
            "mean": np.mean(nndr_values),
            "std": np.std(nndr_values),
            "min": np.min(nndr_values),
            "max": np.max(nndr_values),
        }

        # Risk assessment based on NNDR
        # Lower NNDR indicates higher privacy risk (record is much closer to one real record)
        high_risk_count = int(np.sum(nndr_values < self.threshold))

        results = {
            "nndr_statistics": nndr_stats,
            "high_risk_records": high_risk_count,
            "nndr_values": nndr_values.tolist(),
            "high_risk_proportion": high_risk_count / len(nndr_values),
            "privacy_classification": {
                "realistic": np.sum(
                    nndr_values > self.realistic_threshold
                ),  # Close to 1
                "at_risk": np.sum(nndr_values < self.at_risk_threshold),  # Close to 0
            },
            "interpretation": self._interpret_nndr(nndr_stats, high_risk_count),
        }

        return results

    def _interpret_nndr(self, nndr_stats: dict, high_risk_count: int) -> str:
        """
        Provide interpretation and privacy assessment of NNDR results.

        Args:
            nndr_stats (dict): NNDR statistics (mean, std, min, max).
            high_risk_count (int): Number of records classified as high privacy risk.
        Returns:
            str: Human-readable interpretation and privacy recommendation.
        """
        if high_risk_count == 0:
            return "No records are at high privacy risk based on NNDR threshold."
        elif nndr_stats["mean"] < self.threshold:
            return "Many records are close to real data, privacy risk is high."
        else:
            return "Most records are sufficiently distant from real data, privacy risk is low."
