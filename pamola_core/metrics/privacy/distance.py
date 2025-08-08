"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Privacy Metric Operation - DistanceToClosestRecord
Package:       pamola_core.metrics.privacy
Version:       4.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       Mar 2025
Updated:       2025-07-16
License:       BSD 3-Clause

Description:
    This module implements the Distance to Closest Record (DCR) privacy metric,
    which quantifies the minimum distance from synthetic records to real records
    in a dataset. Lower DCR values indicate higher privacy risk, as synthetic data
    points are closer to original data.

    The implementation supports multiple distance metrics, feature normalization,
    aggregation strategies, and efficient nearest neighbor search using FAISS for
    large-scale datasets. It provides detailed risk assessment and interpretable
    privacy scores for synthetic data evaluation.

Key Features:
    - Custom DCR calculation with sklearn and FAISS support
    - Multiple distance metrics: Euclidean, Manhattan, Cosine, Mahalanobis
    - Feature normalization and robust aggregation (min, mean_k, percentile)
    - Percentile-based risk assessment and privacy scoring
    - Human-readable interpretation and recommendations
    - Seamless integration into PAMOLA.CORE metric pipelines
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from pamola_core.common.enum.distance_metric_type import DistanceMetricType
from pamola_core.metrics.commons.preprocessing import prepare_data_for_distance_metrics

try:
    import faiss
except ImportError:
    faiss = None


class DistanceToClosestRecord:
    """
    DCR measures minimum distance from synthetic to real records.

    Implementation: Custom using scipy.spatial with optimization and FAISS support
    for large-scale datasets.

    This operation implements the Distance to Closest Record (DCR) privacy metric,
    which measures how close synthetic records are to the original training data.
    Lower DCR values indicate higher privacy risk.

    Parameters:
    -----------
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'mahalanobis')
    normalize_features : bool, default=True
        Whether to normalize features using StandardScaler
    percentiles : List[int], default=[5, 25, 50, 75, 95]
        Percentiles to calculate for DCR distribution
    use_faiss : bool, default=False
        Use FAISS for large-scale nearest neighbor search
    aggregation : str, default='min'
        Aggregation method ('min', 'mean_k', 'percentile')
    k_neighbors : int, default=3
        Number of neighbors for mean_k aggregation
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        normalize_features: bool = True,
        percentiles: List[int] = [5, 25, 50, 75, 95],
        use_faiss: bool = False,
        aggregation: str = "min",
        k_neighbors: int = 3,
    ):
        """
        Initialize DistanceToClosestRecord metric with configuration options.

        Parameters:
        -----------
        distance_metric : str
            Distance metric to use ('euclidean', 'manhattan', 'cosine', 'mahalanobis').
        normalize_features : bool
            Whether to normalize features using StandardScaler.
        percentiles : List[int]
            Percentiles to calculate for DCR distribution.
        use_faiss : bool
            Use FAISS for large-scale nearest neighbor search.
        aggregation : str
            Aggregation method ('min', 'mean_k', 'percentile').
        k_neighbors : int
            Number of neighbors for mean_k aggregation.
        """
        self.distance_metric = distance_metric
        self.normalize_features = normalize_features
        self.percentiles = percentiles
        self.use_faiss = use_faiss
        self.aggregation = aggregation
        self.k_neighbors = k_neighbors

    def calculate_metric(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate the Distance to Closest Record (DCR) privacy metric and related statistics
        between two datasets (original and synthetic).

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original/real dataset. Each row is a record, columns are features.
        transformed_df : pd.DataFrame
            Transformed/synthetic dataset. Each row is a synthetic record, columns are features.

        Returns:
        --------
        Dict[str, Any]
            - metrics_dict (Dict[str, Any]): Dictionary containing DCR statistics, risk assessment, privacy score, and interpretation.
        """
        # Preprocess: convert all features to numeric using OrdinalEncoder for object/category columns
        orig_processed = prepare_data_for_distance_metrics(original_df)
        trans_processed = prepare_data_for_distance_metrics(transformed_df)

        # Normalize features if requested, otherwise use raw numeric values
        if self.normalize_features:
            scaler = StandardScaler().fit(orig_processed)
            orig_scaled = scaler.transform(orig_processed)
            trans_scaled = scaler.transform(trans_processed)
        else:
            orig_scaled = orig_processed.values
            trans_scaled = trans_processed.values

        # Calculate distances
        if self.use_faiss and faiss is not None:
            dcr_values = self._calculate_dcr_faiss(orig_scaled, trans_scaled)
        else:
            dcr_values = self._calculate_dcr_sklearn(orig_scaled, trans_scaled)

        # Compute DCR statistics: min, max, mean, std, and percentiles
        dcr_stats = {
            "min": float(np.min(dcr_values)),
            "max": float(np.max(dcr_values)),
            "mean": float(np.mean(dcr_values)),
            "std": float(np.std(dcr_values)),
        }

        # Add percentile statistics
        for p in self.percentiles:
            dcr_stats[f"p{p}"] = float(np.percentile(dcr_values, p))

        # Risk assessment: count records at high, medium, and low risk based on DCR percentiles
        risk_thresholds = {
            "high_risk": int(np.sum(dcr_values < np.percentile(dcr_values, 5))),
            "medium_risk": int(np.sum(dcr_values < np.percentile(dcr_values, 25))),
            "low_risk": int(np.sum(dcr_values >= np.percentile(dcr_values, 25))),
        }

        # Compile all results into a metrics dictionary
        metrics = {
            "dcr_statistics": dcr_stats,
            "risk_assessment": risk_thresholds,
            "proportion_at_risk": float(risk_thresholds["high_risk"] / len(dcr_values)),
            "privacy_score": float(
                self._calculate_privacy_score(
                    dcr_values, num_columns=original_df.shape[1]
                )
            ),
            "interpretation": self._interpret_dcr(dcr_stats),
        }

        return metrics

    def _calculate_dcr_faiss(
        self, orig_data: np.ndarray, trans_data: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Distance to Closest Record (DCR) using FAISS for efficient nearest neighbor search
        between two datasets (original and synthetic).

        Parameters:
        -----------
        orig_data : np.ndarray
            Array of original/real data (n_original, n_features).
        trans_data : np.ndarray
            Array of transformed/synthetic data (n_synthetic, n_features).

        Returns:
        --------
        np.ndarray
            Array of aggregated DCR distances for each synthetic record.
        """
        if faiss is None:
            # FAISS not available, fallback to sklearn
            return self._calculate_dcr_sklearn(
                orig_data.astype(np.float64), trans_data.astype(np.float64)
            )
        try:
            # Ensure data is float32 for FAISS
            orig_data = orig_data.astype(np.float32)
            trans_data = trans_data.astype(np.float32)

            # Build FAISS index
            dimension = orig_data.shape[1]

            if self.distance_metric == DistanceMetricType.EUCLIDEAN:
                index = faiss.IndexFlatL2(dimension)
            elif self.distance_metric == DistanceMetricType.COSINE:
                index = faiss.IndexFlatIP(dimension)
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(orig_data)
                faiss.normalize_L2(trans_data)
            else:
                # Fall back to scipy for other metrics
                return self._calculate_dcr_sklearn(orig_data, trans_data)

            # Add original data to index
            index.add(orig_data)

            # Determine number of neighbors to search
            if self.aggregation == "min":
                k = 1
            elif self.aggregation == "mean_k":
                k = min(self.k_neighbors, orig_data.shape[0])
            else:
                k = max(1, min(10, orig_data.shape[0] // 10))  # For percentile

            # Search for nearest neighbors
            distances, indices = index.search(trans_data, k)

            # For cosine, convert similarity to distance
            if self.distance_metric == "cosine":
                distances = 1.0 - distances
                distances = np.clip(distances, 0, 2)

            # Apply aggregation
            if self.aggregation == "min":
                dcr_values = distances[:, 0]
            elif self.aggregation == "mean_k":
                dcr_values = np.mean(distances, axis=1)
            elif self.aggregation == "percentile":
                dcr_values = np.percentile(distances, 10, axis=1)
            else:
                dcr_values = distances[:, 0]

            # Convert back from squared distances if L2
            if self.distance_metric == "euclidean":
                dcr_values = np.sqrt(np.maximum(dcr_values, 0))  # Ensure non-negative

            return dcr_values

        except Exception:
            return self._calculate_dcr_sklearn(
                orig_data.astype(np.float64), trans_data.astype(np.float64)
            )

    def _interpret_dcr(self, dcr_stats: Dict[str, float]) -> str:
        """
        Provide interpretation and privacy assessment of DCR results.

        Parameters:
        -----------
        dcr_stats : Dict[str, float]
            Dictionary containing DCR statistics (min, max, mean, percentiles, etc.).

        Returns:
        --------
        str
            Human-readable interpretation and privacy recommendation string.
        """
        mean_dcr = dcr_stats["mean"]
        min_dcr = dcr_stats["min"]
        p5_dcr = dcr_stats.get("p5", min_dcr)

        interpretation = []

        # Overall privacy assessment
        if mean_dcr > 1.0:
            interpretation.append(
                "Good privacy: Synthetic records are well-separated from original data."
            )
        elif mean_dcr > 0.5:
            interpretation.append(
                "Moderate privacy: Some synthetic records may be close to original data."
            )
        else:
            interpretation.append(
                "Poor privacy: Synthetic records are very close to original data."
            )

        # Minimum distance concern
        if min_dcr < 0.1:
            interpretation.append(
                "WARNING: At least one synthetic record is extremely close to an original record."
            )
        elif min_dcr < 0.3:
            interpretation.append(
                "CAUTION: Some synthetic records are quite close to original records."
            )

        # 5th percentile analysis
        if p5_dcr < 0.2:
            interpretation.append(
                "High risk: 5% of synthetic records have very small distances to original data."
            )

        # Recommendations
        if mean_dcr < 0.5:
            interpretation.append(
                "RECOMMENDATION: Consider increasing noise or regularization in data generation."
            )

        return " ".join(interpretation)

    def _calculate_dcr_sklearn(
        self, orig_data: np.ndarray, trans_data: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Distance to Closest Record (DCR) using sklearn pairwise distances
        between two datasets (original and synthetic).

        Parameters:
        -----------
        orig_data : np.ndarray
            Array of original/real data (n_original, n_features).
        trans_data : np.ndarray
            Array of transformed/synthetic data (n_synthetic, n_features).

        Returns:
        --------
        np.ndarray
            Array of aggregated DCR distances for each synthetic record.
        """
        # Validate data dimensions
        if orig_data.shape[1] != trans_data.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: original {orig_data.shape[1]}, "
                f"transformed {trans_data.shape[1]}"
            )

        # Remove rows with NaN values in either orig_data or trans_data
        orig_data = orig_data[~np.isnan(orig_data).any(axis=1)]
        trans_data = trans_data[~np.isnan(trans_data).any(axis=1)]

        # If either is empty after removing NaNs, raise error
        if orig_data.shape[0] == 0 or trans_data.shape[0] == 0:
            raise ValueError("No data left after removing rows with NaN values.")

        # Calculate pairwise distances
        distances = pairwise_distances(
            trans_data, orig_data, metric=self.distance_metric
        )

        # Apply aggregation strategy
        if self.aggregation == "min":
            dcr_values = np.min(distances, axis=1)
        elif self.aggregation == "mean_k":
            # Average of k closest distances
            k = min(self.k_neighbors, distances.shape[1])
            sorted_distances = np.sort(distances, axis=1)
            dcr_values = np.mean(sorted_distances[:, :k], axis=1)
        elif self.aggregation == "percentile":
            # Use 10th percentile as more robust measure
            dcr_values = np.percentile(distances, 10, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return dcr_values

    def _calculate_privacy_score(
        self,
        dcr_values: np.ndarray,
        num_columns: int,
    ) -> float:
        """
        Convert the DCR distribution to a privacy score in the range [0, 1].
        Higher scores indicate better privacy (synthetic records are farther from original records).

        Parameters:
        -----------
        dcr_values : np.ndarray
            Array of DCR values for synthetic records.
        columns : Optional[List[str]]
            List of columns used for dimensionality normalization (optional).

        Returns:
        --------
        float
            Privacy score between 0 and 1 (higher = better privacy).
        """
        if len(dcr_values) == 0:
            return 0.0

        # Normalize by data dimensionality
        dimension_factor = np.sqrt(num_columns)

        # Apply robust normalization
        normalized_dcr = np.clip(dcr_values / np.sqrt(dimension_factor), 0, 10)

        # Use sigmoid-like transformation (1 - exp(-x)) for better interpretability
        # Values close to 0 -> score near 0, larger values -> score approaches 1
        scores = 1 - np.exp(-normalized_dcr)

        return float(np.mean(scores))
