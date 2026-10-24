"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
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
    large-scale datasets. The result is a descriptive technical diagnostic of distances;
    it contains no score and no verdict, and no value of it is a probability of
    re-identification.

    Feature space: numeric, datetime and declared-ordinal columns are standardized with one
    fit on the original; nominal categorical columns are one-hot encoded and weighted so that
    a mismatch of any two categories is one unit of Euclidean distance (see
    ``prepare_pair_for_distance_metrics``). All backends receive the same vectors.

Key Features:
    - Custom DCR calculation with sklearn and FAISS support
    - Multiple distance metrics: Euclidean, Manhattan, Cosine, Mahalanobis
    - Feature normalization and robust aggregation (min, mean_k, percentile)
    - Descriptive distance statistics; counts below caller-supplied thresholds (no built-in risk classes)
    - Neutral descriptive text (no verdicts or recommendations)
    - Seamless integration into PAMOLA.CORE metric pipelines
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances_chunked

from pamola_core.common.enum.distance_metric_type import DistanceMetricType
from pamola_core.metrics.commons.preprocessing import (
    prepare_pair_for_distance_metrics,
    scale_pair_for_distance,
)
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.exceptions import (
    DataError,
    InvalidParameterError,
    ResourceError,
    ValidationError,
)

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

# --- Backend selection and resource limits -----------------------------------
#
# The exact distance matrix is n_synthetic x n_original float64 values: ~80 GB at
# 100K x 100K and ~7 TiB at 1M x 1M. Building it whole was the default, so 100K
# rows ended in the OS OOM killer. The exact backend now works in row blocks (peak
# memory is bounded by the working-memory budget below, whatever n is) and refuses,
# with a typed error, work it could not finish in a sane time.

#: Row-block budget, in MiB, for one block of the exact distance matrix. A block
#: plus the temporaries of the aggregation (partition / sort copy) stays within a
#: small multiple of this.
_WORKING_MEMORY_MIB = 256

#: Above this many (original x synthetic) pairs the automatic choice (``use_faiss
#: =None``) moves from the exact search to FAISS - when FAISS gives the same
#: answer for the requested metric and aggregation (see ``_FAISS_EXACT``).
#: 10K x 10K, the size at which the metrics SRS switches to FAISS.
_AUTO_FAISS_MIN_COMPARISONS = 100_000_000

#: Default ceilings on (original x synthetic) pairs, per backend. Beyond them the
#: operation returns a typed limit error instead of running for hours. Measured on
#: 10 features: the block-wise exact search does ~2e7 pairs/s (100K x 100K = 1e10
#: pairs in ~8 min), FAISS ~1.5e9 pairs/s (100K x 100K in ~6 s). Raise the limit
#: with ``max_comparisons`` if the run time is acceptable.
_DEFAULT_MAX_COMPARISONS = {
    "sklearn": 20_000_000_000,
    "faiss": 1_000_000_000_000,
}

#: (metric, aggregation) pairs for which ``_calculate_dcr_faiss`` returns the same
#: values as ``_calculate_dcr_sklearn`` (to float32 rounding), verified by
#: ``tests/metrics/privacy/test_distance_backends.py``. FAISS searches only the k <= 10
#: nearest neighbours, so ``percentile`` (a percentile of ALL distances of a row) is not
#: reproducible with it: ``_calculate_dcr_faiss`` falls back to the exact search for it
#: (visibly, via ``backend_fallback``), and automatic selection never chooses FAISS for it.
_FAISS_EXACT = frozenset(
    {
        ("euclidean", "min"),
        ("euclidean", "mean_k"),
        ("cosine", "min"),
        ("cosine", "mean_k"),
    }
)


def reject_unsupported_distance_metric(metric: Any, metric_name: str) -> None:
    """DCR and NNDR do not support the Mahalanobis distance in 1.0 (string or enum)."""
    value = metric.value if isinstance(metric, DistanceMetricType) else str(metric)
    if value.lower() == "mahalanobis":
        raise InvalidParameterError(
            param_name="distance_metric",
            param_value=value,
            reason=(
                f"the Mahalanobis distance is not supported for {metric_name} in 1.0: the "
                "one-hot features of nominal columns make the covariance singular. Use "
                "'euclidean', 'manhattan' or 'cosine'."
            ),
        )


class DistanceToClosestRecord:
    """
    DCR measures minimum distance from synthetic to real records.

    Implementation: Custom using scipy.spatial with optimization and FAISS support
    for large-scale datasets.

    This operation implements the Distance to Closest Record (DCR) privacy metric,
    which measures how close synthetic records are to the original training data.
    Lower DCR values mean the synthetic records lie closer to original records; the value is a
    distance in the feature space described above, not a probability.

    Parameters
    -----------
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'cosine'); 'mahalanobis' is not
        supported in 1.0
    normalize_features : bool, default=True
        Whether to standardize the non-nominal features (one StandardScaler fit on the original);
        one-hot columns of nominal categoricals are never standardized, they carry a fixed weight
    ordinal_columns : sequence of str or dict, optional
        Categorical columns whose categories have an order (encoded as ranks); all other
        categorical columns are nominal
    percentiles : List[int], default=[5, 25, 50, 75, 95]
        Percentiles to calculate for DCR distribution
    use_faiss : bool or None, default=None
        ``None`` (default) chooses the backend from the input size: the exact
        search for small inputs, FAISS above 10K x 10K record pairs when FAISS
        returns the same values for the requested metric and aggregation
        (euclidean/cosine ``min`` and cosine ``mean_k``), and the exact search
        evaluated in row blocks otherwise. ``True`` forces FAISS where it is
        available; ``False`` forces the exact search.
    aggregation : str, default='min'
        Aggregation method ('min', 'mean_k', 'percentile')
    k_neighbors : int, default=3
        Number of neighbors for mean_k aggregation
    max_comparisons : int, optional
        Largest number of (original x synthetic) record pairs the metric will
        evaluate. Larger inputs raise ``ResourceError`` before any distance is
        computed instead of exhausting memory or running for hours. Defaults to
        2e10 for the exact search (about 140K x 140K) and 1e12 for FAISS.
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        normalize_features: bool = True,
        percentiles: List[int] = [5, 25, 50, 75, 95],
        use_faiss: Optional[bool] = None,
        aggregation: str = "min",
        k_neighbors: int = 3,
        max_comparisons: Optional[int] = None,
        distance_thresholds: Optional[List[float]] = None,
        ordinal_columns: Optional[Any] = None,
    ):
        """
        Initialize DistanceToClosestRecord metric with configuration options.

        Parameters
        -----------
        distance_metric : str
            Distance metric: 'euclidean', 'manhattan' or 'cosine'. 'mahalanobis' (string or
            ``DistanceMetricType``) is not supported for DCR in 1.0 and raises
            ``InvalidParameterError`` here, before any computation.
        normalize_features : bool
            Whether to normalize features using StandardScaler.
        percentiles : List[int]
            Percentiles to calculate for DCR distribution.
        use_faiss : bool or None
            None selects the backend by input size, True forces FAISS, False
            forces the exact search (see the class docstring).
        aggregation : str
            Aggregation method ('min', 'mean_k', 'percentile').
        k_neighbors : int
            Number of neighbors for mean_k aggregation.
        max_comparisons : int or None
            Ceiling on original x synthetic record pairs; None uses the
            per-backend default.
        distance_thresholds : List[float] or None
            Distances (in the units of the normalized feature space) chosen by the caller.
            When given, the result reports how many records lie below each of them
            (``threshold_assessment``). The library applies no threshold of its own.
        ordinal_columns : sequence of str or dict, optional
            Categorical columns whose categories have an order, encoded as ranks (see
            ``prepare_pair_for_distance_metrics``). Every other categorical column is nominal:
            one-hot encoded so that all mismatches are equally far apart.
        """
        reject_unsupported_distance_metric(distance_metric, "DCR")
        self.distance_metric = distance_metric
        self.normalize_features = normalize_features
        self.percentiles = percentiles
        self.use_faiss = use_faiss
        self.aggregation = aggregation
        self.k_neighbors = k_neighbors
        self.max_comparisons = max_comparisons
        self.distance_thresholds = (
            None if distance_thresholds is None else [float(t) for t in distance_thresholds]
        )
        self.ordinal_columns = ordinal_columns
        self._backend_fallback: Optional[Dict[str, str]] = None

    def _metric_name(self) -> str:
        metric = self.distance_metric
        return metric.value if isinstance(metric, DistanceMetricType) else str(metric)

    def _select_backend(self, n_original: int, n_transformed: int) -> str:
        """Choose ``"faiss"`` or ``"sklearn"`` from the configuration and the size.

        Deterministic: depends only on the arguments, the configuration and
        whether FAISS is importable. The exact (sklearn) backend never builds the
        full matrix, so it is a safe answer for every combination FAISS cannot
        reproduce.
        """
        if self.use_faiss is True:
            return "faiss" if faiss is not None else "sklearn"
        if self.use_faiss is False or faiss is None:
            return "sklearn"
        if (
            (self._metric_name(), self.aggregation) in _FAISS_EXACT
            and n_original * n_transformed > _AUTO_FAISS_MIN_COMPARISONS
        ):
            return "faiss"
        return "sklearn"

    def _check_size_limit(
        self, n_original: int, n_transformed: int, n_features: int, backend: str
    ) -> None:
        """Raise ``ResourceError`` when the pair count exceeds the backend's limit."""
        limit = (
            self.max_comparisons
            if self.max_comparisons is not None
            else _DEFAULT_MAX_COMPARISONS[backend]
        )
        pairs = n_original * n_transformed
        if pairs <= limit:
            return
        raise ResourceError(
            operation="DistanceToClosestRecord",
            reason=(
                f"DCR would evaluate {pairs:,} record pairs ({n_original:,} original "
                f"x {n_transformed:,} synthetic records, {n_features} features), above "
                f"the limit of {limit:,} for the {backend} backend. Compute it on a random "
                "sample of the records, or raise `max_comparisons` if the run time "
                "is acceptable. For euclidean or cosine distance with "
                "aggregation='min', `use_faiss=True` is several times faster."
            ),
        )

    def calculate_metric(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate the Distance to Closest Record (DCR) privacy metric and related statistics
        between two datasets (original and synthetic).

        Parameters
        -----------
        original_df : pd.DataFrame
            Original/real dataset. Each row is a record, columns are features.
        transformed_df : pd.DataFrame
            Transformed/synthetic dataset. Each row is a synthetic record, columns are features.

        Returns
        --------
        Dict[str, Any]
            - metrics_dict (Dict[str, Any]): Dictionary containing DCR statistics (and threshold counts when thresholds are supplied), backend and interpretation.
        """
        # One shared schema for both datasets: numeric/datetime/ordinal columns standardized with
        # one fit, nominal categoricals one-hot encoded with a fixed weight (a mismatch = 1).
        orig_processed, trans_processed = prepare_pair_for_distance_metrics(
            original_df, transformed_df, ordinal_columns=self.ordinal_columns
        )
        orig_scaled, trans_scaled = scale_pair_for_distance(
            orig_processed, trans_processed, self.normalize_features
        )

        # Calculate distances
        self._backend_fallback = None
        backend = self._select_backend(len(orig_scaled), len(trans_scaled))
        selected_backend = backend
        self._check_size_limit(
            len(orig_scaled), len(trans_scaled), orig_scaled.shape[1], backend
        )
        if backend == "faiss":
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

        # Compile all results into a metrics dictionary. No self-relative "risk" classes: a
        # percentile of the distribution itself always flags the same share of records.
        metrics = {
            "dcr_statistics": dcr_stats,
            "backend": (
                self._backend_fallback["to"] if self._backend_fallback else selected_backend
            ),
        }
        if self._backend_fallback:
            metrics["backend_fallback"] = dict(self._backend_fallback)
        if self.distance_thresholds:
            metrics["threshold_assessment"] = {
                "thresholds": list(self.distance_thresholds),
                "records_below": {
                    str(t): int(np.sum(dcr_values < t)) for t in self.distance_thresholds
                },
                "proportion_below": {
                    str(t): float(np.mean(dcr_values < t)) for t in self.distance_thresholds
                },
            }
        metrics["interpretation"] = self._interpret_dcr(dcr_stats, metrics.get("threshold_assessment"))

        return metrics

    def _calculate_dcr_faiss(
        self, orig_data: np.ndarray, trans_data: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Distance to Closest Record (DCR) using FAISS for efficient nearest neighbor search
        between two datasets (original and synthetic).

        Parameters
        -----------
        orig_data : np.ndarray
            Array of original/real data (n_original, n_features).
        trans_data : np.ndarray
            Array of transformed/synthetic data (n_synthetic, n_features).

        Returns
        --------
        np.ndarray
            Array of aggregated DCR distances for each synthetic record.
        """
        if faiss is None:
            return self._fall_back("faiss is not installed", orig_data, trans_data)
        if self.aggregation == "percentile":
            return self._fall_back(
                "FAISS searches only the nearest neighbours, so the percentile of all "
                "distances is not reproducible with it",
                orig_data,
                trans_data,
            )
        original_64, transformed_64 = orig_data, trans_data  # the exact path gets full precision
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
                return self._fall_back(
                    f"FAISS does not support the '{self._metric_name()}' metric",
                    original_64,
                    transformed_64,
                )

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

            # FAISS returns SQUARED L2 distances: take the root of every neighbour before any
            # aggregation, so the mean over neighbours is the mean of distances (as in the
            # exact search), not the root of the mean of squares
            if self.distance_metric == "euclidean":
                distances = np.sqrt(np.maximum(distances, 0))

            # For cosine, convert similarity to distance
            if self.distance_metric == "cosine":
                distances = 1.0 - distances
                distances = np.clip(distances, 0, 2)

            # Apply aggregation
            if self.aggregation == "min":
                dcr_values = distances[:, 0]
            elif self.aggregation == "mean_k":
                dcr_values = np.mean(distances, axis=1)
            else:
                dcr_values = distances[:, 0]

            return dcr_values

        except Exception as exc:  # noqa: BLE001 - reported in the result, not hidden
            return self._fall_back(
                f"{type(exc).__name__}: {str(exc)[:200]}", original_64, transformed_64
            )

    def _fall_back(self, reason: str, orig_data: np.ndarray, trans_data: np.ndarray) -> np.ndarray:
        """Compute with the exact search instead of FAISS and say so (one warning, and the
        result carries ``backend_fallback: {from, to, reason}``)."""
        self._backend_fallback = {"from": "faiss", "to": "sklearn", "reason": reason}
        logger.warning("DCR: FAISS backend not used (%s); computing with the exact search", reason)
        return self._calculate_dcr_sklearn(
            orig_data.astype(np.float64), trans_data.astype(np.float64)
        )

    def _interpret_dcr(
        self, dcr_stats: Dict[str, float], threshold_assessment: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Describe the DCR distribution in neutral terms.

        The text states the numbers; it does not classify them as good or poor and gives no
        recommendation (the library issues no verdicts - the reader applies their own criteria).
        """
        parts = [
            "Distance to the closest original record, in the units of the "
            f"{'normalized ' if self.normalize_features else ''}feature space: "
            f"mean {dcr_stats['mean']:.4g}, min {dcr_stats['min']:.4g}"
            + (f", 5th percentile {dcr_stats['p5']:.4g}" if "p5" in dcr_stats else "")
            + f", max {dcr_stats['max']:.4g}."
        ]
        if threshold_assessment:
            counts = ", ".join(
                f"{n} below {t}" for t, n in threshold_assessment["records_below"].items()
            )
            parts.append(f"Records below the supplied thresholds: {counts}.")
        else:
            parts.append("No distance threshold was supplied, so no record is classified.")
        return " ".join(parts)

    def _calculate_dcr_sklearn(
        self, orig_data: np.ndarray, trans_data: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Distance to Closest Record (DCR) using sklearn pairwise distances
        between two datasets (original and synthetic).

        Parameters
        -----------
        orig_data : np.ndarray
            Array of original/real data (n_original, n_features).
        trans_data : np.ndarray
            Array of transformed/synthetic data (n_synthetic, n_features).

        Returns
        --------
        np.ndarray
            Array of aggregated DCR distances for each synthetic record.
        """
        # Validate data dimensions
        if orig_data.shape[1] != trans_data.shape[1]:
            raise ValidationError(
                f"Feature dimension mismatch: original {orig_data.shape[1]}, "
                f"transformed {trans_data.shape[1]}"
            )

        # Remove rows with NaN values in either orig_data or trans_data
        orig_data = orig_data[~np.isnan(orig_data).any(axis=1)]
        trans_data = trans_data[~np.isnan(trans_data).any(axis=1)]

        # If either is empty after removing NaNs, raise error
        if orig_data.shape[0] == 0 or trans_data.shape[0] == 0:
            raise DataError(
                message="No data left after removing rows with NaN values.",
                error_code=ErrorCode.DATA_EMPTY,
            )

        if self.aggregation not in ("min", "mean_k", "percentile"):
            raise InvalidParameterError(
                param_name="aggregation",
                param_value=self.aggregation,
                reason=f"Unknown aggregation method: {self.aggregation}",
            )

        # Guard here as well as in calculate_metric: this method is also reached
        # directly and as the fallback of the FAISS path.
        self._check_size_limit(
            orig_data.shape[0], trans_data.shape[0], orig_data.shape[1], "sklearn"
        )

        n_original = orig_data.shape[0]
        k = min(self.k_neighbors, n_original)

        def aggregate(block: np.ndarray, _start: int) -> np.ndarray:
            """Reduce one row block of the distance matrix to one value per row."""
            if self.aggregation == "min":
                return np.min(block, axis=1)
            if self.aggregation == "mean_k":
                # Average of the k closest distances. Partition first (O(n) per
                # row), then sort only those k so the sum runs in the same order
                # as a full sort and the result is identical to it.
                if 1 <= k < n_original:
                    nearest = np.partition(block, k - 1, axis=1)[:, :k]
                    nearest = np.sort(nearest, axis=1)
                else:
                    nearest = np.sort(block, axis=1)[:, :k]
                return np.mean(nearest, axis=1)
            # Use 10th percentile as more robust measure
            return np.percentile(block, 10, axis=1)

        # Row blocks of the (synthetic x original) distance matrix: the values are
        # those of the dense matrix, the peak memory is the block budget.
        blocks = pairwise_distances_chunked(
            trans_data,
            orig_data,
            metric=self.distance_metric,
            reduce_func=aggregate,
            working_memory=_WORKING_MEMORY_MIB,
        )
        return np.concatenate(list(blocks))

