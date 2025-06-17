"""
PAMOLA.CORE - L-Diversity Processor
----------------------------------------------------
Advanced processor for l-diversity anonymization, ensuring compliance with privacy-preserving techniques.
Designed for scalable, efficient anonymization with modular architecture.

(C) 2024 Realm Inveo Inc., DTX, Titan Technology, & DGT Network Inc.
Licensed under BSD 3-Clause License

## Module Purpose
Provides flexible and performance-optimized data privacy transformations with a configurable and extensible architecture.

## Features:
- **Diversity Calculation**: Distinct, entropy, and recursive (c,l)-diversity.
- **Adaptive l-levels**: Granular group-specific anonymization.
- **Performance Optimized**: NumPy-powered vectorization, Dask integration, parallel processing.
- **Scalability**: Dynamic configuration overrides, horizontal scaling support, memory efficiency.
- **Logging & Monitoring**: Configurable logging, error tracking, progress visualization with `tqdm`.
- **Modular Design**:
  - `calculation.py`: Pamola Core diversity calculations.
  - `apply_model.py`: Model application and transformation.
  - `metrics.py`: Privacy, utility, and fidelity metric assessments.
  - `privacy.py`: Risk and privacy evaluations.
  - `reporting.py`: Compliance and documentation report generation.
  - `visualization.py`: Diversity assessment visualization tools.

## Use Cases:
- Privacy-preserving data anonymization for regulated environments.
- GDPR, HIPAA, CCPA compliance.
- Secure large-scale dataset anonymization for research and analytics.

## TODO:
- Extend adaptive performance scaling.
- Expand metadata-driven anonymization policies.
- Implement additional compliance audit mechanisms.
"""


import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from dask import dataframe as dd

from pamola_core.config import L_DIVERSITY_DEFAULTS
# PAMOLA pamola core imports
from pamola_core.privacy_models.base import BasePrivacyModelProcessor
from pamola_core.utils__old_15_04.group_processing import (
    validate_anonymity_inputs
)
from pamola_core.utils__old_15_04.progress import progress_logger

class LDiversityCalculator(BasePrivacyModelProcessor):
    """
    Advanced L-Diversity Calculation Processor with Centralized Results Storage
    """

    def __init__(
            self,
            l: int = 3,
            diversity_type: str = 'distinct',
            c_value: float = 1.0,
            k: int = 2,
            config_override: Optional[Dict[str, Any]] = None,
            use_dask: bool = False,
            log_level: str = 'INFO',
            adaptive_l: Optional[Dict[Tuple, int]] = None
    ):
        """
        Initialize L-Diversity Processor
        """
        # Initialize configuration
        self.config = dict(L_DIVERSITY_DEFAULTS)
        if config_override:
            self.config.update(config_override)

        # Pamola Core parameters
        self.l = l
        self.diversity_type = diversity_type
        self.c_value = c_value
        self.k = k
        self.use_dask = use_dask

        # Adaptive l-levels support
        self.adaptive_l = adaptive_l or {}

        # Configure logging
        self._setup_logging(log_level)

        # Validate configuration
        self._check_configuration()

        # Centralized results storage
        self._results_cache = {}

    def _setup_logging(self, log_level: str):
        """
        Configure logging with flexible verbosity

        Parameters:
        -----------
        log_level : str
            Logging level (e.g., 'INFO', 'DEBUG', 'WARNING')
        """
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _check_configuration(self):
        """
        Validate configuration parameters

        Raises:
        -------
        ValueError
            If configuration parameters are invalid
        """
        valid_types = ['distinct', 'entropy', 'recursive']
        if self.diversity_type not in valid_types:
            raise ValueError(f"Invalid diversity type. Must be one of {valid_types}")

        if self.l < 1 or self.k < 1:
            raise ValueError("l and k values must be at least 1")

    def _get_adaptive_l(self, group_key: Tuple) -> int:
        """
        Retrieve adaptive l-level for specific group

        Parameters:
        -----------
        group_key : Tuple
            Quasi-identifier group key

        Returns:
        --------
        int
            Adaptive l-level (falls back to default if not specified)
        """
        return self.adaptive_l.get(group_key, self.l)

    def calculate_group_diversity(
            self,
            data: Union[pd.DataFrame, dd.DataFrame],
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            force_recalculate: bool = False
    ) -> pd.DataFrame:
        """
        Calculate diversity metrics with centralized caching

        Parameters:
        -----------
        data : Union[pd.DataFrame, dd.DataFrame]
            Input dataset
        quasi_identifiers : List[str]
            Quasi-identifier columns
        sensitive_attributes : List[str]
            Sensitive attribute columns
        force_recalculate : bool, optional
            Force recalculation even if results exist in cache

        Returns:
        --------
        pd.DataFrame
            Group diversity metrics
        """

        # Create a unique cache key
        cache_key = (
            tuple(quasi_identifiers),
            tuple(sensitive_attributes),
            self.diversity_type
        )

        # Check cached results
        if not force_recalculate and cache_key in self._results_cache:
            return self._results_cache[cache_key]

        # Input validation
        validate_anonymity_inputs(data, quasi_identifiers, self.k)

        # Convert to Dask if required
        if self.use_dask and not isinstance(data, dd.DataFrame):
            data = dd.from_pandas(data, npartitions=4)

        # Group data processing with progress tracking
        grouped = data.groupby(quasi_identifiers)

        # Vectorized processing with NumPy
        diversity_metrics = []

        try:
            progress_bar = progress_logger("Calculating Group Diversity", len(grouped))
            for group_name, group_data in grouped:
                try:
                    # Use adaptive l-level
                    group_key = (group_name if isinstance(group_name, tuple) else (group_name,))
                    adaptive_l = self._get_adaptive_l(group_key)

                    # Process group with NumPy-optimized calculations
                    group_metrics = self._process_group_diversity_vectorized(
                        group_name,
                        group_data,
                        quasi_identifiers,
                        sensitive_attributes,
                        adaptive_l
                    )
                    diversity_metrics.append(group_metrics)
                    progress_bar.update(1)

                except Exception as e:
                    self.logger.warning(f"Error processing group {group_name}: {e}")
            progress_bar.close()
        except Exception as e:
            self.logger.error(f"Error during group diversity calculation: {e}")
            raise

        # Convert to DataFrame
        result = pd.DataFrame(diversity_metrics)

        # Store in centralized cache
        self._results_cache[cache_key] = result

        return result

    # Redirecting methods to dedicated modules
    def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: List[str],
                         sensitive_attributes: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluates privacy risks for the dataset using l-diversity principles.

        This method redirects to the privacy module while providing cached
        diversity calculations for efficiency.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : List[str]
            Columns used as quasi-identifiers
        sensitive_attributes : List[str], optional
            Sensitive attribute columns
        **kwargs : dict
            Additional risk assessment parameters

        Returns:
        --------
        Dict[str, Any]
            Comprehensive privacy risk metrics
        """
        # Import privacy module implementation
        from pamola_core.privacy_models.l_diversity.privacy import LDiversityPrivacyRiskAssessor

        # Create a risk assessor with self as processor for cache access
        risk_assessor = LDiversityPrivacyRiskAssessor(processor=self)

        # Ensure sensitive_attributes is a list
        if sensitive_attributes is None:
            # Try to find potential sensitive attributes (non-quasi-identifiers)
            sensitive_attributes = [col for col in data.columns if col not in quasi_identifiers]

        # Assess privacy risks using cached diversity calculations
        return risk_assessor.assess_privacy_risks(
            data, quasi_identifiers, sensitive_attributes,
            diversity_type=self.diversity_type,
            l_threshold=self.l,
            c_value=self.c_value,
            **kwargs
        )

    def apply_model(self, data: pd.DataFrame, quasi_identifiers: List[str], suppression: bool = True,
                    **kwargs) -> pd.DataFrame:
        """
        Redirects to the actual model application implementation.
        """
        return apply_model_impl(data, quasi_identifiers, suppression, **kwargs)

    def _process_group_diversity_vectorized(
            self,
            group_name: Union[Tuple, Any],
            group_data: Union[pd.DataFrame, dd.DataFrame],
            quasi_identifiers: List[str],
            sensitive_attributes: List[str],
            adaptive_l: int
    ) -> Dict[str, Any]:
        """
        Vectorized group diversity processing with NumPy

        Parameters:
        -----------
        adaptive_l : int
            Adaptive l-level for this specific group
        """
        # Initialize group metrics
        group_metrics = {}

        # Add quasi-identifier values
        group_metrics.update(
            {qi: group_name[i] if isinstance(group_name, tuple) else group_name
             for i, qi in enumerate(quasi_identifiers)}
        )

        # NumPy-optimized calculations for each sensitive attribute
        for sa in sensitive_attributes:
            # Vectorized calculations
            sa_values = group_data[sa].values

            # Distinct values
            distinct_values = len(np.unique(sa_values))
            group_metrics[f"{sa}_distinct"] = distinct_values

            # Entropy calculation
            if self.diversity_type == 'entropy':
                unique_values, counts = np.unique(sa_values, return_counts=True)
                probabilities = counts / len(sa_values)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                group_metrics[f"{sa}_entropy"] = entropy

            # Recursive diversity
            if self.diversity_type == 'recursive':
                unique_values, counts = np.unique(sa_values, return_counts=True)
                # Sort in descending order and take top adaptive_l
                sorted_indices = np.argsort(counts)[::-1]
                top_counts = counts[sorted_indices[:adaptive_l]]

                group_metrics[f"{sa}_recursive"] = (
                    self._check_recursive_diversity_numpy(top_counts, self.c_value, adaptive_l)
                )

        # Add group size
        group_metrics['group_size'] = len(group_data)

        return group_metrics

    def _check_recursive_diversity_numpy(
            self,
            value_counts: np.ndarray,
            c_value: float,
            l_threshold: int
    ) -> bool:
        """
        NumPy-optimized recursive diversity check

        Parameters:
        -----------
        value_counts : np.ndarray
            Counts of top values
        c_value : float
            Recursive diversity parameter
        l_threshold : int
            Minimum threshold for diversity

        Returns:
        --------
        bool
            Whether the group satisfies recursive diversity
        """
        if len(value_counts) < l_threshold:
            return False

        most_frequent = value_counts[0]
        least_frequent_sum = np.sum(value_counts[1:])

        return most_frequent <= c_value * least_frequent_sum