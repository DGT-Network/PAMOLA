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

Module: Correlated Noise Addition Processor
--------------------------------------------
This module provides an implementation of correlated noise addition for
anonymization-preserving data transformations. It extends the BaseNoiseAdditionProcessor
and allows noise injection while maintaining attribute relationships.

Correlated noise addition ensures that statistical dependencies between attributes
remain intact after noise injection. This is useful in cases where correlation structure
must be preserved for downstream analysis.

Common use cases include:
- Privacy-preserving transformations for correlated attributes.
- Ensuring anonymization while maintaining statistical relationships.
- Differential privacy mechanisms for multi-attribute datasets.

This module supports:
- Gaussian, Laplace, and Uniform noise.
- Custom covariance matrix-based correlation preservation.
- Rank-based and copula-based correlation preservation.

NOTE: This module requires `pandas`, `numpy`, and `scipy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC
from typing import Dict, List, Optional, Union, Any
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd
from pamola_core.anonymization.noise_addition.base import (
    BaseNoiseAdditionProcessor,
)
from pamola_core.common.constants import Constants
from pamola_core.common.logging.privacy_logging import log_privacy_transformations, save_privacy_logging
from pamola_core.common.validation.check_column import check_columns_exist


class CorrelatedNoiseAdditionProcessor(BaseNoiseAdditionProcessor, ABC):
    """
    Correlated Noise Addition Processor for preserving attribute relationships.
    This class extends BaseNoiseAdditionProcessor and applies noise to
    numerical attributes while maintaining their correlation structure.

    Methods:
    --------
    - add_noise(): Applies correlated noise to specified numerical columns
      while ensuring relationships between attributes remain intact.
    """

    def __init__(
        self,
        field_groups: Dict[str, List[str]] = {},
        noise_type: str = "gaussian",
        noise_parameters: Dict[str, Union[int, float]] = {},
        correlation_method: str = "covariance",
        preserve_linear: bool = True,
        preserve_rank: bool = False,
        group_specific_params: Dict[str, Dict[str, Union[int, float]]] = {},
        covariance_matrix: Optional[np.ndarray] = None,
        min_val: Dict[str, Union[int, float]] = {},
        max_val: Dict[str, Union[int, float]] = {},
        correlation_tolerance: float = 0.05,
        validation_method: str = "correlation",
        track_correlation: bool = True,
        correlation_log: Optional[str] = None,
    ):
        """
        Initializes the CorrelatedNoiseAdditionProcessor with specified parameters.

        This processor applies correlated noise to groups of numerical attributes 
        while attempting to preserve their statistical relationships.

        Parameters:
        -----------
        field_groups : Dict[str, List[str]]
            Dictionary mapping group names to lists of related fields that should 
            retain correlation after noise application.
        noise_type : str, optional (default="gaussian")
            Type of noise distribution to use. Options: "gaussian", "laplace", "uniform".
        noise_parameters : Dict[str, Union[int, float]], optional (default={})
            Dictionary of additional parameters for noise generation 
            (e.g., {"std_dev": 1.0, "mean": 0.0}).
        correlation_method : str, optional (default="covariance")
            Method for preserving correlation. Options: "covariance", "copula", "rank".
        preserve_linear : bool, optional (default=True)
            Whether to preserve linear relationships between attributes.
        preserve_rank : bool, optional (default=False)
            Whether to preserve rank order of values (useful for non-parametric relationships).
        group_specific_params : Dict[str, Dict[str, Union[int, float]]], optional (default={})
            Dictionary of custom noise parameters for specific field groups.
        covariance_matrix : Optional[np.ndarray], optional (default=None)
            Custom covariance matrix for correlated noise addition. If provided, 
            it overrides automatic computation.
        min_val : Dict[str, float], optional (default={})
            Dictionary specifying the minimum bound for each field after noise is applied.
        max_val : Dict[str, float], optional (default={})
            Dictionary specifying the maximum bound for each field after noise is applied.
        correlation_tolerance : float, optional (default=0.05)
            Allowed deviation in correlation after noise addition.
        validation_method : str, optional (default="correlation")
            Method used for validating correlation preservation. Options: "correlation", "distance".
        track_correlation : bool, optional (default=True)
            Whether to track correlation changes before and after noise application.
        correlation_log : Optional[str], optional (default=None)
            Path to a log file where correlation tracking data is saved, if enabled.
        """
        super().__init__()

        self.field_groups = field_groups or {}
        self.noise_type = noise_type
        self.noise_parameters = noise_parameters or {}
        self.correlation_method = correlation_method
        self.preserve_linear = preserve_linear
        self.preserve_rank = preserve_rank
        self.group_specific_params = group_specific_params or {}
        self.covariance_matrix = covariance_matrix
        self.min_val = min_val
        self.max_val = max_val
        self.correlation_tolerance = correlation_tolerance
        self.validation_method = validation_method
        self.track_correlation = track_correlation
        self.correlation_log = correlation_log
        self.change_log = {}

    def add_noise(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply correlated noise to the dataset while preserving relationships between fields.

        This method applies noise to specified groups of fields while maintaining their
        statistical relationships. It supports different noise types and correlation
        preservation methods.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset to apply noise to
        **kwargs : Dict[str, Any]
            Optional parameters to override instance attributes:
            - field_groups: List[List[str]] - Groups of related fields
            - noise_type: str - Type of noise distribution
            - noise_parameters: Dict - Noise generation parameters
            - correlation_method: str - Method for correlation preservation
            - preserve_linear: bool - Whether to preserve linear relationships
            - preserve_rank: bool - Whether to preserve rank order
            - group_specific_params: Dict - Custom parameters per group
            - covariance_matrix: np.ndarray - Custom covariance matrix
            - min_val: Dict[str, float] - Minimum bounds per field
            - max_val: Dict[str, float] - Maximum bounds per field
            - correlation_tolerance: float - Maximum allowed correlation deviation
            - validation_method: str - Method for validating correlation
            - track_correlation: bool - Whether to track changes
            - correlation_log: str - Path for saving correlation logs

        Returns:
        --------
        pd.DataFrame
            Dataset with correlated noise added

        Raises:
        -------
        TypeError
            If input data is not a pandas DataFrame
        ValueError
            If field groups are empty or fields are missing from dataset
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        # Get parameters with kwargs overrides
        params = self._get_params_with_overrides(**kwargs)
        
        # Validate field groups
        if not params["field_groups"]:
            raise ValueError("Parameter `field_groups` cannot be empty or None")

        # Create copy to avoid modifying original
        modified_data = data.copy()

        # Process each group
        for group, fields in params["field_groups"].items():
            modified_data = self._process_data(
                original_data=data,
                modified_data=modified_data,
                group=group,
                fields=fields,
                params=params
            )

        # Save noise log if tracking is enabled
        if params["track_correlation"] and params["correlation_log"]:
            save_privacy_logging(
                change_log=self.change_log,
                log_str=params["correlation_log"],
                track_changes=params["track_correlation"]
            )

        return modified_data

    def _get_params_with_overrides(self, **kwargs) -> Dict:
        """
        Get parameter dictionary with overrides.

        This method builds a dictionary of configuration parameters by combining the instance-level
        default values with any overrides provided via keyword arguments.

        Parameters:
        -----------
        **kwargs : dict
            Optional keyword arguments to override default instance-level parameters.

        Returns:
        --------
        Dict
            A dictionary of configuration parameters, combining defaults and overrides.
        """
        return {
            "field_groups": kwargs.get("field_groups", self.field_groups),
            "noise_type": kwargs.get("noise_type", self.noise_type),
            "noise_parameters": kwargs.get("noise_parameters", self.noise_parameters),
            "correlation_method": kwargs.get("correlation_method", self.correlation_method),
            "preserve_linear": kwargs.get("preserve_linear", self.preserve_linear),
            "preserve_rank": kwargs.get("preserve_rank", self.preserve_rank),
            "group_specific_params": kwargs.get("group_specific_params", self.group_specific_params),
            "covariance_matrix": kwargs.get("covariance_matrix", self.covariance_matrix),
            "min_val": kwargs.get("min_val", self.min_val),
            "max_val": kwargs.get("max_val", self.max_val),
            "correlation_tolerance": kwargs.get("correlation_tolerance", self.correlation_tolerance),
            "validation_method": kwargs.get("validation_method", self.validation_method),
            "track_correlation": kwargs.get("track_correlation", self.track_correlation),
            "correlation_log": kwargs.get("correlation_log", self.correlation_log),
        }

    def _process_data(
        self,
        original_data: pd.DataFrame,
        modified_data: pd.DataFrame,
        group: str,
        fields: List[str],
        params: Dict
    ) -> pd.DataFrame:
        """
        Process a single field group with noise addition.

        This function applies correlated noise to a specific group of fields while preserving
        statistical relationships and ensuring bounds.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original unmodified dataset before noise application.
        modified_data : pd.DataFrame
            The dataset where noise is applied.
        group : str
            The name of the group of fields being processed.
        fields : List[str]
            The list of fields within the group to which noise will be applied.
        params : Dict
            A dictionary containing noise configuration parameters:
            - **correlation_method** (str): 
                The method used to preserve correlation ("covariance", "copula", "rank").
            - **noise_type** (str): 
                Type of noise distribution ("gaussian", "laplace", "uniform").
            - **noise_params** (Dict): 
                Specific noise parameters like scale or standard deviation.
            - **covariance_matrix** (np.ndarray, optional): 
                Custom covariance matrix for correlated noise.
            - **validation_method** (str, optional): 
                If provided, validates correlation preservation after noise application.
            - **track_correlation** (bool): 
                Whether to track changes in correlation and log noise transformations.
            - **preserve_linear** (bool): 
                If True, ensures noise-added values retain their original linear relationships.
            - **min_val** (Dict[str, Union[int, float]], optional): 
                A dictionary specifying minimum allowable values for each field.
            - **max_val** (Dict[str, Union[int, float]], optional): 
                A dictionary specifying maximum allowable values for each field.

        Returns:
        --------
        pd.DataFrame
            The modified dataset with noise applied.
        """

        # Check if all target columns exist in the data
        check_columns_exist(modified_data, fields)

        # Get group-specific parameters
        noise_params = self._get_noise_params_for_group(group, params)

        # Track original values if needed
        track_correlation = params["track_correlation"]
        original_values = original_data.loc[:, fields].copy() if track_correlation else None
        
        # Apply correlated noise
        modified_data.loc[:, fields] = self._apply_correlated_noise(
            fields_data=modified_data.loc[:, fields],
            correlation_method=params["correlation_method"],
            noise_type=params["noise_type"],
            noise_params=noise_params,
            covariance_matrix=params["covariance_matrix"]
        )

        # Validate correlation if method specified
        if params["validation_method"]:
            self._validate_correlation(
                original_data=original_data.loc[:, fields],
                modified_data=modified_data.loc[:, fields],
                params=params
            )

        # Apply bounds per field if preserving linear relationships
        if noise_params["preserve_linear"]:
            min_val_dict = noise_params["min_val"]
            max_val_dict = noise_params["max_val"]

            for field in fields:
                modified_data.loc[:, field] = modified_data.loc[:, field].clip(
                    lower=min_val_dict.get(field, -float("inf")),
                    upper=max_val_dict.get(field, float("inf"))
                )

        # Track noise changes if enabled
        if track_correlation:
            modified_values = modified_data.loc[:, fields]
            mask_changes = original_values.ne(modified_values).any(axis=1)

            if mask_changes.any().any():
                log_privacy_transformations(
                    self.change_log,
                    operation_name=Constants.OPERATION_NAMES[1],
                    func_name=self._apply_correlated_noise.__name__,
                    mask=mask_changes,
                    original_values=original_values,
                    modified_values=modified_values,
                    fields=fields
                )

        return modified_data

    def _get_noise_params_for_group(self, group: List, params: Dict) -> Dict:
        """
        Retrieve noise parameters for a specific group by merging global defaults and group-specific overrides.

        Parameters:
        -----------
        group : List
            The List of column names that belong to the same field group.
        params : Dict
            Dictionary containing various noise configuration settings, including:
            - "noise_parameters": Global noise settings (default values for all groups).
            - "group_specific_params": Dictionary specifying per-group overrides.
            - "preserve_linear": Global setting for preserving linear relationships.
            - "preserve_rank": Global setting for preserving rank correlations.

        Returns:
        --------
        Dict
            A Dictionary containing the noise parameters specific to the given group, 
            with group-specific settings overriding global defaults.
        """
        global_params = params["noise_parameters"]
        group_params = params["group_specific_params"].get(group, {})

        return {
            "std_dev": group_params.get("std_dev", global_params.get("std_dev", 1.0)),
            "mean": group_params.get("mean", global_params.get("mean", 0.0)),
            "min_val": group_params.get("min_val", global_params.get("min_val", None)),
            "max_val": group_params.get("max_val", global_params.get("max_val", None)),
            "preserve_linear": group_params.get(
                "preserve_linear", params["preserve_linear"]
            ),
            "preserve_rank": group_params.get("preserve_rank", params["preserve_rank"]),
        }

    def _apply_correlated_noise(
        self,
        fields_data: pd.DataFrame,
        correlation_method: str,
        noise_type: str,
        noise_params: dict,
        covariance_matrix: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate noise based on the specified correlation method and noise type.

        Parameters:
        -----------
        fields_data : pd.DataFrame
            The fields to apply noise to.
        correlation_method : str
            The method to use for correlation preservation: "covariance", "copula", or "rank".
        noise_type : str
            Type of noise: "gaussian", "laplace", or "uniform".
        noise_params : dict
            Dictionary containing parameters for noise generation (e.g., std_dev, mean).
        covariance_matrix : Optional[np.ndarray]
            The covariance matrix, if applicable.

        Returns:
        --------
        pd.DataFrame
            Data with applied correlated noise.
        """
        size, num_cols = fields_data.shape

        # Step 1: Generate base noise depending on correlation method
        if correlation_method == "covariance":
            if covariance_matrix is None:
                if num_cols == 1:
                    # Single-variable case: use variance as a 1x1 matrix
                    variance = np.var(fields_data.to_numpy().flatten())
                    covariance_matrix = np.array([[variance]])
                else:
                    # Multi-variable case: use empirical covariance
                    covariance_matrix = np.cov(fields_data.to_numpy().T)

            # Convert to numpy array if it's a list
            if isinstance(covariance_matrix, list):
                covariance_matrix = np.array(covariance_matrix)
                
            if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
                raise ValueError("Covariance matrix must be 2D and square.")

            base_noise = np.random.multivariate_normal(
                mean=[0] * num_cols,
                cov=covariance_matrix,
                size=size
            )

        elif correlation_method == "copula":
            ranks = fields_data.rank(method="average") / (size + 1)
            base_noise = stats.norm.ppf(ranks)
            base_noise -= base_noise.mean(axis=0)

        elif correlation_method == "rank":
            random_noise = np.random.normal(0, 1, (size, num_cols))
            sorted_noise = np.sort(random_noise, axis=0)
            ranks = fields_data.rank(method="average").astype(int) - 1
            base_noise = np.take_along_axis(sorted_noise, ranks.to_numpy(), axis=0)

        else:
            raise ValueError(f"Unsupported correlation method: {correlation_method}")

        # Step 2: Normalize the noise to mean=0, std=1 before scaling
        base_noise = (base_noise - base_noise.mean(axis=0)) / (base_noise.std(axis=0) + 1e-8)

        # Step 3: Scale noise based on type and noise_params
        std_dev = noise_params["std_dev"]
        mean = noise_params["mean"]
        preserve_rank_order = noise_params["preserve_rank"]

        if noise_type == "gaussian":
            base_noise = base_noise * std_dev + mean

        elif noise_type == "laplace":
            base_noise = base_noise * std_dev + mean
            base_noise = np.sign(base_noise) * np.abs(
                np.random.laplace(loc=0, scale=1, size=base_noise.shape)
            )

        elif noise_type == "uniform":
            base_noise = np.random.uniform(low=-std_dev, high=std_dev, size=base_noise.shape)

        # Step 4: Apply noise to the data
        if preserve_rank_order:
            # Apply noise and then re-sort to original rank positions
            perturbed_data = fields_data.to_numpy() + base_noise
            ranked_indices = fields_data.rank(method="average").astype(int) - 1
            perturbed_data_sorted = np.sort(perturbed_data, axis=0)
            fields_data = np.take_along_axis(perturbed_data_sorted, ranked_indices.to_numpy(), axis=0)
        else:
            fields_data += base_noise

        return fields_data

    def _validate_correlation(self, original_data: pd.DataFrame, modified_data: pd.DataFrame, params: Dict) -> None:
        """
        Validate that correlation structure is preserved within specified tolerance.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset before noise application.
        modified_data : pd.DataFrame
            The dataset after noise application.
        params : Dict
            Dictionary containing:
            - "validation_method" (str): "correlation" or "mutual_information".
            - "correlation_tolerance" (float): Maximum allowed deviation.

        Raises:
        -------
        ValueError:
            If correlation deviation exceeds tolerance.
        """
        if not isinstance(original_data, pd.DataFrame) or not isinstance(modified_data, pd.DataFrame):
            raise TypeError("Both original and modified data must be pandas DataFrames")
        
        if original_data.shape != modified_data.shape:
            raise ValueError("Original and modified data must have the same shape")

        validation_method = params["validation_method"]
        tolerance = params["correlation_tolerance"]

        if validation_method == "correlation":
            self._validate_pearson_correlation(original_data, modified_data, tolerance)
        elif validation_method == "mutual_information":
            self._validate_mutual_information(original_data, modified_data, tolerance)
        else:
            raise ValueError(f"Unsupported validation method: {validation_method}")

    def _validate_pearson_correlation(self, original_data: pd.DataFrame, modified_data: pd.DataFrame, tolerance: float) -> None:
        """
        Validate that Pearson correlation is preserved.

        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataset before noise application.
        modified_data : pd.DataFrame
            Dataset after noise application.
        tolerance : float
            Maximum allowed correlation deviation.

        Raises:
        -------
        ValueError:
            If correlation deviation exceeds tolerance.
        """
        original_corr = original_data.corr()
        modified_corr = modified_data.corr()
        
        correlation_diff = np.abs(original_corr - modified_corr)
        max_diff = correlation_diff.max().max()

        if max_diff > tolerance:
            max_idx = np.unravel_index(np.argmax(correlation_diff.values), correlation_diff.shape)
            attr1, attr2 = original_corr.index[max_idx[0]], original_corr.columns[max_idx[1]]
            raise ValueError(f"Correlation deviation ({max_diff:.4f}) between {attr1} and {attr2} exceeds tolerance ({tolerance})!")

    def _validate_mutual_information(self, original_data: pd.DataFrame, modified_data: pd.DataFrame, tolerance: float) -> None:
        """
        Validate that mutual information is preserved.

        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataset before noise application.
        modified_data : pd.DataFrame
            Dataset after noise application.
        tolerance : float
            Maximum allowed mutual information deviation.

        Raises:
        -------
        ValueError:
            If mutual information deviation exceeds tolerance.
        """
        mi_diff = self._compute_mutual_information_difference(original_data, modified_data)
        
        if mi_diff > tolerance:
            raise ValueError(f"Mutual information deviation ({mi_diff:.4f}) exceeds tolerance ({tolerance})!")

    def _compute_mutual_information_difference(self, original_data: pd.DataFrame, modified_data: pd.DataFrame) -> float:
        """
        Compute the absolute difference in average mutual information before and after noise application.

        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataset before noise application.
        modified_data : pd.DataFrame
            Dataset after noise application.

        Returns:
        --------
        float
            Absolute difference in average mutual information.
        """
        def compute_avg_mi(data: pd.DataFrame) -> float:
            mi_values = []
            for col in data.columns:
                other_cols = data.drop(columns=[col])
                if not other_cols.empty:
                    mi = mutual_info_regression(other_cols, data[col])
                    mi_values.extend(mi)
            return np.mean(mi_values) if mi_values else 0.0

        return np.abs(compute_avg_mi(original_data) - compute_avg_mi(modified_data))