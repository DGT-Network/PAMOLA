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

Module: Laplace Noise Addition Processor
-----------------------------------------
This module provides an implementation of Laplace noise addition
for anonymization-preserving data transformations. It extends the
BaseNoiseAdditionProcessor and allows controlled noise injection
to numerical attributes.

Laplace noise addition is widely used in **differential anonymization** as it
ensures anonymization protection by drawing random values from a Laplace
distribution with a specified mean and scale.

Common use cases include:
- Differential anonymization mechanisms.
- Statistical perturbation while maintaining utility.
- Preventing exact inference of original values.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from abc import ABC
from pamola_core.anonymization.noise_addition.base import (
    BaseNoiseAdditionProcessor,
)
from pamola_core.common.constants import Constants
from pamola_core.common.logging.privacy_logging import log_privacy_transformations, save_privacy_logging
from pamola_core.common.validation.check_column import check_columns_exist


class LaplaceNoiseAdditionProcessor(BaseNoiseAdditionProcessor, ABC):
    """
    Laplace Noise Addition Processor for anonymizing numerical attributes.
    This class extends BaseNoiseAdditionProcessor and applies Laplace noise
    to numerical values while preserving statistical properties.

    Methods:
    --------
    - add_noise(): Applies Laplace noise to numerical columns.
    """

    def __init__(
        self,
        target_fields: List[str] = [],
        mean: float = 0.0,
        scale: float = 1.0,
        min_val: Union[int, float] = None,
        max_val: Union[int, float] = None,
        epsilon: float = None,
        sensitivity: float = None,
        field_specific_params: Dict[str, Any] = None,
        adaptive_noise: bool = False,
        preserve_range: bool = True,
        track_noise: bool = True,
        dp_budget_allocation: str = "equal",
        noise_log: str = None,
    ):
        """
        Initializes the Laplace Noise Addition Processor.

        This processor applies Laplace noise to specified fields, supporting differential privacy
        and controlled noise injection to enhance anonymization while maintaining data utility.

        Parameters:
        -----------
        target_fields : List[str], optional (default=[])
            List of column names to which noise should be applied.
        mean : float, optional (default=0.0)
            Mean of the Laplace distribution, controlling the central tendency of the noise.
        scale : float, optional (default=1.0)
            Scale parameter (b) for the Laplace distribution, controlling the spread of the noise.
        min_val : int or float, optional (default=None)
            Minimum allowable value after noise addition. If specified, ensures values do not go below this threshold.
        max_val : int or float, optional (default=None)
            Maximum allowable value after noise addition. If specified, ensures values do not exceed this threshold.
        epsilon : float, optional (default=None)
            Privacy budget for Differential Privacy (DP). If provided, noise is scaled accordingly.
        sensitivity : float, optional (default=None)
            Sensitivity of the function, used in DP noise scaling calculations.
        field_specific_params : Dict[str, Any], optional (default=None)
            Dictionary of per-column noise parameters (e.g., {"age": {"scale": 0.5}}).
        adaptive_noise : bool, optional (default=False)
            If True, dynamically adjusts the noise scale based on data distribution.
        preserve_range : bool, optional (default=True)
            If True, ensures that noise-added values remain within the original data range.
        track_noise : bool, optional (default=False)
            If True, logs noise changes for auditing and tracking.
        dp_budget_allocation : str, optional (default="equal")
            Strategy for distributing the DP budget across fields.
            Options:
            - "equal": Allocates budget evenly across fields.
            - "sensitivity_based": Allocates budget based on sensitivity values.
        noise_log : str, optional (default=None)
            Path to a log file where noise metrics are saved if `track_noise` is enabled.
        """
        super().__init__()

        self.target_fields = target_fields
        self.mean = mean
        self.scale = scale
        self.min_val = min_val
        self.max_val = max_val
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.field_specific_params = field_specific_params or {}
        self.adaptive_noise = adaptive_noise
        self.preserve_range = preserve_range
        self.dp_budget_allocation = dp_budget_allocation
        self.track_noise = track_noise
        self.noise_log = noise_log
        self.change_log = {}

    def add_noise(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply Laplace noise to specified columns in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset containing numerical data.
        kwargs : Dict
            Optional parameters to override instance-level settings.

        Returns:
        --------
        pd.DataFrame
            Dataset with Laplace noise applied.
        """
        # Get final parameters with overrides
        params = self._get_params_with_overrides(**kwargs)

        # Validate essential parameters
        self._validate_params(params)

        # Check if all target columns exist in the data
        check_columns_exist(data, params["target_fields"])

        # Compute sensitivity values based on DP budget allocation
        effective_sensitivity = self._compute_sensitivity(params)

        # Add noise to each target field
        for column in params["target_fields"]:
            column_params = self._get_column_params(column, params, effective_sensitivity)
            data = self._apply_laplace_noise(
                data=data,
                column_name=column,
                mean=column_params["mean"],
                scale=column_params["scale"],
                min_val=column_params["min_val"],
                max_val=column_params["max_val"],
                preserve_range=column_params["preserve_range"],
                track_noise=params["track_noise"]
            )

        # Save noise log if tracking is enabled
        if params["track_noise"] and params["noise_log"]:
            save_privacy_logging(
                change_log=self.change_log,
                log_str=params["noise_log"],
                track_changes=params["track_noise"]
            )

        return data
    
    def _get_params_with_overrides(self, **kwargs) -> Dict:
        """
        Retrieve the final parameters by applying overrides from the provided keyword arguments.

        This method merges instance-level default values with any user-specified overrides 
        from `kwargs`, ensuring that all parameters are properly set.

        Parameters:
        -----------
        kwargs : Dict
            A dictionary containing user-specified parameter overrides.

        Returns:
        --------
        Dict
            A dictionary with the final parameter values after applying overrides.
        """
        return {
            "target_fields": kwargs.get("target_fields", self.target_fields),
            "scale": kwargs.get("scale", self.scale),
            "mean": kwargs.get("mean", self.mean),
            "min_val": kwargs.get("min_val", self.min_val),
            "max_val": kwargs.get("max_val", self.max_val),
            "epsilon": kwargs.get("epsilon", self.epsilon),
            "sensitivity": kwargs.get("sensitivity", self.sensitivity),
            "field_specific_params": kwargs.get("field_specific_params", self.field_specific_params) or {},
            "adaptive_noise": kwargs.get("adaptive_noise", self.adaptive_noise),
            "preserve_range": kwargs.get("preserve_range", self.preserve_range),
            "dp_budget_allocation": kwargs.get("dp_budget_allocation", self.dp_budget_allocation),
            "track_noise": kwargs.get("track_noise", self.track_noise),
            "noise_log": kwargs.get("noise_log", self.noise_log)
        }

    def _validate_params(self, params: Dict[str, Any]):
        """
        Validate the required parameters to ensure they meet expected constraints.

        This method checks that:
        - `scale` is strictly greater than zero.
        - If `epsilon` is provided, it must also be greater than zero.

        Parameters:
        -----------
        params : Dict[str, Any]
            Dictionary of parameters to validate.

        Raises:
        -------
        ValueError:
            If any parameter does not meet the required constraints.
        """
        if params["scale"] <= 0:
            raise ValueError("Scale parameter must be greater than zero.")
    
        if params.get("epsilon") is not None and params["epsilon"] <= 0:
            raise ValueError("Epsilon (privacy budget) must be greater than zero.")
        
    def _get_column_params(self, column: str, params: Dict[str, Any], effective_sensitivity: Dict[str, float]) -> Dict[str, Any]:
        """
        Retrieve the final parameters for a given column, considering per-column overrides and global defaults.

        This method:
        - Extracts column-specific parameters if available, otherwise falls back to global defaults.
        - Adjusts the noise scale dynamically if adaptive noise and differential privacy (DP) settings are enabled.

        Parameters:
        -----------
        column : str
            The name of the column for which parameters are being retrieved.
        params : Dict[str, Any]
            Dictionary containing global and per-column noise addition parameters.
        effective_sensitivity : Dict[str, float]
            Dictionary mapping each column to its computed sensitivity.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the final parameters for the given column
        """
        col_params = params.get("field_specific_params", {}).get(column, {})
        
        col_scale = col_params.get("scale", params["scale"])
        col_mean = col_params.get("mean", params["mean"])
        col_min_val = col_params.get("min_val", params["min_val"])
        col_max_val = col_params.get("max_val", params["max_val"])
        col_adaptive_noise = col_params.get("adaptive_noise", params["adaptive_noise"])
        col_preserve_range = col_params.get("preserve_range", params["preserve_range"])
        col_sensitivity = effective_sensitivity.get(column, params["sensitivity"])

        # Adjust noise scale based on DP settings
        if col_adaptive_noise and params["epsilon"]:
            col_scale = col_sensitivity / params["epsilon"]

        return {
            "scale": col_scale,
            "mean": col_mean,
            "min_val": col_min_val,
            "max_val": col_max_val,
            "preserve_range": col_preserve_range,
            "sensitivity": col_sensitivity
        }

    def _compute_sensitivity(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute sensitivity values based on the chosen DP budget allocation strategy.

        Parameters:
        -----------
        params : Dict
            A dictionary containing the following keys:
            - `dp_budget_allocation`: A strategy for allocating the DP budget (`"equal"` or `"sensitivity_based"`).
            - `sensitivity`: The default sensitivity value to use.
            - `target_fields`: A list of the target columns to apply the sensitivity.
            - `field_specific_params`: A dictionary with field-specific parameters, including `sensitivity`.

        Returns:
        --------
        Dict
            A dictionary with the computed sensitivity values for each target field.
        
        Raises:
        -------
        ValueError:
            If an invalid dp_budget_allocation strategy is provided.
        """
        dp_budget_allocation = params["dp_budget_allocation"]
        target_fields = params["target_fields"]
        field_specific_params = params.get("field_specific_params", {})

        # Ensure target_fields is not empty
        if not target_fields:
            raise ValueError("Target fields cannot be empty.")

        if dp_budget_allocation == "equal":
            # If allocation is equal, all fields get the same sensitivity
            return {col: params["sensitivity"] for col in target_fields}

        elif dp_budget_allocation == "sensitivity_based":
            # If allocation is sensitivity-based, use field-specific sensitivity if available
            return {
                col: field_specific_params.get(col, {}).get("sensitivity", params["sensitivity"])
                for col in target_fields
            }

        else:
            raise ValueError(f"Invalid dp_budget_allocation method: {dp_budget_allocation}")
    
    def _apply_laplace_noise(
        self,
        data: pd.DataFrame,
        column_name: str,
        mean: float,
        scale: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        preserve_range: bool = True,
        track_noise: bool = False,
    ) -> pd.DataFrame:
        """
        Applies Laplace noise to a numerical column while preserving optional constraints.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column_name : str
            The name of the column being processed.
        mean : float
            The mean of the Laplace distribution.
        scale : float
            The scale parameter (b) for the Laplace distribution. Must be > 0.
        min_val : float, optional
            The minimum allowed value after noise is added. If None, no lower bound is applied.
        max_val : float, optional
            The maximum allowed value after noise is added. If None, no upper bound is applied.
        preserve_range : bool
            Whether to ensure that the values remain within their original min/max range.
        track_noise : bool
            Whether to track the added noise for each value (default: False).

        Returns:
        --------
        pd.DataFrame
            The dataframe with Laplace noise applied to the specified column.
        """
        # Preserve NaN values and apply noise only to non-null entries
        mask = ~data[column_name].isna()
        
        # Generate Laplace noise only for valid (non-null) values
        noise = np.random.laplace(loc=mean, scale=scale, size=mask.sum())
        
        # Track original values if needed
        original_values = data.loc[mask, column_name].copy() if track_noise else None

        # Apply noise safely
        data.loc[mask, column_name] += noise

        # Apply range constraints if enabled
        if preserve_range:
            data[column_name] = data[column_name].clip(lower=min_val, upper=max_val)

        # Track noise changes if enabled
        if track_noise:
            modified_values = data.loc[mask, column_name]
            mask_changes = original_values.ne(modified_values)
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[1],
                func_name=self._apply_laplace_noise.__name__,
                mask=mask_changes,
                original_values=original_values,
                modified_values=modified_values,
                fields=[column_name]
            )

        return data