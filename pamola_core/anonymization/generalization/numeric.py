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

Module: Numeric Generalization Processor
-----------------------------------------
This module provides methods for generalizing numerical attributes
to enhance anonymization. It extends the BaseGeneralizationProcessor and
implements techniques such as binning, rounding, and normalization.

Numeric generalization reduces the granularity of numerical values
while preserving statistical distribution and analytical utility.

Common approaches include:
- **Range Binning**: Replacing continuous values with predefined intervals
  (e.g., converting ages into groups like "18-25", "26-35").
- **Precision Reduction**: Rounding numerical values to lower precision
  (e.g., truncating currency values to nearest hundred).
- **Scaling and Normalization**: Transforming values to a common scale
  while preserving proportional relationships.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from abc import ABC
from pamola_core.anonymization.generalization.base import BaseGeneralizationProcessor
from pamola_core.common.constants import Constants
from pamola_core.common.enum.numeric_generalization import NumericMethod
from pamola_core.common.logging.privacy_logging import log_privacy_transformations, save_privacy_logging
from pamola_core.common.validation.check_column import check_columns_exist


class NumericGeneralizationProcessor(BaseGeneralizationProcessor, ABC):
    """
    Numeric Generalization Processor for anonymizing numerical attributes.
    This class extends BaseGeneralizationProcessor and provides techniques
    for generalizing numerical data to enhance anonymization.

    Methods:
    --------
    - bin_values(): Groups numeric values into predefined bins.
    - round_values(): Reduces numerical precision by rounding.
    - normalize_values(): Normalizes numerical values to a standard scale.
    """

    def __init__(
        self, 
        target_fields: List[str] = [],
        method: str = NumericMethod.BINNING.value,
        bins: Optional[Dict[str, List[Union[int, float]]]] = None,
        round_precision: Optional[int] = None, 
        scaling_method: Optional[str] = None, 
        num_bins: int = 5,
        bin_strategy: str = "equal_width",
        auto_bins: bool = False,
        bin_labels: Optional[Dict[str, List[str]]] = None, 
        track_changes: bool = True, 
        preserve_fields: Optional[List[str]] = None, 
        generalization_log: Optional[str] = None
    ):
        """
        Initializes the numeric generalization processor.

        Parameters:
        -----------
        target_fields : List[str], default=[]
            List of numerical fields to apply generalization.
        method : str, default="binning"
            Generalization method to use. Options:
            - "binning": Groups values into bins.
            - "rounding": Rounds values to a specified precision.
            - "scaling": Normalizes values using scaling methods.
        bins : Dict[str, List[Union[int, float]]], optional
            Predefined bins for range-based generalization.
            Example: {"age": [0, 18, 30, 50, 100]}
        round_precision : int, optional (default=None)
            Number of decimal places to retain when using rounding.
            If None, rounding is not applied.
        scaling_method : str, optional (default=None)
            Scaling method to use if `method="scaling"`.
            Options: "min-max", "z-score".
        num_bins : int, default=5
            Number of bins to generate when `auto_bins=True`.
        bin_strategy : str, default="equal_width"
            Strategy for automatic bin generation. Options:
            - "equal_width": Each bin has an equal range.
            - "equal_frequency": Each bin contains an equal number of samples.
            - "quantile": Uses quantile-based binning.
        auto_bins : bool, default=False
            If True, bins are automatically generated.
        bin_labels : Dict[str, List[str]], optional (default={})
            Custom labels for bins. Example:
            {"age": ["Child", "Young Adult", "Adult", "Senior"]}
        track_changes : bool, default=True
            If True, enables tracking of generalization changes.
        preserve_fields : List[str], optional (default=[])
            List of fields to exclude from generalization.
        generalization_log : str, optional (default=None)
            Path to save generalization logs.
        """
        super().__init__()
        self.target_fields = target_fields
        self.method = method
        self.bins = bins or {}
        self.round_precision = round_precision
        self.scaling_method = scaling_method
        self.num_bins = num_bins
        self.bin_strategy = bin_strategy
        self.auto_bins = auto_bins
        self.bin_labels = bin_labels or {}
        self.track_changes = track_changes
        self.preserve_fields = preserve_fields or []
        self.generalization_log = generalization_log
        self.change_log = {}

    def generalize(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply numeric generalization to specified fields.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        **kwargs : Dict, optional
            target_fields : List, optional
                Fields to generalize. Defaults to self.target_fields.
            method : str, optional
                Generalization method ("binning", "rounding", "scaling").
                Defaults to self.method.
            preserve_fields : List, optional
                Fields to exclude from generalization. Defaults to self.preserve_fields.
            track_changes : bool, optional
                Whether to track changes. Defaults to self.track_changes.
            generalization_log : str, optional
                Path to save generalization log.

        Returns:
        --------
        pd.DataFrame
            The dataset with generalized numeric values.
        """
        # Retrieve parameters from kwargs with defaults from instance attributes
        target_fields = kwargs.get("target_fields", self.target_fields)
        method = kwargs.get("method", self.method)
        preserve_fields = kwargs.get("preserve_fields", self.preserve_fields)
        track_changes = kwargs.get("track_changes", self.track_changes)
        generalization_log = kwargs.get("generalization_log", self.generalization_log)

        # Check if all target columns exist in the data
        check_columns_exist(data, target_fields)

        for column in target_fields:
            if column in preserve_fields:
                continue

            if method == NumericMethod.BINNING.value:
                data = self.bin_values(data, column, **kwargs)
            elif method == NumericMethod.ROUNDING.value:
                data = self.round_values(data, column, **kwargs)
            elif method == NumericMethod.SCALING.value:
                data = self.normalize_values(data, column, **kwargs)
            else:
                raise ValueError(f"Invalid generalization method: {method}")

        # Save generalization log if tracking is enabled
        if track_changes and generalization_log:
            save_privacy_logging(
                change_log=self.change_log,
                log_str=generalization_log,
                track_changes=track_changes
            )

        return data

    def bin_values(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Generalize numeric values by mapping them to predefined or auto-generated bins.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column : str
            The column name to be processed.
        **kwargs : Dict, optional
            bins : Dict[str, List[float]], optional
                Predefined bin boundaries. Defaults to self.bins.
            bin_labels : Dict[str, List[str]], optional
                Custom labels for bins. Defaults to self.bin_labels.
            auto_bins : bool, optional
                Whether to generate bins automatically. Defaults to self.auto_bins.
            num_bins : int, optional
                Number of bins to create if auto_bins=True. Defaults to self.num_bins.
            bin_strategy : str, optional
                Strategy for automatic binning ('equal_width', 'equal_frequency', 'quantile').
                Defaults to self.bin_strategy.
            track_changes : bool, default=self.track_changes
                Whether to track transformation changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with numeric values mapped to bins.
        """
        # Retrieve parameters from kwargs with fallback to instance attributes
        col_bins = (kwargs.get("bins") or self.bins).get(column, None)
        col_bin_labels = (kwargs.get("bin_labels") or self.bin_labels).get(column, None)
        auto_bins = kwargs.get("auto_bins", self.auto_bins)
        num_bins = kwargs.get("num_bins", self.num_bins)
        bin_strategy = kwargs.get("bin_strategy", self.bin_strategy)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Track original values if needed
        original_values = data[[column]].copy() if track_changes else None

        # Generate bins if auto_bins is enabled
        if auto_bins or col_bins is None:
            min_val, max_val = data[column].min(), data[column].max()
            if bin_strategy == "equal_width":
                col_bins = np.linspace(min_val, max_val, num_bins + 1)
            elif bin_strategy in {"equal_frequency", "quantile"}:
                col_bins = np.percentile(data[column], np.linspace(0, 100, num_bins + 1))
            else:
                raise ValueError(f"Invalid bin strategy: {bin_strategy}")

        if col_bins is None or len(col_bins) < 2:
            raise ValueError("Bins must be defined or auto-generated for binning.")

        # Convert bins to numpy array for processing
        col_bins = np.array(col_bins)
        if pd.api.types.is_integer_dtype(data[column]):
            col_bins = col_bins.astype(int)

        # Generate bin labels if not provided
        if col_bin_labels is None or len(col_bin_labels) < len(col_bins) - 1:
            col_bin_labels = [f"{col_bins[i]}-{col_bins[i+1]}" for i in range(len(col_bins) - 1)]

        # Apply binning
        data[column] = pd.cut(data[column], bins=col_bins, labels=col_bin_labels, include_lowest=True, duplicates="drop",)
        
        # Track changes efficiently
        if track_changes:
            mask = original_values[column] != data[column]
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.bin_values.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[[column]],
                fields=[column]
            )

        return data

    def round_values(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Reduce precision of numeric values by rounding.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column : str
            The column name to be processed.
        **kwargs : Dict, optional
            round_precision : int, optional
                Decimal places for rounding. Defaults to self.round_precision.
            track_changes : bool, default=self.track_changes
                Whether to track transformation changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with rounded numeric values.
        """
        # Retrieve parameters from kwargs or fallback to instance attributes
        round_precision = kwargs.get("round_precision", self.round_precision)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Ensure round precision is valid
        if not isinstance(round_precision, int):
            raise ValueError("Rounding precision must be a non-negative integer.")
        
        # Track original values if needed
        original_values = data[[column]].copy() if track_changes else None

        # Round values
        data[column] = data[column].round(round_precision)

        # Track changes efficiently
        if track_changes:
            mask = original_values[column] != data[column]
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.round_values.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[[column]],
                fields=[column]
            )
        
        return data

    def normalize_values(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Normalize numeric values to a standard scale.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numeric data.
        column : str
            The column name to be processed.
        **kwargs : Dict, optional
            scaling_method : str, optional
                Scaling method ('min-max', 'z-score'). Defaults to self.scaling_method.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with normalized numeric values.
        """
        # Retrieve parameters from kwargs or fallback to instance attributes
        scaling_method = kwargs.get("scaling_method", self.scaling_method)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Track original values if needed
        original_values = data[[column]].copy() if track_changes else None

         # Apply normalization
        if scaling_method == "min-max":
            min_val, max_val = data[column].min(), data[column].max()
            if max_val == min_val:
                data[column] = 0
            else:
                data[column] = (data[column] - min_val) / (max_val - min_val)

        elif scaling_method == "z-score":
            mean_val, std_val = data[column].mean(), data[column].std()
            if std_val == 0:
                data[column] = 0
            else:
                data[column] = (data[column] - mean_val) / std_val

        else:
            raise ValueError(f"Invalid scaling method: {scaling_method}")

        # Track changes efficiently
        if track_changes:
            mask = original_values[column] != data[column]
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.normalize_values.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[[column]],
                fields=[column]
            )
        
        return data
