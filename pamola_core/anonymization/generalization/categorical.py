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

Module: Categorical Generalization Processor
---------------------------------------------
This module provides methods for generalizing categorical attributes
to enhance anonymization. It extends the BaseGeneralizationProcessor and
implements techniques such as value grouping, frequency smoothing,
and hierarchical generalization.

Categorical generalization replaces specific categorical values with
more abstract, less specific values while preserving statistical
distribution.

Common approaches include:
- **Value Grouping**: Replacing rare values with more common categories.
- **Frequency Smoothing**: Merging infrequent categories into an "Other" group.
- **Hierarchical Generalization**: Mapping values to a higher-level taxonomy
  (e.g., "Car Brand" → "Automobile").
- **k-Anonymity Constraints**: Ensuring that each generalized category has
  at least k occurrences.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from typing import Dict, List, Optional
import pandas as pd
from abc import ABC
from pamola_core.anonymization.generalization.base import BaseGeneralizationProcessor
from pamola_core.common.logging.privacy_logging import log_privacy_transformations, save_privacy_logging
from pamola_core.common.validation.check_column import check_columns_exist
from pamola_core.common.constants import Constants

class CategoricalGeneralizationProcessor(BaseGeneralizationProcessor, ABC):
    """
    Categorical Generalization Processor for anonymizing categorical attributes.
    This class extends BaseGeneralizationProcessor and provides techniques for
    generalizing categorical data to enhance anonymization.

    Methods:
    --------
    - group_values(): Groups similar values into broader categories.
    - smooth_frequencies(): Merges infrequent categories into an "Other" class.
    - apply_hierarchy(): Uses a predefined hierarchy for generalization.
    """

    def __init__(
        self,
        target_fields: List[str] = [],
        method: str = "grouping",
        min_group_size: int = 5,
        threshold: float = 0.05,
        hierarchy: Optional[Dict] = None,
        other_category: str = "Other",
        preserve_fields: Optional[List[str]] = None,
        track_changes: bool = True,
        generalization_log: Optional[str] = None,
    ):
        """
        Initializes the generalization processor with configurable parameters.

        Parameters:
        -----------
        target_fields : List[str], optional
            List of fields to apply generalization. Defaults to an empty list.
        method : str, optional
            The generalization method to use. Options include:
            - "grouping": Groups infrequent categories into a single category.
            - "smoothing": Applies a smoothing technique based on frequency.
            - "hierarchy": Uses a predefined hierarchical mapping.
            Defaults to "grouping".
        min_group_size : int, optional
            The minimum number of occurrences a category must have before it is grouped. 
            Must be at least 1. Defaults to 5.
        threshold : float, optional
            Frequency threshold for category smoothing. Defaults to 0.05.
        hierarchy : Optional[Dict], optional
            A predefined hierarchical mapping for generalization. 
            If not provided, an empty dictionary is used. Defaults to None.
        other_category : str, optional
            Label assigned to grouped categories when using the "grouping" method. 
            Defaults to "Other".
        preserve_fields : Optional[List[str]], optional
            List of fields to exclude from generalization. Defaults to None.
        track_changes : bool, optional
            Whether to track transformation history for auditability. Defaults to True.
        generalization_log : Optional[str], optional
            File path to save generalization logs. Defaults to None.

        Attributes:
        -----------
        change_log : dict
            A dictionary to store changes made during the generalization process.
        """
        super().__init__()
        self.target_fields = target_fields
        self.method = method
        self.min_group_size = max(1, min_group_size)
        self.threshold = threshold
        self.hierarchy = hierarchy or {}
        self.other_category = other_category
        self.preserve_fields = preserve_fields or []
        self.track_changes = track_changes
        self.generalization_log = generalization_log
        self.change_log = {}
    
    def generalize(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply categorical generalization to target fields.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data to be generalized.

        **kwargs : Dict, optional
            target_fields : list, optional
                Fields to generalize. Defaults to self.target_fields.
            method : str, optional
                Generalization method ("binning", "rounding", "scaling").
                Defaults to self.method.
            preserve_fields : list, optional
                Fields to exclude from generalization. Defaults to self.preserve_fields.
            track_changes : bool, optional
                Whether to track changes. Defaults to self.track_changes.
            generalization_log : str, optional
                Path to save generalization log.

        Returns:
        --------
        pd.DataFrame
            The dataset with generalized categorical values.
        """

        # Retrieve parameters from kwargs with defaults from instance attributes
        method = kwargs.get("method", self.method)
        target_fields = kwargs.get("target_fields", self.target_fields)
        preserve_fields = kwargs.get("preserve_fields", self.preserve_fields)
        track_changes = kwargs.get("track_changes", self.track_changes)
        generalization_log = kwargs.get("generalization_log", self.generalization_log)

        # Check if all target columns exist in the data
        check_columns_exist(data, target_fields)

        for column in target_fields:
            if column in preserve_fields:
                continue  # Skip preserved fields
            
            if method == "grouping":
                data = self.group_values(data, column, **kwargs)
            elif method == "smoothing":
                data = self.smooth_frequencies(data, column, **kwargs)
            elif method == "hierarchy":
                data = self.apply_hierarchy(data, column, **kwargs)
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
    
    def group_values(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Group rare categorical values into a common category.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data.
        column : str
            The column name to be processed.

        **kwargs : Dict, optional
            min_group_size : int, optional
                Minimum occurrences for a category to remain unchanged.
                Defaults to self.min_group_size.
            other_category : str, optional
                Label for grouped categories. Defaults to self.other_category.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with grouped categorical values.
        """
        # Retrieve parameters from kwargs with fallback to instance attributes
        min_group_size = kwargs.get("min_group_size", self.min_group_size)
        other_category = kwargs.get("other_category", self.other_category)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Identify rare values
        value_counts = data[column].value_counts()  # Use absolute counts instead of normalized counts
        rare_values = value_counts[value_counts < min_group_size].index

        if rare_values.empty:
            return data  # No rare values, no changes needed

        # Track original values only if needed
        original_values = data[[column]].copy() if track_changes else None

        # Replace rare values efficiently
        mask = data[column].isin(rare_values)
        if mask.any():  # Avoid unnecessary assignment if no changes
            data.loc[mask, column] = other_category

            # Log changes efficiently
            if track_changes:
                log_privacy_transformations(
                    self.change_log,
                    operation_name=Constants.OPERATION_NAMES[0],
                    func_name=self.group_values.__name__,
                    mask=mask,
                    original_values=original_values,
                    modified_values=data[[column]],
                    fields=[column]
                )

        return data
    
    def smooth_frequencies(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Merge infrequent categories into a common category based on a frequency threshold.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data.
        column : str
            The column name to be processed.

        **kwargs : Dict, optional
            threshold : float, optional
                Frequency threshold for merging categories. Categories with a relative frequency
                below this value will be merged. Defaults to self.threshold.
            other_category : str, optional
                Label for grouped categories. Defaults to self.other_category.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with frequency-smoothed categorical values.
        """
        # Retrieve parameters from kwargs with fallback to instance attributes
        threshold = kwargs.get("threshold", self.threshold)
        other_category = kwargs.get("other_category", self.other_category)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Identify rare categories based on frequency threshold
        value_counts = data[column].value_counts(normalize=True)
        rare_values = value_counts[value_counts < threshold].index

        if rare_values.empty:
            return data  # No rare values, no changes needed

        # Track original values only if needed
        original_values = data[[column]].copy() if track_changes else None

        # Replace rare values efficiently
        mask = data[column].isin(rare_values)
        if mask.any():  # Avoid unnecessary assignment
            data.loc[mask, column] = other_category

            # Log changes efficiently
            if track_changes:
                log_privacy_transformations(
                    self.change_log,
                    operation_name=Constants.OPERATION_NAMES[0],
                    func_name=self.smooth_frequencies.__name__,
                    mask=mask,
                    original_values=original_values,
                    modified_values=data[[column]],
                    fields=[column]
                )

        return data
    
    def apply_hierarchy(self, data: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Apply predefined hierarchical generalization to a categorical column.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data.
        column : str
            The column name to be processed.

        **kwargs : Dict, optional
            hierarchy : Dict, optional
                A predefined mapping for categorical values to generalized categories.
                Defaults to self.hierarchy.
            track_changes : bool, optional
                Whether to track transformation changes. Defaults to self.track_changes.

        Returns:
        --------
        pd.DataFrame
            The dataset with hierarchical generalization applied.
        
        Raises:
        -------
        ValueError
            If no hierarchy mapping is provided.
        """
        # Retrieve parameters from kwargs with fallback to instance attributes
        hierarchy = kwargs.get("hierarchy", self.hierarchy)
        track_changes = kwargs.get("track_changes", self.track_changes)

        # Ensure hierarchy mapping is provided
        if not hierarchy:
            raise ValueError("Hierarchy mapping is required for this method.")

        # Track original values if needed
        original_values = data[[column]].copy() if track_changes else None

        # Apply hierarchical generalization using replace (faster than map + fillna)
        data[column] = data[column].replace(hierarchy).fillna('Unknown')  # Handle missing values if required

        # Log changes efficiently
        if track_changes:
            mask = original_values[column] != data[column]
            log_privacy_transformations(
                self.change_log,
                operation_name=Constants.OPERATION_NAMES[0],
                func_name=self.apply_hierarchy.__name__,
                mask=mask,
                original_values=original_values,
                modified_values=data[[column]],
                fields=[column]
            )
        
        return data
