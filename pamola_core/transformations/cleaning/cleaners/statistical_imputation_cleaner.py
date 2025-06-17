"""
Basic Statistical Imputation Cleaner for missing value handling.

This module provides the BasicStatisticalImputationCleaner class for imputing missing values
using common statistical methods based on the distribution of existing values.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pamola_core.common.enum.statistical_method import StatisticalMethod
from pamola_core.transformations.cleaning.cleaners.base_data_cleaner import BaseDataCleaner

# Configure logger
logger = logging.getLogger(__name__)

class BasicStatisticalImputationCleaner(BaseDataCleaner):
    """
    Cleaner for imputing missing values using common statistical methods.
    Supports mean, median, mode.
    """

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Basic Statistical Imputation Cleaner with configuration.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                - target_field (str): Column name to impute.
                - grouping_fields (list of str): Column names are used to group by before imputing.
                - method (str): One of 'mean', 'median', or 'mode'.
        """
        super().__init__(config)
        self.target_field = self.config.get("target_field", "")
        self.grouping_fields = self.config.get("grouping_fields", [])
        self.method = self.config.get("method", StatisticalMethod.MEAN.value)

    def execute(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Execute the imputation based on configuration or provided arguments.

        Args:
            df (pd.DataFrame): Input DataFrame containing missing values.
            **kwargs: Optional overrides for configuration:
                - target_field (str): Column name to impute.
                - grouping_fields (list of str): Column names are used to group by before imputing.
                - method (str): One of 'mean', 'median', or 'mode'.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        target_field = kwargs.get("target_field", self.target_field)
        grouping_fields = kwargs.get("grouping_fields", self.grouping_fields)
        method = kwargs.get("method", self.method)

        if grouping_fields is None:
            grouping_fields = []

        if not target_field:
            raise ValueError("Missing 'target_field' for imputation.")

        if set(target_field).intersection(grouping_fields):
            raise ValueError('target_field and grouping_fields have same value!')

        if method == StatisticalMethod.MEAN.value:
            return self.apply_mean_imputation(df.copy(), target_field, grouping_fields)

        if method == StatisticalMethod.MEDIAN.value:
            return self.apply_median_imputation(df.copy(), target_field, grouping_fields)

        if method == StatisticalMethod.MODE.value:
            return self.apply_mode_imputation(df.copy(), target_field, grouping_fields)

        raise ValueError(f"Unsupported method: {method}")

    def apply_mean_imputation(
            self,
            df: pd.DataFrame,
            target_field: str,
            grouping_fields: List[str]
    ) -> pd.DataFrame:
        """
        Apply mean imputation on a single column.

        Args:
            df (pd.DataFrame): DataFrame with missing values.
            target_field (str): Column to impute.
            grouping_fields (list of str): Column names are used to group by before imputing.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        if not is_numeric_dtype(df[target_field]):
            raise ValueError('target_field must be numeric dtype!')

        try:
            if grouping_fields:
                df_grouped = df.groupby(grouping_fields)
            else:
                df_grouped = df

            df = df.fillna(df_grouped[[target_field]].transform(lambda x: x.fillna(x.mean())))
        except Exception as e:
            logger.error(f"Mean imputation failed for '{target_field}': {e}")

        return df

    def apply_median_imputation(
            self,
            df: pd.DataFrame,
            target_field: str,
            grouping_fields: List[str]
    ) -> pd.DataFrame:
        """
        Apply median imputation.

        Args:
            df (pd.DataFrame): DataFrame with missing values.
            target_field (str): Column to impute.
            grouping_fields (list of str): Column names are used to group by before imputing.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        if not is_numeric_dtype(df[target_field]):
            raise ValueError('target_field must be numeric dtype!')

        try:
            if grouping_fields:
                df_grouped = df.groupby(grouping_fields)
            else:
                df_grouped = df

            df = df.fillna(df_grouped[[target_field]].transform(lambda x: x.fillna(x.median())))
        except Exception as e:
            logger.error(f"Median imputation failed for '{target_field}': {e}")

        return df

    def apply_mode_imputation(
            self,
            df: pd.DataFrame,
            target_field: str,
            grouping_fields: List[str]
    ) -> pd.DataFrame:
        """
        Apply mode imputation.

        Args:
            df (pd.DataFrame): DataFrame with missing values.
            target_field (str): Column to impute.
            grouping_fields (list of str): Column names are used to group by before imputing.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        try:
            if grouping_fields:
                df_grouped = df.groupby(grouping_fields)
            else:
                df_grouped = df

            df = df.fillna(df_grouped[[target_field]].transform(lambda x: x.fillna(x.mode().iloc[0])))
        except Exception as e:
            logger.error(f"Mode imputation failed for '{target_field}': {e}")

        return df
