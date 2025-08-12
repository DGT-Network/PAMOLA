"""
Null Handling Cleaner for missing value handling.

This module provides the NullHandlingCleaner class for handling missing values
based on configured thresholds, supporting removal or flagging strategies.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from pamola_core.transformations.cleaning__old_02_05.cleaners.base_data_cleaner import BaseDataCleaner


# Configure logger
logger = logging.getLogger(__name__)


class NullHandlingCleaner(BaseDataCleaner):
    """
    Cleaner for handling missing values using threshold-based actions.
    Supports removing rows, fields, or flagging based on missing value stats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NullHandlingCleaner with configuration.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                - target_field (str): Column name to process.
                - action (str): One of 'remove_rows', 'remove_fields', or 'flag'.
                - threshold_type (str): 'count', 'percentage', or 'both'.
                - row_threshold (float): Threshold for row removal
                - field_threshold (float): Threshold for field removal
        """
        super().__init__(config)
        self.target_field = self.config.get("target_field", "")
        self.action = self.config.get("action", "remove_rows")
        self.threshold_type = self.config.get('threshold_type', "count")
        self.row_threshold = self.config.get('row_threshold', 1)
        self.field_threshold = self.config.get('field_threshold', 0.8)
 
    def execute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the imputation based on configuration or provided arguments.

        Args:
            df (pd.DataFrame): Input DataFrame containing missing values.
            **kwargs: Optional overrides for configuration:
                - target_field (str): Field to impute.
                - predictor_fields (list): Fields used to predict the target.
                - model_type (str): 'knn', 'random_forest', or 'linear_regression'.
                - model_params (dict): Model parameters.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        action = kwargs.get("action", self.action)
        target_field = kwargs.get("target_field", self.target_field)
        threshold_type = kwargs.get("threshold_type", self.threshold_type)
        row_threshold = kwargs.get("row_threshold", self.row_threshold)
        field_threshold = kwargs.get("field_threshold", self.field_threshold)

        if action == "remove_rows":
            return self._apply_remove_rows(df.copy(), target_field, threshold_type, row_threshold)

        if action == "remove_fields":
           return self._apply_remove_fields(df.copy(), target_field, threshold_type, field_threshold)

        if action == "flag":
            return self._apply_flag(df.copy(), target_field, threshold_type, row_threshold, field_threshold)

        raise ValueError(f"Unsupported action: {action}")

    def _apply_remove_rows(
            self,
            df: pd.DataFrame,
            target_field: str,
            threshold_type: str = "count",
            row_threshold: float = 1
    ) -> pd.DataFrame:
        """
        Handling missing values using removing records.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        frame_fields = self._get_info_df(df, target_field)
        exceed = self._check_threshold(frame_fields, target_field, row_threshold, threshold_type) 
        if exceed:
            df_clean = df.dropna(subset=[target_field])
            return df_clean
        return df

    def _apply_remove_fields(
            self,
            df: pd.DataFrame,
            target_field: str,
            threshold_type: str,
            field_threshold: float
    ) -> pd.DataFrame:
        """
        Handling missing values using removing fields.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        frame_fields = self._get_info_df(df, target_field)
        exceed = self._check_threshold(frame_fields, target_field, field_threshold, threshold_type)
        if exceed:
            df_clean = df.drop(columns=[target_field])
            return df_clean
        return df

    def _apply_flag(
            self,
            df: pd.DataFrame,
            target_field: str,
            threshold_type: str,
            row_threshold: float,
            field_threshold: float
    ) -> pd.DataFrame:
        """
        Filter the DataFrame based on missing value thresholds for a given field.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        target_field : str
            The column to check for missing value thresholds.
        threshold_type : str
            Type of threshold to apply ('count' or 'percentage').
        row_threshold : float
            Row-based threshold to compare against.
        field_threshold : float
            Field-level threshold to compare against.

        Returns:
        --------
        DataFrame
            Filtered DataFrame based on missing values.
        """
        frame_fields = self._get_info_df(df, target_field)
        row = frame_fields.loc[target_field]

        match threshold_type:
            case 'count':
                exceed = row['count'] > row_threshold or row['count'] > row_threshold
            case 'percentage':
                exceed = row['percentage'] > row_threshold or row['percentage'] > field_threshold
            case _:
                raise ValueError(f'Invalid threshold_type: {threshold_type}')

        if exceed:
            df_clean = df[df[target_field].isnull()]
            return df_clean
        return df

    def _get_info_df(
        self,
        df: pd.DataFrame,
        target_field: str
    ) -> pd.DataFrame:
        """
        Get information about missing values in the target field.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        target_field : str
            The column name to analyze missing values.

        Returns:
        --------
        DataFrame
            A DataFrame containing statistics about missing values.
        """
        try:
            df_rows_not_null = df[[target_field]].count().to_frame('not_null')
            df_rows_is_null = df[[target_field]].isnull().sum().to_frame('is_null')

            frame_fields = pd.concat([df_rows_not_null, df_rows_is_null], axis=1)
            frame_fields['total'] = frame_fields['not_null'] + frame_fields['is_null']
            frame_fields['count'] = frame_fields['is_null']
            frame_fields['percentage'] = frame_fields['is_null'] / frame_fields['total']


            return frame_fields

        except Exception as e:
            logger.exception(f"Error during missing value analysis in {target_field}: {str(e)}")
            raise


    def _check_threshold(
        self,
        frame: pd.DataFrame,
        field: str,
        threshold: float,
        threshold_type: Optional[str] = None,
    ) -> bool:
        """
        Check if the field exceeds the threshold based on the config.
        """
        threshold_type = threshold_type or self.threshold_type
        row = frame.loc[field]

        match threshold_type:
            case 'count':
                return row['count'] > threshold
            case 'percentage':
                return row['percentage'] > threshold
            case _:
                raise ValueError(f'Invalid threshold_type: {threshold_type}')