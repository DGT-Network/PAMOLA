"""
    PAMOLA Core - Cleaning Processors Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all Cleaning Processors operations
    in the PAMOLA Core library. It defines the strategies for handling missing values.

    Key features:
    - Null detection and removal.

    This module serves as the foundation for all Cleaning Processors operations in PAMOLA Core.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:
        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd

from typing import List

from pamola_core.cleaning__old_15_04.base import BaseCleaningProcessor

class NullDetectionAndRemoval(BaseCleaningProcessor):
    """
    Operation for identifying and removing records or fields with null values based on specified
    criteria and thresholds.
    """

    def __init__(
            self,
            target_fields: List[str],
            action: str = 'remove_rows',
            threshold_type: str = 'count',
            row_threshold: float = 1.0,
            field_threshold: float = 0.8,
            null_markers: List[str] | None = None,
            treat_empty_as_null: bool = True,
            track_process: bool = True,
            process_log: str = None,
    ):
        """
        Initializes object with configurable options.

        Parameters:
        -----------
        target_fields : list
            Fields to check for null values.
        action : str, default 'remove_rows'
            Action to take on nulls.
        threshold_type : str, default 'count'
            Type of threshold to apply.
        row_threshold : float, default 1.0
            Threshold for row removal.
        field_threshold : float, default 0.8
            Threshold for field removal.
        null_markers : list, optional
            Additional values to treat as null.
        treat_empty_as_null : bool, default True
            Whether to treat empty strings as null.
        track_process : bool, default True
            Whether to track processed values.
        process_log : str, optional
            Path to save processing log.
        """
        super().__init__(
            null_markers = null_markers,
            treat_empty_as_null = treat_empty_as_null,
            track_process = track_process,
            process_log = process_log
        )

        self.target_fields = target_fields
        self.action = action
        self.threshold_type = threshold_type
        self.row_threshold = row_threshold
        self.field_threshold = field_threshold

    def execute(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Operation for identifying and removing records or fields with null values based on specified
        criteria and thresholds.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        kwargs : dict, optional
            Additional parameters for customization.
            - target_fields : list
                Fields to check for null values.
            - action : str, default 'remove_rows'
                Action to take on nulls.
            - threshold_type : str, default 'count'
                Type of threshold to apply.
            - row_threshold : float, default 1.0
                Threshold for row removal.
            - field_threshold : float, default 0.8
                Threshold for field removal.
            - null_markers : list, optional
                Additional values to treat as null.
            - treat_empty_as_null : bool, default True
                Whether to treat empty strings as null.
            - track_process : bool, default True
                Whether to track processed values.
            - process_log : str, optional
                Path to save processing log.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        self.get_custom_parameters(**kwargs)
        self.validate(df)

        match self.action:
            case 'remove_rows':
                return self.apply_remove_rows(df)
            case 'remove_fields':
                return self.apply_remove_fields(df)
            case 'flag':
                return self.apply_flag(df)
            case _:
                raise ValueError(f'Invalid action: {self.action}')

    def get_custom_parameters(
            self,
            **kwargs
    ) -> None:
        """
        Get custom parameters from kwargs.

        Parameters:
        -----------
        kwargs : dict, optional
            Additional parameters for customization.
            - target_fields : list
                Fields to check for null values.
            - action : str, default 'remove_rows'
                Action to take on nulls.
            - threshold_type : str, default 'count'
                Type of threshold to apply.
            - row_threshold : float, default 1.0
                Threshold for row removal.
            - field_threshold : float, default 0.8
                Threshold for field removal.
            - null_markers : list, optional
                Additional values to treat as null.
            - treat_empty_as_null : bool, default True
                Whether to treat empty strings as null.
            - track_process : bool, default True
                Whether to track processed values.
            - process_log : str, optional
                Path to save processing log.
        """
        self.target_fields = kwargs.get('target_fields', self.target_fields)
        self.action = kwargs.get('action', self.action)
        self.threshold_type = kwargs.get('threshold_type', self.threshold_type)
        self.row_threshold = kwargs.get('row_threshold', self.row_threshold)
        self.field_threshold = kwargs.get('field_threshold', self.field_threshold)
        self.null_markers = kwargs.get('null_markers', self.null_markers)
        self.treat_empty_as_null = kwargs.get('treat_empty_as_null', self.treat_empty_as_null)
        self.track_process = kwargs.get('track_process', self.track_process)
        self.process_log = kwargs.get('process_log', self.process_log)

    def validate(
            self,
            df: pd.DataFrame
    ) -> None:
        """
        Validate data.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        """
        if not self.target_fields:
            raise ValueError('target_fields must have value!')

        if df.empty:
            raise ValueError('DataFrame is empty!')

        if self.target_fields == ['*']:
            self.target_fields = df.columns.tolist()

    def apply_remove_rows(
            self,
            df: pd.DataFrame
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
        frame_fields = self.get_info_df(df)

        threshold_fields = {field: False for field in self.target_fields}
        match self.threshold_type:
            case 'count':
                for index, row in frame_fields.iterrows():
                    if row['count'] > self.row_threshold:
                        threshold_fields[str(index)] = True
            case 'percentage':
                for index, row in frame_fields.iterrows():
                    if row['percentage'] > self.row_threshold:
                        threshold_fields[str(index)] = True
            case 'both':
                ...
            case _:
                raise ValueError(f'Invalid threshold_type: {self.threshold_type}')

        subset_fields = [key for key in threshold_fields if threshold_fields[key]]
        df_clean = df.dropna(subset=subset_fields)

        return df_clean

    def apply_remove_fields(
            self,
            df: pd.DataFrame
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
        frame_fields = self.get_info_df(df)

        threshold_fields = {field: False for field in self.target_fields}
        match self.threshold_type:
            case 'count':
                for index, row in frame_fields.iterrows():
                    if row['count'] > self.field_threshold:
                        threshold_fields[str(index)] = True
            case 'percentage':
                for index, row in frame_fields.iterrows():
                    if row['percentage'] > self.field_threshold:
                        threshold_fields[str(index)] = True
            case 'both':
                ...
            case _:
                raise ValueError(f'Invalid threshold_type: {self.threshold_type}')

        columns_fields = [key for key in threshold_fields if threshold_fields[key]]
        df_clean = df.drop(columns=columns_fields)

        return df_clean

    def apply_flag(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handling missing values using common statistical method mode.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        frame_fields = self.get_info_df(df)

        threshold_fields = {field: False for field in self.target_fields}
        match self.threshold_type:
            case 'count':
                for index, row in frame_fields.iterrows():
                    if (row['count'] > self.row_threshold
                            or row['count'] > self.field_threshold):
                        threshold_fields[str(index)] = True
            case 'percentage':
                for index, row in frame_fields.iterrows():
                    if (row['percentage'] > self.row_threshold
                            or row['percentage'] > self.field_threshold):
                        threshold_fields[str(index)] = True
            case 'both':
                ...
            case _:
                raise ValueError(f'Invalid threshold_type: {self.threshold_type}')

        columns_fields = [key for key in threshold_fields if threshold_fields[key]]
        df_clean = df[df[columns_fields].isnull().any(axis=1)]

        return df_clean

    def get_info_df(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get information of DataFrame.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Result data.
        """
        df_rows_not_null = df[self.target_fields].count().to_frame('not_null')
        df_rows_is_null = df[self.target_fields].isnull().sum().to_frame('is_null')

        frame_fields = pd.concat([df_rows_not_null, df_rows_is_null], axis=1)
        frame_fields['total'] = frame_fields['not_null'] + frame_fields['is_null']
        frame_fields['count'] = frame_fields['is_null']
        frame_fields['percentage'] = frame_fields['is_null'] / frame_fields['total']

        return frame_fields
