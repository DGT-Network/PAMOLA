"""
    PAMOLA Core - Cleaning Processors Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all Cleaning Processors operations
    in the PAMOLA Core library. It defines the strategies for handling missing values.

    Key features:
    - Basic statistical imputation.

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

from pandas.api.types import is_numeric_dtype
from pamola_core.cleaning__old_15_04.base import BaseCleaningProcessor

class BasicStatisticalImputation(BaseCleaningProcessor):
    """
    Operation for filling missing values using common statistical methods based on the distribution
    of existing values.
    """

    def __init__(
            self,
            target_fields: List[str],
            grouping_fields: List[str] | None = None,
            method: str = 'mean',
            apply_rounding: bool = False,
            handle_outliers: str = 'include',
            outlier_threshold: float = 3.0,
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
            Fields to impute null values.
        grouping_fields : list, optional
            Fields to group by before imputing.
        method : str, default 'mean'
            Statistical method to use.
        apply_rounding : bool, default False
            Whether to round numeric results.
        handle_outliers : str, default 'include'
            How to handle outliers.
        outlier_threshold : float, default 3.0
            Standard deviations for outlier detection.
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

        if grouping_fields is None:
            grouping_fields = []

        self.target_fields = target_fields
        self.grouping_fields = grouping_fields
        self.method = method
        self.apply_rounding = apply_rounding
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold

    def execute(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Operation for filling missing values using common statistical methods based on the distribution
        of existing values.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        kwargs : dict, optional
            Additional parameters for customization.
            - target_fields : list
                Fields to impute null values.
            - grouping_fields : list, optional
                Fields to group by before imputing.
            - method : str, default 'mean'
                Statistical method to use.
            - apply_rounding : bool, default False
                Whether to round numeric results.
            - handle_outliers : str, default 'include'
                How to handle outliers.
            - outlier_threshold : float, default 3.0
                Standard deviations for outlier detection.
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

        match self.method:
            case 'mean':
                return self.apply_mean(df)
            case 'median':
                return self.apply_median(df)
            case 'mode':
                return self.apply_mode(df)
            case _:
                raise ValueError(f'Invalid method: {self.method}')

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
                Fields to impute null values.
            - grouping_fields : list, optional
                Fields to group by before imputing.
            - method : str, default 'mean'
                Statistical method to use.
            - apply_rounding : bool, default False
                Whether to round numeric results.
            - handle_outliers : str, default 'include'
                How to handle outliers.
            - outlier_threshold : float, default 3.0
                Standard deviations for outlier detection.
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
        self.grouping_fields = kwargs.get('grouping_fields', self.grouping_fields)
        self.method = kwargs.get('method', self.method)
        self.apply_rounding = kwargs.get('apply_rounding', self.apply_rounding)
        self.handle_outliers = kwargs.get('handle_outliers', self.handle_outliers)
        self.outlier_threshold = kwargs.get('outlier_threshold', self.outlier_threshold)
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

        if set(self.target_fields).intersection(self.grouping_fields):
            raise ValueError('target_fields and grouping_fields have same value!')

        if df.empty:
            raise ValueError('DataFrame is empty!')

    def apply_mean(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handling missing values using common statistical method mean.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        for field in self.target_fields:
            if not is_numeric_dtype(df[field]):
                raise ValueError('target_fields must be numeric dtype!')

        if self.grouping_fields:
            df_grouped = df.groupby(self.grouping_fields)
        else:
            df_grouped = df

        df_clean = df.fillna(df_grouped[self.target_fields].transform(lambda x: x.fillna(x.mean())))

        return df_clean

    def apply_median(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handling missing values using common statistical method median.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        for field in self.target_fields:
            if not is_numeric_dtype(df[field]):
                raise ValueError('target_fields must be numeric dtype!')

        if self.grouping_fields:
            df_grouped = df.groupby(self.grouping_fields)
        else:
            df_grouped = df

        df_clean = df.fillna(df_grouped[self.target_fields].transform(lambda x: x.fillna(x.median())))

        return df_clean

    def apply_mode(
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
        if self.grouping_fields:
            df_grouped = df.groupby(self.grouping_fields)
        else:
            df_grouped = df

        df_clean = df.fillna(df_grouped[self.target_fields].transform(lambda x: x.fillna(x.mode().iloc[0])))

        return df_clean
