"""
    PAMOLA Core - Cleaning Processors Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all Cleaning Processors operations
    in the PAMOLA Core library. It defines the strategies for handling missing values.

    Key features:
    - Machine learning imputation.

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
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pamola_core.cleaning.base import BaseCleaningProcessor

class MachineLearningImputation(BaseCleaningProcessor):
    """
    Operation for filling missing values using predictive models trained on non-missing data to
    predict missing values.
    """

    def __init__(
            self,
            target_fields: List[str],
            model_type: str = 'knn',
            model_params: dict | None = None,
            predictor_fields: List[str] | None = None,
            cross_validation: bool = False,
            cv_folds: int = 5,
            feature_scaling: bool = True,
            random_state: int = 42,
            null_markers: List[str] | None = None,
            treat_empty_as_null: bool = True,
            track_process: bool = True,
            process_log: str = None
    ):
        """
        Initializes object with configurable options.

        Parameters:
        -----------
        target_fields : list
            Fields to impute null values.
        model_type : str, default 'knn'
            ML model to use.
        model_params : dict, optional
            Parameters for the ML model.
        predictor_fields : list, optional
            Fields to use as predictors.
        cross_validation : bool, default False
            Whether to use cross-validation.
        cv_folds : int, default 5
            Number of cross-validation folds.
        feature_scaling : bool, default True
            Whether to scale features.
        random_state : int, default 42
            Seed for random operations.
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

        if model_params is None:
            model_params = {}

        if predictor_fields is None:
            predictor_fields = []

        self.target_fields = target_fields
        self.model_type = model_type
        self.model_params = model_params
        self.predictor_fields = predictor_fields
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.feature_scaling = feature_scaling
        self.random_state = random_state

    def execute(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Operation for filling missing values using predictive models trained on non-missing data to
        predict missing values.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.
        kwargs : dict, optional
            Additional parameters for customization.
            - target_fields : list
                Fields to impute null values.
            - model_type : str, default 'knn'
                ML model to use.
            - model_params : dict, optional
                Parameters for the ML model.
            - predictor_fields : list, optional
                Fields to use as predictors.
            - cross_validation : bool, default False
                Whether to use cross-validation.
            - cv_folds : int, default 5
                Number of cross-validation folds.
            - feature_scaling : bool, default True
                Whether to scale features.
            - random_state : int, default 42
                Seed for random operations.
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

        match self.model_type:
            case 'knn':
                return self.apply_knn(df)
            case 'random_forest':
                return self.apply_random_forest(df)
            case 'linear_regression':
                return self.apply_linear_regression(df)
            case _:
                raise ValueError(f'Invalid model_type: {self.model_type}')

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
            - model_type : str, default 'knn'
                ML model to use.
            - model_params : dict, optional
                Parameters for the ML model.
            - predictor_fields : list, optional
                Fields to use as predictors.
            - cross_validation : bool, default False
                Whether to use cross-validation.
            - cv_folds : int, default 5
                Number of cross-validation folds.
            - feature_scaling : bool, default True
                Whether to scale features.
            - random_state : int, default 42
                Seed for random operations.
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
        self.model_type = kwargs.get('model_type', self.model_type)
        self.model_params = kwargs.get('model_params', self.model_params)
        self.predictor_fields = kwargs.get('predictor_fields', self.predictor_fields)
        self.cross_validation = kwargs.get('cross_validation', self.cross_validation)
        self.cv_folds = kwargs.get('cv_folds', self.cv_folds)
        self.feature_scaling = kwargs.get('feature_scaling', self.feature_scaling)
        self.random_state = kwargs.get('random_state', self.random_state)
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

        if len(self.target_fields) > 1:
            raise ValueError('target_fields must have 1 value!!')

    def apply_knn(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handling missing values using K-Nearest Neighbors.

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

        imputer = KNNImputer(**self.model_params)
        arr_np = imputer.fit_transform(df[self.target_fields])
        df_clean = df.fillna(pd.DataFrame(arr_np, columns=self.target_fields))

        return df_clean

    def apply_random_forest(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handling missing values using Random Forest.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        target_field = self.target_fields[0]

        df_with_not_null = df[df[self.target_fields].notnull().any(axis=1)]
        df_with_is_null = df[df[self.target_fields].isnull().any(axis=1)]

        train_set = df_with_not_null[self.predictor_fields+self.target_fields]
        x_train_set = train_set[self.predictor_fields]
        y_train_set = train_set[target_field]

        test_set = df_with_is_null[self.predictor_fields+self.target_fields]
        x_test_set = test_set[self.predictor_fields]
        y_test_set = test_set[target_field]

        random_forest_regressor = RandomForestRegressor(**self.model_params)
        random_forest_regressor.fit(x_train_set, y_train_set)
        y_predict_set = random_forest_regressor.predict(x_test_set)

        test_set.loc[test_set[target_field].isnull(), target_field] = y_predict_set

        df_clean = pd.concat([train_set, test_set]).sort_index()

        return df_clean

    def apply_linear_regression(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handling missing values using Linear Regression.

        Parameters:
        -----------
        df : DataFrame
            The input data to be processed.

        Returns:
        --------
        DataFrame
            Processed data.
        """
        target_field = self.target_fields[0]

        df_with_not_null = df[df[self.target_fields].notnull().any(axis=1)]
        df_with_is_null = df[df[self.target_fields].isnull().any(axis=1)]

        train_set = df_with_not_null[self.predictor_fields+self.target_fields]
        x_train_set = train_set[self.predictor_fields]
        y_train_set = train_set[target_field]

        test_set = df_with_is_null[self.predictor_fields+self.target_fields]
        x_test_set = test_set[self.predictor_fields]
        y_test_set = test_set[target_field]

        linear_regression = LinearRegression(**self.model_params)
        linear_regression.fit(x_train_set, y_train_set)
        y_predict_set = linear_regression.predict(x_test_set)

        test_set.loc[test_set[target_field].isnull(), target_field] = y_predict_set

        df_clean = pd.concat([train_set, test_set]).sort_index()

        return df_clean
