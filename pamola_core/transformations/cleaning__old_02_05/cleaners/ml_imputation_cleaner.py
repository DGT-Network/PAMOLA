"""
Machine Learning Imputation Cleaner for missing value handling.

This module provides the MachineLearningImputationCleaner class for imputing missing values
using predictive models like KNN, Random Forest, or Linear Regression.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Type
from pamola_core.transformations.cleaning__old_02_05.cleaners.base_data_cleaner import BaseDataCleaner
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pamola_core.common.enum.model_type import ModelType

# Configure logger
logger = logging.getLogger(__name__)


class MachineLearningImputationCleaner(BaseDataCleaner):
    """
    Cleaner for imputing missing values using machine learning models.
    Supports KNN, Random Forest, and Linear Regression.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML Imputation Cleaner with configuration.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                - model_type (str): One of 'knn', 'random_forest', or 'linear_regression'.
                - target_field (str): Column name to impute.
                - predictor_fields (list of str): Column names used as features (for RF and LR).
                - model_params (dict): Optional parameters for model constructor.
        """
        super().__init__(config)
        self.model_type = self.config.get("model_type", ModelType.KNN.value)
        self.model_params = self.config.get("model_params", {})
        self.target_field = self.config.get("target_field", "")
        self.predictor_fields = self.config.get("predictor_fields", [])

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
        model_type = kwargs.get("model_type", self.model_type)
        model_params = kwargs.get("model_params", self.model_params)
        target_field = kwargs.get("target_field", self.target_field)
        predictor_fields = kwargs.get("predictor_fields", self.predictor_fields)

        if not target_field:
            raise ValueError("Missing 'target_field' for imputation.")

        if model_type == ModelType.KNN.value:
            return self.apply_knn_imputation(df.copy(), target_field, model_params)

        if model_type in {ModelType.RANDOM_FOREST.value, ModelType.LINEAR_REGRESSION.value} and not predictor_fields:
            raise ValueError("predictor_fields must be provided for supervised models.")

        if model_type == ModelType.RANDOM_FOREST.value:
            return self.apply_random_forest_imputation(df.copy(), target_field, predictor_fields, model_params)

        if model_type == ModelType.LINEAR_REGRESSION.value:
            return self.apply_linear_regression_imputation(df.copy(), target_field, predictor_fields, model_params)

        raise ValueError(f"Unsupported model_type: {model_type}")


    def apply_knn_imputation(self, df: pd.DataFrame, target_field: str, model_params: dict) -> pd.DataFrame:
        """
        Apply KNN imputation on a single column.

        Args:
            df (pd.DataFrame): DataFrame with missing values.
            target_field (str): Column to impute.
            model_params (dict): Parameters for sklearn KNNImputer.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        try:
            imputer = KNNImputer(**model_params)
            df[[target_field]] = imputer.fit_transform(df[[target_field]])
        except Exception as e:
            logger.error(f"KNN imputation failed for '{target_field}': {e}")
        return df

    def apply_random_forest_imputation(
        self, df: pd.DataFrame, target_field: str, predictor_fields: List[str], model_params: dict
    ) -> pd.DataFrame:
        """
        Apply Random Forest imputation.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_field (str): Column with missing values to impute.
            predictor_fields (list): Feature columns used to predict the target.
            model_params (dict): Parameters for sklearn RandomForestRegressor.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        return self.apply_model_imputation(df, target_field, predictor_fields, RandomForestRegressor, model_params)

    def apply_linear_regression_imputation(
        self, df: pd.DataFrame, target_field: str, predictor_fields: List[str], model_params: dict
    ) -> pd.DataFrame:
        """
        Apply Linear Regression imputation.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_field (str): Column with missing values to impute.
            predictor_fields (list): Feature columns used to predict the target.
            model_params (dict): Parameters for sklearn LinearRegression.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        return self.apply_model_imputation(df, target_field, predictor_fields, LinearRegression, model_params)

    def apply_model_imputation(
        self,
        df: pd.DataFrame,
        target_field: str,
        predictor_fields: List[str],
        model_cls: Type,
        model_params: dict
    ) -> pd.DataFrame:
        """
        General-purpose imputation using a supervised learning model.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_field (str): Column to impute.
            predictor_fields (list): Feature columns for prediction.
            model_cls (type): Regressor class (e.g., RandomForestRegressor).
            model_params (dict): Parameters for the model constructor.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        df_train = df[df[target_field].notna()]
        df_test = df[df[target_field].isna()]

        if df_train.empty or df_test.empty:
            logger.warning(f"Skipping imputation for '{target_field}' due to lack of train/test data.")
            return df

        try:
            model = model_cls(**model_params)
            model.fit(df_train[predictor_fields], df_train[target_field])
            predictions = model.predict(df_test[predictor_fields])
            df.loc[df[target_field].isna(), target_field] = predictions
        except Exception as e:
            logger.error(f"Imputation failed for '{target_field}' using {model_cls.__name__}: {e}")
        
        return df