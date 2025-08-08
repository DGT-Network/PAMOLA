"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Regression Utility Metric
Description: Implements the regressors metric
Author: PAMOLA Core Team
License: BSD 3-Clause

This module implements the regressors metric.
"""


from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


class RegressionUtility:
    """Implements the regressors metric."""

    def __init__(
            self,
            models: Optional[List[str]] = None,
            metrics: Optional[List[str]] = None,
            cv_folds: int = 5,
            test_size: float = 0.2
    ):
        """
        Initialize.

        Parameters:
        -----------
        models: list.
            List of models (default: ["linear", "rf", "svr"])
        metrics: list
            Metrics to calculation (default: ["r2", "mae", "mse", "rmse"])
        cv_folds: int
            Cross validation folds (default: 5)
        test_size: float
            Proportion of the dataset to include in the test set (default: 0.2)
        """
        if models is None:
            models = ["linear", "rf", "svr"]

        if metrics is None:
            metrics = ["r2", "mae", "mse", "rmse"]

        self.models = models
        self.metrics = metrics
        self.cv_folds = cv_folds
        self.test_size = test_size

        self.specific_metrics = [] # "pmse"
        self.specific_model_dict = {
            "logistic": LogisticRegression(max_iter=1000)
        }

        self.base_metrics = [metric for metric in self.metrics if metric not in self.specific_metrics]
        self.model_dict = {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(),
            "svr": SVR()
        }

    def calculate_metric(
            self,
            original_df: pd.DataFrame,
            transformed_df: pd.DataFrame,
            value_field: str,
            key_fields: Optional[List[str]] = None,
            aggregation: str = "sum"
    ) -> Dict[str, Any]:
        """
        Calculate metrics.

        Parameters:
        -----------
        original_df: pd.DataFrame
            Original DataFrame.
        transformed_df: pd.DataFrame
            Transformed DataFrame.
        value_field: str
            Target field for analysis
        key_fields: list
            Fields to use for grouped R² calculation
        aggregation: str
            Aggregation method for grouped R² calculation

        Returns:
        --------
        Dict[str, Any]
            Dictionary of metric results.
        """
        # Initialize results dictionary
        results = {}

        # Standard model-based metrics
        model_based = self._calculate_model_based(
            original_df=original_df.copy(deep=True),
            transformed_df=transformed_df.copy(deep=True),
            value_field=value_field
        )
        results.update(model_based)

        # Grouped R2
        if key_fields and value_field:
            r2_grouped = self._calculate_grouped_r2(
                original_df=original_df.copy(deep=True),
                transformed_df=transformed_df.copy(deep=True),
                key_fields=key_fields,
                value_field=value_field,
                aggregation=aggregation
            )
            results["grouped_r2"] = r2_grouped

        return results

    def _calculate_model_based(
            self,
            original_df: pd.DataFrame,
            transformed_df: pd.DataFrame,
            value_field: str
    ) -> Dict[str, Any]:
        """
        Calculate model based metrics.

        Parameters:
        -----------
        original_df: pd.DataFrame
            Original DataFrame.
        transformed_df: pd.DataFrame
            Transformed DataFrame.
        value_field: str
            Target field for analysis

        Returns:
        --------
        Dict[str, Any]
            Dictionary of metric results.
        """
        model_based = {}

        # Setup data frame & cross validation
        original_df = pd.get_dummies(original_df)
        transformed_df = pd.get_dummies(transformed_df)

        X_original = original_df.drop(columns=[value_field])
        y_original = original_df[value_field]

        X_transformed = transformed_df.drop(columns=[value_field])
        y_transformed = transformed_df[value_field]

        cv = None
        X_original_train = None
        y_original_train = None
        X_original_test = None
        y_original_test = None
        X_transformed_train = None
        y_transformed_train = None
        X_transformed_test = None
        y_transformed_test = None
        if self.cv_folds > 2:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        elif self.test_size > 0:
            X_original_train, X_original_test, y_original_train, y_original_test = train_test_split(
                X_original, y_original, test_size=self.test_size, random_state=42, shuffle=True
            )

            X_transformed_train, X_transformed_test, y_transformed_train, y_transformed_test = train_test_split(
                X_transformed, y_transformed, test_size=self.test_size, random_state=42, shuffle=True
            )

        for model_name in self.models:
            model = self.model_dict[model_name]

            # Store values for each fold
            r2_scores = []
            mae_scores = []
            mse_scores = []
            rmse_scores = []

            if self.cv_folds > 2:
                for train_index, test_index in cv.split(X_original, y_original):
                    X_train, X_test = X_original.iloc[train_index], X_transformed.iloc[test_index]
                    y_train, y_test = y_original.iloc[train_index], y_transformed.iloc[test_index]

                    self._calculate_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        metrics=self.base_metrics,
                        r2_scores=r2_scores,
                        mae_scores=mae_scores,
                        mse_scores=mse_scores,
                        rmse_scores=rmse_scores,
                        pmse_scores=[]
                    )

            elif self.test_size > 0:
                X_train, X_test = X_original_train, X_transformed_test
                y_train, y_test = y_original_train, y_transformed_test

                self._calculate_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    metrics=self.base_metrics,
                    r2_scores=r2_scores,
                    mae_scores=mae_scores,
                    mse_scores=mse_scores,
                    rmse_scores=rmse_scores,
                    pmse_scores=[]
                )

            else:
                X_train, X_test = X_original, X_transformed
                y_train, y_test = y_original, y_transformed

                self._calculate_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    metrics=self.base_metrics,
                    r2_scores=r2_scores,
                    mae_scores=mae_scores,
                    mse_scores=mse_scores,
                    rmse_scores=rmse_scores,
                    pmse_scores=[]
                )

            # Aggregate metrics per model
            model_results = {}

            if "r2" in self.metrics:
                model_results["r2"] = float(np.mean(r2_scores))

            if "mae" in self.metrics:
                model_results["mae"] = float(np.mean(mae_scores))

            if "mse" in self.metrics:
                model_results["mse"] = float(np.mean(mse_scores))

            if "rmse" in self.metrics:
                model_results["rmse"] = float(np.mean(rmse_scores))

            if model_results:
                model_based[model_name] = model_results

        # Calculate for pMSE with specific model
        X_combined = pd.concat([X_original, X_transformed], ignore_index=True)
        y_combined = pd.Series(np.concatenate([np.zeros(len(X_original)), np.ones(len(X_transformed))]))

        X_combined_train = X_combined_test = y_combined_train = y_combined_test = None
        if self.cv_folds <= 2 and self.test_size > 0:
            X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
                X_combined, y_combined, test_size=self.test_size, random_state=42, shuffle=True
            )

        for model_name, model in self.specific_model_dict.items():
            # Store values for each fold
            pmse_scores = []

            if self.cv_folds > 2:
                for train_index, test_index in cv.split(X_combined, y_combined):
                    X_train, X_test = X_combined.iloc[train_index], X_combined.iloc[test_index]
                    y_train, y_test = y_combined.iloc[train_index], y_combined.iloc[test_index]

                    self._calculate_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        metrics=self.specific_metrics,
                        r2_scores=[],
                        mae_scores=[],
                        mse_scores=[],
                        rmse_scores=[],
                        pmse_scores=pmse_scores
                    )

            elif self.test_size > 0:
                X_train, X_test = X_combined_train, X_combined_test
                y_train, y_test = y_combined_train, y_combined_test

                self._calculate_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    metrics=self.specific_metrics,
                    r2_scores=[],
                    mae_scores=[],
                    mse_scores=[],
                    rmse_scores=[],
                    pmse_scores=pmse_scores
                )

            else:
                X_train, X_test = X_combined, X_combined
                y_train, y_test = y_combined, y_combined

                self._calculate_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    metrics=self.specific_metrics,
                    r2_scores=[],
                    mae_scores=[],
                    mse_scores=[],
                    rmse_scores=[],
                    pmse_scores=pmse_scores
                )

            # Aggregate metrics per model
            model_results = {}

            if "pmse" in self.metrics:
                model_results["pmse"] = float(np.mean(pmse_scores))

            if model_results:
                model_based[model_name] = model_results

        return model_based

    def _calculate_model(
            self,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            metrics,
            r2_scores,
            mae_scores,
            mse_scores,
            rmse_scores,
            pmse_scores
    ) -> None:
        """
        Calculate model.

        Parameters:
        -----------
        model: ClassifierMixin
            Classifier Mixin.
        X_train: Any
            Data training.
        y_train: Any
            Data training.
        X_test: Any
            Data testing.
        y_test: Any
            Data testing.
        r2_scores: list
            R2 scores.
        mae_scores: list
            MAE scores.
        mse_scores: list
            MSE scores.
        rmse_scores: list
            RMSE scores.
        pmse_scores: list
            pMSE scores.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        if "r2" in metrics:
            r2_scores.append(r2_score(y_test, y_pred))

        if "mae" in metrics:
            mae_scores.append(mean_absolute_error(y_test, y_pred))

        if "mse" in metrics:
            mse_scores.append(mean_squared_error(y_test, y_pred))

        if "rmse" in metrics:
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Calculate specific metrics
        if "pmse" in metrics:
            y_prob = model.predict_proba(X_test)[:, 1]
            pmse_scores.append(mean_squared_error(y_test, y_prob))

    def _calculate_grouped_r2(
            self,
            original_df: pd.DataFrame,
            transformed_df: pd.DataFrame,
            value_field: str,
            key_fields: List[str],
            aggregation: str
    ) -> Dict[str, Any]:
        """
        Calculate R² for grouped data.

        Parameters:
        -----------
        original_df: pd.DataFrame
            Original DataFrame.
        transformed_df: pd.DataFrame
            Transformed DataFrame.
        value_field: str
            Target field for analysis
        key_fields: list
            Fields to use for grouped R² calculation
        aggregation: str
            Aggregation method for grouped R² calculation

        Returns:
        --------
        Dict[str, Any]
            Dictionary of metric results.
        """
        # Create aggregated dictionaries
        x_dict = self._create_value_dictionary(
            df=original_df,
            value_field=value_field,
            key_fields=key_fields,
            aggregation=aggregation
        )
        y_dict = self._create_value_dictionary(
            df=transformed_df,
            value_field=value_field,
            key_fields=key_fields,
            aggregation=aggregation
        )

        # Get common keys
        common_keys = x_dict.keys() & y_dict.keys()
        n = len(common_keys)

        if n == 0:
            return {"r_squared": 0.0, "n_points": 0}

        # Calculate sums
        sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0

        x_values = []
        y_values = []

        for key in common_keys:
            x = x_dict[key]
            y = y_dict[key]

            x_values.append(x)
            y_values.append(y)

            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y

        # Calculate means
        mean_x = sum_x / n
        mean_y = sum_y / n

        # Calculate regression coefficient (slope)
        if (sum_x2 - n * mean_x * mean_x) != 0:
            slope = (sum_xy - n * mean_x * mean_y) / (sum_x2 - n * mean_x * mean_x)
            intercept = mean_y - slope * mean_x
        else:
            return {"r_squared": 0.0, "n_points": n, "error": "Zero variance in x"}

        # Calculate SStot and SSres
        ss_tot = ss_res = 0.0

        for i, key in enumerate(common_keys):
            x = x_values[i]
            y = y_values[i]
            y_pred = slope * x + intercept

            ss_tot += (y - mean_y) ** 2
            ss_res += (y - y_pred) ** 2

        # Calculate R²
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0.0

        return {
            "r_squared": float(r_squared),
            "n_points": n,
            "slope": slope,
            "intercept": intercept,
            "mean_x": mean_x,
            "mean_y": mean_y
        }

    def _create_value_dictionary(
            self,
            df: pd.DataFrame,
            value_field: str,
            key_fields: List[str],
            aggregation: str = "sum"
    ) -> Dict[str, Any]:
        """
        Create aggregated value dictionary.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        value_field : str, optional
            Field to aggregate. If None, performs count
        key_fields : list
            Fields to use as composite key
        aggregation : str
            Aggregation function: "sum", "mean", "min", "max", "count", "first", "last"

        Returns:
        --------
        Dict[str, Any]
            Dictionary with composite keys and aggregated values
        """
        if value_field is None:
            # Count occurrences
            grouped = df.groupby(key_fields).size()
        else:
            # Group and aggregate
            grouped = df.groupby(key_fields)[value_field].aggregate(func=aggregation)

        # Convert to dictionary with composite string keys
        result = {}
        for key, value in grouped.items():
            if isinstance(key, tuple):
                composite_key = "_".join(str(k) for k in key)
            else:
                composite_key = str(key)

            result[composite_key] = float(value)

        return result
