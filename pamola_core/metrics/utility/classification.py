"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Classification Utility Metric
Description: Implements the classifiers metric
Author: PAMOLA Core Team
License: BSD 3-Clause

This module implements the classifiers metric.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


class ClassificationUtility:
    """Implements the classifiers metric."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        cv_folds: int = 5,
        stratified: bool = True,
        test_size: float = 0.2,
    ):
        """
        Initialize.

        Parameters:
        -----------
        models: list.
            List of models (default: ["logistic", "rf", "svm"])
        metrics: list
            Metrics to calculation (default: ["accuracy", "f1", "roc_auc", "precision_recall_tradeoff"])
        cv_folds: int
            Cross validation folds (default: 5)
        stratified: bool
            Whether to use stratified sampling (default: True)
        test_size: float
            Proportion of the dataset to include in the test set (default: 0.2)
        """
        if models is None:
            models = ["logistic", "rf", "svm"]

        if metrics is None:
            metrics = ["accuracy", "f1", "roc_auc", "precision_recall_tradeoff"]

        self.models = models
        self.metrics = metrics
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.test_size = test_size

        self.model_dict = {
            "logistic": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(),
            "svm": SVC(probability=True),
        }

    def calculate_metric(
        self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, value_field: str
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
            Target field for analysis.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of metric results.
        """
        # Initialize results dictionary
        results = {}

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
            if self.stratified:
                cv = StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=42
                )
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        elif self.test_size > 0:
            X_original_train, X_original_test, y_original_train, y_original_test = (
                train_test_split(
                    X_original,
                    y_original,
                    test_size=self.test_size,
                    random_state=42,
                    shuffle=True,
                )
            )

            (
                X_transformed_train,
                X_transformed_test,
                y_transformed_train,
                y_transformed_test,
            ) = train_test_split(
                X_transformed,
                y_transformed,
                test_size=self.test_size,
                random_state=42,
                shuffle=True,
            )

        for model_name in self.models:
            model = self.model_dict[model_name]

            # Store values for each fold
            accuracies = []
            f1_scores = []
            roc_auc_scores = []
            precision_values = []
            recall_values = []
            thresholds_values = []

            if self.cv_folds > 2:
                for train_index, test_index in cv.split(X_original, y_original):
                    X_train, X_test = (
                        X_original.iloc[train_index],
                        X_transformed.iloc[test_index],
                    )
                    y_train, y_test = (
                        y_original.iloc[train_index],
                        y_transformed.iloc[test_index],
                    )

                    self._calculate_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        metrics=self.metrics,
                        accuracies=accuracies,
                        f1_scores=f1_scores,
                        roc_auc_scores=roc_auc_scores,
                        precision_values=precision_values,
                        recall_values=recall_values,
                        thresholds_values=thresholds_values,
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
                    metrics=self.metrics,
                    accuracies=accuracies,
                    f1_scores=f1_scores,
                    roc_auc_scores=roc_auc_scores,
                    precision_values=precision_values,
                    recall_values=recall_values,
                    thresholds_values=thresholds_values,
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
                    metrics=self.metrics,
                    accuracies=accuracies,
                    f1_scores=f1_scores,
                    roc_auc_scores=roc_auc_scores,
                    precision_values=precision_values,
                    recall_values=recall_values,
                    thresholds_values=thresholds_values,
                )

            # Aggregate metrics per model
            model_results = {}

            if "accuracy" in self.metrics:
                model_results["accuracy"] = float(np.mean(accuracies))

            if "f1" in self.metrics:
                model_results["f1"] = float(np.mean(f1_scores))

            if "roc_auc" in self.metrics:
                model_results["roc_auc"] = float(np.mean(roc_auc_scores))

            if "precision_recall_tradeoff" in self.metrics:
                precision_dict = self._calculate_precision_recall_tradeoff(
                    tradeoff_values=precision_values
                )
                recall_dict = self._calculate_precision_recall_tradeoff(
                    tradeoff_values=recall_values
                )
                thresholds_dict = self._calculate_precision_recall_tradeoff(
                    tradeoff_values=thresholds_values
                )

                # Calculate the area under the precision-recall curve (PR AUC)
                pr_auc_dict = {}
                for key in precision_dict.keys():
                    pr_auc_dict[key] = float(auc(recall_dict[key], precision_dict[key]))

                model_results["pr_auc"] = float(np.mean(list(pr_auc_dict.values())))
                model_results["precision_recall_tradeoff"] = {
                    "pr_auc": pr_auc_dict,
                    "precision": precision_dict,
                    "recall": recall_dict,
                    "thresholds": thresholds_dict,
                }

            if model_results:
                results[model_name] = model_results

        return results

    def _calculate_model(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        metrics: List[str],
        accuracies: List[float],
        f1_scores: List[Any],
        roc_auc_scores: List[float],
        precision_values: List[Any],
        recall_values: List[Any],
        thresholds_values: List[Any],
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
        metrics: list
            Metrics to calculation.
        accuracies: list
            Accuracies.
        f1_scores: list
            F1 scores.
        roc_auc_scores: list
            ROC AUC scores.
        precision_values: list
            Precision values.
        recall_values: list
            Recall values.
        thresholds_values: list
            Thresholds values.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Calculate metrics
        if "accuracy" in metrics:
            accuracies.append(accuracy_score(y_test, y_pred))

        if "f1" in metrics:
            f1_scores.append(f1_score(y_test, y_pred, average="macro"))

        if "roc_auc" in metrics:
            if len(set(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                y_bin = label_binarize(y_test, classes=np.unique(y_test))
                roc_auc = roc_auc_score(
                    y_bin, y_prob, average="macro", multi_class="ovr"
                )

            roc_auc_scores.append(float(roc_auc))

        if "precision_recall_tradeoff" in metrics:
            classes_precision = {}
            classes_recall = {}
            classes_thresholds = {}

            if len(set(y_test)) == 2:
                precision, recall, thresholds = precision_recall_curve(
                    y_test, y_prob[:, 1]
                )

                classes_precision["1"] = precision.tolist()
                classes_recall["1"] = recall.tolist()
                classes_thresholds["1"] = thresholds.tolist()
            else:
                y_bin = label_binarize(y_test, classes=np.unique(y_test))
                for i in range(len(np.unique(y_test))):
                    precision, recall, thresholds = precision_recall_curve(
                        y_bin[:, i], y_prob[:, i]
                    )

                    classes_precision[str(i)] = precision.tolist()
                    classes_recall[str(i)] = recall.tolist()
                    classes_thresholds[str(i)] = thresholds.tolist()

            precision_values.append(classes_precision)
            recall_values.append(classes_recall)
            thresholds_values.append(classes_thresholds)

    def _calculate_precision_recall_tradeoff(
        self, tradeoff_values: List[Any]
    ) -> Dict[str, Any]:
        """
        Calculate precision recall tradeoff.

        Parameters:
        -----------
        tradeoff_values: list
            Precision values / Recall values/ Thresholds values.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of precision recall tradeoff.
        """
        ordered_keys = list(tradeoff_values[0].keys())
        seen = set(ordered_keys)

        for d in tradeoff_values[1:]:
            for k in d.keys():
                if k not in seen:
                    ordered_keys.append(k)
                    seen.add(k)

        tradeoff_mean_dict = {}
        for key in ordered_keys:

            tradeoffs = [d[key] for d in tradeoff_values if key in d]

            if not tradeoffs:
                continue

            if len(tradeoffs) > 1:
                max_length = max(len(tradeoff) for tradeoff in tradeoffs)
                padded = [t + [0.0] * (max_length - len(t)) for t in tradeoffs]
                avg_tradeoff = np.mean(padded, axis=0)
                tradeoff_mean_dict[key] = avg_tradeoff.tolist()
            else:
                tradeoff_mean_dict[key] = tradeoffs[0]

        return tradeoff_mean_dict
