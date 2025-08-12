from enum import Enum

class UtilityMetricsType(str, Enum):
    """
    Enum representing supported utility metrics for evaluating
    the functional quality of data post-transformation.

    Regression Metrics:
    - R2: Coefficient of Determination (R²)
    - MSE: Mean Squared Error
    - MAE: Mean Absolute Error

    Classification Metrics:
    - AUROC: Area Under Receiver Operating Characteristic Curve
    - ACCURACY: Classification Accuracy
    - F1: F1-Score (harmonic mean of precision and recall)
    - PRECISION: Precision score
    - RECALL: Recall score
    """

    # Regression Metrics
    R2 = "r2" # R² Score
    MSE = "mse" # Mean Squared Error
    MAE = "mae" # Mean Absolute Error

    # Classification Metrics
    AUROC = "auroc" # Area Under ROC Curve
    ACCURACY = "accuracy" # Classification Accuracy
    F1 = "f1" # F1-Score
    PRECISION = "precision" # Precision Score
    RECALL = "recall" # Recall Score