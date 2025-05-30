from enum import Enum

class ModelType(str, Enum):
    KNN = "knn"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
