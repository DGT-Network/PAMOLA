from enum import Enum

class DistanceMetricType(str, Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"
