from enum import Enum

class StatisticalMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
