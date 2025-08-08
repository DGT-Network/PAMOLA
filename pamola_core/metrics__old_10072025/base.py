"""
PAMOLA.CORE - Base Metrics Module
---------------------------------
This module defines the foundational classes and interfaces for all metrics
in the PAMOLA.CORE system. It provides a unified approach to calculating,
reporting, and interpreting various metrics for privacy, utility, and fidelity.

Key concepts:
- Abstract base classes for all metrics types
- Common interfaces for consistent metric calculation
- Support for different data types and structures
- Standardized result formats and interpretation

The metrics system is organized into several categories:
- Privacy: Metrics related to anonymity, disclosure risk, and re-identification
- Utility: Metrics for measuring information loss and data usefulness
- Fidelity: Metrics for measuring statistical preservation and data quality

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """
    Abstract base class for all metrics in PAMOLA.CORE.

    This class defines the common interface for calculating metrics
    on datasets, ensuring consistent behavior across different
    metric types.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a metric with name and description.

        Parameters:
        -----------
        name : str
            The name of the metric.
        description : str
            A brief description of what the metric measures.
        """
        self.name = name
        self.description = description
        self.last_result = None

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Calculate the metric value based on provided data.

        Parameters will vary based on the specific metric implementation.

        Returns:
        --------
        dict
            Dictionary containing metric values and any additional information.
        """
        pass

    def interpret(self, value: float) -> str:
        """
        Provide an interpretation of the metric value.

        Parameters:
        -----------
        value : float
            The metric value to interpret.

        Returns:
        --------
        str
            Human-readable interpretation of the metric value.
        """
        # Default implementation - should be overridden by subclasses
        return f"{self.name}: {value}"

    def get_last_result(self) -> Dict[str, Any]:
        """
        Get the result of the last calculation.

        Returns:
        --------
        dict or None
            The result of the last calculation, or None if no calculation
            has been performed yet.
        """
        return self.last_result


class PrivacyMetric(BaseMetric):
    """
    Base class for privacy-related metrics.

    Privacy metrics quantify the risk of re-identification, disclosure,
    or privacy leakage in anonymized or synthesized data.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a privacy metric.

        Parameters:
        -----------
        name : str
            The name of the metric.
        description : str
            A brief description of what the metric measures.
        """
        super().__init__(name, description)

    def interpret(self, value: float) -> str:
        """
        Interpret a privacy metric value.

        For privacy metrics, lower values typically indicate better privacy
        (less risk of disclosure).

        Parameters:
        -----------
        value : float
            The privacy metric value (typically a risk percentage).

        Returns:
        --------
        str
            Human-readable interpretation of the privacy risk.
        """
        if value < 5:
            return f"{self.name}: {value:.2f}% - Very low risk"
        elif value < 10:
            return f"{self.name}: {value:.2f}% - Low risk"
        elif value < 20:
            return f"{self.name}: {value:.2f}% - Moderate risk"
        elif value < 50:
            return f"{self.name}: {value:.2f}% - High risk"
        else:
            return f"{self.name}: {value:.2f}% - Very high risk"


class UtilityMetric(BaseMetric):
    """
    Base class for utility-related metrics.

    Utility metrics quantify how useful or valuable the anonymized
    or synthesized data remains for analytical purposes after
    privacy transformations.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a utility metric.

        Parameters:
        -----------
        name : str
            The name of the metric.
        description : str
            A brief description of what the metric measures.
        """
        super().__init__(name, description)

    def interpret(self, value: float) -> str:
        """
        Interpret a utility metric value.

        For utility metrics, higher values typically indicate better utility
        (less information loss).

        Parameters:
        -----------
        value : float
            The utility metric value (typically a percentage of preserved utility).

        Returns:
        --------
        str
            Human-readable interpretation of the utility.
        """
        if value < 50:
            return f"{self.name}: {value:.2f}% - Poor utility"
        elif value < 70:
            return f"{self.name}: {value:.2f}% - Fair utility"
        elif value < 85:
            return f"{self.name}: {value:.2f}% - Good utility"
        elif value < 95:
            return f"{self.name}: {value:.2f}% - Very good utility"
        else:
            return f"{self.name}: {value:.2f}% - Excellent utility"


class FidelityMetric(BaseMetric):
    """
    Base class for fidelity-related metrics.

    Fidelity metrics quantify how well statistical properties and
    relationships in the original data are preserved in the
    anonymized or synthesized data.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a fidelity metric.

        Parameters:
        -----------
        name : str
            The name of the metric.
        description : str
            A brief description of what the metric measures.
        """
        super().__init__(name, description)

    def interpret(self, value: float) -> str:
        """
        Interpret a fidelity metric value.

        For fidelity metrics, higher values typically indicate better
        preservation of statistical properties.

        Parameters:
        -----------
        value : float
            The fidelity metric value (typically a percentage of similarity).

        Returns:
        --------
        str
            Human-readable interpretation of the fidelity.
        """
        if value < 60:
            return f"{self.name}: {value:.2f}% - Poor fidelity"
        elif value < 80:
            return f"{self.name}: {value:.2f}% - Moderate fidelity"
        elif value < 90:
            return f"{self.name}: {value:.2f}% - Good fidelity"
        elif value < 97:
            return f"{self.name}: {value:.2f}% - Very good fidelity"
        else:
            return f"{self.name}: {value:.2f}% - Excellent fidelity"


class QualityMetric(BaseMetric):
    """
    Class for evaluating quality metrics.

    This class provides methods to calculate and interpret quality metrics
    for synthetic datasets compared to real datasets.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize an evaluate quality metric.

        Parameters:
        -----------
        name : str
            The name of the metric.
        description : str
            A brief description of what the metric measures.
        """
        super().__init__(name, description)

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate the quality metric value based on provided real and synthetic data.

        Parameters:
        -----------
        real_data : pd.DataFrame
            The original dataset.
        synthetic_data : pd.DataFrame
            The synthetic dataset.
        kwargs : dict
            Additional parameters for metric calculation.

        Returns:
        --------
        dict
            Dictionary containing metric values and any additional information.
        """
        # Placeholder for actual calculation logic
        return {"quality_metric": 0.0}

    def interpret(self, value: float) -> str:
        """
        Provide an interpretation of the quality metric value.

        Parameters:
        -----------
        value : float
            The quality metric value to interpret.

        Returns:
        --------
        str
            Human-readable interpretation of the quality metric value.
        """
        if value < 50:
            return f"{self.name}: {value:.2f}% - Poor quality"
        elif value < 70:
            return f"{self.name}: {value:.2f}% - Fair quality"
        elif value < 85:
            return f"{self.name}: {value:.2f}% - Good quality"
        elif value < 95:
            return f"{self.name}: {value:.2f}% - Very good quality"
        else:
            return f"{self.name}: {value:.2f}% - Excellent quality"


def round_metric_values(metrics: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
    """
    Round numeric values in a metrics dictionary for better readability.

    Parameters:
    -----------
    metrics : dict
        Dictionary containing metric values.
    decimals : int, optional
        Number of decimal places to round to (default: 2).

    Returns:
    --------
    dict
        Dictionary with rounded metric values.
    """
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = round(value, decimals)
        elif isinstance(value, dict):
            result[key] = round_metric_values(value, decimals)
        else:
            result[key] = value
    return result