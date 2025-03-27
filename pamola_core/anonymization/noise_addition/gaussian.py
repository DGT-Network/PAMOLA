"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Gaussian Noise Addition Processor
-----------------------------------------
This module provides an implementation of Gaussian noise addition
for anonymization-preserving data transformations. It extends the
BaseNoiseAdditionProcessor and allows controlled noise injection
to numerical attributes.

Gaussian noise addition introduces statistical perturbation to
data by drawing random values from a normal distribution with
a specified mean and standard deviation.

Common use cases include:
- Statistical anonymization while preserving general trends.
- Controlled noise for differential anonymization applications.
- Data obfuscation to prevent exact value inference.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import numpy as np
from abc import ABC
from pamola_core.anonymization.noise_addition.base import BaseNoiseAdditionProcessor


class GaussianNoiseAdditionProcessor(BaseNoiseAdditionProcessor, ABC):
    """
    Gaussian Noise Addition Processor for anonymizing numerical attributes.
    This class extends BaseNoiseAdditionProcessor and applies Gaussian noise
    to numerical values while preserving statistical properties.

    Methods:
    --------
    - add_noise(): Applies Gaussian noise to numerical columns.
    """

    def __init__(self, mean: float = 0.0, std_dev: float = 1.0, min_val: float = None, max_val: float = None):
        """
        Initializes the Gaussian noise addition processor.

        Parameters:
        -----------
        mean : float, optional
            The mean of the Gaussian distribution (default: 0.0).
        std_dev : float, optional
            The standard deviation of the Gaussian distribution (default: 1.0).
        min_val : float, optional
            Minimum bound for the noise-added values (default: None, no limit).
        max_val : float, optional
            Maximum bound for the noise-added values (default: None, no limit).
        """
        super().__init__()  # Ensure base class is properly initialized

        if std_dev <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        self.mean = mean
        self.std_dev = std_dev
        self.min_val = min_val
        self.max_val = max_val

    def process(self, data):
        """
        Process the input data.

        Parameters:
        -----------
        data : Any
            The input data to be processed.

        Returns:
        --------
        Processed data, transformed according to the specific processor logic.
        """
        pass

    def add_noise(self, data: pd.DataFrame, columns: list[str], noise_type: str = "gaussian",
                mean: float = None, std_dev: float = None, min_val: float = None,
                max_val: float = None, **kwargs) -> pd.DataFrame:
        """
        Apply Gaussian noise to specified columns in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numerical data.
        columns : list[str]
            List of column names where noise should be injected.
        noise_type : str, optional
            The type of noise to apply (default: "gaussian").
        mean : float, optional
            The mean of the Gaussian distribution (default: instance-level mean).
        std_dev : float, optional
            The standard deviation of the Gaussian distribution (default: instance-level std_dev).
        min_val : float, optional
            Minimum bound for the noise-added values (default: instance-level min_val).
        max_val : float, optional
            Maximum bound for the noise-added values (default: instance-level max_val).

        Returns:
        --------
        pd.DataFrame
            The dataset with Gaussian noise added to specified columns.
        """
        if noise_type != "gaussian":
            raise ValueError(f"GaussianNoiseAdditionProcessor only supports 'gaussian' noise, got '{noise_type}'.")

        mean = mean if mean is not None else self.mean
        std_dev = std_dev if std_dev is not None else self.std_dev
        min_val = min_val if min_val is not None else self.min_val
        max_val = max_val if max_val is not None else self.max_val

        if std_dev <= 0:
            raise ValueError("Standard deviation must be greater than zero.")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not all(col in data.columns for col in columns):
            missing_cols = [col for col in columns if col not in data.columns]
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Clone dataset to avoid modifying original
        data = data.copy()
        
        for column in columns:
            data[column] = self._apply_noise(data[column], mean, std_dev, min_val, max_val)
        
        return data
    
    def _apply_noise(self, series: pd.Series, mean: float, std_dev: float, min_val: float, max_val: float) -> pd.Series:
        """Apply Gaussian noise to a single Pandas Series."""
        noise = np.random.normal(loc=mean, scale=std_dev, size=len(series))
        
        series_noisy = series.fillna(0) + noise
        
        if min_val is not None:
            series_noisy = series_noisy.clip(lower=min_val)
        if max_val is not None:
            series_noisy = series_noisy.clip(upper=max_val)
        
        return series_noisy
