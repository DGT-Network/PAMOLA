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

Module: Uniform Noise Addition Processor
-----------------------------------------
This module provides an implementation of Uniform noise addition
for anonymization-preserving data transformations. It extends the
BaseNoiseAdditionProcessor and allows controlled noise injection
to numerical attributes.

Uniform noise addition introduces random values drawn from a
uniform distribution within a specified range.

Common use cases include:
- Randomized perturbation while maintaining controlled boundaries.
- Masking values to prevent exact inference while retaining utility.
- Data anonymization for machine learning applications.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import numpy as np
from abc import ABC
from core.anonymization.noise_addition.base import BaseNoiseAdditionProcessor

class UniformNoiseAdditionProcessor(BaseNoiseAdditionProcessor, ABC):
    """
    Uniform Noise Addition Processor for anonymizing numerical attributes.
    This class extends BaseNoiseAdditionProcessor and applies uniform noise
    to numerical values while preserving statistical properties.

    Methods:
    --------
    - add_noise(): Applies Uniform noise to numerical columns.
    """

    def __init__(self, min_noise: float = -1.0, max_noise: float = 1.0, min_val: float = None, max_val: float = None):
        """
        Initializes the Uniform noise addition processor.

        Parameters:
        -----------
        min_noise : float, optional
            The minimum possible value for the uniform noise distribution (default: -1.0).
        max_noise : float, optional
            The maximum possible value for the uniform noise distribution (default: 1.0).
        min_val : float, optional
            Minimum bound for the noise-added values (default: None, no limit).
        max_val : float, optional
            Maximum bound for the noise-added values (default: None, no limit).
        """
        super().__init__()  # Ensure base class is properly initialized
        if min_noise > max_noise:
            raise ValueError("min_noise must be <= max_noise.")
        
        self.min_noise = min_noise
        self.max_noise = max_noise
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

    def add_noise(self, data: pd.DataFrame, columns: list[str], noise_type: str = "uniform",
                min_noise: float = None, max_noise: float = None, min_val: float = None, 
                max_val: float = None, **kwargs) -> pd.DataFrame:
        """
        Apply Uniform noise to specified columns in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numerical data.
        columns : list[str]
            List of column names where noise should be injected.
        noise_type : str, optional
            The type of noise to apply (default: "uniform").
        min_noise : float, optional
            The minimum noise value to apply (default: instance-level min_noise).
        max_noise : float, optional
            The maximum noise value to apply (default: instance-level max_noise).
        min_val : float, optional
            Minimum bound for the noise-added values (default: instance-level min_val).
        max_val : float, optional
            Maximum bound for the noise-added values (default: instance-level max_val).

        Returns:
        --------
        pd.DataFrame
            The dataset with Uniform noise added to specified columns.
        """
        if noise_type != "uniform":
            raise ValueError(f"UniformNoiseAdditionProcessor only supports 'uniform' noise, got '{noise_type}'.")

        min_noise = min_noise if min_noise is not None else self.min_noise
        max_noise = max_noise if max_noise is not None else self.max_noise
        min_val = min_val if min_val is not None else self.min_val
        max_val = max_val if max_val is not None else self.max_val

        if min_noise > max_noise:
            raise ValueError("min_noise must be <= max_noise.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not all(col in data.columns for col in columns):
            missing_cols = [col for col in columns if col not in data.columns]
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        data = data.copy()
        for column in columns:
            data[column] = self._apply_noise(data[column], min_noise, max_noise, min_val, max_val)

        return data

    def _apply_noise(self, series: pd.Series, min_noise: float, max_noise: float,
                     min_val: float, max_val: float) -> pd.Series:
        """Applies uniform noise to a single column."""
        noise = np.random.uniform(low=min_noise, high=max_noise, size=len(series))

        series_noisy = series.fillna(0) + noise

        if min_val is not None:
            series_noisy = series_noisy.clip(lower=min_val)
        if max_val is not None:
            series_noisy = series_noisy.clip(upper=max_val)

        return series_noisy
