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

Module: Laplace Noise Addition Processor
-----------------------------------------
This module provides an implementation of Laplace noise addition
for anonymization-preserving data transformations. It extends the
BaseNoiseAdditionProcessor and allows controlled noise injection
to numerical attributes.

Laplace noise addition is widely used in **differential anonymization** as it
ensures anonymization protection by drawing random values from a Laplace
distribution with a specified mean and scale.

Common use cases include:
- Differential anonymization mechanisms.
- Statistical perturbation while maintaining utility.
- Preventing exact inference of original values.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
import pandas as pd
import numpy as np
from abc import ABC
from core.anonymization.noise_addition.base import BaseNoiseAdditionProcessor

class LaplaceNoiseAdditionProcessor(BaseNoiseAdditionProcessor, ABC):
    """
    Laplace Noise Addition Processor for anonymizing numerical attributes.
    This class extends BaseNoiseAdditionProcessor and applies Laplace noise
    to numerical values while preserving statistical properties.

    Methods:
    --------
    - add_noise(): Applies Laplace noise to numerical columns.
    """

    def __init__(self, mean: float = 0.0, scale: float = 1.0, min_val: float = None, max_val: float = None):
        """
        Initializes the Laplace noise addition processor.

        Parameters:
        -----------
        mean : float, optional
            The mean of the Laplace distribution (default: 0.0).
        scale : float, optional
            The scale parameter (b) of the Laplace distribution,
            controlling the spread of the noise (default: 1.0).
        min_val : float, optional
            Minimum bound for the noise-added values (default: None, no limit).
        max_val : float, optional
            Maximum bound for the noise-added values (default: None, no limit).
        """
        self.mean = mean
        self.scale = scale
        self.min_val = min_val
        self.max_val = max_val

    def add_noise(self, data: pd.DataFrame, columns: list[str], noise_type: str = "laplace",
                  mean: float = None, scale: float = None, min_val: float = None, max_val: float = None, **kwargs) -> pd.DataFrame:
        """
        Apply Laplace noise to specified columns in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing numerical data.
        columns : list[str]
            List of column names where noise should be injected.
        noise_type : str, optional
            The type of noise to apply (default: "laplace").
        mean : float, optional
            The mean of the Laplace distribution (default: instance-level mean).
        scale : float, optional
            The scale parameter (b) for the Laplace distribution (default: instance-level scale).
        min_val : float, optional
            Minimum bound for the noise-added values (default: instance-level min_val).
        max_val : float, optional
            Maximum bound for the noise-added values (default: instance-level max_val).

        Returns:
        --------
        pd.DataFrame
            The dataset with Laplace noise added to specified columns.
        """
        if noise_type != "laplace":
            raise ValueError(f"LaplaceNoiseAdditionProcessor only supports 'laplace' noise, got '{noise_type}'.")

        mean = mean if mean is not None else self.mean
        scale = scale if scale is not None else self.scale
        min_val = min_val if min_val is not None else self.min_val
        max_val = max_val if max_val is not None else self.max_val

        for column in columns:
            noise = np.random.laplace(loc=mean, scale=scale, size=len(data))
            data[column] += noise

            if min_val is not None:
                data[column] = data[column].clip(lower=min_val)
            if max_val is not None:
                data[column] = data[column].clip(upper=max_val)

        return data
