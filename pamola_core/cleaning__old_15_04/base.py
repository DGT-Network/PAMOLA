"""
    PAMOLA Core - Cleaning Processors Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all Cleaning Processors operations
    in the PAMOLA Core library. It defines the strategies for handling missing values.

    Key features:
    - Base Processor for Data Cleaning

    This module serves as the foundation for all Cleaning Processors operations in PAMOLA Core.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:
        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd

from abc import ABC, abstractmethod
from typing import List

class BaseCleaningProcessor(ABC):
    """
    Abstract base class for dataset cleaning processors.

    This class serves as a foundation for implementing cleaning mechanisms that helping to
    handling missing values.
    """

    def __init__(
            self,
            null_markers: List[str] | None = None,
            treat_empty_as_null: bool = True,
            track_process: bool = True,
            process_log: str = None,
    ):
        """
        Initializes the BaseCleaningProcessor with configurable options.

        Parameters:
        -----------
        null_markers : list, optional
            Additional values to treat as null.
        treat_empty_as_null : bool, default True
            Whether to treat empty strings as null.
        track_process : bool, default True
            Whether to track processed values.
        process_log : str, optional
            Path to save processing log.
        """
        if null_markers is None:
            null_markers = []

        self.null_markers = null_markers
        self.treat_empty_as_null = treat_empty_as_null
        self.track_process = track_process
        self.process_log = process_log

    @abstractmethod
    def execute(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Handling null, missing, or empty values.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data to be processed.
        kwargs : dict, optional
            Additional parameters for customization.

        Returns:
        --------
        DataFrame
        """
        pass
