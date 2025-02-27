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

Module: Generalization Base Processor
--------------------------------------
This module defines an abstract base class for generalization techniques
in PAMOLA.CORE. It extends the generic BaseProcessor and provides
a structured interface for all generalization methods.

NOTE: This is a template and will be updated as development progresses.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import abstractmethod
import pandas as pd
from core.base_processor import BaseProcessor

class BaseGeneralizationProcessor(BaseProcessor):
    """
    Abstract base class for generalization processors in PAMOLA.CORE.
    This class extends BaseProcessor and defines methods specific to
    generalization techniques.
    """

    @abstractmethod
    def generalize(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply generalization techniques to the given dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset to be generalized.
        kwargs : dict
            Additional parameters for generalization.

        Returns:
        --------
        pd.DataFrame
            The generalized dataset.
        """
        pass
