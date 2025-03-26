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

Module: Attack Simulation
-----------------------
This module provides an abstract base class for attack simulation feature
in PAMOLA.CORE. It defines the general structure and required methods for
implementing specific attack simulation

NOTE: This is a template and will be updated as development progresses.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from abc import abstractmethod
from pamola.core.base_processor import BaseProcessor



class AttackInitialization(BaseProcessor):
    """
    Abstract base class for attack simulation in PAMOLA.CORE.
    This class extends BaseProcessor and declare methods used for attack simulation.
    """

    @abstractmethod
    def preprocess_data(self, data1, data2):
        """
        Data preprocessing: Use TF-ID to convert all string elements of data1 and data2 to numbers

        Parameters:
        -----------
        data1: First dataset
        data_test: Second dataset

        Returns:
        -----------
        data1_final: The dataset contains numeric values corresponding to data1
        data2_final: The dataset contains numeric values corresponding to data2
        """

        pass