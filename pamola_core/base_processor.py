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

Module: Base Processor
-----------------------
This module provides an abstract base class for all data processing modules
in PAMOLA.CORE. It defines the general structure and required methods for
implementing specific processors, such as anonymization, attack simulation,
and data synthesis.

NOTE: This is a template and will be updated as development progresses.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """
    Abstract base class for all data processors in PAMOLA.CORE.
    This class enforces a common interface for all processing modules.
    """

    @abstractmethod
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
