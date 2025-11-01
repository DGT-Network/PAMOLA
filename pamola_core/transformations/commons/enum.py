# =============================================================================
# PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
# -----------------------------------------------------------------------------
# Module:        Enum Definitions
# Package:       pamola_core.transformations.commons
# Author:        PAMOLA Core Team
# Created:       2025
# License:       BSD 3-Clause
#
# Description:
#   Common enum definitions for transformation modules.
# =============================================================================
from enum import Enum

class PartitionMethod(Enum):
    EQUAL_SIZE = "equal_size"
    RANDOM = "random"
    MODULO = "modulo"


class OutputFormat(Enum):
    CSV = "csv"
    JSON = "json"
