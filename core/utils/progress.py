"""
Privacy Utility - Progress Logger
---------------------------------
This module provides helper functions for **logging progress**
in anonymization-preserving transformations, including **k-Anonymity,
l-Diversity, and t-Closeness**.

It supports **real-time logging** and **progress tracking**
for large datasets using `tqdm`.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import logging
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def progress_logger(message: str, total: int):
    """
    Logs progress updates with tqdm.

    Parameters:
    -----------
    message : str
        Descriptive message to display in progress bar.
    total : int
        Total number of iterations (records, groups, etc.).

    Returns:
    --------
    tqdm
        Progress bar object for tracking.
    """
    logging.info(message)
    return tqdm(total=total, desc=message, unit="steps")
