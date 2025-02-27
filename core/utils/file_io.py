"""
PAMOLA.CORE - File Handling Utilities
--------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
privacy-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for secure, structured, and scalable file operations.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: File IO Utilities
--------------------------
This module provides standardized file handling functions for the PAMOLA ecosystem.
It centralizes operations for reading, writing, and saving files, ensuring:
- **Consistent JSON and CSV file handling**.
- **Efficient and modular file writing for reports, metrics, and results**.
- **Secure directory creation before file writes**.
- **Standardized visualization saving for Matplotlib figures**.
- **Logging and error handling to prevent file operation failures**.

Features:
- **write_json()** → Writes a dictionary to a JSON file.
- **write_csv()** → Writes a dictionary to a CSV file.
- **save_plot()** → Saves a Matplotlib figure in various formats (PNG, PDF, SVG).
- **Directory Management** → Ensures paths exist before writing files.
- **Error Handling** → Logs failures and prevents file write corruption.

NOTE: This module requires `json`, `csv`, `matplotlib`, and `os`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import os
import json
import csv
from typing import IO, Union, cast, Optional
from pathlib import Path
import matplotlib.pyplot as plt

def write_json(data: dict, path: Union[str, Path]) -> str:
    """
    Writes JSON data to a file.

    Parameters:
    -----------
    data : dict
        The dictionary to save in JSON format.
    path : str or Path
        The path to save the file.

    Returns:
    --------
    str
        Path to the saved file.
    """
    path = Path(path)  # Ensure path is a Path object

    with path.open(mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)  # type: ignore

    return str(path)


def write_csv(data: dict, path: Union[str, Path]) -> str:
    """
    Writes dictionary data to a CSV file.

    Parameters:
    -----------
    data : dict
        The dictionary to save in CSV format.
    path : str or Path
        The path to save the file.

    Returns:
    --------
    str
        Path to the saved file.
    """
    path = Path(path)  # Ensure path is a Path object

    with open(path, "w", newline="", encoding="utf-8") as f:  # Type hint as TextIO
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in data.items():
            writer.writerow([key, value])

    return str(path)


def save_plot(fig: plt.Figure, save_path: Union[str, Path], save_format: str = "png",
              dpi: int = 300, bbox_inches: str = "tight") -> Optional[str]:
    """
    Saves a Matplotlib figure to a file.

    Parameters:
    -----------
    fig : plt.Figure
        The Matplotlib figure to save.
    save_path : str or Path
        The path (without extension) where the plot should be saved.
    save_format : str, optional
        The format to save the plot in ('png', 'pdf', 'svg') (default: 'png').
    dpi : int, optional
        The resolution of the output image (default: 300).
    bbox_inches : str, optional
        Bounding box settings (default: 'tight').

    Returns:
    --------
    str or None
        The full path to the saved file, or None if save_path is not provided.
    """
    try:
        save_path = Path(save_path).with_suffix(f".{save_format}")  # Ensure correct file extension
        os.makedirs(save_path.parent, exist_ok=True)  # Ensure directory exists

        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format=save_format)
        plt.close(fig)  # Close the figure to free memory

        return str(save_path)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving plot: {e}", exc_info=True)
        return None
