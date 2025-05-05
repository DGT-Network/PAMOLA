"""
PAMOLA.CORE - Logging Utility
---------------------------------------------------------
This module provides functionality to save logs.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Features:
 - Save logs to a file.
 - Support tracking changes during data anonymization.
 - Use JSON format for structured logging.

This utility is useful for privacy-preserving data processing and anonymization techniques.

NOTE: This module requires `json` as a dependency.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import json
import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from pamola_core.common.constants import Constants

def convert_to_json_serializable(obj):
    """Recursively convert non-serializable data types to JSON-compatible formats."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, pd.Int64Dtype.type)):
        return int(obj)  # Convert int64 to int
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return obj.isoformat() if not pd.isna(obj) else None  # Convert datetime to string, NaT to None
    elif pd.isna(obj):  # Handle NaT and NaN
        return None  
    return obj

def save_privacy_logging(
    change_log: Dict,
    log_str: str,
    track_changes: bool = True,
    log_dir: str = Constants.LOG_DIR,
    **kwargs
) -> None:
    """
    Save the log to a specified file.

    Parameters:
    -----------
    - change_log (Dict): Dictionary containing the changes to log.
    - log_str (str): Path to save the log file.
    - track_changes (bool): Whether to save the log based on tracking settings.
    - log_dir (str): Directory where logs should be stored. Defaults to "logs".
    - **kwargs: Additional parameters.

    Returns:
    --------
    None
    """
    if not track_changes or not change_log or not log_str:
        return

    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
    log_path = os.path.join(log_dir, log_str.strip().strip('"'))

    # Convert problematic data types
    clean_log = convert_to_json_serializable(change_log)

    try:
        with open(log_path, "w") as log_file:
            json.dump(clean_log, log_file, indent=4, ensure_ascii=False)
        print(f"Log saved successfully: {log_path}")
    except Exception as e:
        print(f"Error saving log: {e}")


def log_privacy_transformations(
    change_log: Dict,
    operation_name: str,
    func_name: str,
    mask: pd.Series,
    original_values: Union[pd.Series, pd.DataFrame],
    modified_values: Union[pd.Series, pd.DataFrame],
    fields: List
):
    """
    Logs changes in a structured format, preserving individual change records.

    Parameters:
    -----------
    - change_log (dict): Dictionary to store change logs.
    - operation_name (str): Name of the operation performed.
    - func_name (str): Function name where the operation is applied.
    - mask (pd.Series): Boolean mask indicating modified rows.
    - original_values (Union[pd.Series, pd.DataFrame]): Original values before modification.
    - modified_values (Union[pd.Series, pd.DataFrame]): Modified values after transformation.
    - fields (list): List of column names affected by the modification.

    Returns:
    --------
    - None: Updates `change_log` in place.
    """
    if change_log is None or not isinstance(change_log, dict):
        return

    # Ensure original_values and modified_values are DataFrames
    if isinstance(original_values, pd.Series):
        original_values = original_values.to_frame()

    if isinstance(modified_values, pd.Series):
        modified_values = modified_values.to_frame()

    # Get changed indices efficiently
    changed_indices = mask.index[mask].tolist()
    changed_rows = len(changed_indices)

    if changed_rows > 0:
        key = f"{operation_name}-{func_name}-{str(fields)}"

        # Store logs with index, original value, and transformed value
        change_log[key] = {
            "changed_rows": changed_rows,
            "changes": [
                {
                    "index": idx,
                    "original": {field: original_values.at[idx, field] for field in fields if idx in original_values.index},
                    "generalized": {field: modified_values.at[idx, field] for field in fields if idx in modified_values.index},
                }
                for idx in changed_indices
            ],
        }
