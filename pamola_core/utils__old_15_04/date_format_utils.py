"""
PAMOLA.CORE - Date Format Utilities
---------------------------------------------------------
This module contains utility functions and constants for handling date
format detection and validation.

Key features:
- Detect the format of a given date string.

These utilities are used for analyzing and validating date fields in data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from typing import List, Optional
from datetime import datetime
from dateutil import parser

from pamola_core.common.constants import Constants

def detect_date_format(date_str: str, expected_formats: Optional[List[str]] = None) -> str:
    """
    Detect the format of a given date string.

    Parameters:
    -----------
    date_str : str
        The date string to analyze.
    expected_formats : Optional[List[str]]
        A list of expected date formats to validate against. If None, common formats will be used.

    Returns:
    --------
    str
        The detected date format if the date is valid and can be parsed successfully.
        Returns "Invalid Date" if the input is not a valid date.
        Returns "Unknown" if the date is valid but no expected format matches.
    """

    formats = expected_formats or Constants.COMMON_DATE_FORMATS

    if is_validate_date_time(date_str) is False:
        return "Invalid Date"
    
    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except ValueError:
            continue
    
    return "Unknown"

def is_validate_date_time(date_string: str) -> bool:
    """
    Validates if the given date string is a valid date.

    Parameters:
    date_string (str): The date string to be validated.

    Returns:
    bool: True if the date string is a valid date, False otherwise.
    """
    try:
        # Attempt to parse the string into a datetime object using dateutil.parser
        parser.parse(date_string)
        return True
    except (ValueError, OverflowError):
        return False