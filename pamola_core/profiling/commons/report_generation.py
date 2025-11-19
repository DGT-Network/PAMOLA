"""
Report generation utilities for the anonymization project.

This module provides functions for generating and formatting analysis reports
and artifacts, including JSON reports, CSV dictionaries, and metadata formatting.
"""

import json
import logging
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from pamola_core.utils.io import ensure_directory
from pamola_core.utils.logging import configure_logging

# Configure logger using the custom logging utility
logger = configure_logging(level=logging.INFO)


def save_profiling_results(
        result: Dict[str, Any],
        profile_type: str,
        output_name: str,
        format: str = "json",
        output_dir: Optional[str] = None,
        include_timestamp: bool = True
) -> str:
    """
    Save profiling results to a file.

    Parameters:
    -----------
    result : Dict[str, Any]
        Result dictionary to save
    profile_type : str
        Type of profiling being performed
    output_name : str
        Name for the output file (without extension)
    format : str
        Format to save as ('json' or 'csv')
    output_dir : str, optional
        Directory to save results in (default: uses project default)
    include_timestamp : bool
        Whether to include a timestamp in the filename

    Returns:
    --------
    str
        Path to the saved file
    """
    from pamola_core.utils.io import write_json, write_csv

    try:
        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_name}_{timestamp}.{format}"
        else:
            filename = f"{output_name}.{format}"

        # Ensure directory structure exists
        if output_dir:
            base_dir = Path(output_dir)
        else:
            from pamola_core.utils.io import get_output_dir
            base_dir = get_output_dir() / profile_type

        ensure_directory(base_dir)

        # Save in appropriate format
        output_path = base_dir / filename

        if format.lower() == 'json':
            write_json(result, output_path)
        elif format.lower() == 'csv':
            if isinstance(result, pd.DataFrame):
                write_csv(result, output_path)
            else:
                # Try to convert dictionary to DataFrame
                try:
                    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
                    df.columns = ['key', 'value']
                    write_csv(df, output_path)
                except Exception as e:
                    logger.error(f"Failed to convert result to DataFrame for CSV saving: {e}")
                    # Fall back to JSON
                    output_path = base_dir / f"{output_name}.json"
                    write_json(result, output_path)
        else:
            logger.warning(f"Unsupported format {format}, saving as JSON")
            output_path = base_dir / f"{output_name}.json"
            write_json(result, output_path)

        return str(output_path)

    except Exception as e:
        logger.error(f"Error saving profiling results: {e}")
        return ""


def save_value_dictionary(
        value_counts: pd.Series,
        field_name: str,
        profile_type: str,
        min_frequency: int = 2,
        top_n: Optional[int] = None,
        output_dir: Optional[str] = None,
        include_timestamp: bool = True
) -> str:
    """
    Save a dictionary of field values with counts and percentages.

    Parameters:
    -----------
    value_counts : pd.Series
        Series of value counts (from value_counts())
    field_name : str
        Name of the field
    profile_type : str
        Type of profiling being performed
    min_frequency : int
        Minimum frequency to include in dictionary
    top_n : int, optional
        Limit to top N values by frequency
    output_dir : str, optional
        Directory to save dictionary in
    include_timestamp : bool
        Whether to include a timestamp in the filename

    Returns:
    --------
    str
        Path to the saved dictionary file
    """
    try:
        # Create dictionary directory
        if output_dir:
            dict_dir = Path(output_dir) / "dictionaries"
        else:
            from pamola_core.utils.io import get_output_dir
            dict_dir = get_output_dir() / profile_type / "dictionaries"

        ensure_directory(dict_dir)

        # Filter by minimum frequency
        filtered_counts = value_counts[value_counts >= min_frequency]

        # Limit to top_n if specified
        if top_n and len(filtered_counts) > top_n:
            filtered_counts = filtered_counts.nlargest(top_n)

        # Calculate percentages
        total = value_counts.sum()
        percentages = (filtered_counts / total * 100).round(2)

        # Create DataFrame
        df = pd.DataFrame({
            'value': filtered_counts.index,
            'count': filtered_counts.values,
            'percentage': percentages.values
        })

        # Sort by count descending
        df = df.sort_values('count', ascending=False)

        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{field_name}_dictionary_{timestamp}.csv"
        else:
            filename = f"{field_name}_dictionary.csv"

        output_path = dict_dir / filename

        # Save dictionary
        df.to_csv(output_path, index=False)

        return str(output_path)

    except Exception as e:
        logger.error(f"Error saving value dictionary: {e}")
        return ""


def format_operation_metadata(
        field_name: str,
        operation_type: str,
        parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format metadata for an operation to include in reports.

    Parameters:
    -----------
    field_name : str
        Name of the field being analyzed
    operation_type : str
        Type of operation being performed
    parameters : Dict[str, Any]
        Parameters used for the operation

    Returns:
    --------
    Dict[str, Any]
        Formatted metadata dictionary
    """
    return {
        'field_name': field_name,
        'operation_type': operation_type,
        'parameters': parameters,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'  # Version of the metadata format
    }