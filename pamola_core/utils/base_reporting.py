"""
PAMOLA.CORE - Base Reporting Utilities for Privacy Models
---------------------------------------------------------
This module provides the foundation for generating reports about
anonymization transformations. It contains common infrastructure and
utilities that can be used across different anonymization models such as
k-anonymity, l-diversity, t-closeness, and others.

Key features:
- Common report formatting and structure
- Multiple output formats (JSON, HTML, plaintext)
- Report saving and loading utilities
- Extensible interfaces for model-specific reporting

To implement reports for specific anonymization models, extend this module's
base classes and interfaces in dedicated reporting modules (e.g.,
ka_reporting.py for k-anonymity).

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import json
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any

from pamola_core import config
from pamola.pamola_core.utils.file_io import write_json

# Configure logging
logger = logging.getLogger(__name__)


class PrivacyReport(ABC):
    """
    Abstract base class for anonymization model reports.

    This class defines the interface and common functionality
    for all anonymization model reports, enabling consistent report
    generation across different anonymization approaches.
    """

    def __init__(self, report_data: Dict[str, Any], report_type: str):
        """
        Initialize a anonymization report.

        Parameters:
        -----------
        report_data : dict
            Dictionary containing the report data.
        report_type : str
            Type of anonymization model (e.g., 'k-anonymity', 'l-diversity').
        """
        self.report_data = report_data
        self.report_type = report_type
        self.metadata = {
            "creation_time": datetime.now().isoformat(),
            "pamola_version": getattr(config, "PAMOLA_VERSION", "unknown"),
            "report_type": report_type
        }

    @abstractmethod
    def generate(self, include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate the report.

        Parameters:
        -----------
        include_visualizations : bool, optional
            Whether to include visualization paths in the report.

        Returns:
        --------
        dict
            The compiled report.
        """
        pass

    def save(self, output_path: str, format: str = "json") -> str:
        """
        Save the report to a file.

        Parameters:
        -----------
        output_path : str
            Path where to save the report.
        format : str, optional
            Report format: 'json', 'html', or 'text' (default: 'json').

        Returns:
        --------
        str
            Path to the saved report.
        """
        # Generate the report
        report = self.generate()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save in the specified format
        if format.lower() == "json":
            save_json_report(report, output_path)
        elif format.lower() == "html":
            save_html_report(report, output_path)
        elif format.lower() == "text":
            save_text_report(report, output_path)
        else:
            logger.warning(f"Unsupported report format: {format}. Defaulting to JSON.")
            save_json_report(report, output_path)

        logger.info(f"Privacy report saved to {output_path}")
        return output_path

    def get_summary(self) -> str:
        """
        Generate a concise summary of the report.

        Returns:
        --------
        str
            A summary of key information from the report.
        """
        # Generate the report
        report = self.generate()

        # Create a basic summary
        summary = [
            f"PAMOLA {self.report_type.title()} Summary",
            "=" * (len(self.report_type) + 18),
            "",
            f"Generated on: {self.metadata['creation_time']}"
        ]

        return "\n".join(summary)


def save_json_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Saves a report in JSON format using the centralized file IO utility.

    Parameters:
    -----------
    report : dict
        The report data.
    output_path : str
        Path to save the report.
    """
    write_json(report, output_path)



def save_html_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Saves a report in HTML format.

    Parameters:
    -----------
    report : dict
        The report data.
    output_path : str
        Path to save the report.
    """
    # Basic HTML template
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PAMOLA Privacy Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; }}
        .section {{ margin-top: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>PAMOLA Privacy Report</h1>
    <p class="timestamp">Generated on: {report.get("report_metadata", {}).get("creation_time", datetime.now().isoformat())}</p>
"""

    # Add basic sections
    for section_name, section_data in report.items():
        if section_name == "report_metadata":
            continue

        html_content += f"""
    <div class="section">
        <h2>{section_name.replace('_', ' ').title()}</h2>
"""

        # Handle dictionary data
        if isinstance(section_data, dict):
            html_content += """
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
            for key, value in section_data.items():
                # Skip large nested objects
                if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                    html_content += f"""
            <tr><td class="metric">{key.replace('_', ' ').title()}</td><td>{value}</td></tr>
"""
            html_content += """
        </table>
"""
        # Handle list data
        elif isinstance(section_data, list):
            html_content += """
        <ul>
"""
            for item in section_data:
                html_content += f"""
            <li>{item}</li>
"""
            html_content += """
        </ul>
"""
        # Handle plain text
        else:
            html_content += f"""
        <p>{section_data}</p>
"""

        html_content += """
    </div>
"""

    # Close HTML document
    html_content += """
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def save_text_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Saves a report in plain text format.

    Parameters:
    -----------
    report : dict
        The report data.
    output_path : str
        Path to save the report.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write report title and metadata
        report_type = report.get("report_metadata", {}).get("report_type", "Privacy")
        creation_time = report.get("report_metadata", {}).get("creation_time", datetime.now().isoformat())

        f.write(f"PAMOLA {report_type.title()} Report\n")
        f.write("=" * (len(report_type) + 18) + "\n\n")
        f.write(f"Generated on: {creation_time}\n\n")

        # Write report sections
        for section_name, section_data in report.items():
            if section_name == "report_metadata":
                continue

            f.write(f"{section_name.replace('_', ' ').title()}:\n")
            f.write("-" * (len(section_name) + 1) + "\n")

            # Handle dictionary data
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    # Skip large nested objects
                    if not isinstance(value, (dict, list)) or len(str(value)) < 100:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")

            # Handle list data
            elif isinstance(section_data, list):
                for item in section_data:
                    f.write(f"- {item}\n")

            # Handle plain text
            else:
                f.write(f"{section_data}\n")

            f.write("\n")


def load_report(input_path: str) -> Dict[str, Any]:
    """
    Load a previously saved report.

    Parameters:
    -----------
    input_path : str
        Path to the report file.

    Returns:
    --------
    dict
        The loaded report data.
    """
    try:
        if input_path.endswith('.json'):
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Unsupported report format for loading: {input_path}. Only JSON reports can be loaded.")
            return {}
    except Exception as e:
        logger.error(f"Error loading report: {e}")
        return {}


def merge_reports(reports: List[Dict[str, Any]], title: str = "Merged Privacy Report") -> Dict[str, Any]:
    """
    Merge multiple reports into a single comprehensive report.

    Parameters:
    -----------
    reports : list of dict
        List of reports to merge.
    title : str, optional
        Title for the merged report.

    Returns:
    --------
    dict
        The merged report.
    """
    if not reports:
        return {}

    merged = {
        "report_metadata": {
            "creation_time": datetime.now().isoformat(),
            "pamola_version": getattr(config, "PAMOLA_VERSION", "unknown"),
            "report_type": title,
            "merged_from": len(reports)
        }
    }

    # For each report, add its sections to the merged report
    for i, report in enumerate(reports):
        for section_name, section_data in report.items():
            if section_name == "report_metadata":
                continue

            # Add report index suffix to avoid key collisions
            merged_section_name = f"{section_name}_{i + 1}"
            merged[merged_section_name] = section_data

    return merged