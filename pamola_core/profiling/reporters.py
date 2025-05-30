"""
Utilities for reporting profiling results.

This module provides classes and functions for organizing and formatting
profiling results for reporting and visualization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from pamola_core.profiling.commons.base import AnalysisResult
from pamola_core.utils.task_reporting import TaskReporter

# Configure logger
logger = logging.getLogger(__name__)


class ProfileReporter:
    """
    Class for organizing and reporting profiling results.

    This class provides methods for collecting, organizing, and reporting
    the results of profiling operations.
    """

    def __init__(self, task_id: str, reporter: TaskReporter):
        """
        Initialize a new profile reporter.

        Parameters:
        -----------
        task_id : str
            The ID of the profiling task
        reporter : TaskReporter
            The task reporter for tracking progress and artifacts
        """
        self.task_id = task_id
        self.reporter = reporter
        self.results = {}

    def report_field_analysis(self,
                              field_name: str,
                              result: AnalysisResult,
                              artifacts: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Report the results of field analysis.

        Parameters:
        -----------
        field_name : str
            The name of the analyzed field
        result : AnalysisResult
            The results of the analysis
        artifacts : List[Dict[str, Any]], optional
            Additional artifacts to include in the report
        """
        if field_name not in self.results:
            self.results[field_name] = {}

        # Add analysis result
        self.results[field_name]['analysis'] = result.to_dict()

        # Add artifacts
        if artifacts:
            if 'artifacts' not in self.results[field_name]:
                self.results[field_name]['artifacts'] = []
            self.results[field_name]['artifacts'].extend(artifacts)

        # Add artifacts from result
        if result.artifacts:
            if 'artifacts' not in self.results[field_name]:
                self.results[field_name]['artifacts'] = []
            self.results[field_name]['artifacts'].extend(result.artifacts)

    def collect_report(self,
                       include_fields: Optional[List[str]] = None,
                       exclude_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collect all results into a structured report.

        Parameters:
        -----------
        include_fields : List[str], optional
            List of fields to include in the report. If None, all fields are included.
        exclude_fields : List[str], optional
            List of fields to exclude from the report

        Returns:
        --------
        Dict[str, Any]
            The collected report
        """
        # Filter fields if needed
        fields = list(self.results.keys())
        if include_fields:
            fields = [f for f in fields if f in include_fields]
        if exclude_fields:
            fields = [f for f in fields if f not in exclude_fields]

        # Collect results for selected fields
        report = {
            'task_id': self.task_id,
            'field_count': len(fields),
            'fields': {field: self.results[field] for field in fields}
        }

        return report

    def save_report(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the report to a file.

        Parameters:
        -----------
        output_path : str or Path, optional
            Path to save the report. If None, a default path is used.

        Returns:
        --------
        Path
            The path to the saved report
        """
        from pamola_core.utils.io import save_profiling_results
        report = self.collect_report()

        if output_path:
            # Ensure parent directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save report
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return output_path
        else:
            # Use default path
            path = save_profiling_results(report, self.task_id, f'{self.task_id}_full_report')
            return Path(path)

    def generate_html_report(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Generate an HTML report from the collected results.

        Parameters:
        -----------
        output_path : str or Path, optional
            Path to save the HTML report. If None, a default path is used.

        Returns:
        --------
        Path
            The path to the saved HTML report
        """
        try:
            # Try to import the reporting module
            from pamola_core.utils.reporting import generate_report

            # Generate report
            if output_path:
                return generate_report(self.task_id, output_path=output_path)
            else:
                # Use default path
                from pamola_core.utils.io import get_profiling_directory
                html_dir = get_profiling_directory(self.task_id) / 'html'
                html_dir.mkdir(parents=True, exist_ok=True)

                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = html_dir / f'{self.task_id}_report_{timestamp}.html'

                return generate_report(self.taреаsk_id, output_path=output_path)
        except ImportError:
            logger.warning("Module pamola_core.utils.reporting not found. HTML report generation skipped.")
            return Path()
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return Path()