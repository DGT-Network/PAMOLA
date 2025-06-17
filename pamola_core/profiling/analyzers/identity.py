"""
Identity analysis module for the HHR project.

This module provides analyzers and operations for identity fields, following the
operation architecture. It includes analysis of identifier consistency, distribution
of records per identifier, and cross-matching of identifiers.

It integrates with the utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from pamola_core.profiling.commons.identity_utils import (
    analyze_identifier_distribution,
    analyze_identifier_consistency,
    find_cross_matches,
    compute_identifier_stats
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    plot_value_distribution,
    create_bar_plot  # Using bar plot instead of pie chart
)

# Configure logger
logger = logging.getLogger(__name__)


class IdentityAnalyzer:
    """
    Analyzer for identity fields.

    This analyzer provides methods for analyzing identity fields, including
    identifier consistency, distribution of records per identifier, and
    cross-matching of identifiers.
    """

    @staticmethod
    def analyze_identifier_distribution(df: pd.DataFrame,
                                        id_field: str,
                                        entity_field: Optional[str] = None,
                                        top_n: int = 15) -> Dict[str, Any]:
        """
        Analyze the distribution of entities per identifier.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        id_field : str
            Identifier field to analyze (e.g., 'UID')
        entity_field : str, optional
            Entity identifier field (e.g., 'resume_id')
        top_n : int
            Number of top examples to include

        Returns:
        --------
        Dict[str, Any]
            Analysis results including distribution statistics
        """
        return analyze_identifier_distribution(df, id_field, entity_field, top_n)

    @staticmethod
    def analyze_identifier_consistency(df: pd.DataFrame,
                                       id_field: str,
                                       reference_fields: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency between an identifier and reference fields.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        id_field : str
            Identifier field to analyze
        reference_fields : List[str]
            Fields that define an entity's identity

        Returns:
        --------
        Dict[str, Any]
            Analysis results including consistency statistics
        """
        return analyze_identifier_consistency(df, id_field, reference_fields)

    @staticmethod
    def find_cross_matches(df: pd.DataFrame,
                           id_field: str,
                           reference_fields: List[str],
                           min_similarity: float = 0.8,
                           fuzzy_matching: bool = False) -> Dict[str, Any]:
        """
        Find cases where reference fields match but identifiers differ.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        id_field : str
            Identifier field to analyze
        reference_fields : List[str]
            Fields that define an entity's identity
        min_similarity : float
            Minimum similarity for fuzzy matching
        fuzzy_matching : bool
            Whether to use fuzzy matching

        Returns:
        --------
        Dict[str, Any]
            Cross-matching analysis results
        """
        return find_cross_matches(df, id_field, reference_fields, min_similarity, fuzzy_matching)

    @staticmethod
    def compute_identifier_stats(df: pd.DataFrame,
                                 id_field: str,
                                 entity_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute basic statistics about an identifier field.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        id_field : str
            Identifier field to analyze
        entity_field : str, optional
            Entity identifier field for relation analysis

        Returns:
        --------
        Dict[str, Any]
            Basic statistics about the identifier
        """
        return compute_identifier_stats(df, id_field, entity_field)


@register(version="1.0.0")
class IdentityAnalysisOperation(FieldOperation):
    """
    Operation for analyzing identity fields and their consistency.

    This operation analyzes:
    1. Distribution of records per identifier
    2. Consistency between identifiers and reference fields
    3. Cross-matching of identifiers
    """

    def __init__(self,
                 uid_field: str,
                 reference_fields: List[str],
                 id_field: Optional[str] = None,
                 top_n: int = 15,
                 check_cross_matches: bool = None,
                 include_timestamps: bool = None,
                 min_similarity: float = 0.8,
                 fuzzy_matching: bool = None,
                 description: str = ""):
        """
        Initialize the identity analysis operation.

        Parameters:
        -----------
        uid_field : str
            Primary identifier field to analyze (e.g., 'UID')
        reference_fields : List[str]
            Fields that can be used to identify an entity (e.g., ['first_name', 'last_name', 'birth_day'])
        id_field : Optional[str]
            Entity identifier field (e.g., 'resume_id'), used to analyze entities per identifier
        description : str
            Description of the operation (optional)
        """
        super().__init__(
            field_name=uid_field,
            description=description or f"Analysis of identity field '{uid_field}'"
        )
        self.reference_fields = reference_fields
        self.id_field = id_field
        self.top_n = top_n
        self.check_cross_matches = check_cross_matches
        self.include_timestamps = include_timestamps
        self.min_similarity = min_similarity
        self.fuzzy_matching = fuzzy_matching

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the identity analysis operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : ProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation:
            - top_n : int, number of top entries to include (default: 15)
            - check_cross_matches : bool, whether to analyze cross matches (default: True)
            - include_timestamps : bool, whether to include timestamps (default: True)
            - min_similarity : float, similarity threshold for fuzzy matching (default: 0.8)
            - fuzzy_matching : bool, whether to use fuzzy matching (default: False)

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs
        global distribution_analysis, cross_match_analysis
        top_n = kwargs.get('top_n', self.top_n)
        check_cross_matches = kwargs.get('check_cross_matches', self.check_cross_matches)
        include_timestamps = kwargs.get('include_timestamps', self.include_timestamps)
        min_similarity = kwargs.get('min_similarity', self.min_similarity)
        fuzzy_matching = kwargs.get('fuzzy_matching', self.fuzzy_matching)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        visualizations_dir = dirs['visualizations']
        output_dir = dirs['output']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "field": self.field_name})

        try:
            # Get DataFrame from data source
            df = load_data_operation(data_source)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Check if required fields exist
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame"
                )

            # Validate reference fields
            valid_reference_fields = [field for field in self.reference_fields if field in df.columns]
            if not valid_reference_fields:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"None of the reference fields {self.reference_fields} found in DataFrame"
                )

            # Log missing reference fields
            if len(valid_reference_fields) < len(self.reference_fields):
                missing_fields = set(self.reference_fields) - set(valid_reference_fields)
                logger.warning(f"Some reference fields are missing: {missing_fields}")
                reporter.add_operation(
                    f"Missing reference fields for {self.field_name}",
                    status="warning",
                    details={"missing_fields": list(missing_fields)}
                )

            # Check id_field existence
            valid_id_field = self.id_field in df.columns if self.id_field else False
            if self.id_field and not valid_id_field:
                logger.warning(f"ID field {self.id_field} not found in DataFrame")
                reporter.add_operation(
                    f"Missing ID field {self.id_field}",
                    status="warning",
                    details={"missing_field": self.id_field}
                )

            # Add operation to reporter
            reporter.add_operation(f"Analyzing identity field: {self.field_name}", details={
                "field_name": self.field_name,
                "reference_fields": valid_reference_fields,
                "id_field": self.id_field if valid_id_field else None,
                "operation_type": "identity_analysis"
            })

            # Adjust progress tracker total steps if provided
            total_steps = 4  # Preparation, identifier stats, consistency, distribution
            if check_cross_matches:
                total_steps += 1  # Add step for cross-matching

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            # Step 1: Basic identifier statistics
            identifier_stats = IdentityAnalyzer.compute_identifier_stats(
                df,
                self.field_name,
                self.id_field if valid_id_field else None
            )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Basic identifier statistics complete"})

            # Step 2: Analyze identifier consistency
            consistency_analysis = IdentityAnalyzer.analyze_identifier_consistency(
                df,
                self.field_name,
                valid_reference_fields
            )

            # Save consistency analysis results
            consistency_filename = get_timestamped_filename(f"{self.field_name}_consistency", "json",
                                                            include_timestamps)
            consistency_path = output_dir / consistency_filename

            write_json(consistency_analysis, consistency_path)
            result.add_artifact("json", consistency_path, f"{self.field_name} consistency analysis")
            reporter.add_artifact("json", str(consistency_path), f"{self.field_name} consistency analysis")

            # Create visualization for consistency analysis
            if 'match_percentage' in consistency_analysis:
                # Create data for visualization - consistency percentage
                consistency_data = {
                    'Consistent': consistency_analysis['match_percentage'],
                    'Inconsistent': 100 - consistency_analysis['match_percentage']
                }

                viz_filename = get_timestamped_filename(f"{self.field_name}_consistency", "png", include_timestamps)
                viz_path = visualizations_dir / viz_filename

                # Use create_bar_plot instead of create_pie_chart
                viz_result = create_bar_plot(
                    data=consistency_data,
                    output_path=str(viz_path),
                    title=f"{self.field_name} Consistency Analysis",
                    orientation="h"  # horizontal bars to better show the proportions
                )

                if not viz_result.startswith("Error"):
                    result.add_artifact("png", viz_path, f"{self.field_name} consistency visualization")
                    reporter.add_artifact("png", str(viz_path), f"{self.field_name} consistency visualization")
                else:
                    logger.warning(f"Error creating consistency visualization: {viz_result}")

            # Save examples of mismatches
            if 'mismatch_examples' in consistency_analysis and consistency_analysis['mismatch_examples']:
                mismatch_filename = get_timestamped_filename(f"{self.field_name}_mismatch_examples", "json",
                                                             include_timestamps)
                mismatch_path = output_dir / mismatch_filename

                mismatch_examples = {
                    'mismatch_examples': consistency_analysis['mismatch_examples'],
                    'mismatch_count': consistency_analysis.get('mismatch_count', 0),
                    'total_records': consistency_analysis.get('total_records', 0)
                }

                write_json(mismatch_examples, mismatch_path)
                result.add_artifact("json", mismatch_path, f"{self.field_name} mismatch examples")
                reporter.add_artifact("json", str(mismatch_path), f"{self.field_name} mismatch examples")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Identifier consistency analysis complete"})

            # Step 3: Analyze distribution of records per identifier
            if valid_id_field:
                distribution_analysis = IdentityAnalyzer.analyze_identifier_distribution(
                    df,
                    self.field_name,
                    self.id_field,
                    top_n
                )

                # Save distribution analysis results
                distribution_filename = get_timestamped_filename(f"{self.field_name}_distribution", "json",
                                                                 include_timestamps)
                distribution_path = output_dir / distribution_filename

                write_json(distribution_analysis, distribution_path)
                result.add_artifact("json", distribution_path, f"Records per {self.field_name} distribution analysis")
                reporter.add_artifact("json", str(distribution_path),
                                      f"Records per {self.field_name} distribution analysis")

                # Create visualization for distribution analysis
                if 'distribution' in distribution_analysis:
                    viz_filename = get_timestamped_filename(f"{self.field_name}_count_distribution", "png",
                                                            include_timestamps)
                    viz_path = visualizations_dir / viz_filename

                    # Create visualization using the visualization module
                    viz_result = plot_value_distribution(
                        data=distribution_analysis['distribution'],
                        output_path=str(viz_path),
                        title=f"Records per {self.field_name} Distribution",
                        max_items=top_n
                    )

                    if not viz_result.startswith("Error"):
                        result.add_artifact("png", viz_path,
                                            f"Records per {self.field_name} distribution visualization")
                        reporter.add_artifact("png", str(viz_path),
                                              f"Records per {self.field_name} distribution visualization")
                    else:
                        logger.warning(f"Error creating distribution visualization: {viz_result}")
            else:
                logger.warning(f"Skipping distribution analysis. ID field not found: {self.id_field}")
                reporter.add_operation(
                    f"Skipping distribution analysis for {self.field_name}",
                    status="warning",
                    details={"reason": f"ID field {self.id_field} not found"}
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Record distribution analysis complete"})

            # Step 4: Cross-matching analysis
            if check_cross_matches and valid_reference_fields:
                cross_match_analysis = IdentityAnalyzer.find_cross_matches(
                    df,
                    self.field_name,
                    valid_reference_fields,
                    min_similarity,
                    fuzzy_matching
                )

                # Save cross-match analysis results
                cross_match_filename = get_timestamped_filename(f"{self.field_name}_cross_match", "json",
                                                                include_timestamps)
                cross_match_path = output_dir / cross_match_filename

                write_json(cross_match_analysis, cross_match_path)
                result.add_artifact("json", cross_match_path, f"{self.field_name} cross-match analysis")
                reporter.add_artifact("json", str(cross_match_path), f"{self.field_name} cross-match analysis")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Cross-match analysis complete"})
            else:
                if not check_cross_matches:
                    logger.info("Skipping cross-match analysis as per configuration")
                elif not valid_reference_fields:
                    logger.warning("Skipping cross-match analysis. No valid reference fields")

            # Add metrics to the result
            result.add_metric("total_records", identifier_stats.get('total_records', 0))
            result.add_metric("unique_identifiers", identifier_stats.get('unique_identifiers', 0))
            result.add_metric("identifier_coverage", identifier_stats.get('coverage_percentage', 0))

            if consistency_analysis:
                result.add_metric("consistent_percentage", consistency_analysis.get('match_percentage', 0))
                result.add_metric("inconsistent_count", consistency_analysis.get('mismatch_count', 0))

            if valid_id_field and 'distribution_analysis' in locals():
                result.add_metric("max_records_per_identifier", distribution_analysis.get('max_count', 0))
                result.add_metric("avg_records_per_identifier", distribution_analysis.get('avg_count', 0))

            if check_cross_matches and 'cross_match_analysis' in locals():
                result.add_metric("cross_match_count", cross_match_analysis.get('total_cross_matches', 0))

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "unique_identifiers": identifier_stats.get('unique_identifiers', 0),
                "consistency_percentage": consistency_analysis.get('match_percentage', 0),
                "reference_fields_used": valid_reference_fields,
                "cross_matches_found": cross_match_analysis.get('total_cross_matches',
                                                                0) if 'cross_match_analysis' in locals() else None
            })

            return result

        except Exception as e:
            logger.exception(f"Error in identity analysis operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing identity field {self.field_name}: {str(e)}"
            )

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare required directories for artifacts.

        Parameters:
        -----------
        task_dir : Path
            Base directory for the task

        Returns:
        --------
        Dict[str, Path]
            Dictionary of directory paths
        """
        # Create required directories
        output_dir = task_dir / 'output'
        visualizations_dir = task_dir / 'visualizations'
        dictionaries_dir = task_dir / 'dictionaries'

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)
        ensure_directory(dictionaries_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir,
            'dictionaries': dictionaries_dir
        }


def analyze_identities(
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        identity_fields: Dict[str, Dict[str, Any]] = None,
        **kwargs) -> Dict[str, OperationResult]:
    """
    Analyze multiple identity fields in a dataset.

    Parameters:
    -----------
    data_source : DataSource
        Source of data for the operations
    task_dir : Path
        Directory where task artifacts should be saved
    reporter : Any
        Reporter object for tracking progress and artifacts
    identity_fields : Dict[str, Dict[str, Any]], optional
        Dictionary mapping field names to their configuration. Each configuration
        should include 'reference_fields' (list) and optionally 'id_field' (str).
    **kwargs : dict
        Additional parameters for the operations:
        - top_n: int, number of top entries to include (default: 15)
        - check_cross_matches: bool, whether to analyze cross matches (default: True)
        - include_timestamps: bool, whether to include timestamps (default: True)

    Returns:
    --------
    Dict[str, OperationResult]
        Dictionary mapping field names to their operation results
    """
    # Get DataFrame from data source
    df = load_data_operation(data_source)
    # Use get_dataframe safely
    if df is None:
        reporter.add_operation("Identity fields analysis", status="error",
                               details={"error": "No valid DataFrame found in data source"})
        return {}

    # If no identity fields specified, try to detect them (this is a simplified approach)
    if identity_fields is None:
        identity_fields = {}

        # Look for potential ID fields
        potential_id_fields = [col for col in df.columns if
                               'id' in col.lower() or 'uuid' in col.lower() or 'uid' in col.lower()]

        # Look for potential reference fields (name fields, dates, etc.)
        potential_reference_fields = [col for col in df.columns if
                                      'name' in col.lower() or
                                      'date' in col.lower() or
                                      'birth' in col.lower() or
                                      'gender' in col.lower()]

        # Create a simple configuration for detected ID fields
        for id_field in potential_id_fields:
            entity_field = None
            # Try to find a related entity field for this ID field
            for other_id in potential_id_fields:
                if other_id != id_field:
                    entity_field = other_id
                    break

            identity_fields[id_field] = {
                'reference_fields': potential_reference_fields,
                'id_field': entity_field
            }

    # Report on fields to be analyzed
    reporter.add_operation("Identity fields analysis", details={
        "fields_count": len(identity_fields),
        "fields": list(identity_fields.keys()),
        "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
    })

    # Track progress if enabled
    track_progress = kwargs.get('track_progress', True)
    overall_tracker = None

    if track_progress and identity_fields:
        overall_tracker = ProgressTracker(
            total=len(identity_fields),
            description=f"Analyzing {len(identity_fields)} identity fields",
            unit="fields",
            track_memory=True
        )

    # Initialize results dictionary
    results = {}

    # Process each field
    for i, (field, config) in enumerate(identity_fields.items()):
        if field in df.columns:
            try:
                # Update overall progress tracker
                if overall_tracker:
                    overall_tracker.update(0, {"field": field, "progress": f"{i + 1}/{len(identity_fields)}"})

                logger.info(f"Analyzing identity field: {field}")

                # Get configuration for this field
                reference_fields = config.get('reference_fields', [])
                id_field = config.get('id_field')

                # Create and execute operation
                operation = IdentityAnalysisOperation(
                    field,
                    reference_fields=reference_fields,
                    id_field=id_field
                )
                result = operation.execute(data_source, task_dir, reporter, **kwargs)

                # Store result
                results[field] = result

                # Update overall tracker after successful analysis
                if overall_tracker:
                    if result.status == OperationStatus.SUCCESS:
                        overall_tracker.update(1, {"field": field, "status": "completed"})
                    else:
                        overall_tracker.update(1, {"field": field, "status": "error",
                                                   "error": result.error_message})

            except Exception as e:
                logger.error(f"Error analyzing identity field {field}: {e}", exc_info=True)

                reporter.add_operation(f"Analyzing {field} field", status="error",
                                       details={"error": str(e)})

                # Update overall tracker in case of error
                if overall_tracker:
                    overall_tracker.update(1, {"field": field, "status": "error"})

    # Close overall progress tracker
    if overall_tracker:
        overall_tracker.close()

    # Report summary
    success_count = sum(1 for r in results.values() if r.status == OperationStatus.SUCCESS)
    error_count = sum(1 for r in results.values() if r.status == OperationStatus.ERROR)

    reporter.add_operation("Identity fields analysis completed", details={
        "fields_analyzed": len(results),
        "successful": success_count,
        "failed": error_count
    })

    return results