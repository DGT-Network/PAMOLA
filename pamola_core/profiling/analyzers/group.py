"""
Group analysis operations for the HHR project.

This module provides classes for analyzing groups of records with the same
identifier. It integrates with IO, reporting, visualization and progress
tracking systems to execute analysis tasks and generate artifacts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from pamola_core.profiling.commons.group_utils import (
    calculate_field_variation,
    calculate_weighted_variation,
    calculate_change_frequency,
    analyze_cross_groups,
    extract_group_metadata,
    analyze_collapsibility,
    identify_change_patterns,
    calculate_variation_distribution,
    analyze_group_in_chunks,
    estimate_resources
)
from pamola_core.utils.io import (
    write_json,
    get_timestamped_filename,
    save_dataframe
)
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    plot_value_distribution,
    create_heatmap,
    plot_group_variation_distribution
)

# Configure logger
logger = logging.getLogger(__name__)


class GroupAnalyzer:
    """
    Analyzer for groups of records with the same identifier.

    Improved implementation with focus on resume_id grouping and
    detection of variations in personal identification fields.
    """

    def analyze(self, df: pd.DataFrame, group_field: str,
                fields_weights: Dict[str, float], **kwargs) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Analyze variation within groups identified by group_field.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to analyze
        group_field : str
            Field to group by (e.g., 'resume_id')
        fields_weights : Dict[str, float]
            Dictionary of fields and their weights for variation calculation
        **kwargs : dict
            Additional parameters:
            - min_group_size: Minimum size of groups to analyze (default: 2)
            - handle_nulls: How to handle nulls ('as_value', 'exclude', default: 'as_value')
            - analyze_collapsibility: Whether to analyze group collapsibility (default: False)
            - collapsibility_threshold: Threshold for collapsibility analysis (default: 0.2)
            - metadata_fields: List of fields to extract as group metadata (default: [])
            - analyze_changes: Whether to analyze change patterns (default: False)
            - group_metadata: Additional metadata about the groups (default: {})
            - group_context: Context information about groups (default: '')
            - subtable_name: Name of the subtable being analyzed (default: '')
            - set_name: Name/number of the field set being analyzed (for multiple sets, default: '')
            - use_chunks: Whether to process in chunks for large datasets (default: True)
            - chunk_size: Size of chunks for processing (default: 50000)
            - progress_tracker: Optional progress tracker
            - timestamp_field: Field containing timestamps for temporal analysis (default: None)
        """
        # Extract parameters from kwargs
        min_group_size = kwargs.get('min_group_size', 2)
        handle_nulls = kwargs.get('handle_nulls', 'as_value')
        analyze_collapsibility_flag = kwargs.get('analyze_collapsibility', False)
        collapsibility_threshold = kwargs.get('collapsibility_threshold', 0.2)
        metadata_fields = kwargs.get('metadata_fields', [])
        analyze_changes_flag = kwargs.get('analyze_changes', False)
        group_metadata = kwargs.get('group_metadata', {})
        group_context = kwargs.get('group_context', '')
        subtable_name = kwargs.get('subtable_name', '')
        set_name = kwargs.get('set_name', '')
        use_chunks = kwargs.get('use_chunks', True)
        chunk_size = kwargs.get('chunk_size', 50000)
        progress_tracker = kwargs.get('progress_tracker')
        timestamp_field = kwargs.get('timestamp_field', None)

        # Check for required fields
        if group_field not in df.columns:
            logger.error(f"Group field {group_field} not found in DataFrame")
            return {
                'error': f"Group field {group_field} not found in DataFrame"
            }, pd.DataFrame()

        # Validate fields_weights
        valid_fields_weights = {}
        for field, weight in fields_weights.items():
            if field in df.columns:
                valid_fields_weights[field] = weight
            else:
                logger.warning(f"Field {field} not found in DataFrame, removing from weights")

        if not valid_fields_weights:
            logger.error("No valid fields found in weights dictionary")
            return {
                'error': "No valid fields found in weights dictionary"
            }, pd.DataFrame()

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Validating fields"})

        # Determine whether to use chunked processing
        is_large_df = use_chunks and len(df) > chunk_size

        # Process data
        if is_large_df:
            logger.info(f"Processing large dataset with {len(df)} rows in chunks")

            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(1, {"step": "Processing in chunks", "total_rows": len(df)})

            # Process in chunks
            variation_results, analysis_stats = analyze_group_in_chunks(
                df=df,
                group_field=group_field,
                fields_weights=valid_fields_weights,
                chunk_size=chunk_size,
                min_group_size=min_group_size,
                handle_nulls=handle_nulls
            )
        else:
            # Group data by the group field
            logger.info(f"Processing dataset with {len(df)} rows directly")
            groups = df.groupby(group_field)

            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(1, {"step": "Grouping data", "group_count": len(groups)})

            # Prepare results containers
            variation_results = []

            # Process each group
            processed_groups = 0
            total_groups = len(groups)

            for group_id, group_df in groups:
                # Skip groups that are too small
                if len(group_df) < min_group_size:
                    continue

                # Calculate variation
                variation = calculate_weighted_variation(
                    group_df,
                    valid_fields_weights,
                    handle_nulls=handle_nulls
                )

                # Calculate variations for individual fields
                field_variations = {
                    field: calculate_field_variation(group_df, field, handle_nulls)
                    for field in valid_fields_weights
                }

                # Calculate change frequency if requested
                change_frequency = None
                if analyze_changes_flag:
                    change_frequency = calculate_change_frequency(
                        group_df,
                        list(valid_fields_weights.keys()),
                        handle_nulls=handle_nulls
                    )

                # Analyze temporal patterns if timestamp field is provided
                temporal_patterns = None

                # Get timestamp field value from kwargs
                ts_field = kwargs.get('timestamp_field', None)

                # Explicitly avoid using 'timestamp_field' variable name to prevent any confusion
                if ts_field is not None:
                    # Additional validation to ensure ts_field is a valid column
                    if isinstance(ts_field, str) and ts_field in group_df.columns:
                        try:
                            # Sort by timestamp using a validated column name
                            sorted_df = group_df.sort_values(by=ts_field)

                            # Track changes in key fields over time
                            temporal_patterns = {}
                            for field in valid_fields_weights:
                                if field in sorted_df.columns:
                                    # Get sequence of values
                                    value_sequence = sorted_df[field].fillna('NULL').tolist()
                                    # Only store if there are actual changes
                                    if len(set(value_sequence)) > 1:
                                        temporal_patterns[field] = value_sequence
                        except Exception as e:
                            logger.warning(f"Error analyzing temporal patterns: {str(e)}")
                    else:
                        logger.warning(f"Timestamp field '{ts_field}' not found in DataFrame columns")

                # Extract metadata if fields specified
                group_metadata_values = None
                if metadata_fields:
                    group_metadata_values = extract_group_metadata(group_df, metadata_fields)

                # Prepare result record
                result_record = {
                    'group_id': group_id,
                    'size': len(group_df),
                    'variation': variation,
                    'field_variations': field_variations
                }

                if change_frequency:
                    result_record['change_frequency'] = change_frequency

                if temporal_patterns:
                    result_record['temporal_patterns'] = temporal_patterns

                if group_metadata_values:
                    result_record['metadata'] = group_metadata_values

                variation_results.append(result_record)

                # Update progress periodically
                processed_groups += 1
                if progress_tracker and processed_groups % 100 == 0:
                    progress_tracker.update(0, {
                        "step": "Processing groups",
                        "progress": f"{processed_groups}/{total_groups}"
                    })

            # Calculate overall statistics
            analysis_stats = {
                'group_field': group_field,
                'fields_analyzed': list(valid_fields_weights.keys()),
                'fields_weights': valid_fields_weights,
                'total_groups': len(groups),
                'analyzed_groups': len(variation_results),
                'min_group_size': min_group_size
            }

            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(1,
                                        {"step": "Completed group analysis", "groups_analyzed": len(variation_results)})

        # Calculate overall statistics for fields
        if variation_results:
            variations = [r['variation'] for r in variation_results]

            # Calculate variation distribution
            variation_distribution = calculate_variation_distribution(variations)
            analysis_stats['variation_distribution'] = variation_distribution

            if len(variations) > 0:
                analysis_stats['overall_stats'] = {
                    'min_variation': min(variations),
                    'max_variation': max(variations),
                    'mean_variation': sum(variations) / len(variations),
                    'median_variation': sorted(variations)[len(variations) // 2]
                }

            # Calculate collapsibility
            if analyze_collapsibility_flag:
                collapsibility_analysis = analyze_collapsibility(
                    variation_results,
                    threshold=collapsibility_threshold
                )
                analysis_stats['collapsibility_analysis'] = collapsibility_analysis

                # Update progress if tracker provided
                if progress_tracker:
                    progress_tracker.update(1, {
                        "step": "Collapsibility analysis",
                        "collapsible_groups": collapsibility_analysis.get('collapsible_groups_count', 0)
                    })

            # Identify change patterns
            if analyze_changes_flag:
                change_patterns = identify_change_patterns(variation_results)
                analysis_stats['change_patterns'] = change_patterns

                # Update progress if tracker provided
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Change pattern analysis"})

            # Add metadata if provided
            if group_metadata:
                analysis_stats['group_metadata'] = group_metadata

            if group_context:
                analysis_stats['group_context'] = group_context

            if subtable_name:
                analysis_stats['subtable_name'] = subtable_name

            # Set name for the field set if provided
            if set_name:
                analysis_stats['set_name'] = set_name

        # Convert variation_results to DataFrame for easier handling
        df_variation_results = pd.DataFrame(variation_results) if variation_results else pd.DataFrame()

        return analysis_stats, df_variation_results

    @staticmethod
    def analyze_cross_groups(df: pd.DataFrame, primary_group_field: str,
                             secondary_identifier_fields: List[str], **kwargs) -> Dict[str, Any]:
        """
        Analyze relationships between different group identifiers.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to analyze
        primary_group_field : str
            Primary field to group by (e.g., 'resume_id')
        secondary_identifier_fields : List[str]
            Fields that form secondary identifiers (e.g., ['first_name', 'last_name', 'birth_day'])
        **kwargs : dict
            Additional parameters:
            - min_group_size: Minimum size of groups to analyze (default: 2)
            - handle_nulls: How to handle nulls (default: 'exclude')
            - threshold: Minimum confidence threshold (default: 0.8)
            - progress_tracker: Optional progress tracker

        Returns:
        --------
        Dict[str, Any]
            Analysis results containing relationships between identifiers
        """
        min_group_size = kwargs.get('min_group_size', 2)
        handle_nulls = kwargs.get('handle_nulls', 'exclude')
        threshold = kwargs.get('threshold', 0.8)
        progress_tracker = kwargs.get('progress_tracker')

        if progress_tracker:
            progress_tracker.update(1, {"step": "Starting cross-group analysis"})

        # Perform cross-group analysis
        results = analyze_cross_groups(
            df=df,
            primary_group_field=primary_group_field,
            secondary_identifier_fields=secondary_identifier_fields,
            min_group_size=min_group_size,
            handle_nulls=handle_nulls,
            threshold=threshold
        )

        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Completed cross-group analysis",
                "cross_group_count": results.get('cross_group_count', 0)
            })

        return results

    @staticmethod
    def plot_group_variation_distribution(variations: List[float], title: str = "Group Variation Distribution",
                                          output_path: Optional[str] = None, color: str = 'blue') -> str:
        """
        Create a visualization of group variation distribution using visualization.py API.

        Parameters:
        -----------
        variations : List[float]
            List of variation values
        title : str
            Title for the plot
        output_path : str, optional
            Path to save the visualization
        color : str
            Color for the bars

        Returns:
        --------
        str
            Path to the saved visualization or error message
        """
        # Use calculate_variation_distribution from group_utils
        distribution_data = calculate_variation_distribution(variations)

        # Use the visualization module directly
        return plot_value_distribution(
            data=distribution_data,
            output_path=output_path,
            title=title
        )

    @staticmethod
    def plot_group_size_distribution(df: pd.DataFrame, group_field: str,
                                     title: str = "Group Size Distribution",
                                     output_path: Optional[str] = None, color: str = 'green') -> str:
        """
        Create a visualization of group size distribution using visualization.py API.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to analyze
        group_field : str
            Field to group by
        title : str
            Title for the plot
        output_path : str, optional
            Path to save the visualization
        color : str
            Color for the bars

        Returns:
        --------
        str
            Path to the saved visualization or error message
        """
        try:
            # Calculate group sizes
            group_sizes = df.groupby(group_field).size()

            # Create size categories
            categories = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101+"]
            counts = [0] * len(categories)

            # Aggregate group sizes
            for size in group_sizes:
                if size == 1:
                    counts[0] += 1
                elif size <= 5:
                    counts[1] += 1
                elif size <= 10:
                    counts[2] += 1
                elif size <= 20:
                    counts[3] += 1
                elif size <= 50:
                    counts[4] += 1
                elif size <= 100:
                    counts[5] += 1
                else:
                    counts[6] += 1

            # Create data in a dictionary format
            data = {categories[i]: counts[i] for i in range(len(categories))}

            # Use the visualization module directly
            return plot_value_distribution(
                data=data,
                output_path=output_path,
                title=title
            )
        except Exception as e:
            # In case of error, log details and return error message
            logger.error(f"Error creating size distribution plot: {str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    def plot_field_variation_heatmap(variation_results: List[Dict[str, Any]],
                                     title: str = "Field Variation Heatmap",
                                     max_groups: int = 20,
                                     output_path: Optional[str] = None) -> str:
        """
        Create a heatmap visualization of field variations across groups using the visualization.py API.

        Parameters:
        -----------
        variation_results : List[Dict[str, Any]]
            List of group variation results
        title : str
            Title for the plot
        max_groups : int
            Maximum number of groups to include
        output_path : str, optional
            Path to save the visualization

        Returns:
        --------
        str
            Path to the saved visualization or error message
        """
        try:
            if not variation_results:
                return "Error: No variation results to visualize"

            # Sort groups by overall variation
            sorted_results = sorted(variation_results, key=lambda x: x.get('variation', 0), reverse=True)

            # Limit to max_groups
            selected_results = sorted_results[:max_groups]

            # Extract field variations
            field_names = []

            # First, collect all unique field names
            for result in selected_results:
                field_variations = result.get('field_variations', {})
                for field in field_variations.keys():
                    if field not in field_names:
                        field_names.append(field)

            # Create a proper data structure for the heatmap
            # It should be a dictionary with nested dictionaries representing x-y coordinates
            heatmap_data = {}

            for result in selected_results:
                group_id = str(result.get('group_id', 'Unknown'))
                field_variations = result.get('field_variations', {})

                if group_id not in heatmap_data:
                    heatmap_data[group_id] = {}

                for field in field_names:
                    variation = field_variations.get(field)
                    if variation is not None:
                        heatmap_data[group_id][field] = float(variation)

            # Use visualization module to create heatmap
            return create_heatmap(
                data=heatmap_data,
                output_path=output_path,
                title=title,
                x_label="Fields",
                y_label="Group IDs",
                colorscale="Blues",
                reverse_scale=False
            )
        except Exception as e:
            # Log the error and return error message
            logger.error(f"Error creating field variation heatmap: {str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    def estimate_resources(df: pd.DataFrame, group_field: str, fields_weights: Dict[str, float]) -> Dict[
        str, Any]:
        """
        Estimate resources needed for group analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to analyze
        group_field : str
            Field to group by
        fields_weights : Dict[str, float]
            Dictionary of fields and their weights

        Returns:
        --------
        Dict[str, Any]
            Estimated resource requirements
        """
        return estimate_resources(df, group_field, fields_weights)


@register(override=True)
class GroupOperation(BaseOperation):
    """
    Operation for analyzing groups of records.

    This class provides methods to execute group analysis and generate
    artifacts within a profiling task.

    Optimized for analyzing resume_id groups and detecting variations in
    personal identification fields.
    """

    def __init__(self, group_field: str, fields_weights: Dict[str, float],
                 min_group_size: int = 2, set_name: str = "",
                 description: str = "", timestamp_field: Optional[str] = None,
                 analyze_personal_fields: bool = True,
                 analyze_changes: bool = True):
        """
        Initialize a group analysis operation.

        Parameters:
        -----------
        group_field : str
            Field to group by (e.g., 'resume_id')
        fields_weights : Dict[str, float]
            Dictionary of fields and their weights for variation calculation
        min_group_size : int
            Minimum size of groups to analyze
        set_name : str
            Optional name for the field set
        description : str
            Description of the operation
        timestamp_field : str, optional
            Field containing timestamps for temporal analysis
        analyze_personal_fields : bool
            Whether to specifically analyze personal identification fields
        analyze_changes : bool
            Whether to analyze changes within groups
        """
        desc = description or f"Analysis of group variations by {group_field}"
        super().__init__(f"Group analysis by {group_field}", desc)

        self.group_field = group_field
        self.fields_weights = fields_weights
        self.min_group_size = min_group_size
        self.set_name = set_name
        self.analyzer = GroupAnalyzer()
        self.timestamp_field = timestamp_field
        self.analyze_personal_fields = analyze_personal_fields
        self.analyze_changes = analyze_changes

    def _generate_visualizations(self,
                                 df: pd.DataFrame,
                                 analysis_results: Dict[str, Any],
                                 variation_df: pd.DataFrame,
                                 visualizations_dir: Path,
                                 include_timestamp: bool,
                                 title_prefix: str,
                                 result: OperationResult,
                                 reporter: Any,
                                 set_suffix: str = ""):
        """
        Generate visualizations for the group analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        variation_df : pd.DataFrame
            DataFrame with variation results
        visualizations_dir : Path
            Directory to save visualizations (typically not used, save to root dir)
        include_timestamp : bool
            Whether to include timestamps in filenames
        title_prefix : str
            Prefix for plot titles
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        set_suffix : str
            Suffix for filenames based on set name
        """
        # Generate variation distribution visualization
        if 'overall_stats' in analysis_results and variation_df is not None and not variation_df.empty:
            variations = variation_df['variation'].tolist() if 'variation' in variation_df.columns else []

            if variations:
                dist_filename = get_timestamped_filename(f"group_variation_distribution{set_suffix}", "png",
                                                         include_timestamp)
                dist_path = dist_filename  # Save to root directory

                title = f"{title_prefix} Variation Distribution"
                if self.set_name:
                    title += f" - Set {self.set_name}"

                # Use the visualization method from GroupAnalyzer
                dist_result = plot_group_variation_distribution(
                    results={"variation_distribution": analysis_results['variation_distribution']},
                    output_path=str(dist_path),
                    title=title
                )

                if not dist_result.startswith("Error"):
                    result.add_artifact("png", dist_path,
                                        f"Group variation distribution{' for set ' + self.set_name if self.set_name else ''}")
                    reporter.add_artifact("png", str(dist_path),
                                          f"Group variation distribution{' for set ' + self.set_name if self.set_name else ''}")

        # Generate group size distribution visualization
        size_filename = get_timestamped_filename(f"group_size_distribution{set_suffix}", "png", include_timestamp)
        size_path = size_filename  # Save to root directory

        title = f"{title_prefix} Size Distribution"
        if self.set_name:
            title += f" - Set {self.set_name}"

        size_result = self.analyzer.plot_group_size_distribution(
            df=df,
            group_field=self.group_field,
            title=title,
            output_path=str(size_path)
        )

        if not size_result.startswith("Error"):
            result.add_artifact("png", size_path,
                                f"Group size distribution{' for set ' + self.set_name if self.set_name else ''}")
            reporter.add_artifact("png", str(size_path),
                                  f"Group size distribution{' for set ' + self.set_name if self.set_name else ''}")

        # Generate field variation heatmap if there's enough data
        if variation_df is not None and not variation_df.empty and len(variation_df) >= 5:
            heatmap_filename = get_timestamped_filename(f"field_variation_heatmap{set_suffix}", "png",
                                                        include_timestamp)
            heatmap_path = heatmap_filename  # Save to root directory

            title = f"{title_prefix} Field Variation Heatmap"
            if self.set_name:
                title += f" - Set {self.set_name}"

            # Extract variation results for heatmap
            variation_results = []

            # Extract field_variations data from DataFrame
            for _, row in variation_df.iterrows():
                row_dict = {}
                for col in row.index:
                    row_dict[col] = row[col]

                # Check if field_variations exists directly
                if 'field_variations' in row_dict:
                    variation_results.append(row_dict)
                else:
                    # Need to reconstruct field_variations from columns
                    # Try to identify fields that store variations
                    field_vars = {}
                    for col in row.index:
                        if col.startswith('field_variation_') and col != 'field_variations':
                            field_name = col.replace('field_variation_', '')
                            field_vars[field_name] = row[col]

                    if field_vars:
                        variation_results.append({
                            'group_id': row['group_id'] if 'group_id' in row else 'Unknown',
                            'variation': row['variation'] if 'variation' in row else 0.0,
                            'field_variations': field_vars
                        })

            if variation_results:
                heatmap_result = self.analyzer.plot_field_variation_heatmap(
                    variation_results=variation_results,
                    title=title,
                    output_path=str(heatmap_path)
                )

                if not heatmap_result.startswith("Error"):
                    result.add_artifact("png", heatmap_path,
                                        f"Field variation heatmap{' for set ' + self.set_name if self.set_name else ''}")
                    reporter.add_artifact("png", str(heatmap_path),
                                          f"Field variation heatmap{' for set ' + self.set_name if self.set_name else ''}")

    def execute(self, data_source: DataSource, task_dir: Path, reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None, **kwargs) -> OperationResult:
        """
        Execute group analysis and generate artifacts.

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
            Additional parameters:
            - title_prefix: Prefix for plot titles (default: 'Group')
            - generate_plots: Whether to generate plots (default: True)
            - save_details: Whether to save detailed group information (default: True)
            - handle_nulls: How to handle nulls (default: 'as_value')
            - analyze_cross_groups: Whether to analyze cross-group relationships (default: False)
            - secondary_identifier_fields: Fields forming secondary identifiers for cross-group analysis
            - analyze_collapsibility: Whether to analyze group collapsibility (default: True)
            - collapsibility_threshold: Threshold for collapsibility analysis (default: 0.2)
            - analyze_changes: Whether to analyze change patterns (default: False)
            - metadata_fields: List of fields to extract as group metadata (default: [])
            - include_timestamp: Whether to include timestamp in filenames (default: True)

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        dictionaries_dir = dirs['dictionaries']

        # Extract parameters from kwargs
        title_prefix = kwargs.get('title_prefix', 'Group')
        generate_plots = kwargs.get('generate_plots', True)
        save_details = kwargs.get('save_details', True)
        handle_nulls = kwargs.get('handle_nulls', 'as_value')
        analyze_cross_groups_flag = kwargs.get('analyze_cross_groups', False)
        secondary_identifier_fields = kwargs.get('secondary_identifier_fields', [])
        analyze_collapsibility = kwargs.get('analyze_collapsibility', True)
        collapsibility_threshold = kwargs.get('collapsibility_threshold', 0.2)
        metadata_fields = kwargs.get('metadata_fields', [])
        include_timestamp = kwargs.get('include_timestamp', True)

        # Use instance variables if provided
        analyze_changes = kwargs.get('analyze_changes', self.analyze_changes)

        # Create the main result object
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.update(1, {"step": "Preparation", "group_field": self.group_field})

        try:
            # Get DataFrame from data source
            df = data_source.get_dataframe("main")
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Check if field exists
            if self.group_field not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Group field {self.group_field} not found in DataFrame"
                )

            # Add operation to reporter
            reporter.add_operation(f"Analyzing groups by {self.group_field}", details={
                "group_field": self.group_field,
                "fields_weights": self.fields_weights,
                "min_group_size": self.min_group_size,
                "set_name": self.set_name,
                "operation_type": "group_analysis"
            })

            # Enhance fields_weights with personal identification fields if requested
            if self.analyze_personal_fields:
                # Check for personal identification fields in the DataFrame
                id_fields = {}

                # Standard personal identification fields with preset weights
                id_field_weights = {
                    'first_name': 0.3,
                    'last_name': 0.4,
                    'middle_name': 0.1,
                    'birth_day': 0.2,
                    'gender': 0.1,
                    'file_as': 0.3
                }

                # Add any personal fields that exist in df but not in fields_weights
                for field, weight in id_field_weights.items():
                    if field in df.columns and field not in self.fields_weights:
                        id_fields[field] = weight

                # Log which personal fields were added
                if id_fields:
                    logger.info(f"Adding personal identification fields to analysis: {list(id_fields.keys())}")

                    # Create a copy to avoid modifying the original
                    enhanced_fields_weights = self.fields_weights.copy()
                    enhanced_fields_weights.update(id_fields)
                else:
                    enhanced_fields_weights = self.fields_weights
            else:
                enhanced_fields_weights = self.fields_weights

            # Execute the analyzer
            analysis_params = {
                'min_group_size': self.min_group_size,
                'handle_nulls': handle_nulls,
                'analyze_collapsibility': analyze_collapsibility,
                'collapsibility_threshold': collapsibility_threshold,
                'analyze_changes': analyze_changes,
                'metadata_fields': metadata_fields,
                'set_name': self.set_name,
                'progress_tracker': progress_tracker,
                'timestamp_field': self.timestamp_field
            }

            analysis_results, variation_df = self.analyzer.analyze(
                df=df,
                group_field=self.group_field,
                fields_weights=enhanced_fields_weights,
                **analysis_params
            )

            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analysis complete", "field": self.group_field})

            # Determine file suffix based on set_name
            set_suffix = f"_set{self.set_name}" if self.set_name else ""

            # Save analysis results to JSON
            stats_filename = get_timestamped_filename(f"group_variation{set_suffix}", "json", include_timestamp)
            stats_path = stats_filename  # Save to root directory

            write_json(analysis_results, stats_path)
            result.add_artifact("json", stats_path,
                                f"Group variation analysis{' for set ' + self.set_name if self.set_name else ''}")

            # Add to reporter
            reporter.add_artifact("json", str(stats_path),
                                  f"Group variation analysis{' for set ' + self.set_name if self.set_name else ''}")

            # Save detailed group information if requested
            if save_details and not variation_df.empty:
                details_filename = get_timestamped_filename(f"group_variation_details{set_suffix}", "csv",
                                                            include_timestamp)
                details_path = Path(dictionaries_dir) / details_filename

                save_dataframe(variation_df, details_path)
                result.add_artifact("csv", details_path,
                                    f"Detailed group variation information{' for set ' + self.set_name if self.set_name else ''}")
                reporter.add_artifact("csv", str(details_path),
                                      f"Detailed group variation information{' for set ' + self.set_name if self.set_name else ''}")

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualizations if requested
            if generate_plots:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualizations"})

                self._generate_visualizations(
                    df=df,
                    analysis_results=analysis_results,
                    variation_df=variation_df,
                    visualizations_dir=task_dir,  # Pass task_dir instead of visualizations_dir
                    include_timestamp=include_timestamp,
                    title_prefix=title_prefix,
                    result=result,
                    reporter=reporter,
                    set_suffix=set_suffix
                )

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Visualizations complete"})

            # Perform cross-group analysis if requested
            if analyze_cross_groups_flag and secondary_identifier_fields:
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Starting cross-group analysis"})

                cross_group_results = self.analyzer.analyze_cross_groups(
                    df=df,
                    primary_group_field=self.group_field,
                    secondary_identifier_fields=secondary_identifier_fields,
                    progress_tracker=progress_tracker
                )

                # Save cross-group results
                cross_group_filename = get_timestamped_filename(f"cross_group_identifiers{set_suffix}", "json",
                                                                include_timestamp)
                cross_group_path = cross_group_filename  # Save to root directory

                write_json(cross_group_results, cross_group_path)
                result.add_artifact("json", cross_group_path, "Cross-group identifier analysis")
                reporter.add_artifact("json", str(cross_group_path), "Cross-group identifier analysis")

                # Save detailed cross-group mapping if there are results
                if cross_group_results.get('cross_group_details'):
                    mapping_filename = get_timestamped_filename(f"cross_group_mapping{set_suffix}", "csv",
                                                                include_timestamp)
                    mapping_path = Path(dictionaries_dir) / mapping_filename

                    mapping_df = pd.DataFrame(cross_group_results['cross_group_details'])
                    save_dataframe(mapping_df, mapping_path)

                    result.add_artifact("csv", mapping_path, "Cross-group mapping details")
                    reporter.add_artifact("csv", str(mapping_path), "Cross-group mapping details")

                if progress_tracker:
                    progress_tracker.update(1, {"step": "Cross-group analysis complete"})

            # Add metrics to the result
            result.add_metric("total_groups", analysis_results.get('total_groups', 0))
            result.add_metric("analyzed_groups", analysis_results.get('analyzed_groups', 0))
            result.add_metric("mean_variation", analysis_results.get('overall_stats', {}).get('mean_variation', 0))

            if 'collapsibility_analysis' in analysis_results:
                collapsibility = analysis_results['collapsibility_analysis']
                result.add_metric("collapsible_groups_count", collapsibility.get('collapsible_groups_count', 0))
                result.add_metric("collapsible_groups_percentage",
                                  collapsibility.get('collapsible_groups_percentage', 0))

            # Update progress to completion
            if progress_tracker:
                progress_tracker.update(1, {"step": "Operation complete", "status": "success"})

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of groups by {self.group_field} completed", details={
                "analyzed_groups": analysis_results.get('analyzed_groups', 0),
                "mean_variation": analysis_results.get('overall_stats', {}).get('mean_variation', 0),
                "collapsible_groups": analysis_results.get('collapsibility_analysis', {}).get(
                    'collapsible_groups_count', 0) if 'collapsibility_analysis' in analysis_results else 0
            })

            return result

        except Exception as e:
            logger.exception(f"Error in group analysis for {self.group_field}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing groups by {self.group_field}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing groups by {self.group_field}: {str(e)}"
            )