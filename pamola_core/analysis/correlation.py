"""
PAMOLA.CORE - Correlation & Relationships Module
------------------------------------------------
Module:        Correlation Analyzer
Package:       pamola_core.analysis
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides robust correlation analysis utilities that normalize all results to
  pandas DataFrames for consistent downstream processing and visualization.
  Includes automatic categorical-to-numeric mapping, input validation, multiple
  correlation methods (pearson/spearman/kendall) and optional chart generation
  (matrix / heatmap) via the visualization subsystem.

Key Features:
  - Normalizes correlation outputs to DataFrames for consistent APIs
  - Supports single-variable, pairwise and selected-variable analyses
  - Automatic categorical string mapping to numeric (configurable common mappings)
  - Input validation with clear error messages for missing/invalid columns
  - Optional chart generation using create_correlation_matrix and create_heatmap
  - Backward-compatible raw_result outputs (Series / scalar / DataFrame)
  - Safe defaults and logging for production use

Dependencies:
  - pandas  - DataFrame operations
  - numpy   - Numeric operations and NaN handling
  - typing  - Type hints and validation
  - datetime/pathlib/logging - IO and logging utilities
  - pamola_core.utils.visualization - chart generation helpers
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Any, Tuple, Set
import logging
from pamola_core.utils.visualization import create_correlation_matrix, create_heatmap

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    A class for performing correlation analysis on pandas DataFrames with visualization.

    All results are normalized to DataFrames for consistent chart generation.
    """

    SUPPORTED_METHODS = ["pearson", "spearman", "kendall"]
    SUPPORTED_CHARTS = ["matrix", "heatmap"]
    SUPPORTED_VIZ_FORMATS = ["png", "jpg", "svg", "html"]
    CATEGORICAL_MAPPINGS = {
        # Common yes/no variations
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        # Boolean string variations
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
    }

    def __init__(self):
        """Initialize the CorrelationAnalyzer."""

    def _validate_method(self, method: str) -> None:
        """Validate the correlation method."""
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Method must be one of {self.SUPPORTED_METHODS}, got '{method}'"
            )

    def _validate_viz_format(self, viz_format: str) -> None:
        """Validate the vizualization format."""
        if viz_format not in self.SUPPORTED_VIZ_FORMATS:
            raise ValueError(
                f"Vizualization format must be one of {self.SUPPORTED_VIZ_FORMATS}, got '{viz_format}'"
            )

    def _validate_output_chart(self, output_chart: Union[str, List[str]]) -> List[str]:
        """Validate and normalize output_chart parameter."""
        if isinstance(output_chart, str):
            chart_types = [output_chart]
        elif isinstance(output_chart, list):
            chart_types = output_chart
        else:
            raise ValueError(
                f"output_chart must be str or List[str], got {type(output_chart)}"
            )

        invalid_charts = [
            chart for chart in chart_types if chart not in self.SUPPORTED_CHARTS
        ]
        if invalid_charts:
            raise ValueError(
                f"Invalid chart types: {invalid_charts}. Must be one of {self.SUPPORTED_CHARTS}"
            )

        return chart_types

    def _validate_columns(self, df: pd.DataFrame, columns: Optional[List[str]]) -> None:
        """Validate that specified columns exist in the DataFrame."""
        if columns is not None:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    def _map_binary_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert binary columns (boolean or categorical strings with ≤2 values) to numeric."""

        # Handle boolean and object columns
        if series.dtype in ["bool", "object"]:
            unique_vals = series.dropna().astype(str).str.lower().unique()

            if len(unique_vals) <= 2 and all(
                val in self.CATEGORICAL_MAPPINGS for val in unique_vals
            ):
                # Use custom mapping if values are in CATEGORICAL_MAPPINGS
                mapping = {}
                for val in series.dropna().unique():
                    mapping[val] = self.CATEGORICAL_MAPPINGS[str(val).lower()]
                return series.map(mapping).fillna(series)
            elif series.dtype == "bool":
                # Default boolean conversion: True -> 1, False -> 0
                return series.astype(int)

        return series

    def _prepare_data(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Prepare data for correlation analysis."""
        data = df.copy(deep=True)

        if columns is not None:
            data = data[columns]

        # Single loop: convert binary to numeric AND filter zero variance columns
        processed_cols = []
        removed_cols = []

        for col in data.columns:
            # Step 1: Convert binary to numeric
            converted_series = self._map_binary_to_numeric(data[col])

            # Step 2: Check if it's numeric after conversion
            if pd.api.types.is_numeric_dtype(converted_series):
                # Step 3: Check variance (zero variance = constant column)
                variance = converted_series.var()
                if variance > 1e-10:  # Keep columns with variance > threshold
                    processed_cols.append(col)
                    data[col] = converted_series  # Update with converted values
                else:
                    removed_cols.append(col)
                    # Don't add to processed_cols = effectively removed
            # If not numeric after conversion, column is automatically excluded

        # Report what was removed
        if removed_cols:
            logger.warning(f"Removed constant columns (zero variance): {removed_cols}")

        # Filter to only processed columns
        if not processed_cols:
            raise ValueError("No suitable columns found for correlation analysis")

        final_data = data[processed_cols]

        logger.info(f"Final columns for correlation: {list(final_data.columns)}")
        return final_data

    def _calculate_correlation_result(
        self, clean_data: pd.DataFrame, columns: Optional[List[str]], method: str
    ) -> Tuple[pd.DataFrame, str]:
        """
        Calculate correlation and return normalized DataFrame result with result type.

        Returns:
            Tuple of (DataFrame result, result_type)
            result_type can be: "all_variables", "single_variable", "pairwise", "selected_variables"
        """
        if columns is None:
            # Return correlation matrix of all variables
            result = clean_data.corr(method=method)
            result_type = "all_variables"

        elif len(columns) == 1:
            # Return correlation of specified column with all others
            col_name = columns[0]
            if col_name not in clean_data.columns:
                raise ValueError(f"Column '{col_name}' not found in numeric columns")

            corr_matrix = clean_data.corr(method=method)
            correlation_series = corr_matrix[col_name].drop(
                col_name
            )  # Exclude self-correlation

            # Use vertical DataFrame (N rows × 1 column)
            result = pd.DataFrame(
                correlation_series.values.reshape(-1, 1),  # Convert to vertical
                columns=[f"{col_name}_correlation"],
                index=correlation_series.index,
            )
            result_type = "single_variable"

        elif len(columns) == 2:
            # For 2 columns, return full 2x2 correlation matrix instead of 1x1
            col1, col2 = columns[0], columns[1]

            missing = [col for col in [col1, col2] if col not in clean_data.columns]
            if missing:
                raise ValueError(f"Columns not found in numeric columns: {missing}")

            # Return full 2x2 correlation matrix for better visualization
            result = clean_data[[col1, col2]].corr(method=method)
            result_type = "pairwise"

        else:
            # Return correlation matrix of specified columns
            available_cols = [col for col in columns if col in clean_data.columns]

            if not available_cols:
                raise ValueError("None of the specified columns are numeric/boolean")

            if len(available_cols) < len(columns):
                missing = [col for col in columns if col not in available_cols]
                raise ValueError(f"Some columns are not numeric/boolean: {missing}")

            result = clean_data[available_cols].corr(method=method)
            result_type = "selected_variables"

        return result, result_type

    def _generate_charts(
        self,
        result_df: pd.DataFrame,
        result_type: str,
        method: str,
        chart_types: List[str],
        analysis_dir: str,
        viz_format: str = "html",
    ) -> Union[str, List[str], None]:
        """
        Generate charts based on DataFrame results - now with automatic skip logic for insufficient columns.

        Args:
            result_df: Always a pandas DataFrame
            result_type: Type of result for title customization
            method: Correlation method used
            chart_types: List of chart types to generate
            analysis_dir: Directory to save analysis outputs
            viz_format: Output format for charts (png, jpg, svg, html), default is "html"

        Returns:
            Single file path, list of file paths, or None if charts were skipped
        """
        file_paths = []
        analysis_dir = Path(analysis_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define titles based on result type
        title_templates = {
            "all_variables": f"{method.title()} Correlation - All Variables",
            "single_variable": f"{method.title()} Correlation - Single Variable Analysis",
            "pairwise": f"{method.title()} Correlation - Pairwise Analysis",
            "selected_variables": f"{method.title()} Correlation - Selected Variables",
        }

        for chart_type in chart_types:
            try:
                # Per-chart checks: ensure the DataFrame shape is appropriate for the requested chart
                if chart_type not in {"matrix", "heatmap"}:
                    logger.warning(f"Unsupported chart type requested: {chart_type}")
                    continue

                # Require at least 2x2 for matrix/heatmap visualizations
                if result_df.shape[0] < 2 or result_df.shape[1] < 2:
                    # Allow an exception: pairwise (2 cols) handled by above check; otherwise skip
                    logger.warning(
                        f"Skipping {chart_type}: requires ≥2x2. Shape: {result_df.shape}"
                    )
                    continue

                analysis_filename = (
                    f"correlation_{method}_{chart_type}_{timestamp}.{viz_format}"
                )
                analysis_path = analysis_dir / analysis_filename
                # Generate title
                base_title = title_templates.get(
                    result_type, f"{method.title()} Correlation"
                )
                title = f"{base_title} ({chart_type.title()})"

                # Smart annotation logic based on chart type and DataFrame size
                num_vars = max(result_df.shape)

                if chart_type == "matrix":
                    # Matrix: Always annotate (purpose is to read exact values)
                    annotate = True
                    annotation_params = {
                        "annotation_format": ".3f",  # More precision for exact reading
                        "mask_diagonal": False,  # Show self-correlations (always 1.0)
                        "mask_upper": False,  # Show full matrix
                    }

                elif chart_type == "heatmap":
                    # Heatmap: Conditional annotation (depends on size)
                    annotate = num_vars <= 10  # Only annotate if ≤10 variables
                    annotation_params = {
                        "annotation_format": ".2f" if annotate else None,
                        "annotation_color_threshold": 0.5 if annotate else None,
                    }

                # Common parameters
                common_params = {
                    "data": result_df,
                    "output_path": str(analysis_path),
                    "title": title,
                    "x_label": "Variables",
                    "y_label": "Variables",
                    "backend": "plotly",
                    "viz_format": viz_format,
                    "annotate": annotate,
                }

                # Chart-specific generation
                if chart_type == "matrix":
                    path = create_correlation_matrix(
                        **common_params,
                        colorscale="RdBu_r",  # Diverging: negative=red, positive=blue
                        colorbar_title="Correlation",
                        significant_threshold=0.5,  # Highlight strong correlations
                        **annotation_params,
                    )

                elif chart_type == "heatmap":
                    path = create_heatmap(
                        **common_params,
                        colorscale="Viridis",  # Sequential: better for pattern recognition
                        colorbar_title="Correlation Value",
                        **annotation_params,
                    )

                if path and isinstance(path, str) and not path.startswith("Error"):
                    file_paths.append(path)
                    logger.info(f"Generated {chart_type} chart: {Path(path).name}")
                else:
                    logger.warning(f"Failed to generate {chart_type}: {path}")

            except Exception as e:
                logger.warning(f"Failed to generate {chart_type}: {str(e)}")
                continue

        return (
            file_paths[0]
            if len(chart_types) == 1
            else file_paths if file_paths else None
        )

    def analyze_correlation(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        plot: bool = True,
        output_chart: Union[str, List[str]] = "heatmap",
        analysis_dir: str = "",
        viz_format: str = "html",
    ) -> Dict[str, Any]:
        """
        Analyze correlation coefficients with optional visualization.

        All results are returned as DataFrames for consistency.

        Args:
            df: Input pandas DataFrame
            columns: Optional list of column names to analyze
            method: Correlation method ("pearson", "spearman", "kendall")
            plot: If True, generate charts according to output_chart
            output_chart: Chart type(s) to generate - "matrix", "heatmap", or list of both
            analysis_dir: Directory to save analysis outputs
            viz_format: Output format for charts (png, jpg, svg, html), default is "html"
        Returns:
            Dictionary with keys:
            - "result": Always a pandas DataFrame (normalized from all result types)
            - "result_type": String indicating the type of analysis performed
            - "path": File path(s) to generated chart(s) or None if skipped
            - "raw_result": Original result before DataFrame normalization (for backward compatibility)
        """
        # Validate inputs
        self._validate_method(method)
        self._validate_viz_format(viz_format)
        self._validate_columns(df, columns)
        chart_types = self._validate_output_chart(output_chart)

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Prepare data
        clean_data = self._prepare_data(df, columns)

        # Calculate correlation - now always returns DataFrame
        result_df, result_type = self._calculate_correlation_result(
            clean_data, columns, method
        )

        # Store raw result for backward compatibility
        if result_type == "single_variable":
            # Reconstruct proper Series from vertical DataFrame
            raw_result = pd.Series(
                result_df.iloc[:, 0],
                index=result_df.index,
                name=result_df.columns[0].replace("_correlation", ""),
            )
        elif result_type == "pairwise":
            # For backward compatibility, return the correlation value between the two columns
            col1, col2 = result_df.index[0], (
                result_df.columns[1]
                if len(result_df.columns) > 1
                else result_df.columns[0]
            )
            raw_result = (
                result_df.loc[col1, col2] if col1 != col2 else result_df.iloc[0, 1]
            )
        else:
            raw_result = result_df.copy()

        # Generate charts if requested - automatically skips if insufficient data
        chart_paths = None
        if plot:
            chart_paths = self._generate_charts(
                result_df, result_type, method, chart_types, analysis_dir, viz_format
            )

        return {
            "result": result_df,  # Always DataFrame now
            "result_type": result_type,  # New field to indicate analysis type
            "raw_result": raw_result,  # Backward compatibility
            "path": chart_paths,  # Will be None if charts were skipped
        }


# Convenience function
def analyze_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    plot: bool = True,
    output_chart: Union[str, List[str]] = "heatmap",
    analysis_dir: str = "",
    viz_format: str = "html",
) -> Dict[str, Any]:
    """
    Convenience function to calculate correlation without instantiating the class.
    """
    analyzer = CorrelationAnalyzer()
    return analyzer.analyze_correlation(
        df, columns, method, plot, output_chart, analysis_dir, viz_format
    )
