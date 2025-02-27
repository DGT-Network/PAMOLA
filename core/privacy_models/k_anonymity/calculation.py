"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: k-Anonymity Processor
------------------------------
This module implements **k-Anonymity**, ensuring that each
record in a dataset is **indistinguishable from at least `k-1` others**
based on a set of quasi-identifiers.

Features:
- **Efficient Pandas operations (`groupby()`, `value_counts()`)**.
- **Modular functions extracted into `core.utils.anonymization` for reuse**.
- **Masking & suppression options** for anonymization.
- **Adaptive k-levels** for different groups.
- **Tracking `k` values per row** in a new column.
- **Progress tracking (`tqdm`) for large datasets**.
- **Parallel processing with `Dask` for very large datasets**.
- **Integrated metrics** for privacy, utility, and fidelity assessment.
- **Visualization tools** for k-anonymity assessment.
- **Comprehensive logging** for debugging and auditing.
- **Reporting capabilities** for documentation and compliance.
- **Optimized memory usage** for large datasets.

NOTE: This module requires `pandas`, `numpy`, `tqdm`, `matplotlib`, and `dask`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Standard libraries
import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Optional, Any

# Data processing libraries
import pandas as pd
from dask import dataframe as dd

# PAMOLA imports
from core import config
# Import reporting modules
from core.privacy_models.k_anonymity.ka_reporting import (
    generate_compliance_report
)
from core.utils.file_io import write_json, write_csv
from core.privacy_models.base import BasePrivacyModelProcessor
from core.metrics.fidelity.statistical_fidelity import StatisticalFidelityMetric, calculate_fidelity_metrics
# Import metrics modules
from core.metrics.privacy.disclosure_risk import DisclosureRiskMetric, KAnonymityRiskMetric, \
    calculate_disclosure_risk_metrics
from core.metrics.utility.information_loss import InformationLossMetric, calculate_information_loss_metrics
from core.utils import progress
from core.utils.group_processing import compute_group_sizes, adaptive_k_lookup, validate_anonymity_inputs, \
    optimize_memory_usage

# Visualization libraries

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KAnonymityProcessor(BasePrivacyModelProcessor, ABC):
    """
    k-Anonymity Processor for enforcing k-anonymity in datasets.

    This class extends BasePrivacyModelProcessor and applies k-Anonymity
    by grouping records based on quasi-identifiers and ensuring each
    group meets the required `k` threshold.

    Methods:
    --------
    - evaluate_privacy(): Assesses the dataset's k-anonymity level.
    - apply_model(): Applies k-Anonymity by suppressing or masking non-compliant records.
    - enrich_with_k_values(): Adds k-values as a column to the dataset.
    - calculate_risk(): Calculates re-identification risk for each record.
    - calculate_metrics(): Calculates comprehensive metrics for privacy, utility, and fidelity.
    - generate_report(): Creates a comprehensive anonymization report.
    - visualize_k_distribution(): Plots the distribution of k-values.
    - visualize_risk_heatmap(): Creates a heatmap of re-identification risk.
    """

    def __init__(self,
                 k: int = 3,
                 adaptive_k: Optional[Dict[Tuple, int]] = None,
                 suppression: bool = True,
                 mask_value: str = "MASKED",
                 use_dask: bool = False,
                 log_level: str = "INFO",
                 progress_tracking: bool = True,
                 config_override: Optional[Dict[str, Any]] = None):
        """
        Initializes the k-Anonymity processor.

        Parameters:
        -----------
        k : int, optional
            The minimum group size required for k-anonymity (default: 3).
        adaptive_k : dict, optional
            A dictionary defining different `k` levels per quasi-identifier group.
        suppression : bool, optional
            Whether to suppress non-compliant records (default: True).
        mask_value : str, optional
            The value to replace masked cells with (default: "MASKED").
        use_dask : bool, optional
            Whether to use Dask for parallel processing on large datasets (default: False).
        log_level : str, optional
            The logging level (default: "INFO").
        progress_tracking : bool, optional
            Whether to display progress bars (default: True).
        config_override : dict, optional
            Dictionary with configuration values to override defaults.
        """
        # Merge configuration: defaults from config, overridden by explicit parameters
        self.config = getattr(config, "K_ANONYMITY_DEFAULTS", {
            "k": 5,
            "use_dask": False,
            "mask_value": "MASKED",
            "suppression": True,
            "visualization": {
                "hist_bins": 20,
                "save_format": "png",
            }
        })

        # Apply configuration overrides if provided
        if config_override:
            self.config.update(config_override)

        # Set parameters, prioritizing explicitly provided values over config
        self.k = k if k is not None else self.config.get("k", 3)
        self.adaptive_k = adaptive_k if adaptive_k else {}
        self.suppression = suppression if suppression is not None else self.config.get("suppression", True)
        self.mask_value = mask_value if mask_value is not None else self.config.get("mask_value", "MASKED")
        self.use_dask = use_dask if use_dask is not None else self.config.get("use_dask", False)
        self.progress_tracking = progress_tracking

        # Configure visualization settings
        self.viz_config = self.config.get("visualization", {})

        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(numeric_level)

        # Performance metrics and report data
        self.execution_times = {}
        self.report_data = {}

        # Initialize metrics
        self.risk_metric = DisclosureRiskMetric()
        self.k_risk_metric = KAnonymityRiskMetric(k_threshold=self.k)
        self.info_loss_metric = InformationLossMetric()
        self.stat_fidelity_metric = StatisticalFidelityMetric()

        logger.info(
            f"Initialized KAnonymityProcessor with k={self.k}, suppression={self.suppression}, use_dask={self.use_dask}")
        logger.debug(f"Full configuration: {self.config}")

    def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the dataset's k-Anonymity level.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to be evaluated.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        kwargs : dict
            Additional parameters for evaluation.
            - detailed_metrics: bool - Whether to include detailed metrics
            - risk_threshold: float - Threshold for high-risk records (default: 0.5)
            - sensitive_attributes: list - Columns containing sensitive information
            - store_for_report: bool - Whether to store results for later reporting (default: True)

        Returns:
        --------
        dict
            Privacy assessment results:
            - "min_k": The minimum k found in the dataset.
            - "group_sizes": A dictionary of group sizes.
            - "at_risk_records": Count of records failing k-Anonymity.
            - "at_risk_groups": Count of groups failing k-Anonymity.
            - "compliant": Whether the dataset meets k-anonymity
            - Additional metrics if detailed_metrics=True
        """
        start_time = time()
        logger.info(f"Evaluating k-anonymity for {len(data)} records with {len(quasi_identifiers)} quasi-identifiers")

        try:
            # Input validation
            validate_anonymity_inputs(data, quasi_identifiers, self.k)

            # Extract additional parameters
            detailed_metrics = kwargs.get('detailed_metrics', False)
            risk_threshold = kwargs.get('risk_threshold', 0.5)
            sensitive_attributes = kwargs.get('sensitive_attributes', None)
            store_for_report = kwargs.get('store_for_report', True)

            # Use metrics classes to evaluate privacy
            risk_results = calculate_disclosure_risk_metrics(
                data=data,
                quasi_identifiers=quasi_identifiers,
                sensitive_attributes=sensitive_attributes,
                k_threshold=self.k,
                risk_threshold=risk_threshold,
                detailed=detailed_metrics
            )

            # Extract key metrics
            k_anonymity_risk = risk_results["k_anonymity_risk"]
            disclosure_risk = risk_results["disclosure_risk"]

            # Prepare result
            result = {
                "min_k": k_anonymity_risk["min_k"],
                "at_risk_records": k_anonymity_risk["records_in_small_groups"],
                "at_risk_groups": k_anonymity_risk["groups_below_threshold"],
                "percentage_at_risk": k_anonymity_risk["percent_records_at_risk"],
                "compliant": k_anonymity_risk["compliant"],
                "prosecutor_risk": disclosure_risk["prosecutor_risk"],
                "journalist_risk": disclosure_risk["journalist_risk"],
                "marketer_risk": disclosure_risk["marketer_risk"]
            }

            # Add l-diversity results if sensitive attributes provided
            if sensitive_attributes and "l_diversity_risk" in risk_results:
                result["l_diversity"] = risk_results["l_diversity_risk"]

            # Add detailed metrics if requested
            if detailed_metrics and "group_sizes" in disclosure_risk:
                result["group_sizes"] = disclosure_risk["group_sizes"]
                result["k_distribution"] = k_anonymity_risk.get("k_distribution", {})

            # Record execution time
            end_time = time()
            execution_time = end_time - start_time
            self.execution_times['evaluate_privacy'] = execution_time
            result['execution_time'] = execution_time

            # Store results for reporting if requested
            if store_for_report:
                self.report_data['privacy_evaluation'] = result
                self.report_data['quasi_identifiers'] = quasi_identifiers
                if sensitive_attributes:
                    self.report_data['sensitive_attributes'] = sensitive_attributes
                self.report_data['record_count'] = len(data)
                self.report_data['evaluation_timestamp'] = datetime.now().isoformat()

            logger.info(f"k-Anonymity evaluation completed in {execution_time:.2f} seconds. "
                        f"Min k: {result['min_k']}, At-risk records: {result['at_risk_records']}")
            return result

        except Exception as e:
            logger.error(f"Error during privacy evaluation: {e}")
            raise

    def apply_model(self, data: pd.DataFrame, quasi_identifiers: List[str], suppression: Optional[bool] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply k-Anonymity by suppressing or masking non-compliant records.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to be anonymized.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        suppression : bool, optional
            Whether to suppress or mask records (overrides self.suppression if provided).
        kwargs : dict
            Additional parameters for model application.
            - npartitions: int - Number of partitions for Dask (default: 4)
            - add_k_column: bool - Whether to add a column with k values (default: False)
            - mask_columns: list - Columns to mask (default: quasi_identifiers)
            - return_info: bool - Whether to return info dictionary (default: False)
            - optimize_memory: bool - Whether to optimize memory usage (default: False)
            - store_for_report: bool - Whether to store results for later reporting (default: True)
            - original_data: pd.DataFrame - Original data for metrics calculation

        Returns:
        --------
        pd.DataFrame or Tuple[pd.DataFrame, Dict]
            The anonymized dataset, optionally with info dictionary if return_info=True.
        """
        start_time = time()
        progress_bar = None

        try:
            # Input validation
            validate_anonymity_inputs(data, quasi_identifiers, self.k)

            # Use provided suppression flag or instance default
            suppression = self.suppression if suppression is None else suppression

            # Extract additional parameters
            npartitions = kwargs.get('npartitions', 4)
            add_k_column = kwargs.get('add_k_column', False)
            mask_columns = kwargs.get('mask_columns', quasi_identifiers)
            return_info = kwargs.get('return_info', False)
            optimize_memory = kwargs.get('optimize_memory', False)
            store_for_report = kwargs.get('store_for_report', True)
            original_data = kwargs.get('original_data', None)

            logger.info(f"Applying k-anonymity (k={self.k}) on {len(data)} records using "
                        f"{'suppression' if suppression else 'masking'}")

            # Optimize memory if requested
            if optimize_memory:
                logger.info("Optimizing memory usage")
                data = optimize_memory_usage(data, quasi_identifiers)

            # Convert to Dask DataFrame for large datasets if enabled
            if self.use_dask and not isinstance(data, dd.DataFrame):
                logger.info(f"Converting to Dask DataFrame with {npartitions} partitions")
                data = dd.from_pandas(data, npartitions=npartitions)

            # Calculate group sizes and identify at-risk groups
            group_sizes = compute_group_sizes(data, quasi_identifiers, self.use_dask)
            k_thresholds = adaptive_k_lookup(group_sizes, self.k, self.adaptive_k)
            at_risk_mask = pd.Series(group_sizes < k_thresholds)
            at_risk_groups = group_sizes[at_risk_mask].index

            # Initialize progress tracking if enabled
            if self.progress_tracking:
                progress_bar = progress.progress_logger(
                    f"Applying k-Anonymity: {'Suppression' if suppression else 'Masking'}",
                    len(at_risk_groups)
                )

            # Apply anonymization
            if suppression:
                # Suppression: Remove records in at-risk groups
                logger.info(f"Suppressing {(at_risk_mask * group_sizes).sum()} records in {len(at_risk_groups)} groups")
                result = data[~data.set_index(quasi_identifiers).index.isin(at_risk_groups)]
            else:
                # Masking: Replace values in specified columns
                logger.info(f"Masking {len(mask_columns)} columns in {len(at_risk_groups)} groups")
                result = data.copy() if not self.use_dask else data

                for i, col in enumerate(mask_columns):
                    if col in data.columns:
                        result.loc[data.set_index(quasi_identifiers).index.isin(at_risk_groups), col] = self.mask_value
                        if progress_bar:
                            progress_bar.update(1)
                    else:
                        logger.warning(f"Column {col} not found in dataset, skipping masking")

            # Add k-values column if requested
            if add_k_column:
                logger.info("Adding k-values column to the result")
                k_column = f"K_{'_'.join([qi[:3].upper() for qi in quasi_identifiers])}"
                result = result.merge(
                    group_sizes.rename(k_column),
                    left_on=quasi_identifiers,
                    right_index=True,
                    how='left'
                )

            # Convert back to pandas DataFrame if using Dask
            if self.use_dask and isinstance(result, dd.DataFrame):
                logger.info("Converting result back to pandas DataFrame")
                result = result.compute()

            # Calculate metrics if original data is provided
            metrics = {}
            if isinstance(original_data, pd.DataFrame):
                logger.info("Calculating metrics for anonymized data")
                metrics = self.calculate_metrics(original_data, result, quasi_identifiers)
            else:
                logger.warning("Skipping metric calculation: original_data is None or invalid")

            # Record execution time
            end_time = time()
            execution_time = end_time - start_time
            self.execution_times['apply_model'] = execution_time

            # Prepare info dictionary
            info = {
                "original_records": len(data),
                "anonymized_records": len(result),
                "records_removed": len(data) - len(result) if suppression else 0,
                "groups_affected": len(at_risk_groups),
                "execution_time": execution_time,
                "method": "suppression" if suppression else "masking",
                "k_value": self.k,
                "quasi_identifiers": quasi_identifiers,
                "timestamp": datetime.now().isoformat()
            }

            # Add metrics to info if calculated
            if metrics:
                info["metrics"] = metrics

            # Store results for reporting if requested
            if store_for_report:
                self.report_data['anonymization_result'] = info
                self.report_data['anonymization_timestamp'] = datetime.now().isoformat()
                self.report_data['anonymization_method'] = "suppression" if suppression else "masking"
                if metrics:
                    self.report_data['metrics'] = metrics

            logger.info(f"k-Anonymity application completed in {execution_time:.2f} seconds. "
                        f"Result has {len(result)} records ({info['records_removed']} removed).")

            # Return result with or without info
            return (result, info) if return_info else result

        except Exception as e:
            logger.error(f"Error during model application: {e}")
            raise
        finally:
            # Ensure progress bar is closed
            if progress_bar:
                progress_bar.close()

    def enrich_with_k_values(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """
        Adds a column with the k-anonymity value for each record.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to enrich.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.

        Returns:
        --------
        pd.DataFrame
            Dataset with added k-value column.
        """
        logger.info(f"Enriching dataset with k-values for {len(quasi_identifiers)} quasi-identifiers")

        try:
            # Input validation
            validate_anonymity_inputs(data, quasi_identifiers, self.k)

            # Compute group sizes
            group_sizes = compute_group_sizes(data, quasi_identifiers, self.use_dask)

            # Generate column name
            k_column = f"K_{'_'.join([qi[:3].upper() for qi in quasi_identifiers])}"

            # Convert to Dask if enabled
            if self.use_dask and not isinstance(data, dd.DataFrame):
                data = dd.from_pandas(data, npartitions=4)

            # Add k-values column
            result = data.merge(
                group_sizes.rename(k_column),
                left_on=quasi_identifiers,
                right_index=True,
                how='left'
            )

            # Add risk column based on k-values
            risk_column = f"RISK_{'_'.join([qi[:3].upper() for qi in quasi_identifiers])}"
            result[risk_column] = result[k_column].apply(lambda k: 1 / k if k > 0 else 1)

            # Convert back if using Dask
            if self.use_dask and isinstance(result, dd.DataFrame):
                result = result.compute()

            logger.info(f"Added {k_column} and {risk_column} columns to dataset")
            return result

        except Exception as e:
            logger.error(f"Error during dataset enrichment: {e}")
            raise

    def calculate_risk(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """
        Calculates re-identification risk for each record based on k-anonymity.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to analyze.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.

        Returns:
        --------
        pd.DataFrame
            Dataset with added risk column.
        """
        try:
            # Get dataset with k-values
            result = self.enrich_with_k_values(data, quasi_identifiers)
            return result

        except Exception as e:
            logger.error(f"Error during risk calculation: {e}")
            raise

    def calculate_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame,
                          quasi_identifiers: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Calculates comprehensive metrics for anonymized data.

        This method combines privacy, utility, and fidelity metrics
        to provide a complete assessment of the anonymization process.

        Parameters:
        -----------
        original_data : pd.DataFrame
            The original dataset.
        anonymized_data : pd.DataFrame
            The anonymized dataset.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        **kwargs : dict
            Additional parameters for metrics calculation.
            - sensitive_attributes: list - Columns containing sensitive information
            - numerical_columns: list - Columns to treat as numerical
            - distribution_tests: bool - Whether to include distribution tests (default: False)

        Returns:
        --------
        dict
            Dictionary with all calculated metrics:
            - "privacy_metrics": Disclosure risk metrics
            - "utility_metrics": Information loss metrics
            - "fidelity_metrics": Statistical fidelity metrics
        """
        logger.info("Calculating comprehensive metrics for anonymized data")

        try:
            metrics = {}

            # Extract additional parameters
            sensitive_attributes = kwargs.get('sensitive_attributes', None)
            numerical_columns = kwargs.get('numerical_columns', None)
            distribution_tests = kwargs.get('distribution_tests', False)

            # Calculate privacy metrics
            privacy_metrics = calculate_disclosure_risk_metrics(
                data=anonymized_data,
                quasi_identifiers=quasi_identifiers,
                sensitive_attributes=sensitive_attributes,
                k_threshold=self.k
            )
            metrics["privacy_metrics"] = privacy_metrics

            # Calculate utility metrics
            utility_metrics = calculate_information_loss_metrics(
                original_data=original_data,
                anonymized_data=anonymized_data,
                numerical_columns=numerical_columns
            )
            metrics["utility_metrics"] = utility_metrics

            # Calculate fidelity metrics
            fidelity_metrics = calculate_fidelity_metrics(
                original_data=original_data,
                anonymized_data=anonymized_data,
                distribution_tests=distribution_tests,
                columns=numerical_columns
            )
            metrics["fidelity_metrics"] = fidelity_metrics

            # Store metrics for reporting
            self.report_data['metrics'] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error during metrics calculation: {e}")
            raise

    def generate_report(self, output_path: Optional[str] = None, include_visualizations: bool = True,
                        report_format: str = "json", regulation: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a comprehensive report about the anonymization process.

        Parameters:
        -----------
        output_path : str, optional
            Path to save the report.
        include_visualizations : bool, optional
            Whether to include visualizations in the report.
        report_format : str, optional
            Report format: 'json', 'html', or 'text'.
        regulation : str, optional
            If provided, generates a compliance report for the specified regulation
            (e.g., 'GDPR', 'HIPAA').

        Returns:
        --------
        dict
            The complete anonymization report.
        """
        try:
            # Prepare report data
            report_data = {
                "k_anonymity_configuration": {
                    "k_value": self.k,
                    "suppression": self.suppression,
                    "mask_value": self.mask_value if not self.suppression else None,
                    "adaptive_k": bool(self.adaptive_k),
                },
                "execution_times": self.execution_times,
                **self.report_data,  # Merge existing report data
            }

            # Use compliance report if regulation is specified
            if regulation:
                return generate_compliance_report(report_data=report_data, output_path=output_path,
                                                  regulation=regulation)

            # Save the report if an output path is provided
            if output_path:
                if report_format.lower() == "json":
                    write_json(report_data, output_path)
                else:
                    logger.warning(f"Unsupported report format `{report_format}`. Saving as JSON instead.")
                    write_json(report_data, output_path)

            return report_data

        except Exception as e:
            logger.error(f"Error generating anonymization report: {e}", exc_info=True)
            raise

    def visualize_k_distribution(self, data: pd.DataFrame, k_column: str, save_path: Optional[str] = None, **kwargs):
        """
        Creates a visualization of k-anonymity distribution.

        This is a wrapper around the visualization utility function.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset with k-values.
        k_column : str
            Name of the column containing k-values.
        save_path : str, optional
            Path to save the visualization.
        **kwargs : dict
            Additional parameters for visualization.

        Returns:
        --------
        tuple
            Figure object and path to saved figure (if saved, otherwise None).
        """
        # Get visualization settings from config
        bins = kwargs.get('bins', self.viz_config.get('hist_bins', 20))
        save_format = kwargs.get('save_format', self.viz_config.get('save_format', 'png'))

        try:
            from core.privacy_models.k_anonymity.ka_visualization import visualize_k_distribution

            # Call the visualization utility
            fig, saved_path = visualize_k_distribution(
                data=data,
                k_column=k_column,
                save_path=save_path,
                bins=bins,
                save_format=save_format,
                **kwargs
            )

            # Store path for reporting if saved
            if saved_path and hasattr(self, 'report_data'):
                if 'visualization_paths' not in self.report_data:
                    self.report_data['visualization_paths'] = {}
                self.report_data['visualization_paths']['k_distribution'] = saved_path

            return fig, saved_path

        except Exception as e:
            logger.error(f"Error during k-distribution visualization: {e}")
            raise

    def visualize_risk_heatmap(self, data: pd.DataFrame, risk_column: str, feature_columns: List[str],
                               save_path: Optional[str] = None, **kwargs):
        """
        Creates a heatmap visualizing re-identification risk.

        This is a wrapper around the visualization utility function.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with risk values.
        risk_column : str
            Name of the column containing risk values.
        feature_columns : list[str]
            Columns to include in the heatmap.
        save_path : str, optional
            Path to save the visualization.
        **kwargs : dict
            Additional parameters for visualization.

        Returns:
        --------
        tuple
            Figure object and path to saved figure (if saved, otherwise None).
        """
        # Get visualization settings from config
        save_format = kwargs.get('save_format', self.viz_config.get('save_format', 'png'))

        try:
            from core.privacy_models.k_anonymity.ka_visualization import visualize_risk_heatmap

            # Call the visualization utility
            fig, saved_path = visualize_risk_heatmap(
                data=data,
                risk_column=risk_column,
                feature_columns=feature_columns,
                save_path=save_path,
                save_format=save_format,
                **kwargs
            )

            # Store path for reporting if saved
            if saved_path and hasattr(self, 'report_data'):
                if 'visualization_paths' not in self.report_data:
                    self.report_data['visualization_paths'] = {}
                self.report_data['visualization_paths']['risk_heatmap'] = saved_path

            return fig, saved_path

        except Exception as e:
            logger.error(f"Error during risk heatmap visualization: {e}")
            raise

    def _validate_input(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> None:
        """
        Validates input data and parameters.

        This method uses the validate_anonymity_inputs utility function.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to validate.
        quasi_identifiers : list[str]
            List of quasi-identifiers to validate.
        """
        validate_anonymity_inputs(data, quasi_identifiers, self.k)

    def get_execution_summary(self) -> Dict[str, float]:
        """
        Returns a summary of execution times for different operations.

        Returns:
        --------
        dict
            Dictionary of operation names and execution times.
        """
        return self.execution_times

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current configuration of the k-anonymity processor.

        Returns:
        --------
        dict
            Dictionary containing all configuration parameters.
        """
        return self.config


    def export_metrics(self, path: str, format: str = "json") -> str:
        """
        Exports calculated metrics to a file.

        Parameters:
        -----------
        path : str
            Path where to save the metrics.
        format : str, optional
            Format to save metrics in: 'json', 'csv' (default: 'json').

        Returns:
        --------
        str
            Path to the saved metrics file.
        """
        if not hasattr(self, "report_data") or "metrics" not in self.report_data:
            logger.warning("No metrics available to export")
            return ""  # Return an empty string

        try:
            metrics = self.report_data["metrics"]

            # Create directory if it doesn't exist
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Export metrics in the requested format
            if format.lower() == "json":
                return write_json(metrics, path)

            elif format.lower() == "csv":
                return write_csv(metrics, path)

            else:
                logger.warning(f"Unsupported format: {format}. Using JSON instead.")
                return write_json(metrics, path)

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}", exc_info=True)
            raise

    def compare_anonymized_datasets(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame,
                                    quasi_identifiers: List[str], **kwargs) -> Dict[str, Any]:
        """
        Compares two anonymized datasets to evaluate their relative privacy and utility.

        Parameters:
        -----------
        dataset1 : pd.DataFrame
            First anonymized dataset.
        dataset2 : pd.DataFrame
            Second anonymized dataset.
        quasi_identifiers : list[str]
            List of quasi-identifiers used for anonymization.
        **kwargs : dict
            Additional parameters for comparison.

        Returns:
        --------
        dict
            Comparison results with privacy and utility metrics for both datasets.
        """
        try:
            logger.info("Comparing two anonymized datasets")

            # Calculate k-anonymity metrics for both datasets
            ka_risk1 = KAnonymityRiskMetric(k_threshold=self.k)
            ka_metrics1 = ka_risk1.calculate(dataset1, quasi_identifiers)

            ka_risk2 = KAnonymityRiskMetric(k_threshold=self.k)
            ka_metrics2 = ka_risk2.calculate(dataset2, quasi_identifiers)

            # If original data is provided, calculate utility metrics
            utility_comparison = {}
            original_data = kwargs.get('original_data', None)
            if isinstance(original_data, pd.DataFrame):  # Ensures correct type
                logger.info("Calculating utility metrics for anonymized data")
                info_loss1 = InformationLossMetric()
                utility1 = info_loss1.calculate(original_data, dataset1)

                info_loss2 = InformationLossMetric()
                utility2 = info_loss2.calculate(original_data, dataset2)

                utility_comparison = {
                    "dataset1": {
                        "information_loss": utility1["overall_information_loss"],
                        "records_loss": utility1["total_records_loss"]
                    },
                    "dataset2": {
                        "information_loss": utility2["overall_information_loss"],
                        "records_loss": utility2["total_records_loss"]
                    },
                    "difference": {
                        "information_loss": utility1["overall_information_loss"] - utility2["overall_information_loss"],
                        "records_loss": utility1["total_records_loss"] - utility2["total_records_loss"]
                    }
                }

            # Prepare comparison result
            result = {
                "privacy_comparison": {
                    "dataset1": {
                        "min_k": ka_metrics1["min_k"],
                        "compliant": ka_metrics1["compliant"],
                        "at_risk_records": ka_metrics1["records_in_small_groups"],
                        "at_risk_percentage": ka_metrics1["percent_records_at_risk"]
                    },
                    "dataset2": {
                        "min_k": ka_metrics2["min_k"],
                        "compliant": ka_metrics2["compliant"],
                        "at_risk_records": ka_metrics2["records_in_small_groups"],
                        "at_risk_percentage": ka_metrics2["percent_records_at_risk"]
                    },
                    "difference": {
                        "min_k": ka_metrics1["min_k"] - ka_metrics2["min_k"],
                        "at_risk_percentage": ka_metrics1["percent_records_at_risk"] - ka_metrics2[
                            "percent_records_at_risk"]
                    }
                }
            }

            # Add utility comparison if available
            utility1 = {}  # Initialize empty dictionaries
            utility2 = {}
            if utility_comparison:
                result["utility_comparison"] = utility_comparison

                # Calculate privacy-utility trade-off score
                # Higher score means better balance (higher privacy with lower information loss)
                privacy_score1 = min(ka_metrics1["min_k"] / self.k * 100, 100)
                privacy_score2 = min(ka_metrics2["min_k"] / self.k * 100, 100)

                utility_score1 = 100 - utility1["overall_information_loss"]
                utility_score2 = 100 - utility2["overall_information_loss"]

                tradeoff_score1 = 0.5 * privacy_score1 + 0.5 * utility_score1
                tradeoff_score2 = 0.5 * privacy_score2 + 0.5 * utility_score2

                result["privacy_utility_tradeoff"] = {
                    "dataset1_score": tradeoff_score1,
                    "dataset2_score": tradeoff_score2,
                    "better_tradeoff": "dataset1" if tradeoff_score1 > tradeoff_score2 else "dataset2"
                }

            logger.info(f"Dataset comparison completed")
            return result

        except Exception as e:
            logger.error(f"Error during dataset comparison: {e}")
            raise

    def batch_anonymize(self, data: pd.DataFrame, quasi_identifiers: List[str],
                        k_values: List[int], **kwargs) -> Dict[int, pd.DataFrame]:
        """
        Anonymizes a dataset using multiple k values for comparison.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to anonymize.
        quasi_identifiers : list[str]
            List of quasi-identifiers to use.
        k_values : list[int]
            List of k values to try.
        **kwargs : dict
            Additional parameters for anonymization.

        Returns:
        --------
        dict
            Dictionary mapping k values to anonymized datasets.
        """
        try:
            logger.info(f"Batch anonymizing dataset with {len(k_values)} different k values")

            results = {}
            comparison_metrics = []

            # Save original k value
            original_k = self.k

            for k in k_values:
                logger.info(f"Anonymizing with k={k}")

                # Set k value for this iteration
                self.k = k

                # Anonymize data
                anonymized_data, info = self.apply_model(
                    data=data,
                    quasi_identifiers=quasi_identifiers,
                    return_info=True,
                    original_data=data,
                    **kwargs
                )

                # Store result
                results[k] = anonymized_data

                # Store metrics for comparison
                if 'metrics' in info:
                    comparison_metrics.append({
                        "k_value": k,
                        "privacy": info['metrics'].get('privacy_metrics', {}),
                        "utility": info['metrics'].get('utility_metrics', {}),
                        "records_removed": info['records_removed'],
                        "percent_removed": round(100 * info['records_removed'] / info['original_records'], 2)
                    })

            # Restore original k value
            self.k = original_k

            # Store comparison metrics for reporting
            self.report_data['batch_comparison'] = comparison_metrics

            logger.info(f"Batch anonymization completed for {len(k_values)} k values")
            return results

        except Exception as e:
            logger.error(f"Error during batch anonymization: {e}")
            # Restore original k value in case of error
            original_k: int = self.k  # Ensure a valid default value
            self.k = original_k
            raise