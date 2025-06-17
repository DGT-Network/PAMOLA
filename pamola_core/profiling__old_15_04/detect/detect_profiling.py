"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
---------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for privacy-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

Licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Data Profiling Processor
--------------------------------
Detects direct identifiers, quasi-identifiers, sensitive attributes,
indirect identifiers, and non-sensitive attributes to support privacy-preserving data transformations.

NOTE: Requires `pandas` and `numpy`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import re
import gc
import time
from typing import List, Tuple, Dict, Any
import pandas as pd
import dask.dataframe as dd
from joblib import Parallel, delayed
from abc import ABC
import logging
import psutil
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from pamola_core.common.regex.patterns import Patterns
from pamola_core.profiling__old_15_04.base import BaseProfilingProcessor

logger = logging.getLogger(__name__)


class SystemResourceMonitor:
    """Monitor and manage system resources for optimal processing."""

    def __init__(self):
        self.total_memory = psutil.virtual_memory().total
        self.cpu_count = psutil.cpu_count(logical=True)
        self.available_memory = psutil.virtual_memory().available

    def get_optimal_settings(self, data_size: int, num_columns: int) -> Dict[str, Any]:
        """
        Determine optimal processing settings based on system capacity and data characteristics.

        Args:
            data_size: Size of dataset in bytes
            num_columns: Number of columns in dataset

        Returns:
            Dict containing optimal processing settings
        """
        # Get current system status
        memory_info = psutil.virtual_memory()
        cpu_info = psutil.cpu_percent(interval=1)

        # Calculate available resources
        available_memory = memory_info.available
        memory_usage_percent = memory_info.percent

        # Determine processing strategy based on data size and system capacity
        memory_ratio = data_size / available_memory

        # Small dataset: Use joblib with threading
        if data_size < 100 * 1024 * 1024:  # < 100MB
            return {
                "strategy": "joblib_threading",
                "n_jobs": min(self.cpu_count, num_columns, 4),
                "backend": "threading",
                "chunk_size": None,
                "use_dask": False,
            }

        # Medium dataset: Use joblib with processes or Dask based on memory
        elif data_size < 500 * 1024 * 1024:  # < 500MB
            if memory_usage_percent < 70:  # System has enough memory
                return {
                    "strategy": "joblib_processes",
                    "n_jobs": min(self.cpu_count // 2, num_columns, 4),
                    "backend": "multiprocessing",
                    "chunk_size": None,
                    "use_dask": False,
                }
            else:  # Memory constrained, use Dask
                return {
                    "strategy": "dask_local",
                    "n_jobs": min(self.cpu_count // 2, 4),
                    "backend": None,
                    "chunk_size": 64 * 1024 * 1024,  # 64MB chunks
                    "use_dask": True,
                    "n_partitions": max(
                        2, min(8, math.ceil(data_size / (64 * 1024 * 1024)))
                    ),
                }

        # Large dataset: Use Dask with optimized settings
        else:  # >= 500MB
            # Check if Dask is available and working
            try:
                import dask.dataframe as dd_test

                # Test basic Dask functionality
                test_df = pd.DataFrame({"test": [1, 2, 3]})
                test_ddf = dd_test.from_pandas(test_df, npartitions=1)
                _ = test_ddf.sum().compute()
                dask_available = True
            except Exception as e:
                logger.warning(f"Dask not available or not working: {str(e)}")
                dask_available = False

            if dask_available:
                # Calculate optimal chunk size based on available memory
                optimal_chunk_size = min(
                    available_memory // 8,  # Use 1/8 of available memory per chunk
                    128 * 1024 * 1024,  # Max 128MB per chunk
                )

                n_partitions = max(
                    4, min(16, math.ceil(data_size / optimal_chunk_size))
                )

                return {
                    "strategy": "dask_optimized",
                    "n_jobs": min(self.cpu_count, 6),  # Cap at 6 workers for stability
                    "backend": None,
                    "chunk_size": optimal_chunk_size,
                    "use_dask": True,
                    "n_partitions": n_partitions,
                    "scheduler": (
                        "threads" if memory_usage_percent < 60 else "synchronous"
                    ),
                }
            else:
                # Fallback to joblib with processes for large datasets
                return {
                    "strategy": "joblib_processes_large",
                    "n_jobs": min(self.cpu_count // 2, 4),
                    "backend": "multiprocessing",
                    "chunk_size": None,
                    "use_dask": False,
                }


class DetectProfilingProcessor(BaseProfilingProcessor, ABC):
    """
    Processor for detecting direct identifiers, quasi-identifiers, sensitive attributes,
    indirect identifiers, and non-sensitive attributes in datasets.
    Auto-manages Dask and Joblib based on system capacity.
    """

    # Category constants
    CATEGORY_DIRECT = "direct"
    CATEGORY_QUASI = "quasi"
    CATEGORY_SENSITIVE = "sensitive"
    CATEGORY_INDIRECT = "indirect"
    CATEGORY_NON_SENSITIVE = "non_sensitive"
    CATEGORY_ERROR = "error"

    def __init__(
        self,
        unique_threshold: float = 0.9,
        sample_size: int = 1000,
        sensitive_keywords: List[str] = None,
        indirect_keywords: List[str] = None,
        timeout_seconds: int = 300,
    ):
        """
        Initialize DetectProfilingProcessor with auto-resource management.

        Args:
            unique_threshold (float): Threshold for direct identifier detection.
            sample_size (int): Number of samples for pattern matching.
            sensitive_keywords (List[str], optional): Keywords for sensitive attributes.
            indirect_keywords (List[str], optional): Keywords for indirect identifiers.
            timeout_seconds (int): Processing timeout in seconds.
        """
        super().__init__()
        self.unique_threshold = unique_threshold
        self.sample_size = sample_size
        self.timeout_seconds = timeout_seconds
        self.resource_monitor = SystemResourceMonitor()

        self.sensitive_keywords = sensitive_keywords or [
            "disease",
            "income",
            "salary",
            "condition",
            "health",
            "ssn",
            "medical",
            "insurance",
            "credit",
            "bank",
            "tax",
            "password",
            "social security",
            "ethnicity",
            "religion",
            "gender",
            "disability",
        ]
        self.indirect_keywords = indirect_keywords or [
            "job",
            "role",
            "title",
            "department",
            "organization",
            "employer",
            "company",
        ]

    def _process_column(
        self,
        series_data,
        col: str,
        patterns: dict,
        unique_threshold: float,
        sample_size: int,
        sensitive_keywords: List[str],
        indirect_keywords: List[str],
        is_dask: bool = False,
    ) -> Tuple:
        """
        Process a single column to determine its attribute type.
        Works with both pandas Series and Dask Series.

        Parameters:
            series_data (pd.Series or dask.Series): The column data to analyze.
            col (str): The name of the column.
            patterns (dict): Dictionary of regex patterns used to detect sensitive formats (e.g., emails, SSNs).
            unique_threshold (float): Threshold above which a column is considered a direct identifier.
            sample_size (int): Number of values to sample from object columns for pattern matching.
            sensitive_keywords (List[str]): Keywords indicating highly sensitive attributes (e.g., 'name', 'email').
            indirect_keywords (List[str]): Keywords indicating indirect identifiers (e.g., 'zipcode', 'region').
            is_dask (bool): Whether the column is a Dask Series (True) or pandas Series (False).

        Returns:
            Tuple[str, str]: A tuple containing the column name and the inferred category:
                - CATEGORY_SENSITIVE
                - CATEGORY_DIRECT
                - CATEGORY_INDIRECT
                - CATEGORY_QUASI
                - CATEGORY_NON_SENSITIVE
                - CATEGORY_ERROR (in case of failure)
        """
        try:
            col_lower = col.lower()

            # 1. Check sensitive attributes first (no computation needed)
            if any(keyword in col_lower for keyword in sensitive_keywords):
                return col, self.CATEGORY_SENSITIVE

            # 4. Check indirect identifiers (no computation needed)
            if any(keyword in col_lower for keyword in indirect_keywords):
                return col, self.CATEGORY_INDIRECT

            # Handle Dask vs Pandas operations
            if is_dask:
                # For Dask series - compute operations carefully
                try:
                    # Calculate all required values ​​in one go
                    computed_series = series_data.compute()
                    unique_values = computed_series.nunique()
                    row_count = len(computed_series)
                    series_dtype = computed_series.dtype

                    # Sample data for object dtype
                    if pd.api.types.is_object_dtype(series_dtype):
                        # Get non-null values first
                        non_null_series = computed_series.dropna().astype(str)
                        series_length = len(non_null_series)

                        if series_length > 0:
                            # Calculate sample size
                            actual_sample_size = min(sample_size, series_length)
                            # Sample from Dask series
                            if actual_sample_size < series_length:
                                sample_data = non_null_series.sample(
                                    n=actual_sample_size, random_state=42
                                )
                            else:
                                sample_data = non_null_series
                        else:
                            sample_data = pd.Series([], dtype=str)
                    else:
                        sample_data = pd.Series([], dtype=str)
                except Exception as e:
                    logger.warning(f"Dask computation error for column {col}: {str(e)}")
                    # Fallback to basic analysis
                    unique_values = 0
                    row_count = 1
                    sample_data = pd.Series([], dtype=str)
            else:
                # For pandas series
                unique_values = series_data.nunique(dropna=True)
                row_count = len(series_data)
                series_dtype = series_data.dtype

                # Sample data for pattern checking
                if pd.api.types.is_object_dtype(series_dtype):
                    non_null_values = series_data.dropna().astype(str)
                    sample_data = (
                        non_null_values.sample(
                            min(sample_size, len(non_null_values)), random_state=42
                        )
                        if len(non_null_values) > 0
                        else pd.Series([], dtype=str)
                    )
                else:
                    sample_data = pd.Series([], dtype=str)

            # Calculate cardinality ratio
            cardinality_ratio = unique_values / row_count if row_count > 0 else 0

            # 2. Check direct identifiers by cardinality
            if cardinality_ratio > unique_threshold:
                return col, self.CATEGORY_DIRECT

            # 3. Check patterns for object dtype columns
            if pd.api.types.is_object_dtype(series_dtype) and len(sample_data) > 0:
                for pattern in patterns.values():
                    if sample_data.str.contains(pattern, regex=True, na=False).any():
                        return col, self.CATEGORY_DIRECT

            # 5. Check quasi-identifiers
            if 0.1 < cardinality_ratio < unique_threshold:
                return col, self.CATEGORY_QUASI

            # 6. Default to non-sensitive
            return col, self.CATEGORY_NON_SENSITIVE

        except Exception as e:
            logger.error(f"Error processing column {col}: {str(e)}")
            return col, self.CATEGORY_ERROR

    def _process_with_joblib(
        self,
        df: pd.DataFrame,
        settings: dict,
        patterns: dict,
        unique_threshold: float,
        sample_size: int,
        sensitive_keywords: List[str],
        indirect_keywords: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Process dataset using Joblib parallel processing.

        Parameters:
            df (pd.DataFrame): The input pandas DataFrame to be analyzed.
            settings (dict): Configuration for Joblib processing, including:
                - 'strategy' (str): Label indicating the strategy being used (for logging/debugging).
                - 'n_jobs' (int): Number of parallel workers to use.
                - 'backend' (str): Backend for Joblib ('threading' or 'loky').
            patterns (dict): Dictionary of regex patterns used to match known sensitive data formats.
            unique_threshold (float): Threshold ratio above which a column is considered a direct identifier.
            sample_size (int): Number of sample values to evaluate patterns in object columns.
            sensitive_keywords (List[str]): List of keywords that, if found in column names, classify them as sensitive.
            indirect_keywords (List[str]): List of keywords that, if found in column names, classify them as indirect identifiers.

        Returns:
            List[Tuple[str, str]]: A list of tuples with column names and their inferred data sensitivity category.
                                Possible categories: 'sensitive', 'direct', 'indirect',
                                'quasi', 'non-sensitive', or 'error'.
        """
        logger.info(
            f"Processing with joblib: {settings['strategy']}, n_jobs={settings['n_jobs']}"
        )

        try:
            results = Parallel(
                n_jobs=settings["n_jobs"],
                backend=settings["backend"],
                timeout=self.timeout_seconds,
            )(
                delayed(self._process_column)(
                    df[col],
                    col,
                    patterns,
                    unique_threshold,
                    sample_size,
                    sensitive_keywords,
                    indirect_keywords,
                    False,
                )
                for col in df.columns
            )
            return results
        except Exception as e:
            logger.error(f"Joblib processing failed: {str(e)}")
            raise

    def _process_with_dask(
        self,
        df: pd.DataFrame,
        settings: dict,
        patterns: dict,
        unique_threshold: float,
        sample_size: int,
        sensitive_keywords: List[str],
        indirect_keywords: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Process dataset using Dask local processing.

        Parameters:
            df (pd.DataFrame): The input pandas DataFrame to be processed.
            settings (dict): Configuration settings for Dask, including:
                - 'strategy' (str): A label for the processing strategy used (for logging/debugging).
                - 'n_partitions' (int): Number of partitions to split the DataFrame into.
                - 'scheduler' (str, optional): Dask scheduler to use ('threads', 'processes', or 'synchronous').
            patterns (dict): Dictionary of regex patterns to identify specific data types (e.g., emails, phone numbers).
            unique_threshold (float): Ratio threshold for determining direct identifiers.
            sample_size (int): Number of samples to use when testing patterns in text columns.
            sensitive_keywords (List[str]): Keywords to match against column names for detecting sensitive attributes.
            indirect_keywords (List[str]): Keywords to match against column names for detecting indirect identifiers.

        Returns:
            List[Tuple[str, str]]: A list of (column_name, category) tuples, where category is one of:
                'sensitive', 'direct', 'indirect', 'quasi', 'non-sensitive', or 'error'.
        """
        logger.info(
            f"Processing with Dask: {settings['strategy']}, partitions={settings['n_partitions']}"
        )

        try:
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=settings["n_partitions"])

            # Use appropriate scheduler
            scheduler = settings.get("scheduler", "threads")

            # Process columns using simple Dask operations
            results = []

            # Set scheduler context
            if scheduler == "synchronous":
                import dask

                with dask.config.set(scheduler="synchronous"):
                    for col in df.columns:
                        try:
                            computed_result = self._process_column(
                                ddf[col],
                                col,
                                patterns,
                                unique_threshold,
                                sample_size,
                                sensitive_keywords,
                                indirect_keywords,
                                True,
                            )
                            results.append(computed_result)
                        except Exception as e:
                            logger.error(
                                f"Error processing column {col} with Dask: {str(e)}"
                            )
                            results.append((col, self.CATEGORY_ERROR))
            else:
                # Use threads scheduler (default)
                for col in df.columns:
                    try:
                        computed_result = self._process_column(
                            ddf[col],
                            col,
                            patterns,
                            unique_threshold,
                            sample_size,
                            sensitive_keywords,
                            indirect_keywords,
                            True,
                        )
                        results.append(computed_result)
                    except Exception as e:
                        logger.error(
                            f"Error processing column {col} with Dask: {str(e)}"
                        )
                        results.append((col, self.CATEGORY_ERROR))

            return results

        except Exception as e:
            logger.error(f"Dask processing failed: {str(e)}")
            # Fallback to joblib if Dask fails
            logger.info("Falling back to joblib processing")
            fallback_settings = {"n_jobs": 2, "backend": "threading"}
            return self._process_with_joblib(
                df,
                fallback_settings,
                patterns,
                unique_threshold,
                sample_size,
                sensitive_keywords,
                indirect_keywords,
            )
        finally:
            # Force cleanup
            gc.collect()

    def _process_with_threading(
        self,
        df: pd.DataFrame,
        settings: dict,
        patterns: dict,
        unique_threshold: float,
        sample_size: int,
        sensitive_keywords: List[str],
        indirect_keywords: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Process dataset using ThreadPoolExecutor for fine-grained control.

        Parameters:
            df (pd.DataFrame): The input pandas DataFrame to process.
            settings (dict): Configuration settings containing the number of worker threads (expects 'n_jobs' key).
            patterns (dict): Dictionary of regex patterns used to detect specific data types (e.g., email, phone).
            unique_threshold (float): Threshold (0 to 1) above which a column is considered a direct identifier.
            sample_size (int): Number of values to sample from the column for pattern matching.
            sensitive_keywords (List[str]): List of keywords that identify sensitive fields based on column names.
            indirect_keywords (List[str]): List of keywords that identify indirect identifiers based on column names.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains:
                - str: The column name.
                - str: The detected category (e.g., 'sensitive', 'direct', 'indirect', 'quasi', 'non-sensitive', or 'error').
        """
        logger.info(
            f"Processing with ThreadPoolExecutor: n_workers={settings['n_jobs']}"
        )

        results = []
        with ThreadPoolExecutor(max_workers=settings["n_jobs"]) as executor:
            # Submit all column processing tasks
            future_to_col = {
                executor.submit(
                    self._process_column,
                    df[col],
                    col,
                    patterns,
                    unique_threshold,
                    sample_size,
                    sensitive_keywords,
                    indirect_keywords,
                    False,
                ): col
                for col in df.columns
            }

            # Collect results with timeout
            for future in as_completed(future_to_col, timeout=self.timeout_seconds):
                try:
                    result = future.result(timeout=30)  # 30s per column
                    results.append(result)
                except Exception as e:
                    col = future_to_col[future]
                    logger.error(f"Error processing column {col}: {str(e)}")
                    results.append((col, self.CATEGORY_ERROR))

        return results

    def execute(
        self,
        df: pd.DataFrame,
        **kwargs,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Detects attributes using auto-managed Dask/Joblib based on system capacity.

        Parameters:
            df (pd.DataFrame): The dataset to analyze.

        Kwargs:
            unique_threshold (float, optional): Threshold for identifying direct identifiers.
            sensitive_keywords (List[str], optional): Keywords to detect sensitive attributes.
            indirect_keywords (List[str], optional): Keywords to detect indirect identifiers.
            sample_size (int, optional): Number of samples for pattern matching.

        Returns:
            Tuple[List[str], List[str], List[str], List[str], List[str]]: Lists of
            direct identifiers, quasi-identifiers, sensitive attributes, indirect identifiers,
            and non-sensitive attributes.
        """
        start_time = time.time()

        try:
            # Get parameters from kwargs or defaults
            unique_threshold = kwargs.get("unique_threshold", self.unique_threshold)
            sensitive_keywords = kwargs.get(
                "sensitive_keywords", self.sensitive_keywords
            )
            indirect_keywords = kwargs.get("indirect_keywords", self.indirect_keywords)
            sample_size = kwargs.get("sample_size", self.sample_size)

            if len(df) == 0:
                return [], [], [], [], []

            # Analyze dataset characteristics
            data_size = df.memory_usage(deep=True).sum()
            num_columns = len(df.columns)
            num_rows = len(df)

            logger.info(
                f"Dataset: {num_rows:,} rows, {num_columns} columns, "
                f"{data_size / (1024**2):.1f} MB"
            )

            # Get optimal processing settings based on system capacity
            settings = self.resource_monitor.get_optimal_settings(
                data_size, num_columns
            )
            logger.info(f"Selected strategy: {settings['strategy']}")

            # Compile regex patterns once
            patterns = {
                "email": re.compile(Patterns.EMAIL_REGEX),
                "phone": re.compile(Patterns.PHONE_REGEX),
                "credit_card": re.compile(Patterns.CREDIT_CARD),
            }

            # Process based on selected strategy
            try:
                # Log settings
                logger.info("Settings used for processing:")
                for key, value in settings.items():
                    logger.info("  - %s: %s", key, value)

                if settings["strategy"] == "joblib_threading":
                    results = self._process_with_joblib(
                        df,
                        settings,
                        patterns,
                        unique_threshold,
                        sample_size,
                        sensitive_keywords,
                        indirect_keywords,
                    )
                elif settings["strategy"] in [
                    "joblib_processes",
                    "joblib_processes_large",
                ]:
                    results = self._process_with_joblib(
                        df,
                        settings,
                        patterns,
                        unique_threshold,
                        sample_size,
                        sensitive_keywords,
                        indirect_keywords,
                    )
                elif settings["use_dask"]:
                    results = self._process_with_dask(
                        df,
                        settings,
                        patterns,
                        unique_threshold,
                        sample_size,
                        sensitive_keywords,
                        indirect_keywords,
                    )
                else:
                    # Fallback to threading
                    results = self._process_with_threading(
                        df,
                        settings,
                        patterns,
                        unique_threshold,
                        sample_size,
                        sensitive_keywords,
                        indirect_keywords,
                    )
            except Exception as processing_error:
                logger.error(
                    f"Primary processing strategy failed: {str(processing_error)}"
                )
                logger.info("Falling back to simple threading approach")

                # Ultimate fallback - simple threading
                fallback_settings = {
                    "strategy": "fallback_threading",
                    "n_jobs": min(4, num_columns),
                    "backend": "threading",
                }
                results = self._process_with_threading(
                    df,
                    fallback_settings,
                    patterns,
                    unique_threshold,
                    sample_size,
                    sensitive_keywords,
                    indirect_keywords,
                )

            # Initialize category lists
            category_lists = {
                self.CATEGORY_DIRECT: [],
                self.CATEGORY_QUASI: [],
                self.CATEGORY_SENSITIVE: [],
                self.CATEGORY_INDIRECT: [],
                self.CATEGORY_NON_SENSITIVE: [],
            }

            # Organize results
            for col, category in results:
                if category in category_lists:
                    category_lists[category].append(col)
                else:
                    logger.warning(f"Column {col} had processing errors")

            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")

            return (
                category_lists[self.CATEGORY_DIRECT],
                category_lists[self.CATEGORY_QUASI],
                category_lists[self.CATEGORY_SENSITIVE],
                category_lists[self.CATEGORY_INDIRECT],
                category_lists[self.CATEGORY_NON_SENSITIVE],
            )

        except Exception as e:
            logger.error(f"Error in execute method: {str(e)}")
            gc.collect()  # Force cleanup on error
            raise
