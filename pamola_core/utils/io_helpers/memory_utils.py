"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Memory Management Utilities
Description: Tools for estimating, optimizing, and monitoring memory use during I/O operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Dynamic estimation of optimal chunk sizes based on system memory
- DataFrame memory optimization to reduce in-memory footprint
- Process-level memory monitoring and reporting for progress bars
"""


import logging
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("pamola_core.utils.io_helpers.memory_utils")

# Constants for memory size conversions
BYTES_PER_KB = 1024
BYTES_PER_MB = BYTES_PER_KB * 1024
BYTES_PER_GB = BYTES_PER_MB * 1024


def get_system_memory() -> Dict[str, float]:
   """
   Get information about system memory.

   Returns:
   --------
   Dict[str, float]
       Dictionary with system memory information in GB
   """
   try:
       import psutil
       virtual_memory = psutil.virtual_memory()
       return {
           "total_gb": virtual_memory.total / BYTES_PER_GB,
           "available_gb": virtual_memory.available / BYTES_PER_GB,
           "used_gb": virtual_memory.used / BYTES_PER_GB,
           "percent_used": virtual_memory.percent
       }
   except ImportError:
       # Fallback if psutil is not available
       import platform

       if platform.system() == 'Windows':
           # Very rough estimate for Windows
           return {
               "total_gb": 16.0,  # Assume 16GB as default
               "available_gb": 8.0,  # Assume 50% available
               "used_gb": 8.0,
               "percent_used": 50.0
           }
       elif platform.system() == 'Darwin' or platform.system() == 'Linux':
           # Rough estimate based on process size
           import resource
           rusage = resource.getrusage(resource.RUSAGE_SELF)
           process_memory_gb = rusage.ru_maxrss / (1024 * 1024)  # Convert KB to GB

           return {
               "total_gb": 16.0,  # Assume 16GB as default
               "process_gb": process_memory_gb,
               "percent_used": None  # Cannot determine accurately
           }
       else:
           # Default fallback
           return {
               "total_gb": 16.0,  # Assume 16GB as default
               "available_gb": 8.0,  # Assume 50% available
               "used_gb": 8.0,
               "percent_used": 50.0
           }


def get_process_memory_usage() -> Dict[str, float]:
   """
   Get memory usage information for the current process.

   Returns:
   --------
   Dict[str, float]
       Dictionary with process memory usage information
   """
   try:
       import psutil
       process = psutil.Process(os.getpid())
       memory_info = process.memory_info()

       return {
           "rss_bytes": memory_info.rss,  # Resident Set Size
           "rss_mb": memory_info.rss / BYTES_PER_MB,
           "rss_gb": memory_info.rss / BYTES_PER_GB,
           "vms_bytes": memory_info.vms,  # Virtual Memory Size
           "vms_mb": memory_info.vms / BYTES_PER_MB,
           "vms_gb": memory_info.vms / BYTES_PER_GB,
           "percent": process.memory_percent()
       }
   except ImportError:
       # Fallback if psutil is not available
       import resource
       rusage = resource.getrusage(resource.RUSAGE_SELF)

       # rusage.ru_maxrss gives kilobytes on most systems
       memory_kb = rusage.ru_maxrss

       return {
           "rss_bytes": memory_kb * BYTES_PER_KB,
           "rss_mb": memory_kb / 1024,
           "rss_gb": memory_kb / (1024 * 1024),
           "percent": None  # Cannot determine accurately
       }


def estimate_dataframe_size(df: pd.DataFrame, deep: bool = True) -> Dict[str, float]:
   """
   Estimate the memory size of a DataFrame.

   Parameters:
   -----------
   df : pd.DataFrame
       DataFrame to estimate
   deep : bool
       Whether to perform a deep analysis (more accurate but slower)

   Returns:
   --------
   Dict[str, float]
       Dictionary with size information
   """
   try:
       memory_usage = df.memory_usage(deep=deep)
       total_bytes = memory_usage.sum()

       # Calculate individual column sizes
       # Using .to_dict() approach to avoid FutureWarning about Series.__getitem__
       # treating keys as positions being deprecated
       column_sizes = memory_usage.to_dict()

       return {
           "total_bytes": total_bytes,
           "total_kb": total_bytes / BYTES_PER_KB,
           "total_mb": total_bytes / BYTES_PER_MB,
           "total_gb": total_bytes / BYTES_PER_GB,
           "rows": len(df),
           "columns": len(df.columns),
           "bytes_per_row": total_bytes / len(df) if len(df) > 0 else 0,
           "column_sizes_bytes": column_sizes,
           "column_sizes_mb": {col: size / BYTES_PER_MB for col, size in column_sizes.items()}
       }
   except Exception as e:
       logger.warning(f"Error estimating DataFrame size: {e}")

       # Provide rough estimate
       columns = len(df.columns)
       rows = len(df)

       # Rough estimate: 8 bytes per cell for numeric data
       estimated_bytes = rows * columns * 8

       return {
           "total_bytes_estimated": estimated_bytes,
           "total_mb_estimated": estimated_bytes / BYTES_PER_MB,
           "rows": rows,
           "columns": columns,
           "estimation_error": str(e)
       }


def estimate_csv_size(file_path: Union[str, Path],
                     encoding: str = "utf-8",
                     delimiter: str = ",",
                     sample_rows: int = 1000) -> Dict[str, Any]:
   """
   Estimate the memory requirements for loading a CSV file.

   Parameters:
   -----------
   file_path : str or Path
       Path to the CSV file
   encoding : str
       File encoding (default: "utf-8")
   delimiter : str
       Field delimiter (default: ",")
   sample_rows : int
       Number of rows to sample (default: 1000)

   Returns:
   --------
   Dict[str, Any]
       Dictionary with size estimates
   """
   file_path = Path(file_path)

   if not file_path.exists():
       return {
           "error": "File not found",
           "file_path": str(file_path)
       }

   # Get the file size on disk
   file_size_bytes = file_path.stat().st_size

   try:
       # Count the total number of lines
       with open(file_path, 'r', encoding=encoding) as f:
           total_lines = sum(1 for _ in f)

       # Read a sample of rows
       sample_df = pd.read_csv(
           file_path,
           encoding=encoding,
           delimiter=delimiter,
           nrows=sample_rows
       )

       # Calculate memory usage of the sample
       sample_size = estimate_dataframe_size(sample_df)

       # Estimate for full file
       if sample_rows > 0 and total_lines > 1:  # Account for header
           scale_factor = (total_lines - 1) / sample_rows
           estimated_bytes = sample_size["total_bytes"] * scale_factor

           return {
               "file_size_bytes": file_size_bytes,
               "file_size_mb": file_size_bytes / BYTES_PER_MB,
               "total_lines": total_lines,
               "estimated_memory_bytes": estimated_bytes,
               "estimated_memory_mb": estimated_bytes / BYTES_PER_MB,
               "estimated_memory_gb": estimated_bytes / BYTES_PER_GB,
               "expansion_ratio": estimated_bytes / file_size_bytes if file_size_bytes > 0 else None,
               "sample_rows": sample_rows,
               "sample_columns": len(sample_df.columns),
               "bytes_per_row": sample_size["bytes_per_row"]
           }
       else:
           # File is empty or has only a header
           return {
               "file_size_bytes": file_size_bytes,
               "file_size_mb": file_size_bytes / BYTES_PER_MB,
               "total_lines": total_lines,
               "warning": "File appears to be empty or has only a header"
           }

   except UnicodeDecodeError:
       # Try a binary counting method for files with encoding issues
       try:
           with open(file_path, 'rb') as f:
               total_lines = sum(1 for _ in f)

           # Make a rough estimate based on file size
           return {
               "file_size_bytes": file_size_bytes,
               "file_size_mb": file_size_bytes / BYTES_PER_MB,
               "total_lines_estimated": total_lines,
               "estimated_memory_bytes": file_size_bytes * 5,  # Rough estimate: 5x expansion
               "estimated_memory_mb": (file_size_bytes * 5) / BYTES_PER_MB,
               "warning": "Encoding error during estimation - using rough approximation"
           }
       except Exception as e:
           return {
               "file_size_bytes": file_size_bytes,
               "file_size_mb": file_size_bytes / BYTES_PER_MB,
               "error": f"Error estimating file size: {str(e)}"
           }
   except Exception as e:
       return {
           "file_size_bytes": file_size_bytes,
           "file_size_mb": file_size_bytes / BYTES_PER_MB,
           "error": f"Error estimating file size: {str(e)}"
       }


def get_optimal_chunk_size(file_path: Union[str, Path],
                          available_memory_mb: Optional[float] = None,
                          memory_factor: float = 0.5,
                          min_chunk_size: int = 1000,
                          max_chunk_size: int = 1000000,
                          encoding: str = "utf-8",
                          delimiter: str = ",") -> int:
   """
   Calculate the optimal chunk size for reading a file.

   Parameters:
   -----------
   file_path : str or Path
       Path to the file
   available_memory_mb : float, optional
       Available memory in MB, will be auto-detected if None
   memory_factor : float
       Fraction of available memory to use (0.0 to 1.0)
   min_chunk_size : int
       Minimum chunk size in rows
   max_chunk_size : int
       Maximum chunk size in rows
   encoding : str
       File encoding (default: "utf-8")
   delimiter : str
       Field delimiter for CSV files (default: ",")

   Returns:
   --------
   int
       Optimal chunk size in rows
   """
   # Auto-detect available memory if not provided
   if available_memory_mb is None:
       system_memory = get_system_memory()
       available_memory_mb = system_memory["available_gb"] * 1024 * memory_factor

   # Estimate file memory requirements
   size_estimate = estimate_csv_size(file_path, encoding=encoding, delimiter=delimiter)

   # Check if we have valid bytes_per_row information
   if "bytes_per_row" in size_estimate and size_estimate["bytes_per_row"] > 0:
       # Calculate chunks based on bytes per row
       bytes_per_row = size_estimate["bytes_per_row"]
       available_memory_bytes = available_memory_mb * BYTES_PER_MB

       # Calculate chunk size with a safety margin
       optimal_rows = int(available_memory_bytes * memory_factor / bytes_per_row)

       # Ensure chunk size is within bounds
       chunk_size = max(min_chunk_size, min(optimal_rows, max_chunk_size))

       logger.debug(f"Calculated optimal chunk size: {chunk_size} rows "
                    f"(using {memory_factor * 100:.0f}% of {available_memory_mb:.1f} MB)")

       return chunk_size
   else:
       # Fallback to default if estimation failed
       default_chunk_size = 100000
       logger.warning(f"Could not estimate optimal chunk size. Using default: {default_chunk_size} rows")
       return default_chunk_size


def estimate_file_memory(file_path: Union[str, Path]) -> Dict[str, Any]:
   """
   Estimate memory requirements for loading a file based on its format.

   Parameters:
   -----------
   file_path : str or Path
       Path to the file

   Returns:
   --------
   Dict[str, Any]
       Dictionary with memory requirement estimates
   """
   file_path = Path(file_path)

   # Check if file exists
   if not file_path.exists():
       return {
           "error": "File not found",
           "file_path": str(file_path)
       }

   # Get the file extension
   extension = file_path.suffix.lower()

   # Get the file size on disk
   file_size_bytes = file_path.stat().st_size
   file_size_mb = file_size_bytes / BYTES_PER_MB

   # Define memory multiplication factors for different file types
   memory_factors = {
       ".csv": 5.0,  # CSV can expand significantly in memory
       ".tsv": 5.0,  # TSV similar to CSV
       ".txt": 3.0,  # Text files have variable expansion
       ".json": 4.0,  # JSON typically expands quite a bit
       ".parquet": 1.5,  # Parquet is more memory-efficient
       ".xlsx": 7.0,  # Excel files expand significantly
       ".xls": 7.0,  # Excel files expand significantly
       ".pkl": 1.2,  # Pickle files are usually close to in-memory size
       ".pickle": 1.2,  # Pickle files are usually close to in-memory size
   }

   # Get the memory factor for this file type
   memory_factor = memory_factors.get(extension, 3.0)  # Default factor

   # Estimate memory requirements
   estimated_memory_mb = file_size_mb * memory_factor

   # For different file types, use specific estimation methods
   if extension in [".csv", ".tsv", ".txt"]:
       delimiter = "," if extension == ".csv" else "\t"

       # For very large files, avoid full estimation
       if file_size_mb > 1000:  # 1GB
           return {
               "file_size_bytes": file_size_bytes,
               "file_size_mb": file_size_mb,
               "estimated_memory_mb": estimated_memory_mb,
               "estimated_memory_gb": estimated_memory_mb / 1024,
               "memory_factor": memory_factor,
               "file_type": extension.lstrip("."),
               "warning": "Large file - detailed estimation skipped"
           }
       else:
           # Use detailed CSV estimation
           csv_estimate = estimate_csv_size(
               file_path,
               encoding="utf-8" if extension == ".txt" else "utf-8",
               delimiter=delimiter
           )

           # Add file type information
           csv_estimate["file_type"] = extension.lstrip(".")

           return csv_estimate
   else:
       # Basic estimation for other file types
       return {
           "file_size_bytes": file_size_bytes,
           "file_size_mb": file_size_mb,
           "estimated_memory_mb": estimated_memory_mb,
           "estimated_memory_gb": estimated_memory_mb / 1024,
           "memory_factor": memory_factor,
           "file_type": extension.lstrip(".")
       }


def optimize_dataframe_memory(df: pd.DataFrame,
                             categorical_threshold: float = 0.5,
                             inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
   """
   Optimize memory usage of a DataFrame by converting data types.

   Parameters:
   -----------
   df : pd.DataFrame
       DataFrame to optimize
   categorical_threshold : float
       Threshold for converting object columns to categorical (ratio of unique values to total values)
   inplace : bool
       Whether to modify the DataFrame in place

   Returns:
   --------
   Tuple[pd.DataFrame, Dict[str, Any]]
       Optimized DataFrame and dictionary with optimization information
   """
   # Get memory usage before optimization
   before_size = estimate_dataframe_size(df)

   # Create a copy if not in-place
   if not inplace:
       df = df.copy()

   # Track changes for reporting
   conversions = {}

   # Process numeric columns
   for col in df.select_dtypes(include=['int']).columns:
       # Check the range of values
       col_min = df[col].min()
       col_max = df[col].max()

       # Convert to smaller int types if possible
       if col_min >= 0:
           if col_max < 256:
               df[col] = df[col].astype(np.uint8)
               conversions[col] = f"int -> uint8"
           elif col_max < 65536:
               df[col] = df[col].astype(np.uint16)
               conversions[col] = f"int -> uint16"
           elif col_max < 4294967296:
               df[col] = df[col].astype(np.uint32)
               conversions[col] = f"int -> uint32"
       else:
           if col_min > -128 and col_max < 128:
               df[col] = df[col].astype(np.int8)
               conversions[col] = f"int -> int8"
           elif col_min > -32768 and col_max < 32768:
               df[col] = df[col].astype(np.int16)
               conversions[col] = f"int -> int16"
           elif col_min > -2147483648 and col_max < 2147483648:
               df[col] = df[col].astype(np.int32)
               conversions[col] = f"int -> int32"

   # Process float columns
   for col in df.select_dtypes(include=['float']).columns:
       # Convert to float32 if precision loss is acceptable
       if df[col].notna().all():  # Only convert if no NaN values
           df[col] = df[col].astype(np.float32)
           conversions[col] = f"float -> float32"

   # Process object columns - consider converting to categorical
   for col in df.select_dtypes(include=['object']).columns:
       # Count unique values
       n_unique = df[col].nunique()
       n_total = len(df)

       # Convert to categorical if below threshold
       if n_unique / n_total < categorical_threshold:
           df[col] = df[col].astype('category')
           conversions[col] = f"object -> category ({n_unique} categories)"

   # Get memory usage after optimization
   after_size = estimate_dataframe_size(df)

   # Calculate savings
   savings_bytes = before_size["total_bytes"] - after_size["total_bytes"]
   savings_percent = (savings_bytes / before_size["total_bytes"]) * 100 if before_size["total_bytes"] > 0 else 0

   # Prepare optimization info
   optimization_info = {
       "before_bytes": before_size["total_bytes"],
       "before_mb": before_size["total_mb"],
       "after_bytes": after_size["total_bytes"],
       "after_mb": after_size["total_mb"],
       "savings_bytes": savings_bytes,
       "savings_mb": savings_bytes / BYTES_PER_MB,
       "savings_percent": savings_percent,
       "conversions": conversions
   }

   return df, optimization_info


def check_memory_critical(threshold_percent: float = 90.0) -> bool:
   """
   Check if system memory usage is critical.

   Parameters:
   -----------
   threshold_percent : float
       Threshold percentage for critical memory usage

   Returns:
   --------
   bool
       True if memory usage is critical, False otherwise
   """
   try:
       system_memory = get_system_memory()
       return system_memory["percent_used"] > threshold_percent
   except:
       # Fallback - check if we can allocate a small test array
       try:
           # Try to allocate a 100MB array
           test_array = np.ones((100 * 1024 * 1024) // 8, dtype=np.float64)
           del test_array
           return False
       except MemoryError:
           return True


def calculate_safe_chunk_count(file_size_mb: float,
                              available_memory_gb: Optional[float] = None,
                              memory_factor: float = 0.5,
                              min_chunks: int = 2) -> int:
   """
   Calculate a safe number of chunks for processing a large file.

   Parameters:
   -----------
   file_size_mb : float
       File size in MB
   available_memory_gb : float, optional
       Available memory in GB, will be auto-detected if None
   memory_factor : float
       Fraction of available memory to use (0.0 to 1.0)
   min_chunks : int
       Minimum number of chunks

   Returns:
   --------
   int
       Safe number of chunks
   """
   if available_memory_gb is None:
       system_memory = get_system_memory()
       available_memory_gb = system_memory["available_gb"]

   # Convert to consistent units (MB)
   available_memory_mb = available_memory_gb * 1024 * memory_factor

   # Assume worst-case 5x memory expansion from disk to memory
   estimated_memory_mb = file_size_mb * 5

   # Calculate chunks needed
   chunks_needed = max(min_chunks, int(estimated_memory_mb / available_memory_mb) + 1)

   return chunks_needed