Based on my analysis of the provided files and requirements, I'll outline the key changes needed for the PAMOLA.CORE IO module refactoring project. The goal is to enhance the existing IO system by integrating the new cryptography subsystem, adding selective loading, multi-file dataset support, standardized error handling, and improving memory management.

# Implementation Order

## 1. Create New Helper Modules

First, we need to create these new helper modules:

### A. error_utils.py
- Create standardized error structures
- Implement error creation helpers
- Add error handling decorators

### B. memory_utils.py 
- Implement file size estimation
- Add memory usage tracking 
- Create functions for chunking decisions

### C. multi_file_utils.py
- Implement vertical file stacking
- Add batch processing support
- Create memory-aware multi-file reading

## 2. Update Existing Helper Modules

Next, we need to update these existing modules:

### A. csv_utils.py
- Add column filtering support
- Add row selection parameters
- Enhance memory usage reporting

### B. format_utils.py
- Improve format detection for encrypted files
- Add format validation functions
- Enhance file metadata handling

### C. directory_utils.py
- Add secure temporary file management
- Improve directory cleanup for crypto operations
- Add path normalization utilities

## 3. Update Core IO.py Module

Finally, update the main IO module:

### A. Update Function Signatures
- Add `encryption_key` parameter to all read/write functions
- Add `columns`, `nrows`, `skiprows` parameters to read functions
- Ensure backward compatibility

### B. Add New Functions
- Implement `read_multi_csv()` and similar helpers
- Add `estimate_file_memory()`
- Create `get_file_metadata()`

### C. Refactor Existing Functions
- Integrate with crypto_utils
- Update error handling using error_utils
- Enhance progress tracking and reporting

# Detailed Implementation Plan

## 1. error_utils.py

```python
def create_error_info(error_type: str, message: str, resolution: str = None, details: dict = None) -> dict:
    """Create standardized error information dictionary."""
    
def handle_io_errors(func):
    """Decorator to standardize error handling for IO operations."""
    
def is_recoverable_error(error_info: dict) -> bool:
    """Check if an error is potentially recoverable."""
```

## 2. memory_utils.py

```python
def estimate_file_memory(file_path: Union[str, Path], sample_rows: int = 1000) -> dict:
    """Estimate memory requirements for loading a file."""
    
def get_optimal_chunk_size(file_path: Union[str, Path], available_memory_mb: int) -> int:
    """Calculate optimal chunk size for reading a file."""
    
def report_memory_usage(df: pd.DataFrame, detailed: bool = False) -> dict:
    """Report memory usage of a DataFrame."""
```

## 3. multi_file_utils.py

```python
def stack_files_vertically(file_paths: List[Union[str, Path]], **kwargs) -> pd.DataFrame:
    """Stack multiple files vertically (row-wise)."""
    
def process_files_in_batches(file_paths: List[Union[str, Path]], batch_size: int, **kwargs) -> pd.DataFrame:
    """Process multiple files in memory-efficient batches."""
    
def get_common_columns(file_paths: List[Union[str, Path]]) -> List[str]:
    """Identify columns common to all files in a dataset."""
```

## 4. Update io.py

Add new parameters to existing functions:
```python
def read_full_csv(file_path: Union[str, Path], 
                 encoding: str = "utf-8",
                 delimiter: str = ",",
                 quotechar: str = '"',
                 show_progress: bool = True,
                 use_dask: bool = False,
                 encryption_key: Optional[str] = None,
                 columns: Optional[List[str]] = None,
                 nrows: Optional[int] = None,
                 skiprows: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
    """Reads an entire CSV file into a DataFrame."""
```

Add new functions:
```python
def read_multi_csv(file_paths: List[Union[str, Path]], **kwargs) -> pd.DataFrame:
    """Reads multiple CSV files and stacks them vertically."""
    
def estimate_file_memory(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Estimates memory requirements for loading a file."""
```

This implementation plan follows the requirements specified in the SRS and maintains backward compatibility while adding the new requested features. I'll now proceed with creating the specific code for these modules based on your guidance.