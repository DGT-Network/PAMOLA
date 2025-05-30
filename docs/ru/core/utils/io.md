# PAMOLA.CORE I/O Module Documentation

## Overview

The `pamola_core.utils.io` module provides a robust and comprehensive system for file operations, optimized for both small and large datasets. The module handles file reading/writing in various formats, directory management, and data transformation capabilities with a focus on performance, error handling, and user experience.

## Pamola Core Architecture

The I/O system is structured with a clear separation of concerns:

- **Main Module (`io.py`)**: Provides the public API and coordinates operations
- **Helper Modules**: Handle specialized functionality:
  - `directory_utils.py`: Directory and path management
  - `csv_utils.py`: CSV file operations
  - `json_utils.py`: JSON data handling
  - `format_utils.py`: Format detection and validation
  - `dask_utils.py`: Integration with Dask for large datasets
  - `crypto_utils.py`: Encryption/decryption capabilities
  - `image_utils.py`: Image and plot handling

This modular approach ensures clean separation of concerns and makes the system easier to maintain and extend.

## Features

- High-performance CSV reading with chunking and progress tracking
- Support for large datasets via optional Dask integration
- Customizable encoding, delimiters, and text qualifiers
- Memory-efficient processing
- JSON handling with customizable formatting
- Integration with encryption/decryption capabilities
- Support for CSV, JSON, Parquet, Pickle, and image formats
- Directory creation and management
- Timestamped file naming conventions

## Dependencies

### Pamola Core Dependencies
- `pandas`: For DataFrame operations
- `numpy`: For array processing and type handling
- `psutil`: For memory usage tracking

### Internal Dependencies
- `pamola_core.utils.logging`: For centralized logging
- `pamola_core.utils.progress`: For standardized progress tracking
- `pamola_core.utils.crypto`: For data encryption/decryption

### Optional Dependencies
- `dask`: For distributed computing with large datasets
- `pyarrow`: For Parquet file format support
- `openpyxl`: For Excel file support (future)
- `matplotlib`: For image/plot saving

## Directory Management Functions

### ensure_directory

```python
ensure_directory(directory: Union[str, Path]) -> Path
```

Ensures the specified directory exists, creating it if necessary.

**Parameters:**
- `directory`: Path to the directory to ensure exists

**Returns:**
- Pathlib object for the ensured directory

**Example:**

```python
from pamola_core.utils.io import ensure_directory

# Create a directory if it doesn't exist
output_dir = ensure_directory("/path/to/output")
```

### get_timestamped_filename

```python
get_timestamped_filename(base_name: str, extension: str = "csv", include_timestamp: bool = True) -> str
```

Creates a timestamped filename.

**Parameters:**
- `base_name`: Base name for the file
- `extension`: File extension (default: "csv")
- `include_timestamp`: Whether to include a timestamp in the filename (default: True)

**Returns:**
- Timestamped filename

**Example:**

```python
from pamola_core.utils.io import get_timestamped_filename

# Generate a timestamped filename
filename = get_timestamped_filename("results", "json")
# Example output: "results_20250314_153045.json"
```

### get_file_stats

```python
get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]
```

Gets statistics about a file.

**Parameters:**
- `file_path`: Path to the file

**Returns:**
- Dictionary with file statistics (size, creation time, modification time, etc.)

**Example:**

```python
from pamola_core.utils.io import get_file_stats

# Get file statistics
stats = get_file_stats("data.csv")
print(f"File size: {stats['size_mb']:.2f} MB")
print(f"Created: {stats['creation_time']}")
```

### list_directory_contents

```python
list_directory_contents(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]
```

Lists the contents of a directory.

**Parameters:**
- `directory`: Path to the directory
- `pattern`: Glob pattern for filtering files (default: "*")
- `recursive`: Whether to search recursively (default: False)

**Returns:**
- List of paths to the files matching the pattern

**Example:**

```python
from pamola_core.utils.io import list_directory_contents

# List all CSV files in a directory
csv_files = list_directory_contents("/path/to/data", "*.csv")

# List all files in a directory and its subdirectories
all_files = list_directory_contents("/path/to/data", recursive=True)
```

### clear_directory

```python
clear_directory(directory: Union[str, Path], ignore_patterns: Optional[List[str]] = None, confirm: bool = True) -> int
```

Clears all files and subdirectories in the specified directory.

**Parameters:**
- `directory`: Path to the directory to clear
- `ignore_patterns`: List of glob patterns to ignore
- `confirm`: Whether to ask for confirmation before clearing (default: True)

**Returns:**
- Number of items removed

**Example:**

```python
from pamola_core.utils.io import clear_directory

# Clear a directory but keep log files
removed_count = clear_directory("/path/to/temp", ignore_patterns=["*.log"])
print(f"Removed {removed_count} files")
```

## CSV Reading Functions

### read_csv_in_chunks

```python
read_csv_in_chunks(file_path: Union[str, Path], chunk_size: int = 100000, encoding: str = "utf-8", 
                  delimiter: str = ",", quotechar: str = '"', show_progress: bool = True, 
                  use_dask: bool = False, encryption_key: Optional[str] = None) -> Iterator[pd.DataFrame]
```

Reads a very large CSV file in chunks, yielding each chunk as a DataFrame.

**Parameters:**
- `file_path`: Path to the CSV file
- `chunk_size`: Number of rows to read per chunk (default: 100,000)
- `encoding`: File encoding (default: "utf-8")
- `delimiter`: Field delimiter (default: ",")
- `quotechar`: Text qualifier character (default: '"')
- `show_progress`: Whether to display a progress bar (default: True)
- `use_dask`: Whether to use Dask for larger-than-memory datasets (default: False)
- `encryption_key`: Key for decrypting encrypted files

**Yields:**
- DataFrame containing each chunk of data

**Example:**

```python
from pamola_core.utils.io import read_csv_in_chunks

# Process a large CSV file in chunks
for chunk in read_csv_in_chunks("large_file.csv", chunk_size=200000):
    # Process each chunk
    processed_data = process_chunk(chunk)
    # Do something with processed data
```

### read_full_csv

```python
read_full_csv(file_path: Union[str, Path], encoding: str = "utf-8", delimiter: str = ",", 
             quotechar: str = '"', show_progress: bool = True, use_dask: bool = False, 
             encryption_key: Optional[str] = None) -> pd.DataFrame
```

Reads an entire CSV file into a DataFrame. For large files, consider using read_csv_in_chunks instead.

**Parameters:**
- `file_path`: Path to the CSV file
- `encoding`: File encoding (default: "utf-8")
- `delimiter`: Field delimiter (default: ",")
- `quotechar`: Text qualifier character (default: '"')
- `show_progress`: Whether to display a progress bar (default: True)
- `use_dask`: Whether to use Dask for larger-than-memory datasets (default: False)
- `encryption_key`: Key for decrypting encrypted files

**Returns:**
- DataFrame containing the entire file

**Example:**

```python
from pamola_core.utils.io import read_full_csv

# Read a CSV file with custom encoding
df = read_full_csv("data.csv", encoding="utf-8", delimiter=";")

# Read a large file with Dask
big_df = read_full_csv("huge_file.csv", use_dask=True)
```

## CSV Writing Functions

### write_dataframe_to_csv

```python
write_dataframe_to_csv(df: pd.DataFrame, file_path: Union[str, Path], encoding: str = "utf-8", 
                      delimiter: str = ",", quotechar: str = '"', index: bool = False, 
                      show_progress: bool = True, use_dask: bool = False, 
                      encryption_key: Optional[str] = None) -> Path
```

Writes a DataFrame to a CSV file.

**Parameters:**
- `df`: DataFrame to write
- `file_path`: Path to save the CSV file
- `encoding`: File encoding (default: "utf-8")
- `delimiter`: Field delimiter (default: ",")
- `quotechar`: Text qualifier character (default: '"')
- `index`: Whether to write row indices (default: False)
- `show_progress`: Whether to display a progress bar (default: True)
- `use_dask`: Whether to use Dask for larger datasets (default: False)
- `encryption_key`: Key for encrypting the file

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import write_dataframe_to_csv

# Write a DataFrame to CSV
output_path = write_dataframe_to_csv(df, "output.csv", encoding="utf-8")

# Write a large DataFrame using Dask
output_path = write_dataframe_to_csv(big_df, "large_output.csv", use_dask=True)

# Write encrypted data
output_path = write_dataframe_to_csv(sensitive_df, "encrypted.csv",
                                     encryption_key="your-secure-key")
```

### write_chunks_to_csv

```python
write_chunks_to_csv(chunks: Iterator[pd.DataFrame], file_path: Union[str, Path], 
                   encoding: str = "utf-8", delimiter: str = ",", quotechar: str = '"', 
                   index: bool = False, encryption_key: Optional[str] = None) -> Path
```

Writes an iterator of DataFrame chunks to a CSV file.

**Parameters:**
- `chunks`: Iterator of DataFrame chunks to write
- `file_path`: Path to save the CSV file
- `encoding`: File encoding (default: "utf-8")
- `delimiter`: Field delimiter (default: ",")
- `quotechar`: Text qualifier character (default: '"')
- `index`: Whether to write row indices (default: False)
- `encryption_key`: Key for encrypting the file

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import write_chunks_to_csv


# Generate chunks of data
def generate_chunks():
    for i in range(5):
        yield pd.DataFrame({'A': range(i * 100, (i + 1) * 100), 'B': range(i * 100, (i + 1) * 100)})


# Write chunks to a single CSV file
output_path = write_chunks_to_csv(generate_chunks(), "chunked_output.csv")
```

## JSON Reading and Writing Functions

### read_json

```python
read_json(file_path: Union[str, Path], encoding: str = "utf-8", 
         encryption_key: Optional[str] = None) -> Dict[str, Any]
```

Reads a JSON file into a dictionary.

**Parameters:**
- `file_path`: Path to the JSON file
- `encoding`: File encoding (default: "utf-8")
- `encryption_key`: Key for decrypting the file

**Returns:**
- Dictionary containing the JSON data

**Example:**

```python
from pamola_core.utils.io import read_json

# Read a JSON configuration file
config = read_json("config.json")

# Read an encrypted JSON file
sensitive_data = read_json("secure_data.json", encryption_key="your-secure-key")
```

### write_json

```python
write_json(data: Union[Dict[str, Any], List[Any]], file_path: Union[str, Path], 
          encoding: str = "utf-8", indent: int = 2, ensure_ascii: bool = False, 
          convert_numpy: bool = True, encryption_key: Optional[str] = None) -> Path
```

Writes a dictionary to a JSON file.

**Parameters:**
- `data`: Dictionary or list to write
- `file_path`: Path to save the JSON file
- `encoding`: File encoding (default: "utf-8")
- `indent`: Number of spaces for indentation (default: 2)
- `ensure_ascii`: Whether to escape non-ASCII characters (default: False)
- `convert_numpy`: Whether to convert NumPy types to standard Python types (default: True)
- `encryption_key`: Key for encrypting the file

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import write_json

# Write analysis results to JSON
results = {"mean": 42.5, "median": 41.0, "std_dev": 5.2}
output_path = write_json(results, "analysis_results.json")

# Write encrypted sensitive data
sensitive_data = {"ssn": "123-45-6789", "dob": "1980-01-01"}
output_path = write_json(sensitive_data, "secure_data.json",
                         encryption_key="your-secure-key")
```

### append_to_json_array

```python
append_to_json_array(item: Dict[str, Any], file_path: Union[str, Path], 
                    encoding: str = "utf-8", indent: int = 2, convert_numpy: bool = True, 
                    create_if_missing: bool = True, encryption_key: Optional[str] = None) -> Path
```

Appends an item to a JSON array file. If the file doesn't exist or doesn't contain a valid JSON array, a new array is created.

**Parameters:**
- `item`: Item to append to the array
- `file_path`: Path to the JSON file
- `encoding`: File encoding (default: "utf-8")
- `indent`: Number of spaces for indentation (default: 2)
- `convert_numpy`: Whether to convert NumPy types to standard Python types (default: True)
- `create_if_missing`: Whether to create the file if it doesn't exist (default: True)
- `encryption_key`: Key for encrypting/decrypting the file

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import append_to_json_array

# Append a new log entry to a log file
log_entry = {
    "timestamp": "2025-03-14T15:30:45",
    "level": "INFO",
    "message": "Operation completed successfully"
}
append_to_json_array(log_entry, "app_log.json")
```

### merge_json_objects

```python
merge_json_objects(item: Dict[str, Any], file_path: Union[str, Path], 
                  encoding: str = "utf-8", indent: int = 2, convert_numpy: bool = True, 
                  create_if_missing: bool = True, overwrite_existing: bool = True, 
                  recursive_merge: bool = False, encryption_key: Optional[str] = None) -> Path
```

Merges a dictionary with an existing JSON object file. If the file doesn't exist, a new JSON object is created.

**Parameters:**
- `item`: Dictionary to merge with existing JSON object
- `file_path`: Path to the JSON file
- `encoding`: File encoding (default: "utf-8")
- `indent`: Number of spaces for indentation (default: 2)
- `convert_numpy`: Whether to convert NumPy types to standard Python types (default: True)
- `create_if_missing`: Whether to create the file if it doesn't exist (default: True)
- `overwrite_existing`: Whether to overwrite existing keys (default: True)
- `recursive_merge`: Whether to recursively merge nested dictionaries (default: False)
- `encryption_key`: Key for encrypting/decrypting the file

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import merge_json_objects

# Update configuration settings
new_settings = {"max_connections": 100, "timeout": 30}
merge_json_objects(new_settings, "settings.json")

# Recursively merge nested configuration
advanced_settings = {
    "database": {
        "pool_size": 20,
        "timeout": 5
    }
}
merge_json_objects(advanced_settings, "settings.json", recursive_merge=True)
```

## Parquet Reading and Writing Functions

### read_parquet

```python
read_parquet(file_path: Union[str, Path], **kwargs) -> pd.DataFrame
```

Reads a Parquet file into a DataFrame.

**Parameters:**
- `file_path`: Path to the Parquet file
- `**kwargs`: Additional arguments to pass to pandas.read_parquet

**Returns:**
- DataFrame containing the file data

**Example:**

```python
from pamola_core.utils.io import read_parquet

# Read a Parquet file
df = read_parquet("data.parquet")

# Read specific columns from a Parquet file
df = read_parquet("data.parquet", columns=["name", "age", "salary"])
```

### write_parquet

```python
write_parquet(df: pd.DataFrame, file_path: Union[str, Path], 
             compression: str = "snappy", index: bool = False, **kwargs) -> Path
```

Writes a DataFrame to a Parquet file.

**Parameters:**
- `df`: DataFrame to write
- `file_path`: Path to save the Parquet file
- `compression`: Compression algorithm (default: "snappy")
- `index`: Whether to include the index (default: False)
- `**kwargs`: Additional arguments to pass to pd.DataFrame.to_parquet

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import write_parquet

# Write a DataFrame to a Parquet file with Snappy compression
output_path = write_parquet(df, "data.parquet")

# Use a different compression algorithm
output_path = write_parquet(df, "data_zstd.parquet", compression="zstd")
```

## Image/Plot Utilities

### save_plot

```python
save_plot(plot_fig, file_path: Union[str, Path], dpi: int = 300, **kwargs) -> Path
```

Saves a matplotlib figure to a file.

**Parameters:**
- `plot_fig`: The matplotlib figure to save
- `file_path`: Path to save the image
- `dpi`: Dots per inch for raster formats (default: 300)
- `**kwargs`: Additional arguments to pass to fig.savefig

**Returns:**
- Path to the saved file

**Example:**

```python
import matplotlib.pyplot as plt
from pamola_core.utils.io import save_plot

# Create a matplotlib figure
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
ax.set_title("Sample Plot")

# Save the figure
output_path = save_plot(fig, "sample_plot.png", dpi=150)
```

## Data Transformation Functions

### save_dataframe

```python
save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], 
              format: str = "csv", **kwargs) -> Path
```

Saves a DataFrame to a file in the specified format.

**Parameters:**
- `df`: DataFrame to save
- `file_path`: Path to save the file
- `format`: File format: "csv", "json", "parquet", "pickle" (default: "csv")
- `**kwargs`: Additional arguments to pass to the underlying save function

**Returns:**
- Path to the saved file

**Example:**

```python
from pamola_core.utils.io import save_dataframe

# Save in different formats
csv_path = save_dataframe(df, "data.csv")
json_path = save_dataframe(df, "data.json", format="json", orient="records")
parquet_path = save_dataframe(df, "data.parquet", format="parquet", compression="zstd")
pickle_path = save_dataframe(df, "data.pkl", format="pickle")
```

### read_dataframe

```python
read_dataframe(file_path: Union[str, Path], format: Optional[str] = None, **kwargs) -> pd.DataFrame
```

Reads a file into a DataFrame based on the file extension or specified format.

**Parameters:**
- `file_path`: Path to the file
- `format`: File format: "csv", "json", "parquet", "pickle". If None, inferred from file extension.
- `**kwargs`: Additional arguments to pass to the underlying read function

**Returns:**
- DataFrame containing the file data

**Example:**

```python
from pamola_core.utils.io import read_dataframe

# Read different formats, automatically detecting the format from extension
csv_df = read_dataframe("data.csv")
json_df = read_dataframe("data.json")
parquet_df = read_dataframe("data.parquet")
pickle_df = read_dataframe("data.pkl")

# Explicitly specify format with custom parameters
df = read_dataframe("custom.data", format="csv", delimiter="|", encoding="latin1")
```

## Advanced Usage Patterns

### Processing Large Files with Chunks and Dask

For extremely large files that may not fit in memory, use chunks or Dask:

```python
from pamola_core.utils.io import read_csv_in_chunks, write_dataframe_to_csv

# Process a very large file in chunks
processed_chunks = []
for chunk in read_csv_in_chunks("huge_file.csv", chunk_size=100000):
    # Apply transformations
    processed_chunk = transform_data(chunk)
    processed_chunks.append(processed_chunk)

# Write processed chunks back to disk
for i, chunk in enumerate(processed_chunks):
    output_path = f"processed_chunk_{i}.csv"
    write_dataframe_to_csv(chunk, output_path)

# Alternatively, use Dask for distributed processing
from pamola_core.utils.io import read_full_csv

df = read_full_csv("huge_file.csv", use_dask=True)
result = df.groupby('category').mean().compute()  # Dask computation
```

### Working with Encrypted Data

For sensitive data that requires encryption:

```python
from pamola_core.utils.io import write_dataframe_to_csv, read_full_csv, write_json, read_json

# Generate or obtain a secure encryption key
from pamola_core.utils.crypto import generate_key

encryption_key = generate_key()  # Or load from secure storage

# Save the encryption key securely (separate from the data)
with open("secure_key.txt", "w") as f:
    f.write(encryption_key)

# Write encrypted data
write_dataframe_to_csv(sensitive_df, "encrypted_data.csv", encryption_key=encryption_key)
write_json(sensitive_config, "encrypted_config.json", encryption_key=encryption_key)

# Later, read the encrypted data
with open("secure_key.txt", "r") as f:
    encryption_key = f.read().strip()

df = read_full_csv("encrypted_data.csv", encryption_key=encryption_key)
config = read_json("encrypted_config.json", encryption_key=encryption_key)
```

### Directory Management and Timestamped Files

For organizing output with timestamped files:

```python
from pamola_core.utils.io import ensure_directory, get_timestamped_filename, write_dataframe_to_csv

# Create output directories
base_dir = ensure_directory("/path/to/reports")
daily_dir = ensure_directory(base_dir / "daily")
archive_dir = ensure_directory(base_dir / "archive")

# Generate a timestamped filename
filename = get_timestamped_filename("sales_report", "csv")
# Example: "sales_report_20250314_153045.csv"

# Write data with the timestamped filename
output_path = daily_dir / filename
write_dataframe_to_csv(sales_df, output_path)
```

### Working with Multiple File Formats

For applications that need to work with different file formats:

```python
from pamola_core.utils.io import save_dataframe, read_dataframe, format_utils

# Detect format from file extension
file_path = "data.parquet"
format_name = format_utils.detect_format_from_extension(file_path)
print(f"Detected format: {format_name}")

# Check if a format is supported
if format_utils.is_format_supported(format_name):
    df = read_dataframe(file_path)
    # Process the data...

    # Save in multiple formats for different use cases
    save_dataframe(df, "data_for_excel.csv")  # For Excel users
    save_dataframe(df, "data_for_analysis.parquet", format="parquet")  # For analysts
    save_dataframe(df, "data_for_web.json", format="json", orient="records")  # For web API
```

## Error Handling Best Practices

The I/O module provides detailed error messages and logs but generally propagates exceptions to the caller. Here's how to handle common errors:

```python
from pamola_core.utils.io import read_full_csv, write_json
import logging

logger = logging.getLogger(__name__)

# Handle file not found
try:
    df = read_full_csv("data.csv")
except FileNotFoundError:
    logger.warning("Data file not found, using empty DataFrame")
    df = pd.DataFrame()

# Handle JSON parsing errors
try:
    config = read_json("config.json")
except json.JSONDecodeError:
    logger.error("Invalid JSON format in config file")
    config = {"use_defaults": True}
except FileNotFoundError:
    logger.info("Config file not found, creating with defaults")
    config = {"use_defaults": True}
    write_json(config, "config.json")

# Handle permission errors
try:
    write_dataframe_to_csv(df, "/protected/path/data.csv")
except PermissionError:
    logger.error("Cannot write to protected directory")
    alternative_path = "fallback_data.csv"
    logger.info(f"Writing to alternative location: {alternative_path}")
    write_dataframe_to_csv(df, alternative_path)
```

## Performance Optimization Tips

1. **Choose the right format for your data:**
   - CSV: Good for compatibility, slower for large datasets
   - Parquet: Excellent for large datasets, supports columnar operations
   - Pickle: Fast but Python-specific, good for temporary storage

2. **Use chunking for large files:**
   ```python
   # Instead of
   df = read_full_csv("huge_file.csv")  # May cause memory issues
   
   # Use
   for chunk in read_csv_in_chunks("huge_file.csv", chunk_size=100000):
       process_chunk(chunk)  # Process one piece at a time
   ```

3. **Enable Dask for very large datasets:**
   ```python
   df = read_full_csv("enormous_file.csv", use_dask=True)
   ```

4. **Select appropriate encodings:**
   - UTF-8 is more space-efficient but may not handle all characters
   - UTF-16 handles international characters well but uses more space

5. **Disable progress bars for automated processes:**
   ```python
   df = read_full_csv("data.csv", show_progress=False)  # Better for automated jobs
   ```

6. **Use appropriate compression:**
   ```python
   # Snappy: Fast compression/decompression, moderate size
   write_parquet(df, "data.parquet", compression="snappy")
   
   # ZSTD: Better compression, slightly slower
   write_parquet(df, "data.parquet", compression="zstd")
   ```

## Security Considerations

1. **Encryption key management:**
   - Never hardcode encryption keys in your code
   - Store keys separately from encrypted data
   - Consider using a key management service for production

2. **File permissions:**
   - Ensure appropriate file system permissions on sensitive data
   - The I/O module does not change file permissions by default

3. **Temporary files:**
   - Encryption operations may create temporary files that are automatically deleted
   - In case of errors, check for orphaned temporary files

4. **Sensitive data handling:**
   - Be selective about what data to encrypt (encryption adds overhead)
   - Consider storing sensitive fields in separate files from non-sensitive data

## Future Extensions

The I/O module is designed for extensibility. Upcoming features include:

1. **Excel support**: Direct reading/writing Excel files (in development)
2. **Additional formats**: Support for specialized formats like HDF5, Arrow, or Feather
3. **Database integration**: Direct connections to databases like SQLite, PostgreSQL, etc.
4. **Cloud storage**: Integration with S3, Azure Blob Storage, or other cloud storage
5. **Streaming capabilities**: Enhanced streaming for real-time data processing
6. **Compression options**: More fine-grained control over compression algorithms

## Appendix: Helper Module Details

### directory_utils.py

Provides utilities for managing directories, paths, and file timestamps:
- `ensure_directory`: Creates directories if they don't exist
- `get_timestamped_filename`: Creates filenames with timestamps
- `get_file_stats`: Gets statistics about files
- `list_directory_contents`: Lists directory contents with filtering
- `clear_directory`: Removes files from a directory with options
- `make_unique_path`: Creates unique paths by appending counters

### csv_utils.py

Provides utilities for working with CSV files:
- `estimate_csv_size`: Estimates CSV file size based on DataFrame content
- `prepare_csv_reader_options`: Prepares options for CSV reading
- `prepare_csv_writer_options`: Prepares options for CSV writing
- `count_csv_lines`: Counts lines in a CSV file
- `validate_csv_structure`: Validates CSV structure against requirements
- `monitor_csv_operation`: Creates progress bars for CSV operations
- `report_memory_usage`: Reports memory usage during operations

### json_utils.py

Provides utilities for working with JSON data:
- `convert_numpy_types`: Converts NumPy types to standard Python types
- `validate_json_structure`: Validates JSON against expected schema
- `merge_json_objects_in_memory`: Merges JSON objects with options
- `prettify_json`: Converts dictionaries to prettified JSON strings
- `detect_array_or_object`: Detects if JSON string is array or object
- `extract_json_subset`: Extracts subset of keys from JSON object
- `prepare_json_writer_options`: Prepares options for JSON writing

### format_utils.py

Provides utilities for handling different file formats:
- `detect_format_from_extension`: Detects format from file extension
- `is_format_supported`: Checks if format is supported
- `get_format_extension`: Gets standard extension for format
- `check_pyarrow_available`: Checks for pyarrow availability
- `check_openpyxl_available`: Checks for openpyxl availability
- `check_matplotlib_available`: Checks for matplotlib availability
- `convert_dataframe_to_json`: Converts DataFrame to JSON
- `get_pandas_dtypes_info`: Gets detailed DataFrame type information
- `get_dataframe_stats`: Gets statistics about DataFrame

### dask_utils.py

Provides integration with Dask for large-scale data processing:
- `is_dask_available`: Checks if Dask is available
- `read_csv_in_chunks`: Reads CSV with Dask and yields chunks
- `read_full_csv`: Reads full CSV with Dask
- `write_dataframe_to_csv`: Writes DataFrame with Dask
- `compute_dask_stats`: Computes statistics for Dask DataFrame

### crypto_utils.py

Provides utilities for encrypting and decrypting data:
- `encrypt_file_content`: Encrypts data with error handling
- `decrypt_file_content`: Decrypts data with error handling
- `decrypt_file`: Reads and decrypts a file
- `encrypt_file`: Encrypts and saves a file
- `encrypt_content_to_file`: Encrypts content and saves to file
- `is_encrypted_data`: Checks if data is in encrypted format
- `get_encryption_metadata`: Extracts metadata from encrypted data
- `safe_remove_temp_file`: Safely removes temporary files after operations

### image_utils.py

Provides utilities for working with images and plots:
- `save_plot`: Saves matplotlib figure to a file
- `get_figure_format`: Determines figure format from file extension
- `prepare_figure_options`: Prepares options for figure creation
- `get_optimal_figure_size`: Calculates optimal figure size based on content

## Conclusion

The PAMOLA.CORE I/O module provides a robust, flexible, and efficient system for file operations with a clean separation of concerns. Its modular design makes it easy to maintain and extend, while its comprehensive error handling and performance optimizations make it suitable for both small and large-scale data processing tasks.

By providing a standardized interface for file operations, the module helps ensure consistency across the PAMOLA.CORE project and reduces the risk of errors and inconsistencies in data handling. The integration with encryption, progress tracking, and memory monitoring further enhances its utility for production applications.

When using this module, remember that it's designed to be used explicitly - all parameters should be passed directly to functions rather than relying on global configurations or defaults. This approach enhances clarity and maintainability, even if it sometimes requires more verbose function calls.