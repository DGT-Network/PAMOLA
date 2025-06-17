# Final AI-Ready SRS for PAMOLA.CORE IO Module Refactoring

## System Context
- **io.py**: Facade module providing public API with minimal logic inside
- **io_helpers/**: Package of supporting modules containing specialized logic
- **crypto_utils.py**: Already implemented encryption subsystem - do not modify, only use as provided
- **data_ops layer**: Higher-level client that calls io.py but should not handle file details

## Required Refactoring Targets

| **Area**                          | **Action Required**                                                                                                                                         |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Encryption integration            | Update all read/write functions to support `encryption_key` and offload to `io_helpers.crypto_utils` (`encrypt_file`, `decrypt_file`). Remove legacy crypto.   |
| Selective loading                 | All read functions must accept `columns`, `nrows`, `skiprows` and apply them efficiently, using pandas/dask where possible.                                    |
| Multi-file datasets               | Implement `read_multi_csv()` and similar helpers that **stack files vertically**. No horizontal joins or ID merges in io.py. Batch handling goes to helpers.  |
| Error handling                    | Use `error_utils.create_error_info()` consistently across all functions for structured error outputs `{error_type, message, resolution}`.                     |
| Memory estimation                 | Add `estimate_file_memory()` and related functions to expose size/memory hints; no dynamic memory balancing yet.                                             |
| Progress tracking                 | Integrate `pamola_core.utils.progress` into all long-running functions; helpers should handle finer-grained progress if needed.                                     |
| Directory & temp management       | Use `directory_utils` for safe path handling, temp files, and cleanup (especially for decrypted intermediates).                                               |
| Code modularity                   | Any function exceeding ~100 lines or complex branching must move detailed steps to helpers; `io.py` should remain a clean orchestration layer.               |
| Public API stability              | Keep all current function names and signatures. Add only optional parameters. Ensure no existing behavior breaks.                                             |

## Implementation Scope

### Phase 1 (Required)

1. **Crypto Integration**
   - Add `encryption_key` parameter to all read/write functions
   - Use existing `crypto_utils.{encrypt_file, decrypt_file, encrypt_data, decrypt_data}` functions
   - DO NOT modify crypto_utils.py, only use it as provided
   - Handle secure temp files for encryption/decryption

2. **Data Selection**
   - Add `columns` parameter to all read functions
   - Add `nrows`, `skiprows` parameters where applicable
   - Implement efficient column filtering after loading

3. **Multi-File Support**
   - Create `read_multi_csv` function for vertical concatenation
   - Implement memory-aware batch processing
   - Support progress tracking across files

4. **Error Handling**
   - Create `error_utils.py` with standardized error objects
   - Implement consistent error handling decorator
   - Include detailed context in error messages

5. **Memory Management**
   - Add memory estimation functions
   - Implement memory-aware chunking
   - Provide memory usage reporting

### Phase 2 (Excluded from current refactoring)

- Streaming I/O for large files
- Advanced schema validation and management
- Parallel/async multi-file merges
- Cloud storage integration
- Key rotation support

## Helper Modules

1. **Core Updates**
   - `csv_utils.py`: Add encryption support, optimize chunking
   - `format_utils.py`: Enhance format detection including encrypted files
   - `directory_utils.py`: Add secure temp file management

2. **New Modules**
   - `error_utils.py`: Standardized error handling
   - `memory_utils.py`: Memory estimation and management
   - `multi_file_utils.py`: Multi-file processing utilities

## Code Implementation Constraints

- **Language**: Python 3.9+
- **Docstrings**: English, following Google or NumPy style
- **Type Annotations**: Full, explicit on all functions 
- **Function Size**: Any function exceeding ~100 lines or with complex branching should move details to helpers
- **Direct References**: Explicitly reference helpers, e.g., `crypto_utils.decrypt_file()`, not indirect calls
- **No Dead Code**: Remove unused legacy branches (especially old crypto paths)
- **Testability**: Design helpers for isolated unit testing; avoid tight coupling

## Required API Extensions

### File Reading Functions

```python
def read_full_csv(..., encryption_key: Optional[str] = None, columns: Optional[List[str]] = None, nrows: Optional[int] = None, skiprows: Optional[Union[int, List[int]]] = None)
def read_parquet(..., encryption_key: Optional[str] = None, columns: Optional[List[str]] = None, nrows: Optional[int] = None)
def read_excel(..., encryption_key: Optional[str] = None, columns: Optional[List[str]] = None, nrows: Optional[int] = None)
def read_json(..., encryption_key: Optional[str] = None)
def read_text(..., encryption_key: Optional[str] = None, columns: Optional[List[str]] = None, nrows: Optional[int] = None)
```

### File Writing Functions

```python
def write_dataframe_to_csv(..., encryption_key: Optional[str] = None)
def write_parquet(..., encryption_key: Optional[str] = None)
def write_json(..., encryption_key: Optional[str] = None)
```

### New Functions

```python
def read_multi_csv(file_paths: List[Union[str, Path]], **kwargs) -> pd.DataFrame
def estimate_file_memory(file_path: Union[str, Path]) -> Dict[str, Any]
```

## Implementation Priority

1. Add encryption support to main io.py functions
2. Add data selection parameters
3. Create error_utils.py and standardize error handling
4. Implement multi-file support
5. Add memory management utilities

## Delivery Package

- Updated **io.py** with all public functions refactored
- Updated/added helpers in **io_helpers/** as needed (csv_utils, error_utils, memory_utils, multi_file_utils)
- Updated docstrings and type hints throughout
- Removed legacy crypto integration
- Ready-to-run unit tests for all new and refactored helpers