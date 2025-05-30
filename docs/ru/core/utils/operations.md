# PAMOLA.CORE Operations Structure and Organization Guide

## 1. Purpose and General Concept

Operations in PAMOLA.CORE are atomic computational units that perform specific data processing tasks, such as:
- Data profiling (analysis of numeric, text, and categorical fields)
- Data anonymization
- Attack simulation on data
- Calculation of data quality and completeness metrics

Each operation takes input data (DataFrames or file references), processes it, and creates artifacts (JSON reports, visualizations, dictionaries, modified data), saving them in a standardized directory structure.

Operations are designed to be executed within tasks, not independently. Tasks orchestrate multiple operations, manage their dependencies, and aggregate their results.

## 2. Package and Module Organization

```
pamola_core/
├── utils/
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── op_base.py            # Base operation classes
│   │   ├── op_data_source.py     # Data source handling
│   │   ├── op_result.py          # Operation result classes
│   │   ├── op_registry.py        # Operation registration
│   │   └── op_config.py          # Operation configuration
│   ├── io.py                     # Data input/output
│   ├── visualization.py          # Data visualization
│   ├── progress.py               # Progress tracking
│   ├── logging.py                # Logging
│   ├── crypto.py                 # Data encryption
│   └── resource_monitor.py       # Resource usage monitoring
├── profiling/
│   ├── commons/                  # Common components for profiling
│   │   ├── __init__.py
│   │   ├── numeric_utils.py      # Numeric analysis utilities
│   │   ├── text_utils.py         # Text analysis utilities
│   │   └── ... 
│   ├── numeric.py                # Numeric field analysis operation
│   ├── categorical.py            # Categorical field analysis operation
│   └── ...
├── anonymization/                # Anonymization operations
│   ├── commons/
│   │   └── ...
│   ├── k_anonymity.py
│   └── ...
└── security/                     # Attack simulation operations
    ├── commons/
    │   └── ...
    └── ...
```

## 3. Key Classes and Interfaces

### 3.1. Base Operation Classes (`op_base.py`)

#### `BaseOperation`

Abstract base class for all operations, defining a unified interface:

```python
class BaseOperation(ABC):
    """Base class for all operations."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.version = "1.0.0"  # Operation implementation version
    
    @abstractmethod
    def execute(self, 
                data_source: DataSource, 
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """Execute the operation."""
        pass
    
    def run(self, 
            data_source: DataSource, 
            task_dir: Path,
            reporter: Any,
            track_progress: bool = True,
            resource_monitoring: bool = False,
            **kwargs) -> OperationResult:
        """Run with timing and error handling."""
        pass
        
    def validate_inputs(self, data_source: DataSource, **kwargs) -> bool:
        """Validate operation inputs before execution."""
        pass
        
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return estimated resource requirements for this operation."""
        pass
```

#### `FieldOperation`

Specialized base class for operations on specific fields:

```python
class FieldOperation(BaseOperation):
    """Base class for operations that process specific fields."""
    
    def __init__(self, field_name: str, description: str = ""):
        super().__init__(f"{field_name} analysis", description or f"Analysis of {field_name} field")
        self.field_name = field_name
        
    def validate_field(self, df: pd.DataFrame) -> bool:
        """Validate that the field exists and is of the expected type."""
        pass
```

#### `DataFrameOperation`

Specialized base class for operations on entire DataFrames:

```python
class DataFrameOperation(BaseOperation):
    """Base class for operations that process entire DataFrames."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame meets the operation's requirements."""
        pass
```

### 3.2. Data Sources (`op_data_source.py`)

#### `DataSource`

Class encapsulating access to data from different sources:

```python
class DataSource:
    """Represents a data source for operations."""
    
    def __init__(self,
                 dataframes: Dict[str, pd.DataFrame] = None,
                 file_paths: Dict[str, Path] = None,
                 encryption_keys: Dict[str, str] = None):
        self.dataframes = dataframes or {}
        self.file_paths = {k: Path(v) if isinstance(v, str) else v 
                           for k, v in (file_paths or {}).items()}
        self.encryption_keys = encryption_keys or {}
    
    def get_dataframe(self, name: str, load_if_path: bool = True) -> Optional[pd.DataFrame]:
        """Get a DataFrame by name."""
        pass
    
    def get_file_path(self, name: str) -> Optional[Path]:
        """Get a file path by name."""
        pass
        
    def stream_chunks(self, name: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Stream data in chunks for memory-efficient processing."""
        pass
        
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata about the data source."""
        pass
```

### 3.3. Operation Results (`op_result.py`)

#### `OperationStatus`

Enumeration of operation execution statuses:

```python
class OperationStatus(enum.Enum):
    """Status codes for operation results."""
    SUCCESS = "success"
    WARNING = "warning"  # Completed with some issues
    ERROR = "error"
    SKIPPED = "skipped"
    CACHED = "cached"    # Result loaded from cache
```

#### `OperationArtifact`

Class representing a single artifact produced by an operation:

```python
class OperationArtifact:
    """Represents a single artifact produced by an operation."""
    
    def __init__(self, 
                 artifact_type: str, 
                 path: Union[str, Path], 
                 description: str = "",
                 metadata: Dict[str, Any] = None):
        self.artifact_type = artifact_type
        self.path = Path(path) if isinstance(path, str) else path
        self.description = description
        self.metadata = metadata or {}
        self.creation_time = datetime.now().isoformat()
        self.checksum = self._calculate_checksum()
        
    def _calculate_checksum(self) -> str:
        """Calculate a checksum for the artifact for verification."""
        pass
```

#### `OperationResult`

Class representing the complete result of an operation:

```python
class OperationResult:
    """Represents the result of an operation."""
    
    def __init__(self, 
                 status: OperationStatus = OperationStatus.SUCCESS,
                 artifacts: List[OperationArtifact] = None,
                 metrics: Dict[str, Any] = None,
                 error_message: str = None,
                 execution_time: float = None,
                 version: str = None,
                 resources_used: Dict[str, Any] = None):
        self.status = status
        self.artifacts = artifacts or []
        self.metrics = metrics or {}
        self.error_message = error_message
        self.execution_time = execution_time
        self.creation_time = datetime.now().isoformat()
        self.version = version
        self.resources_used = resources_used or {}
    
    def add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = ""):
        """Add an artifact to the result."""
        pass
    
    def add_metric(self, name: str, value: Any):
        """Add a metric to the result."""
        pass
        
    def to_json(self) -> str:
        """Serialize the result to JSON."""
        pass
        
    def save(self, path: Path) -> Path:
        """Save the result to a file."""
        pass
        
    @classmethod
    def load(cls, path: Path) -> 'OperationResult':
        """Load a result from a file."""
        pass
```

### 3.4. Operation Configuration (`op_config.py`)

#### `OperationConfig`

Class for managing operation configuration:

```python
class OperationConfig:
    """Configuration for operations."""
    
    def __init__(self,
                 config_dict: Dict[str, Any] = None,
                 config_file: Optional[Path] = None):
        self.config = config_dict or {}
        if config_file:
            self.load_from_file(config_file)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        pass
        
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        pass
        
    def load_from_file(self, path: Path) -> None:
        """Load configuration from a file."""
        pass
        
    def save_to_file(self, path: Path) -> None:
        """Save configuration to a file."""
        pass
        
    def merge(self, other: 'OperationConfig') -> 'OperationConfig':
        """Merge with another configuration."""
        pass
```

## 4. Typical Operation Structure

Each operation should contain 2-3 key classes:

1. **Analyzer** - responsible for the analytical part: calculating metrics, statistics, extracting patterns
2. **Operation** - responsible for the overall execution including saving results, generating artifacts, logging
3. **Utils** (optional, in commons) - helper functions and classes to support operations of a specific type

### 4.1. Example Structure for Numeric Field Analysis

#### `pamola_core.profiling.commons.numeric_utils.py`

```python
"""Utility functions for numeric field analysis."""

def calculate_extended_stats(data: pd.Series, near_zero_threshold: float = 1e-10) -> Dict[str, Any]:
    """Calculate extended statistics for numeric data."""
    pass

def calculate_percentiles(data: pd.Series) -> Dict[str, float]:
    """Calculate percentiles for numeric data."""
    pass

def calculate_histogram(data: pd.Series, bins: int) -> Dict[str, Any]:
    """Calculate histogram data."""
    pass

def detect_outliers(data: pd.Series) -> Dict[str, Any]:
    """Detect outliers using the IQR method."""
    pass

def test_normality(data: pd.Series, method: str = "all") -> Dict[str, Any]:
    """Test for normal distribution."""
    pass
```

#### `pamola_core.profiling.numeric.py`

```python
"""Numeric field analysis in PAMOLA.CORE."""

class NumericAnalyzer:
    """Analyzer for numeric fields."""
    
    def analyze(self, df: pd.DataFrame, field_name: str, **kwargs) -> Dict[str, Any]:
        """Analyze a numeric field in the DataFrame."""
        pass
        
    def estimate_resources(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """Estimate resources needed for the analysis."""
        pass
        
    def validate_field(self, df: pd.DataFrame, field_name: str) -> bool:
        """Validate if the field is suitable for numeric analysis."""
        pass

class NumericOperation(FieldOperation):
    """Operation for numeric field analysis."""
    
    def __init__(self, field_name: str, bins: int = 10):
        super().__init__(field_name, f"Analysis of numeric field {field_name}")
        self.bins = bins
        self.analyzer = NumericAnalyzer()
    
    def execute(self, data_source: DataSource, task_dir: Path, reporter: Any, 
                progress_tracker: Optional[ProgressTracker] = None, **kwargs) -> OperationResult:
        """Execute the numeric field analysis operation."""
        pass
```

## 5. Standards and Conventions

### 5.1. Data Input

- All operations must accept data through `DataSource`
- For field operations, always specify `field_name`
- Explicitly check that fields exist in the DataFrame before analysis
- Handle missing values according to established rules
- Validate input data quality before processing

### 5.2. Data Output

- All artifacts must be saved in standard subdirectories:
  - `task_dir/` - root directory of the task
  - `task_dir/output/` - for modified data
  - `task_dir/dictionaries/` - for extracted dictionaries
  - `task_dir/visualizations/` - for visualizations
- Logs should be written to `task_dir.parent/logs/`
- All artifacts must be registered in `OperationResult` and `reporter`
- Include version information and checksums with artifacts

### 5.3. Reporting and Logging

- All operations should use `progress_tracker` for progress tracking
- All operations should use logging with appropriate importance levels
- Errors should be logged with full stack traces
- Operation results must include:
  - Execution status
  - List of artifacts with descriptions
  - Key metrics
  - Execution time
  - Resource usage information

### 5.4. Error Handling

- All operations must catch exceptions and return `OperationResult` with `ERROR` status
- Error messages should be informative and include the root cause
- Errors should be logged but not interrupt the task's overall execution
- When possible, operations should return partial results even when errors occur

### 5.5. Internationalization

- All code comments and documentation must be in English
- Support processing data in multiple languages where applicable
- Handle different character encodings properly
- Support regional data formats (dates, currencies, etc.)

## 6. Integration with Other Components

### 6.1. Integration with Task System

Operations should be easily integrated into higher-level tasks:

```python
# Example of using an operation in a task
task_dir = Path("/path/to/task")
data_source = DataSource.from_file_path("data.csv")
reporter = Reporter()  # Object for collecting reports

# Create and run operation
operation = NumericOperation("salary")
result = operation.run(data_source, task_dir, reporter)

# Use results
if result.status == OperationStatus.SUCCESS:
    # Process successful result
    for artifact in result.artifacts:
        if artifact.artifact_type == "json":
            # Process JSON artifact
else:
    # Handle error
    logger.error(f"Operation failed: {result.error_message}")
```

### 6.2. Integration with Visualization

- All visualizations should be created through `pamola_core.utils.visualization`
- Visualizations should be optional (parameter `generate_plots`)
- If visualization errors occur, the operation should continue execution
- Visualization parameters should be configurable

### 6.3. Integration with I/O System

- All data reading/writing operations should be performed through `pamola_core.utils.io`
- For large files, stream processing and chunking should be supported
- Support encrypted data handling through `pamola_core.utils.crypto`
- Standardize artifact paths and naming conventions

### 6.4. Integration with Reporting System

The `reporter` interface must support:

- Recording operation start, progress, and completion
- Registering artifacts with types and descriptions
- Logging metrics and key findings
- Aggregating results from multiple operations
- Generating task-level summary reports

### 6.5. Integration with Caching System

Operations should support result caching:

- Check if a valid cached result exists before execution
- Save results to cache after successful execution
- Invalidate cache when input data or parameters change
- Support forced cache bypass for debugging

## 7. Key Supporting Modules

The following supporting modules are required for operation implementation:

### 7.1. `pamola_core.utils.io.py`

Input/output utilities for working with various data formats:

```python
# Key functions
def save_profiling_results(result: Dict, task_dir: Path, name: str) -> Path:
    """Save profiling results to a JSON file."""
    pass
    
def read_full_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    pass
    
def read_csv_in_chunks(path: Path, chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
    """Read a CSV file in chunks for memory-efficient processing."""
    pass
    
def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    pass
    
def write_dataframe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Write a DataFrame to a CSV file."""
    pass
```

### 7.2. `pamola_core.utils.visualization.py`

Visualization utilities for creating standardized plots and charts:

```python
# Key functions
def create_bar_plot(data: Dict, title: str, path: Path, **kwargs) -> Path:
    """Create a bar plot and save it to a file."""
    pass
    
def create_histogram(data: Dict, title: str, path: Path, **kwargs) -> Path:
    """Create a histogram and save it to a file."""
    pass
    
def plot_text_length_distribution(data: Dict, title: str, path: Path) -> Path:
    """Create a text length distribution plot and save it to a file."""
    pass
    
def create_correlation_matrix(data: pd.DataFrame, title: str, path: Path) -> Path:
    """Create a correlation matrix heatmap and save it to a file."""
    pass
```

### 7.3. `pamola_core.utils.progress.py`

Progress tracking utilities for monitoring operation execution:

```python
class ProgressTracker:
    """Track progress of an operation."""
    
    def __init__(self, total: int, description: str, unit: str = "steps"):
        """Initialize the progress tracker."""
        pass
        
    def update(self, n: int, info: Dict = None):
        """Update progress by n units."""
        pass
        
    def close(self):
        """Mark the progress as complete."""
        pass
        
    def set_description(self, description: str):
        """Set a new description for the progress bar."""
        pass
```

### 7.4. `pamola_core.utils.logging.py`

Logging utilities for standardized log handling:

```python
def configure_logging(log_file=None, level=logging.INFO, name="pamola_core"):
    """Configure logging for the project."""
    pass
    
def configure_task_logging(task_dir, level=logging.INFO):
    """Configure logging for a specific task."""
    pass
    
def get_logger(name):
    """Get a logger for the specified module."""
    pass
```

### 7.5. `pamola_core.utils.crypto.py` (optional)

Utilities for encrypting and decrypting data:

```python
def encrypt_file(path: Path, key: str) -> Path:
    """Encrypt a file."""
    pass
    
def decrypt_file(path: Path, key: str) -> Path:
    """Decrypt a file."""
    pass
    
def encrypt_dataframe(df: pd.DataFrame, key: str) -> bytes:
    """Encrypt a DataFrame."""
    pass
    
def decrypt_dataframe(data: bytes, key: str) -> pd.DataFrame:
    """Decrypt a DataFrame."""
    pass
```

### 7.6. `pamola_core.utils.resource_monitor.py`

Utilities for monitoring resource usage:

```python
class ResourceMonitor:
    """Monitor resource usage during operation execution."""
    
    def __init__(self):
        """Initialize the resource monitor."""
        pass
        
    def start(self):
        """Start monitoring resources."""
        pass
        
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return resource usage statistics."""
        pass
        
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        pass
```

## 8. Refactoring Existing Modules

When refactoring existing modules to transition to the new architecture, follow these steps:

### 8.1. Separate Analysis and Operation Logic

1. **Extract analytical logic** into an `<Type>Analyzer` class:
   - Move data processing and analysis functions
   - Add validation and resource estimation methods
   - Focus on pure analysis with no I/O dependencies

2. **Move infrastructure code** to an `<Type>Operation` class:
   - Handle saving results, reporting, and artifact generation
   - Implement progress tracking and error handling
   - Integrate with the task system

3. **Extract helper functions** to `commons/<type>_utils.py`:
   - Move reusable calculation functions
   - Create utility functions for domain-specific logic
   - Ensure functions are stateless and testable

### 8.2. Adapt Method Signatures

1. Update method signatures to match base classes:
   - `execute()` method should conform to the base class interface
   - Add support for `ProgressTracker` and `DataSource`
   - Standardize parameter names and types

2. Add new required methods:
   - Validation methods to check inputs
   - Resource estimation methods
   - Documentation with complete parameter descriptions

### 8.3. Update Logging and Error Handling

1. Replace direct logging with `self.logger`:
   - Use appropriate log levels (INFO, WARNING, ERROR)
   - Add context to log messages
   - Standardize log format

2. Implement exception handling:
   - Catch and handle exceptions properly
   - Create `OperationResult` with `ERROR` status
   - Provide detailed error messages

### 8.4. Standardize Output Data

1. Return `OperationResult` instead of dictionaries:
   - Set appropriate status
   - Add all artifacts with descriptions
   - Include key metrics and execution time

2. Register all artifacts in the reporter:
   - Conform to the artifact directory structure
   - Use standardized file naming conventions
   - Include version and metadata

### 8.5. Testing and Validation

1. Test the refactored module with:
   - Different types of input data
   - Edge cases (empty data, all nulls, etc.)
   - Error conditions
   - Resource limits

2. Validate results match the original implementation:
   - Compare key metrics
   - Check artifact content
   - Verify performance characteristics

## 9. Creating New Operations

When creating new operations, follow these steps:

### 9.1. Choose the Base Class

1. Choose the appropriate base class:
   - `FieldOperation` for operations on specific fields
   - `DataFrameOperation` for operations on entire tables
   - Create a custom class inheriting from `BaseOperation` for special cases

2. Determine the operation scope:
   - What data does it process?
   - What artifacts does it produce?
   - What parameters can users configure?

### 9.2. Define Initialization Parameters

1. Define what is required to configure the operation:
   - Required parameters (like field names)
   - Optional parameters with sensible defaults
   - Validation for parameter values

2. Document all parameters clearly:
   - Purpose of each parameter
   - Valid value ranges
   - Default values and their implications

### 9.3. Implement the Analyzer Class

1. Create a separate `<Type>Analyzer` class:
   - Implement the pamola core analytical logic
   - Add validation methods
   - Add resource estimation methods
   - Keep it independent of I/O and reporting

2. Define the analysis interface:
   - Clear method signatures with typing
   - Comprehensive documentation
   - Error handling for analytical operations

### 9.4. Implement the Operation

1. Implement the `execute()` method:
   - Get data from `DataSource`
   - Process data through the `Analyzer`
   - Create and save artifacts
   - Track progress via `progress_tracker`
   - Return an `OperationResult`

2. Handle edge cases and errors:
   - Input validation
   - Resource constraints
   - Data quality issues
   - Unexpected outputs

### 9.5. Create Helper Utilities

1. For complex operations, create utilities in the commons package:
   - Reusable algorithms
   - Domain-specific functions
   - Data transformation utilities
   - Specialized statistical calculations

2. Ensure utilities are:
   - Well-documented
   - Thoroughly tested
   - Stateless when possible
   - Performance-optimized

### 9.6. Create Tests

1. Test the operation with various inputs:
   - Normal cases
   - Edge cases
   - Error cases
   - Performance tests for large datasets

2. Verify outputs:
   - Artifact correctness
   - Statistical accuracy
   - Error handling behavior
   - Resource usage

## 10. Modules to Implement or Refactor

For a full implementation of the future operations infrastructure, the following modules need to be implemented or refactored:

### 10.1. Pamola Core Infrastructure

- [x] `pamola_core.utils.tasks.op_base.py` - Base operation classes
- [x] `pamola_core.utils.tasks.op_data_source.py` - Data source abstraction
- [x] `pamola_core.utils.tasks.op_result.py` - Operation results representation
- [ ] `pamola_core.utils.tasks.op_registry.py` - Operation registration
- [ ] `pamola_core.utils.tasks.op_config.py` - Operation configuration management
- [ ] `pamola_core.utils.tasks.op_cache.py` - Result caching mechanisms

### 10.2. Utility Modules

- [ ] `pamola_core.utils.io.py` - Data input/output
- [ ] `pamola_core.utils.visualization.py` - Data visualization
- [ ] `pamola_core.utils.progress.py` - Progress tracking
- [ ] `pamola_core.utils.logging.py` - Logging
- [ ] `pamola_core.utils.crypto.py` - Encryption (optional)
- [ ] `pamola_core.utils.resource_monitor.py` - Resource usage monitoring

### 10.3. Profiling Modules to Refactor

- [ ] `pamola_core.profiling.commons.numeric_utils.py` and `pamola_core.profiling.numeric.py`
- [ ] `pamola_core.profiling.commons.categorical_utils.py` and `pamola_core.profiling.categorical.py`
- [ ] `pamola_core.profiling.commons.date_utils.py` and `pamola_core.profiling.date.py`
- [ ] `pamola_core.profiling.commons.correlation_utils.py` and `pamola_core.profiling.correlation.py`
- [ ] `pamola_core.profiling.commons.email_utils.py` and `pamola_core.profiling.email.py`
- [ ] `pamola_core.profiling.commons.phone_utils.py` and `pamola_core.profiling.phone.py`

### 10.4. New Modules to Implement

- [ ] `pamola_core.profiling.commons.text_utils.py` and `pamola_core.profiling.text.py`
- [ ] `pamola_core.profiling.commons.longtext_utils.py` and `pamola_core.profiling.longtext.py`
- [ ] `pamola_core.profiling.commons.privacy_utils.py` and `pamola_core.profiling.privacy.py`
- [ ] `pamola_core.profiling.commons.mvf_utils.py` and `pamola_core.profiling.mvf.py`
- [ ] `pamola_core.profiling.commons.group_utils.py` and `pamola_core.profiling.group.py`

## 11. Additional Considerations

### 11.1. Caching and Reusability

Operations should support result caching to avoid redundant computation:

- Cache results based on input data checksum and operation parameters
- Provide cache invalidation mechanisms when input data changes
- Allow forced cache bypass when needed
- Store cache metadata to verify cache validity

### 11.2. Resource Management

Operations should be aware of resource constraints and adapt accordingly:

- Monitor memory usage during execution
- Adjust chunk sizes based on available memory
- Support resource usage limits (memory, CPU)
- Provide resource usage estimates before execution
- Log resource usage metrics after execution

### 11.3. Result Versioning and Reproducibility

Ensure reproducibility and traceability of operation results:

- Version all artifacts with operation version and parameters
- Store complete operation configuration with results
- Include checksums for input data references
- Support result comparison between different runs
- Maintain metadata about the execution environment

### 11.4. Internationalization and Localization

Support for international data and users:

- Handle text processing in multiple languages
- Support regional data formats (dates, numbers, etc.)
- Localize error messages and reports
- Encode/decode text properly for different character sets

### 11.5. Error Handling and Recovery

Robust error handling mechanisms:

- Categorize errors (data errors, resource errors, configuration errors)
- Provide recovery mechanisms for transient failures
- Support checkpointing for long-running operations
- Allow partial results from failed operations when applicable
- Detailed error reporting with context information

### 11.6. Integration with Task System

Operations must integrate seamlessly with the task system:

- Support dependency specification between operations
- Allow task-level configuration overrides
- Report progress and status to the task manager
- Support conditional execution based on previous operation results
- Enable artifact discovery between operations

### 11.7. Security Considerations

Ensure secure handling of data throughout operations:

- Support encrypted data sources
- Secure artifact storage
- Audit access to sensitive data
- Clean up temporary files securely
- Validate untrusted inputs

### 11.8. API and UI Integration

Provide integration points for APIs and user interfaces:

- Well-defined operation metadata for discovery
- Progress and status reporting interfaces
- Standardized artifact representation for rendering
- Operation parameter validation
- Self-documentation of operations for UI generation