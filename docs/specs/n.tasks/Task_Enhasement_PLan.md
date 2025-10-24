# PAMOLA Core Tasks Package Refactoring Plan

## Overview

This document outlines the phased refactoring plan for the `pamola_core.utils.tasks` package, transforming the monolithic `BaseTask` implementation into a module-based system following the facade pattern. The plan includes module specifications, dependencies, and implementation sequence.

## Current Module Structure

```
pamola_core/utils/tasks/
├── __init__.py
├── base_task.py           # Base class for task implementation
├── task_config.py         # Configuration loading and management
├── task_registry.py       # Task registration and lookup
├── task_reporting.py      # Task execution reporting and artifact tracking
├── project_config_loader.py # Project configuration management
├── path_security.py       # Path security validation
└── execution_log.py       # Task execution history tracking
```

## Target Module Structure

```
pamola_core/utils/tasks/
├── __init__.py
├── base_task.py           # Facade implementation (refactored)
├── task_config.py         # Configuration (already refactored)
├── task_registry.py       # Task registration (unchanged)
├── task_reporting.py      # Task reporting (unchanged)
├── project_config_loader.py # Project configuration (unchanged)
├── path_security.py       # Path security (unchanged)
├── execution_log.py       # Execution logging (enhanced)
├── directory_manager.py   # NEW: Directory structure management
├── encryption_manager.py  # NEW: Encryption functionality
├── operation_executor.py  # NEW: Operation execution with retry
├── context_manager.py     # NEW: Task state management
├── log_manager.py         # NEW: Advanced logging features
└── llm_integration.py     # NEW: LLM integration (optional)
```

## Implementation Phases

### Phase 1: Pamola Core Infrastructure Enhancements

**Focus**: Implement immediate operational improvements without major restructuring

#### 1.1. Enhance `execution_log.py`

**Priority**: High  
**Dependencies**: None  
**Enhancements**:
- Add checkpoint functionality
- Implement execution state serialization
- Add tracking for partial execution history

```python
# Key additions to execution_log.py
def save_execution_checkpoint(task_id: str, state: Dict[str, Any], checkpoint_name: Optional[str] = None) -> Path:
    """Save execution state to a checkpoint file."""

def load_execution_checkpoint(task_id: str, checkpoint_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load execution state from a checkpoint file."""

def get_latest_checkpoint(task_id: str) -> Optional[str]:
    """Get the name of the latest checkpoint for a task."""
```

#### 1.2. Create `log_manager.py`

**Priority**: Medium  
**Dependencies**: None  
**Implementation**:
- Extract logging functionality from `BaseTask`
- Add log rotation capabilities
- Implement dual logging (project/task specific)

```python
# log_manager.py
class TaskLogManager:
    def __init__(self, task_id: str, task_config: TaskConfig):
        """Initialize log manager for a specific task."""
        
    def setup_logging(self, max_bytes: int = 10_485_760, backup_count: int = 5) -> logging.Logger:
        """Configure logging with rotation capability."""
        
    def get_log_paths(self) -> Dict[str, Path]:
        """Get paths to log files."""
```

### Phase 2: Security and Resilience

**Focus**: Enhance security features and implement retry capabilities

#### 2.1. Create `encryption_manager.py`

**Priority**: High  
**Dependencies**: None  
**Implementation**:
- Extract encryption logic from `BaseTask`
- Implement secure key handling
- Add data redaction for reports and logs

```python
# encryption_manager.py
class TaskEncryptionManager:
    def __init__(self, task_config: TaskConfig, logger: logging.Logger):
        """Initialize encryption manager."""
        
    def initialize(self) -> bool:
        """Initialize encryption based on configuration."""
        
    def get_encryption_context(self) -> EncryptionContext:
        """Get secure encryption context."""
        
    def redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from data structures."""
```

#### 2.2. Create `operation_executor.py`

**Priority**: High  
**Dependencies**: None  
**Implementation**:
- Extract operation execution from `BaseTask`
- Implement retry logic with exponential backoff
- Add selective retry based on exception types

```python
# operation_executor.py
class TaskOperationExecutor:
    def __init__(self, task_config: TaskConfig, logger: logging.Logger, reporter: TaskReporter):
        """Initialize operation executor."""
        
    def execute_operation(self, operation: BaseOperation, params: Dict[str, Any]) -> OperationResult:
        """Execute a single operation."""
        
    def execute_with_retry(self, operation: BaseOperation, params: Dict[str, Any], 
                          max_retries: int = None, backoff_factor: float = None) -> OperationResult:
        """Execute operation with retry logic."""
        
    def is_retriable_error(self, exception: Exception) -> bool:
        """Determine if an exception should trigger retry."""
```

### Phase 3: Task State Management

**Focus**: Implement task checkpointing and directory management

#### 3.1. Create `directory_manager.py`

**Priority**: Medium  
**Dependencies**: `path_security.py`  
**Implementation**:
- Extract directory management from `BaseTask` and `task_utils.py`
- Implement standard task directory structure
- Add path generation for artifacts

```python
# directory_manager.py
class TaskDirectoryManager:
    def __init__(self, task_config: TaskConfig):
        """Initialize directory manager."""
        
    def ensure_directories(self) -> Dict[str, Path]:
        """Create and validate all required task directories."""
        
    def get_directory(self, dir_type: str) -> Path:
        """Get path to specific directory type."""
        
    def get_artifact_path(self, artifact_name: str, artifact_type: str, 
                         subdir: str = "output", include_timestamp: bool = True) -> Path:
        """Generate standardized path for an artifact."""
        
    def clean_temp_directory(self) -> bool:
        """Clean temporary files and directories."""
```

#### 3.2. Create `context_manager.py`

**Priority**: High  
**Dependencies**: `execution_log.py`, `directory_manager.py`  
**Implementation**:
- Implement task state serialization
- Add automatic checkpointing
- Support for resumable execution

```python
# context_manager.py
class TaskContextManager:
    def __init__(self, task_id: str, task_dir: Path, logger: logging.Logger):
        """Initialize context manager."""
        
    def save_execution_state(self, state: Dict[str, Any], checkpoint_name: Optional[str] = None) -> Path:
        """Save execution state to a checkpoint file."""
        
    def restore_execution_state(self, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """Restore execution state from a checkpoint file."""
        
    def create_automatic_checkpoint(self, operation_index: int, metrics: Dict[str, Any]) -> str:
        """Create an automatic checkpoint at the current execution point."""
        
    def can_resume_execution(self) -> Tuple[bool, Optional[str]]:
        """Check if task execution can be resumed from a checkpoint."""
```

### Phase 4: Advanced Features

**Focus**: Add parallel execution and LLM integration

#### 4.1. Enhance `operation_executor.py` with Parallel Execution

**Priority**: Medium  
**Dependencies**: Initial `operation_executor.py`  
**Enhancements**:
- Add parallel execution capabilities
- Implement worker pool management
- Add dependency resolution for operations

```python
# Additions to operation_executor.py
def execute_operations_parallel(self, operations: List[BaseOperation], 
                               common_params: Dict[str, Any]) -> Dict[str, OperationResult]:
    """Execute operations in parallel using process pool."""
    
def construct_execution_plan(self, operations: List[BaseOperation]) -> List[List[BaseOperation]]:
    """Construct execution plan considering dependencies."""
```

#### 4.2. Create `llm_integration.py` (Optional)

**Priority**: Low  
**Dependencies**: None  
**Implementation**:
- Add LLM integration capabilities
- Implement templating for prompts
- Add response parsing functionality

```python
# llm_integration.py
class LLMIntegrationHelper:
    def __init__(self, task_config: TaskConfig, logger: logging.Logger):
        """Initialize LLM integration helper."""
        
    def prepare_prompt(self, template_name: str, **variables) -> str:
        """Prepare a prompt for LLM using template and parameters."""
        
    def execute_query(self, prompt: str, model: Optional[str] = None, **params) -> str:
        """Execute a query to LLM and return the response."""
        
    def process_response(self, response: str, output_format: Optional[str] = None) -> Any:
        """Process LLM response into the desired format."""
```

### Phase 5: Facade Implementation

**Focus**: Refactor `BaseTask` into a facade that delegates to specialized components

#### 5.1. Refactor `base_task.py`

**Priority**: High  
**Dependencies**: All component modules from Phases 1-4  
**Implementation**:
- Refactor `BaseTask` to use the new components
- Maintain backward compatibility
- Add support for new features through component delegation

```python
# Refactored base_task.py
class BaseTask:
    def __init__(self, task_id: str, task_type: str, description: str, 
                input_datasets: Optional[Dict[str, str]] = None, 
                auxiliary_datasets: Optional[Dict[str, str]] = None, 
                version: str = "1.0.0"):
        """Initialize task with basic information."""
        # Pamola Core attributes
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.version = version
        self.input_datasets = input_datasets or {}
        self.auxiliary_datasets = auxiliary_datasets or {}
        
        # Components (initialized in initialize())
        self.config = None
        self.dir_manager = None
        self.log_manager = None
        self.logger = None
        self.encryption_manager = None
        self.operation_executor = None
        self.context_manager = None
        self.reporter = None
        
        # Status tracking
        self.status = "pending"
        self.error_info = None
        
    def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize task components."""
        # Implementation delegates to components
        
    # Rest of the methods delegate to appropriate components
```

## Testing Strategy

For each module:

1. **Unit Tests**: Create isolated tests for each module's functionality
2. **Integration Tests**: Test interactions between related modules
3. **Backward Compatibility Tests**: Verify that existing task scripts continue to function

## Migration Strategy

1. **Documentation**: Create detailed migration guide for task developers
2. **Deprecation Notices**: Add deprecation warnings for direct component access
3. **Examples**: Update example tasks to use the new component model
4. **Transition Support**: Support both direct and delegated approaches for one development cycle


This plan allows for iterative implementation, with early phases focusing on immediate improvements while later phases address architectural enhancements. The sequencing ensures that dependencies are met and that the most valuable enhancements are delivered early.