# PAMOLA Operations & Tasks Framework: Architectural Analysis

## Executive Summary

PAMOLA Core implements a two-tier execution framework:
- **Operations (Ops)** -- atomic units of data processing
- **Tasks (Jobs)** -- orchestrators that compose, configure, and execute sequences of operations

This document analyzes how the `utils/ops/` and `utils/tasks/` subsystems are built, what contracts they enforce, and what it means for a module to be a "real operation" or "executable task" versus a plain function.

---

## 1. Operations Framework (`utils/ops/`)

### 1.1 Class Hierarchy

```
BaseOperation (ABC)                     # utils/ops/op_base.py
  ├── FieldOperation (ABC)              # Per-column operations
  │     └── [Concrete: CategoricalOperation, DateOperation, EmailOperation, ...]
  └── DataFrameOperation (ABC)          # Whole-DataFrame operations
        └── [Concrete: CorrelationMatrixOperation, KAnonymityProfilerOperation, ...]
```

**Key principle:** `BaseOperation` is the single contract. Everything that wants to participate in the framework MUST extend it (directly or via `FieldOperation`/`DataFrameOperation`).

### 1.2 The Operation Contract

A class becomes a "real operation" by inheriting from `BaseOperation` and fulfilling these obligations:

| Obligation | Mechanism | File |
|---|---|---|
| **Identity** | `name`, `description`, `version` attributes | `op_base.py:139-293` |
| **Scope** | `OperationScope` (datasets, fields, field_groups) | `op_base.py:52-128` |
| **Configuration** | `OperationConfig` with JSON Schema validation | `op_config.py` |
| **Execute contract** | Abstract `execute(data_source, task_dir, reporter, progress_tracker)` | `op_base.py:356-386` |
| **Result contract** | Must return `OperationResult` with status, artifacts, metrics | `op_result.py` |
| **Auto-registration** | `register_operation()` called in `__init__` | `op_base.py:304` |
| **Cache key generation** | Override `_get_cache_parameters()` for deterministic caching | `op_base.py:843-856` |

### 1.3 Execution Lifecycle

The `run()` method (NOT `execute()`) is the entry point called by the Task framework. It wraps `execute()` with a full lifecycle:

```
run()
  1. Validate encryption/vectorization/dask settings
  2. save_config(task_dir)        -- atomic JSON write (REQ-OPS-004)
  3. _log_operation_start()       -- parameter logging (secrets redacted)
  4. Create HierarchicalProgressTracker if needed
  5. Register operation with reporter
  6. Build execution_params (pre_process + process + output configs)
  7. execute(...)                  -- ABSTRACT: subclass implements this
  8. Set execution_time on result
  9. Standardize error results via ErrorHandler
 10. Report final status to reporter
 11. _log_operation_end()

Exception path:
  - Caught by ErrorHandler.handle_error()
  - Returns OperationResult with ERROR status
  - Error code from ErrorCode enum (centralized error taxonomy)
```

### 1.4 What `execute()` Receives

```python
def execute(
    self,
    data_source: DataSource,        # Unified data abstraction (DataFrames + file paths)
    task_dir: Path,                  # Artifact output directory
    reporter: Any,                   # TaskReporter for audit trail
    progress_tracker: HierarchicalProgressTracker,
    **kwargs,                        # Merged pre_process + process + output configs
) -> OperationResult:
```

The `DataSource` abstraction (`op_data_source.py`) provides:
- `get_dataframe(name)` -- lazy loading from files or in-memory
- `suggest_engine(name)` -- auto-selects pandas vs. dask based on data size
- Schema validation and type checking
- Multi-dataset support (primary + auxiliary)

### 1.5 Result & Artifact System

`OperationResult` (`op_result.py`) is a structured container:

```
OperationResult
  ├── status: OperationStatus (SUCCESS|WARNING|ERROR|SKIPPED|PARTIAL_SUCCESS|PENDING)
  ├── artifacts: List[OperationArtifact]
  │     └── Each: type, path, description, category, tags, checksum
  ├── artifact_groups: Dict[str, ArtifactGroup]
  ├── metrics: Dict[str, Any]            # Nested metric structure
  ├── error_message, error_trace, exception
  └── execution_time: float
```

Artifacts support:
- Direct file registration (`add_artifact`)
- Writer-based registration (`register_artifact_via_writer` -> DataWriter)
- Validation (existence, size, type, checksum)
- Grouping and tagging

### 1.6 Operation Registry (`op_registry.py`)

The registry provides:
- **Auto-registration**: Every `BaseOperation.__init__` calls `register_operation()`
- **Lazy loading**: `lazily_load_operation()` attempts import from common paths
- **Discovery**: `discover_operations('pamola_core')` recursively scans packages
- **Metadata extraction**: Parameters, category, version -- all introspected automatically
- **Version management**: Semantic versioning with constraint checking (`>=1.0.0`, `1.x.x`)
- **Dependency resolution**: Operations can declare dependencies on other operations
- **Factory**: `create_operation_instance(name, **kwargs)` -- instantiation by name

Decorator form:
```python
@register(version="2.0.0", dependencies=[{"name": "BaseProfiler", "version": ">=1.0.0"}])
class MyOperation(FieldOperation):
    ...
```

### 1.7 Configuration Management (`op_config.py`)

`OperationConfig` provides:
- JSON Schema validation (via `validate_json_schema()`)
- Save/load to/from JSON files
- Type-safe access (`get()`, `__getitem__`, `__contains__`)
- `OperationConfigRegistry` -- maps operation types to their config classes

---

## 2. Tasks Framework (`utils/tasks/`)

### 2.1 What Is a Task (Job)?

A **Task** is an executable unit of work that:
1. Loads and validates configuration
2. Creates directory structure for artifacts
3. Configures a sequence of Operations
4. Executes those Operations in order (with retry, checkpointing, progress)
5. Collects results, generates reports, records execution history

### 2.2 The Task Contract

A class becomes a "real task" by inheriting from `BaseTask` and implementing:

| Obligation | Mechanism | File |
|---|---|---|
| **Identity** | `task_id`, `task_type`, `description`, `version` | `base_task.py:131-160` |
| **Default config** | Override `get_default_config()` | `base_task.py:217-249` |
| **Operation setup** | Override `configure_operations()` | `base_task.py:722-754` |
| **Registration** | `task_registry.register_task_class(task_id, cls)` | `task_registry.py` |

Everything else is provided by the framework.

### 2.3 Task Lifecycle

```
run(args, force_restart, enable_checkpoints)
  │
  ├── 1. initialize(args)
  │     ├── load_task_config()         -- YAML/JSON with priority cascade
  │     ├── _setup_logging()           -- dual logging (project + task)
  │     ├── create_directory_manager() -- standardized directory structure
  │     ├── TaskReporter()             -- report generation
  │     ├── TaskDependencyManager()    -- check prerequisite tasks
  │     ├── TaskEncryptionManager()    -- encryption setup (none/simple/age)
  │     ├── TaskContextManager()       -- checkpoint/resume support
  │     ├── TaskProgressManager()      -- progress bars
  │     ├── TaskOperationExecutor()    -- execution engine with retry
  │     ├── DataSource()               -- input datasets
  │     └── DataWriter()               -- output handling
  │
  ├── 2. configure_operations()        -- SUBCLASS IMPLEMENTS THIS
  │     └── self.add_operation(cls_or_name, **kwargs)
  │
  ├── 3. execute()
  │     ├── Resume from checkpoint if enabled
  │     └── _run_operations(start_idx)
  │           ├── For each operation:
  │           │   ├── _prepare_operation_parameters()
  │           │   ├── operation_executor.execute_with_retry()
  │           │   ├── Collect results, artifacts, metrics
  │           │   └── Create automatic checkpoint
  │           └── Handle errors (continue_on_error flag)
  │
  └── 4. finalize(success)
        ├── reporter.add_task_summary()
        ├── reporter.save_report()     -- JSON report
        ├── record_task_execution()    -- execution log (cross-run)
        └── Cleanup resources
```

### 2.4 Component Managers (Facade Pattern)

BaseTask delegates to specialized managers:

| Manager | Responsibility | File |
|---|---|---|
| `DirectoryManager` | Create/validate task directory structure | `directory_manager.py` |
| `TaskEncryptionManager` | Encryption lifecycle (init, check, context) | `encryption_manager.py` |
| `TaskContextManager` | Checkpoints, state serialization, resume | `context_manager.py` |
| `TaskDependencyManager` | Dependency resolution and validation | `dependency_manager.py` |
| `TaskProgressManager` | Progress bars and operation tracking | `progress_manager.py` |
| `TaskOperationExecutor` | Operation execution with retry/backoff | `operation_executor.py` |
| `TaskReporter` | Report generation, artifact tracking | `task_reporting.py` |
| `DataSource` | Unified data input abstraction | `op_data_source.py` |
| `DataWriter` | Consistent output writing with encryption | `op_data_writer.py` |

### 2.5 Operation Executor with Retry (`operation_executor.py`)

`TaskOperationExecutor` provides:
- **Single execution**: `execute_operation(op, params)`
- **Retry with backoff**: `execute_with_retry(op, params, max_retries, backoff_factor, ...)`
  - Exponential backoff with jitter
  - Selective retry based on exception type (ConnectionError, TimeoutError, IOError)
  - Never-retry list (MemoryError, TypeError, ValueError, KeyboardInterrupt)
  - `on_retry` callback
- **Batch sequential**: `execute_operations(ops, common_params, continue_on_error)`
- **Batch parallel**: `execute_operations_parallel(ops, common_params, max_workers)`
  - Uses `ProcessPoolExecutor`
  - Falls back to sequential on failure
- Execution statistics tracking

### 2.6 Task Registry (`task_registry.py`)

Mirrors the operation registry but for tasks:
- `register_task_class(task_id, cls)` -- register by ID
- `discover_task_classes(package_paths)` -- recursive package scanning
- `create_task_instance(task_id, **kwargs)` -- factory
- `check_task_dependencies(task_id, ...)` -- cross-execution-log validation
- Validation: checks for `BaseTask` inheritance, required methods (`configure_operations`, `run`, `initialize`, `execute`, `finalize`)

### 2.7 Configuration Cascade (`task_config.py`)

Configuration priority (highest to lowest):
1. **Command-line args** -- runtime overrides
2. **Task-specific config** -- `configs/{task_id}/config.yaml`
3. **Project config** -- `pamola_project.yaml`
4. **Default config** -- from `get_default_config()`

Includes:
- Project root discovery
- Path resolution and security validation
- Environment variable support
- Encryption key path resolution

### 2.8 Checkpoint & Resume (`context_manager.py`)

`TaskContextManager` provides:
- **Automatic checkpointing** after each operation
- **State serialization** -- operation index, metrics, custom state
- **Resumable execution** -- `can_resume_execution()` -> `restore_execution_state()`
- File-lock-based atomic checkpoint writes
- Integration with execution log for cross-run persistence
- Checkpoint verification and integrity checking

### 2.9 Execution Log (`execution_log.py`)

Project-level persistent history:
- `record_task_execution()` -- records task run with metadata
- `find_latest_execution()` -- query by task_id
- File-lock-based concurrent access
- Used by dependency manager to verify prerequisite tasks completed successfully

---

## 3. What Makes Something a "Real Operation" vs. "Just a Function"

### 3.1 The Gap

Many existing PAMOLA modules do NOT use the Ops framework:

| Module | Current Pattern | What's Missing |
|---|---|---|
| **Profiling analyzers** | Named `*Operation` but custom classes | No `BaseOperation` inheritance, no registry, no `OperationResult` |
| **Analysis functions** | Pure functions (`analyze_*`) | No class, no lifecycle, no artifact management |
| **Attack classes** | Extend `AttackInitialization/PreprocessData` | Separate hierarchy, no `execute()` contract |
| **IO handlers** | Protocol implementations | Different abstraction level (data access, not processing) |
| **Privacy models** | Extend `BasePrivacyModelProcessor` | Separate hierarchy, custom lifecycle |

### 3.2 What a "Real Operation" Gets You

By inheriting from `BaseOperation`, a component automatically gains:

1. **Lifecycle management** -- `run()` handles timing, logging, error handling
2. **Registry participation** -- discoverable, instantiable by name, version-tracked
3. **Configuration schema** -- validated, serializable, auditable (`config.json`)
4. **Artifact management** -- structured output with checksums, categories, tags
5. **Progress tracking** -- hierarchical progress bars
6. **Caching** -- deterministic cache keys from parameters + data hash
7. **Encryption** -- transparent output encryption
8. **Error standardization** -- centralized error codes, structured error results
9. **Reporter integration** -- automatic audit trail
10. **Task composability** -- can be added to any Task via `add_operation()`
11. **Retry support** -- transient failures handled by TaskOperationExecutor
12. **Dask/parallel** -- auto-engine selection, chunked processing, vectorization

### 3.3 What a "Real Task" Gets You

By inheriting from `BaseTask`, a component gains:

1. **Full lifecycle** -- init/configure/execute/finalize with error handling
2. **Configuration cascade** -- project -> task -> args priority
3. **Directory management** -- standardized structure for outputs
4. **Dependency validation** -- prerequisite tasks checked before execution
5. **Checkpoint/resume** -- automatic checkpointing, resumable execution
6. **Dual logging** -- project-level + task-specific logs
7. **Encryption lifecycle** -- key management, mode selection
8. **Execution history** -- persistent log across runs
9. **Progress visualization** -- operation-level progress bars
10. **Report generation** -- structured JSON reports with artifacts
11. **Operation orchestration** -- sequential/parallel execution with retry

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER / SCRIPT                                 │
│                        task.run(args={...})                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          BaseTask.run()                                 │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │initialize │→│configure_operations│→│ execute  │→│  finalize     │   │
│  └──────────┘  └──────────────────┘  └────┬─────┘  └──────────────┘   │
│       │                                    │                            │
│  ┌────┴──────────────────┐    ┌────────────┴──────────────┐            │
│  │ Component Managers:   │    │ TaskOperationExecutor:     │            │
│  │ - DirectoryManager    │    │  for op in operations:     │            │
│  │ - EncryptionManager   │    │    execute_with_retry(op)  │            │
│  │ - ContextManager      │    │      │                     │            │
│  │ - DependencyManager   │    │      ▼                     │            │
│  │ - ProgressManager     │    │    op.run(data_source,     │            │
│  │ - TaskReporter        │    │         task_dir,          │            │
│  │ - DataSource          │    │         reporter, ...)     │            │
│  │ - DataWriter          │    │      │                     │            │
│  └───────────────────────┘    │      ▼                     │            │
│                               │  ┌──────────────────┐     │            │
│                               │  │BaseOperation.run()│     │            │
│                               │  │  save_config()   │     │            │
│                               │  │  log_start()     │     │            │
│                               │  │  execute(...)     │◄── abstract     │
│                               │  │  log_end()        │     │            │
│                               │  │  → OperationResult│     │            │
│                               │  └──────────────────┘     │            │
│                               └───────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Framework Registries

### Operation Registry (in-memory, global)

```
_OPERATION_REGISTRY     : Dict[str, Type]        # class_name -> class
_OPERATION_METADATA     : Dict[str, Dict]        # class_name -> {module, category, parameters}
_OPERATION_DEPENDENCIES : Dict[str, List[Dict]]  # class_name -> [{name, version}]
_OPERATION_VERSIONS     : Dict[str, str]         # class_name -> semver string
```

### Task Registry (in-memory, global)

```
_task_classes : Dict[str, Type]                  # task_id -> class
```

### Execution Log (file-based, project-level)

```
configs/execution_log.json                       # Persistent cross-run history
  └── {task_id: {success, execution_time, report_path, input_datasets, ...}}
```

---

## 6. Summary: The Transformation Required

To convert a "plain function" or "custom class" into a framework-compliant component:

### For an Operation:

```python
# FROM: plain function
def analyze_something(df, field_name, threshold=0.5):
    result = ...
    return result

# TO: framework operation
class AnalyzeSomethingOperation(FieldOperation):
    def __init__(self, field_name: str, threshold: float = 0.5, **kwargs):
        super().__init__(field_name=field_name, **kwargs)
        self.threshold = threshold

    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        df = data_source.get_dataframe("main")
        dirs = self._prepare_directories(task_dir)
        result = OperationResult(status=OperationStatus.SUCCESS)

        # ... processing logic ...

        result.add_artifact("json", dirs["output"] / "analysis.json", "Analysis results")
        result.add_metric("threshold", self.threshold)
        return result

    def _get_cache_parameters(self):
        return {"threshold": self.threshold}
```

### For a Task:

```python
class AnalysisTask(BaseTask):
    def __init__(self):
        super().__init__(
            task_id="analysis_task",
            task_type="analysis",
            description="Run full analysis pipeline",
        )

    def configure_operations(self):
        self.add_operation(AnalyzeSomethingOperation, field_name="age", threshold=0.3)
        self.add_operation(AnalyzeSomethingOperation, field_name="income", threshold=0.5)

# Usage:
task = AnalysisTask()
task.run(args={"input_file": "data.csv"})
```

This transformation gives the component access to the entire infrastructure: registry, retry, checkpoints, encryption, reporting, progress tracking, and composability.
