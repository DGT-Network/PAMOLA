# Software Requirements Specification
# PAMOLA Core Configuration and Path Management Enhancement

## 1. Introduction

### 1.1 Purpose
This document specifies requirements for enhancing PAMOLA Core's configuration management and path resolution mechanism to ensure consistent and reliable path handling across various execution contexts.

### 1.2 Scope
The enhancements will standardize configuration formats, establish reliable path resolution, fix issues with relative paths, and provide a clear API for consistent path access.

### 1.3 Definitions

| Term | Definition |
|------|------------|
| Project Root | The root directory of a specific implementation project (e.g., PAMOLA.CORE) which uses PAMOLA.CORE. Contains project-specific configurations, logs, and includes PAMOLA.CORE as a package. |
| PAMOLA.CORE | A framework package within the project that provides reusable operations and utilities. May be shared across multiple implementation projects. |
| Data Repository | A potentially separate location containing standardized data directories (`raw/`, `processed/`, `reports/`). May be on a different drive/server from the Project Root. |
| Task | A user-level script with a unique identifier (e.g., "t_1P1") that orchestrates operations. |
| Operation | A pamola core functional module that performs specific data transformations. |
| Task Directory | The specific directory for a task's outputs, typically at `{data_repository}/processed/{task_id}/`. |
| Configuration Cascade | The priority order for configuration: command-line args → task config → project config → defaults. |

## 2. Project Structure

### 2.1 Hierarchy
```
{project_root}/                    # Implementation Project (e.g., PAMOLA.CORE)
    ├── configs/                   # Configuration directory
    │   ├── prj_config.yaml        # Project-level configuration (YAML)
    │   └── {task_id}.json         # Task-specific configurations (JSON)
    ├── logs/                      # Log files
    ├── pamola_core/                      # PAMOLA.CORE framework
    │   ├── utils/
    │   ├── profiling/
    │   └── ...
    ├── scripts/                   # Task scripts
    │   └── mock/
    │       └── t_1P1_group_profile.py
    └── {data_repository}/         # May be external to project
        ├── raw/                   # Raw input data
        ├── processed/             # Directory for task outputs
        │   └── {task_id}/         # Task-specific directories
        │       ├── input/         # Task input directory
        │       ├── output/        # Task output directory
        │       ├── temp/          # Temporary working files
        │       └── dictionaries/  # Extracted data fragments
        └── reports/               # Task reports
```

### 2.2 Project Root Discovery

**REQ-PRD-01:** The system shall discover the project root using one or more of the following methods:
1. Presence of a `.pamola` marker file
2. Presence of a `configs` directory containing `prj_config.yaml`
3. Presence of a `core` directory containing PAMOLA pamola core modules
4. Value of the `PAMOLA_PROJECT_ROOT` environment variable
5. Searching parent directories up to a reasonable limit

**REQ-PRD-02:** If the project root cannot be reliably determined, the system shall fail with a clear error message directing the user to set the `PAMOLA_PROJECT_ROOT` environment variable.

## 3. Configuration Requirements

### 3.1 Project Configuration

**REQ-PC-01:** The system shall support a project-level configuration file at `{project_root}/configs/prj_config.yaml` using YAML format for improved readability and comment support.

**REQ-PC-02:** The project configuration shall define essential path locations:
- `project_root` (absolute path to implementation project)
- `data_repository` (absolute or relative path to data storage)

**REQ-PC-03:** The project configuration shall define directory structure:
- `raw` (directory for input data)
- `processed` (directory for task outputs)
- `reports` (directory for task reports)
- `logs` (directory for log files)
- `configs` (directory for configuration files)

**REQ-PC-04:** The project configuration shall support default settings for:
- Logging (level, format)
- Performance (chunk_size, memory_limits, encoding, delimiter)
- Encryption (use_encryption, encryption_mode, key_path)
- Large data processing (use_dask, chunk_size)
- Task defaults (continue_on_error, parallel_processes)

**REQ-PC-05:** The project configuration shall support task-specific overrides in a `tasks` section.

**REQ-PC-06:** The project configuration shall define standard task directory suffixes in a `task_dir_suffixes` list, with defaults including:
- `input` (directory for task inputs)
- `output` (directory for task outputs)
- `temp` (directory for temporary files)
- `dictionaries` (directory for extracted data fragments)

**Sample Project Configuration**:
```yaml
# Project configuration
project_root: "D:/VK/_DEVEL/PAMOLA.CORE"
data_repository: "D:/VK/_DEVEL/PAMOLA.CORE/data"

# Directory structure
directory_structure:
  raw: "raw"
  processed: "processed"
  reports: "reports"
  logs: "logs"
  configs: "configs"

# Standard task directory suffixes
task_dir_suffixes:
  - "input"
  - "output"
  - "temp"
  - "dictionaries"

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Default performance settings
performance:
  chunk_size: 100000
  default_encoding: "utf-8"
  default_delimiter: ","
  memory_limit_mb: 1000
  use_dask: false

# Default encryption settings
encryption:
  use_encryption: false
  encryption_mode: "none"  # Accepted values: none, simple, fernet, age
  key_path: null

# Task defaults
task_defaults:
  continue_on_error: true
  parallel_processes: 4

# Task-specific configuration overrides
tasks:
  t_1I:
    description: "Initial data ingestion"
    dependencies: []
  t_1P1:
    description: "Group profiling"
    dependencies: ["t_1I"]
```

### 3.2 Task Configuration

**REQ-TC-01:** The system shall support task-specific configuration files at `{project_root}/configs/{task_id}.json` using JSON format.

**REQ-TC-02:** Task configuration shall inherit from project configuration with the ability to override values.

**REQ-TC-03:** Task configuration shall include:
- Task metadata (ID, type, description)
- Dependencies on other tasks
- Behavioral settings (error handling, encryption)
- Task-specific parameters (fields, datasets)
- Operation-specific parameters

**REQ-TC-04:** If a task configuration file doesn't exist, the system shall create one with default values merged from the `task_defaults` section of the project configuration, including a placeholder description and appropriate values for encryption and large data processing.

### 3.3 Configuration Formats

**REQ-CF-01:** Project configuration shall use YAML format with support for comments, while task configurations shall use JSON format.

**REQ-CF-02:** The system shall support variable substitution in configuration values using `${variable}` syntax.

**REQ-CF-03:** Variables available for substitution shall include:
- `project_root` - Absolute path to project root
- `data_repository` - Absolute path to data repository
- `task_id` - Current task identifier
- Any top-level configuration keys

## 4. Path Resolution Requirements

### 4.1 Path Resolution

**REQ-PATH-01:** All relative paths shall be resolved to absolute paths during initialization.

**REQ-PATH-02:** Path resolution shall be independent of the current working directory.

**REQ-PATH-03:** All file paths shall be resolved relative to either project root or data repository.

**REQ-PATH-04:** Tasks shall never directly construct paths using string concatenation.

**REQ-PATH-05:** The system shall provide API methods for accessing common path locations.

**REQ-PATH-06:** Path resolution shall support both project-relative and data-repository-relative paths.

**REQ-PATH-07:** Path resolution shall handle path separator differences across operating systems.

### 4.2 Path Resolution API

**REQ-API-01:** The system shall provide methods for accessing project-related paths:
- `get_project_root()` - Absolute path to project root
- `get_project_config_dir()` - Path to configs directory 
- `get_project_logs_dir()` - Path to logs directory

**REQ-API-02:** The system shall provide methods for accessing data repository paths:
- `get_data_repository()` - Absolute path to data repository
- `get_raw_data_dir()` - Path to raw data directory
- `get_processed_dir()` - Path to processed data directory
- `get_reports_dir()` - Path to reports directory

**REQ-API-03:** The system shall provide methods for accessing task-specific paths:
- `get_task_dir(task_id=None)` - Path to task directory
- `get_task_input_dir(task_id=None)` - Path to task input directory
- `get_task_output_dir(task_id=None)` - Path to task output directory
- `get_task_temp_dir(task_id=None)` - Path to task temporary directory
- `get_task_dictionaries_dir(task_id=None)` - Path to task dictionaries directory

**REQ-API-04:** The system shall provide methods for dependency management:
- `get_dependency_output(dependency_id, file_pattern=None)` - Path(s) to dependency outputs
  - If `dependency_id` is a Path object or absolute path string, use it directly
  - If `dependency_id` is a task ID string, resolve to the task's output directory
- `get_dependency_report(dependency_id)` - Path to dependency report

### 4.3 Path Security

**REQ-SEC-01:** The system shall validate all paths for security before access.

**REQ-SEC-02:** Path validation shall prevent:
- Directory traversal attacks (`..`, `.`)
- Access to system directories
- Command injection via special characters
- References to home directories (`~`)

**REQ-SEC-03:** The system shall provide detailed logging for path security violations.

**REQ-SEC-04:** The system shall raise a `PathSecurityError` and abort initialization when an invalid or unsafe path is detected, unless a `--force` CLI flag is set.

## 5. Task and Operation Integration

### 5.1 Task Path Integration

**REQ-TI-01:** The `BaseTask` class shall initialize and ensure existence of all standard directory suffixes defined in the project configuration.

**REQ-TI-02:** The `BaseTask` class shall provide helper methods for accessing common paths.

**REQ-TI-03:** The `BaseTask` class shall validate to prevent direct use of relative string paths.

**REQ-TI-04:** The `BaseTask` class shall provide clear error messages for path-related failures.

### 5.2 Operation Integration

**REQ-OI-01:** Operations shall receive paths only through the task framework.

**REQ-OI-02:** Operations shall not make assumptions about the current working directory.

**REQ-OI-03:** Operation results shall be saved to paths provided by the task framework.

**REQ-OI-04:** Task shall pass infrastructure flags (`use_encryption`, `encryption_mode`, `use_dask`) to operations whose constructors accept these parameters.

## 6. Backward Compatibility

**REQ-BC-01:** The system shall maintain backward compatibility with existing task scripts where possible.

**REQ-BC-02:** The system shall provide clear deprecation warnings for legacy path usage patterns.

**REQ-BC-03:** The system shall support a transition period where both old and new path resolution methods work.

## 7. Implementation Requirements

### 7.1 TaskConfig Enhancements

**REQ-TE-01:** `TaskConfig` shall support variable substitution in configuration values.

**REQ-TE-02:** `TaskConfig` shall convert all relative paths to absolute during initialization.

**REQ-TE-03:** `TaskConfig` shall provide helper methods for accessing common paths.

**REQ-TE-04:** `TaskConfig` shall validate and sanitize all paths before access.

**REQ-TE-05:** `TaskConfig` shall log path resolution for debugging.

### 7.2 Error Handling

**REQ-EH-01:** The system shall provide specific exception types for path-related errors.

**REQ-EH-02:** Error messages shall include both the original path and the context of the failure.

**REQ-EH-03:** The system shall include suggested fixes in error messages where possible.

## 8. Testing Requirements

**REQ-TEST-01:** Unit tests shall verify path resolution in different execution contexts.

**REQ-TEST-02:** Unit tests shall verify configuration loading and cascade.

**REQ-TEST-03:** Unit tests shall verify helper methods for path access.

**REQ-TEST-04:** Integration tests shall verify task execution from various working directories.

**REQ-TEST-05:** Integration tests shall verify path resolution across task pipelines.

## 9. Documentation Requirements

**REQ-DOC-01:** The system shall provide documentation on the configuration format and structure.

**REQ-DOC-02:** The system shall provide documentation on the path resolution API.

**REQ-DOC-03:** The system shall provide migration guidelines for task developers.

**REQ-DOC-04:** The system shall provide examples of proper path handling in tasks.