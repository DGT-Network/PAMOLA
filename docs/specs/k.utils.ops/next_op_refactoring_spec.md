SRS: Enhancement of `pamola_core/utils/ops` Infrastructure

## 1. Purpose and Scope

**1.1 Purpose:** Define a clear, complete specification for extending the PAMOLA.CORE operations framework (`pamola_core/utils/ops`) to support unified data writing, reproducible configuration, and developer productivity tools, enabling AI‑driven code generation with minimal re‑explanation.

**1.2 Scope:**

- New modules: `op_data_writer.py`, `op_test_helpers.py`, `templates/`
- Updates to: `op_base.py`, `op_result.py`
- Existing modules (`op_data_source.py`, helpers, reader) are considered stable and *out of scope* for refactoring here.

## 2. Terminology & Definitions

| Term       | Definition                                                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Operation  | A unit of work (e.g., data profiling, generalization) that runs in its own `task_dir` and produces metrics, transformed data, and logs. |
| `task_dir` | Root directory for a single operation run, provided by the user script, containing subfolders for outputs, dictionaries, logs, etc.     |
| Artifact   | Any file produced by an operation (CSV, JSON, Parquet, metrics files, visualizations).                                                  |
| DataWriter | New utility class responsible for writing operation outputs in a consistent, encrypted, and timestamped way.                            |

## 3. Package Structure (`pamola_core/utils/ops`)

```
pamola_core/utils/ops/
  ├── op_base.py              # BaseOperation, lifecycle, logging, save_config
  ├── op_config.py            # OperationConfig, parameter validation
  ├── op_cache.py             # OperationCache
  ├── op_data_source.py       # DataSource (stable)
  ├── op_data_reader.py       # DataReader (stable)
  ├── op_registry.py          # OperationRegistry
  ├── op_result.py            # OperationResult, Artifact registration
  ├── op_data_writer.py       # (NEW) DataWriter
  ├── op_test_helpers.py      # (NEW) Testing stubs & fixtures
  ├── templates/              # (NEW) Operation skeletons and config examples
```

## 4. External Dependencies

- **Logging:** `pamola_core/utils/logging.py`
- **Progress:** `pamola_core/utils/progress.py`
- **I/O:** `pamola_core/utils/io.py`
- **Visualization:** `pamola_core/utils/visualization.py`

These modules must remain backwards-compatible; new code should import and use their public APIs without modification.

## 5. New Components

### 5.1 `op_data_writer.py`

- **Class:** `DataWriter`
- **Responsibilities:**
  - Write DataFrames, JSON objects, Parquet tables, and arbitrary artifacts
  - Enforce directory structure under `{task_dir}`:
    - Outputs → `{task_dir}/output/`
    - Dictionaries/Extracts → `{task_dir}/dictionaries/`
  - Support optional encryption (`encryption_key` parameter)
  - Generate timestamped filenames (e.g., `20250503T142530_mydata.csv`)
  - Validate write success and raise descriptive errors

### 5.2 `op_test_helpers.py`

- **Utilities for unit testing operations:**
  - `MockDataSource` returning in-memory DataFrames
  - `StubDataWriter` capturing writes in temp folders
  - Assert functions: `assert_artifact_exists()`, `assert_metrics_content()`
  - Helper to create a temporary `task_dir` and `OperationConfig`

### 5.3 `templates/`

- **Contents:**
  - `operation_skeleton.py`: Boilerplate for a new `MyOperation(BaseOperation)` with placeholders for `execute()`
  - `config_example.json`: Sample JSON config with schema annotations
  - `README.md`: Instructions for extending and testing operations

## 6. Modifications to Existing Modules

### 6.1 `op_base.py`

- **Add method** `save_config(self, task_dir: Path)`:
  - Serializes `self.config.to_dict()` to `{task_dir}/config.json`
  - Records operation name and version

### 6.2 `op_result.py`

- **Enhance** `OperationResult`:
  - Provide `register_artifact_via_writer(writer: DataWriter, obj, subdir: str, name: str)` helper to delegate file writing
  - Deprecate direct calls to `open()/to_csv()` in favor of `DataWriter`

## 7. Detailed Requirements

| ID          | Description                                                                                                                                       | Priority |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| REQ-OPS-001 | `DataWriter.write_dataframe(df, name, task_dir, encryption_key=None)` writes CSV under `task_dir/output/` with timestamp and returns full path.   | High     |
| REQ-OPS-002 | `DataWriter.write_json(obj, name, task_dir)` writes `.json` under `task_dir/` and validates JSON schema (if provided).                            | High     |
| REQ-OPS-003 | `DataWriter.write_parquet(df, name, task_dir, **opts)` writes Parquet with optional compression.                                                  | Medium   |
| REQ-OPS-004 | `BaseOperation.save_config(task_dir)` writes `config.json` atomically before execution begins.                                                    | High     |
| REQ-OPS-005 | `OperationResult.register_artifact_via_writer()` uses `DataWriter` so operation code no longer directly writes files.                             | Medium   |
| REQ-OPS-006 | Test helpers must allow instantiation of any `BaseOperation` subclass without real I/O or encryption, and verify expected artifacts are recorded. | Medium   |
| REQ-OPS-007 | Templates must include comments indicating where to inject business logic, import patterns, and required dependency references.                   | Low      |

## 8. Design Decisions and Clarifications

Below are specific design decisions for `op_data_writer.py` based on stakeholder feedback:

1. **Encryption Support**

   - Controlled by explicit parameter `encrypt: bool` or `encryption_key`. No automatic sensitivity detection.

2. **Overwrite Behavior**

   - Default `overwrite = true`. To preserve old files, operations may pass `timestamp_in_name = true`.

3. **Progress Tracking**

   - `DataWriter` accepts an external progress tracker. If none provided, uses a no-op stub.

4. **Return Values**

   - Methods return a `WriterResult(path: Path, size_bytes: int, timestamp: datetime, format: str)` structure for metadata extensibility.

5. **Directory Creation**

   - `DataWriter` creates required subdirectories (`output`, `dictionaries`, `logs`) if they do not exist.

6. **Dask Support**

   - Automatically handles Dask DataFrames by partitioned `to_parquet` or `to_csv`. Returns list of paths or root directory.

7. **Logging**

   - Accepts a `logger` instance; defaults to `logging.getLogger(__name__)`. Logs INFO/ERROR for start, completion, and errors.

8. **Format Specification**

   - Format is always explicit via `format` parameter (`csv`, `json`, `parquet`, etc.). No auto-detection.

9. **Error Handling**

   - Raises a `DataWriteError` on failure with detailed message and traceback.

10. **Artifact Registration**

    - `DataWriter` does not auto-register artifacts. Caller must use `OperationResult.register_artifact(...)`.

## 9. Test Helpers Design Decisions

These clarifications guide the implementation of `op_test_helpers.py` (§5.2):

1. **MockDataSource Scope**  
   - **Supported Methods:** Only pamola core methods required by operations tests, such as `read()` and `get_schema()`. No need to implement the full `DataSource` API (e.g., chunked reads) unless specifically tested.  
   - **Error Simulation:** Do not simulate I/O failures by default; error paths can be added later if a specific operation requires testing of failure modes.

2. **StubDataWriter Behavior**  
   - **Actual Writes vs. Recording:** It should perform real writes into a temporary directory (using `tempfile.TemporaryDirectory`) to validate file system interactions.  
   - **Call Tracking:** Internally record each call’s parameters and return values in a list for assertions (e.g., `self.calls.append(...)`).

3. **Assertion Helpers**  
   - **`assert_metrics_content` Matching:** Default to **partial matching**—verify that all keys in `expected_metrics` exist and match, while allowing extra fields.  
   - **`assert_artifact_exists` Checking:** Only verify that the file exists at the expected path; content checks may be done separately by the test.

4. **`create_test_operation_env` Initialization**  
   - **Minimal `OperationConfig` Fields:** Include required schema fields such as `operation_name`, `version`, and any operation-specific parameters used by tests.  
   - **Directory Setup:** Create standard subdirectories under `task_dir` (`output`, `dictionaries`, `logs`) so that tests can write into these paths without errors.

## 10. Out of Scope

- Unified **reporting** abstraction (e.g., summary HTML/Markdown) — handled by user-level scripts
- Multi-task orchestration and JSON report generation — beyond single-operation focus

## 11. Modernization Approach Decisions

Below are explicit answers to the previously posed questions. Any future adjustments should be marked with `# TODO` in the module header.

1. **Class-Based vs Functional Style (op_registry.py)**  
   **Answer:** Retain the existing functional API (`register_operation()`, `get_operation()` etc.) with global functions and registry dictionaries. This minimizes changes to existing operations. For future major releases, a thin `OperationRegistry` wrapper class may be introduced via deprecation shim.  

2. **Global `operation_cache` Instance (op_cache.py)**  
   **Answer:** Preserve the global `operation_cache = OperationCache()` for backward compatibility and minimal refactoring. Document how users can instantiate custom `OperationCache` if needed.  

3. **JSON-Schema Validation (op_config.py)**  
   **Answer:** Move direct `jsonschema.validate` calls into a new helper in `pamola_core.utils.io.validate_json_schema(schema, data)`. In `op_config.py`, call this helper. This unifies validation without removing existing schema logic.  

4. **Artifact Validation & Checksum (op_result.py)**  
   **Answer:** Keep existing filesystem logic in `OperationArtifact` temporarily, but annotate with `# TODO: delegate metadata extraction (checksum, size) to pamola_core.utils.io.get_file_metadata()`. Full extraction to `io` helpers can follow in later refactoring.  

5. **Req-ID in Docstrings**  
   **Answer:** Include `REQ-OPS-00X` only on high- and medium-priority public methods. Reference remaining SRS items at the module-level docstring.  

6. **Error Class Standardization**  
   **Answer:** Introduce a base `OpsError(Exception)` for all new errors. Existing errors (e.g., `ConfigSaveError`, `DataWriteError`) may temporarily remain, but must subclass `OpsError` going forward.  

7. **Type Annotations**  
   **Answer:** Annotate all public methods with full type hints (`-> None`, `-> WriterResult`, etc.). Private/internal methods may be annotated later as needed.  

8. **Performance Considerations (op_cache.py)**  
   **Answer:** Retain synchronous API for `OperationCache`. If thread safety is required, add a simple `threading.Lock` inside `OperationCache` methods. No async support at this stage.

---

*All future implementation details and minor refactorings should be annotated in module headers with `# TODO` comments to track deferred work.*

