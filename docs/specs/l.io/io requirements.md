### Refactoring Requirements for `pamola_core.utils.io.py` and `io_helpers` (PAMOLA Project)

---

### Architectural Priorities

- Maintain a strong facade: `pamola_core.utils.io.py` remains the public entry point, delegating complex logic to `io_helpers`.
- Integrate fully with the updated cryptographic subsystem (`pamola_core.utils.io_helpers.crypto_utils`).
- Preserve existing public APIs for backward compatibility.
- Clearly separate phase 1 (MVP) and phase 2 (future) improvements.
- Minimize unnecessary complexity; focus only on what directly supports PET tasks.

---

### Requirements Table with Priorities

| Category                          | Requirement                                                                                           | Priority    |
|-----------------------------------|------------------------------------------------------------------------------------------------------|-------------|
| **Crypto Integration**            | Switch all encryption/decryption calls to the new `pamola_core.utils.io_helpers.crypto_utils`.               | Phase 1 ✅  |
|                                   | Remove duplication or inconsistencies across modules.                                                | Phase 1 ✅  |
|                                   | Preserve public APIs to ensure backward compatibility.                                               | Phase 1 ✅  |
|                                   | Improve code comments and docstrings for new crypto modes (none, simple, age).                       | Phase 1 ✅  |
|                                   | Unit-test crypto integration across formats (CSV, JSON, Parquet).                                    | Phase 1 ✅  |
|                                   | TODO: Add streaming (age) encryption for large binary formats.                                       | Phase 2 ⚠️ |
|                                   | TODO: Support cloud storage + encryption pipelines.                                                  | Phase 2 ⚠️ |
|                                   | TODO: Implement key management (key rotation, audit trails).                                         | Phase 2 ⚠️ |
|                                   |                                                                                                      |             |
| **File Read/Write Operations**    | Add universal support for options: `columns`, `nrows`, `skiprows` (where applicable).                | Phase 1 ✅  |
|                                   | Implement multi-file CSV support (`read_multi_csv`) as a dedicated helper module.                    | Phase 1 ✅  |
|                                   | TODO: Support parallel or batched multi-part Parquet processing.                                     | Phase 2 ⚠️ |
|                                   | TODO: Leave full streaming implementation as a backlog item (not needed for PET tasks).              | Phase 2 ⚠️ |
|                                   |                                                                                                      |             |
| **Progress and Logging**          | Maintain or enhance integration with `progress` tracking.                                             | Phase 1 ✅  |
|                                   | Ensure compatibility with `pamola_core.utils.logging`; add contextual markers (e.g., TaskID).               | Phase 1 ✅  |
|                                   | TODO: Add memory monitor (e.g., using psutil) for large operations.                                 | Phase 2 ⚠️ |
|                                   |                                                                                                      |             |
| **Metadata and Schema**           | TODO: Add a `get_file_metadata` API for all supported formats.                                       | Phase 2 ⚠️ |
|                                   | TODO: Implement schema validation against expected schemas.                                          | Phase 2 ⚠️ |
|                                   |                                                                                                      |             |
| **Error Handling**                | Introduce a centralized error handler decorator for all I/O functions.                               | Phase 1 ✅  |
|                                   | Standardize success/warning/error report objects for downstream use.                                 | Phase 1 ✅  |
|                                   | TODO: Expand fallback strategies (e.g., fallback to pandas if Dask fails).                          | Phase 2 ⚠️ |
|                                   |                                                                                                      |             |
| **Documentation and Testing**     | Update docstrings for all affected functions.                                                       | Phase 1 ✅  |
|                                   | Write or update unit tests for all new/changed functions.                                           | Phase 1 ✅  |
|                                   | Prepare a developer-facing API doc with examples and usage notes.                                   | Phase 1 ✅  |
|                                   | TODO: Prepare extended architectural documentation with diagrams and flowcharts.                    | Phase 2 ⚠️ |

---

### Development Principles

✅ Do not break existing APIs unless strictly necessary.
✅ Only add features essential for MVP PET operations.
✅ Document and test all phase 1 changes.
✅ Defer lower-priority or more experimental features to phase 2 or backlog.

---

### Next Step

Prepare a detailed SRS (Software Requirements Specification) draft based on this table for approval before implementation.

