### 📦 Function Redistribution Table

| **Functionality Area**              | **Assigned To io.py**                                                | **Assigned To DataSource**                                                  | **Assigned To Helpers**                                                   |
|-------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Basic file read/write (CSV, Parquet, Excel, JSON, Text) | ✅ Full responsibility (all formats, including encryption integration) | ❌ Only routes calls to io.py, never reads directly                        | ❌ Only light wrappers, no I/O logic                                        |
| Encryption handling                 | ✅ Fully implemented (via io_helpers.crypto_utils)                    | ❌ Only passes encryption_key when calling io.py                           | ❌ Thin adapters if format-specific, but generally stay in io.py            |
| Chunked reading                     | ✅ Provided via io_helpers.chunk_utils or pandas/dask                 | ❌ Only calls io.py or helper method to get generator                      | ✅ Manage chunk caching or tracking, but no I/O                            |
| Schema building and validation      | ❌ Only exposes raw metadata if needed                               | ✅ Coordinates schema requests, caches schema info                         | ✅ Implements schema checks, dtype comparisons, compatibility logic         |
| Multi-file dataset merging          | ✅ Provides underlying file reads                                     | ✅ Controls batch coordination, merging strategies                         | ✅ Provides batch helpers, memory estimators, batching logic                |
| Progress tracking                   | ✅ Integrated via io_helpers or pamola_core.utils.progress                   | ❌ Only triggers progress events when needed                               | ✅ Manages detailed progress event steps, but reuses io.py events          |
| Memory estimation/management        | ✅ Supplies file-level memory hints                                   | ✅ Decides batching or fallback strategies based on memory constraints     | ✅ Calculates detailed estimates, recommends splits, manages memory limits  |
| Error handling                      | ✅ Provides standardized error objects from I/O level                 | ✅ Maps or wraps errors into DataSource-level abstractions                 | ✅ Offers utility functions for detailed error reporting                   |
| Public API                          | ✅ Focused on file-level I/O (no DataSource abstractions)             | ✅ Provides clear methods: get_dataframe, add_file_path, add_dataframe, etc. | ❌ Only offers internal support functions, not public API                  |

---

### 🔧 Refactoring Recommendations

1️⃣ **Eliminate duplication**
- Remove direct file reading logic from DataSource; ensure all `.read_*` operations call io.py.
- In helpers (like file_helpers), remove embedded read logic and redirect to io.py.
- Use io.py encryption support consistently; helpers should not implement their own crypto handling.

2️⃣ **Define clear responsibilities**
- io.py → Owns all I/O, format parsing, encryption, chunking, progress tracking.
- DataSource → Manages what to load, when, from where; maintains DataFrame objects, caches, metadata.
- Helpers → Provide specialized calculations (schema, memory, batching) but do not perform raw I/O.

3️⃣ **Reduce DataSource size and complexity**
- Strip down to essential coordination methods:
    - add_dataframe, add_file_path
    - get_dataframe, get_dataframe_chunks
    - get_schema, validate_schema
    - release_dataframe, get_large_dataframe
    - add_multi_file_dataset (calls io + helpers)
- Move detailed logic (chunking, schema checking, error formatting) into helpers or io.py.

4️⃣ **Provide a minimal, strong public API**
- Keep DataSource focused on: unified access to data, caching, and high-level coordination.
- Let io.py remain the authoritative source for reading, writing, encryption, and file-specific behaviors.
- Ensure helpers stay internal (hidden from external users) and serve only to keep DataSource clean.

---

### ✅ Final Notes

This approach ensures:
- Consistency across all data operations.
- Minimal risk of duplicate logic or diverging behaviors.
- Better maintainability and testability (each layer has a narrow, clear scope).
- Faster onboarding for developers using the public API.

Once approved, I can help prepare a detailed SRS or refactoring plan with milestones.

