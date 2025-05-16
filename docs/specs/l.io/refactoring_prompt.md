# PAMOLA.CORE IO Module Refactoring Project

## Project Context

I'm working on refactoring the PAMOLA.CORE `io.py` module and its associated helpers to enhance functionality, integrate with a new cryptography subsystem, and improve code organization. This is a critical infrastructure component that handles all file operations for the privacy-preserving AI data processing framework.

## What I Need Help With

I need assistance refactoring the existing code to meet the requirements in the attached SRS. The work involves:

1. Updating the public API in `io.py` to add new parameters while maintaining backward compatibility
2. Creating/updating helper modules in the `io_helpers` package
3. Integrating with the existing crypto system
4. Adding new functionality for selective loading, multi-file processing, etc.
5. Implementing standardized error handling and memory management

## Key Files To Review

I'm attaching the following files:

1. `io.py` - The main module to be refactored
2. The `io_helpers` package files:
   - `crypto_utils.py` (DO NOT MODIFY - already updated)
   - `csv_utils.py` 
   - `dask_utils.py`
   - `directory_utils.py`
   - `format_utils.py`
   - `image_utils.py`
   - `json_utils.py`
3. `crypto.md` - Documentation on the new cryptography subsystem

## Important Notes Not Covered in SRS

1. **Import Structure**: 
   - Do not use wildcard imports (`from x import *`)
   - Prefer explicit imports and group them logically
   - The import structure in the current files should be maintained where possible

2. **Exception Handling**:
   - Catch specific exceptions rather than broad `Exception` where possible
   - Log exceptions before re-raising them
   - Use consistent exception naming (e.g., `FileReadError`, `DecryptionError`)

3. **Logging**:
   - All functions should have appropriate debug/info/error logging
   - Log entry/exit for major operations
   - Include relevant context in log messages (file paths, operation types)

4. **Memory Management**:
   - Be careful with large file operations
   - Use generators and iterators where appropriate
   - Clear variables that are no longer needed

5. **Naming Conventions**:
   - Private helper functions should start with underscore
   - Constants should be UPPER_CASE
   - Class names should be CamelCase
   - Function and variable names should be snake_case

6. **Implementation Order**:
   - Start with the basic error_utils.py module
   - Then move to the pamola core functionality in io.py
   - Finally implement the specialized helpers

## Expectations for Code Quality

1. **Readability** is a primary concern - code should be clear and well-documented
2. **Type hints** must be comprehensive
3. **Docstrings** should follow Google style and include examples where helpful
4. **Variable names** should be descriptive and meaningful
5. **Comments** should explain "why" not "what"
6. **Error messages** should be actionable and user-friendly

## Deliverables

Please generate the refactored code according to the SRS, addressing the points above. For each file, provide:

1. The complete refactored code
2. A brief summary of changes made
3. Any implementation notes or concerns

Let's start with the `error_utils.py` module as it will be used by many other components.