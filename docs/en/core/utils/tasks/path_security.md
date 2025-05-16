# PathSecurity Module Documentation

## Overview

The `path_security.py` module provides robust security validation for file paths in the PAMOLA Core framework. It helps prevent directory traversal attacks, access to system directories, and other path-based security vulnerabilities. This module is critical for ensuring that file operations within PAMOLA Core are performed in a secure manner, reducing the risk of path-based attacks or accidental access to sensitive system files.

## Key Features

- **Path Traversal Prevention**: Blocks attempts to use parent directory traversal (`..`)
- **System Directory Protection**: Prevents access to sensitive system directories
- **External Path Validation**: Controls access to paths outside allowed directories
- **Symbolic Link Validation**: Checks for symbolic links that might bypass restrictions
- **Flexible Strictness**: Configurable strictness for different security requirements
- **Platform-Specific Protection**: System-specific path security checks for Windows, Linux, and macOS
- **Batch Validation**: Support for validating multiple paths at once
- **Path Normalization**: Standardizes and validates paths for consistent handling

## Dependencies

- `logging`: Standard library module
- `platform`: For system-specific checks
- `pathlib.Path`: For path manipulation

## Exception Classes

- **PathSecurityError**: Exception raised for path security violations

## Key Functions

### validate_path_security

```python
def validate_path_security(
    path: Union[str, Path],
    allowed_paths: Optional[List[Union[str, Path]]] = None,
    allow_external: bool = False,
    strict_mode: bool = True
) -> bool
```

Validate that a path is safe to use. This function checks that a path doesn't contain potentially dangerous components like path traversal sequences.

**Parameters:**
- `path`: Path to validate
- `allowed_paths`: List of allowed external paths (absolute)
- `allow_external`: Whether to allow external paths outside data repository
- `strict_mode`: If True, raises PathSecurityError for unsafe paths

**Returns:**
- True if the path is safe, False otherwise

**Raises:**
- `PathSecurityError`: If the path is unsafe and strict_mode is True

### is_within_allowed_paths

```python
def is_within_allowed_paths(
    path: Path,
    allowed_paths: List[Union[str, Path]]
) -> bool
```

Check if a path is within any of the allowed paths.

**Parameters:**
- `path`: Path to check
- `allowed_paths`: List of allowed parent paths

**Returns:**
- True if path is within any allowed path, False otherwise

### get_system_specific_dangerous_paths

```python
def get_system_specific_dangerous_paths() -> List[str]
```

Get a list of system-specific paths that should be protected.

**Returns:**
- List of dangerous system paths for the current operating system

### validate_paths

```python
def validate_paths(
    paths: List[Union[str, Path]],
    allowed_paths: Optional[List[Union[str, Path]]] = None,
    allow_external: bool = False
) -> Tuple[bool, List[str]]
```

Validate multiple paths at once.

**Parameters:**
- `paths`: List of paths to validate
- `allowed_paths`: List of allowed external paths
- `allow_external`: Whether to allow external paths

**Returns:**
- Tuple containing:
  - Boolean indicating if all paths are valid
  - List of error messages for invalid paths

### is_potentially_dangerous_path

```python
def is_potentially_dangerous_path(path: Union[str, Path]) -> bool
```

Check if a path might be potentially dangerous without raising exceptions. This is a convenience method for quick checks without stopping execution.

**Parameters:**
- `path`: Path to check

**Returns:**
- True if path might be dangerous, False if likely safe

### normalize_and_validate_path

```python
def normalize_and_validate_path(
    path: Union[str, Path],
    base_dir: Optional[Path] = None,
    allowed_paths: Optional[List[Union[str, Path]]] = None,
    allow_external: bool = False
) -> Path
```

Normalize a path and validate its security. If the path is relative, it will be resolved against the base_dir.

**Parameters:**
- `path`: Path to normalize and validate
- `base_dir`: Base directory for resolving relative paths
- `allowed_paths`: List of allowed external paths
- `allow_external`: Whether to allow external paths

**Returns:**
- Normalized path object

**Raises:**
- `PathSecurityError`: If the path fails security validation

## Security Checks

The module implements several layers of security checks:

### 1. Dangerous Pattern Detection

The following patterns are detected and flagged as potentially dangerous:

- `..`: Parent directory traversal
- `~`: Home directory reference
- `|`: Command chaining (Windows)
- `;`: Command chaining (Unix)
- `&`: Command chaining
- `$`: Variable substitution
- `` ` ``: Command substitution
- `\\x`: Hex escape
- `\\u`: Unicode escape

### 2. System Directory Protection

The module checks for attempts to access system directories, with platform-specific checks:

#### Windows System Directories:
- `C:\Windows`
- `C:\Program Files`
- `C:\Program Files (x86)`
- `C:\Users\Default`
- `C:\ProgramData`
- `C:\System Volume Information`
- `C:\$Recycle.Bin`

#### Unix System Directories (Linux/macOS):
- `/bin`
- `/sbin`
- `/etc`
- `/dev`
- `/sys`
- `/proc`
- `/boot`
- `/lib`
- `/lib64`
- `/usr/bin`
- `/usr/sbin`
- `/usr/lib`
- `/var/run`
- `/var/lock`

### 3. Symbolic Link Validation

The module checks if a path contains symbolic links that might lead outside allowed paths by resolving the path and comparing it to the original.

### 4. External Path Validation

When `allow_external` is False, the module ensures that paths are within the allowed directories specified in `allowed_paths`.

## Usage Examples

### Basic Path Validation

```python
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError
from pathlib import Path

# Validate a safe path
try:
    is_safe = validate_path_security(
        path="data/processed/t_1P1/output/results.csv",
        strict_mode=True
    )
    print(f"Path is safe: {is_safe}")
except PathSecurityError as e:
    print(f"Path security error: {e}")

# Validate an unsafe path
try:
    is_safe = validate_path_security(
        path="../../../etc/passwd",
        strict_mode=True
    )
    print(f"Path is safe: {is_safe}")
except PathSecurityError as e:
    print(f"Path security error: {e}")
```

### Path Validation with Allowed Paths

```python
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError
from pathlib import Path

# Define allowed paths
allowed_paths = [
    Path("/data/repository"),
    Path("/opt/external/data")
]

# Validate a path within allowed paths
try:
    is_safe = validate_path_security(
        path="/data/repository/processed/t_1P1/output/results.csv",
        allowed_paths=allowed_paths,
        allow_external=False,
        strict_mode=True
    )
    print(f"Path is safe: {is_safe}")
except PathSecurityError as e:
    print(f"Path security error: {e}")

# Validate a path outside allowed paths
try:
    is_safe = validate_path_security(
        path="/home/user/data.csv",
        allowed_paths=allowed_paths,
        allow_external=False,
        strict_mode=True
    )
    print(f"Path is safe: {is_safe}")
except PathSecurityError as e:
    print(f"Path security error: {e}")
```

### Batch Path Validation

```python
from pamola_core.utils.tasks.path_security import validate_paths
from pathlib import Path

# List of paths to validate
paths = [
    "data/processed/t_1P1/output/results.csv",
    "data/raw/input.csv",
    "../../../etc/passwd",
    "C:/Windows/System32/config/SAM"
]

# Validate all paths
all_valid, errors = validate_paths(
    paths=paths,
    allowed_paths=[Path("data")],
    allow_external=False
)

if all_valid:
    print("All paths are valid")
else:
    print("Some paths are invalid:")
    for error in errors:
        print(f"  - {error}")
```

### Non-Strict Path Checking

```python
from pamola_core.utils.tasks.path_security import validate_path_security
from pathlib import Path

# Check a potentially dangerous path without raising an exception
is_safe = validate_path_security(
    path="/etc/passwd",
    strict_mode=False
)

if is_safe:
    print("Path is considered safe")
else:
    print("Path is potentially dangerous, but operation continues")
```

### Path Normalization and Validation

```python
from pamola_core.utils.tasks.path_security import normalize_and_validate_path, PathSecurityError
from pathlib import Path

# Base directory for relative paths
base_dir = Path("/data/project")

# Normalize and validate a relative path
try:
    normalized_path = normalize_and_validate_path(
        path="processed/results.csv",
        base_dir=base_dir,
        allow_external=False
    )
    print(f"Normalized path: {normalized_path}")
except PathSecurityError as e:
    print(f"Path security error: {e}")

# Try to normalize a dangerous path
try:
    normalized_path = normalize_and_validate_path(
        path="../../etc/passwd",
        base_dir=base_dir,
        allow_external=False
    )
    print(f"Normalized path: {normalized_path}")
except PathSecurityError as e:
    print(f"Path security error: {e}")
```

### Quick Check without Exception Handling

```python
from pamola_core.utils.tasks.path_security import is_potentially_dangerous_path
from pathlib import Path

# Check paths quickly
paths = [
    "data/processed/output.csv",
    "../config/settings.ini",
    "~/.ssh/id_rsa",
    "/etc/passwd"
]

for path in paths:
    if is_potentially_dangerous_path(path):
        print(f"WARNING: {path} might be dangerous")
    else:
        print(f"{path} appears safe")
```

## Integration with Other Modules

The `path_security.py` module is designed to integrate with other PAMOLA Core modules:

### Integration with TaskDirectoryManager

```python
# In TaskDirectoryManager
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError

def normalize_and_validate_path(self, path: Union[str, Path]) -> Path:
    """Normalize a path and validate its security."""
    try:
        path_obj = Path(path) if isinstance(path, str) else path
        
        # If path is relative, resolve against task directory
        if not path_obj.is_absolute():
            path_obj = self.task_dir / path_obj
            
        # Validate path security
        if not validate_path_security(
            path=path_obj,
            allowed_paths=[self.project_root, self.data_repository],
            allow_external=getattr(self.config, 'allow_external_paths', False)
        ):
            raise PathSecurityError(f"Path failed security validation: {path_obj}")
            
        return path_obj
        
    except Exception as e:
        self.logger.error(f"Path validation error: {e}")
        raise
```

### Integration with ProjectConfigLoader

```python
# In ProjectConfigLoader
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError

def _resolve_key_path(self) -> Path:
    """Resolve the encryption key path safely."""
    try:
        key_path = self.config.get('encryption', {}).get('key_path')
        if not key_path:
            return None
            
        key_path_obj = Path(key_path)
        
        # If relative, resolve against project root
        if not key_path_obj.is_absolute():
            key_path_obj = self.project_root / key_path_obj
            
        # Validate path security
        if not validate_path_security(
            path=key_path_obj,
            allowed_paths=[self.project_root],
            allow_external=False
        ):
            raise PathSecurityError(f"Key path failed security validation: {key_path_obj}")
            
        return key_path_obj
        
    except Exception as e:
        self.logger.error(f"Key path resolution error: {e}")
        raise
```

### Integration with BaseTask

```python
# In BaseTask
from pamola_core.utils.tasks.path_security import PathSecurityError

def load_dataset(self, dataset_path: Union[str, Path]) -> Any:
    """Load a dataset securely."""
    try:
        # Use directory manager to resolve and validate path
        validated_path = self.directory_manager.normalize_and_validate_path(dataset_path)
        
        # Now we can safely load the dataset
        return self.data_source.load_from_path(validated_path)
        
    except PathSecurityError as e:
        self.logger.error(f"Path security error when loading dataset: {e}")
        raise
```

## Security Best Practices

When using the path security module, follow these best practices:

1. **Always Validate User Input**: Any path that comes from user input should be validated

2. **Use Strict Mode**: Use strict_mode=True (default) in production to raise exceptions for invalid paths

3. **Define Allowed Paths**: Always provide a list of allowed paths when validating external paths

4. **Prefer Relative Paths**: Use relative paths when possible and resolve them against safe base directories

5. **Check Symbolic Links**: Be aware that symbolic links can bypass directory restrictions

6. **Handle Exceptions**: Always catch and handle PathSecurityError exceptions for better user experience

7. **Log Security Violations**: Log path security violations for audit and debugging

8. **Validate Before I/O**: Always validate paths before performing file I/O operations

9. **Use normalize_and_validate_path**: Use this helper function to both normalize and validate paths in one step

10. **Platform-Specific Awareness**: Be aware of platform-specific path issues (e.g., case sensitivity)

## Common Security Risks

The module helps protect against these common security risks:

1. **Directory Traversal Attacks**: Attempts to access files outside the intended directory using `../` sequences

2. **Absolute Path Attacks**: Attempts to directly reference sensitive system files using absolute paths

3. **Symbolic Link Attacks**: Using symbolic links to reference files outside the allowed directories

4. **Command Injection**: Using special characters to inject commands via path names

5. **Environment Variable Expansion**: Using environment variables in paths to access unintended locations

6. **Unicode Normalization Attacks**: Using different Unicode representations of the same character to bypass filters

7. **Relative Path Confusion**: Misinterpreting relative paths due to unclear base directory contexts

## Limitations and Considerations

1. **Performance Impact**: Path validation adds some performance overhead

2. **False Positives**: Some legitimate paths might be flagged as dangerous

3. **System-Specific Paths**: The list of system paths is not exhaustive and may vary by system

4. **Race Conditions**: Path validation is subject to race conditions if files change between checks and operations

5. **Symlink Resolution**: Some complex symlink scenarios might not be fully detected

6. **Case Sensitivity**: Path validation behavior may differ on case-sensitive vs. case-insensitive file systems

7. **Platform Differences**: Path handling differs between operating systems

## Security Testing Guidelines

To test the security of your path handling:

1. **Test Path Traversal**: Attempt to access files outside allowed directories using `../` sequences

2. **Test Absolute Paths**: Try to access system directories using absolute paths

3. **Test Special Characters**: Include special characters in paths to check for command injection

4. **Test Unicode Characters**: Use different Unicode representations of characters

5. **Test Symbolic Links**: Create symbolic links to files outside allowed directories

6. **Test Edge Cases**: Test empty paths, very long paths, and paths with unusual characters

7. **Test Cross-Platform**: Test path validation on different operating systems

8. **Fuzzing**: Use automated fuzzing to find unexpected edge cases