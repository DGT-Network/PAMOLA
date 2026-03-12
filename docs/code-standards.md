# PAMOLA.CORE Code Standards

**Version:** 0.1.0
**Last Updated:** 2026-03-12

## Overview

This document defines the coding standards, conventions, and best practices for PAMOLA.CORE development. All contributors must follow these standards to ensure code quality, maintainability, and consistency.

## Core Principles

- **YAGNI** (You Aren't Gonna Need It): Implement only what's needed
- **KISS** (Keep It Simple, Stupid): Favor simple solutions
- **DRY** (Don't Repeat Yourself): Avoid code duplication
- **Privacy-First**: All operations designed with privacy preservation in mind

## Python Coding Conventions

### Style Guide

**PEP 8 Compliance:**
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters (soft), 120 (hard)
- Use `snake_case` for functions, variables, and modules
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants
- Import order: standard library → third-party → local imports

**Example:**
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party
import pandas as pd
from pydantic import BaseModel, Field

# Local
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.helpers import generate_data_hash
```

### Type Hints

**Required Type Annotations:**
- All function signatures must include type hints
- Use `DataFrameType` alias for Union[pd.DataFrame, dd.DataFrame]
- Use `Optional` for nullable parameters
- Use `List`, `Dict`, `Tuple` for collections

**Example:**
```python
from typing import Dict, List, Optional, Union
from pamola_core.common.type_aliases import DataFrameType

def process_data(
    data: DataFrameType,
    fields: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> DataFrameType:
    """Process data with specified fields."""
    if config is None:
        config = {}
    # Implementation
    return data
```

### Documentation Strings

**Google Style Docstrings:**
```python
def mask_email(email: str, mask_char: str = "*") -> str:
    """Mask an email address for privacy preservation.

    This function masks the local part of an email address while preserving
    the domain part. The masking preserves the format for usability.

    Args:
        email: The email address to mask.
        mask_char: The character to use for masking (default: "*").

    Returns:
        The masked email address.

    Raises:
        ValueError: If email is invalid or empty.

    Example:
        >>> mask_email("john.doe@example.com")
        'j***@example.com'
    """
    # Implementation
    pass
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | `snake_case` | `masking_operations.py` |
| Classes | `PascalCase` | `MaskingOperation` |
| Functions | `snake_case` | `mask_email_address` |
| Variables | `snake_case` | `masked_email` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_MASK_CHAR` |
| Private | `_leading_underscore` | `_internal_method` |
| Protected | `_leading_underscore` | `_protected_method` |

## Architecture Patterns

### Operation-Based Framework

**All Operations Must:**
1. Inherit from `BaseOperation` or domain-specific base
2. Use Pydantic schemas for configuration
3. Support both pandas and Dask DataFrames
4. Return `OperationResult` with status, artifacts, metrics
5. Register in `OperationRegistry`

**Template:**
```python
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_registry import register_operation
from pydantic import BaseModel, Field

class MyCustomConfig(BaseModel):
    """Configuration for MyCustomOperation."""

    param1: str = Field(default="value", description="Parameter 1")
    param2: int = Field(default=10, ge=0, le=100, description="Parameter 2")

@register_operation
class MyCustomOperation(BaseOperation):
    """Custom operation for specific use case."""

    def __init__(self, config: MyCustomConfig):
        super().__init__(config)
        self.param1 = config.param1
        self.param2 = config.param2

    def _validate_input(self, data: DataFrameType) -> None:
        """Validate input data."""
        # Validation logic
        pass

    def _execute(self, data: DataFrameType) -> DataFrameType:
        """Execute the operation."""
        # Implementation
        return data

    def _finalize(self, result: OperationResult) -> OperationResult:
        """Finalize the operation."""
        # Add metrics, artifacts
        return result
```

### Schema-Driven Configuration

**Pydantic Schema Requirements:**
- All configuration must use Pydantic `BaseModel`
- Use `Field()` for metadata (description, default, constraints)
- Use validators for complex validation logic
- Support JSON serialization

**Example:**
```python
from pydantic import BaseModel, Field, field_validator

class MaskingConfig(BaseModel):
    """Configuration for masking operations."""

    mask_char: str = Field(
        default="*",
        min_length=1,
        max_length=1,
        description="Character to use for masking"
    )

    preserve_format: bool = Field(
        default=True,
        description="Preserve original format (e.g., dashes in phone)"
    )

    fields: List[str] = Field(
        default_factory=list,
        description="List of fields to mask"
    )

    @field_validator("mask_char")
    @classmethod
    def validate_mask_char(cls, v: str) -> str:
        """Validate mask character is safe."""
        if v in (" ", "\t", "\n"):
            raise ValueError("Whitespace characters not allowed as mask char")
        return v
```

### Error Handling

**Principles:**
- Use specific exception types
- Provide clear error messages
- Log errors with context
- Handle edge cases gracefully

**Example:**
```python
from pamola_core.utils import logging

logger = logging.get_logger(__name__)

class MaskingError(Exception):
    """Base exception for masking operations."""
    pass

def mask_value(value: str, mask_char: str = "*") -> str:
    """Mask a string value."""
    try:
        if not value:
            raise ValueError("Cannot mask empty value")
        if not mask_char:
            raise ValueError("Mask character cannot be empty")

        # Implementation
        return masked_value

    except ValueError as e:
        logger.error(f"Masking failed for value '{value}': {e}")
        raise MaskingError(f"Failed to mask value: {e}") from e
```

### Logging

**Logging Standards:**
- Use `pamola_core.utils.logging.get_logger(__name__)`
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include context in log messages
- Log operation start/end, parameters, results

**Example:**
```python
from pamola_core.utils import logging

logger = logging.get_logger(__name__)

def process_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Process dataset with logging."""
    logger.info(f"Processing dataset with {len(data)} records")

    try:
        result = data.pipe(transform_step1).pipe(transform_step2)
        logger.info(f"Successfully processed {len(result)} records")
        return result

    except Exception as e:
        logger.error(f"Failed to process dataset: {e}", exc_info=True)
        raise
```

## Testing Standards

### Test Organization

**Test Structure:**
```python
# tests/anonymization/masking/test_masking_op.py

import pytest
import pandas as pd
from pamola_core.anonymization.masking import MaskingOperation
from pamola_core.anonymization.schemas import MaskingConfig

class TestMaskingOperation:
    """Test suite for MaskingOperation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            "name": ["John Doe", "Jane Smith"],
            "email": ["john@example.com", "jane@example.com"]
        })

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MaskingConfig(
            fields=["name", "email"],
            mask_char="*"
        )

    def test_full_masking(self, sample_data, config):
        """Test full masking operation."""
        # Arrange
        op = MaskingOperation(config)

        # Act
        result = op.execute(sample_data)

        # Assert
        assert result.status == OperationStatus.SUCCESS
        assert "*" in result.data["name"][0]
```

### Testing Patterns

**1. Internal Logic Testing:**
```python
def test_internal_method_directly():
    """Test _internal_method without full pipeline."""
    op = MaskingOperation(config)
    result = op._mask_single_value("test@email.com")
    assert result == "t***@email.com"
```

**2. Disposable Resource Fixtures:**
```python
import tempfile
import os

@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("name,email\nJohn,john@example.com\n")
        temp_path = f.name

    yield temp_path

    os.unlink(temp_path)
```

**3. Strategy Injection Matrix:**
```python
@pytest.mark.parametrize("strategy,expected", [
    ("full", "j***@example.com"),
    ("partial", "j***@e***.com"),
    ("preserve", "j***@example.com"),
])
def test_masking_strategies(strategy, expected):
    """Test multiple masking strategies."""
    config = MaskingConfig(strategy=strategy)
    op = MaskingOperation(config)
    result = op.execute(sample_data)
    assert expected in result.data["email"][0]
```

**4. Metric State Verification:**
```python
def test_metric_calculation():
    """Test operation produces correct metrics."""
    op = MaskingOperation(config)
    result = op.execute(sample_data)

    assert "masked_count" in result.metrics
    assert result.metrics["masked_count"] == 2
    assert result.metrics["masking_ratio"] == 1.0
```

**5. Stochastic Control Pattern:**
```python
def test_noise_with_fixed_seed():
    """Test noise operation is reproducible with seed."""
    config = NoiseConfig(epsilon=1.0, seed=42)
    op = NoiseOperation(config)

    result1 = op.execute(sample_data)
    result2 = op.execute(sample_data)

    pd.testing.assert_frame_equal(result1.data, result2.data)
```

**6. Side-Effect Safety Check:**
```python
def test_non_target_fields_unchanged():
    """Verify non-target columns are not modified."""
    config = MaskingConfig(fields=["email"])
    op = MaskingOperation(config)

    original_names = sample_data["name"].copy()
    result = op.execute(sample_data)

    pd.testing.assert_series_equal(result.data["name"], original_names)
```

**7. Edge Case Matrix:**
```python
@pytest.mark.parametrize("data_type,test_data", [
    ("empty", pd.DataFrame()),
    ("nulls", pd.DataFrame({"email": [None, "test@example.com"]})),
    ("mixed", pd.DataFrame({"email": ["valid@email.com", 123, None]})),
])
def test_edge_cases(data_type, test_data):
    """Test operation handles edge cases gracefully."""
    op = MaskingOperation(config)
    result = op.execute(test_data)
    assert result.status in [OperationStatus.SUCCESS, OperationStatus.PARTIAL]
```

### Robustness Requirements

**Every Operation Must Handle:**
- Empty DataFrames
- Null/NaN values
- Mixed types
- Infinite values
- Missing columns
- Invalid data types

**Example:**
```python
def _validate_input(self, data: DataFrameType) -> None:
    """Validate input with comprehensive checks."""
    if data.empty:
        logger.warning("Empty DataFrame provided")
        return

    missing_fields = set(self.fields) - set(data.columns)
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    for field in self.fields:
        if data[field].isna().all():
            logger.warning(f"Field '{field}' is all NaN values")
```

## Development Workflow

### Version Management

**Version Definition:**
- Version defined in `pyproject.toml`
- Synced to `pamola_core/_version.py` via `setup.py`
- Update only `pyproject.toml` for version changes

**Example:**
```toml
[project]
name = "pamola-core"
version = "0.1.0"  # Update this only
```

### Pre-Commit/Pre-Push Rules

**Before Commit:**
1. Run linting: `ruff check pamola_core/`
2. Run tests: `pytest tests/`
3. Check for syntax errors
4. Verify no confidential information committed

**Before Push:**
1. Ensure all tests pass
2. Run full test suite
3. Update documentation if needed
4. Create clean, professional commit messages

### Commit Message Format

**Conventional Commits:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

**Example:**
```
feat(anonymization): add partial masking with pattern support

Implement partial masking operation that preserves format while
masking sensitive portions of data. Supports custom regex patterns
for flexible masking strategies.

Closes #123
```

### Code Review Standards

**Before Submitting PR:**
1. Code compiles without errors
2. All tests pass
3. New code has test coverage
4. Documentation updated
5. No linting errors (minor style OK)
6. Commit history clean

**Review Criteria:**
- Functionality: Does it work as intended?
- Design: Is it well-architected?
- Safety: Does it handle edge cases?
- Performance: Is it efficient?
- Documentation: Is it clear?

## File Organization

### File Size Management

**Maximum Lines:**
- Code files: 200 lines (preferred), 400 lines (hard limit)
- Split large files into smaller modules
- Use composition over inheritance

**Example Split:**
```
# Before (500 lines)
masking_operations.py

# After
masking/
├── __init__.py
├── base_masking_op.py    # Base classes (100 lines)
├── full_masking_op.py    # Full masking (80 lines)
├── partial_masking_op.py # Partial masking (120 lines)
└── masking_utils.py      # Utilities (100 lines)
```

### File Naming

**Use kebab-case:**
- ✅ `masking-operation.py`
- ✅ `profile-data-analyzer.py`
- ✅ `privacy-metrics-calculator.py`
- ❌ `maskingOperation.py`
- ❌ `profile_data_analyzer.py`

**Descriptive Names:**
- ✅ `hash-based-pseudonymization-op.py`
- ❌ `op.py`
- ❌ `utils.py`

## Performance Standards

### Memory Efficiency

**Guidelines:**
- Use chunk-based processing for large datasets
- Prefer generators over lists for large sequences
- Release resources explicitly (context managers)
- Profile memory usage for operations > 1M records

**Example:**
```python
def process_large_dataset(data: pd.DataFrame, chunk_size: int = 10000):
    """Process large dataset in chunks."""
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        result = process_chunk(chunk)
        results.append(result)
    return pd.concat(results)
```

### Caching Strategy

**When to Cache:**
- Expensive computations (metric calculations)
- NLP model loading
- Dictionary lookups
- Schema validation results

**Example:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_entity_dictionary(entity_type: str) -> Dict:
    """Load and cache entity dictionary."""
    path = f"pamola_core/resources/entities/{entity_type}.json"
    with open(path) as f:
        return json.load(f)
```

## Security Standards

### Cryptography

**Guidelines:**
- Use `cryptography` library for secure operations
- SHA3 for irreversible hashing
- AES-256 for reversible pseudonymization
- Never store plaintext passwords or keys
- Use environment variables for configuration

**Example:**
```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

def hash_value(value: str, salt: bytes) -> str:
    """Hash value using SHA3-256."""
    digest = hashes.Hash(hashes.SHA3_256())
    digest.update(value.encode())
    digest.update(salt)
    return digest.finalize().hex()
```

### Data Protection

**Principles:**
- No API keys or credentials in code
- Use environment variables for secrets
- Log sensitive data sparingly
- Encrypt stored mappings
- Secure by default

**Example:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")
```

## Documentation Standards

### Code Documentation

**Requirements:**
- All public functions must have docstrings
- Complex algorithms need inline comments
- Docstrings must include Args, Returns, Raises, Examples
- Keep docstrings up-to-date with code changes

### API Documentation

**Standards:**
- Document all public APIs
- Provide usage examples
- Explain parameters and return types
- Note any exceptions raised
- Include version information

## Module-Specific Standards

### Anonymization Module

**Requirements:**
- All operations inherit from `AnonymizationOperation`
- Use masking presets where applicable
- Validate input data types
- Track privacy metrics
- Support both pandas and Dask

### Metrics Module

**Requirements:**
- Return normalized scores (0-1) where applicable
- Include verdict generation (PASS/WARN/FAIL)
- Cache expensive calculations
- Handle edge cases (empty, nulls)
- Provide detailed metric breakdowns

### Profiling Module

**Requirements:**
- Use appropriate analyzers for data types
- Cache analysis results
- Provide recommendations
- Handle missing values gracefully
- Support incremental analysis

### Fake Data Module

**Requirements:**
- Use PRNG with seed management
- Support cultural/linguistic variations
- Validate generated data format
- Provide metrics for quality assessment
- Support reversible mapping

## Quality Assurance

### Code Quality Checklist

Before marking code as complete:
- [ ] Code compiles without errors
- [ ] All tests pass (pytest)
- [ ] New code has test coverage
- [ ] Docstrings complete and accurate
- [ ] Type hints present
- [ ] No hard-coded secrets
- [ ] Error handling comprehensive
- [ ] Logging added appropriately
- [ ] Performance acceptable
- [ ] Documentation updated

### Static Analysis

**Tools:**
- `ruff` for linting
- `mypy` for type checking
- `pytest` for testing
- `coverage` for test coverage

**Usage:**
```bash
# Linting
ruff check pamola_core/

# Type checking
mypy pamola_core/

# Testing
pytest tests/ -v

# Coverage
pytest --cov=pamola_core tests/
```

## Continuous Integration

### CI Pipeline Requirements

**Each PR Must:**
1. Pass all tests
2. Meet code coverage thresholds (95%+)
3. Pass static analysis
4. Build successfully
5. Pass security scans

### Testing Requirements

**Minimum Coverage:**
- Overall: 95%
- New code: 100%
- Critical paths: 100%

**Test Types:**
- Unit tests: Function-level testing
- Integration tests: Multi-operation testing
- Edge case tests: Error handling
- Performance tests: Large dataset testing

## References

- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [PEP 257 Docstrings](https://peps.python.org/pep-0257/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)
- [project-overview-pdr.md](./project-overview-pdr.md) - Product requirements
- [codebase-summary.md](./codebase-summary.md) - Codebase overview
- [system-architecture.md](./system-architecture.md) - Architecture details
