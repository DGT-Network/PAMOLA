# TaskContext Documentation

**Module:** `pamola_core.utils.tasks.task_context`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Classes](#core-classes)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)
10. [Summary](#summary)

## Overview

The `TaskContext` module provides a lightweight dataclass for carrying execution context (seed, RNG, task_dir) for running tasks. It implements **FR-EP3-CORE-043** — reproducibility via seed propagation — enabling deterministic operation sequences across different operations within the same task.

The primary purpose is to ensure that randomized operations produce consistent, reproducible results when given the same master seed, while allowing different operations to receive different derived seeds to maintain independence.

## Key Features

- **Master Seed Management**: Single seed controls all randomness in a task
- **Deterministic Per-Operation Seeds**: Each operation gets its own reproducible seed derived from master seed
- **Seed Independence**: Different operations get uncorrelated seeds despite sharing master seed
- **RNG Instance**: Built-in random.Random instance initialized with master seed
- **Task Directory Reference**: Carries task_dir for operation artifact management
- **Reproducibility Guarantee**: Same master seed + same sequence = identical results

## Architecture

```
TaskContext
  ├── seed (Optional[int])
  │   └── Master seed for entire task
  │
  ├── task_dir (Optional[Path])
  │   └── Root directory for task artifacts
  │
  ├── rng (random.Random) [derived]
  │   └── Private RNG instance for seed generation
  │
  └── get_operation_seed(op_name: str) -> Optional[int]
      └── Generates deterministic per-operation seed
```

### Seed Derivation Flow

```
Master Seed (e.g., 42)
      ↓
[hash master_seed : operation_name]
      ↓
SHA-256 digest
      ↓
First 4 bytes → unsigned int
      ↓
Per-operation Seed (deterministic)
```

## Dependencies

| Module | Purpose |
|--------|---------|
| `random` | random.Random for RNG management |
| `hashlib` | SHA-256 hashing for seed derivation |
| `dataclasses` | @dataclass decorator |
| `pathlib` | Path type for task directories |

## Core Classes

### TaskContext

A dataclass that carries execution context for a running task, ensuring reproducibility of randomized operations.

#### Fields

```python
@dataclass
class TaskContext:
    seed: Optional[int] = None
    task_dir: Optional[Path] = None
    rng: random.Random = field(init=False)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | Optional[int] | None | Master random seed; None for non-deterministic |
| `task_dir` | Optional[Path] | None | Root task directory for artifacts |
| `rng` | random.Random | Initialized | Private RNG instance (init=False) |

#### Constructor

```python
def __init__(
    self,
    seed: Optional[int] = None,
    task_dir: Optional[Path] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | Optional[int] | None | Master seed for reproducibility |
| `task_dir` | Optional[Path] | None | Task directory path |

**Behavior:**

- Initializes `seed` and `task_dir` from parameters
- Automatically creates `rng` instance via `__post_init__()` initialized with master seed
- If `seed` is None, `rng` is seeded from system entropy (non-deterministic)

#### Methods

##### get_operation_seed()

```python
def get_operation_seed(self, op_name: str) -> Optional[int]
```

**Purpose:** Return a deterministic seed for a named operation.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_name` | str | Operation class name or unique identifier |

**Returns:** Integer seed if master seed is set, None otherwise

**Algorithm:**

1. If master seed is None, return None (non-deterministic mode)
2. Create string: `"{master_seed}:{op_name}"`
3. Hash with SHA-256
4. Take first 4 bytes as big-endian unsigned int
5. Return resulting seed

**Properties:**

- **Deterministic**: Same `(seed, op_name)` pair always produces same result
- **Independent**: Different op_names produce uncorrelated seeds
- **Reproducible**: Runs with same seed produce identical sequences

**Example:**
```python
ctx = TaskContext(seed=42)

# Same operation, same seed
seed1 = ctx.get_operation_seed("AnonymizationOp")
seed2 = ctx.get_operation_seed("AnonymizationOp")
assert seed1 == seed2  # True

# Different operations, different seeds
seed3 = ctx.get_operation_seed("ProfilingOp")
assert seed1 != seed3  # True (uncorrelated)
```

## Usage Examples

### Example 1: Basic Task Reproducibility

```python
from pathlib import Path
from pamola_core.utils.tasks.task_context import TaskContext

# Create context with fixed seed
ctx = TaskContext(
    seed=12345,
    task_dir=Path("/tmp/my_task")
)

# All operations in this task will be reproducible
# Same seed + operation name = same results

# Example: RandomMaskingOperation uses the seed
masking_seed = ctx.get_operation_seed("RandomMaskingOperation")
print(f"Masking seed: {masking_seed}")

# Run again with same context
ctx2 = TaskContext(seed=12345)
masking_seed2 = ctx2.get_operation_seed("RandomMaskingOperation")
assert masking_seed == masking_seed2  # True - reproducible
```

### Example 2: Non-Deterministic Mode

```python
from pamola_core.utils.tasks.task_context import TaskContext

# Create context without seed (non-deterministic)
ctx = TaskContext()  # seed=None

# Operations run non-deterministically
seed1 = ctx.get_operation_seed("RandomOp")
print(f"Seed 1: {seed1}")  # None

# Perfect for non-repeatable requirements
```

### Example 3: Integration with TaskRunner

```python
from pathlib import Path
from pamola_core.utils.tasks.task_runner import TaskRunner
from pamola_core.utils.tasks.task_context import TaskContext

# TaskRunner uses TaskContext internally
runner = TaskRunner(
    task_id="anon_task_001",
    task_type="anonymization",
    description="Anonymize sensitive data",
    additional_options={"seed": 42}  # Passed to TaskContext
)

# TaskRunner automatically creates TaskContext with seed
# All operations will be deterministic

# Run task - same input + same seed = same output
result = runner.run_operations()
```

### Example 4: Multi-Operation Determinism

```python
from pamola_core.utils.tasks.task_context import TaskContext
from anonymization.operations import RandomMaskingOperation, NoiseAdditionOperation
import numpy as np

# Create deterministic context
ctx = TaskContext(seed=999)

# First operation uses derived seed
masking_seed = ctx.get_operation_seed("RandomMaskingOperation")
np.random.seed(masking_seed)
masked_data = apply_random_masking(data)

# Second operation uses independent seed
noise_seed = ctx.get_operation_seed("NoiseAdditionOperation")
np.random.seed(noise_seed)
noisy_data = add_noise(masked_data)

# Running again with same context produces identical results
# (assuming same input data)
```

### Example 5: Seed Distribution Across Operations

```python
from pamola_core.utils.tasks.task_context import TaskContext

ctx = TaskContext(seed=42)

# Show different operations get different seeds
operations = [
    "MaskingOperation",
    "GeneralizationOperation",
    "SuppressionOperation",
    "NoiseOperation"
]

seeds = {}
for op_name in operations:
    seed = ctx.get_operation_seed(op_name)
    seeds[op_name] = seed
    print(f"{op_name}: {seed}")

# All different, all deterministic
assert len(set(seeds.values())) == len(operations)  # True
```

### Example 6: Task Directory Reference

```python
from pathlib import Path
from pamola_core.utils.tasks.task_context import TaskContext

# Create context with task directory
ctx = TaskContext(
    seed=42,
    task_dir=Path("/projects/anonymization_task_001")
)

# Operations can access task directory
output_dir = ctx.task_dir / "output"
log_dir = ctx.task_dir / "logs"
artifacts_dir = ctx.task_dir / "artifacts"

# Operations can create deterministic artifacts
op_seed = ctx.get_operation_seed("SyntheticDataGeneration")
np.random.seed(op_seed)
synthetic_data = generate_synthetic_data(original_data)

# Save with deterministic content
synthetic_data.to_csv(
    output_dir / "synthetic_data.csv",
    index=False
)
```

## Best Practices

1. **Use Fixed Seeds for Production**
   - Set `seed` parameter for reproducible production runs
   - Document the seed used for audit trails
   - Allows debugging and replication of results

2. **Use None for Non-Deterministic Scenarios**
   - Leave `seed=None` for exploration and testing
   - Better variance in synthetic data generation
   - Useful for robustness testing

3. **Create Fresh Contexts Per Task**
   - Don't reuse TaskContext across multiple tasks
   - Each task should have its own context instance
   - Prevents unintended seed correlation

4. **Pass Seeds Through Task Configuration**
   - Store seeds in task configuration files
   - Use `additional_options` in TaskRunner
   - Enables seed tracking and reproducibility

5. **Document Seed Usage**
   - Record which seed was used for which task
   - Essential for regulatory compliance and audits
   - Helpful for debugging and replication

6. **Verify Reproducibility**
   - Run same task with same seed twice
   - Compare outputs to verify determinism
   - Catch non-deterministic operations

7. **Use Unique Operation Names**
   - Operation names should be class names or unique identifiers
   - Different names produce different seeds
   - Prevents accidental seed collision

## Troubleshooting

### Issue: Operations Produce Different Results with Same Seed

**Cause:** Operations not using the seed from TaskContext

**Solution:**
```python
# Operation must explicitly use the seed from TaskContext
class MyRandomOperation(BaseOperation):
    def execute(self, data_source, task_dir, reporter, **kwargs):
        # Get seed from context (passed via kwargs)
        seed = kwargs.get('seed')

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Now operations will be deterministic
```

### Issue: Seeds Are the Same for Different Operations

**Cause:** Not using get_operation_seed() - using master seed directly

**Solution:**
```python
# Wrong - all operations get same seed
ctx = TaskContext(seed=42)
seed = ctx.seed  # All operations get 42

# Correct - each operation gets unique derived seed
seed = ctx.get_operation_seed("OperationName")  # Unique seed
```

### Issue: Need to Debug Specific Operation

**Solution:**
```python
# Verify the seed being used
ctx = TaskContext(seed=42)
op_seed = ctx.get_operation_seed("MyOperation")
print(f"Operation seed: {op_seed}")

# Set breakpoints in operation to verify seed is applied
# Add logging to confirm randomness is controlled
import numpy as np
np.random.seed(op_seed)
print(f"First random value: {np.random.random()}")  # Always same with seed
```

## Related Components

| Component | Purpose | Integration |
|-----------|---------|-------------|
| `TaskRunner` | Creates and manages TaskContext | Creates instance in `__init__` |
| `BaseTask` | Task execution framework | Uses TaskContext for operations |
| `BaseOperation` | Operation execution | Receives seed via parameters |
| `task_runner.py` | Task orchestration | Injects seeds into operations |
| `operation_executor.py` | Executes operations | Passes seeds to operations |

## Summary

The `TaskContext` class is a simple but powerful tool for ensuring reproducibility in randomized operations. By providing a centralized master seed that generates deterministic per-operation seeds, it enables:

- **Reproducibility**: Same seed → same results
- **Independence**: Different operations get uncorrelated seeds
- **Traceability**: Single seed controls all randomness
- **Auditability**: Seed documented in task logs

Key strengths:
- Lightweight and focused design
- Clear API with single public method
- Integrates seamlessly with TaskRunner
- Supports both deterministic and non-deterministic modes

Use TaskContext whenever:
- Your task contains randomized operations
- You need reproducible results for auditing/compliance
- You want deterministic testing and debugging
- You need to distribute seeds across multiple operations
