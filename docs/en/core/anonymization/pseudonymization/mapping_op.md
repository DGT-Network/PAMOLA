# Consistent Mapping Pseudonymization Operation

**Module:** `pamola_core.anonymization.pseudonymization.mapping_op`
**Version:** 1.1.0
**Last Updated:** 2026-04-20
**Status:** Stable

## Overview

The `ConsistentMappingPseudonymizationOperation` provides reversible pseudonymization by maintaining a bidirectional mapping between original values and generated pseudonyms. All mappings are stored encrypted using AES-256-GCM, enabling controlled re-identification when authorized. Guarantees that the same input always maps to the same pseudonym (within and across datasets sharing the same mapping file).

## Constructor Signature

```python
def __init__(
    self,
    field_name: str,
    mapping_encryption_key: Union[str, bytes],
    # ==== Fields ====
    additional_fields: Optional[List[str]] = None,
    # ==== Mapping storage ====
    mapping_file: Optional[Union[str, Path]] = None,
    mapping_format: str = "csv",
    create_if_not_exists: bool = True,
    backup_on_update: bool = True,
    persist_frequency: int = 1000,
    # ==== Pseudonym generation ====
    pseudonym_type: str = "uuid",
    pseudonym_prefix: Optional[str] = None,
    pseudonym_suffix: Optional[str] = None,
    pseudonym_length: int = 36,
    # ==== Compound & privacy ====
    quasi_identifiers: Optional[List[str]] = None,
    compound_mode: bool = False,
    compound_separator: str = "|",
    compound_null_handling: str = "skip",
    **kwargs,
):
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Primary field to pseudonymize |
| `mapping_encryption_key` | Union[str, bytes] | Required | 256-bit AES-GCM key (64-char hex string or 32 bytes) for mapping storage |
| `additional_fields` | Optional[List[str]] | None | Additional fields used for compound pseudonymization |
| `mapping_file` | Optional[Union[str, Path]] | None | Custom mapping file name (auto-generated if None) |
| `mapping_format` | str | "csv" | Storage format for mapping: `"csv"` or `"json"` |
| `create_if_not_exists` | bool | True | Create mapping file if it does not exist |
| `backup_on_update` | bool | True | Back up mapping file before each update |
| `persist_frequency` | int | 1000 | Persist to disk after N newly created mappings |
| `pseudonym_type` | str | "uuid" | Pseudonym format: `"uuid"`, `"sequential"`, or `"random_string"` |
| `pseudonym_prefix` | Optional[str] | None | Prefix prepended to each pseudonym |
| `pseudonym_suffix` | Optional[str] | None | Suffix appended to each pseudonym |
| `pseudonym_length` | int | 36 | Target length for `"random_string"` pseudonyms |
| `quasi_identifiers` | Optional[List[str]] | None | Quasi-identifier columns for privacy metrics |
| `compound_mode` | bool | False | Whether to map `field_name` together with `additional_fields` |
| `compound_separator` | str | "\|" | Separator used between fields in compound mode |
| `compound_null_handling` | str | "skip" | Null handling in compound mode: `"skip"`, `"empty"`, or `"raise"` |
| `**kwargs` | dict | - | Additional keyword arguments passed to `AnonymizationOperation` |

## Key Methods

### execute()
Executes the consistent mapping pseudonymization operation on the input data source.

```python
def execute(
    self,
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> OperationResult:
```

**Features:**
- Full integration with PAMOLA framework (caching, progress tracking, metrics)
- Bidirectional mapping with encrypted storage (AES-256-GCM)
- Guaranteed consistency: same input → same pseudonym
- Thread-safe mapping access via internal lock
- Automatic incremental persistence every `persist_frequency` writes
- Backup-before-update for safe recovery

### process_batch()
Processes a batch of rows, looking up existing mappings or generating new ones.

### process_value()
Resolves a single value to its pseudonym; creates a new mapping if the value is unseen.

## Pseudonym Types

| Type | Example | Notes |
|------|---------|-------|
| `uuid` | `a1b2c3d4-e5f6-4708-8910-abcdef012345` | RFC 4122 UUID (36 chars) |
| `sequential` | `1`, `2`, `3`, ... | Monotonically increasing integers (efficient, enumerable) |
| `random_string` | `Kd9mP2xL...` | Random alphanumeric, length = `pseudonym_length` |

## Mapping Storage

- **Encryption:** AES-256-GCM with the provided key.
- **Format:** CSV or JSON. Both are encrypted at rest.
- **File name:** Auto-derived from field name if `mapping_file` is not provided.
- **Persistence:** Mappings are persisted after every `persist_frequency` new entries and at the end of each run.
- **Backup:** When `backup_on_update=True`, a `.bak` copy is created before each write.

## Usage Examples

### Basic UUID Pseudonymization
```python
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.utils.ops.op_data_source import DataSource

op = ConsistentMappingPseudonymizationOperation(
    field_name="email",
    mapping_encryption_key="ab" * 32,  # 64 hex chars = 256-bit key
    pseudonym_type="uuid",
)

data_source = DataSource.from_file_path("users.csv", name="main")
result = op.execute(
    data_source=data_source,
    task_dir=Path("output/task_001"),
    reporter=None,
)
```

### Sequential Pseudonyms with Prefix
```python
op = ConsistentMappingPseudonymizationOperation(
    field_name="customer_id",
    mapping_encryption_key=bytes.fromhex("ab" * 32),
    pseudonym_type="sequential",
    pseudonym_prefix="CUST_",
)
# Output: "CUST_1", "CUST_2", ...
```

### Random-String with Custom Length
```python
op = ConsistentMappingPseudonymizationOperation(
    field_name="ssn",
    mapping_encryption_key="ab" * 32,
    pseudonym_type="random_string",
    pseudonym_length=12,
    mapping_format="json",
)
```

### Compound Mapping (Multi-Field)
```python
op = ConsistentMappingPseudonymizationOperation(
    field_name="first_name",
    additional_fields=["last_name", "dob"],
    mapping_encryption_key="ab" * 32,
    compound_mode=True,
    compound_separator="|",
    compound_null_handling="skip",
)
```

### Reuse Existing Mapping File
```python
op = ConsistentMappingPseudonymizationOperation(
    field_name="email",
    mapping_encryption_key="ab" * 32,
    mapping_file=Path("shared/email_mapping.enc.csv"),
    create_if_not_exists=False,  # Fail if file is missing
)
```

## Integration with Base Class

**Reference:** This operation inherits from the abstract base class documented in [base_anonymization_op.md](../base_anonymization_op.md). See that file for all shared parameters, methods, conditional processing, and k-anonymity integration details.

## Security Considerations

- **Encryption Key:** Must be a 256-bit key, either 64-character hex string or 32 raw bytes. Invalid format or size raises `ValidationError` at construction time.
- **Key Handling:** The public attribute is removed after init; key is stored privately as `_mapping_encryption_key` to prevent accidental serialization/logging.
- **Reversibility:** Authorized holders of the key can reverse-lookup original values from pseudonyms. Protect the key with the same rigor as the source data.
- **Storage at Rest:** Every mapping write is AES-256-GCM encrypted; tampering is detected via the GCM authentication tag.
- **Backup:** Before each persistence, a `.bak` copy is created to allow recovery from interrupted writes.

## Metrics

The operation collects:
- **Mapping Hits / Misses:** How many lookups hit existing mappings vs. required new generation
- **Total New Mappings:** Cumulative count of newly created mappings
- **Lookup Timing:** Mean lookup time (Welford running mean)
- **Collision Count:** Pseudonym collisions requiring regeneration
- **Persist Count:** Number of times mappings were flushed to disk

## Related Components

- `AnonymizationOperation` (base class)
- `ConsistentMappingPseudonymizationConfig` (configuration schema)
- `MappingStorage` (encrypted bidirectional mapping store)
- `PseudonymGenerator` (UUID / sequential / random_string generator)

## Changelog

**v1.1.0 (2026-04-20)**
- Added Welford running mean for lookup timing (O(1) memory)
- Hardened key handling: public hex attr removed after init
- Schema exclude/tooltip/ui schema artifacts added
- Error handling updated to use `ErrorCode.MAPPING_ERROR` with correct `context` + `reason` kwargs

**v1.0.0 (2025-11-01)**
- Initial stable release
- Bidirectional encrypted mapping with AES-256-GCM
- UUID, sequential, and random_string pseudonym types
- Incremental persistence with backup-on-update
