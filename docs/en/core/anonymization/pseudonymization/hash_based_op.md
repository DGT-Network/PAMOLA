# Hash-Based Pseudonymization Operation

**Module:** `pamola_core.anonymization.pseudonymization.hash_based_op`
**Version:** 2.0.0
**Last Updated:** 2026-04-20
**Status:** Stable

## Overview

The `HashBasedPseudonymizationOperation` performs irreversible pseudonymization using cryptographic hash functions (SHA3-256/512). It transforms sensitive identifiers into pseudonyms that cannot be reversed to recover the original values, while preserving referential integrity (same input → same pseudonym across rows and datasets when the same salt/pepper are used). Supports salt + optional session pepper, multiple output encodings, compound (multi-field) hashing, and collision tracking.

## Constructor Signature

```python
def __init__(
    self,
    field_name: str,
    # ==== Fields & algorithm ====
    additional_fields: Optional[List[str]] = None,
    algorithm: str = "sha3_256",
    # ==== Salt & pepper ====
    salt_config: Optional[Dict[str, Any]] = None,
    salt_file: Optional[Path] = None,
    use_pepper: bool = True,
    pepper_length: int = 32,
    # ==== Output format ====
    hash_output_format: str = "hex",
    output_length: Optional[int] = None,
    output_prefix: Optional[str] = None,
    output_suffix: Optional[str] = None,
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
| `additional_fields` | Optional[List[str]] | None | Additional fields used for compound hashing |
| `algorithm` | str | "sha3_256" | Hash algorithm: `"sha3_256"` or `"sha3_512"` |
| `salt_config` | Optional[Dict[str, Any]] | None | Salt configuration (`source`: parameter/file/field; `value` or `field_name`) |
| `salt_file` | Optional[Path] | None | Path to salt file when `salt_config.source == "file"` |
| `use_pepper` | bool | True | Whether to use a session-scoped pepper for extra security |
| `pepper_length` | int | 32 | Pepper length in bytes |
| `hash_output_format` | str | "hex" | Output encoding: `"hex"`, `"base64"`, `"base32"`, `"base58"`, or `"uuid"` |
| `output_length` | Optional[int] | None | Truncate output to specified length (characters) |
| `output_prefix` | Optional[str] | None | Prefix prepended to each pseudonym |
| `output_suffix` | Optional[str] | None | Suffix appended to each pseudonym |
| `quasi_identifiers` | Optional[List[str]] | None | Quasi-identifier columns for privacy metrics |
| `compound_mode` | bool | False | Whether to hash `field_name` together with `additional_fields` |
| `compound_separator` | str | "\|" | Separator used between fields in compound mode |
| `compound_null_handling` | str | "skip" | Null handling in compound mode: `"skip"`, `"empty"`, or `"raise"` |
| `**kwargs` | dict | - | Additional keyword arguments passed to `AnonymizationOperation` |

## Key Methods

### execute()
Executes the hash-based pseudonymization operation on the input data source.

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
- Support for pandas and Dask DataFrames
- Deterministic across runs ONLY with `use_pepper=False` and a non-trivial fixed salt; with the default session pepper the same input gets a different pseudonym in every run (consistent within one run)
- Compound hashing across multiple fields
- Collision probability estimation and tracking
- Conditional processing (mode, condition_field, ka_risk_field)

### process_batch()
Processes a batch of rows, producing pseudonyms for each value.

### process_value()
Hashes a single value using the configured algorithm, salt, and pepper.

## Salt Configuration

| Source | Example | Notes |
|--------|---------|-------|
| `parameter` | `{"source": "parameter", "value": "ab" * 32}` | Inline 256-bit hex value |
| `file` | `{"source": "file"}` + `salt_file=Path(...)` | Load from external file |
| `field` | `{"source": "field", "field_name": "salt_col"}` | Per-record salt from another column |

## Output Formats

| Format | Example (truncated) |
|--------|---------------------|
| `hex` | `eff8c0fc416d38cffcd9730b2a964658...` |
| `base64` | `7/jA/EFtOM/82XMLKpZGWMfp...` |
| `base32` | `57ZMB7CBNU4M77WNOMFSVFSGLD...` |
| `base58` | `CsG6fsL3Jz6CvGANXLhTgZ...` |
| `uuid` | `eff8c0fc-416d-38cf-fcd9-730b2a964658` |

## Usage Examples

### Basic Hash Pseudonymization
```python
from pamola_core.anonymization.pseudonymization.hash_based_op import (
    HashBasedPseudonymizationOperation,
)
from pamola_core.utils.ops.op_data_source import DataSource

op = HashBasedPseudonymizationOperation(
    field_name="email",
    algorithm="sha3_256",
    salt_config={"source": "parameter", "value": "ab" * 32},
    use_pepper=False,
    hash_output_format="hex",
)

data_source = DataSource.from_file_path("users.csv", name="main")
result = op.execute(
    data_source=data_source,
    task_dir=Path("output/task_001"),
    reporter=None,
)
```

### UUID-Format Output
```python
op = HashBasedPseudonymizationOperation(
    field_name="user_id",
    hash_output_format="uuid",
    output_prefix="usr_",
)
# Output: "usr_eff8c0fc-416d-38cf-fcd9-730b2a964658"
```

### Compound Hashing (Multi-Field)
```python
op = HashBasedPseudonymizationOperation(
    field_name="first_name",
    additional_fields=["last_name", "dob"],
    compound_mode=True,
    compound_separator="|",
    compound_null_handling="skip",
)
# Hashes "<first_name>|<last_name>|<dob>" as a single identifier
```

### Truncated Output with File Salt
```python
op = HashBasedPseudonymizationOperation(
    field_name="ssn",
    salt_config={"source": "file"},
    salt_file=Path("secrets/salt.bin"),
    output_length=16,
    hash_output_format="base58",
)
```

## Integration with Base Class

**Reference:** This operation inherits from the abstract base class documented in [base_anonymization_op.md](../base_anonymization_op.md). See that file for all shared parameters, methods, conditional processing, and k-anonymity integration details.

## Security Considerations

- **Algorithm:** Uses SHA3 (not SHA1/MD5); both `sha3_256` and `sha3_512` are collision-resistant.
- **Salt:** Required to prevent rainbow-table attacks. Use at least 256-bit random salt.
- **Construction:** `H(H(value || salt) || pepper)` with SHA3-256/512 (single pass; it is neither HMAC nor a key-stretching KDF). `H(value || salt)` alone when `use_pepper=False`.
- **Pepper:** `use_pepper=True` (default) draws fresh random bytes (`secrets`) for EVERY run and keeps them in memory only. Consequence: pseudonyms are consistent inside one run (all chunks and workers) but **cannot be linked across runs, and the pepper cannot be supplied or restored**. If you need the same pseudonym for the same input in two runs or two datasets, use `use_pepper=False` with a high-entropy secret salt (a 256-bit random value, kept as a secret) - or the consistent-mapping operation.
- **Irreversibility:** No mapping is stored; the hash cannot be inverted. It can still be **reproduced for every candidate value**: see the low-entropy warning.
- **Low-entropy warning (SRS 7.3).** Hashing a field whose possible values are few or guessable (a status, a year, a postcode, a first name, a small id range) does not hide it: whoever can enumerate the candidates and knows the salt (a fixed salt without pepper, or a leaked salt) hashes each and reads the original back. When the source field has at most 1000 distinct values the operation logs one warning per run (counts only) and records `pseudonymization.low_entropy_source` and `distinct_source_values` in the metrics. The 1000 threshold is a heuristic on the data, not a proof either way: a field with many values can still be guessable (phone numbers, national ids).
- **Secrets are not serialized:** the salt value, the pepper and the output encryption key are not part of the object's pickled state, `copy`, or Dask tokenization (the config's salt is redacted); they are handed to the operation's own parallel workers only for the duration of its parallel run. An operation copied or pickled elsewhere cannot hash and must be recreated from its parameters.

## Metrics

The operation collects:
- **Pseudonymization Rate:** Percentage of values successfully transformed
- **Collision Count / Probability:** Number and estimated probability of hash collisions
- **Cache Hit Rate:** LRU cache efficiency
- **Hash Computation Time:** Total and per-record timing
- **Performance Metrics:** Throughput and memory usage

## Related Components

- `AnonymizationOperation` (base class)
- `HashBasedPseudonymizationConfig` (configuration schema)
- `HashGenerator` (cryptographic hashing)
- `PseudonymizationCache` (LRU result cache)

## Changelog

**v2.0.0 (2026-04-20)**
- Parameter `output_format` renamed to `hash_output_format` (disambiguates from generic output format)
- Added `compound_null_handling` strategy options
- Collision probability estimation hardened
- Schema exclude/tooltip/ui schema artifacts added

**v1.0.1 (2025-12-01)**
- Bug fixes for salt file loading and pepper generation

**v1.0.0 (2025-10-15)**
- Initial stable release with SHA3-256/512 support
- Multiple output formats (hex, base64, base32, base58, uuid)
- Compound and single-field hashing