## Mini-Specification: Refactoring the PAMOLA PAMOLA.CORE Cryptographic Subsystem

### Introduction and Motivation
PAMOLA PAMOLA.CORE handles multiple task-driven data flows, where individual tasks generate or modify sensitive data during processing. Many of these operations involve intermediate outputs or cached files that may contain personally identifiable information (PII), financial records, or algorithmic outcomes. These intermediate versions increase the attack surface unless protected appropriately. 

This specification introduces a unified encryption subsystem designed to:
- Reduce risks associated with temporary file exposure.
- Ensure minimal friction for developers working in trusted environments.
- Enable upgrade paths to more advanced security practices (key isolation, hardware tokens, side-channel mitigation).

The system supports three operational modes to provide flexibility across development, testing, and production contexts.

### Goals
- Support three encryption modes:
  - `none`: No encryption, useful for testing and debugging.
  - `simple`: AES-GCM based in-house encryption.
  - `age`: Integration with the external `age` CLI tool for stream encryption.
- Ensure modular and extensible architecture.
- Maintain compatibility with legacy data formats.
- Implement safe key management for the `simple` mode.
- Prepare foundation for advanced protections (e.g., hardware-bound keys, memory cleansing).

### Architecture Overview
```
┌────────────┐
│  API Layer │ ← pamola_core/utils/io_helpers/crypto_utils.py
└─────┬──────┘
      │
┌─────▼──────┐
│  Router    │ ← pamola_core/utils/io_helpers/crypto_router.py
└─────┬──────┘
      │
┌─────▼────────────────────┐
│    Crypto Providers      │ ← pamola_core/utils/crypto_helpers/providers/
└─────┬──────────┬─────────┘
      │          │
 ┌────▼──┐   ┌────▼────┐
 │simple │   │  age    │
 └───────┘   └─────────┘
```

### Key Concepts

- **Router Layer** selects the appropriate provider based on mode and invokes corresponding logic.
- **Providers** implement `encrypt_file()` and `decrypt_file()` based on a shared interface.
- **Key Store** supports secure storage of encryption keys for `simple` mode using AES-GCM and a master key.
- **Audit Logging** captures all encryption/decryption actions.
- **Legacy Migration** supports conversion of old format encrypted files.

### Mode Capabilities

| Mode   | Description                          | Encryption | Streaming | Key Store |
|--------|--------------------------------------|------------|-----------|-----------|
| none   | Plain copy, for debugging/testing     | No         | N/A       | No        |
| simple | AES-GCM, JSON metadata format         | Yes        | No        | Yes       |
| age    | External tool, stream-capable         | Yes        | Yes       | Optional  |

### Implementation Highlights

- **File metadata** will include `mode`, `version`, `timestamp`, and `description` fields to support future parsing.
- **Router** will auto-detect format when decrypting (via JSON keys or binary header detection).
- **Error handling** will be unified under a shared exception hierarchy with context-aware logging.
- **Master key** will be stored in `configs/master.key`, with structure and checks for future secure upgrades.
- **Key DB** will reside in `configs/keys.db` and store per-task keys as encrypted JSON records.

### Cryptographic Details

#### Internal (simple mode)
- **Algorithm**: AES-GCM (256-bit key, 12-byte IV, PBKDF2-SHA256 if password-based)
- **Format**: JSON structure with fields: `algorithm`, `iv`, `salt`, `data`, `mode`, `version`, and optional `file_info`
- **Libraries**: Uses `cryptography.hazmat` primitives for AES-GCM and PBKDF2

#### External (age mode)
- **Algorithm**: XChaCha20-Poly1305 (via age CLI tool)
- **Format**: Stream-based binary format defined by age (`age-encryption.org/v1`)
- **Invocation**: CLI call via `subprocess`.
- **Key handling options**:
  - *Option 1 – Passphrase-based*: passphrase is passed via stdin at runtime. No persistent key material is stored.
  - *Option 2 – Public/Private Key-based*: files are encrypted using recipient public keys and decrypted using identity private keys. Keys are stored in `~/.config/age/` or passed explicitly.
- **Planned default**: Public/private key mode is preferred for secure deployments. Scripts may use key path configuration or environment detection.

### Key Management Description
- **EncryptedKeyStore** manages per-task keys stored in `keys.db`, encrypted with a symmetric master key.
- **Master key** is stored in plain text only for MVP under `configs/master.key`, with read/write warnings logged.
- **Security checks**: File permissions (`chmod 600`) and exposure detection through `is_master_key_exposed()`.
- **Future plans**: Transition to system-level keyring or encrypted container for the master key.
- **Note on age mode**: In public/private key mode, long-term key storage is handled externally, and PAMOLA only references keys via config/environment.

### Mode Detection

A helper function `detect_encryption_mode(file_path)` will inspect the file and determine the correct decryption mode:
- If valid JSON with `mode`, `algorithm`, or known fields → assume `simple`
- If binary header starts with `age-encryption.org/` → assume `age`
- Otherwise → fallback to `none` or legacy detection

### Pamola Core Functions by Module

| Module               | Functions                                                                 |
|----------------------|---------------------------------------------------------------------------|
| `crypto_utils.py`    | `encrypt_file()`, `decrypt_file()` — entry points with optional mode arg  |
| `crypto_router.py`   | `encrypt_file_router()`, `decrypt_file_router()`, `detect_encryption_mode()` |
| `simple_provider.py` | `encrypt_file()`, `decrypt_file()` — AES-GCM based file operations        |
| `age_provider.py`    | `encrypt_file()`, `decrypt_file()` — CLI call to age with mode control    |
| `key_store.py`       | `store_task_key()`, `load_task_key()`, `is_master_key_exposed()`          |
| `audit.py`           | `log_crypto_operation()`                                                   |
| `legacy_migration.py`| `detect_legacy_format()`, `migrate_legacy_file()`                         |

### Deliverables

1. **Router module**: `crypto_router.py`
2. **Providers**:
   - `none_provider.py`
   - `simple_provider.py`
   - `age_provider.py`
3. **Key store module**: `key_store.py`
4. **Error definitions**: `errors.py`
5. **Logging**: `audit.py`
6. **Migration support**: `legacy_migration.py`

### Testing Plan

- Unit tests for each provider and router logic
- Integration tests for file roundtrips (enc/dec)
- Legacy file migration test
- Negative tests for error conditions (bad keys, file corruption)

### Implementation Sequence

1. Define error classes and base interfaces (CryptoProvider, exceptions)
2. Implement `none_provider` as baseline and test router dispatch
3. Build `simple_provider` with encryption metadata and metadata validation
4. Add `EncryptedKeyStore` with task-key encryption using master key
5. Integrate router logic + `detect_encryption_mode()`
6. Implement `age_provider` with config-driven CLI calls (passphrase + keypair)
7. Add `audit.py` for crypto operation logging
8. Finalize `crypto_utils.py` with public API wrappers
9. Build `legacy_migration.py` for auto-conversion and compatibility testing
10. Run full test suite; enable incremental rollout across modules

This specification provides a structured, forward-compatible path to implementing robust and flexible encryption capabilities in PAMOLA PAMOLA.CORE. The architecture is modular and mode-aware, supporting extensibility, auditability, and security-focused growth.

