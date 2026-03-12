# PAMOLA.CORE Security and Performance Architecture

**Version:** 0.1.0
**Last Updated:** 2026-03-12

## Overview

This document describes the security architecture, performance optimization strategies, and deployment patterns for PAMOLA.CORE.

## Security Architecture

### Cryptography Integration

```mermaid
graph TB
    subgraph "Hash Operations"
        Input[Input Data]
        Salt[Random Salt]
        Hash[SHA3-256]
        Output[Hashed Value]
    end

    subgraph "Encryption Operations"
        Plain[Plain Data]
        Key[AES-256 Key]
        Encrypt[Encrypt]
        Encrypted[Encrypted Data]
        Decrypt[Decrypt]
    end

    subgraph "Key Management"
        KeyGen[Key Generation]
        KeyStore[Key Storage]
        KeyRot[Key Rotation]
    end

    Input --> Hash
    Salt --> Hash
    Hash --> Output

    Plain --> Encrypt
    Key --> Encrypt
    Encrypt --> Encrypted

    Encrypted --> Decrypt
    Key --> Decrypt
    Decrypt --> Plain

    KeyGen --> KeyStore
    KeyStore --> KeyRot
    KeyRot --> Key
```

### Data Protection Flow

```mermaid
graph TB
    subgraph "Input Protection"
        Validate[Validate Input]
        Sanitize[Sanitize Data]
        LogSafe[Log Safely]
    end

    subgraph "Processing Protection"
        Minimize[Minimize Data Exposure]
        Mask[Mask Sensitive Data]
        Encrypt[Encrypt Mappings]
    end

    subgraph "Output Protection"
        Filter[Filter Results]
        Truncate[Truncate Logs]
        SecureStore[Secure Storage]
    end

    Validate --> Sanitize
    Sanitize --> LogSafe
    LogSafe --> Minimize
    Minimize --> Mask
    Mask --> Encrypt
    Encrypt --> Filter
    Filter --> Truncate
    Truncate --> SecureStore
```

### Security Features

| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **Hashing** | SHA3-256 | Irreversible pseudonymization |
| **Encryption** | AES-256 | Reversible pseudonymization with mapping storage |
| **Key Management** | Secure key generation and rotation | Protect encryption keys |
| **Input Validation** | Pydantic schemas | Prevent injection attacks |
| **Audit Trail** | manifest.json | Full operation reproducibility |

## Performance Architecture

### Memory Management

```mermaid
graph TB
    subgraph "Memory Strategies"
        Chunk[Chunk-Based Processing]
        Stream[Stream Processing]
        Lazy[Lazy Evaluation]
    end

    subgraph "Memory Optimization"
        Reuse[Object Reuse]
        Release[Explicit Release]
        Profile[Memory Profiling]
    end

    subgraph "Scaling"
        Small[< 1M rows]
        Medium[1M - 10M rows]
        Large[> 10M rows]
    end

    Chunk --> Reuse
    Stream --> Release
    Lazy --> Profile

    Small --> InMemory[In-Memory]
    Medium --> Chunked[Chunked]
    Large --> Distributed[Distributed]

    InMemory --> Reuse
    Chunked --> Chunk
    Distributed --> Stream
```

### Parallel Processing

```mermaid
graph TB
    subgraph "Task Distribution"
        Task[Task Runner]
        Dep[Dependency Graph]
        Schedule[Scheduler]
    end

    subgraph "Execution"
        Parallel[Parallel Tasks]
        Sequential[Sequential Tasks]
    end

    subgraph "Coordination"
        Sync[Synchronization]
        Barrier[Barrier]
        Reduce[Reduce]
    end

    Task --> Dep
    Dep --> Schedule
    Schedule --> Parallel
    Schedule --> Sequential

    Parallel --> Sync
    Parallel --> Barrier
    Parallel --> Reduce
```

### Performance Optimization Strategies

| Strategy | Implementation | Use Case |
|----------|---------------|----------|
| **Vectorization** | NumPy/pandas operations | Single-field transformations |
| **Chunking** | Dask partitions | Large datasets (>1M rows) |
| **Caching** | Result cache with TTL | Repeated operations |
| **Lazy Loading** | Dask lazy evaluation | Memory-constrained environments |
| **Progress Tracking** | Hierarchical tracker | Long-running operations |

## Deployment Architecture

### Package Structure

```mermaid
graph TB
    subgraph "Package"
        Core[pamola-core]
        Extras[Extras]
        Dev[Dev Dependencies]
    end

    subgraph "Core Package"
        Anon[Anonymization]
        Metrics[Metrics]
        Profile[Profiling]
        Trans[Transformations]
        Fake[Fake Data]
        Utils[Utilities]
    end

    subgraph "Extras"
        Fast[Fast: Polars, DuckDB]
        Profiling[Profiling: YData, Presidio]
        NER[NER: spaCy]
        DP[DP: OpenDP]
    end

    subgraph "Dev"
        Test[pytest]
        Lint[ruff]
        Coverage[coverage]
    end

    Core --> Anon
    Core --> Metrics
    Core --> Profile
    Core --> Trans
    Core --> Fake
    Core --> Utils

    Extras --> Fast
    Extras --> Profiling
    Extras --> NER
    Extras --> DP

    Dev --> Test
    Dev --> Lint
    Dev --> Coverage
```

### Installation Options

```bash
# Core package
pip install pamola-core

# With performance extras
pip install pamola-core[fast]       # + Polars, ConnectorX, DuckDB

# With profiling extras
pip install pamola-core[profiling]  # + YData-profiling, Presidio

# With NER support
pip install pamola-core[ner]        # + spaCy for short text NER

# With differential privacy
pip install pamola-core[dp]         # + OpenDP for formal DP guarantees

# Development installation
pip install pamola-core[dev]        # + pytest, coverage, ruff
```

### Deployment Considerations

| Environment | Memory | Recommended Setup |
|-------------|--------|-------------------|
| **Development** | 4GB+ | Core + dev dependencies |
| **Testing** | 8GB+ | Core + all extras + dev |
| **Production (Small)** | 8GB+ | Core + required extras |
| **Production (Large)** | 16GB+ | Core + fast + Dask cluster |

## Compliance Considerations

### GDPR Alignment

- **Pseudonymization** (Art. 4(5)): Hash-based and mapping-based operations
- **Data Minimization** (Art. 25): Suppression and filtering operations
- **Security** (Art. 32): Encryption and secure storage

### HIPAA Safe Harbor

- 18 identifier types supported for removal/generalization
- Configurable suppression thresholds
- Audit trail via manifest.json

## References

- [system-architecture.md](./system-architecture.md) - Core system architecture
- [architecture-data-flows.md](./architecture-data-flows.md) - Data flows and component interactions
- [project-roadmap.md](./project-roadmap.md) - Development roadmap
