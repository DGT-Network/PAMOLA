# PAMOLA.CORE - Product Development Requirements

**Version:** 0.1.0
**Last Updated:** 2026-03-12
**Status:** Active Development

## Executive Summary

PAMOLA.CORE is an open-source Python library (3.11-3.12) for privacy-preserving data processing, developed by Realm Inveo Inc. and DGT Network Inc. It provides a comprehensive, operations-first framework for data anonymization, privacy analysis, synthetic data generation, and regulatory compliance support.

### Core Value Proposition

- **Operations-First Approach**: Direct transforms (mask, suppress, generalize) rather than constraint optimization
- **Full Audit Trail**: Reproducible pipelines with manifest.json tracking every operation
- **Practical Risk Assessment**: Data-release attacks to test re-identification risk
- **DP-Semantics Noise**: Calibrated noise with transparency reports
- **Comprehensive Metrics**: Privacy, utility, fidelity, and quality metrics with verdicts

## Problem Statement

Organizations handling sensitive data face critical challenges:

1. **Fragmented Tooling**: ARX (Java, GUI-focused), Faker/Presidio (no pipeline), DP libraries (narrow scope)
2. **Missing Capabilities**: Short text handling, risk testing, reproducibility
3. **Compliance Burden**: GDPR, HIPAA, CCPA requirements demand audit-ready evidence
4. **Data Sharing Risks**: Need to assess re-identification risk before data release

### Target Users

| User Type | Needs | PAMOLA.CORE Solutions |
|-----------|-------|----------------------|
| **Data Engineers** | Privacy-safe data preparation for ML | Anonymization operations, pipeline runtime |
| **Data Scientists** | Data access without privacy violations | Synthetic data, risk assessment |
| **Compliance Officers** | Audit-ready evidence | manifest.json, attack reports, metrics |
| **Healthcare Analysts** | HIPAA Safe Harbor de-identification | Healthcare-specific masking, suppression |
| **Financial Analysts** | GDPR/PCI-compliant data processing | Pseudonymization, noise, generalization |

## Functional Requirements

### Core Capabilities (REQ-CORE)

#### REQ-CORE-001: Data Anonymization
- **Description**: Transform sensitive data to prevent re-identification
- **Operations**:
  - Masking: Full masking, partial masking (position/pattern/preset strategies)
  - Suppression: Cell, attribute, and record-level suppression with outlier detection
  - Generalization: Categorical (hierarchy/frequency), numeric (binning/rounding), datetime
  - Pseudonymization: Hash-based (SHA3, irreversible), mapping-based (AES-256, reversible)
- **Acceptance Criteria**: All operations produce auditable results in manifest.json

#### REQ-CORE-002: Privacy Profiling
- **Description**: Analyze data for privacy risks before processing
- **Analyzers**:
  - Identity analysis (uniqueness, quasi-identifiers)
  - Categorical analysis (distribution, entropy)
  - Numeric analysis (outliers, distribution)
  - Date/currency analysis (patterns, formats)
  - Text analysis (PII detection, patterns)
- **Acceptance Criteria**: Profiling produces actionable risk recommendations

#### REQ-CORE-003: Privacy Metrics
- **Description**: Measure privacy, utility, fidelity, and quality
- **Metric Categories**:
  - Privacy: DCR, NNDR, Uniqueness, K-Anonymity, L-Diversity, Disclosure Risk
  - Utility: Classification, Regression, Information Loss
  - Fidelity: Statistical Fidelity, KS Test, KL Divergence
  - Quality: Distribution tests, correlation
- **Acceptance Criteria**: All metrics produce verdicts (PASS/WARN/FAIL)

#### REQ-CORE-004: Data-Release Attacks
- **Description**: Simulate practical re-identification attacks
- **Attack Types**:
  - CVPL: Information leakage between releases
  - Fellegi-Sunter Linkage: Record matching to external data
  - Singling-out: Unique record identification
  - Attribute Inference: Sensitive attribute prediction
- **Acceptance Criteria**: Attack results include risk scores and mitigation recommendations

#### REQ-CORE-005: Noise Injection (DP-Semantics)
- **Description**: Add calibrated noise with differential privacy semantics
- **Distributions**: Laplace, Gaussian, Exponential
- **Features**:
  - Secure random generation
  - Configurable sensitivity and epsilon
  - Clipping bounds for value constraints
  - Noise transparency reports
- **Acceptance Criteria**: Noise parameters fully documented in noise_report.json

#### REQ-CORE-006: Synthetic Data Generation
- **Description**: Generate privacy-safe synthetic datasets
- **Generators**: SDV-based (CTGAN, TVAE), pattern-based
- **Entity Types**: Names, emails, phones, organizations, addresses
- **Acceptance Criteria**: Synthetic data passes fidelity metrics vs original

#### REQ-CORE-007: Operation Framework
- **Description**: Base classes for all operations
- **Components**:
  - BaseOperation: Lifecycle management, progress tracking, Dask support
  - OperationRegistry: Dynamic registration and discovery
  - OperationConfig: JSON Schema validation, serialization
  - OperationResult: Standardized results with status, artifacts, metrics
  - OperationScope: Dataset, field, and group boundaries
- **Acceptance Criteria**: All operations inherit from BaseOperation

#### REQ-CORE-008: Task Orchestration
- **Description**: Pipeline execution with reproducibility
- **Features**:
  - Hierarchical progress tracking
  - Checkpoint/resume support
  - Dependency management
  - Seed management for reproducibility
  - Manifest generation (full audit trail)
- **Acceptance Criteria**: Every task run produces reproducible manifest.json

### Non-Functional Requirements

#### REQ-NFR-001: Performance
- Support for datasets up to 10M records via Dask integration
- Chunk-based processing for memory efficiency
- Operation result caching for performance optimization

#### REQ-NFR-002: Usability
- CLI interface for common operations
- Python API for programmatic access
- Clear error messages and logging
- Comprehensive documentation with examples

#### REQ-NFR-003: Reliability
- 95%+ test coverage target
- Robustness: Handle empty DataFrames, null values, mixed types
- Graceful degradation for unsupported data types

#### REQ-NFR-004: Security
- Cryptography helpers for secure hashing (SHA3)
- AES-256 encryption for reversible pseudonymization
- No API keys or credentials in code
- Secure random number generation

#### REQ-NFR-005: Compatibility
- Python 3.11-3.12 support
- Pandas and Dask DataFrame support
- Cross-platform (Linux, macOS, Windows)

### Data Flow Requirements

#### REQ-DATA-001: Input/Output
- **Input Formats**: CSV, JSON, Excel, Parquet
- **Output Formats**: CSV, Parquet (with metadata JSON)
- **Streaming**: Support for chunk-based processing

#### REQ-DATA-002: Metadata
- manifest.json with operation history
- metrics_detail.json with all metric calculations
- noise_report.json for noise operations
- attack reports with risk scores

## Technical Constraints

### Technology Stack
- **Language**: Python 3.11-3.12 (strictly enforced)
- **Data Processing**: pandas, numpy, dask, pyarrow
- **ML/Stats**: scikit-learn, scipy, torch, sdv
- **NLP**: spacy, nltk, langdetect, fasttext
- **Visualization**: matplotlib, seaborn, plotly
- **Validation**: pydantic

### Architecture Constraints
- **Pattern**: Feature-area modular library
- **Core Framework**: Operation-based (all operations inherit from BaseOperation)
- **Schema-Driven**: Pydantic schemas for all configurations
- **Dual-Engine**: Pandas and Dask DataFrames with auto-switching

### Licensing
- **License**: BSD 3-Clause
- **Copyright**: Dual copyright (Realm Inveo Inc. & DGT Network Inc.)
- **IP Rights**: All IP belongs exclusively to Realm Inveo Inc.

## Regulatory Context

### Supported Regulations

| Regulation | Relevant Capabilities | Notes |
|------------|----------------------|-------|
| **GDPR** | Pseudonymization, data minimization (Art. 25, 32) | Technical controls only |
| **HIPAA** | Safe Harbor de-identification support | Healthcare-specific operations |
| **CCPA/CPRA** | Data suppression, anonymization workflows | Consumer privacy rights |

### Important Disclaimers
> **Compliance Note**: PAMOLA.CORE provides technical capabilities only. Legal compliance requires organizational policies, procedures, and legal guidance beyond software tools.

## Use Cases

### UC-001: ML Dataset Preparation
**Actor**: Data Scientist
**Goal**: Prepare privacy-safe dataset for ML training

**Flow**:
1. Profile dataset to identify quasi-identifiers
2. Apply masking to direct identifiers (name, email, phone)
3. Generalize quasi-identifiers (age, zipcode, gender)
4. Add noise to numeric attributes (salary)
5. Run privacy metrics (k-anonymity, l-diversity)
6. Run attack suite to assess re-identification risk
7. Validate metrics pass thresholds
8. Export anonymized dataset with manifest

**Success Criteria**: All metrics pass, attack risk < 5%

### UC-002: Healthcare Data De-identification
**Actor**: Healthcare Analyst
**Goal**: HIPAA Safe Harbor de-identification

**Flow**:
1. Identify HIPAA identifiers (names, dates, locations)
2. Apply healthcare-specific masking (ICD codes, prescriptions)
3. Suppress records with rare diagnoses
4. Generalize dates to year, locations to state
5. Validate 18 HIPAA identifiers removed/suppressed
6. Run Safe Harbor validation
7. Generate de-identification report

**Success Criteria**: Safe Harbor validation passes

### UC-003: Data Sharing with External Partner
**Actor**: Compliance Officer
**Goal**: Share data with verifiable privacy guarantees

**Flow**:
1. Define data sharing agreement requirements
2. Configure anonymization pipeline
3. Run pipeline with seed for reproducibility
4. Execute attack suite (linkage, singling-out, attribute inference)
5. Generate compliance report with manifest
6. Review attack risk scores
7. Export dataset with full documentation

**Success Criteria**: Attack reports show acceptable risk, manifest available for audit

## Success Metrics

### Technical Metrics
- **Test Coverage**: 95%+ across all modules
- **Performance**: Process 1M records in < 5 minutes
- **Reliability**: < 1% error rate on valid inputs
- **Documentation**: 100% of public APIs documented

### Adoption Metrics
- **PyPI Downloads**: Track monthly downloads
- **GitHub Stars**: Community engagement
- **Issues/PRs**: Community contributions
- **Citations**: Academic and industry references

### Quality Metrics
- **Bug Resolution Time**: < 7 days median
- **Release Frequency**: Quarterly minor releases
- **API Stability**: No breaking changes in minor versions

## Open Questions

1. **Formal DP Guarantees**: Should we integrate OpenDP or diffprivlib for formal differential privacy?
2. **Cloud Integration**: Should we add native cloud storage support (S3, GCS, Azure)?
3. **UI Component**: Should we provide a basic web UI for pipeline configuration?
4. **Performance Benchmarks**: What are the target performance numbers for dataset sizes?
5. **Enterprise Features**: What enterprise features (RBAC, audit logging) are needed?

## References

- [README.md](../README.md) - Project overview and quick start
- [architecture.md](./architecture.md) - Technical architecture
- [code-standards.md](./code-standards.md) - Development guidelines
- [system-architecture.md](./system-architecture.md) - Detailed architecture with diagrams
- [PAMOLA Knowledge Base](https://realmdata.io/kb) - Privacy engineering resources
