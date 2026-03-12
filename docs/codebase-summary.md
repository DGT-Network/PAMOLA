# PAMOLA.CORE Codebase Summary

**Version:** 0.1.0
**Last Updated:** 2026-03-12
**Total Python Files:** 569
**Total Lines of Code:** ~240,000

## Overview

PAMOLA.CORE is a modular Python library for privacy-preserving data processing. The codebase follows an **Operation-Based Framework** pattern where all privacy and data processing tasks inherit from base classes and use Pydantic schemas for configuration management.

### Core Architecture Pattern

```
BaseOperation (abstract base)
    ├── AnonymizationOperation
    ├── MetricsOperation
    ├── TransformationOperation
    ├── ProfilingOperation
    └── ... (domain-specific operations)
```

Each operation:
- Inherits from `BaseOperation` in `pamola_core/utils/ops/op_base.py`
- Uses Pydantic schemas for configuration validation
- Supports both pandas and Dask DataFrames
- Produces `OperationResult` with status, artifacts, and metrics
- Registers in `OperationRegistry` for discovery

## Module Structure

### Core Modules (by LOC)

| Module | Files | Lines | Description |
|--------|-------|-------|-------------|
| **utils/** | 141 | 86,185 | Operation framework, tasks, reporting, NLP, crypto |
| **anonymization/** | 87 | 44,793 | Masking, suppression, noise, generalization, pseudonymization |
| **profiling/** | 98 | 43,034 | Data analyzers (identity, date, currency, text, etc.) |
| **transformations/** | 57 | 19,642 | Cleaning, merging, splitting, imputation |
| **fake_data/** | 49 | 18,869 | Synthetic data generators and dictionaries |
| **metrics/** | 46 | 12,516 | Utility, privacy, fidelity metrics |
| **privacy_models/** | 25 | 10,554 | k-anon, l-div, t-close, differential privacy |
| **common/** | 32 | 2,552 | Shared utilities (enums, helpers, validation, logging) |
| **attacks/** | 10 | 1,901 | Privacy attack simulations |
| **analysis/** | 7 | 1,796 | Statistical analysis utilities |
| **io/** | 8 | 1,329 | CSV, JSON, Excel, Parquet adapters |
| **configs/** | 4 | 1,475 | Configuration variables and settings |

### Directory Tree

```
pamola_core/
├── __init__.py
├── _version.py
├── anonymization/          # Privacy operations
│   ├── base_anonymization_op.py
│   ├── masking/            # Full/partial masking operations
│   ├── suppression/        # Cell/attribute/record suppression
│   ├── generalization/     # Categorical/numeric/datetime generalization
│   ├── noise/              # DP-semantics noise operations
│   ├── pseudonymization/   # Hash/mapping-based operations
│   └── commons/            # Shared utilities (patterns, presets, utils)
├── profiling/              # Data analysis
│   ├── base_profile_op.py
│   ├── analyzers/          # Identity, date, currency, text analyzers
│   └── commons/            # Analysis utilities
├── transformations/        # Data transformations
│   ├── base_transformation_op.py
│   ├── cleaning/           # Invalid value handling
│   ├── merging/            # Dataset merging
│   ├── splitting/          # Dataset/field splitting
│   ├── aggregation/        # Record aggregation
│   └── imputation/         # Missing value imputation
├── fake_data/              # Synthetic data
│   ├── base_fake_data_op.py
│   ├── generators/         # Name, email, phone, org generators
│   ├── dictionaries/       # Entity dictionaries
│   ├── mappers/            # Value mapping
│   └── commons/            # PRNG, metrics, validation
├── metrics/                # Privacy/utility/fidelity metrics
│   ├── base_metrics_op.py
│   ├── privacy/            # DCR, NNDR, uniqueness metrics
│   ├── utility/            # Classification, regression metrics
│   ├── fidelity/           # KS test, KL divergence
│   ├── quality/            # Statistical quality metrics
│   └── commons/            # Aggregation, normalization
├── privacy_models/         # Formal privacy models
│   ├── k_anonymity.py
│   ├── l_diversity.py
│   ├── t_closeness.py
│   └── differential_privacy.py
├── attacks/                # Privacy attack simulation
│   ├── cvpl_attack_op.py
│   ├── linkage_attack_op.py
│   ├── singling_out_op.py
│   └── attribute_inference_op.py
├── analysis/               # Statistical analysis
│   ├── correlation.py
│   ├── dataset_summary.py
│   └── privacy_risk.py
├── io/                     # Data adapters
│   ├── csv_adapter.py
│   ├── json_adapter.py
│   ├── excel_adapter.py
│   └── parquet_adapter.py
├── utils/                  # Core framework
│   ├── ops/                # Operation framework
│   │   ├── op_base.py      # BaseOperation class
│   │   ├── op_registry.py  # Operation registry
│   │   ├── op_config.py    # Configuration management
│   │   ├── op_result.py    # Result objects
│   │   ├── op_scope.py     # Operation boundaries
│   │   └── op_test_helpers.py
│   ├── tasks/              # Task execution
│   │   ├── task_runner.py
│   │   ├── progress.py     # HierarchicalProgressTracker
│   │   └── checkpoints.py
│   ├── nlp/                # NLP utilities
│   │   ├── entity/         # Entity recognition
│   │   ├── category_matching.py
│   │   └── cache.py
│   ├── schema_helpers/     # Schema building
│   ├── reporting/          # Report generation
│   ├── crypto_helpers/     # Cryptography utilities
│   └── io_helpers/         # IO utilities
├── common/                 # Shared utilities
│   ├── enums.py
│   ├── helpers.py
│   ├── validation.py
│   └── logging.py
├── configs/                # Configuration
│   ├── privacy_levels.py
│   ├── masking_defaults.py
│   └── metric_thresholds.py
└── resources/              # Static resources
    ├── entities/           # Entity dictionaries
    ├── stopwords/          # NLP stopwords
    └── tokenization/       # Tokenization data
```

## Key Components

### 1. Operation Framework (`utils/ops/`)

**Base Classes:**
- `BaseOperation`: Abstract base for all operations
  - Lifecycle: `__init__` → `validate_input` → `execute` → `finalize`
  - Progress tracking via `HierarchicalProgressTracker`
  - Dask support for large datasets
  - Integrated logging and error handling

- `AnonymizationOperation`: Specialized for privacy operations
- `MetricsOperation`: Specialized for metric calculations
- `TransformationOperation`: Specialized for data transformations

**Configuration:**
- `OperationConfig`: Pydantic-based configuration
  - JSON Schema validation
  - Serialization/deserialization
  - Default values and constraints

**Registry:**
- `OperationRegistry`: Dynamic operation registration
  - Decorator-based registration: `@register_operation`
  - Operation discovery by name
  - Metadata extraction
  - Version management

**Results:**
- `OperationResult`: Standardized result objects
  - Status: SUCCESS, FAILURE, PARTIAL
  - Artifacts: Output data, reports, plots
  - Metrics: Key-value metric storage
  - Metadata: Execution metadata

### 2. Anonymization Module (`anonymization/`)

**12 Operations across 5 categories:**

| Category | Operations | Key Features |
|----------|------------|--------------|
| **Masking** | FullMaskingOperation, PartialMaskingOperation | Position/pattern/preset strategies, format-preserving |
| **Suppression** | CellOperation, AttributeOperation, RecordOperation | Outlier detection, conditional suppression |
| **Generalization** | CategoricalOp, NumericOp, DateTimeOp | Hierarchy/frequency/merge strategies |
| **Noise** | UniformNumericOp, UniformTemporalOp | Laplace/Gaussian/Exponential, secure random |
| **Pseudonymization** | HashBasedOp, MappingOp | SHA3/AES-256, reversible/irreversible |

**Common Utilities (`anonymization/commons/`):**
- `masking_patterns.py`: Regex-based pattern masking
- `masking_presets.py`: Pre-configured presets (email, phone, SSN, credit card)
- `category_mapping.py`: Thread-safe category mapping engine
- `hierarchy_dictionary.py`: Hierarchical generalization support
- `metric_utils.py`: Privacy metric calculations

### 3. Profiling Module (`profiling/`)

**Analyzers (98 modules):**
- Identity analysis: uniqueness, quasi-identifiers
- Categorical analysis: distribution, entropy, rare categories
- Numeric analysis: outliers, distribution, statistics
- Date analysis: patterns, formats, ranges
- Currency analysis: symbols, formats, ranges
- Text analysis: PII detection, patterns, length
- Email/phone analysis: format validation, patterns
- Multi-value field (MVF) analysis: array/list fields
- Correlation analysis: feature correlations

**Key Classes:**
- `ProfileOperation`: Base profiling operation
- `anonymity.py`: k-anonymity, l-diversity calculations
- `attribute_utils.py`: Quasi-identifier detection
- `group_utils.py`: Group-based analysis

### 4. Metrics Module (`metrics/`)

**4 Categories with Composite Scoring:**

| Category | Metrics | Use Case |
|----------|---------|----------|
| **Privacy** | DCR, NNDR, Uniqueness, K-Anon, L-Div, Disclosure Risk | Re-identification risk |
| **Utility** | Classification, Regression, Information Loss, F1, R² | Data usefulness |
| **Fidelity** | Statistical Fidelity, KS Test, KL Divergence | Distribution similarity |
| **Quality** | KS, KL, Wasserstein, Pearson Correlation | Data quality |

**Composite Scoring:**
- Weighted aggregation across categories
- Verdict generation: PASS/WARN/FAIL
- Threshold-based evaluation

### 5. Transformations Module (`transformations/`)

**Operations:**
- Cleaning: Remove invalid values, standardize formats
- Merging: Join datasets with configurable strategies
- Splitting: Split by fields, ID values, or criteria
- Aggregation: Group-by aggregation with custom functions
- Imputation: Mean/median/mode imputation, KNN imputation

### 6. Fake Data Module (`fake_data/`)

**Generators:**
- Names (first, last, full)
- Emails (domain-specific patterns)
- Phone numbers (country-specific)
- Organizations (company names, departments)
- Addresses (street, city, state, zip)
- Custom entity generation

**Features:**
- PRNG with seed management
- Mapping storage with encryption
- Dictionary-based generation
- Cultural/linguistic variations

### 7. Attacks Module (`attacks/`)

**Attack Simulations:**
- **CVPL**: Cross-View Privacy Leakage (between releases)
- **Linkage**: Fellegi-Sunter record linkage
- **Singling-out**: Unique record identification
- **Attribute Inference**: Sensitive attribute prediction

**Output:**
- Risk scores (0-1)
- Verdicts (SAFE/RISKY/CRITICAL)
- Mitigation recommendations

### 8. Privacy Models Module (`privacy_models/`)

**Formal Privacy Models:**
- **k-anonymity**: Each record indistinguishable from k-1 others
- **l-diversity**: Each equivalence class has l diverse values
- **t-closeness**: Distribution close to original within threshold t
- **Differential Privacy**: ε-differential privacy guarantees

### 9. Task Orchestration (`utils/tasks/`)

**TaskRunner:**
- Pipeline execution with operation chaining
- Checkpoint/resume support
- Hierarchical progress tracking
- Seed management for reproducibility
- Manifest generation (full audit trail)

**Output Structure:**
```
task_dir/
├── manifest.json          # Full reproducibility record
├── output/                # Transformed data
├── metrics/               # Privacy/utility metrics
├── attacks/               # Attack simulation results
├── plots/                 # Visualizations
├── dictionaries/          # Extracted mappings
└── logs/                  # Execution logs
```

## Technology Stack

### Core Dependencies

| Category | Libraries | Version | Purpose |
|----------|-----------|---------|---------|
| **Data** | pandas, numpy, pyarrow, scipy | 2.2.2, 1.26.4, 14.0.2, 1.16.3 | Data processing |
| **Big Data** | dask[complete] | 2025.11.0 | Out-of-core processing |
| **ML/Stats** | scikit-learn, torch, sdv | 1.7.2, 2.8.0, 1.18.0 | Modeling and synthesis |
| **NLP** | spacy, nltk, langdetect, fasttext | 3.8.9, 3.9.2, 1.0.9, 0.9.2 | Text processing |
| **Visualization** | matplotlib, seaborn, plotly | 3.8.2, 0.13.2, 6.4.0 | Charts and plots |
| **Validation** | pydantic | 2.12.4 | Schema validation |
| **Testing** | pytest | 8.4.2 | Test framework |
| **Cryptography** | cryptography, bcrypt | 46.0.3, 4.3.0 | Secure operations |

### Python Version
- **Required**: Python 3.11-3.12 (strictly enforced)
- **Type Hints**: Extensive use of `typing` module
- **Async**: Limited async support (mostly synchronous)

## Data Flow Pattern

### Typical Pipeline Flow

```
1. LOAD DATA
   └─> pamola_core/io/ adapters (CSV, JSON, Excel, Parquet)

2. PROFILE DATA
   └─> pamola_core/profiling/ analyzers
       └─> Identify quasi-identifiers, sensitive attributes

3. TRANSFORM/ANONYMIZE
   └─> pamola_core/anonymization/ operations
       └─> Mask, suppress, generalize, noise, pseudonymize

4. EVALUATE
   └─> pamola_core/metrics/ operations
       └─> Privacy, utility, fidelity metrics

5. TEST (Optional)
   └─> pamola_core/attacks/ operations
       └─> Linkage, singling-out, attribute inference

6. SYNTHESIZE (Optional)
   └─> pamola_core/fake_data/ operations
       └─> Generate synthetic data

7. REPORT
   └─> manifest.json with full audit trail
```

## Extension Points

### Adding New Operations

1. **Inherit from Base Class:**
   ```python
   from pamola_core.utils.ops.op_base import BaseOperation

   class MyCustomOperation(BaseOperation):
       def _execute(self, data):
           # Implementation
           pass
   ```

2. **Define Pydantic Schema:**
   ```python
   from pydantic import BaseModel, Field

   class MyCustomConfig(BaseModel):
       param1: str = Field(default="value")
       param2: int = Field(ge=0, le=100)
   ```

3. **Register Operation:**
   ```python
   from pamola_core.utils.ops.op_registry import register_operation

   @register_operation
   class MyCustomOperation(BaseOperation):
       pass
   ```

4. **Add Tests:**
   ```python
   # tests/my_custom_operation/test_my_custom_op.py
   def test_my_custom_operation():
       # Test implementation
       pass
   ```

### Adding New Analyzers

1. Create analyzer in `pamola_core/profiling/analyzers/`
2. Inherit from existing analyzer base classes
3. Add commons utilities to `pamola_core/profiling/commons/`
4. Register in profiling operation registry

### Adding New Metrics

1. Create metric in `pamola_core/metrics/{category}/`
2. Inherit from `MetricsOperation`
3. Define aggregation logic
4. Add verdict thresholds in config

## Testing Structure

### Test Organization

```
tests/
├── anonymization/
│   ├── masking/
│   ├── suppression/
│   ├── generalization/
│   └── commons/
├── profiling/
│   ├── analyzers/
│   └── commons/
├── transformations/
│   ├── merging/
│   ├── splitting/
│   └── commons/
├── fake_data/
│   ├── generators/
│   ├── dictionaries/
│   └── commons/
├── metrics/
│   ├── privacy/
│   ├── utility/
│   ├── fidelity/
│   └── commons/
├── utils/
│   ├── nlp/
│   └── ops/
└── conftest.py              # Shared fixtures
```

### Testing Patterns

1. **Internal Logic Testing**: Test `_internal_method` directly
2. **Disposable Resources**: Use `tempfile` for test assets
3. **Strategy Injection**: Test multiple algorithms via config
4. **Metric Verification**: Assert on `op.metrics` dictionary
5. **Direct Utility Testing**: Test helpers in isolation
6. **Stochastic Control**: Fix seeds for randomized operations
7. **Side-Effect Safety**: Verify non-target columns unchanged
8. **Edge Case Matrix**: Parameterized tests for edge cases

## Key Design Patterns

### 1. Operation-Based Framework
All operations inherit from `BaseOperation` with standardized lifecycle.

### 2. Schema-Driven Configuration
Pydantic schemas for all configurations with validation.

### 3. Registry Pattern
Dynamic operation registration and discovery via decorators.

### 4. Result Objects
Standardized `OperationResult` with status, artifacts, metrics.

### 5. Progress Tracking
Hierarchical progress tracking for nested operations.

### 6. Dual-Engine Support
Pandas and Dask DataFrames with auto-switching.

### 7. Caching
Operation result caching for performance optimization.

### 8. Task Orchestration
Dependency management and execution context for pipelines.

## Performance Characteristics

### Memory Efficiency
- Chunk-based processing for large datasets
- Dask integration for out-of-core computation
- Operation result caching
- Lazy evaluation in transformations

### Scalability
- Support for datasets up to 10M records
- Parallel processing via Dask
- Configurable chunk sizes
- Memory profiling utilities

### Caching Strategy
- Operation result caching
- NLP model caching
- Dictionary pre-loading
- Schema validation caching

## Security Considerations

### Cryptography
- SHA3 for irreversible hashing
- AES-256 for reversible pseudonymization
- Secure random number generation
- Encrypted mapping storage

### Data Protection
- No API keys or credentials in code
- Environment variable configuration
- Secure by default (masking, suppression)
- Audit trail via manifest.json

### Compliance Support
- GDPR: Pseudonymization, data minimization
- HIPAA: Safe Harbor de-identification
- CCPA: Data suppression, anonymization

## Current Limitations

1. **No Formal DP**: DP-semantics noise but no formal guarantees
2. **Text Handling**: Short text only (long text in separate package)
3. **Model-Centric Attacks**: Not included (in PAMOLA.SYNT)
4. **Cloud Storage**: No native cloud storage support
5. **Real-time Processing**: Batch processing only

## Future Enhancements

1. **OpenDP Integration**: Formal differential privacy
2. **Cloud Storage**: Native S3, GCS, Azure support
3. **Web UI**: Basic pipeline configuration interface
4. **Performance**: Optimize for 100M+ records
5. **Enterprise Features**: RBAC, audit logging, SSO

## References

- [architecture.md](./architecture.md) - Detailed architecture
- [module-catalog.md](./module-catalog.md) - Complete module listing
- [project-overview-pdr.md](./project-overview-pdr.md) - Product requirements
- [code-standards.md](./code-standards.md) - Development guidelines
- [system-architecture.md](./system-architecture.md) - Architecture with diagrams
