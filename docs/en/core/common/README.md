# Common Module Documentation Index

**Last Updated:** 2026-03-23
**Documentation Version:** 1.0
**Module:** `pamola_core.common`

## Overview

Complete documentation for the `pamola_core.common` module, providing shared utilities, enumerations, type definitions, constants, and helper functions used throughout PAMOLA.CORE.

## Documentation Structure

```
docs/en/core/common/
├── README.md                          (This file - navigation index)
├── common_overview.md                 (Module architecture & features)
├── constants.md                       (Global constants reference)
├── type_aliases.md                    (Type definitions & configs)
├── enum/                              (Enumeration documentation)
│   ├── enums_reference.md             (Quick reference of all enums)
│   ├── encryption_mode.md             (Encryption modes)
│   ├── mask_strategy.md               (Masking strategies)
│   ├── distance_metric_type.md        (Distance metrics)
│   ├── fidelity_metrics.md            (Statistical similarity metrics)
│   ├── privacy_metrics_type.md        (Privacy evaluation metrics)
│   └── [Additional enums available]
├── helpers/                           (Helper utilities)
│   └── helpers_overview.md            (Data manipulation helpers)
├── logging/                           (Logging functionality)
│   └── logging_config.md              (Privacy-aware logging)
├── validation/                        (Validation utilities)
│   └── validation_helpers.md          (Column validation)
└── regex/                             (Pattern references)
    └── (patterns.py documentation in patterns)
```

## Quick Navigation

### Getting Started
1. **New to the module?** Start with [common_overview.md](./common_overview.md)
2. **Need specific enum?** Check [enums_reference.md](./enum/enums_reference.md)
3. **Working with data?** See [helpers_overview.md](./helpers/helpers_overview.md)

### By Component

#### Enumerations (Type-Safe Configuration)
| Document | Contains | Purpose |
|----------|----------|---------|
| [enums_reference.md](./enum/enums_reference.md) | Quick reference table | All 17+ enums at a glance |
| [encryption_mode.md](./enum/encryption_mode.md) | EncryptionMode | Task encryption configuration |
| [mask_strategy.md](./enum/mask_strategy.md) | MaskStrategyEnum | Anonymization masking strategies |
| [distance_metric_type.md](./enum/distance_metric_type.md) | DistanceMetricType | Distance calculations |
| [fidelity_metrics.md](./enum/fidelity_metrics.md) | FidelityMetrics(Type) | Statistical similarity metrics |
| [privacy_metrics_type.md](./enum/privacy_metrics_type.md) | PrivacyMetricsType | Privacy preservation metrics |

#### Core Modules
| Document | Contains | Purpose |
|----------|----------|---------|
| [constants.md](./constants.md) | Constants class | Operation names, date formats, type mappings |
| [type_aliases.md](./type_aliases.md) | Type aliases, CryptoConfig | File path types, encryption configuration |
| [helpers_overview.md](./helpers/helpers_overview.md) | DataHelper, dataset_helper | Data processing and transformation |
| [logging_config.md](./logging/logging_config.md) | Privacy logging functions | Change tracking and audit trails |
| [validation_helpers.md](./validation/validation_helpers.md) | check_columns_exist() | Column validation utilities |

## Document Overview

### 1. common_overview.md
**High-level module architecture and features**
- Module structure and organization
- Core components and classes
- Dependencies and relationships
- Usage patterns
- Best practices
- Quick reference table

**Start here if:** You're new to the common module or need architectural context

### 2. constants.md
**Global constants and standards**
- Operation names (generalization, noise_addition)
- 40+ date format patterns
- Frequency mappings (D, W, M, Q, Y)
- Pandas dtype mappings
- Artifact categories
- Safe evaluation globals
- I/O configuration constants

**Use this when:** You need standard values or date formats

### 3. type_aliases.md
**Type definitions and configuration classes**
- PathLike and PathLikeOrList type aliases
- DataFrameType (pandas/dask support)
- CryptoConfig class (encryption settings)
- FileCryptoConfig class (file+crypto pairing)
- convert_to_flatten_dict() function
- Serialization methods

**Use this when:** Working with crypto, file paths, or type hints

### 4. Enumeration Documents

#### enums_reference.md
**Quick-reference of all enumerations**
- Summary table of all 17+ enums
- Member listings by category
- Detailed enum definitions
- Import patterns
- Selection guide by use case

**Start here for:** Finding the right enum for your task

#### Individual Enum Documents
Each enum has detailed documentation covering:
- Purpose and use cases
- All members with values and descriptions
- Detailed metric/strategy definitions
- Usage examples
- Best practices
- Related components

**Examples:**
- [encryption_mode.md](./enum/encryption_mode.md) - Task encryption (NONE, SIMPLE, AGE)
- [mask_strategy.md](./enum/mask_strategy.md) - Masking methods (FIXED, PATTERN, RANDOM, WORDS)
- [fidelity_metrics.md](./enum/fidelity_metrics.md) - Statistical similarity (KS, KL, JS, WASSERSTEIN)
- [privacy_metrics_type.md](./enum/privacy_metrics_type.md) - Privacy metrics (k-anonymity, l-diversity, etc.)

### 5. helpers_overview.md
**Data processing and transformation utilities**
- DataHelper class (40+ methods for data operations)
- dataset_helper (train-test splitting)
- math_helper (numeric utilities)
- data_profiler (analysis utilities)
- Comprehensive workflows for common tasks
- Type detection and conversion
- Privacy transformation utilities

**Use this when:** Processing, validating, or transforming data

### 6. logging_config.md
**Privacy-aware logging and audit trails**
- save_privacy_logging() - Save logs to JSON
- log_privacy_transformations() - Track individual changes
- convert_to_json_serializable() - Type conversion
- Audit trail workflows
- Logging configuration
- Error handling

**Use this when:** Tracking anonymization changes or creating audit trails

### 7. validation_helpers.md
**Column and field validation**
- check_columns_exist() - Validate DataFrame columns
- Error handling and exceptions
- Validation workflows
- Batch validation patterns
- Best practices

**Use this when:** Ensuring required columns exist before processing

## Common Workflows

### Workflow 1: Data Anonymization with Logging

```python
from pamola_core.common.helpers.data_helper import DataHelper
from pamola_core.common.logging.privacy_logging import log_privacy_transformations, save_privacy_logging
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum

# Validate input
check_columns_exist(df, ["age", "name"])

# Apply anonymization
change_log = {}
original_age = df["age"].copy()
df["age"] = DataHelper.bin_numeric(df["age"], num_bins=5)

# Log changes
log_privacy_transformations(
    change_log,
    "generalization",
    "age_binning",
    df["age"] != original_age,
    original_age,
    df["age"],
    ["age"]
)

# Save audit trail
save_privacy_logging(change_log, "audit.json")
```

### Workflow 2: Privacy-Utility Evaluation

```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics
from pamola_core.common.enum.distance_metric_type import DistanceMetricType

# Assess privacy preservation
privacy_metrics = {
    PrivacyMetricsType.K_ANONYMITY: compute_k_anonymity(df),
    PrivacyMetricsType.L_DIVERSITY: compute_l_diversity(df),
    PrivacyMetricsType.DCR: compute_dcr(df)
}

# Assess data utility
fidelity_metrics = {
    FidelityMetrics.KS: compute_ks(original, anonymized),
    FidelityMetrics.JS: compute_js(original, anonymized)
}
```

### Workflow 3: Encryption Configuration

```python
from pamola_core.common.type_aliases import CryptoConfig, FileCryptoConfig

# Create encryption config
crypto = CryptoConfig(
    mode="age",
    algorithm="AES-256",
    key_path="/secure/key.pem"
)

# Associate with files
file_config = FileCryptoConfig(
    file_paths=["/data/sensitive1.csv", "/data/sensitive2.csv"],
    crypto_config=crypto
)

# Serialize/deserialize
config_dict = file_config.to_dict()
restored_config = FileCryptoConfig.from_dict(config_dict)
```

## Learning Path

### Beginner (Just Starting)
1. Read [common_overview.md](./common_overview.md) - Get the big picture
2. Review [enums_reference.md](./enum/enums_reference.md) - Understand available types
3. Check [constants.md](./constants.md) - See standard values

### Intermediate (Building Solutions)
1. Study [helpers_overview.md](./helpers/helpers_overview.md) - Learn data utilities
2. Review [validation_helpers.md](./validation/validation_helpers.md) - Ensure data quality
3. Check specific enum docs for your use case

### Advanced (Complex Operations)
1. Deep dive into specific enum documentation (e.g., [fidelity_metrics.md](./enum/fidelity_metrics.md))
2. Study [logging_config.md](./logging/logging_config.md) - Build audit trails
3. Review [type_aliases.md](./type_aliases.md) - Advanced type handling

## Key Concepts

### Type Safety
All configuration is done through enumerations, not strings, preventing typos and enabling IDE autocomplete.

```python
# Good - type-safe
mode = EncryptionMode.AGE

# Avoid - error-prone
mode = "age"
```

### Constants Over Hardcoding
Central constants ensure consistency and maintainability.

```python
# Good
date_fmt = Constants.COMMON_DATE_FORMATS[0]

# Avoid
date_fmt = "%Y-%m-%d"
```

### Privacy by Design
Logging and validation are built in to support audit trails and data quality.

```python
# Good - tracks changes
log_privacy_transformations(...)
save_privacy_logging(...)

# Avoid - no audit trail
# Process without logging
```

## Related Documentation

- **Parent:** [`pamola_core` Main Documentation](../)
- **Sibling:** [anonymization](../anonymization/), [metrics](../metrics/), [helpers](../utils/)
- **Downstream:** All modules depend on common utilities

## Quick Import Reference

```python
# Enumerations
from pamola_core.common.enum.encryption_mode import EncryptionMode
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

# Constants and Types
from pamola_core.common.constants import Constants
from pamola_core.common.type_aliases import CryptoConfig, PathLike, DataFrameType

# Helpers
from pamola_core.common.helpers.data_helper import DataHelper
from pamola_core.common.helpers.dataset_helper import split_dataset

# Logging
from pamola_core.common.logging.privacy_logging import save_privacy_logging, log_privacy_transformations

# Validation
from pamola_core.common.validation.check_column import check_columns_exist
```

## Contributing

When adding to this documentation:
1. Follow the existing template structure
2. Include practical examples
3. Document related components
4. Update this index
5. Keep individual files under 300 lines
6. Use type hints in code examples

## Maintenance

Documentation is maintained alongside source code. When:
- **Adding new enum**: Create enum documentation file + update enums_reference.md
- **Adding new constant**: Update constants.md
- **Adding new helper**: Update helpers_overview.md
- **Breaking changes**: Update all affected documentation files

## Support

For questions about the common module:
1. Check the appropriate documentation file
2. Review code examples in the documentation
3. Check related component documentation
4. File an issue with specific questions

## Changelog

**v1.0 (2026-03-23)**
- Initial comprehensive documentation
- 12 documentation files covering all components
- Complete enum references
- Practical workflow examples
- Best practices guide
