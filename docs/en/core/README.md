# PAMOLA.CORE Documentation Index

**Last Updated:** 2026-03-27
**Version:** 0.0.1
**Python:** 3.10-3.12
**Total Docs:** 221 files (Tier 1 + Tier 2)

---

## Tier 1 — Public API

### [Anonymization](./anonymization/) (13 docs)
Privacy-preserving transformations: masking, generalization, noise, suppression, pseudonymization.

| Doc | Source |
|-----|--------|
| [Base Anonymization Op](./anonymization/base_anonymization_op.md) | `BaseAnonymizationOperation` |
| [Categorical Generalization](./anonymization/generalization/categorical.md) | hierarchy, frequency, rare merge |
| [Numeric Generalization](./anonymization/generalization/numeric.md) | binning, rounding, range |
| [Datetime Generalization](./anonymization/generalization/datetime_op.md) | rounding, binning, component, relative |
| [Full Masking](./anonymization/masking/full_masking_op.md) | complete field masking |
| [Partial Masking](./anonymization/masking/partial_masking_op.md) | fixed, pattern, random, words |
| [Uniform Numeric Noise](./anonymization/noise/uniform_numeric_op.md) | additive/multiplicative noise |
| [Uniform Temporal Noise](./anonymization/noise/uniform_temporal_op.md) | temporal shift noise |
| [Hash-based Pseudonymization](./anonymization/pseudonymization/hash_based_op.md) | SHA3, output formats |
| [Mapping Pseudonymization](./anonymization/pseudonymization/mapping_op.md) | reversible UUID/sequential |
| [Attribute Suppression](./anonymization/suppression/attribute_op.md) | column removal |
| [Cell Suppression](./anonymization/suppression/cell_op.md) | 7 strategies: null, mean, median, mode, constant, group_mean, group_mode |
| [Record Suppression](./anonymization/suppression/record_op.md) | row removal by condition |

### [Profiling](./profiling/) (14 docs)
Data profiling analyzers for field-level analysis.

| Doc | Source |
|-----|--------|
| [Anonymity](./profiling/anonymity.md) | k-anonymity assessment |
| [Attribute](./profiling/attribute.md) | column role classification |
| [Categorical](./profiling/categorical.md) | frequency, distribution, anomaly |
| [Correlation](./profiling/correlation.md) | Pearson, Spearman, Kendall, Cramer's V |
| [Currency](./profiling/currency.md) | currency detection, parsing |
| [Date](./profiling/date.md) | date parsing, distribution, anomaly |
| [Email](./profiling/email.md) | email validation, domain analysis |
| [Email Dask](./profiling/email_dask.md) | distributed email analysis |
| [Group](./profiling/group.md) | group variation, cross-group |
| [Identity](./profiling/identity.md) | identifier statistics |
| [MVF](./profiling/mvf.md) | multi-valued field parsing |
| [Numeric](./profiling/numeric.md) | statistics, normality, outliers |
| [Phone](./profiling/phone.md) | phone parsing, country codes |
| [Text](./profiling/text.md) | text analysis, categorization |

### [Transformations](./transformations/) (9 docs)
Data transformation operations.

| Doc | Source |
|-----|--------|
| [Base Transformation Op](./transformations/base_transformation_op.md) | `BaseTransformationOperation` |
| [Clean Invalid Values](./transformations/cleaning/clean_invalid_values.md) | null/invalid handling |
| [Add/Modify Fields](./transformations/field_ops/add_modify_fields.md) | field creation/modification |
| [Remove Fields](./transformations/field_ops/remove_fields.md) | field removal |
| [Aggregate Records](./transformations/grouping/aggregate_records_op.md) | group-by aggregation |
| [Impute Missing Values](./transformations/imputation/impute_missing_values.md) | statistical imputation |
| [Merge Datasets](./transformations/merging/merge_datasets_op.md) | multi-dataset merge |
| [Split by ID Values](./transformations/splitting/split_by_id_values_op.md) | partition by ID |
| [Split Fields](./transformations/splitting/split_fields_op.md) | field-group splitting |

### [Metrics](./metrics/) (22 docs)
Privacy, fidelity, utility, and quality metrics.

| Doc | Source |
|-----|--------|
| [Base Metrics Op](./metrics/base_metrics_op.md) | `MetricsOperation` |
| [Fidelity Ops](./metrics/operations/fidelity_ops.md) | fidelity metric operation |
| [Privacy Ops](./metrics/operations/privacy_ops.md) | privacy metric operation |
| [Utility Ops](./metrics/operations/utility_ops.md) | utility metric operation |
| [KL Divergence](./metrics/fidelity/distribution/kl_divergence.md) | information-theoretic |
| [KS Test](./metrics/fidelity/distribution/ks_test.md) | non-parametric distribution |
| [Statistical Fidelity](./metrics/fidelity/statistical_fidelity.md) | property preservation |
| [Distance (DCR)](./metrics/privacy/distance.md) | distance to closest record |
| [Identity](./metrics/privacy/identity.md) | identity disclosure |
| [Neighbor (NNDR)](./metrics/privacy/neighbor.md) | nearest neighbor distance ratio |
| [Disclosure Risk](./metrics/privacy/disclosure_risk.md) | prosecutor/journalist/marketer |
| [Classification](./metrics/utility/classification.md) | F1, precision, recall |
| [Regression](./metrics/utility/regression.md) | MSE, R2 |
| [Quality metrics](./metrics/quality/) | KS, KL, Pearson, Wasserstein |

### [Fake Data](./fake_data/) (5 docs)
Synthetic data generation operations.

| Doc | Source |
|-----|--------|
| [Base Generator Op](./fake_data/base_generator_op.md) | `GeneratorOperation` |
| [Email Op](./fake_data/operations/email_op.md) | `FakeEmailOperation` |
| [Name Op](./fake_data/operations/name_op.md) | `FakeNameOperation` |
| [Organization Op](./fake_data/operations/organization_op.md) | `FakeOrganizationOperation` |
| [Phone Op](./fake_data/operations/phone_op.md) | `FakePhoneOperation` |

### [Analysis](./analysis/) (7 docs)
Dataset-level analysis functions.

| Doc | Source |
|-----|--------|
| [Dataset Summary](./analysis/dataset_summary.md) | `analyze_dataset_summary()` |
| [Privacy Risk](./analysis/privacy_risk.md) | `calculate_full_risk()` |
| [Descriptive Stats](./analysis/descriptive_stats.md) | `analyze_descriptive_stats()` |
| [Distribution](./analysis/distribution.md) | `visualize_distribution_df()` |
| [Correlation](./analysis/correlation.md) | `analyze_correlation()` |
| [Field Analysis](./analysis/field_analysis.md) | `analyze_field_level()` |

### [I/O](../../../pamola_core/io/) (2 docs)
Data readers: CSV, JSON, Excel, Parquet.

| Doc | Source |
|-----|--------|
| [I/O Guide](./utils/io.md) | `read_csv`, `read_json`, `read_excel`, `read_parquet` |

### [Errors](./errors/) (7 docs)
Error handling framework.

| Doc | Source |
|-----|--------|
| [Overview](./errors/errors_overview.md) | Module architecture |
| [BasePamolaError](./errors/base_pamola_error.md) | Base exception class |
| [auto_exception](./errors/auto_exception.md) | Exception decorator |
| [ErrorHandler](./errors/error_handler.md) | Centralized handler |
| [ErrorCode](./errors/error_codes.md) | 80+ error codes |
| [TaskInitializationError](./errors/task_initialization_error.md) | Task-specific error |

---

## Tier 2 — Key Abstractions

### [Privacy Models](./privacy_models/) (15 docs)
Privacy model processors.

| Doc | Source |
|-----|--------|
| [Overview](./privacy_models/privacy_models_overview.md) | Comparison table |
| [k-Anonymity](./privacy_models/k_anonymity/) | 3 docs: processor, report, visualization |
| [l-Diversity](./privacy_models/l_diversity/) | 8 docs: calculator, report, metrics, risk, strategies |
| [t-Closeness](./privacy_models/t_closeness/t_closeness_processor.md) | Wasserstein distance |
| [Differential Privacy](./privacy_models/differential_privacy/dp_processor.md) | Laplace/Gaussian noise |

### [Attacks](./attacks/) (9 docs)
Privacy attack simulation.

| Doc | Source |
|-----|--------|
| [Overview](./attacks/attacks_overview.md) | Attack comparison |
| [Linkage Attack](./attacks/linkage_attack.md) | exact, probabilistic, cluster-vector |
| [Membership Inference](./attacks/membership_inference.md) | DCR, NNDR, model-based |
| [Attribute Inference](./attacks/attribute_inference.md) | entropy-based |
| [DCR](./attacks/distance_to_closest_record.md) | distance metric |
| [NNDR](./attacks/nearest_neighbor_distance_ratio.md) | nearest neighbor |
| [Attack Metrics](./attacks/attack_metrics.md) | accuracy, ASR, RRS |

### [CLI](./cli/) (9 docs)
Command-line interface.

| Doc | Source |
|-----|--------|
| [Overview](./cli/cli_overview.md) | CLI system |
| [list-ops](./cli/commands/list_ops.md) | Operation discovery |
| [run](./cli/commands/run.md) | Task/operation execution |
| [schema](./cli/commands/schema.md) | Parameter inspection |
| [validate](./cli/commands/validate.md) | Config validation |

### [Utils/Ops](./utils/ops/) (10 docs)
Operation framework.

| Doc | Source |
|-----|--------|
| [BaseOperation](./utils/ops/op_base.md) | Core operation lifecycle |
| [OperationConfig](./utils/ops/op_config.md) | Schema-based config |
| [DataSource](./utils/ops/op_data_source.md) | Data access |
| [DataWriter](./utils/ops/op_data_writer.md) | Output writing |
| [OperationResult](./utils/ops/op_result.md) | Result container |
| [OperationRegistry](./utils/ops/op_registry.md) | Op discovery |
| [DataReader](./utils/ops/op_data_reader.md) | Unified reading |

### [Utils/Tasks](./utils/tasks/) (18 docs)
Task orchestration framework.

| Doc | Source |
|-----|--------|
| [BaseTask](./utils/tasks/base_task.md) | Task lifecycle |
| [TaskConfig](./utils/tasks/task_config.md) | Configuration loading |
| [TaskRunner](./utils/tasks/task_runner.md) | Operation orchestration |
| [TaskContext](./utils/tasks/task_context.md) | Reproducibility |
| + 14 manager docs | dependency, encryption, progress, path security, etc. |

### [Utils/NLP](./utils/nlp/) (29 docs)
NLP subsystem: tokenization, entity extraction, LLM integration.

| Doc | Source |
|-----|--------|
| [Base](./utils/nlp/base.md) | DependencyManager, normalize_language_code |
| [Cache](./utils/nlp/cache.md) | MemoryCache, FileCache, ModelCache |
| [Tokenization](./utils/nlp/tokenization.md) | SpaCyTokenizer, TransformersTokenizer |
| [Entity/](./utils/nlp/entity/) | 5 entity extractors |
| [LLM/](./utils/nlp/LLM/) | 9 LLM integration docs |
| + 13 utility docs | clustering, minhash, stopwords, etc. |

### [Common](./common/) (20 docs)
Shared enumerations, constants, helpers.

| Doc | Source |
|-----|--------|
| [Constants](./common/constants.md) | 40+ date formats, operation names |
| [Type Aliases](./common/type_aliases.md) | CryptoConfig, FileCryptoConfig |
| [Enums](./common/enum/) | 12 enum docs (EncryptionMode, MaskStrategy, etc.) |
| [Helpers](./common/helpers/) | DataHelper, math_helper, data_profiler |

### [Configs](./configs/) (4 docs) | [Catalogs](./catalogs/) (2 docs)

See detailed listings above.

---

## Quick Start

```python
from pamola_core import (
    # Anonymization
    CategoricalGeneralizationOperation,
    FullMaskingOperation,
    UniformNumericNoiseOperation,
    # Profiling
    NumericAnalysisOperation,
    CategoricalAnalysisOperation,
    # Metrics
    FidelityMetricOperation,
    PrivacyMetricOperation,
    # Analysis
    analyze_dataset_summary,
    calculate_full_risk,
    # I/O
    read_csv, read_json,
)
```

---

## Sync Prompts

Run after code changes:

```
Please run the docs sync prompt at plans/prompts/prompt-update-docs-pamola-core.md
Please run the tests sync prompt at plans/prompts/prompt-update-tests-pamola-core.md
```
