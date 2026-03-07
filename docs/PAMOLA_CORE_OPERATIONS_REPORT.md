# PAMOLA CORE - Operations Report

**Generated:** 2026-03-07
**Repository:** DGT-Network/PAMOLA
**Package:** `pamola_core`

---

## Table of Contents

1. [Operations Framework Overview](#operations-framework-overview)
2. [Operations Summary](#operations-summary)
3. [Profiling Operations](#1-profiling--pamolacoreprofiling)
4. [Anonymization Operations](#2-anonymization--pamolacoreanonimization)
5. [Fake Data Generation Operations](#3-fake-data-generation--pamolacoresfakedata)
6. [Metrics Operations](#4-metrics--pamolacoremetrics)
7. [Transformation Operations](#5-transformations--pamolacoretransformations)
8. [Attack Simulation (Legacy)](#6-attack-simulation-legacy--pamolacoreattacks)
9. [Analysis Utilities (Non-framework)](#7-analysis-utilities-non-framework--pamolacoreanalysis)
10. [Privacy Models (Non-framework)](#8-privacy-models-non-framework--pamolacoreprivacymodels)

---

## Operations Framework Overview

PAMOLA CORE uses a structured operations framework located in `pamola_core/utils/ops/`. Operations are built around several key components:

| Component | Module | Description |
|-----------|--------|-------------|
| `BaseOperation` | `op_base.py` | Abstract base class with lifecycle management (`run()` / `execute()`) |
| `FieldOperation` | `op_base.py` | Base for per-field operations (extends `BaseOperation`) |
| `DataFrameOperation` | `op_base.py` | Base for whole-DataFrame operations (extends `BaseOperation`) |
| `OperationConfig` | `op_config.py` | JSON Schema-validated configuration |
| `OperationResult` | `op_result.py` | Result/artifact/metrics collection |
| `OperationRegistry` | `op_registry.py` | Auto-discovery and registration via `@register` decorator |
| `OperationCache` | `op_cache.py` | Thread-safe caching with staleness/size management |
| `DataSource` | `op_data_source.py` | Unified DataFrame/file data access |
| `DataWriter` | `op_data_writer.py` | Structured output with encryption support |

### Readiness Levels

| Level | Description |
|-------|-------------|
| **Production** | Full framework integration, registered via `@register`, config schema, tests, visualization |
| **Stable** | Full framework integration, may lack some optional features (e.g., Dask support) |
| **Draft** | Implements framework interfaces but may have incomplete features |
| **Legacy** | Does not use the ops framework; uses older patterns (ABC, standalone classes) |
| **Utility** | Standalone functions/classes, not operations per se |

---

## Operations Summary

| Category | Operations | Framework-based | Legacy/Utility |
|----------|-----------|-----------------|----------------|
| Profiling | 15 | 15 | 0 |
| Anonymization | 10 | 10 | 0 |
| Fake Data | 4 | 4 | 0 |
| Metrics | 3 | 3 | 0 |
| Transformations | 8 | 8 | 0 |
| Attacks | 6 | 0 | 6 |
| Analysis | 5 | 0 | 5 |
| Privacy Models | 4 | 0 | 4 |
| **Total** | **55** | **40** | **15** |

---

## 1. Profiling (`pamola_core.profiling`)

Base class: `FieldOperation` / `BaseOperation` (from `utils/ops/op_base.py`)
All registered via `@register(version="1.0.0")` | Module version: 2.0.0

### 1.1 `profiling.analyzers.categorical` -- CategoricalOperation

| | |
|---|---|
| **Class** | `CategoricalOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Frequency distribution, cardinality metrics, anomaly detection, and dictionary creation for categorical fields |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `top_n` | `int` | `15` | Number of top values to include |
| `min_frequency` | `int` | `1` | Minimum frequency for dictionary inclusion |
| `profile_type` | `str` | `"categorical"` | Profile type identifier |
| `analyze_anomalies` | `bool` | `True` | Enable anomaly detection (typos, rare values) |

---

### 1.2 `profiling.analyzers.numeric` -- NumericOperation

| | |
|---|---|
| **Class** | `NumericOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Statistical analysis, distribution profiling, outlier detection, and normality testing for numeric fields |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `bins` | `int` | `10` | Number of histogram bins |
| `detect_outliers` | `bool` | `True` | Enable outlier detection |
| `test_normality` | `bool` | `True` | Enable normality tests |
| `near_zero_threshold` | `float` | `1e-10` | Threshold for near-zero detection |
| `profile_type` | `str` | `"numeric"` | Profile type identifier |

---

### 1.3 `profiling.analyzers.date` -- DateOperation

| | |
|---|---|
| **Class** | `DateOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Date/datetime field profiling with temporal distribution, anomaly detection, and format analysis |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `min_year` | `int` | `1940` | Minimum valid year for anomaly detection |
| `max_year` | `int` | `2005` | Maximum valid year for anomaly detection |
| `id_column` | `str` | `None` | Optional ID column reference |
| `uid_column` | `str` | `None` | Optional UID column reference |
| `profile_type` | `str` | `"date"` | Profile type identifier |
| `is_birth_date` | `bool` | `None` | Hint for birth date-specific analysis |

---

### 1.4 `profiling.analyzers.email` -- EmailOperation

| | |
|---|---|
| **Class** | `EmailOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Email field profiling: domain distribution, format validation, privacy risk analysis |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `top_n` | `int` | `20` | Number of top domains |
| `min_frequency` | `int` | `1` | Minimum frequency for dictionary |
| `profile_type` | `str` | `"email"` | Profile type identifier |
| `analyze_privacy_risk` | `bool` | `True` | Enable privacy risk analysis |

> **Note:** A Dask-optimized variant exists in `email_dask.py` with identical interface.

---

### 1.5 `profiling.analyzers.phone` -- PhoneOperation

| | |
|---|---|
| **Class** | `PhoneOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Phone number profiling: format detection, country/operator code analysis, validation |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `min_frequency` | `int` | `1` | Minimum frequency for dictionary |
| `patterns_csv` | `str` | `None` | Path to custom phone patterns CSV |
| `country_codes` | `List[str]` | `None` | List of expected country codes |

---

### 1.6 `profiling.analyzers.currency` -- CurrencyOperation

| | |
|---|---|
| **Class** | `CurrencyOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Currency/monetary field profiling: parsing, statistical analysis, outlier detection, locale-aware formatting |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `locale` | `str` | `"en_US"` | Locale for currency parsing |
| `bins` | `int` | `10` | Number of histogram bins |
| `detect_outliers` | `bool` | `True` | Enable outlier detection |
| `test_normality` | `bool` | `True` | Enable normality tests |

---

### 1.7 `profiling.analyzers.text` -- TextSemanticCategorizerOperation

| | |
|---|---|
| **Class** | `TextSemanticCategorizerOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Semantic categorization of text fields: NER, clustering, dictionary matching, topic extraction |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `id_field` | `str` | `None` | ID field for reference |
| `entity_type` | `str` | `"generic"` | Expected entity type |
| `dictionary_path` | `str/Path` | `None` | Path to external dictionary |
| `min_word_length` | `int` | `3` | Minimum word length for analysis |
| `clustering_threshold` | `float` | `0.7` | Similarity threshold for clustering |
| `use_ner` | `bool` | `True` | Enable Named Entity Recognition |
| `perform_categorization` | `bool` | `True` | Enable semantic categorization |
| `perform_clustering` | `bool` | `True` | Enable value clustering |
| `match_strategy` | `str` | `"specific_first"` | Dictionary matching strategy |

---

### 1.8 `profiling.analyzers.mvf` -- MVFOperation

| | |
|---|---|
| **Class** | `MVFOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Multi-Value Field (MVF) profiling: parsing, distribution analysis, and dictionary creation for fields containing multiple values |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `top_n` | `int` | `20` | Number of top values |
| `min_frequency` | `int` | `1` | Minimum frequency for dictionary |
| `format_type` | `str` | `None` | Expected value format |
| `parse_kwargs` | `Dict` | `None` | Additional parsing parameters |

---

### 1.9 `profiling.analyzers.correlation` -- CorrelationOperation

| | |
|---|---|
| **Class** | `CorrelationOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Pairwise correlation analysis between two fields with automatic method selection (Pearson, Spearman, Cramer's V, etc.) |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field1` | `str` | *required* | First field name |
| `field2` | `str` | *required* | Second field name |
| `method` | `str` | `None` | Correlation method (auto-detected if `None`) |
| `null_handling` | `str` | `"drop"` | How to handle nulls |
| `mvf_parser` | `str` | `None` | MVF parser for multi-value fields |

---

### 1.10 `profiling.analyzers.correlation` -- CorrelationMatrixOperation

| | |
|---|---|
| **Class** | `CorrelationMatrixOperation` |
| **Extends** | `BaseOperation` |
| **Readiness** | Production |
| **Description** | Full correlation matrix computation across multiple fields with heatmap visualization |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fields` | `List[str]` | *required* | Fields to include in the matrix |
| `methods` | `Dict[str, str]` | `None` | Per-field method overrides |
| `min_threshold` | `float` | `0.3` | Minimum correlation threshold for display |
| `null_handling` | `str` | `"drop"` | How to handle nulls |

---

### 1.11 `profiling.analyzers.identity` -- IdentityAnalysisOperation

| | |
|---|---|
| **Class** | `IdentityAnalysisOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Identity field analysis: uniqueness, cross-match detection, similarity analysis, and privacy risk assessment |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uid_field` | `str` | *required* | Primary UID field |
| `reference_fields` | `List[str]` | *required* | Fields to cross-reference |
| `id_field` | `str` | `None` | Optional secondary ID field |
| `top_n` | `int` | `15` | Number of top results |
| `check_cross_matches` | `bool` | `True` | Enable cross-field matching |
| `min_similarity` | `float` | `0.8` | Minimum similarity for matching |
| `fuzzy_matching` | `bool` | `False` | Enable fuzzy string matching |

---

### 1.12 `profiling.analyzers.anonymity` -- KAnonymityProfilerOperation

| | |
|---|---|
| **Class** | `KAnonymityProfilerOperation` |
| **Extends** | `BaseOperation` |
| **Readiness** | Production |
| **Description** | K-anonymity profiling: equivalence class analysis, risk scoring, and optional DataFrame enrichment with k-values |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"KAnonymityProfiler"` | Operation name |
| `quasi_identifiers` | `List[str]` | `None` | QI fields to analyze |
| `analysis_mode` | `str` | `"ANALYZE"` | Mode: `ANALYZE`, `ENRICH`, or `BOTH` |
| `threshold_k` | `int` | `5` | Minimum acceptable k value |
| `export_metrics` | `bool` | `True` | Export metrics to JSON |
| `max_combinations` | `int` | `50` | Max QI combinations to test |
| `output_field_suffix` | `str` | `"k_anon"` | Suffix for enrichment column |
| `quasi_identifier_sets` | `List[List[str]]` | `None` | Predefined QI sets |
| `id_fields` | `List[str]` | `None` | ID fields to exclude |

---

### 1.13 `profiling.analyzers.attribute` -- DataAttributeProfilerOperation

| | |
|---|---|
| **Class** | `DataAttributeProfilerOperation` |
| **Extends** | `BaseOperation` |
| **Readiness** | Production |
| **Description** | Automatic attribute type detection and classification (PII, quasi-identifiers, sensitive data) for all dataset columns |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"DataAttributeProfiler"` | Operation name |
| `dictionary_path` | `str/Path` | `None` | Custom dictionary path |
| `language` | `str` | `"english"` | Language for NLP analysis |
| `sample_size` | `int` | `10` | Sample rows for type inference |
| `max_columns` | `int` | `None` | Max columns to process |

---

### 1.14 `profiling.analyzers.group` -- GroupAnalyzerOperation

| | |
|---|---|
| **Class** | `GroupAnalyzerOperation` |
| **Extends** | `FieldOperation` |
| **Readiness** | Production |
| **Description** | Group-level analysis: variance, distribution, hashing, MinHash similarity within grouped data |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Field to analyze |
| `fields_config` | `Dict[str, int]` | *required* | Fields configuration with thresholds |
| `text_length_threshold` | `int` | `100` | Max text length for hashing |
| `variance_threshold` | `float` | `0.2` | Variance significance threshold |
| `large_group_threshold` | `int` | `100` | Threshold for large group handling |
| `large_group_variance_threshold` | `float` | `0.05` | Variance threshold for large groups |
| `hash_algorithm` | `str` | `"md5"` | Hash algorithm |
| `minhash_similarity_threshold` | `float` | `0.7` | MinHash similarity threshold |

---

### 1.15 `profiling.analyzers.longtext` -- (Placeholder)

| | |
|---|---|
| **Class** | N/A |
| **Readiness** | Draft (empty file) |
| **Description** | Reserved for long text field profiling |
| **Updated** | 2025-08-08 |

---

## 2. Anonymization (`pamola_core.anonymization`)

Base class: `AnonymizationOperation` (extends `BaseOperation`)
Module version: 3.0.0 | All registered via `@register`

### 2.1 Generalization (`anonymization.generalization`)

#### 2.1.1 `generalization.numeric_op` -- NumericGeneralizationOperation

| | |
|---|---|
| **Class** | `NumericGeneralizationOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Description** | Generalize numeric fields via binning, rounding, or range-based strategies |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `strategy` | `str` | *required* | `"binning"`, `"rounding"`, or `"range"` |
| `bin_count` | `int` | varies | Number of bins (binning strategy) |
| `precision` | `int` | varies | Rounding precision |
| `range_limits` | `List` | `None` | Custom range boundaries |

---

#### 2.1.2 `generalization.categorical_op` -- CategoricalGeneralizationOperation

| | |
|---|---|
| **Class** | `CategoricalGeneralizationOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="4.0.0")` |
| **Description** | Categorical data generalization with frequency-based, hierarchy, and merge strategies |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `strategy` | `str` | *required* | Generalization strategy |
| `category_mappings` | `Dict` | `None` | Custom category mappings |
| `hierarchy_config` | `Dict` | `None` | Hierarchy configuration |

---

#### 2.1.3 `generalization.datetime_op` -- DateTimeGeneralizationOperation

| | |
|---|---|
| **Class** | `DateTimeGeneralizationOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Description** | Generalize datetime fields via rounding, binning, component-based, or relative strategies |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `strategy` | `str` | *required* | `"rounding"`, `"binning"`, `"component"`, `"relative"` |
| `rounding_unit` | `str` | varies | Rounding unit (year, month, day, etc.) |
| `bin_type` | `str` | `None` | Bin type for binning strategy |
| `interval_size` | `int` | `None` | Interval size for bins |
| `reference_date` | `str` | `None` | Reference date for relative strategy |
| `custom_bins` | `List` | `None` | Custom bin boundaries |

---

### 2.2 Masking (`anonymization.masking`)

#### 2.2.1 `masking.full_masking_op` -- FullMaskingOperation

| | |
|---|---|
| **Class** | `FullMaskingOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="4.0.0")` |
| **Description** | Complete value replacement with mask characters, with optional format preservation |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `mask_char` | `str` | `"*"` | Mask character |
| `preserve_length` | `bool` | varies | Preserve original value length |
| `fixed_length` | `int` | `None` | Fixed mask length |
| `random_mask` | `bool` | `False` | Use random mask characters |
| `preserve_format` | `bool` | `False` | Preserve value format (e.g., email structure) |
| `format_patterns` | `Dict` | `None` | Custom format preservation patterns |

---

#### 2.2.2 `masking.partial_masking_op` -- PartialMaskingOperation

| | |
|---|---|
| **Class** | `PartialMaskingOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="4.0.0")` |
| **Description** | Selective masking of value portions while preserving specified prefix/suffix parts |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `mask_char` | `str` | `"*"` | Mask character |
| `mask_strategy` | `str` | varies | Masking strategy |
| `mask_percentage` | `float` | varies | Percentage of value to mask |
| `unmasked_prefix` | `int` | `0` | Characters to keep at start |
| `unmasked_suffix` | `int` | `0` | Characters to keep at end |

---

### 2.3 Noise (`anonymization.noise`)

#### 2.3.1 `noise.uniform_numeric_op` -- UniformNumericNoiseOperation

| | |
|---|---|
| **Class** | `UniformNumericNoiseOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="1.0.0")` |
| **Description** | Add uniformly distributed random noise to numeric fields |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `noise_range` | `float` | *required* | Noise amplitude range |
| `noise_type` | `str` | `"additive"` | `"additive"` or `"multiplicative"` |
| `output_min` | `float` | `None` | Minimum output value clamp |
| `output_max` | `float` | `None` | Maximum output value clamp |
| `preserve_zero` | `bool` | `False` | Keep zero values unchanged |
| `scale_by_std` | `bool` | `False` | Scale noise by field's std deviation |

---

#### 2.3.2 `noise.uniform_temporal_op` -- UniformTemporalNoiseOperation

| | |
|---|---|
| **Class** | `UniformTemporalNoiseOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="1.0.0")` |
| **Description** | Add uniform random time shifts to datetime fields |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `noise_range_days` | `int` | `0` | Noise range in days |
| `noise_range_hours` | `int` | `0` | Noise range in hours |
| `noise_range_minutes` | `int` | `0` | Noise range in minutes |
| `noise_range_seconds` | `int` | `0` | Noise range in seconds |
| `direction` | `str` | `"both"` | `"both"`, `"forward"`, or `"backward"` |
| `min_datetime` | `str` | `None` | Minimum datetime boundary |
| `max_datetime` | `str` | `None` | Maximum datetime boundary |
| `preserve_special_dates` | `bool` | `False` | Keep special dates unchanged |

---

### 2.4 Pseudonymization (`anonymization.pseudonymization`)

#### 2.4.1 `pseudonymization.hash_based_op` -- HashBasedPseudonymizationOperation

| | |
|---|---|
| **Class** | `HashBasedPseudonymizationOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Description** | Irreversible hash-based pseudonymization with cryptographic security |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `algorithm` | `str` | `"SHA3-256"` | Hash algorithm (`SHA3-256`, `SHA3-512`) |
| `salt` | `str` | `None` | Salt value for hashing |
| `pepper` | `str` | `None` | Pepper value (stored separately) |
| `output_format` | `str` | `"hex"` | `"hex"`, `"base64"`, or `"uuid"` |

---

#### 2.4.2 `pseudonymization.mapping_op` -- MappingPseudonymizationOperation

| | |
|---|---|
| **Class** | `MappingPseudonymizationOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Description** | Reversible mapping-based pseudonymization with encrypted mapping storage |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `pseudonym_type` | `str` | `"UUID"` | `"UUID"`, `"sequential"`, or `"random_string"` |
| `mapping_storage_path` | `str/Path` | `None` | Path to store/load mapping file |
| `encryption_key` | `str` | `None` | Key for mapping file encryption |

---

### 2.5 Suppression (`anonymization.suppression`)

#### 2.5.1 `suppression.attribute_op` -- AttributeSuppressionOperation

| | |
|---|---|
| **Class** | `AttributeSuppressionOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="1.0.0")` |
| **Description** | Remove entire columns (attributes) from datasets |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Primary field to suppress |
| `additional_fields` | `List[str]` | `None` | Additional fields to suppress |
| `suppression_mode` | `str` | varies | Suppression mode |
| `save_suppressed_schema` | `bool` | `False` | Save schema of suppressed fields |

---

#### 2.5.2 `suppression.cell_op` -- CellSuppressionOperation

| | |
|---|---|
| **Class** | `CellSuppressionOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="1.0.0")` |
| **Description** | Replace individual cell values with NULL, statistical substitutes, or constants |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `suppression_strategy` | `str` | `"null"` | `"null"`, `"mean"`, `"median"`, `"mode"`, `"group_mean"`, `"group_mode"`, `"constant"` |
| `outlier_method` | `str` | `None` | Outlier detection method |
| `rare_value_threshold` | `float` | `None` | Threshold for rare value detection |

---

#### 2.5.3 `suppression.record_op` -- RecordSuppressionOperation

| | |
|---|---|
| **Class** | `RecordSuppressionOperation` |
| **Extends** | `AnonymizationOperation` |
| **Readiness** | Production |
| **Registered** | `@register(version="1.0.0")` |
| **Description** | Remove entire rows (records) based on configurable conditions |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field for condition evaluation |
| `suppression_condition` | `str` | `"null"` | `"null"`, `"value"`, `"range"`, `"risk"`, `"custom"` |
| `suppression_values` | `List` | `None` | Values triggering suppression |
| `suppression_range` | `Dict` | `None` | Range boundaries for suppression |
| `save_suppressed_records` | `bool` | `False` | Save suppressed records to separate file |

---

## 3. Fake Data Generation (`pamola_core.fake_data`)

Base class: `GeneratorOperation` (extends `BaseOperation`)
Module version: 3.0.0 | All registered via `@register(version="1.0.0")`

### 3.1 `fake_data.operations.name_op` -- FakeNameOperation

| | |
|---|---|
| **Class** | `FakeNameOperation` |
| **Extends** | `GeneratorOperation` |
| **Readiness** | Production |
| **Description** | Synthetic name generation preserving linguistic and gender characteristics |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `language` | `str` | `None` | Language/locale for names |
| `gender_field` | `str` | `None` | Field containing gender data |
| `gender_from_name` | `bool` | `False` | Infer gender from original name |
| `format` | `str` | varies | Name format template |
| `f_m_ratio` | `float` | `None` | Female/male ratio |
| `use_faker` | `bool` | `False` | Use Faker library as source |
| `case` | `str` | `None` | Output case transformation |

---

### 3.2 `fake_data.operations.email_op` -- FakeEmailOperation

| | |
|---|---|
| **Class** | `FakeEmailOperation` |
| **Extends** | `GeneratorOperation` |
| **Readiness** | Production |
| **Description** | Synthetic email generation with configurable format and domain patterns |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `domains` | `List[str]` | `None` | Allowed email domains |
| `format` | `str` | varies | Email format template |
| `format_ratio` | `Dict` | `None` | Format distribution ratios |
| `first_name_field` | `str` | `None` | Field with first names |
| `last_name_field` | `str` | `None` | Field with last names |
| `preserve_domain_ratio` | `bool` | `False` | Preserve original domain distribution |

---

### 3.3 `fake_data.operations.phone_op` -- FakePhoneOperation

| | |
|---|---|
| **Class** | `FakePhoneOperation` |
| **Extends** | `GeneratorOperation` |
| **Readiness** | Production |
| **Description** | Synthetic phone number generation with country/operator format preservation |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `country_codes` | `List[str]` | `None` | Allowed country codes |
| `operator_codes_dict` | `Dict` | `None` | Operator codes by country |
| `format` | `str` | varies | Phone number format |
| `default_country` | `str` | `None` | Default country code |
| `preserve_country_code` | `bool` | `False` | Keep original country code |
| `preserve_operator_code` | `bool` | `False` | Keep original operator code |

---

### 3.4 `fake_data.operations.organization_op` -- FakeOrganizationOperation

| | |
|---|---|
| **Class** | `FakeOrganizationOperation` |
| **Extends** | `GeneratorOperation` |
| **Readiness** | Production |
| **Description** | Synthetic organization name generation with regional and type preservation |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | `str` | *required* | Target field |
| `organization_type` | `str` | `None` | Organization type filter |
| `dictionaries` | `Dict` | `None` | Custom name dictionaries |
| `prefixes` | `List[str]` | `None` | Allowed name prefixes |
| `suffixes` | `List[str]` | `None` | Allowed name suffixes |
| `region` | `str` | `None` | Regional naming conventions |
| `preserve_type` | `bool` | `False` | Preserve original organization type |

---

## 4. Metrics (`pamola_core.metrics`)

Base class: `MetricsOperation` (extends `BaseOperation`)
Module version: 4.0.0 | All registered via `@register(version="4.0.0")`

### 4.1 `metrics.operations.fidelity_ops` -- FidelityOperation

| | |
|---|---|
| **Class** | `FidelityOperation` |
| **Extends** | `MetricsOperation` |
| **Readiness** | Production |
| **Description** | Calculate data fidelity metrics comparing original and anonymized datasets |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fidelity_metrics` | `List[str]` | *required* | Metrics to calculate: `"KL"` (KL Divergence), `"KS"` (Kolmogorov-Smirnov) |
| `metric_params` | `Dict` | `None` | Per-metric configuration parameters |

---

### 4.2 `metrics.operations.privacy_ops` -- PrivacyMetricOperation

| | |
|---|---|
| **Class** | `PrivacyMetricOperation` |
| **Extends** | `MetricsOperation` |
| **Readiness** | Production |
| **Description** | Calculate privacy risk metrics for anonymized data |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `privacy_metrics` | `List[str]` | *required* | Metrics: `"DCR"` (Distance to Closest Record), `"NNDR"` (Nearest Neighbor Distance Ratio), `"Uniqueness"` |
| `metric_params` | `Dict` | `None` | Per-metric configuration parameters |

---

### 4.3 `metrics.operations.utility_ops` -- UtilityMetricOperation

| | |
|---|---|
| **Class** | `UtilityMetricOperation` |
| **Extends** | `MetricsOperation` |
| **Readiness** | Production |
| **Description** | Calculate data utility metrics measuring analytical usefulness after anonymization |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `utility_metrics` | `List[str]` | *required* | Metrics: `"ClassificationUtility"`, `"RegressionUtility"` |
| `metric_params` | `Dict` | `None` | Per-metric configuration parameters |

---

## 5. Transformations (`pamola_core.transformations`)

Base class: `TransformationOperation` (extends `BaseOperation`)
Module version: 1.0.0

### 5.1 Field Operations (`transformations.field_ops`)

#### 5.1.1 `field_ops.add_modify_fields` -- AddOrModifyFieldsOperation

| | |
|---|---|
| **Class** | `AddOrModifyFieldsOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Add or modify fields based on lookups, conditions, or computed values |
| **Updated** | 2026-03-06 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_operations` | `List[Dict]` | *required* | List of field operation definitions |
| `lookup_tables` | `Dict` | `None` | Lookup table configurations |

---

#### 5.1.2 `field_ops.remove_fields` -- RemoveFieldsOperation

| | |
|---|---|
| **Class** | `RemoveFieldsOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Remove one or more specified fields from a dataset |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fields_to_remove` | `List[str]` | *required* | Fields to remove |
| `pattern` | `str` | `None` | Regex pattern for field matching |

---

### 5.2 Cleaning (`transformations.cleaning`)

#### 5.2.1 `cleaning.clean_invalid_values` -- CleanInvalidValuesOperation

| | |
|---|---|
| **Class** | `CleanInvalidValuesOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Nullify or replace values violating defined constraints (type, range, format) |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_constraints` | `Dict` | *required* | Per-field validation constraints |
| `whitelist_path` | `str/Path` | `None` | Path to allowed values list |
| `blacklist_path` | `str/Path` | `None` | Path to forbidden values list |
| `null_replacement` | `Any` | `None` | Replacement value for invalid entries |

---

### 5.3 Imputation (`transformations.imputation`)

#### 5.3.1 `imputation.impute_missing_values` -- ImputeMissingValuesOperation

| | |
|---|---|
| **Class** | `ImputeMissingValuesOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Replace missing or invalid values using statistical functions (mean, median, mode, etc.) |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_strategies` | `Dict` | *required* | Per-field imputation strategies |
| `invalid_values` | `List` | `None` | Values to treat as missing |

---

### 5.4 Grouping (`transformations.grouping`)

#### 5.4.1 `grouping.aggregate_records_op` -- AggregateRecordsOperation

| | |
|---|---|
| **Class** | `AggregateRecordsOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Group records and apply aggregation functions (sum, mean, count, etc.) per field |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_by_fields` | `List[str]` | *required* | Fields to group by |
| `aggregation_config` | `Dict` | *required* | Per-field aggregation functions |

---

### 5.5 Merging (`transformations.merging`)

#### 5.5.1 `merging.merge_datasets_op` -- MergeDatasetOperation

| | |
|---|---|
| **Class** | `MergeDatasetOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Merge datasets with various strategies (binning, rounding, range-based matching) |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merge_strategy` | `str` | *required* | Merge strategy type |
| `relationship_types` | `Dict` | `None` | Relationship type definitions |

---

### 5.6 Splitting (`transformations.splitting`)

#### 5.6.1 `splitting.split_fields_op` -- SplitFieldsOperation

| | |
|---|---|
| **Class** | `SplitFieldsOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Split dataset into multiple groups of fields (vertical partitioning) |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_field` | `str` | *required* | ID field to preserve in all partitions |
| `field_groups` | `Dict[str, List[str]]` | *required* | Named groups of fields |
| `include_id_field` | `bool` | `True` | Include ID field in each partition |

---

#### 5.6.2 `splitting.split_by_id_values_op` -- SplitByIDValuesOperation

| | |
|---|---|
| **Class** | `SplitByIDValuesOperation` |
| **Extends** | `TransformationOperation` |
| **Readiness** | Production |
| **Description** | Split datasets by ID values or automatic partitioning (horizontal partitioning) |
| **Updated** | 2026-03-05 |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_field` | `str` | *required* | ID field for partitioning |
| `value_groups` | `Dict[str, List]` | `None` | Named groups of ID values |
| `number_of_partitions` | `int` | `None` | Auto-partition count |
| `partition_method` | `str` | `"round_robin"` | Partitioning method |

---

## 6. Attack Simulation -- Legacy (`pamola_core.attacks`)

> **Note:** This package does NOT use the PAMOLA operations framework (`utils/ops`). It uses a standalone ABC-based architecture with `AttackInitialization` as base class. Marked as legacy/template code.

Base class: `AttackInitialization` (ABC) -> `PreprocessData` -> attack classes

### 6.1 `attacks.membership_inference` -- MembershipInference

| | |
|---|---|
| **Class** | `MembershipInference` |
| **Extends** | `PreprocessData` |
| **Readiness** | Legacy |
| **Description** | Membership Inference Attack (MIA) simulation -- determines if a record was in the training set |
| **Updated** | 2026-03-05 |

---

### 6.2 `attacks.attribute_inference` -- AttributeInference

| | |
|---|---|
| **Class** | `AttributeInference` |
| **Extends** | `PreprocessData` |
| **Readiness** | Legacy |
| **Description** | Attribute inference attack -- predicts sensitive attribute values from quasi-identifiers |
| **Updated** | 2026-03-05 |

---

### 6.3 `attacks.linkage_attack` -- LinkageAttack

| | |
|---|---|
| **Class** | `LinkageAttack` |
| **Extends** | `PreprocessData` |
| **Readiness** | Legacy |
| **Description** | Linkage attack simulation -- attempts to re-identify records across datasets |
| **Updated** | 2026-03-05 |

---

### 6.4 `attacks.distance_to_closest_record` -- DistanceToClosestRecord

| | |
|---|---|
| **Class** | `DistanceToClosestRecord` |
| **Extends** | `PreprocessData` |
| **Readiness** | Legacy |
| **Description** | Calculate Distance to Closest Record (DCR) metric for privacy risk assessment |
| **Updated** | 2026-03-05 |

---

### 6.5 `attacks.nearest_neighbor_distance_ratio` -- NearestNeighborDistanceRatio

| | |
|---|---|
| **Class** | `NearestNeighborDistanceRatio` |
| **Extends** | `PreprocessData` |
| **Readiness** | Legacy |
| **Description** | Calculate Nearest Neighbor Distance Ratio (NNDR) for privacy evaluation |
| **Updated** | 2026-03-05 |

---

### 6.6 `attacks.attack_metrics` -- AttackMetrics

| | |
|---|---|
| **Class** | `AttackMetrics` |
| **Extends** | `PreprocessData` |
| **Readiness** | Legacy |
| **Description** | Evaluate attack performance: accuracy, precision, recall, AUC, advantage metrics |
| **Updated** | 2026-03-05 |

---

## 7. Analysis Utilities -- Non-framework (`pamola_core.analysis`)

> **Note:** This package provides standalone analysis functions and classes, NOT framework-based operations. These are utility modules used by other components.

### 7.1 `analysis.dataset_summary` -- DatasetAnalyzer

| | |
|---|---|
| **Class** | `DatasetAnalyzer` |
| **Type** | Utility class |
| **Readiness** | Stable |
| **Description** | Dataset summarization: type detection, missing value analysis, outlier detection (IQR) |
| **Updated** | 2026-03-05 |
| **Key function** | `analyze_dataset_summary(df, ...)` |

---

### 7.2 `analysis.descriptive_stats` -- analyze_descriptive_stats

| | |
|---|---|
| **Type** | Standalone function |
| **Readiness** | Stable |
| **Description** | Descriptive statistics computation for dataset fields |
| **Updated** | 2026-03-05 |
| **Key function** | `analyze_descriptive_stats(df, ...)` |

---

### 7.3 `analysis.field_analysis` -- analyze_field_level

| | |
|---|---|
| **Type** | Standalone function |
| **Readiness** | Stable |
| **Description** | Per-field detailed analysis with type-specific metrics |
| **Updated** | 2026-03-05 |
| **Key function** | `analyze_field_level(df, ...)` |

---

### 7.4 `analysis.correlation` -- CorrelationAnalyzer

| | |
|---|---|
| **Class** | `CorrelationAnalyzer` |
| **Type** | Utility class |
| **Readiness** | Stable |
| **Description** | Correlation analysis with visualization support |
| **Updated** | 2026-03-05 |
| **Key function** | `analyze_correlation(df, ...)` |

---

### 7.5 `analysis.privacy_risk` -- calculate_full_risk

| | |
|---|---|
| **Type** | Standalone function |
| **Readiness** | Stable |
| **Description** | Full privacy risk calculation: k-anonymity, l-diversity, t-closeness assessment |
| **Updated** | 2026-03-05 |
| **Key function** | `calculate_full_risk(df, quasi_identifiers, ...)` |

---

### 7.6 `analysis.distribution`

| | |
|---|---|
| **Type** | Utility module |
| **Readiness** | Stable |
| **Description** | Distribution analysis utilities |
| **Updated** | 2025-11-19 |

---

## 8. Privacy Models -- Non-framework (`pamola_core.privacy_models`)

> **Note:** Privacy models use their own `BasePrivacyModelProcessor` ABC hierarchy, separate from the operations framework. These are calculation/evaluation engines for privacy metrics.

### 8.1 `privacy_models.k_anonymity.calculation` -- KAnonymityProcessor

| | |
|---|---|
| **Class** | `KAnonymityProcessor` |
| **Extends** | `BasePrivacyModelProcessor` (ABC) |
| **Readiness** | Stable |
| **Description** | K-anonymity model evaluation and reporting |
| **Updated** | 2026-03-05 |

---

### 8.2 `privacy_models.l_diversity.calculation` -- LDiversityCalculator

| | |
|---|---|
| **Class** | `LDiversityCalculator` |
| **Extends** | `BasePrivacyModelProcessor` |
| **Readiness** | Stable |
| **Description** | L-diversity model computation with attribute risk analysis, visualization, and reporting |
| **Updated** | 2026-03-05 |

---

### 8.3 `privacy_models.t_closeness.calculation` -- TCloseness

| | |
|---|---|
| **Class** | `TCloseness` |
| **Extends** | `BasePrivacyModelProcessor` |
| **Readiness** | Stable |
| **Description** | T-closeness privacy model evaluation |
| **Updated** | 2026-03-05 |

---

### 8.4 `privacy_models.differential_privacy.calculation` -- DifferentialPrivacyProcessor

| | |
|---|---|
| **Class** | `DifferentialPrivacyProcessor` |
| **Extends** | `BasePrivacyModelProcessor` |
| **Readiness** | Stable |
| **Description** | Differential privacy implementation using Laplace mechanism |
| **Updated** | 2026-03-05 |

---

## Appendix A: Base Class Hierarchy

```
BaseOperation (utils/ops/op_base.py)
 |
 +-- FieldOperation (utils/ops/op_base.py)
 |    +-- CategoricalOperation (profiling)
 |    +-- NumericOperation (profiling)
 |    +-- DateOperation (profiling)
 |    +-- EmailOperation (profiling)
 |    +-- PhoneOperation (profiling)
 |    +-- CurrencyOperation (profiling)
 |    +-- TextSemanticCategorizerOperation (profiling)
 |    +-- MVFOperation (profiling)
 |    +-- CorrelationOperation (profiling)
 |    +-- IdentityAnalysisOperation (profiling)
 |    +-- GroupAnalyzerOperation (profiling)
 |
 +-- DataFrameOperation (utils/ops/op_base.py)
 |
 +-- TransformationOperation (transformations/base_transformation_op.py)
 |    +-- AddOrModifyFieldsOperation
 |    +-- RemoveFieldsOperation
 |    +-- CleanInvalidValuesOperation
 |    +-- ImputeMissingValuesOperation
 |    +-- AggregateRecordsOperation
 |    +-- MergeDatasetOperation
 |    +-- SplitFieldsOperation
 |    +-- SplitByIDValuesOperation
 |
 +-- AnonymizationOperation (anonymization/base_anonymization_op.py)
 |    +-- NumericGeneralizationOperation
 |    +-- CategoricalGeneralizationOperation
 |    +-- DateTimeGeneralizationOperation
 |    +-- FullMaskingOperation
 |    +-- PartialMaskingOperation
 |    +-- UniformNumericNoiseOperation
 |    +-- UniformTemporalNoiseOperation
 |    +-- HashBasedPseudonymizationOperation
 |    +-- MappingPseudonymizationOperation
 |    +-- AttributeSuppressionOperation
 |    +-- CellSuppressionOperation
 |    +-- RecordSuppressionOperation
 |
 +-- GeneratorOperation (fake_data/base_generator_op.py)
 |    +-- FakeNameOperation
 |    +-- FakeEmailOperation
 |    +-- FakePhoneOperation
 |    +-- FakeOrganizationOperation
 |
 +-- MetricsOperation (metrics/base_metrics_op.py)
 |    +-- FidelityOperation
 |    +-- PrivacyMetricOperation
 |    +-- UtilityMetricOperation
 |
 +-- KAnonymityProfilerOperation (profiling)
 +-- CorrelationMatrixOperation (profiling)
 +-- DataAttributeProfilerOperation (profiling)


AttackInitialization (ABC) (attacks/base.py)
 +-- PreprocessData (attacks/preprocess_data.py)
      +-- MembershipInference
      +-- AttributeInference
      +-- LinkageAttack
      +-- DistanceToClosestRecord
      +-- NearestNeighborDistanceRatio
      +-- AttackMetrics


BasePrivacyModelProcessor (ABC) (privacy_models/base.py)
 +-- KAnonymityProcessor
 +-- LDiversityCalculator
 +-- TCloseness
 +-- DifferentialPrivacyProcessor
```

## Appendix B: Shared BaseOperation Parameters

All framework-based operations (extending `BaseOperation`) inherit the following parameter categories:

| Category | Parameters |
|----------|-----------|
| **Processing** | `mode` (REPLACE/ENRICH), `column_prefix`, `output_field_name`, `null_strategy`, `engine` (pandas/dask/auto), `chunk_size` |
| **Performance** | `optimize_memory`, `adaptive_chunk_size`, `use_dask`, `npartitions`, `dask_partition_size`, `use_vectorization`, `parallel_processes` |
| **Caching** | `use_cache`, `force_recalculation` |
| **Output** | `output_format` (csv/json/parquet), `save_output`, `generate_visualization` |
| **Visualization** | `visualization_theme`, `visualization_backend` (plotly/matplotlib), `visualization_strict`, `visualization_timeout` |
| **Security** | `use_encryption`, `encryption_mode` (age/simple/none), `encryption_key` |
