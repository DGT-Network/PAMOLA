# Metrics Module Documentation Index

Quick navigation guide for all metrics module documentation.

## Core Framework

- **[Base Metrics Operation](./base_metrics_op.md)** - Foundation class for all metrics operations with lifecycle management, caching, and distributed processing support.

## Commons (Utilities & Shared Components)

Reusable utilities for metric calculation, data processing, and quality assessment.

### Data Processing & Validation
- **[Preprocessing](./commons/preprocessing.md)** - Convert mixed data types to numeric format for distance calculations
- **[Validation](./commons/validation.md)** - Dataset compatibility and metric input validation
- **[Validation Rules](./commons/validation_rules.md)** - Extensible framework for data quality rules (email, phone, date, SSN validation)

### Aggregation & Normalization
- **[Aggregation](./commons/aggregation.md)** - Aggregate column/metric scores into dataset-level metrics
- **[Normalize](./commons/normalize.md)** - Normalize values, arrays, and metrics to standard scales
- **[Safe Instantiate](./commons/safe_instantiate.md)** - Safe class instantiation from dynamic config

### Scoring & Assessment
- **[Quality Scoring](./commons/quality_scoring.md)** - Comprehensive data quality calculation (completeness, validity, diversity)
- **[Predicted Utility Scoring](./commons/predicted_utility_scoring.md)** - Quick utility assessment on sampled subsets
- **[Risk Scoring](./commons/risk_scoring.md)** - Re-identification risk estimation with attribute role detection
- **[Schema Manager](./commons/schema_manager.md)** - Field definitions, validation rule assignment, schema persistence

## Fidelity Metrics

Measure preservation of statistical properties and distribution characteristics.

### Distribution-Based
- **[KL Divergence](./fidelity/distribution/kl_divergence.md)** - Kullback-Leibler divergence with confidence levels
- **[KS Test](./fidelity/distribution/ks_test.md)** - Kolmogorov-Smirnov distribution test

### Statistical Properties
- **[Statistical Fidelity](./fidelity/statistical_fidelity.md)** - Mean, variance, correlation preservation metrics

### Operations
- **[Fidelity Operations](./operations/fidelity_ops.md)** - FidelityOperation wrapper for fidelity metric workflows

## Privacy Metrics

Evaluate disclosure risk and privacy preservation.

### Distance-Based
- **[Distance to Closest Record](./privacy/distance.md)** - DCR privacy metric with FAISS support for large-scale nearest neighbor search

### Risk & Disclosure
- **[Disclosure Risk](./privacy/disclosure_risk.md)** - Prosecutor, journalist, marketer risk models
- **[Identity Risk](./privacy/identity.md)** - Record uniqueness and identity disclosure risk
- **[Neighbor Risk](./privacy/neighbor.md)** - Neighborhood-based privacy risk assessment

### Operations
- **[Privacy Operations](./operations/privacy_ops.md)** - PrivacyMetricOperation wrapper for privacy metric workflows

## Utility Metrics

Assess data utility and analytical value after anonymization.

### Classification
- **[Classification Utility](./utility/classification.md)** - Classification model performance assessment
- **[F1 Score](./utility/f1_score.md)** - F1 Score for binary and multi-class classification

### Regression
- **[Regression Utility](./utility/regression.md)** - Regression model performance assessment
- **[Mean Squared Error](./utility/mean_squared_error.md)** - MSE/RMSE for regression evaluation
- **[R² Score](./utility/r2_score.md)** - Coefficient of determination

### Information Loss
- **[Information Loss](./utility/information_loss.md)** - Generalization, suppression, and overall information loss
- **[L-Diversity Loss](./utility/ldiversity_loss.md)** - L-diversity preservation as utility metric

### Operations
- **[Utility Operations](./operations/utility_ops.md)** - UtilityMetricOperation wrapper for utility metric workflows

## Quality Metrics

Assess data quality, distribution similarity, and synthetic data fidelity.

### Distribution Comparison
- **[Kolmogorov-Smirnov Test](./quality/kolmogorov_smirnov_test.md)** - Non-parametric distribution test
- **[Kullback-Leibler Divergence](./quality/kullback_leibler_divergence.md)** - Information-theoretic distribution divergence
- **[Wasserstein Distance](./quality/wasserstein_distance.md)** - Optimal transport distance for distributions

### Statistical Relationships
- **[Pearson Correlation](./quality/pearson_correlation.md)** - Linear relationship preservation in anonymized data

## Operations Wrappers

Higher-level operation interfaces for metric workflows.

- **[Fidelity Operations](./operations/fidelity_ops.md)** - FidelityOperation for end-to-end fidelity assessment
- **[Privacy Operations](./operations/privacy_ops.md)** - PrivacyMetricOperation for privacy evaluation
- **[Utility Operations](./operations/utility_ops.md)** - UtilityMetricOperation for utility assessment

---

## Quick Reference by Use Case

### "How much privacy is preserved?"
→ Start with **[Privacy Operations](./operations/privacy_ops.md)**
→ Then explore **[Distance to Closest Record](./privacy/distance.md)**, **[Disclosure Risk](./privacy/disclosure_risk.md)**

### "Is my anonymized data still useful?"
→ Start with **[Utility Operations](./operations/utility_ops.md)**
→ Then explore **[Classification Utility](./utility/classification.md)**, **[Regression Utility](./utility/regression.md)**, **[Information Loss](./utility/information_loss.md)**

### "Does my synthetic data match the original?"
→ Start with **[Fidelity Operations](./operations/fidelity_ops.md)**
→ Then explore **[Statistical Fidelity](./fidelity/statistical_fidelity.md)**, **[KL Divergence](./fidelity/distribution/kl_divergence.md)**, **[Wasserstein Distance](./quality/wasserstein_distance.md)**

### "What's the data quality before metrics?"
→ Start with **[Quality Scoring](./commons/quality_scoring.md)**
→ Then explore **[Validation Rules](./commons/validation_rules.md)**

### "How risky is this data?"
→ Start with **[Risk Scoring](./commons/risk_scoring.md)**
→ Then explore **[Disclosure Risk](./privacy/disclosure_risk.md)**, **[Distance to Closest Record](./privacy/distance.md)**

---

## Documentation Statistics

| Category | Count | Status |
|----------|-------|--------|
| Total documented metrics | 31+ | Complete |
| Commons utilities | 9 | Complete |
| Fidelity metrics | 3 | Complete |
| Privacy metrics | 3 | Complete |
| Utility metrics | 5 | Complete |
| Quality metrics | 4 | Complete |
| Operations wrappers | 3 | Complete |
| Framework/Base | 1 | Complete |

---

## Related Documentation

- **[Project Overview & PDR](../../../project-overview-pdr.md)** - Project requirements and architecture
- **[Code Standards](../../../code-standards.md)** - Development guidelines
- **[System Architecture](../../../system-architecture.md)** - System design documentation
- **[Codebase Summary](../../../codebase-summary.md)** - Detailed codebase structure

---

*Last Updated: March 2026*
*Documentation Status: Complete & Current*
