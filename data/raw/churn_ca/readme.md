# Synthetic Customer Churn Dataset - Canadian Banking

> **⚠️ Synthetic Data Notice**
>
> This dataset is **fully synthetic** - programmatically generated for testing purposes.
> **No real personal or financial information is included.**
>
> - Not collected from any external sources or financial institutions
> - Not derived from any real customer or banking data
> - Any resemblance to actual persons, accounts, or institutions is coincidental
>
> *This documentation is for technical reference, not legal advice.*

---

## Overview

Synthetic customer churn dataset simulating a hypothetical Canadian retail bank. Designed for testing churn prediction models, privacy-preserving ML techniques, and anonymization pipelines. Multiple dataset versions available for different testing scenarios.

**Domain:** Canadian retail banking  
**Target Variable:** `ChurnProbability` (continuous 0-1)  
**Primary Use Cases:** Churn prediction, privacy testing, data quality validation

---

## Data Generation

**Method:** Rule-based synthetic generation with controlled distributions

| Component | Approach |
|-----------|----------|
| Demographics | Faker-style generation with Canadian distributions |
| Financial attributes | Log-normal distributions matching retail banking patterns |
| Churn labels | Rule-based probability calculation from feature combinations |
| Geographic | Canadian provinces with realistic city distributions |
| Data quality issues | Intentionally introduced in DIRTY variants |

**Reproducibility:** JSON passport file contains generation parameters.

---

## Dataset Versions

| Version | Filename | Records | Size | Use Case |
|---------|----------|---------|------|----------|
| **Micro** | `S_CHURN_BANK_CANADA_50.csv` | 50 | ~25 KB | Unit testing, validation |
| **Small** | `S_CHURN_BANK_CANADA_2K.csv` | 2,000 | ~1 MB | Development, testing |
| **Small + Issues** | `S_CHURN_BANK_CANADA_2K_DIRTY.csv` | 2,000 | ~1 MB | Data quality testing |
| **Issue Catalog** | `S_CHURN_BANK_CANADA_2K_DIRTY_ISSUES.csv` | - | - | Documents injected issues |
| **Medium** | `S_CHURN_BANK_CANADA_10K.csv` | 10,000 | ~5 MB | Integration testing |
| **Passport** | `S_CHURN_BANK_CANADA_50.json` | - | - | Metadata and statistics |

### DIRTY Dataset Variants

The `_DIRTY` variants contain intentionally introduced data quality issues:

| Issue Type | Description | Purpose |
|------------|-------------|---------|
| Missing values | Additional nulls in optional fields | Completeness testing |
| Format violations | Case inconsistencies, whitespace | Validation testing |
| Outliers | Extreme values in numeric fields | Anomaly detection |
| Duplicates | Near-duplicate records | Deduplication testing |

The `_DIRTY_ISSUES.csv` file documents all injected issues with row references.

---

## Target Variable

| Property | Value |
|----------|-------|
| **Column** | `ChurnProbability` |
| **Type** | Continuous probability [0, 1] |
| **Distribution** | Right-skewed (most customers low risk) |
| **Statistics (50-sample)** | Min: 0.04, Max: 0.53, Mean: 0.19, Std: 0.13 |

**Binary threshold:** Use 0.5 for classification tasks, or optimize based on business cost.

---

## Field Specifications

### Privacy Classification Summary

| Category | Count | Fields |
|----------|-------|--------|
| **Direct Identifiers** | 7 | ID, FULLNAME, FAMILYID, MobilePhone, Email, CardNumber, AccountType |
| **Quasi-Identifiers** | 6 | CITY, PROVINCE, AGE, overdraft_usage, branch_visits, digital_usage_level |
| **Sensitive Attributes** | 4 | Income, credit_score, loan_status, credit_card_status |
| **Behavioral/Indirect** | 20 | SEX, IsMarried, RACE, IsHomeOwner, transaction features, satisfaction metrics |

### Direct Identifiers

| Field | Type | Unique Ratio | Anonymization |
|-------|------|--------------|---------------|
| `ID` | Integer | 1.00 | Drop or hash with salt |
| `FULLNAME` | Text | 1.00 | Drop |
| `FAMILYID` | Categorical | 1.00* | Drop (92% missing) |
| `MobilePhone` | Text | ~1.00 | Mask (keep area code) |
| `Email` | Text | ~1.00 | Domain only |
| `CardNumber` | Text | ~1.00 | Mask (show last 4) |
| `AccountType` | Categorical | High | Generalize |

### Quasi-Identifiers

| Field | Type | Values | Anonymization |
|-------|------|--------|---------------|
| `AGE` | Integer | 21-52 (mean 36) | 5-year bins |
| `CITY` | Text | 29 cities | Generalize to region |
| `PROVINCE` | Categorical | ON 64%, BC 16%, QC 16%, AB 4% | Group to East/Central/West |
| `overdraft_usage` | Float | Continuous | Binning |
| `branch_visits` | Integer | Count | Capping |
| `digital_usage_level` | Categorical | Low/Medium/High | Preserve |

### Sensitive Attributes

| Field | Type | Distribution | Anonymization |
|-------|------|--------------|---------------|
| `Income` | Float | Mean $75K, Std $26K | Laplace noise (epsilon=0.1) |
| `credit_score` | Float | Mean 680, Range 414-850 | Categorize to 5 bands |
| `loan_status` | Categorical | 24% missing, Mortgage 79% | Handle missing, generalize |
| `credit_card_status` | Categorical | Active/Inactive/None | Preserve |

### Demographic Fields

| Field | Type | Distribution | Notes |
|-------|------|--------------|-------|
| `SEX` | Categorical | F 60%, M 40% | Low re-identification risk |
| `IsMarried` | Boolean | True 40%, False 60% | Binary |
| `RACE` | Categorical | White 60%, Asian 26%, Other 14% | Consider suppression |
| `IsHomeOwner` | Boolean | Binary | Correlates with Income |
| `HOMEVALUE` | Float | Nullable | Only for homeowners |
| `RENTVALUE` | Float | Nullable | Only for renters |
| `IsEduBachelors` | Boolean | Binary | Education indicator |
| `IsUnemployed` | Boolean | Binary | Employment status |

### Financial Behavior Fields

| Field | Type | Description |
|-------|------|-------------|
| `AME` | Float | Average monthly expenses |
| `AMB` | Float | Average monthly balance |
| `avg12_tx` | Float | Average transactions (12 months) |
| `avg12_tx_volume` | Float | Average transaction volume |
| `BankIsPrimary` | Boolean | Primary banking relationship |

### Engagement Fields

| Field | Type | Description |
|-------|------|-------------|
| `customer_service` | Integer | Service interactions count |
| `complaints_filed` | Integer | Complaint count |
| `satisfaction_level` | Float | Satisfaction score |
| `BankPresenceRating` | Float | Engagement rating (scale 0-1)* |

*Note: BankPresenceRating uses 0-1 scale (observed range 0.1-0.18)

---

## Known Data Quality Issues

### Documented Anomalies

| Issue | Field | Description | Handling |
|-------|-------|-------------|----------|
| Missing data | `FAMILYID` | 92% null | Drop column |
| Missing data | `loan_status` | 24% null | Impute mode or preserve |
| Text artifacts | `MobilePhone` | 'T' suffix (e.g., "807-881-7806T") | Strip suffix |
| Text artifacts | `CardNumber` | 'T' in middle (e.g., "4514 0112T 1742") | Remove 'T' |
| Scale issue | `BankPresenceRating` | 0-0.18 instead of expected 0-10 | Document or rescale |

### Data Cleaning Recommendations

```python
# Handle missing values
missing_patterns = {
    'FAMILYID': 'drop_column',      # 92% missing
    'loan_status': 'impute_mode'    # 24% missing
}

# Clean text artifacts
text_cleaning = {
    'MobilePhone': lambda x: x.rstrip('T') if x else x,
    'CardNumber': lambda x: x.replace('T', '') if x else x
}
```

---

## Testing Capabilities

### ML Tasks

| Task | Target | Metrics |
|------|--------|---------|
| Binary Classification | ChurnProbability > threshold | AUROC, F1, Precision/Recall |
| Regression | ChurnProbability continuous | MAE, RMSE, R-squared |
| Calibration | Probability calibration | Brier score, calibration curve |

### Privacy Testing

| Test Type | Method | Target |
|-----------|--------|--------|
| K-anonymity | QI combinations | k >= 3 (micro), k >= 5 (larger) |
| L-diversity | Sensitive attribute distribution | l >= 2 |
| Membership inference | Attack simulation | < 10% success rate |
| Attribute inference | Predict credit_score from QIs | Measure disclosure risk |

### Data Quality Testing

| Test | Dataset | Purpose |
|------|---------|---------|
| Completeness | All | Detect missing value patterns |
| Validity | DIRTY variant | Detect format violations |
| Consistency | All | Cross-field validation |
| Uniqueness | All | Duplicate detection |

---

## Anonymization Pipeline

### Recommended Sequence

```
1. Data Cleaning
   ├── Handle missing values (FAMILYID: drop, loan_status: impute)
   ├── Clean text artifacts (phone 'T' suffix, card 'T')
   └── Validate value ranges

2. Direct Identifier Processing
   ├── Drop: FULLNAME, FAMILYID
   ├── Hash: ID (SHA-256 + project salt)
   └── Mask: MobilePhone, CardNumber, Email

3. Quasi-Identifier Generalization
   ├── AGE -> 5-year bins ([20-24], [25-29], ...)
   ├── CITY -> Region mapping (GTA, Ottawa Region, etc.)
   └── PROVINCE -> Geographic groups (East/Central/West)

4. Sensitive Attribute Protection
   ├── Income + Laplace noise (epsilon=0.1)
   ├── credit_score -> Categories (Poor/Fair/Good/VeryGood/Excellent)
   └── loan_status -> Binary (HasLoan / NoLoan)

5. Validation
   ├── Verify k-anonymity achieved
   ├── Test ML model performance preservation
   └── Validate distribution similarity (KS test)
```

### Target Privacy Metrics

| Metric | Micro (50) | Small (2K) | Medium (10K) |
|--------|------------|------------|--------------|
| k-anonymity | >= 3 | >= 5 | >= 5 |
| l-diversity | >= 2 | >= 2 | >= 3 |
| Unique QI combinations | < 30% | < 20% | < 15% |
| Re-identification risk | < 20% | < 10% | < 5% |

---

## File Structure

```
churn_ca/
├── S_CHURN_BANK_CANADA_50.csv          # Micro dataset (unit testing)
├── S_CHURN_BANK_CANADA_50.json         # Passport/metadata
├── S_CHURN_BANK_CANADA_2K.csv          # Small dataset (development)
├── S_CHURN_BANK_CANADA_2K_DIRTY.csv    # With data quality issues
├── S_CHURN_BANK_CANADA_2K_DIRTY_ISSUES.csv  # Issue documentation
├── S_CHURN_BANK_CANADA_10K.csv         # Medium dataset (integration)
└── README.md                           # This documentation
```

---

## Usage Examples

### Load and Explore

```python
import pandas as pd

# Load appropriate version
df = pd.read_csv("data/raw/churn_ca/S_CHURN_BANK_CANADA_2K.csv")

# Check target distribution
print(df['ChurnProbability'].describe())

# Identify high-risk customers
high_risk = df[df['ChurnProbability'] > 0.3]
```

### Test Anonymization

```python
from pamola_core.profiling import ProfileOperation
from pamola_core.anonymization import KAnonymityOperation

# Profile quasi-identifiers
quasi_ids = ['AGE', 'CITY', 'PROVINCE', 'SEX']
profile = ProfileOperation(params={"fields": quasi_ids}).run(df)

# Apply k-anonymity
anon_result = KAnonymityOperation(
    params={"k": 5, "quasi_identifiers": quasi_ids}
).run(df)
```

---

## Known Limitations

1. **Sample size effects** - 50-record micro dataset may not represent full distributions
2. **Geographic concentration** - 64% Ontario skews geographic analysis
3. **No temporal features** - No time-series data for trend analysis
4. **Perfect regularization** - Clean data may not reflect real-world noise (use DIRTY variant)
5. **Single churn definition** - Rule-based labels may not match real churn patterns

---

## License

Apache 2.0 - same as PAMOLA.CORE repository.

See [LICENSE](../../../LICENSE) for full terms.

---

**Maintainer:** [Realm Inveo Inc.](https://realmdata.io)  
**Repository:** [github.com/DGT-Network/PAMOLA](https://github.com/DGT-Network/PAMOLA)  
**Version:** 1.0.0 (Epic 2 Testing Suite)
