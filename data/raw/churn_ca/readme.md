# S_CHURN_BANK_CANADA DATASET SPECIFICATION

## 1. Dataset Overview

### 1.1 Metadata

* **Dataset Name**: S_CHURN_BANK_CANADA
* **Data Type**: Synthetic (Surrogate) Data
* **Source**: Generated based on marginal distributions for a hypothetical Canadian bank (RBC-like) with rule-based churn labels
* **Format**: CSV (Comma-Separated Values)
* **Encoding**: UTF-8
* **Delimiter**: Comma (,)
* **Text Qualifier**: Double quotes (")
* **Header**: First row contains field names
* **Missing Values**: Present in some fields (notably FAMILYID, loan_status)
* **Data Quality**: Cleaned, strong regularization, distributions ~normal-like

### 1.2 Available Versions

| Version | Filename | Records | Size | Use Case |
| --- | --- | --- | --- | --- |
| **Micro** | **S_CHURN_BANK_CANADA_50.csv** | **50** | **~25 KB** | **Unit testing, validation** |
| Small | S_CHURN_BANK_CANADA_2K.csv | 2,000 | ~1 MB | Development, testing |
| Medium | S_CHURN_BANK_CANADA_10K.csv | 10,000 | ~5 MB | Integration testing |
| Large | S_CHURN_BANK_CANADA_365K.csv | 365,000 | ~180 MB | Production simulation |

### 1.3 Target Variable Statistics (50-record sample)

* **Target Column**: `ChurnProbability`
* **Type**: Continuous probability [0,1]
* **Distribution**:
  * Min: 0.04
  * Max: 0.53
  * Mean: 0.1902
  * Std: 0.1302
  * Median: ~0.15 (estimated)

## 2. Business Context and Use Cases

### 2.1 Primary ML Tasks

1. **Binary Classification**: Predict customer churn (threshold ChurnProbability)
2. **Regression**: Direct churn probability prediction
3. **Calibration Testing**: Evaluate probability calibration
4. **Privacy-Preserving ML**: DP-SGD, federated learning experiments

### 2.2 Privacy Risk Evaluation

* **Membership Inference Attacks (MIA)**: Test susceptibility
* **Attribute Inference**: Evaluate sensitive attribute disclosure
* **Re-identification Risk**: Assess quasi-identifier combinations

## 3. Enhanced Field Specifications

### 3.1 Updated Privacy Classification (Based on Actual Data Analysis)

| Category | Count | Fields |
| --- | --- | --- |
| **Direct Identifiers** | 7   | ID, FULLNAME, FAMILYID, MobilePhone, Email, AccountType*, account_age* |
| **Quasi-Identifiers** | 6   | CITY, PROVINCE, AGE, overdraft_usage, branch_visits, digital_usage_level |
| **Sensitive Attributes** | 4   | Income, credit_score, loan_status, credit_card_status |
| **Indirect Attributes** | 20  | SEX, IsMarried, RACE, IsHomeOwner, HOMEVALUE, RENTVALUE, IsEduBachelors, IsUnemployed, AME, AMB, avg12_tx, BankIsPrimary, avg12_tx_volume, customer_service, complaints_filed, satisfaction_level, CardNumber, CardType, BankPresenceRating, ChurnProbability |

*Note: AccountType and account_age reclassified as direct identifiers due to high uniqueness in small samples

### 3.2 Detailed Field Analysis with Observed Statistics (50-record sample)

| Field | Type | Non-null | Unique | Unique Ratio | Observed Distribution | Anonymization Priority |
| --- | --- | --- | --- | --- | --- | --- |
| **ID** | Integer | 50  | 50  | 1.00 | Range: 3455-360515 | Remove/Pseudonymize |
| **SEX** | Categorical | 50  | 2   | 0.04 | F:60%, M:40% | Low risk |
| **CITY** | Text | 50  | 29  | 0.58 | Toronto:22%, Ottawa:8% | Generalize to region |
| **PROVINCE** | Categorical | 50  | 4   | 0.08 | ON:64%, BC:16%, QC:16% | Group to regions |
| **FULLNAME** | Text | 50  | 50  | 1.00 | All unique | Remove |
| **AGE** | Integer | 50  | 25  | 0.50 | μ=36.3, σ=8.05, Range:21-52 | 5-year bins |
| **IsMarried** | Boolean | 50  | 2   | 0.04 | True:40%, False:60% | Preserve |
| **FAMILYID** | Categorical | 4   | 4   | 1.00 | 92% missing | Handle missing/remove |
| **RACE** | Categorical | 50  | 5   | 0.10 | White:60%, Asian:26% | Consider suppression |
| **Income** | Float | 50  | 50  | 1.00 | μ=74966, σ=25935 | Add noise ±5% |
| **credit_score** | Float | 50  | 46  | 0.92 | μ=679.6, σ=90.5, Range:414-850 | Categorize |
| **loan_status** | Categorical | 38  | 4   | 0.11 | 24% missing, Mortgage:79% | Handle missing |
| **BankPresenceRating** | Float | 50  | 9   | 0.18 | μ=0.143, σ=0.018 | Scale issue detected* |

*Note: BankPresenceRating shows values 0.1-0.18 instead of expected 0-10 scale

### 3.3 Critical Data Quality Observations

1. **Missing Data Patterns**:
  
  * FAMILYID: 92% missing (only 4 of 50 records)
  * loan_status: 24% missing (12 of 50 records)
2. **Unique Value Ratios**:
  
  * High (>0.8): ID, FULLNAME, Income, AME, AMB, credit_score, MobilePhone, CardNumber, Email
  * Medium (0.4-0.8): AGE, CITY, ChurnProbability
  * Low (<0.4): Most categorical fields
3. **Anomalies Detected**:
  
  * BankPresenceRating: Scale appears to be 0-1 instead of 0-10
  * Phone numbers have 'T' suffix (e.g., "807-881-7806T")
  * Card numbers have 'T' in middle (e.g., "4514 0112T 1742 1725")

## 4. Updated Anonymization Strategy

### 4.1 Field-Specific Recommendations

| Field | Strategy | Parameters | Rationale |
| --- | --- | --- | --- |
| **High Priority (Direct Identifiers)** |     |     |     |
| ID  | Drop or hash | SHA-256+salt | No ML value |
| FULLNAME | Drop | -   | No ML value |
| FAMILYID | Drop | -   | 92% missing |
| MobilePhone | Mask | Keep area code | Preserve geographic info |
| Email | Domain only | Keep provider type | Preserve service preference |
| **Medium Priority (Quasi-Identifiers)** |     |     |     |
| AGE | Generalize | 5-year bins | Balance utility/privacy |
| CITY | Hierarchical | City→Region | Preserve geographic patterns |
| PROVINCE | Group | East/Central/West | Maintain regional effects |
| **Low Priority (Sensitive)** |     |     |     |
| Income | Add noise | Laplace ε=0.1 | Preserve distribution |
| credit_score | Categorize | 5 bands | Standard industry practice |

### 4.2 k-Anonymity Evaluation

Based on 50-record analysis:

* **Current k-anonymity**: k=1 (multiple unique combinations)
* **Target k-anonymity**: k≥3 for micro dataset, k≥5 for larger sets
* **Critical QI combinations**:
  * {AGE, SEX, PROVINCE}: Requires generalization
  * {CITY, IsMarried, AccountType}: High uniqueness

## 5. Statistical Validation Metrics

### 5.1 Distribution Preservation Tests

| Metric | Test | Threshold | Critical Fields |
| --- | --- | --- | --- |
| **Numerical** | KS Statistic | <0.15 | Income, credit_score, AGE |
| **Categorical** | Chi-Square | p>0.05 | PROVINCE, AccountType |
| **Correlation** | Pearson R | >0.85 | Feature pairs |
| **Target** | AUROC | >0.90 of original | ChurnProbability prediction |

### 5.2 Privacy Metrics

| Metric | Current (50 records) | Target |
| --- | --- | --- |
| k-anonymity | 1   | ≥3  |
| l-diversity | N/A | ≥2  |
| Unique combinations | >60% | <20% |
| Re-identification risk | High | <10% |

## 6. Implementation Guidelines for PAMOLA

### 6.1 Data Loading Considerations

    # Handle missing values appropriately
    missing_patterns = {
        'FAMILYID': 'drop_column',  # 92% missing
        'loan_status': 'impute_mode'  # 24% missing
    }
    
    # Fix scale issues
    scale_corrections = {
        'BankPresenceRating': lambda x: x * 100  # Convert 0-0.18 to 0-18
    }
    
    # Clean text anomalies
    text_cleaning = {
        'MobilePhone': lambda x: x.rstrip('T'),
        'CardNumber': lambda x: x.replace('T', '')
    }

### 6.2 Anonymization Pipeline Sequence

    1. Data Cleaning
       ├── Handle missing values
       ├── Fix scale issues
       └── Clean text artifacts
    
    2. Direct Identifier Processing
       ├── Drop: FULLNAME, FAMILYID
       ├── Hash: ID
       └── Mask: MobilePhone, CardNumber, Email
    
    3. Quasi-Identifier Generalization
       ├── AGE → 5-year bins
       ├── CITY → Region mapping
       └── PROVINCE → Geographic groups
    
    4. Sensitive Attribute Protection
       ├── Income + Laplace noise
       ├── credit_score → Categories
       └── loan_status → Binary
    
    5. Validation
       ├── Check k-anonymity
       ├── Measure utility loss
       └── Verify distributions

### 6.3 Quality Assurance Checks

1. **Pre-processing**:
  
  * Verify no duplicate IDs
  * Check value ranges
  * Validate categorical levels
2. **Post-anonymization**:
  
  * Confirm k≥3 achieved
  * Test ML model performance
  * Validate statistical properties
3. **Edge Cases**:
  
  * Handle missing FAMILYID gracefully
  * Preserve loan_status distribution despite missingness
  * Maintain churn probability calibration

## 7. Testing Recommendations

### 7.1 Unit Testing (50-record dataset)

* Individual field transformations
* Missing value handling
* Scale corrections
* Text cleaning

### 7.2 Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage |
| --- | --- | --- |
| 50 records | <1 second | <10 MB |
| 2K records | <10 seconds | <50 MB |
| 365K records | <5 minutes | <2 GB |

### 7.3 Privacy Attack Simulations

* **Linkage Attack**: Use {AGE, CITY, PROVINCE} as QIs
* **Membership Inference**: Target high-income individuals
* **Attribute Inference**: Predict credit_score from other fields

## 8. Known Issues and Limitations

### 8.1 Data Quality Issues

1. **Scale inconsistency**: BankPresenceRating (0-0.18 vs expected 0-10)
2. **Text artifacts**: 'T' suffixes in phone/card numbers
3. **Missing data**: Significant missingness in FAMILYID, loan_status

### 8.2 Statistical Limitations

1. **Small sample effects**: 50 records may not represent full distribution
2. **Perfect regularization**: Overly clean data may not reflect real-world noise
3. **Temporal absence**: No time-based features for trend analysis

### 8.3 Privacy Considerations

1. **High uniqueness**: Many fields have unique ratio >0.8
2. **Geographic concentration**: 64% from Ontario
3. **Demographic skew**: May not represent full population diversity

This enhanced specification incorporates empirical observations from the actual 50-record dataset and provides practical guidance for PAMOLA implementation and testing.
