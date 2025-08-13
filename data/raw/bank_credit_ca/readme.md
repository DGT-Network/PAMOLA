# Bank Credit Datasets for Privacy-Preserving Analytics Testing

## Overview

Two comprehensive synthetic financial datasets designed for testing privacy-preserving analytics, anonymization techniques, and federated learning scenarios. These datasets simulate real-world financial data patterns while containing no actual personal information.

## Datasets

### 1. FIN_BANK_APPS - Bank Loan Applications Dataset
**File:** `FIN_BANK_APPS_10k.csv`  
**Records:** 10,000 synthetic bank loan applications  
**Purpose:** Testing anonymization of financial application data with behavioral patterns  
**Passport:** `FIN_BANK_APPS_passport.json`

### 2. FIN_BUREAU - Credit Bureau Dataset  
**File:** `FIN_BUREAU_10k.csv`  
**Records:** 10,000 synthetic credit bureau records  
**Purpose:** Testing privacy techniques on credit scoring and risk assessment data  
**Passport:** `FIN_BUREAU_passport.json`

## Field Descriptions

### FIN_BANK_APPS Fields (38 fields)

#### Direct Identifiers (2 fields)
- `application_id`: Unique application identifier (string)
- `bank_internal_id`: Bank's internal customer hash (string, 16 chars)

#### Quasi-Identifiers (11 fields)
- `birth_year`: Year of birth (1945-2004)
- `sex`: Gender (F/M/X)
- `home_postal_fsa`: Canadian postal code FSA (first 3 chars, e.g., "M5V")
- `home_province`: Province code (e.g., "ON", "QC")
- `employment_type`: Full-time/Part-time/Self-employed/Contract/Retired/Student/Unemployed
- `employer_industry`: Industry sector (Technology/Healthcare/Retail/Finance/etc.)
- `pay_frequency`: Payment frequency (monthly/biweekly/weekly)
- `device_type`: Application device (mobile/desktop)
- `browser`: Browser used (Chrome/Firefox/Safari/Edge/Other)
- `os`: Operating system (Windows/macOS/iOS/Android/Linux)
- `product`: Loan product type (CreditCard/PersonalLoan/AutoLoan/LineOfCredit)

#### Sensitive Attributes (15 fields)
- `gt_person_id`: Ground-truth person ID for evaluation only
- `link_shared_id`: Cross-dataset linkage hash (~62% populated)
- `declared_income_cad`: Self-reported annual income ($12K-$400K)
- `verified_income_cad`: Bank-verified income estimate
- `avg_balance_3m_cad`: Average account balance over 3 months
- `inflow_payroll_3m_cad`: Total payroll deposits over 3 months
- `card_spend_3m_cad`: Credit/debit card spending over 3 months
- `atm_withdraw_3m_cad`: ATM withdrawals over 3 months
- `nsf_count_3m`: Non-sufficient funds incidents (0-4)
- `late_fee_count_3m`: Late payment fees (0-3)
- `overdraft_days_3m`: Days in overdraft (0-7)
- `ip_lat`/`ip_lng`: IP geolocation coordinates
- `internal_risk_score`: Bank's risk assessment (0.0-1.0)
- `default_12m`: Binary flag for default within 12 months

#### Indirect/Utility Fields (10 fields)
- `application_ts`: Application timestamp
- `amount_requested`: Loan amount requested ($500-$100K)
- `term_months`: Loan term (12-72 months)
- `interest_rate`: Annual percentage rate (0.01-0.309)
- `employment_tenure_months`: Time at current employer (0-175 months)
- `dependents`: Number of dependents (0-5)
- `residence_months`: Time at current address (0-235 months)
- `home_lat`/`home_lng`: Home address coordinates (~20% missing)
- `device_id`: Unique device identifier
- `created_at`/`updated_at`: Record timestamps
- `source_system`: Data source tag ("bank_app_v1")

### FIN_BUREAU Fields (30 fields)

#### Direct Identifiers (2 fields)
- `bureau_row_id`: Unique row identifier (string)
- `bureau_internal_id`: Bureau's internal person hash (string, 16 chars)

#### Quasi-Identifiers (4 fields)
- `birth_year`: Year of birth (1945-2004)
- `sex`: Gender (F/M/X)
- `home_postal_fsa`: Canadian postal code FSA (e.g., "P5N")
- `home_province`: Province code (e.g., "ON", "BC")

#### Sensitive Attributes (12 fields)
- `gt_person_id`: Ground-truth person ID for evaluation only
- `link_shared_id`: Cross-dataset linkage hash (~66% populated)
- `bureau_score`: Credit score (346-900, median 678)
- `credit_limit_total`: Sum of all credit limits ($775-$300K)
- `balance_total`: Sum of all balances ($144-$250K)
- `delinq_30d_12m`: 30-day delinquencies in past 12 months (0-5)
- `delinq_60d_12m`: 60-day delinquencies in past 12 months (0-3)
- `delinq_90d_12m`: 90-day delinquencies in past 12 months (0-3)
- `derogatory_count`: Number of derogatory items (0-2)
- `bankruptcy_flag`: Bankruptcy indicator (boolean)
- `months_since_bankruptcy`: Time since bankruptcy (0-228 months, 97% null)
- `consumer_proposal_flag`: Consumer proposal indicator (boolean)

#### Indirect/Utility Fields (12 fields)
- `bureau_pull_ts`: Credit check timestamp
- `inquiries_12m`: Credit inquiries in past 12 months (0-9)
- `inquiries_90d`: Credit inquiries in past 90 days (0-5)
- `tradelines_open`: Number of open credit accounts (0-16)
- `tradelines_revolving`: Revolving credit accounts (0-13)
- `tradelines_installment`: Installment loans (0-9)
- `tradelines_mortgage`: Mortgage accounts (0-5)
- `credit_utilization_pct`: Current credit utilization (0-162%)
- `max_utilization_12m_pct`: Maximum utilization in 12 months (0-209%)
- `addresses_count_24m`: Address changes in 24 months (0-7)
- `employers_count_24m`: Employer changes in 24 months (0-6)
- `thin_file_flag`: Limited credit history indicator (boolean)
- `created_at`/`updated_at`: Record timestamps
- `source_system`: Data source tag ("bureau_v1")

## Key Data Characteristics

### Cross-Dataset Linkage
- **Shared Records:** ~3,800-4,000 individuals appear in both datasets
- **Linkage Field:** `link_shared_id` enables privacy-preserving joins
- **Coverage:** 62% in BANK_APPS, 66% in BUREAU have linkage IDs
- **Ground Truth:** `gt_person_id` for validation (remove before production)

### Distribution Highlights

#### FIN_BANK_APPS
- **Products:** Balanced distribution across 4 loan types (~25% each)
- **Default Rate:** 6.44% within 12 months
- **Income:** Median $65,599 CAD (range $12K-$400K)
- **Risk Score:** Mean 0.35 on 0-1 scale
- **Device Usage:** 70% mobile, 30% desktop applications
- **Geographic Coverage:** All Canadian provinces represented

#### FIN_BUREAU
- **Credit Scores:** Median 678 (range 346-900)
- **Thin Files:** 12.09% with limited credit history
- **Bankruptcies:** 2.96% have bankruptcy records
- **Credit Utilization:** Mean 47%, max 162%
- **Delinquencies:** 60% have at least one 30-day late payment
- **Tradelines:** Average 5 open accounts per person

## Testing Capabilities

### Privacy & Anonymization
- **K-anonymity:** Test with quasi-identifier combinations (target k≥5)
- **L-diversity:** Validate sensitive attribute diversity within groups
- **Differential Privacy:** Apply Laplace noise (ε≈0.2) to numeric fields
- **Generalization:** Test hierarchies (exact value → range → category)
- **Suppression:** Cell, record, and attribute-level suppression strategies

### Federated Learning Scenarios
- **Vertical FL:** Different features for same entities via `link_shared_id`
- **Horizontal FL:** Same features across different geographic regions
- **Cross-Silo:** Bank data vs. bureau data collaboration
- **Privacy-Preserving Joins:** Test secure multi-party computation

### Attack Simulations
- **Linkage Attacks:** Using quasi-identifier combinations
- **Membership Inference:** Validate with `gt_person_id`
- **Attribute Inference:** From correlated features
- **Re-identification Risk:** Measure uniqueness of combinations

## Data Quality Features

### Realistic Patterns
- **Income Verification:** Verified income ≈ 0.95 × declared income (±noise)
- **Geographic Consistency:** 92% have matching IP and home locations
- **Temporal Logic:** Application dates precede default dates
- **Credit Relationships:** Higher utilization correlates with lower scores

### Edge Cases & Anomalies
- **Outliers:** 2% extreme incomes (>$300K), scores (<400 or >850)
- **Mixed Currency:** 4.26% USD transactions in CAD dataset
- **Geographic Outliers:** 8% IP location ≠ home province
- **Data Quality Issues:** Realistic missing patterns (e.g., bankruptcy dates)

## Anonymization Pipeline Recommendations

### Priority 1: Direct Identifiers
- Hash all IDs with project-specific salt (SHA-256)
- Remove or pseudonymize internal identifiers
- Drop evaluation fields (`gt_person_id`) before release

### Priority 2: Quasi-Identifiers  
- Generalize birth_year to 5-year buckets
- Coarsen postal codes to first character or province
- Group rare categories (<5 occurrences)
- Apply k-anonymity validation

### Priority 3: Sensitive Fields
- Add Laplace noise to financial amounts (ε≈0.2)
- Bucket continuous values (income ranges, score bands)
- Round geographic coordinates to 2 decimal places
- Coarsen timestamps to day or week level

## File Structure
```
financial_test_data/
├── FIN_BANK_APPS_10k.csv           # Bank applications dataset
├── FIN_BANK_APPS_passport.json     # Detailed specifications
├── FIN_BUREAU_10k.csv              # Credit bureau dataset  
├── FIN_BUREAU_passport.json        # Detailed specifications
└── README.md                        # This documentation
```

## Usage Notes

- **Evaluation Only:** Fields marked with `gt_` prefix are for testing only
- **Synthetic Data:** All records are artificially generated - no real PII
- **Passport Files:** Refer to JSON passports for complete field specifications
- **Linkage Testing:** Use `link_shared_id` for privacy-preserving joins
- **Compliance Testing:** Suitable for GDPR, CCPA, and fair lending scenarios

---
*Generated for PAMOLA Epic 2 Testing Suite - Version 1.0.0*
