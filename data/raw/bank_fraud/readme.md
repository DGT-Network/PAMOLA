# Canadian Fraud Detection Dataset for Privacy-Preserving Analytics

## Overview

Comprehensive synthetic payment fraud dataset designed for testing privacy-preserving fraud detection, anonymization techniques, and real-time risk assessment scenarios. This dataset simulates Canadian card and payment transactions with realistic fraud patterns while containing no actual personal information.

## Dataset Files

### Core Dataset
**File:** `FRAUD_CA_V1_10k.csv`  
**Records:** 10,000 synthetic payment transactions  
**Period:** April 1, 2025 - August 11, 2025 (temporal sequence)  
**Fraud Rate:** ~5% labeled fraud cases  
**Purpose:** Testing anonymization of payment fraud detection systems

### Supporting Files
**Passport:** `FRAUD_CA_V1_passport.json` - Detailed field specifications and anonymization guidelines  
**Dictionary:** `FRAUD_CA_V1_dictionary.csv` - Reference codes and category mappings

## Field Descriptions (51 fields)

### Transaction Identifiers (4 fields)
- `transaction_id`: Unique transaction identifier (UUID format)
- `account_id`: Customer account ID (numeric, ~2,200 unique accounts)
- `merchant_id`: Merchant identifier (numeric, ~800 unique merchants)
- `device_id`: Device fingerprint (stable per account, "_new" suffix indicates unfamiliar device)

### Transaction Details (8 fields)
- `transaction_ts`: Transaction timestamp with millisecond precision
- `amount`: Transaction amount in native currency ($0.01 - $9,999)
- `currency`: Transaction currency (CAD 96%, USD 4%)
- `fx_rate_to_cad`: Foreign exchange rate to CAD (1.0 for CAD transactions)
- `channel`: Transaction channel (POS 54.6%, Ecommerce 35.1%, Interac 10.2%)
- `payment_method`: Payment type (chip/tap/swipe/keyed/web/app/interac)
- `card_present`: Physical card presence indicator (boolean)
- `primary_card_brand`: Card network (Visa/Mastercard/Amex/Discover/Interac)

### Merchant Information (9 fields)
- `merchant_category_code`: 4-digit MCC code
- `merchant_category`: MCC description (Grocery/Restaurant/Gas/Entertainment/etc.)
- `merchant_city`: Merchant location city
- `merchant_province`: Canadian province code
- `merchant_country`: Country code (CA/US)
- `merchant_lat`/`merchant_lng`: Merchant coordinates (jittered, ~23% missing)
- `merchant_risk_score`: Intrinsic merchant risk rating (0.0-1.0)
- `is_high_risk_mcc`: High-risk merchant category flag

### Customer Profile (7 fields)
- `home_city`: Account holder's city
- `home_province`: Account holder's province
- `home_country`: Account holder's country (Canada)
- `home_postal_fsa`: Canadian postal code FSA (first 3 chars, e.g., "M5V")
- `home_lat`/`home_lng`: Home coordinates (jittered, ~20% missing)
- `segment`: Customer segment (standard/premium/corporate)
- `kyc_level`: Know Your Customer level (basic/standard/enhanced)

### Device & Channel Attributes (6 fields)
- `device_type`: Device category (mobile/desktop/pos_terminal)
- `browser`: Browser type for online transactions (~55% null for POS)
- `os`: Operating system for online transactions (~55% null for POS)
- `device_trust_score`: Device reputation score (0.0-1.0)
- `new_device`: First-time device flag for account
- `three_ds_result`: 3D Secure authentication result (pass/fail/not_attempted/na)

### Geospatial Features (4 fields)
- `ip_lat`/`ip_lng`: IP geolocation for online transactions (~19% missing)
- `dist_km_home_to_merchant`: Haversine distance from home to merchant (~37% missing)
- `dist_km_ip_to_merchant`: Haversine distance from IP to merchant (~35% missing)

### Temporal Features (8 fields)
- `hour`: Hour of day (0-23, local time)
- `weekday`: Day of week (0=Monday, 6=Sunday)
- `is_night`: Night-time transaction flag (10pm-6am)
- `is_weekend`: Weekend transaction flag
- `prev_ts`: Previous transaction timestamp for account (~22% null for first transactions)
- `secs_since_prev_txn_acct`: Seconds since last account transaction (-1 if first)
- `txns_last_1h_acct`: Count of account transactions in past hour
- `amt_last_1h_acct`: Sum of account amounts in past hour
- `txns_last_1h_device`: Count of device transactions in past hour

### Risk Indicators (5 fields)
- `acct_median_amount`: Historical median transaction amount for account
- `amount_to_median_ratio`: Current amount / median amount ratio
- `is_cross_border`: International transaction flag (4.09%)
- `is_fraud`: Fraud label (5% positive rate)
- `chargeback_date`: Date of chargeback if fraud (~83% null for non-fraud)

## Key Data Characteristics

### Transaction Patterns
- **Volume Distribution:** Median $36, Mean $82, Max $9,999
- **Time Patterns:** 18% night-time, 29% weekend transactions
- **Geographic Coverage:** All Canadian provinces + 4% US merchants
- **Payment Methods:** 
  - In-person: chip (35%), tap (15%), swipe (5%)
  - Online: web (25%), app (10%), keyed (5%)
  - Interac: 10%

### Fraud Characteristics
- **Base Rate:** 5% fraud (500 fraudulent transactions)
- **High-Risk Categories:** 40% of transactions in high-risk MCCs
- **New Device Fraud:** 15% transactions from unfamiliar devices
- **Chargeback Lag:** Median 62 days, 90th percentile 109 days
- **Cross-Border Risk:** 4% international transactions

### Customer Behavior
- **Active Accounts:** ~2,200 unique customers
- **Transaction Frequency:** Mean 4.5 transactions per account
- **Geographic Mobility:** 37% transactions >100km from home
- **Device Patterns:** Average 2.3 devices per account

## Testing Capabilities

### Fraud Detection & Prevention
- **Real-time Risk Scoring:** Test with 51 engineered features
- **Velocity Checks:** Rolling windows (1-hour aggregations)
- **Geographic Analysis:** Distance-based anomaly detection
- **Device Fingerprinting:** New device detection and trust scoring
- **Merchant Risk Assessment:** Risk scores and MCC categorization

### Privacy & Anonymization
- **K-anonymity:** Test on location and demographic quasi-identifiers
- **Differential Privacy:** Apply to amounts and distances (ε≈0.3)
- **Temporal Anonymization:** Timestamp coarsening and jittering
- **Geographic Generalization:** Coordinate rounding and region mapping
- **Pseudonymization:** Consistent hashing for IDs across timeframes

### Machine Learning Scenarios
- **Imbalanced Classification:** 5% fraud rate challenges
- **Temporal Validation:** Time-based train/test splits
- **Feature Engineering:** 51 pre-computed risk indicators
- **Concept Drift:** 4-month period for drift detection
- **Online Learning:** Sequential transaction processing

## Anonymization Pipeline Recommendations

### Priority 1: Direct Identifiers
- Hash `account_id`, `merchant_id`, `device_id` with SHA-256 + salt
- Maintain consistency for temporal analysis
- Remove raw identifiers before distribution

### Priority 2: Quasi-Identifiers
- Generalize `home_postal_fsa` to first character if k<5
- Group rare cities/provinces into "Other"
- Coarsen coordinates to 2 decimal places or city centroids
- Aggregate device/browser/OS combinations

### Priority 3: Sensitive Attributes
- Add Laplace noise to amounts (ε≈0.3)
- Bucket distances into quantile ranges
- Coarsen timestamps to hour or day level
- Cap velocity features at 99th percentile

### Priority 4: Temporal Consistency
- Apply consistent time shifts per account
- Preserve transaction sequences
- Maintain relative time intervals
- Protect against timing attacks

## Data Quality Features

### Realistic Patterns
- **Temporal Logic:** Transactions follow realistic daily/weekly patterns
- **Geographic Consistency:** 92% transactions within expected regions
- **Amount Distributions:** Log-normal with realistic outliers
- **Fraud Clusters:** Correlated fraud patterns (device, time, merchant)

### Edge Cases & Anomalies
- **Micro-payments:** 1% transactions under $1
- **Large Transactions:** 1% over $1,000 (max $9,999)
- **Clock Skew:** Some negative time-since-previous values
- **Missing Geolocation:** ~20% records lack coordinates
- **New Device Bursts:** Concentrated around fraud events

## Usage Scenarios

### Scenario 1: Real-time Fraud Detection
Test streaming fraud detection with sub-second latency requirements

### Scenario 2: Privacy-Preserving Model Training
Develop fraud models on anonymized data maintaining >90% accuracy

### Scenario 3: Federated Learning
Simulate multi-bank fraud detection without sharing raw transactions

### Scenario 4: Synthetic Data Generation
Use as seed data for generating larger synthetic fraud datasets

### Scenario 5: Regulatory Compliance
Test PCI-DSS and privacy regulation compliance workflows

## File Structure
```
fraud_test_data/
├── FRAUD_CA_V1_10k.csv         # Main transaction dataset
├── FRAUD_CA_V1_passport.json   # Detailed field specifications
├── FRAUD_CA_V1_dictionary.csv  # Reference codes and mappings
└── README.md                    # This documentation
```

## Dataset Statistics

### Coverage Metrics
- **Geographic:** All 13 Canadian provinces/territories
- **Temporal:** 133 days of continuous transactions
- **Merchants:** 800+ unique merchants across 15 categories
- **Accounts:** 2,200+ unique customer accounts

### Quality Metrics
- **Completeness:** Core fields 100% populated
- **Consistency:** Logical relationships maintained
- **Accuracy:** Realistic statistical distributions
- **Timeliness:** Temporal sequences preserved

## Known Limitations

1. **Simplified Fraud Patterns:** Real fraud is more complex and evolving
2. **Geographic Bias:** Concentrated in major Canadian cities
3. **Limited International:** Only 4% cross-border transactions
4. **Static Risk Scores:** Merchant/device scores don't evolve
5. **Perfect Labels:** No label noise or delayed fraud detection

## Compliance & Ethics

- **100% Synthetic:** No real payment or personal information
- **PCI-DSS Compliant:** Suitable for payment card testing
- **Privacy-First:** Designed for privacy regulation compliance
- **Fair Testing:** Balanced demographic representation
- **Ethical AI:** Supports responsible fraud detection development

---
*Generated for PAMOLA Epic 2 Testing Suite - Canadian Fraud Detection Module*  
*Version 1.0 - August 2025*
