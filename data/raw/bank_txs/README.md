# Synthetic Banking Dataset - Multi-Table Edition

> **⚠️ Synthetic Data Notice**
> 
> This dataset is **fully synthetic** - programmatically generated for testing purposes.**No real personal or financial information is included.**
> 
> * Not collected from any external sources
> * Not derived from any real customer, account, or transaction data
> * Any resemblance to actual persons, accounts, or transactions is coincidental
> 
> *This documentation is for technical reference, not legal advice.*

* * *

## Overview

Comprehensive synthetic banking dataset with 6 linked tables simulating a Canadian retail bank. Designed for testing data profiling, privacy-preserving analytics, cross-table linkage attacks, and anonymization pipelines.

**Total Records:** 16,100 across 6 tables**Unique Customers:** 1,500**Relationships:** 5 foreign key relationships**Data Quality Issues:** Intentionally included for testing

* * *

## Data Generation

**Method:** Programmatic generation using Python (seed=42 for reproducibility)

| Component | Approach |
| --- | --- |
| Identifiers | Sequential with synthetic prefix `PMLA-*-SYN` |
| Names | Faker-style from curated Canadian name lists |
| Addresses | Synthetic Canadian addresses with valid postal codes |
| Financial values | Log-normal distributions with realistic ranges |
| Timestamps | Realistic patterns over 6+ months |
| Relationships | Referential integrity maintained across tables |

**Generated:** January 2026 for PAMOLA Epic 2 testing

* * *

## Dataset Structure

    bank_txs/
    ├── CUSTOMERS.csv        # 1,500 customer profiles with PII
    ├── ACCOUNTS.csv         # 2,100 bank accounts
    ├── TRANSACTIONS.csv     # 10,000 account transactions
    ├── LOANS.csv            # 800 loan records
    ├── CREDIT_CARDS.csv     # 1,200 credit card accounts
    ├── FEEDBACK.csv         # 500 customer feedback records
    ├── BANK_TXS_passport.json   # Metadata and statistics
    └── README.md            # This documentation

* * *

## Entity Relationship Diagram

    ┌─────────────────┐
    │   CUSTOMERS     │
    │  (customer_id)  │──────────┬─────────────┬─────────────┬─────────────┐
    └────────┬────────┘          │             │             │             │
             │                   │             │             │             │
             │ 1:N               │ 1:N         │ 1:N         │ 1:N         │
             ▼                   ▼             ▼             ▼             ▼
    ┌─────────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
    │    ACCOUNTS     │  │    LOANS    │  │  CREDIT  │  │   FEEDBACK   │  │          │
    │  (account_id)   │  │  (loan_id)  │  │   CARDS  │  │ (feedback_id)│  │          │
    └────────┬────────┘  └─────────────┘  │ (card_id)│  └──────────────┘  │          │
             │                            └──────────┘                     │          │
             │ 1:N                                                         │          │
             ▼                                                             │          │
    ┌─────────────────┐                                                    │          │
    │  TRANSACTIONS   │                                                    │          │
    │(transaction_id) │                                                    │          │
    └─────────────────┘                                                    │          │

* * *

## Table Specifications

### 1. CUSTOMERS (1,500 records)

Customer profiles with direct identifiers and demographics.

| Field | Type | PII Category | Description |
| --- | --- | --- | --- |
| `customer_id` | string | Direct ID | Synthetic ID: `PMLA-C000001-SYN` |
| `first_name` | string | Direct ID | First name |
| `last_name` | string | Direct ID | Last name |
| `birth_date` | date | Quasi-ID | Date of birth |
| `age` | integer | Quasi-ID | Age in years |
| `gender` | string | Quasi-ID | Male/Female/Other/Prefer not to say |
| `address` | string | Direct ID | Street address |
| `city` | string | Quasi-ID | City name |
| `province` | string | Quasi-ID | Province code (ON, BC, QC, etc.) |
| `postal_code` | string | Quasi-ID | Full postal code (e.g., M5V 2H1) |
| `country` | string | Non-PII | Country (Canada) |
| `phone` | string | Direct ID | Phone number |
| `email` | string | Direct ID | Email address |
| `employer` | string | Quasi-ID | Employer name (nullable) |
| `employment_status` | string | Sensitive | Employment status |
| `annual_income` | float | Sensitive | Annual income CAD (nullable) |
| `segment` | string | Non-PII | RETAIL/SMALL_BUSINESS/PREMIUM/PRIVATE |
| `kyc_level` | string | Non-PII | BASIC/STANDARD/ENHANCED |
| `risk_rating` | string | Sensitive | LOW/MEDIUM/HIGH |
| `customer_since` | date | Quasi-ID | Account opening date |
| `is_active` | boolean | Non-PII | Active customer flag |

### 2. ACCOUNTS (2,100 records)

Bank account records linked to customers.

| Field | Type | PII Category | Description |
| --- | --- | --- | --- |
| `account_id` | string | Direct ID | Synthetic ID: `PMLA-A000001-SYN` |
| `customer_id` | string | FK  | Reference to CUSTOMERS |
| `account_type` | string | Non-PII | CHECKING/SAVINGS/BUSINESS/JOINT |
| `account_number` | string | Direct ID | Masked: `****1234` |
| `currency` | string | Non-PII | CAD/USD |
| `balance` | float | Sensitive | Current balance |
| `available_balance` | float | Sensitive | Available balance |
| `hold_amount` | float | Sensitive | Hold amount |
| `interest_rate` | float | Non-PII | Interest rate % |
| `overdraft_limit` | integer | Sensitive | Overdraft limit |
| `opened_date` | date | Quasi-ID | Account opening date |
| `last_activity_date` | date | Quasi-ID | Last activity date |
| `status` | string | Non-PII | ACTIVE/DORMANT/CLOSED/FROZEN |
| `branch_id` | string | Quasi-ID | Branch identifier |

### 3. TRANSACTIONS (10,000 records)

Account transaction records with temporal and behavioral data.

| Field | Type | PII Category | Description |
| --- | --- | --- | --- |
| `transaction_id` | string | Direct ID | UUID |
| `account_id` | string | FK  | Reference to ACCOUNTS |
| `transaction_type` | string | Non-PII | DEPOSIT/WITHDRAWAL/TRANSFER/etc. |
| `amount` | float | Sensitive | Transaction amount |
| `currency` | string | Non-PII | CAD/USD |
| `timestamp` | datetime | Quasi-ID | Transaction timestamp |
| `channel` | string | Quasi-ID | ONLINE/MOBILE/BRANCH/ATM/PHONE |
| `status` | string | Non-PII | COMPLETED/PENDING/FAILED/REVERSED |
| `description` | string | Non-PII | Transaction description |
| `merchant_name` | string | Quasi-ID | Merchant name (POS only) |
| `merchant_category` | string | Quasi-ID | Merchant category (POS only) |
| `reference_number` | string | Non-PII | Reference number |
| `balance_after` | float | Sensitive | Balance after transaction |
| `ip_address` | string | Quasi-ID | IP address (online/mobile) |
| `device_id` | string | Quasi-ID | Device fingerprint (online/mobile) |
| `geolocation` | string | Quasi-ID | Lat,Lng coordinates |
| `is_international` | boolean | Non-PII | Cross-border flag |
| `fx_rate` | float | Non-PII | Exchange rate |
| `anomaly_flag` | integer | Non-PII | Anomaly indicator (0/1) |

### 4. LOANS (800 records)

Loan applications and status records.

| Field | Type | PII Category | Description |
| --- | --- | --- | --- |
| `loan_id` | string | Direct ID | Synthetic ID: `PMLA-L000001-SYN` |
| `customer_id` | string | FK  | Reference to CUSTOMERS |
| `loan_type` | string | Non-PII | MORTGAGE/AUTO/PERSONAL/etc. |
| `principal_amount` | float | Sensitive | Loan principal |
| `interest_rate` | float | Non-PII | Interest rate % |
| `term_months` | integer | Non-PII | Loan term |
| `monthly_payment` | float | Sensitive | Monthly payment |
| `outstanding_balance` | float | Sensitive | Outstanding balance |
| `application_date` | date | Quasi-ID | Application date |
| `approval_date` | date | Quasi-ID | Approval date |
| `disbursement_date` | date | Quasi-ID | Disbursement date |
| `maturity_date` | date | Quasi-ID | Maturity date |
| `status` | string | Sensitive | APPROVED/ACTIVE/CLOSED/REJECTED/DEFAULTED |
| `collateral_type` | string | Non-PII | Collateral type |
| `credit_score_at_application` | integer | Sensitive | Credit score |
| `debt_to_income_ratio` | float | Sensitive | DTI ratio |
| `late_payments_count` | integer | Sensitive | Late payment count |
| `days_past_due` | integer | Sensitive | Days past due |

### 5. CREDIT_CARDS (1,200 records)

Credit card account records.

| Field | Type | PII Category | Description |
| --- | --- | --- | --- |
| `card_id` | string | Direct ID | Synthetic ID: `PMLA-CC000001-SYN` |
| `customer_id` | string | FK  | Reference to CUSTOMERS |
| `card_type` | string | Non-PII | VISA/MASTERCARD/AMEX/DISCOVER |
| `card_tier` | string | Non-PII | CLASSIC/GOLD/PLATINUM/INFINITE |
| `card_number_masked` | string | Direct ID | Masked: `****-****-****-1234` |
| `credit_limit` | float | Sensitive | Credit limit |
| `current_balance` | float | Sensitive | Current balance |
| `available_credit` | float | Sensitive | Available credit |
| `minimum_payment_due` | float | Sensitive | Minimum payment |
| `payment_due_date` | date | Non-PII | Payment due date |
| `last_payment_date` | date | Quasi-ID | Last payment date |
| `last_payment_amount` | float | Sensitive | Last payment amount |
| `interest_rate_purchase` | float | Non-PII | Purchase APR |
| `interest_rate_cash_advance` | float | Non-PII | Cash advance APR |
| `annual_fee` | float | Non-PII | Annual fee |
| `rewards_points` | integer | Non-PII | Rewards balance |
| `rewards_program` | string | Non-PII | Rewards program type |
| `issued_date` | date | Quasi-ID | Card issue date |
| `expiry_date` | string | Quasi-ID | Expiry YYYY-MM |
| `status` | string | Non-PII | ACTIVE/SUSPENDED/CLOSED/LOST_STOLEN |
| `utilization_rate` | float | Sensitive | Credit utilization % |

### 6. FEEDBACK (500 records)

Customer feedback and complaint records.

| Field | Type | PII Category | Description |
| --- | --- | --- | --- |
| `feedback_id` | string | Direct ID | Synthetic ID: `PMLA-FB000001-SYN` |
| `customer_id` | string | FK  | Reference to CUSTOMERS |
| `feedback_type` | string | Non-PII | COMPLAINT/SUGGESTION/PRAISE/INQUIRY/DISPUTE |
| `category` | string | Non-PII | Service category |
| `channel` | string | Quasi-ID | Submission channel |
| `subject` | string | Non-PII | Subject line |
| `description` | string | Sensitive | Feedback text |
| `submitted_date` | date | Quasi-ID | Submission date |
| `resolution_status` | string | Non-PII | Resolution status |
| `resolution_date` | date | Quasi-ID | Resolution date |
| `assigned_to` | string | Non-PII | Agent ID |
| `priority` | string | Non-PII | LOW/MEDIUM/HIGH/CRITICAL |
| `satisfaction_score` | integer | Sensitive | CSAT score 1-5 |
| `response_time_hours` | integer | Non-PII | Response time |

* * *

## Intentional Data Quality Issues

This dataset contains **intentionally introduced issues** for testing data quality detection.

### Summary

| Issue Type | Table | Count | Purpose |
| --- | --- | --- | --- |
| **Outliers** | ACCOUNTS | ~45 | High-balance accounts (>$100K) |
| **Outliers** | TRANSACTIONS | ~200 | Large transactions (>$50K) |
| **Format Violations** | CUSTOMERS | ~45 | Name formatting issues |
| **Format Violations** | TRANSACTIONS | ~300 | Description formatting issues |
| **Missing Values** | Multiple | ~2% | Nullable fields |

### Format Violation Types

| Type | Description | Example |
| --- | --- | --- |
| Case | Inconsistent capitalization | `JOHN` or `john` instead of `John` |
| Whitespace | Extra spaces/tabs | ` John ` or `John\t` |
| Encoding | Character substitution | Cyrillic `а` instead of Latin `a` |
| Truncation | Shortened values | `DEP` instead of `DEPOSIT` |

### Anomaly Flags

Transactions table includes `anomaly_flag` field (0/1) marking records suitable for anomaly detection testing. Approximately 5% of transactions are flagged.

* * *

## Key Statistics

### Customer Distribution

| Segment | Count | %   |
| --- | --- | --- |
| RETAIL | ~900 | 60% |
| SMALL_BUSINESS | ~300 | 20% |
| PREMIUM | ~225 | 15% |
| PRIVATE | ~75 | 5%  |

### Transaction Types

| Type | Count | %   |
| --- | --- | --- |
| WITHDRAWAL | ~2,000 | 20% |
| DEPOSIT | ~1,800 | 18% |
| TRANSFER_OUT | ~1,500 | 15% |
| TRANSFER_IN | ~1,500 | 15% |
| BILL_PAYMENT | ~1,200 | 12% |
| POS_PURCHASE | ~1,000 | 10% |
| ATM_WITHDRAWAL | ~500 | 5%  |
| INTERAC_ETRANSFER | ~300 | 3%  |
| PAYROLL | ~200 | 2%  |

### Loan Portfolio

| Type | Count | Avg Amount |
| --- | --- | --- |
| MORTGAGE | ~240 | $650K |
| AUTO | ~200 | $45K |
| PERSONAL | ~200 | $25K |
| LINE_OF_CREDIT | ~80 | $50K |
| STUDENT | ~40 | $50K |
| BUSINESS | ~40 | $250K |

* * *

## Synthetic Fingerprints

All IDs contain synthetic markers for lineage tracking.

| Table | Pattern | Example |
| --- | --- | --- |
| CUSTOMERS | `PMLA-C{index}-SYN` | `PMLA-C000001-SYN` |
| ACCOUNTS | `PMLA-A{index}-SYN` | `PMLA-A000001-SYN` |
| LOANS | `PMLA-L{index}-SYN` | `PMLA-L000001-SYN` |
| CREDIT_CARDS | `PMLA-CC{index}-SYN` | `PMLA-CC000001-SYN` |
| FEEDBACK | `PMLA-FB{index}-SYN` | `PMLA-FB000001-SYN` |
| TRANSACTIONS | UUID v4 | `550e8400-e29b-...` |

**Purpose:**

* Prevent misrepresentation as real data
* Enable data lineage tracking
* Support audit and compliance testing

* * *

## Testing Capabilities

### Privacy & Anonymization

| Technique | Tables | Application |
| --- | --- | --- |
| K-anonymity | CUSTOMERS | Quasi-ID combinations (age, city, gender) |
| L-diversity | CUSTOMERS | Sensitive attribute diversity (income, risk) |
| Differential Privacy | TRANSACTIONS | Noise injection on amounts |
| Pseudonymization | All | Replace fingerprinted IDs |
| Generalization | CUSTOMERS | Age ranges, postal FSA only |
| Suppression | CUSTOMERS | Remove outlier records |

### Cross-Table Linkage Testing

| Attack Type | Tables | Method |
| --- | --- | --- |
| Direct linkage | All | Via customer_id FK |
| Transitive linkage | ACCOUNTS -> TRANSACTIONS | Account patterns |
| Temporal correlation | TRANSACTIONS + FEEDBACK | Timeline analysis |
| Behavioral fingerprinting | TRANSACTIONS | Spending patterns |

### Data Quality Testing

| Test | Tables | Target Issues |
| --- | --- | --- |
| Completeness | All | Missing values |
| Validity | CUSTOMERS, TRANSACTIONS | Format violations |
| Consistency | All | Referential integrity |
| Accuracy | ACCOUNTS, TRANSACTIONS | Outlier detection |

* * *

## Usage Examples

### Load All Tables

    import pandas as pd
    
    customers = pd.read_csv("data/raw/bank_txs/CUSTOMERS.csv")
    accounts = pd.read_csv("data/raw/bank_txs/ACCOUNTS.csv")
    transactions = pd.read_csv("data/raw/bank_txs/TRANSACTIONS.csv")
    loans = pd.read_csv("data/raw/bank_txs/LOANS.csv")
    credit_cards = pd.read_csv("data/raw/bank_txs/CREDIT_CARDS.csv")
    feedback = pd.read_csv("data/raw/bank_txs/FEEDBACK.csv")

### Join Customer with Transactions

    # Get customer transaction history
    customer_txns = transactions.merge(
        accounts[["account_id", "customer_id"]],
        on="account_id"
    ).merge(
        customers[["customer_id", "first_name", "last_name", "city"]],
        on="customer_id"
    )

### Test K-Anonymity

    from pamola_core.attacks import KAnonymityCheck
    
    quasi_ids = ["age", "gender", "city", "province"]
    result = KAnonymityCheck(
        data=customers,
        quasi_identifiers=quasi_ids,
        k_threshold=5
    ).evaluate()

* * *

## Known Limitations

1. **Simplified relationships** - Real banking has more complex account structures
2. **Static balances** - Balance_after in transactions is approximate
3. **No inter-account transfers** - Transfer_in/out don't link to counterparty
4. **Limited temporal patterns** - No seasonal or trend effects
5. **Uniform geographic distribution** - Real data has regional concentration

* * *

## License

Apache 2.0 - same as PAMOLA.CORE repository.

See [LICENSE](../../../LICENSE) for full terms.

* * *

**Maintainer:** [Realm Inveo Inc.](https://realmdata.io)**Repository:** [github.com/DGT-Network/PAMOLA](https://github.com/DGT-Network/PAMOLA)**Version:** 2.0.0 (Epic 2 Testing Suite - Multi-Table Edition)
