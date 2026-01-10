# Synthetic 3-Way Cross-Party Anti-Fraud Dataset

> **⚠️ Synthetic Data Notice - No Real Financial Information**
> 
> This dataset is **fully synthetic** - programmatically generated for testing purposes.**No real Personally Identifiable Information (PII) or financial data is included.**
> 
> * Not collected from any bank, payment processor, or telecommunications provider
> * Not derived from any real transaction logs, fraud cases, or customer records
> * Not extracted from any financial institution, card network, or carrier system
> * All customer IDs, transaction amounts, phone numbers, and behavioral signals are artificially generated
> * Any resemblance to actual persons, institutions, or fraud patterns is coincidental
> 
> This dataset is designed for **testing federated learning, privacy-preserving analytics, PSI protocols, and synthetic data generation pipelines**.It is not intended for production fraud detection, credit decisions, or customer risk assessment.
> 
> *This documentation is for technical reference, not legal or financial advice.*

* * *

## Overview

Synthetic dataset designed for demonstrating **privacy-preserving cross-party fraud detection** using Vertical Federated Learning (VFL) and Private Set Intersection (PSI).

**Key Feature:** ~8,300 synthetic persons distributed across three parties with ~4,600 appearing in all datasets via PSI-compatible linkage tokens.

* * *

## Scenario

Three parties collaborate to detect fraud that no single party can identify alone:

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │    BANK     │     │     PSP     │     │   TELECOM   │
    │  (Active)   │◄───►│  (Passive)  │◄───►│  (Passive)  │
    │  Has Label  │     │  Payments   │     │  SIM/OTP    │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                     │                   │
           └─────────────────────┼───────────────────┘
                               │
                        ┌─────▼───────┐
                        │   PSI/VFL   │
                        │  Protocols  │
                        └─────────────┘

### Use Cases

| Application | Description |
| --- | --- |
| **Privacy Engineering** | Data profiling, synthetic data generation, PSI linkage demonstration |
| **Federated Learning** | 3-party VFL training (Bank=active with labels, PSP/Telecom=passive) |
| **Cross-Party Analytics** | Testing secure multi-party computation protocols |

* * *

## Data Generation

**Method:** Rule-based synthetic generation with controlled distributions and cross-party linkage

| Component | Approach |
| --- | --- |
| Customer demographics | Canadian population distributions (provinces, FSAs) |
| Transaction amounts | Log-normal distribution with realistic ranges |
| Fraud patterns | 4 cross-party scenarios with correlated signals |
| Linkage tokens | Per-party HMAC-SHA256 with independent salts |
| Behavioral signals | Correlated across parties for fraud cases |

**Seed:** 42 (reproducible)

* * *

## Dataset Contents

### Files

| File | Records | Description |
| --- | --- | --- |
| `BANK_ANTIFRAUD.csv` | 46,003 | Bank transactions with fraud labels |
| `PAYMENT_ANTIFRAUD.csv` | 46,231 | PSP payment authorizations |
| `TELECOM_ANTIFRAUD.csv` | 29,044 | Telecom events (OTP, SIM swap) |
| `CROSSPARTY_FRAUD_SCENARIOS.csv` | 217 | Cross-party fraud pattern examples |
| `CROSSPARTY_LINKAGE_STATS.json` | —   | Linkage overlap statistics |
| `*_passport.json` | —   | Schema + field statistics |
| `MANIFEST.json` | —   | File checksums (SHA256) |

### Variants

    release/
    ├── public/           # For external sharing
    │   ├── *.csv         # NO gt_person_id
    │   └── MANIFEST.json
    │
    ├── dev/              # For internal testing
    │   ├── *.csv         # WITH gt_person_id
    │   └── PSI_GROUND_TRUTH.json
    │
    └── tools/
        ├── generate_antifraud_crossparty.py
        ├── prepare_release.py
        └── validate.py

* * *

## Cross-Party Linkage for Federated Learning

### Linkage Statistics

| Metric | Value |
| --- | --- |
| Total unique persons | ~8,300 |
| Shared across all 3 parties | ~4,600 (55%) |
| Bank-PSP overlap | ~7,200 |
| Bank-Telecom overlap | ~6,500 |

### Linkage Fields

| Field              | Description                          | Purpose                              |     |
| ------------------ | ------------------------------------ | ------------------------------------ | --- |
| `psi_phone_token`  | HMAC-SHA256(phone, PARTY_SALT)[:32]  | Privacy-preserving cross-party joins |     |
| `psi_email_token`  | HMAC-SHA256(email, PARTY_SALT)[:32]  | Additional linkage key               |     |
| `psi_device_token` | HMAC-SHA256(device, PARTY_SALT)[:32] | Device fingerprint linkage           |     |
| `gt_person_id`     | Ground-truth ID (dev only)           | Validation and testing               |     |

**Note:** PSI tokens are NOT directly joinable — each party uses independent salts. PSI protocol required for matching.

### Federated Learning Scenarios

**Vertical Federated Learning:**

    Bank Dataset    → Account data, transaction history, fraud labels
          ↓ PSI protocol
    PSP Dataset     → Payment velocity, risk scores, CVV/AVS checks
          ↓ PSI protocol  
    Telecom Dataset → SIM swap flags, OTP patterns, device history
    
    Use Case: Detect SIM-swap fraud combining all three signal sources

* * *

## Schema

### BANK_ANTIFRAUD.csv (36 fields)

#### Identifiers

| Field | Type | Description | Anonymization |
| --- | --- | --- | --- |
| `bank_record_id` | string | Unique row ID: `PMLA-BA-*-SYN` | Hash with salt |
| `psi_phone_token` | string | Bank's phone PSI token | PSI-compatible |
| `psi_email_token` | string | Bank's email PSI token | PSI-compatible |
| `customer_id` | string | Internal customer ID | Hash with salt |
| `gt_person_id` | string | Ground-truth ID (dev only) | **Remove before release** |

#### Transaction Data

| Field | Type | Description |
| --- | --- | --- |
| `transaction_ts` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount (CAD) |
| `account_type` | enum | checking, savings, credit |
| `channel` | enum | online, mobile, branch, atm |

#### Behavioral Signals

| Field | Type | Description |
| --- | --- | --- |
| `is_new_device` | bool | First time device seen |
| `is_new_ip` | bool | First time IP seen |
| `login_attempts_24h` | int | Login attempts in 24h |
| `failed_logins_24h` | int | Failed logins in 24h |
| `transactions_24h` | int | Transactions in 24h |
| `amount_24h` | float | Total amount in 24h |

#### Labels

| Field | Type | Rate | Description |
| --- | --- | --- | --- |
| **`is_fraud_bank`** | bool | ~2% | **LABEL: fraud indicator** |
| `fraud_type_bank` | string | —   | Fraud category |

### PAYMENT_ANTIFRAUD.csv (28 fields)

#### Transaction Data

| Field | Type | Description |
| --- | --- | --- |
| `payment_record_id` | string | Unique row ID |
| `authorization_ts` | datetime | Auth timestamp |
| `amount` | float | Transaction amount |
| `merchant_mcc` | int | Merchant category code |
| `pos_entry_mode` | enum | chip, contactless, ecommerce |

#### Verification Signals

| Field | Type | Description |
| --- | --- | --- |
| `cvv_match` | bool | CVV verification result |
| `avs_match` | bool | Address verification result |
| `3ds_result` | string | 3D Secure authentication result |

#### Velocity and Risk

| Field | Type | Description |
| --- | --- | --- |
| `velocity_1h` | int | Transactions in last hour |
| `velocity_24h` | int | Transactions in 24h |
| `cross_border_flag` | bool | International transaction |
| `risk_score` | int | PSP risk score (0-1000) |
| `is_fraud_payment` | bool | Fraud indicator |

### TELECOM_ANTIFRAUD.csv (24 fields)

#### Subscriber Data

| Field | Type | Description |
| --- | --- | --- |
| `telecom_record_id` | string | Unique row ID |
| `subscriber_id` | string | Internal subscriber ID |
| `event_type` | enum | call, sms, data, otp |
| `event_ts` | datetime | Event timestamp |

#### SIM and Device Signals

| Field | Type | Description |
| --- | --- | --- |
| `sim_swap_date` | date | Last SIM swap date |
| `sim_swap_count_90d` | int | SIM swaps in 90 days |
| `otp_requests_24h` | int | OTP requests in 24h |
| `unique_devices_30d` | int | Unique devices in 30 days |
| `international_roaming_flag` | bool | Roaming active |
| `is_fraud_telecom` | bool | Fraud indicator |

* * *

## Cross-Party Fraud Scenarios

| Scenario | Description | Cross-Party Signals |
| --- | --- | --- |
| **SIM Swap Attack** | Fraudster swaps SIM to intercept OTP | Telecom: SIM swap + OTP burst → Bank: new device + high-value txn |
| **Account Takeover** | Credential compromise | Bank: new IP + failed logins → PSP: velocity spike |
| **Card Testing** | Testing stolen card numbers | PSP: micro-transactions + CVV failures → Telecom: OTP attempts |
| **Identity Theft** | New account fraud | Bank: new customer → PSP: immediate high spend |

### SIM Swap Attack Signature (Example)

    Telecom signals:
      - otp_requests_24h: 8-20 (burst)
      - sim_swap_count_90d: 1-3 (recent swap)
    
    Bank signals:
      - is_new_device: True
      - login_attempts_24h: 5-12
      - amount: 3-8x typical
    
    PSP signals:
      - velocity_24h: elevated
      - 3ds_result: authenticated (intercepted OTP)
    
    Without telecom data: appears as normal authenticated transaction
    With cross-party integration: high fraud probability detected

* * *

## Demonstrated Results (Fixed Cohort Evaluation)

Evaluated on 6,531 persons present in all three datasets:

| Model | ROC-AUC | Recall@1%FPR | Incremental |
| --- | --- | --- | --- |
| Bank-only | 0.869 | 44.2% | baseline |
| Bank + PSP | 0.874 | **58.1%** | +14pp |
| Bank + PSP + Telecom | 0.873 | **60.5%** | +2.3pp |
| VFL 3-party | 0.816 | —   | gap: 5.7% |

**Key Insight:** PSP data adds **+14 percentage points** to fraud recall at 1% FPR on the same population.

**VFL Gap Analysis:** 5.7% gap represents realistic privacy-utility tradeoff for Hardy et al. VFL protocol with linear models (~94% utility retention).

* * *

## Quick Start

### 1. Validate Integrity

    python tools/validate.py --dir ./public

### 2. Load Data (Python)

    import pandas as pd
    
    bank = pd.read_csv('public/BANK_ANTIFRAUD.csv')
    psp = pd.read_csv('public/PAYMENT_ANTIFRAUD.csv')
    telecom = pd.read_csv('public/TELECOM_ANTIFRAUD.csv')
    
    print(f"Bank: {len(bank):,} records, {bank['is_fraud_bank'].sum()} fraud")
    print(f"PSP: {len(psp):,} records")
    print(f"Telecom: {len(telecom):,} records")

### 3. PSI Linkage Demo

    # In real PSI, parties compute set intersection without revealing identifiers
    # Each party has tokens generated with their own secret salt:
    
    # Bank has:    psi_phone_token = 'a3f2b7c4...' (HMAC with BANK_SALT)
    # PSP has:     psi_phone_token = 'x9y8z7w6...' (HMAC with PSP_SALT)
    # Telecom has: psi_phone_token = 'j5k6l7m8...' (HMAC with TEL_SALT)
    
    # PSI protocol determines if they refer to same person
    # WITHOUT revealing the underlying phone number or the salts

### 4. Regenerate Dataset

    python tools/generate_antifraud_crossparty.py --seed 42 --output-dir ./output
    python tools/prepare_release.py --input-dir ./output --output-dir ./release

* * *

## Testing Capabilities

### Privacy and Anonymization

| Technique | Application |
| --- | --- |
| PSI Protocols | Cross-party linkage without revealing identifiers |
| Vertical FL | Collaborative model training without data sharing |
| Differential Privacy | Adding noise to aggregated statistics |
| Secure MPC | Multi-party computation for fraud scoring |

### Cross-Party Challenges

| Challenge | Description |
| --- | --- |
| Linkage attack prevention | Test resistance using quasi-identifiers |
| Inference control | Prevent attribute disclosure across parties |
| Population overlap bias | Fixed cohort evaluation methodology |
| Privacy-utility tradeoff | VFL gap measurement |

* * *

## Synthetic Fingerprints

All IDs contain synthetic markers for lineage tracking:

| Dataset | Field | Pattern | Example |
| --- | --- | --- | --- |
| Bank | bank_record_id | `PMLA-BA-*-SYN` | `PMLA-BA-000001-SYN` |
| Payment | payment_record_id | `PMLA-PA-*-SYN` | `PMLA-PA-000001-SYN` |
| Telecom | telecom_record_id | `PMLA-TA-*-SYN` | `PMLA-TA-000001-SYN` |
| All (dev) | gt_person_id | `PERSON-*` | `PERSON-000001` |

* * *

## Known Limitations

1. **Simplified fraud patterns** - Real fraud is more sophisticated and evolving
2. **No temporal dependencies** - Events are not sequentially correlated
3. **Perfect PSI assumption** - Real-world identifier matching is messier
4. **Static behavior** - No concept drift or seasonal variations
5. **Balanced overlap** - Real cross-party overlap is often smaller
6. **Canadian geography only** - Single jurisdiction simplification

* * *

## Intended Use

This dataset is intended for:

* ✅ Testing federated learning algorithms
* ✅ Developing privacy-preserving cross-party analytics
* ✅ Evaluating PSI protocol implementations
* ✅ Testing synthetic data generation pipelines
* ✅ Educational purposes in privacy engineering
* ✅ Academic publications (with citation)

This dataset is **not** intended for:

* ❌ Production fraud detection systems
* ❌ Training models for real-world deployment
* ❌ Credit decisions or customer risk assessment
* ❌ Re-identification research
* ❌ Regulatory compliance certification

* * *

## File Structure

    3way-antifraud/
    ├── public/
    │   ├── BANK_ANTIFRAUD.csv
    │   ├── PAYMENT_ANTIFRAUD.csv
    │   ├── TELECOM_ANTIFRAUD.csv
    │   ├── CROSSPARTY_FRAUD_SCENARIOS.csv
    │   ├── CROSSPARTY_LINKAGE_STATS.json
    │   ├── *_passport.json
    │   ├── MANIFEST.json
    │   ├── README.md
    │   └── LICENSE
    │
    ├── dev/
    │   ├── (same as public)
    │   ├── gt_person_id columns
    │   └── PSI_GROUND_TRUTH.json
    │
    └── tools/
        ├── generate_antifraud_crossparty.py
        ├── prepare_release.py
        └── validate.py

* * *

## Citation

    @dataset{crossparty_antifraud_2024,
      title={Synthetic 3-Way Cross-Party Anti-Fraud Dataset},
      author={REALM Data Team},
      year={2024},
      publisher={REALM Data},
      version={1.0.0},
      url={https://github.com/realm-data/synthetic-datasets}
    }

* * *

## License

Apache 2.0

See [LICENSE](./LICENSE) for full terms.

* * *

## Changelog

### v1.0.0 (2024-01)

* Initial release
* 3 party datasets (Bank, PSP, Telecom)
* Per-party PSI tokens with independent salts
* Fixed cohort evaluation results
* Public/Dev variants with ground truth separation

* * *

**Maintainer:** [REALM Data](https://realmdata.io)  | **Repository:** [github.com/DGT-Network/PAMOLA](https://github.com/DGT-NETWORK/PAMOLA/data) | **Version:** 1.0.0
