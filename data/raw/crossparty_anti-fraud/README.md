# PAMOLA 3-Way Cross-Party Anti-Fraud Dataset

**Version:** 1.0.0 | **License:** Apache-2.0 | **Maintainers:** PAMOLA Team (REALM Data)

* * *

## Overview

Synthetic dataset designed for demonstrating **privacy-preserving cross-party fraud detection** using Vertical Federated Learning (VFL) and Private Set Intersection (PSI).

### Use Cases

| Project         | Use Case                                                    |
| --------------- | ----------------------------------------------------------- |
| **PAMOLA CORE** | Data profiling, synthetic data generation, PSI linkage demo |
| **GUARDORA**    | 3-party VFL training (Bank=active, PSP/Telecom=passive)     |

### Scenario

Three parties collaborate to detect fraud that no single party can identify alone:

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │    BANK     │     │     PSP     │     │   TELECOM   │
    │  (Active)   │◄───►│  (Passive)  │◄───►│  (Passive)  │
    │  Has Label  │     │  Payments   │     │  SIM/OTP    │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                        ┌──────▼──────┐
                        │   PSI/VFL   │
                        │  Protocols  │
                        └─────────────┘

* * *

## Dataset Contents

### Files

| File | Records | Description |
| --- | --- | --- |
| `BANK_ANTIFRAUD.csv` | 46,003 | Bank transactions with fraud labels |
| `PAYMENT_ANTIFRAUD.csv` | 46,231 | PSP payment authorizations |
| `TELECOM_ANTIFRAUD.csv` | 29,044 | Telecom events (OTP, SIM swap) |
| `CROSSPARTY_FRAUD_SCENARIOS.csv` | 217 | Cross-party fraud patterns |
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

## Schema

### Linkage Keys (PSI Tokens)

Each party generates **independent PSI tokens** using party-specific salts:

| Field | Description |
| --- | --- |
| `psi_phone_token` | HMAC-SHA256(phone, PARTY_SALT)[:32] |
| `psi_email_token` | HMAC-SHA256(email, PARTY_SALT)[:32] |
| `psi_device_token` | HMAC-SHA256(device_fp, PARTY_SALT)[:32] |

**Note:** Tokens are NOT directly joinable. PSI protocol required for matching.

### BANK_ANTIFRAUD.csv

| Field | Type | Description |
| --- | --- | --- |
| `bank_record_id` | string | Unique record ID (PMLA-BA-*-SYN) |
| `psi_phone_token` | string | Bank's phone PSI token |
| `psi_email_token` | string | Bank's email PSI token |
| `customer_id` | string | Internal customer ID |
| `account_type` | enum | checking, savings, credit |
| `transaction_ts` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount (CAD) |
| `channel` | enum | online, mobile, branch, atm |
| `is_new_device` | bool | First time device seen |
| `is_new_ip` | bool | First time IP seen |
| `login_attempts_24h` | int | Login attempts in 24h |
| `failed_logins_24h` | int | Failed logins in 24h |
| `transactions_24h` | int | Transactions in 24h |
| `amount_24h` | float | Total amount in 24h |
| **`is_fraud_bank`** | bool | **LABEL: fraud indicator** |
| `fraud_type_bank` | string | Fraud category |

### PAYMENT_ANTIFRAUD.csv

| Field | Type | Description |
| --- | --- | --- |
| `payment_record_id` | string | Unique record ID |
| `psi_phone_token` | string | PSP's phone PSI token |
| `authorization_ts` | datetime | Auth timestamp |
| `amount` | float | Transaction amount |
| `merchant_mcc` | int | Merchant category code |
| `pos_entry_mode` | enum | chip, contactless, ecommerce |
| `cvv_match` | bool | CVV verification result |
| `avs_match` | bool | Address verification result |
| `velocity_1h` | int | Transactions in last hour |
| `velocity_24h` | int | Transactions in 24h |
| `cross_border_flag` | bool | International transaction |
| `risk_score` | int | PSP risk score (0-1000) |
| `is_fraud_payment` | bool | Fraud indicator |

### TELECOM_ANTIFRAUD.csv

| Field | Type | Description |
| --- | --- | --- |
| `telecom_record_id` | string | Unique record ID |
| `psi_phone_token` | string | Telecom's phone PSI token |
| `subscriber_id` | string | Internal subscriber ID |
| `event_type` | enum | call, sms, data, otp |
| `event_ts` | datetime | Event timestamp |
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

* * *

## Demonstrated Results (Fixed Cohort)

Evaluated on 6,531 persons present in all three datasets:

| Model | ROC-AUC | Recall@1%FPR | Incremental |
| --- | --- | --- | --- |
| Bank-only | 0.869 | 44.2% | baseline |
| Bank + PSP | 0.874 | **58.1%** | +14pp |
| Bank + PSP + Telecom | 0.873 | **60.5%** | +2.3pp |
| VFL 3-party | 0.816 | —   | gap: 5.7% |

**Key Insight:** PSP data adds **+14 percentage points** to fraud recall at 1% FPR.

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
    # For demo purposes, we show the concept:
    
    # Bank has: psi_phone_token = 'a3f2b7c4...'
    # PSP has:  psi_phone_token = 'x9y8z7w6...'  (different salt!)
    #
    # PSI protocol determines if they refer to same person
    # WITHOUT revealing the underlying phone number

### 4. Regenerate Dataset

    python tools/generate_antifraud_crossparty.py --seed 42 --output-dir ./output
    python tools/prepare_release.py --input-dir ./output --output-dir ./release

* * *

## Data Quality Notes

### Realism

* Canadian geography (provinces, FSAs, area codes)
* Realistic fraud rates (~3.5%)
* Correlated cross-party signals for fraud scenarios
* Log-normal transaction amounts

### Synthetic Limitations

* No temporal dependencies between events
* Simplified fraud patterns
* No seasonal variations
* Linkage based on hashed identifiers (not raw PII)

### Privacy

* **PUBLIC variant:** No `gt_person_id`, cannot directly link records
* **DEV variant:** Includes ground truth for testing only
* All identifiers are synthetic (no real PII)

* * *

## Intended Use

### Permitted

✅ Privacy-preserving ML research✅ VFL/PSI protocol development✅ PAMOLA/GUARDORA demonstration✅ Academic publications (with citation)

### Not Permitted

❌ Production fraud detection systems❌ Training models for real-world deployment❌ Re-identification research❌ Commercial use without license

* * *

## Citation

    @dataset{pamola_antifraud_2024,
      title={PAMOLA 3-Way Cross-Party Anti-Fraud Synthetic Dataset},
      author={REALM Data Team},
      year={2024},
      publisher={REALM Data},
      version={1.0.0},
      url={https://github.com/realm-data/pamola-datasets}
    }

* * *

## Changelog

### v1.0.0 (2024-01)

* Initial release
* 3 party datasets (Bank, PSP, Telecom)
* Per-party PSI tokens
* Fixed cohort evaluation results
* Public/Dev variants

* * *

## Contact

* **Project:** [PAMOLA](https://pamola.realmdata.io)
* **Issues:** GitHub Issues
* **Email:** pamola@realmdata.io
