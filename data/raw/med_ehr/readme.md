# Synthetic Electronic Health Records Dataset

> **⚠️ Synthetic Data Notice - No Real Patient Information**
>
> This dataset is **fully synthetic** - programmatically generated for testing purposes.
> **No real Protected Health Information (PHI) is included.**
>
> - Not collected from any healthcare systems, EHRs, or medical facilities
> - Not derived from any real patient records, clinical encounters, or claims data
> - Not extracted from any hospital databases, insurance systems, or health registries
> - All patient IDs, diagnoses, medications, lab results, and clinical notes are artificially generated
> - Any resemblance to actual patients, providers, or healthcare encounters is coincidental
>
> This dataset is designed for **testing privacy-preserving analytics and anonymization pipelines**.
> It is not intended for clinical decision-making or medical research.
>
> *This documentation is for technical reference, not legal or medical advice.*

---

## Overview

Comprehensive synthetic outpatient EHR dataset designed for testing privacy-preserving healthcare analytics, medical data anonymization, and clinical decision support systems. This dataset simulates Canadian outpatient encounters with realistic clinical patterns while containing no actual patient information.

**Domain:** Canadian outpatient healthcare  
**Records:** 10,000 synthetic encounters  
**Patients:** ~4,600 unique synthetic individuals  
**Key Feature:** PHI detection ground-truth flags for validation  
**Primary Use Cases:** PHI detection, text de-identification, healthcare analytics testing

---

## Data Generation

**Method:** Rule-based synthetic generation with clinical pattern modeling

| Component | Approach |
|-----------|----------|
| Patient demographics | Faker-style with Canadian population distributions |
| Clinical encounters | Template-based with realistic visit patterns |
| Diagnoses | ICD-10-CA codes with age/gender-appropriate distributions |
| Lab results | LOINC codes with normal/abnormal value distributions |
| Medications | Common drug classes with appropriate indications |
| Clinical notes | Template generation with intentional PHI injection |
| Geography | Canadian provinces with jittered coordinates |

**PHI Injection:** 52% of clinical notes contain synthetic PHI for detection testing.

**Reproducibility:** Passport JSON contains generation parameters and statistics.

---

## Dataset Files

| File | Description |
|------|-------------|
| `EHR_FHIR_LITE_OUTPATIENT_10k.csv` | Main dataset (10,000 encounters) |
| `EHR_FHIR_LITE_OUTPATIENT_passport.json` | Field specifications and statistics |
| `EHR_FHIR_LITE_OUTPATIENT_dictionary.csv` | Medical code mappings |

**Period:** December 2024 - August 2025 (simulated)  
**Coding Systems:** ICD-10-CA (diagnoses), LOINC (labs), CCI (procedures)

---

## Field Specifications (58 fields)

### Patient and Provider Identifiers

| Field | Type | Description | Anonymization |
|-------|------|-------------|---------------|
| `patient_id` | string | Synthetic patient ID (~4,600 unique) | Hash with salt |
| `provider_id` | string | Synthetic provider ID (~1,200 unique) | Hash with salt |
| `encounter_id` | string | Unique encounter ID | Hash with salt |

### Facility Information

| Field | Type | Missing % | Description |
|-------|------|-----------|-------------|
| `facility_id` | string | 0% | Facility identifier (~200 facilities) |
| `facility_name` | string | 0% | Synthetic facility name |
| `facility_city` | string | 0% | Facility city |
| `facility_province` | string | 0% | Province code |
| `facility_country` | string | 0% | Country (CA) |
| `facility_lat` | float | 21% | Latitude (jittered) |
| `facility_lng` | float | 21% | Longitude (jittered) |
| `provider_specialty` | string | 0% | Medical specialty |

### Patient Demographics

| Field | Type | Description | Anonymization |
|-------|------|-------------|---------------|
| `birth_year` | integer | Year of birth (1930-2015) | 5-year bins |
| `sex` | string | F/M/X | Keep |
| `language_pref` | string | EN/FR/Bilingual | Keep |
| `home_city` | string | City of residence | Generalize to region |
| `home_province` | string | Province | Keep or group |
| `home_postal_fsa` | string | Postal FSA (first 3 chars) | First char only if k<5 |
| `home_lat` | float | Latitude (21% missing) | Round to 2 decimals |
| `home_lng` | float | Longitude (21% missing) | Round to 2 decimals |

### Encounter Details

| Field | Type | Description |
|-------|------|-------------|
| `encounter_start` | datetime | Visit start timestamp |
| `encounter_end` | datetime | Visit end timestamp |
| `encounter_duration_min` | integer | Duration (median 35 min) |
| `hour` | integer | Start hour (0-23) |
| `weekday` | integer | Day of week (0=Monday) |
| `is_weekend` | boolean | Weekend indicator |

### Clinical Information - Diagnoses

| Field | Type | Description |
|-------|------|-------------|
| `diagnoses_icd10` | string | ICD-10-CA codes (semicolon-separated) |
| `diag_primary` | string | Primary diagnosis code |
| `chief_complaint_text` | string | Presenting complaint (10 categories) |
| `triage_note` | string | Triage assessment (10 templates) |

### Clinical Information - Procedures and Medications

| Field | Type | Missing % | Description |
|-------|------|-----------|-------------|
| `procedures` | string | 66% | Procedure codes (semicolon-separated) |
| `medications_text` | string | 25% | Medication list (pipe-separated) |

### Clinical Information - Laboratory Results

| Field | Type | Description |
|-------|------|-------------|
| `labs_json` | JSON | Lab results array with LOINC codes |
| `abnormal_lab_count` | integer | Count of abnormal results (0-10) |

**Lab JSON Structure:**
```json
{
  "loinc": "2345-7",
  "name": "Glucose",
  "value": 95,
  "units": "mg/dL",
  "ref_range": "70-100",
  "abnormal_flag": "N"
}
```

### Clinical Information - Vital Signs

| Field | Type | Range | Units |
|-------|------|-------|-------|
| `vital_hr` | integer | 40-180 | bpm |
| `vital_sbp` | integer | 80-200 | mmHg |
| `vital_dbp` | integer | 40-120 | mmHg |
| `vital_rr` | integer | 8-40 | breaths/min |
| `vital_spo2` | integer | 85-100 | % |
| `vital_temp_c` | float | 35.0-40.0 | Celsius |

### Chronic Conditions

| Field | Type | Prevalence | Description |
|-------|------|------------|-------------|
| `cond_dm2` | boolean | 25% | Type 2 diabetes |
| `cond_htn` | boolean | 35% | Hypertension |
| `cond_ckd` | boolean | 8% | Chronic kidney disease |
| `cond_asthma` | boolean | 12% | Asthma |

### Clinical Notes and PHI Detection

| Field | Type | Description |
|-------|------|-------------|
| `clinical_note` | string | Synthetic note (120-600 chars) |
| `has_phi_in_text` | boolean | PHI presence flag (52% true) |
| `phi_email_count` | integer | Emails in note (0-3) |
| `phi_phone_count` | integer | Phones in note (0-3) |
| `phi_date_count` | integer | Dates in note (0-5) |

### Billing and Claims

| Field | Type | Description |
|-------|------|-------------|
| `payer_type` | string | Public 70%, Private 25%, Self-pay 5% |
| `allowed_amount` | float | Claim amount ($10-$500, median $54) |
| `patient_copay` | float | Copayment ($0-$100) |
| `claim_status` | string | Paid 75%, Adjusted 10%, Pending 10%, Denied 5% |

### Outcomes and Follow-up

| Field | Type | Rate | Description |
|-------|------|------|-------------|
| `revisit_30d` | boolean | 18.8% | 30-day revisit flag |
| `mortality_30d` | boolean | 1.32% | 30-day mortality flag |
| `next_encounter_start` | datetime | 54% | Next visit date |

### System Metadata

| Field | Type | Description |
|-------|------|-------------|
| `source_system` | string | Source EHR system identifier |
| `created_at` | datetime | Record creation timestamp |
| `updated_at` | datetime | Last update timestamp |
| `distance_home_to_facility_km` | float | Travel distance (67% populated) |

---

## Key Data Characteristics

### Patient Population

| Metric | Value |
|--------|-------|
| Total patients | ~4,600 unique |
| Age distribution | Median 45, range 10-95 |
| Gender | Female 52%, Male 47%, Other 1% |
| Language | English 60%, French 30%, Bilingual 10% |
| Geography | All Canadian provinces |

### Clinical Patterns

| Metric | Value |
|--------|-------|
| Encounters per patient | Average 2.2 |
| Visit duration | Median 35 min (IQR: 21-48) |
| Weekend visits | 15% |
| After-hours | 20% |

### Top Primary Diagnoses

| ICD-10 | Description | Prevalence |
|--------|-------------|------------|
| B34.9 | Viral infection | 10% |
| E11.9 | Type 2 diabetes | 9% |
| F41.1 | Anxiety disorder | 8% |
| H66.90 | Otitis media | 8% |
| G43.909 | Migraine | 8% |
| I10 | Essential hypertension | 7% |
| J06.9 | Upper respiratory infection | 6% |
| K21.9 | GERD | 5% |
| J45.909 | Asthma | 5% |
| K40.90 | Hernia | 5% |

---

## PHI Detection Testing

### Ground-Truth Flags

The dataset includes ground-truth PHI flags for validating detection systems:

| Flag | Description | Purpose |
|------|-------------|---------|
| `has_phi_in_text` | Binary PHI presence | Classification accuracy |
| `phi_email_count` | Email count in note | Detection recall |
| `phi_phone_count` | Phone count in note | Detection recall |
| `phi_date_count` | Date count in note | Detection recall |

### PHI Patterns in Clinical Notes

| PHI Type | Example Pattern | Prevalence |
|----------|-----------------|------------|
| Email | "contact patient at john.doe@email.com" | ~15% |
| Phone | "callback number 416-555-1234" | ~20% |
| Date | "follow-up scheduled for March 15, 2025" | ~30% |
| Name | "discussed with spouse Jane" | ~10% |

### Recommended Validation

```python
from pamola_core.text import PIIDetector

detector = PIIDetector(entities=['EMAIL', 'PHONE', 'DATE', 'PERSON'])

# Detect PHI in clinical notes
results = detector.scan(df['clinical_note'])

# Validate against ground truth
precision = calculate_precision(results, df['has_phi_in_text'])
recall_email = calculate_recall(results['EMAIL'], df['phi_email_count'])
recall_phone = calculate_recall(results['PHONE'], df['phi_phone_count'])
```

---

## Testing Capabilities

### Healthcare Analytics

| Use Case | Relevant Fields |
|----------|-----------------|
| Clinical decision support | diagnoses, labs, vitals, conditions |
| Risk stratification | revisit_30d, mortality_30d, conditions |
| Quality metrics | encounter patterns, outcomes |
| Population health | geographic distribution, prevalence |
| Utilization analysis | visit patterns, provider workload |

### Privacy and Anonymization

| Technique | Application |
|-----------|-------------|
| PHI detection | Clinical notes with ground-truth flags |
| K-anonymity | Demographic quasi-identifiers |
| Text de-identification | NER redaction testing |
| Geographic privacy | Coordinate generalization |
| Temporal anonymization | Date shifting/coarsening |

### Interoperability Testing

| Standard | Application |
|----------|-------------|
| FHIR | Resource structure compatibility |
| ICD-10-CA | Diagnosis code validation |
| LOINC | Lab result coding |
| CCI | Procedure coding |

---

## Anonymization Pipeline

### Priority 1: Direct Identifiers

```
- patient_id, provider_id, encounter_id -> SHA-256 + salt
- Maintain referential integrity across encounters
- Remove identifiers from clinical_note text
```

### Priority 2: Quasi-Identifiers

```
- birth_year -> 5-year bands
- home_postal_fsa -> First character if k<5
- facility_city -> Group rare values to "Other"
- provider_specialty -> Broader categories
```

### Priority 3: Clinical Text

```
- clinical_note -> NER redaction
  - Email: [EMAIL_REDACTED]
  - Phone: [PHONE_REDACTED]
  - Date: [DATE_REDACTED]
  - Name: [NAME_REDACTED]
- Preserve clinical terminology and diagnostic content
```

### Priority 4: Sensitive Attributes

```
- allowed_amount, patient_copay -> Laplace noise (epsilon ~0.3)
- encounter_start/end -> Coarsen to date or week
- coordinates -> Round to 2 decimals
- distance_home_to_facility_km -> Cap at 95th percentile
```

---

## Known Edge Cases

| Issue | Prevalence | Description |
|-------|------------|-------------|
| Clock skew | ~0.5% | Negative encounter durations |
| Missing geography | ~2% | Province code "ZZ" |
| Incomplete labs | ~1% | Missing lab results |
| CSV injection | ~0.3% | Special characters in facility names |
| Outlier vitals | Rare | Extreme but clinically possible values |

---

## File Structure

```
med_ehr/
├── EHR_FHIR_LITE_OUTPATIENT_10k.csv        # Main dataset
├── EHR_FHIR_LITE_OUTPATIENT_passport.json  # Generation metadata
├── EHR_FHIR_LITE_OUTPATIENT_dictionary.csv # Code mappings
└── README.md                                # This documentation
```

---

## Usage Examples

### Load and Explore

```python
import pandas as pd

df = pd.read_csv("data/raw/med_ehr/EHR_FHIR_LITE_OUTPATIENT_10k.csv")

# Check PHI distribution
print(f"Records with PHI: {df['has_phi_in_text'].mean():.1%}")
print(f"Total embedded emails: {df['phi_email_count'].sum()}")
print(f"Total embedded phones: {df['phi_phone_count'].sum()}")

# Clinical summary
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Unique diagnoses: {df['diag_primary'].nunique()}")
```

### Test PHI Detection

```python
from pamola_core.text import PIIDetector

detector = PIIDetector()
results = detector.scan(df['clinical_note'])

# Compare with ground truth
tp = ((results['has_phi']) & (df['has_phi_in_text'])).sum()
fp = ((results['has_phi']) & (~df['has_phi_in_text'])).sum()
fn = ((~results['has_phi']) & (df['has_phi_in_text'])).sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
```

### Clinical Risk Model

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Predict 30-day readmission
features = ['vital_hr', 'vital_sbp', 'abnormal_lab_count', 
            'cond_dm2', 'cond_htn', 'encounter_duration_min']

X = df[features].fillna(0)
y = df['revisit_30d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)
```

---

## Known Limitations

1. **Simplified clinical logic** - Real medicine has more complex relationships
2. **Limited specialties** - Focus on primary care encounters
3. **Static conditions** - Chronic diseases do not progress over time
4. **Perfect documentation** - No missing or incorrect clinical entries
5. **Geographic simplification** - Urban bias in facility distribution
6. **No longitudinal complexity** - Limited patient history depth

---

## Intended Use

This dataset is intended for:

- ✅ Testing PHI detection and de-identification algorithms
- ✅ Developing privacy-preserving healthcare analytics
- ✅ Training and evaluating anonymization pipelines
- ✅ Educational purposes in health informatics
- ✅ Benchmarking NER models for clinical text

This dataset is **not** intended for:

- ❌ Clinical decision-making or patient care
- ❌ Medical research or epidemiological studies
- ❌ Training production clinical AI systems
- ❌ Regulatory compliance certification

---

## License

Apache 2.0 - same as PAMOLA.CORE repository.

See [LICENSE](../../../LICENSE) for full terms.

---

**Maintainer:** [Realm Inveo Inc.](https://realmdata.io)  
**Repository:** [github.com/DGT-Network/PAMOLA](https://github.com/DGT-Network/PAMOLA)  
**Version:** 1.0.0 (Epic 2 Testing Suite - Healthcare Module)
