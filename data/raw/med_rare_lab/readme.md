# Synthetic Medical Rare Disease & Laboratory Datasets

> **⚠️ Synthetic Data Notice - No Real Patient Information**
>
> These datasets are **fully synthetic** - programmatically generated for testing purposes.
> **No real Protected Health Information (PHI) is included.**
>
> - Not collected from any healthcare systems, disease registries, or laboratory information systems
> - Not derived from any real patient records, transplant registries, or clinical databases
> - Not extracted from any hospital systems, rare disease foundations, or lab networks
> - All patient IDs, diagnoses, genetic data, lab results, and clinical notes are artificially generated
> - Any resemblance to actual patients, providers, or healthcare encounters is coincidental
>
> These datasets are designed for **testing federated learning, privacy-preserving analytics, and anonymization pipelines**.
> They are not intended for clinical decision-making, medical research, or patient care.
>
> *This documentation is for technical reference, not legal or medical advice.*

---

## Overview

Two comprehensive synthetic medical datasets designed for testing privacy-preserving analytics in rare disease registries and laboratory information systems. These datasets simulate Canadian healthcare data with realistic clinical patterns and intentional patient overlap for federated learning scenarios.

**Key Feature:** ~2,200 patients appear in both datasets via `link_shared_id` for cross-silo analytics testing.

---

## Data Generation

**Method:** Rule-based synthetic generation with controlled distributions and cross-dataset linkage

| Component | Approach |
|-----------|----------|
| Patient demographics | Faker-style with Canadian population distributions |
| Rare diseases | 12 disease groups with appropriate genetic variants |
| Transplant data | Realistic waitlist and outcome patterns |
| Laboratory results | LOINC-coded tests with normal/abnormal distributions |
| Clinical notes | Template generation with intentional PHI injection |
| Cross-linkage | Shared patient pool with consistent `link_shared_id` |

**Seed:** 42 (reproducible)

---

## Datasets

### 1. MED_RARE_TXP_REGISTRY - Rare Disease & Transplant Registry

| Property | Value |
|----------|-------|
| **File** | `MED_RARE_TXP_REGISTRY_10k.csv` |
| **Records** | 10,000 |
| **Unique Patients** | ~5,100 |
| **Purpose** | Testing anonymization of rare disease cohorts |
| **Passport** | `MED_RARE_TXP_REGISTRY_passport.json` |

### 2. MED_LAB - Laboratory Results System

| Property | Value |
|----------|-------|
| **File** | `MED_LAB_10k.csv` |
| **Records** | 10,000 |
| **Unique Patients** | ~3,600 |
| **Purpose** | Testing privacy techniques on clinical laboratory data |
| **Passport** | `MED_LAB_passport.json` |

### Supporting Files

| File | Description |
|------|-------------|
| `MED_RARE_TXP_REGISTRY_dictionary.csv` | Medical code mappings and terminology |

---

## Cross-Dataset Linkage for Federated Learning

### Linkage Statistics

| Metric | Value |
|--------|-------|
| Total unique patients | ~6,500 |
| Shared patients (in both) | ~2,200 (34%) |
| Registry-only patients | ~2,900 |
| Lab-only patients | ~1,400 |

### Linkage Fields

| Field | Registry Coverage | Lab Coverage | Purpose |
|-------|-------------------|--------------|---------|
| `link_shared_id` | ~50% | ~69% | Privacy-preserving cross-dataset joins |
| `gt_person_id` | 100% | 100% | Ground-truth for validation (remove before production) |

### Federated Learning Scenarios

**Vertical Federated Learning:**
```
Registry Dataset → Demographics, disease info, transplant data, genetics
       ↓ link_shared_id
Laboratory Dataset → Lab results, biomarkers, test patterns

Use Case: Predict transplant outcomes using combined clinical and lab data
```

**Horizontal Federated Learning:**
```
Hospital A: Registry subset (provinces ON, QC)
Hospital B: Registry subset (provinces BC, AB)
Hospital C: Registry subset (provinces Atlantic)

Use Case: Train rare disease models across institutions without data sharing
```

---

## MED_RARE_TXP_REGISTRY Fields (36 fields)

### Patient Identifiers and Demographics

| Field | Type | Description | Anonymization |
|-------|------|-------------|---------------|
| `registry_row_id` | string | Unique row ID: `PMLA-RR{index}-SYN` | Hash with salt |
| `registry_internal_id` | string | Registry's stable patient hash | Hash with salt |
| `gt_person_id` | string | Ground-truth ID (~5,100 unique) | **Remove before release** |
| `link_shared_id` | string | Cross-dataset linkage (50% populated) | Hash with salt |
| `birth_year` | integer | Year of birth (1930-2015) | 5-year bins |
| `sex` | string | F/M/X | Keep |
| `home_postal_fsa` | string | Canadian postal FSA | First char if k<5 |
| `home_province` | string | Province code | Keep or group |
| `language_pref` | string | EN/FR/Bilingual | Keep |

### Disease and Transplant Information

| Field | Type | Description |
|-------|------|-------------|
| `disease_group` | string | Rare disease category (12 groups) |
| `transplant_type` | string | Kidney/Liver/Heart/Lung/Pancreas/HSCT |
| `diagnosis_date` | date | Initial diagnosis date |
| `referral_date` | date | Transplant program referral |
| `waitlist_status` | string | Not Listed/Active/Inactive/Removed |
| `transplant_date` | date | Actual transplant date (~15% transplanted) |
| `donor_type` | string | Deceased/Living (when applicable) |

### Genetic and Immunological Data

| Field | Type | Description |
|-------|------|-------------|
| `hla_typing` | string | HLA typing (A/B/DR alleles) |
| `pra_percent` | float | Panel reactive antibody (0-100%) |
| `genotype_gene` | string | Relevant gene symbol |
| `genotype_variant` | string | Variant notation |
| `immunosuppression` | string | Medication regimen (semicolon-separated) |

### Transplant Center Information

| Field | Type | Missing % | Description |
|-------|------|-----------|-------------|
| `center_id` | string | 0% | Center ID: `PMLA-CTR{index}-SYN` |
| `center_name` | string | 0% | Synthetic center name |
| `center_city` | string | 0% | Center city |
| `center_province` | string | 0% | Center province |
| `center_lat` | float | ~21% | Latitude (jittered) |
| `center_lng` | float | ~21% | Longitude (jittered) |

### Clinical Notes and PHI Detection

| Field | Type | Description |
|-------|------|-------------|
| `clinician_note` | string | Clinical narrative (120-600 chars) |
| `has_phi_in_text` | boolean | PHI presence flag (~48% true) |
| `phi_email_count` | integer | Emails in note (0-3) |
| `phi_phone_count` | integer | Phones in note (0-3) |
| `phi_date_count` | integer | Dates in note (0-5) |

### Outcomes

| Field | Type | Rate | Description |
|-------|------|------|-------------|
| `hospitalized_30d` | boolean | ~15% | 30-day hospitalization |
| `graft_failure_1y` | boolean | ~2% | 1-year graft failure |
| `mortality_1y` | boolean | ~2% | 1-year mortality |

---

## MED_LAB Fields (30 fields)

### Patient and Order Identifiers

| Field | Type | Description | Anonymization |
|-------|------|-------------|---------------|
| `lab_row_id` | string | Unique row ID: `PMLA-LR{index}-SYN` | Hash with salt |
| `lab_internal_id` | string | Lab's stable patient hash | Hash with salt |
| `gt_person_id` | string | Ground-truth ID (~3,600 unique) | **Remove before release** |
| `link_shared_id` | string | Cross-dataset linkage (~69% populated) | Hash with salt |
| `order_id` | string | Lab order number | Hash with salt |

### Laboratory Site Information

| Field | Type | Missing % | Description |
|-------|------|-----------|-------------|
| `site_id` | string | 0% | Site ID: `PMLA-LAB{index}-SYN` |
| `site_name` | string | 0% | Laboratory name |
| `site_city` | string | 0% | Lab city |
| `site_province` | string | 0% | Lab province |
| `site_lat` | float | ~21% | Latitude |
| `site_lng` | float | ~21% | Longitude |

### Test Information

| Field | Type | Description |
|-------|------|-------------|
| `test_loinc` | string | LOINC code |
| `test_name` | string | Human-readable test name |
| `specimen_type` | string | Blood/Serum/Plasma/Urine/Sputum |
| `collection_ts` | datetime | Sample collection timestamp |
| `received_ts` | datetime | Sample receipt timestamp |

### Test Results

| Field | Type | Description |
|-------|------|-------------|
| `result_value` | float | Numeric result |
| `result_units` | string | Measurement units |
| `ref_range` | string | Reference range |
| `abnormal_flag` | string | N (normal) or A (abnormal) - ~18% abnormal |

### Notes and PHI Detection

| Field | Type | Description |
|-------|------|-------------|
| `note` | string | Result comments (~16% populated) |
| `has_phi_in_text` | boolean | PHI presence flag (~2% true) |
| `phi_email_count` | integer | Emails in note |
| `phi_phone_count` | integer | Phones in note |
| `phi_date_count` | integer | Dates in note |

---

## Key Data Characteristics

### Disease Distribution (Registry)

| Disease Group | Prevalence |
|---------------|------------|
| Hemophilia A | ~8% |
| Idiopathic Pulmonary Fibrosis | ~8% |
| Sickle Cell Disease | ~8% |
| Cystic Fibrosis | ~8% |
| ALS | ~8% |
| Phenylketonuria | ~8% |
| Huntington Disease | ~8% |
| Marfan Syndrome | ~8% |
| Wilson Disease | ~8% |
| Gaucher Disease | ~8% |
| Fabry Disease | ~8% |
| Pompe Disease | ~8% |

### Transplant Statistics (Registry)

| Metric | Value |
|--------|-------|
| Transplanted | ~15% |
| Waitlist Active | ~25% |
| Graft Failure (1yr) | ~2% |
| Mortality (1yr) | ~2% |

### Laboratory Tests (Lab)

| LOINC | Test Name | Units |
|-------|-----------|-------|
| 19123-9 | Hemoglobin | g/dL |
| 777-3 | Platelets | 10*3/uL |
| 2951-2 | Sodium | mmol/L |
| 1920-8 | AST | U/L |
| 1742-6 | ALT | U/L |
| 2160-0 | Creatinine | mg/dL |
| 3094-0 | BUN | mg/dL |
| 2345-7 | Glucose | mg/dL |
| 6690-2 | WBC | 10*3/uL |
| 4544-3 | Hematocrit | % |

---

## Testing Capabilities

### Privacy and Anonymization

| Technique | Application |
|-----------|-------------|
| K-anonymity | Demographic quasi-identifiers (k>=5 for rare diseases) |
| L-diversity | Genetic markers diversity |
| Differential Privacy | Lab values (epsilon ~0.3) |
| Secure MPC | Cross-dataset analysis protocols |
| PHI Detection | Clinical notes with ground-truth flags |

### Cross-Dataset Challenges

| Challenge | Description |
|-----------|-------------|
| Linkage attack prevention | Test resistance using quasi-identifiers |
| Inference control | Prevent attribute disclosure across datasets |
| Temporal privacy | Protect sequential lab results |
| Rare disease privacy | Extra protection for small cohorts |

### Clinical Research Applications

| Use Case | Relevant Data |
|----------|---------------|
| Transplant outcomes | Waitlist, outcomes, lab trends |
| Rare disease natural history | Diagnosis dates, progression |
| Biomarker discovery | Lab patterns, outcomes correlation |
| Genotype-phenotype | Genetic variants, disease severity |

---

## Anonymization Pipeline

### Priority 1: Identifier Protection

```python
# Pseudonymize all IDs consistently across datasets
ids_to_hash = ['registry_internal_id', 'lab_internal_id', 'link_shared_id']
for id_field in ids_to_hash:
    df[id_field] = hash_with_salt(df[id_field], project_salt)

# CRITICAL: Remove ground-truth IDs before any release
df = df.drop(columns=['gt_person_id'])
```

### Priority 2: Quasi-Identifier Management

```
- birth_year -> 5-year bands
- home_postal_fsa -> First character if k<5
- disease_group -> Consider grouping rare conditions
- center_city/site_city -> Region or suppress
```

### Priority 3: Sensitive Data Protection

```
- Clinical notes -> NER redaction (EMAIL, PHONE, DATE, NAME)
- Lab values -> Laplace noise (preserve clinical validity)
- Timestamps -> Coarsen to week/month
- Coordinates -> Round to city centroids
```

### Priority 4: Cross-Dataset Consistency

```
- Apply same anonymization to shared fields
- Maintain referential integrity of link_shared_id
- Validate linkage after anonymization
- Preserve temporal relationships
```

---

## Synthetic Fingerprints

All IDs contain synthetic markers for lineage tracking:

| Dataset | Field | Pattern | Example |
|---------|-------|---------|---------|
| Registry | registry_row_id | `PMLA-RR{index}-SYN` | `PMLA-RR000001-SYN` |
| Registry | center_id | `PMLA-CTR{index}-SYN` | `PMLA-CTR000005-SYN` |
| Lab | lab_row_id | `PMLA-LR{index}-SYN` | `PMLA-LR000001-SYN` |
| Lab | site_id | `PMLA-LAB{index}-SYN` | `PMLA-LAB000010-SYN` |
| Both | gt_person_id | `PMLA-P{index}-SYN` | `PMLA-P005466-SYN` |

---

## File Structure

```
med_rare_lab/
├── MED_RARE_TXP_REGISTRY_10k.csv        # Registry dataset
├── MED_RARE_TXP_REGISTRY_passport.json  # Registry metadata
├── MED_LAB_10k.csv                      # Laboratory dataset
├── MED_LAB_passport.json                # Lab metadata
├── MED_RARE_TXP_REGISTRY_dictionary.csv # Code mappings
└── README.md                            # This documentation
```

---

## Usage Examples

### Load Both Datasets

```python
import pandas as pd

registry = pd.read_csv("data/raw/med_rare_lab/MED_RARE_TXP_REGISTRY_10k.csv")
lab = pd.read_csv("data/raw/med_rare_lab/MED_LAB_10k.csv")

print(f"Registry: {len(registry)} records, {registry['gt_person_id'].nunique()} patients")
print(f"Lab: {len(lab)} records, {lab['gt_person_id'].nunique()} patients")
```

### Find Shared Patients

```python
# Using ground-truth (for validation only)
registry_patients = set(registry['gt_person_id'])
lab_patients = set(lab['gt_person_id'])
shared = registry_patients & lab_patients
print(f"Shared patients: {len(shared)}")

# Using link_shared_id (privacy-preserving)
registry_links = set(registry['link_shared_id'].dropna())
lab_links = set(lab['link_shared_id'].dropna())
linkable = registry_links & lab_links
print(f"Linkable via link_shared_id: {len(linkable)}")
```

### Join Datasets for Analysis

```python
# Get transplant patients with their lab data
transplanted = registry[registry['transplant_date'].notna()]

# Join via link_shared_id
merged = transplanted.merge(
    lab,
    on='link_shared_id',
    suffixes=('_reg', '_lab')
)

# Analyze lab patterns for transplant outcomes
outcome_analysis = merged.groupby('graft_failure_1y').agg({
    'result_value': 'mean',
    'abnormal_flag': lambda x: (x == 'A').mean()
})
```

---

## Known Limitations

1. **Simplified biology** - Real disease mechanisms are more complex
2. **Static genetics** - No somatic mutations or disease evolution
3. **Limited medications** - Simplified immunosuppression regimens
4. **Perfect linkage** - Real-world matching is messier
5. **Temporal simplification** - No complex longitudinal patterns
6. **Geographic simplification** - Urban bias in facility distribution

---

## Intended Use

These datasets are intended for:

- ✅ Testing federated learning algorithms
- ✅ Developing privacy-preserving cross-silo analytics
- ✅ Evaluating linkage attack resistance
- ✅ Testing PHI detection and de-identification
- ✅ Educational purposes in health informatics

These datasets are **not** intended for:

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
**Version:** 1.0.0 (Epic 2 Testing Suite - Federated Healthcare Module)
