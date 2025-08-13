# Medical Rare Disease & Laboratory Datasets for Privacy-Preserving Healthcare Analytics

## Overview

Two comprehensive synthetic medical datasets designed for testing privacy-preserving analytics in rare disease registries and laboratory information systems. These datasets simulate Canadian healthcare data with realistic clinical patterns while containing no actual patient information. Designed specifically for federated learning scenarios with intentional patient overlap.

## Datasets

### 1. MED_RARE_TXP_REGISTRY - Rare Disease & Transplant Registry
**File:** `MED_RARE_TXP_REGISTRY_10k.csv`  
**Records:** 10,000 synthetic patient registry records  
**Patients:** ~5,100 unique individuals  
**Purpose:** Testing anonymization of rare disease cohorts and transplant registries  
**Passport:** `MED_RARE_TXP_REGISTRY_passport.json`

### 2. MED_LAB - Laboratory Results System  
**File:** `MED_LAB_10k.csv`  
**Records:** 10,000 synthetic laboratory test results  
**Patients:** ~3,600 unique individuals  
**Purpose:** Testing privacy techniques on clinical laboratory data  
**Passport:** `MED_LAB_passport.json`

### Supporting Files
**Dictionary:** `MED_RARE_TXP_REGISTRY_dictionary.csv` - Medical code mappings, reference ranges, and terminology

## Cross-Dataset Linkage for Federated Learning

### Linkage Capabilities
- **Shared Patients:** ~2,000-2,500 individuals appear in both datasets
- **Linkage Field:** `link_shared_id` enables privacy-preserving joins
  - Registry: 59% have linkage IDs
  - Laboratory: 56% have linkage IDs
- **Ground Truth:** `gt_person_id` for validation (remove before production)
- **Overlap Design:** Simulates real-world scenario where transplant patients have extensive lab monitoring

## Field Descriptions

### MED_RARE_TXP_REGISTRY Fields (36 fields)

#### Patient Identifiers & Demographics (9 fields)
- `registry_row_id`: Unique row identifier
- `registry_internal_id`: Registry's stable patient hash
- `gt_person_id`: Ground-truth ID for evaluation (~5,100 unique)
- `link_shared_id`: Cross-dataset linkage hash (59% populated)
- `birth_year`: Year of birth (1930-2015)
- `sex`: Biological sex (F/M/X)
- `home_postal_fsa`: Canadian postal FSA
- `home_province`: Province code
- `language_pref`: Language preference (EN/FR/Bilingual)

#### Disease & Transplant Information (7 fields)
- `disease_group`: Rare disease category (12 groups)
  - Top conditions: Hemophilia A, IPF, Sickle Cell, PKU, ALS
- `transplant_type`: Organ/tissue type (Kidney/Liver/Heart/Lung/Pancreas/HSCT)
- `diagnosis_date`: Initial diagnosis date
- `referral_date`: Transplant program referral
- `waitlist_status`: Not Listed/Active/Inactive/Removed (55% not listed)
- `transplant_date`: Actual transplant date (25% transplanted)
- `donor_type`: Deceased/Living donor (when applicable)

#### Genetic & Immunological Data (5 fields)
- `hla_typing`: HLA typing results (A/B/DR alleles)
- `pra_percent`: Panel reactive antibody percentage (0-100%)
- `genotype_gene`: Relevant gene symbol
- `genotype_variant`: Variant notation
- `immunosuppression`: Medication regimen (semicolon-separated)

#### Transplant Center Information (6 fields)
- `center_id`: Transplant center identifier (~100 centers)
- `center_name`: Center name
- `center_city`: Center location city
- `center_province`: Center province
- `center_lat`/`center_lng`: Center coordinates (~21% missing)

#### Clinical Notes & PHI Detection (5 fields)
- `clinician_note`: Clinical narrative (120-600 chars)
- `has_phi_in_text`: PHI presence flag (48% of records)
- `phi_email_count`: Embedded emails (0-3)
- `phi_phone_count`: Embedded phones (0-3)
- `phi_date_count`: Embedded dates (0-5)

#### Outcomes (3 fields)
- `hospitalized_30d`: 30-day hospitalization flag
- `graft_failure_1y`: 1-year graft failure (1.94% rate)
- `mortality_1y`: 1-year mortality (1.93% rate)

#### Metadata (3 fields)
- `created_at`: Record creation date
- `updated_at`: Last modification date
- `source_system`: Source system identifier

### MED_LAB Fields (30 fields)

#### Patient & Order Identifiers (6 fields)
- `lab_row_id`: Unique row identifier
- `lab_internal_id`: Lab's stable patient hash
- `gt_person_id`: Ground-truth ID (~3,600 unique)
- `link_shared_id`: Cross-dataset linkage (56% populated)
- `order_id`: Laboratory order number
- `birth_year`, `sex`, `home_postal_fsa`, `home_province`, `language_pref`: Demographics

#### Laboratory Site Information (6 fields)
- `site_id`: Lab facility identifier (~100 sites)
- `site_name`: Laboratory name
- `site_city`: Lab location city
- `site_province`: Lab province
- `site_lat`/`site_lng`: Lab coordinates (~23% missing)

#### Test Information (6 fields)
- `test_loinc`: LOINC code for test type
- `test_name`: Human-readable test name
- `specimen_type`: Sample type (Blood/Urine/Plasma/Serum/Sputum)
- `collection_ts`: Sample collection timestamp
- `received_ts`: Sample receipt timestamp

#### Test Results (4 fields)
- `result_value`: Numeric result value
- `result_units`: Measurement units
- `ref_range`: Reference range string
- `abnormal_flag`: Abnormal result indicator (6.69% abnormal)

#### Clinical Notes & PHI (5 fields)
- `note`: Result comments/notes (~16% populated)
- `has_phi_in_text`: PHI presence flag (16% of records)
- `phi_email_count`: Embedded emails
- `phi_phone_count`: Embedded phones
- `phi_date_count`: Embedded dates

## Key Data Characteristics

### MED_RARE_TXP_REGISTRY Statistics

#### Disease Distribution
- **Hemophilia A:** 8.8%
- **Idiopathic Pulmonary Fibrosis:** 8.6%
- **Sickle Cell Disease:** 8.6%
- **Cystic Fibrosis:** 8.2%
- **ALS:** 8.2%
- **Other rare diseases:** 57.6%

#### Transplant Statistics
- **Transplanted:** 25.14% of patients
- **Waitlist Active:** 14.8%
- **Graft Failure Rate:** 1.94% at 1 year
- **Mortality Rate:** 1.93% at 1 year
- **Equal distribution** across organ types (~16.7% each)

#### Immunological Markers
- **PRA Levels:** Median 21%, IQR 9-34%
- **HLA Typing:** Complete for all patients
- **Immunosuppression:** Triple therapy most common

### MED_LAB Statistics

#### Test Volume & Types
- **Top Tests:**
  - Complete Blood Count components (19123-9, 777-3)
  - Basic Metabolic Panel (2951-2, 1920-8)
  - Liver Function Tests (1742-6)
  - Inflammatory Markers (2160-0)
- **Specimen Types:** Equal distribution (~20% each type)
- **Abnormal Rate:** 6.69% of results flagged

#### Result Distributions
- **Median Value:** 17.24 (varies by test)
- **25th Percentile:** 4.47
- **75th Percentile:** 116.93
- **Wide range** reflecting different test types

## Testing Capabilities

### Federated Learning Scenarios

#### Vertical Federated Learning
```
Registry Dataset → Patient demographics, disease info, transplant data
     ↓ link_shared_id
Laboratory Dataset → Lab results, test patterns, biomarkers

Use Case: Predict transplant outcomes using combined clinical and lab data
```

#### Horizontal Federated Learning
```
Hospital A: Registry subset (provinces ON, QC)
Hospital B: Registry subset (provinces BC, AB)
Hospital C: Registry subset (provinces Atlantic)

Use Case: Train rare disease models across institutions
```

#### Cross-Silo Analytics
```
Registry Silo: Long-term outcomes, genetic data
Laboratory Silo: Real-time biomarkers, trends

Use Case: Early detection of transplant rejection
```

### Privacy & Anonymization Testing

#### Multi-Dataset Challenges
- **Linkage Attack Prevention:** Test resistance using quasi-identifiers
- **Inference Control:** Prevent attribute disclosure across datasets
- **Temporal Privacy:** Protect sequential lab results
- **Rare Disease Privacy:** Extra protection for small cohorts

#### Anonymization Strategies
- **K-anonymity:** Minimum group size 5 for rare diseases
- **L-diversity:** Ensure diversity in genetic markers
- **Differential Privacy:** Add noise to lab values (ε≈0.3)
- **Secure Multi-party Computation:** Test protocols for joint analysis

### Clinical Research Applications

#### Transplant Outcomes Research
- Survival analysis with censored data
- Rejection prediction using lab trends
- Donor-recipient matching optimization
- Immunosuppression protocol comparison

#### Rare Disease Studies
- Natural history modeling
- Biomarker discovery
- Genotype-phenotype correlations
- Treatment response prediction

## Data Quality Features

### Realistic Clinical Correlations
- **Disease-Transplant Logic:** Appropriate organ needs by disease
- **Lab-Clinical Alignment:** Abnormal labs correlate with outcomes
- **Temporal Consistency:** Lab results track disease progression
- **Geographic Patterns:** Center distribution matches population

### Intentional Complexity
- **Missing Data Patterns:** Realistic clinical missingness
- **PHI Contamination:** 48% registry, 16% lab notes contain PHI
- **Duplicate Testing:** Multiple labs per patient over time
- **Referral Patterns:** Geographic referral networks

## Anonymization Pipeline Recommendations

### Priority 1: Identifier Protection
```python
# Pseudonymize all IDs consistently across datasets
ids_to_hash = ['registry_internal_id', 'lab_internal_id', 
               'gt_person_id', 'link_shared_id']
for id_field in ids_to_hash:
    df[id_field] = hash_with_salt(df[id_field], project_salt)
```

### Priority 2: Quasi-Identifier Management
- Generalize `birth_year` to 5-year bands
- Coarsen `postal_fsa` to first character for k<5
- Group rare diseases into broader categories
- Aggregate lab sites by region

### Priority 3: Sensitive Data Protection
- Redact PHI from clinical notes using NER
- Add Laplace noise to lab values (preserve clinical validity)
- Coarsen timestamps to week/month level
- Round geographic coordinates to city centroids

### Priority 4: Cross-Dataset Consistency
- Maintain referential integrity of `link_shared_id`
- Apply same anonymization rules to shared fields
- Preserve temporal relationships
- Validate linkage after anonymization

## Usage Scenarios

### Scenario 1: Federated Transplant Model
Train rejection prediction model across multiple centers without data sharing

### Scenario 2: Rare Disease Natural History
Analyze disease progression using combined registry and lab data

### Scenario 3: Privacy-Preserving Biomarker Discovery
Identify lab patterns predictive of outcomes while protecting patient identity

### Scenario 4: Synthetic Data Generation
Generate larger synthetic cohorts maintaining clinical relationships

### Scenario 5: Regulatory Compliance Testing
Validate PIPEDA/HIPAA compliance for multi-institutional studies

## File Structure
```
medical_rare_disease_data/
├── MED_RARE_TXP_REGISTRY_10k.csv        # Registry dataset
├── MED_RARE_TXP_REGISTRY_passport.json  # Registry specifications
├── MED_LAB_10k.csv                      # Laboratory dataset
├── MED_LAB_passport.json                # Lab specifications
├── MED_RARE_TXP_REGISTRY_dictionary.csv # Medical terminology
└── README.md                             # This documentation
```

## Dataset Integration Statistics

### Linkage Metrics
- **Total Unique Patients:** ~6,700 across both datasets
- **Overlapping Patients:** ~2,000-2,500 (30-37%)
- **Registry-only Patients:** ~2,600
- **Lab-only Patients:** ~1,100
- **Linkage Coverage:** 85% of true overlaps have `link_shared_id`

### Use Case Coverage
- **Transplant Monitoring:** Complete pre/post lab data
- **Rare Disease Cohorts:** 12 disease groups
- **Genetic Testing:** HLA + disease variants
- **Outcome Tracking:** 1-year follow-up data

## Compliance & Ethics

- **100% Synthetic:** No real patient information
- **PIPEDA/HIPAA Aligned:** Suitable for privacy regulation testing
- **Rare Disease Considerations:** Extra privacy for vulnerable populations
- **Clinical Validity:** Medically reviewed relationships
- **Ethical AI:** Supports responsible healthcare AI development

## Known Limitations

1. **Simplified Biology:** Real disease mechanisms more complex
2. **Static Genetics:** No somatic mutations or evolution
3. **Limited Medications:** Simplified immunosuppression regimens
4. **Perfect Linkage:** Real-world matching is messier
5. **Temporal Simplification:** No complex longitudinal patterns

---
*Generated for PAMOLA Epic 2 Testing Suite - Federated Healthcare Analytics Module*  
*Version 1.0 - August 2025*
