# Electronic Health Records Dataset for Privacy-Preserving Healthcare Analytics

## Overview

Comprehensive synthetic outpatient EHR dataset designed for testing privacy-preserving healthcare analytics, medical data anonymization, and clinical decision support systems. This dataset simulates Canadian outpatient encounters with realistic clinical patterns while containing no actual patient information.

## Dataset Files

### Core Dataset
**File:** `EHR_FHIR_LITE_OUTPATIENT_10k.csv`  
**Records:** 10,000 synthetic outpatient encounters  
**Period:** December 1, 2024 - August 11, 2025  
**Scope:** Canadian healthcare system (ICD-10-CA, LOINC, CCI coding)  
**Purpose:** Testing anonymization of electronic health records and claims data

### Supporting Files
**Passport:** `EHR_FHIR_LITE_OUTPATIENT_passport.json` - Detailed field specifications and anonymization guidelines  
**Dictionary:** `EHR_FHIR_LITE_OUTPATIENT_dictionary.csv` - Medical code mappings and reference values

## Field Descriptions (58 fields)

### Patient & Provider Identifiers (3 fields)
- `patient_id`: Stable patient identifier (~4,600 unique patients)
- `provider_id`: Healthcare provider identifier (~1,200 unique providers)
- `encounter_id`: Unique encounter/visit identifier

### Facility Information (9 fields)
- `facility_id`: Healthcare facility identifier (~200 facilities)
- `facility_name`: Facility/clinic name
- `facility_city`: Facility location city
- `facility_province`: Canadian province code
- `facility_country`: Country (CA)
- `facility_lat`/`facility_lng`: Facility coordinates (jittered, ~21% missing)
- `provider_specialty`: Provider specialty (Family Medicine/Internal Medicine/Emergency/Cardiology/etc.)

### Patient Demographics (8 fields)
- `birth_year`: Year of birth (1930-2015)
- `sex`: Biological sex (F/M/X)
- `language_pref`: Language preference (EN/FR/EN/FR bilingual)
- `home_city`: Patient's city of residence
- `home_province`: Patient's province
- `home_postal_fsa`: Canadian postal FSA (first 3 chars)
- `home_lat`/`home_lng`: Home coordinates (jittered, ~21% missing)

### Encounter Details (6 fields)
- `encounter_start`: Visit start timestamp
- `encounter_end`: Visit end timestamp
- `encounter_duration_min`: Duration in minutes (median 35 min)
- `hour`: Start hour (0-23)
- `weekday`: Day of week (0=Monday)
- `is_weekend`: Weekend indicator

### Clinical Information - Diagnoses (4 fields)
- `diagnoses_icd10`: ICD-10-CA codes (semicolon-separated list)
- `diag_primary`: Primary diagnosis code
- `chief_complaint_text`: Presenting complaint (10 categories)
- `triage_note`: Triage assessment note (10 standard templates)

### Clinical Information - Procedures & Medications (2 fields)
- `procedures`: Procedure codes (semicolon-separated, ~34% populated)
- `medications_text`: Medication list (pipe-separated, ~75% populated)

### Clinical Information - Laboratory Results (2 fields)
- `labs_json`: JSON array of lab results with LOINC codes (~99% populated)
  - Structure: `{loinc, name, value, units, ref_range, abnormal_flag}`
- `abnormal_lab_count`: Count of abnormal lab results (0-10)

### Clinical Information - Vital Signs (6 fields)
- `vital_hr`: Heart rate (40-180 bpm)
- `vital_sbp`: Systolic blood pressure (80-200 mmHg)
- `vital_dbp`: Diastolic blood pressure (40-120 mmHg)
- `vital_rr`: Respiratory rate (8-40 breaths/min)
- `vital_spo2`: Oxygen saturation (85-100%)
- `vital_temp_c`: Body temperature (35.0-40.0°C)

### Chronic Conditions (4 fields)
- `cond_dm2`: Type 2 diabetes flag (25% prevalence)
- `cond_htn`: Hypertension flag (35% prevalence)
- `cond_ckd`: Chronic kidney disease flag (8% prevalence)
- `cond_asthma`: Asthma flag (12% prevalence)

### Clinical Notes & PHI Detection (5 fields)
- `clinical_note`: Synthetic clinical note (120-600 chars)
- `has_phi_in_text`: PHI presence flag (52% of records)
- `phi_email_count`: Email addresses in note (0-3)
- `phi_phone_count`: Phone numbers in note (0-3)
- `phi_date_count`: Dates in note (0-5)

### Billing & Claims (4 fields)
- `payer_type`: Insurance type (public 70%, private 25%, self-pay 5%)
- `allowed_amount`: Allowed claim amount ($10-$500, median $54)
- `patient_copay`: Patient copayment ($0-$100)
- `claim_status`: Claim status (paid 75%, adjusted 10%, pending 10%, denied 5%)

### Outcomes & Follow-up (3 fields)
- `revisit_30d`: 30-day revisit flag (18.8% rate)
- `mortality_30d`: 30-day mortality flag (1.32% rate)
- `next_encounter_start`: Next visit date (~54% have follow-up)

### System Metadata (4 fields)
- `source_system`: Source EHR system identifier
- `created_at`: Record creation timestamp
- `updated_at`: Last update timestamp
- `distance_home_to_facility_km`: Travel distance (~67% populated)

## Key Data Characteristics

### Patient Population
- **Total Patients:** ~4,600 unique individuals
- **Age Distribution:** Median age 45, range 10-95 years
- **Gender:** Female 52%, Male 47%, Other 1%
- **Language:** English 60%, French 30%, Bilingual 10%
- **Geography:** All Canadian provinces represented

### Clinical Patterns
- **Visit Volume:** Average 2.2 encounters per patient
- **Visit Duration:** Median 35 minutes (IQR: 21-48 min)
- **Weekend Visits:** 15% of encounters
- **After-hours:** 20% outside 9am-5pm

### Top Primary Diagnoses
1. **B34.9** - Viral infection (10%)
2. **E11.9** - Type 2 diabetes (9%)
3. **F41.1** - Anxiety disorder (8%)
4. **H66.90** - Otitis media (8%)
5. **G43.909** - Migraine (8%)
6. **I10** - Essential hypertension (7%)
7. **J06.9** - Upper respiratory infection (6%)
8. **K21.9** - GERD (5%)
9. **J45.909** - Asthma (5%)
10. **K40.90** - Hernia (5%)

### Laboratory Testing
- **Test Frequency:** 99% of encounters include labs
- **Common Tests:** CBC, Basic Metabolic Panel, Lipid Panel, HbA1c
- **Abnormal Results:** Mean 2.5 abnormal values per encounter
- **LOINC Codes:** 50+ unique test types

### Medication Patterns
- **Prescription Rate:** 75% of encounters
- **Common Classes:** Antibiotics, Antihypertensives, Analgesics, Antidiabetics
- **Polypharmacy:** 15% of patients on 5+ medications

## Testing Capabilities

### Healthcare Analytics
- **Clinical Decision Support:** Test with diagnoses, labs, vitals
- **Risk Stratification:** 30-day readmission and mortality prediction
- **Quality Metrics:** Care gap identification, chronic disease management
- **Population Health:** Disease prevalence, geographic patterns
- **Utilization Analysis:** Visit patterns, provider workload

### Privacy & Anonymization
- **PHI Detection:** 52% of clinical notes contain synthetic PHI
- **K-anonymity:** Test on demographic quasi-identifiers
- **Text De-identification:** Redact names, dates, IDs from notes
- **Geographic Privacy:** Coordinate generalization and jittering
- **Temporal Anonymization:** Date shifting and coarsening

### Interoperability Testing
- **FHIR Compliance:** FHIR-lite structure compatibility
- **Code Systems:** ICD-10-CA, LOINC, CCI validation
- **Data Exchange:** HL7 message generation testing
- **Claims Processing:** Billing and reimbursement workflows

## Anonymization Pipeline Recommendations

### Priority 1: Direct Identifiers
- Pseudonymize `patient_id`, `provider_id`, `facility_id` with SHA-256 + salt
- Maintain referential integrity across encounters
- Remove any embedded identifiers in text fields

### Priority 2: Quasi-Identifiers
- Generalize `birth_year` to 5-year bands
- Coarsen `home_postal_fsa` to first character if k<5
- Group rare `facility_city` values into "Other"
- Aggregate provider specialties into broader categories

### Priority 3: Clinical Data
- Redact PHI from clinical notes using NER
  - Email patterns: `[EMAIL_REDACTED]`
  - Phone patterns: `[PHONE_REDACTED]`
  - Date patterns: `[DATE_REDACTED]`
  - Name patterns: `[NAME_REDACTED]`
- Preserve clinical content and medical terminology
- Maintain diagnostic code integrity

### Priority 4: Sensitive Attributes
- Add Laplace noise to billing amounts (ε≈0.3)
- Coarsen timestamps to date or week level
- Round geographic coordinates to 2 decimals
- Cap distance measurements at 95th percentile

## Data Quality Features

### Realistic Clinical Patterns
- **Comorbidity Correlations:** Diabetes + hypertension co-occurrence
- **Seasonal Variations:** Respiratory infections peak in winter
- **Age-appropriate Diagnoses:** Pediatric vs. geriatric conditions
- **Vital Sign Correlations:** BP, heart rate relationships
- **Lab Value Distributions:** Realistic normal and abnormal ranges

### Edge Cases & Anomalies
- **Clock Skew:** ~0.5% negative encounter durations
- **Missing Geography:** ~2% with province code "ZZ"
- **Incomplete Labs:** ~1% missing lab results
- **CSV Injection:** ~0.3% facility names with special characters
- **Outlier Vitals:** Extreme but clinically possible values

## Usage Scenarios

### Scenario 1: Clinical Risk Prediction
Develop 30-day readmission models using diagnoses, labs, and vitals

### Scenario 2: PHI Detection Testing
Validate de-identification algorithms on clinical notes with known PHI

### Scenario 3: Population Health Dashboard
Create geographic disease prevalence maps with privacy preservation

### Scenario 4: Quality Measure Calculation
Compute HEDIS/quality metrics on anonymized patient cohorts

### Scenario 5: Federated Learning
Simulate multi-hospital model training without data sharing

## File Structure
```
medical_test_data/
├── EHR_FHIR_LITE_OUTPATIENT_10k.csv        # Main EHR dataset
├── EHR_FHIR_LITE_OUTPATIENT_passport.json  # Field specifications
├── EHR_FHIR_LITE_OUTPATIENT_dictionary.csv # Code mappings
└── README.md                                # This documentation
```

## Dataset Statistics

### Coverage Metrics
- **Temporal:** 8+ months of continuous encounters
- **Geographic:** All Canadian provinces
- **Clinical:** 50+ diagnosis codes, 50+ lab types, 20+ medications
- **Providers:** 1,200+ across 10 specialties
- **Facilities:** 200+ healthcare sites

### Quality Metrics
- **Completeness:** Core fields 100% populated
- **Consistency:** Logical clinical relationships maintained
- **Accuracy:** Medically plausible values and patterns
- **Timeliness:** Temporal sequences preserved

## Compliance & Ethics

- **100% Synthetic:** No real patient information
- **HIPAA/PIPEDA Aligned:** Suitable for privacy regulation testing
- **Clinical Validity:** Medically reviewed patterns and relationships
- **Ethical AI:** Supports responsible healthcare AI development
- **Bias Testing:** Demographic diversity for fairness evaluation

## Known Limitations

1. **Simplified Clinical Logic:** Real medicine is more complex
2. **Limited Specialties:** Focus on primary care encounters
3. **Static Conditions:** Chronic diseases don't progress
4. **Perfect Documentation:** No missing or incorrect entries
5. **Geographic Simplification:** Urban bias in facility distribution

---
*Generated for PAMOLA Epic 2 Testing Suite - Healthcare Analytics Module*  
*Version 1.0 - August 2025*
