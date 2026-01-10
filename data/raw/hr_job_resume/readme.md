# Synthetic HR Profiles and Resumes Dataset

> **⚠️ Synthetic Data Notice**
>
> This dataset is **fully synthetic** - programmatically generated for testing purposes.
> **No real personal information (PII) is included.**
>
> - Not collected from any external sources, job boards, or HR systems
> - Not derived from any real resumes, CVs, or employee records
> - All names, emails, phone numbers, addresses, and SINs are artificially generated
> - Any resemblance to actual persons or organizations is coincidental
>
> *This documentation is for technical reference, not legal advice.*

---

## Overview

Synthetic HR profiles and resume dataset simulating Canadian IT/Finance/Marketing professionals. Designed for testing PII detection, text anonymization, NER redaction, and privacy-preserving HR analytics. Contains long-form text fields with intentionally embedded PII patterns for detection testing.

**Domain:** Canadian HR/Recruitment  
**Records:** 10,000 synthetic profiles  
**Key Feature:** Ground-truth PII flags for detection validation  
**Primary Use Cases:** PII detection, text anonymization, NER testing, privacy analytics

---

## Data Generation

**Method:** Rule-based synthetic generation with controlled PII injection

| Component | Approach |
|-----------|----------|
| Names | Faker library with Canadian/multicultural name distributions |
| Contact info | Synthetic emails, phones, addresses (Canadian formats) |
| Locations | Canadian cities/provinces with jittered coordinates |
| Salaries | Log-normal distributions matching Canadian IT/Finance market |
| Text fields | Template-based generation with intentional PII injection |
| SINs | Synthetic Canadian SIN format (fake_sin ~1% populated) |

**PII Injection:** 75.97% of records have PII embedded in text fields for detection testing.

**Reproducibility:** Passport JSON contains generation parameters.

---

## Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `HR_RESUMES_CA_V2_10k.csv` | 10,000 | Main synthetic profiles dataset |
| `HR_RESUMES_SAMPLE_CA.csv` | ~100 | Small sample for quick testing |
| `HR_RESUMES_SAMPLE_CA_10k.csv` | 10,000 | Full sample variant |
| `HR_JOB_SAMPLE.csv` | ~500 | Synthetic job postings |
| `HR_RESUMES_DICTIONARY_CA.csv` | - | Field definitions and metadata |
| `HR_RESUMES_CA_V2_passport.json` | - | Generation parameters and statistics |

---

## Privacy Classification

### Summary by Category

| Category | Count | Fields |
|----------|-------|--------|
| **Direct Identifiers (D)** | 13 | entity_id, resume_id, first_name, last_name, email, phone, address_line1, postal_code, linkedin_url, github_url, personal_website, twitter_handle, fake_sin |
| **Quasi-Identifiers (Q)** | 20 | location_city, location_province, location_country, geo_lat, geo_lng, job_title_current, company_current, education_level, degree_major, university_name, graduation_year, languages, company_size_bucket, employer_industry, seniority_level, job_level, work_authorization, remote_preference, relocation_preference |
| **Sensitive (S)** | 12 | base_salary_cad, bonus_cad, equity_cad, total_comp_cad, current_salary_cad, hourly_rate, currency, pay_frequency, summary, experience_summary, bio_note, recruiter_notes |
| **Indirect/Utility (I)** | 20 | created_at, updated_at, snapshot_date, row_version, data_source, skills, split, member_of_training, is_management, income_bracket, is_outlier_salary, dup_group_size, has_pii_in_text, pii_email_count, pii_phone_count, preferences_json, role_start_date, role_end_date, specialization, years_of_experience |

---

## Field Specifications

### Direct Identifiers

| Field | Type | Missing % | Unique Ratio | Anonymization |
|-------|------|-----------|--------------|---------------|
| `entity_id` | string | 0% | 0.45 | Hash (SHA-256 + salt) |
| `resume_id` | string | 0% | 1.00 | Hash (SHA-256 + salt) |
| `first_name` | string | 0% | low | Drop or replace with synthetic |
| `last_name` | string | 0% | low | Drop or replace with synthetic |
| `email` | string | 12% | 0.42 | Mask username, keep domain type |
| `phone` | string | 23% | 0.53 | Mask digits, keep area code |
| `address_line1` | string | 0% | 0.45 | Drop |
| `postal_code` | string | 0% | 0.45 | Generalize to FSA (first 3 chars) |
| `linkedin_url` | string | 0% | 0.43 | Drop path, keep provider only |
| `github_url` | string | 25% | 0.05 | Drop path, keep provider only |
| `personal_website` | string | 86% | 0.25 | Drop |
| `twitter_handle` | string | 88% | 0.71 | Drop |
| `fake_sin` | string | 99% | 0.42 | Drop (synthetic but high-risk pattern) |

### Quasi-Identifiers

| Field | Type | Missing % | Anonymization |
|-------|------|-----------|---------------|
| `location_city` | string | 0% | Hierarchical: City -> Province -> Country |
| `location_province` | string | 0% | Group to regions (East/Central/West) |
| `location_country` | string | 0% | Keep (all Canada) |
| `geo_lat` | number | 29% | Coarsen to 2 decimals or city centroid |
| `geo_lng` | number | 29% | Coarsen to 2 decimals or city centroid |
| `job_title_current` | string | 0% | Map to job families |
| `company_current` | string | 0% | Pseudonymize or generalize to industry |
| `education_level` | string | 0% | Keep |
| `degree_major` | string | 15% | Keep or group rare majors |
| `university_name` | string | 12% | Keep or generalize to tier |
| `graduation_year` | string | 8% | 5-year bins |
| `languages` | string | 0% | Keep |
| `company_size_bucket` | string | 0% | Keep |
| `employer_industry` | string | 0% | Keep |
| `seniority_level` | string | 0% | Keep |
| `job_level` | string | 0% | Keep |
| `work_authorization` | string | 5% | Keep or group rare categories |
| `remote_preference` | string | 0% | Keep |
| `relocation_preference` | string | 0% | Keep |

### Sensitive Attributes (Compensation)

| Field | Type | Missing % | Anonymization |
|-------|------|-----------|---------------|
| `base_salary_cad` | number | 0% | Laplace noise (epsilon ~0.2) or bucket |
| `bonus_cad` | number | 0% | Laplace noise or bucket |
| `equity_cad` | number | 0% | Laplace noise or bucket |
| `total_comp_cad` | number | 0% | Laplace noise or bucket |
| `current_salary_cad` | number | 8% | Laplace noise or bucket |
| `hourly_rate` | number | 90% | Bucket to ranges |
| `currency` | string | 0% | Keep (CAD/USD) |
| `pay_frequency` | string | 0% | Keep (annual/hourly) |

### Sensitive Attributes (Text Fields)

| Field | Type | Length | Contains PII | Anonymization |
|-------|------|--------|--------------|---------------|
| `summary` | string | 100-420 chars | Possible | NER redaction |
| `experience_summary` | string | 180-600 chars | Possible | NER redaction |
| `bio_note` | string | 100-600 chars | **Yes (injected)** | NER redaction |
| `recruiter_notes` | string | 100-600 chars | **Yes (injected)** | NER redaction |

### Ground-Truth PII Flags

| Field | Type | Description |
|-------|------|-------------|
| `has_pii_in_text` | boolean | True if PII injected in bio/recruiter notes |
| `pii_email_count` | integer | Count of emails embedded in text |
| `pii_phone_count` | integer | Count of phones embedded in text |

These fields enable precision/recall measurement for PII detection systems.

---

## Long Text & PII Detection Testing

### PII Injection Patterns

The `bio_note` and `recruiter_notes` fields contain intentionally embedded PII:

| PII Type | Example Pattern | Detection Target |
|----------|-----------------|------------------|
| Email | "contact me at john.doe@email.com" | EMAIL entity |
| Phone | "reach me at 416-555-1234" | PHONE entity |
| URL | "portfolio at example.com/portfolio" | URL pattern |
| Name | "worked with Sarah Johnson" | PERSON entity |
| Organization | "previously at Acme Corp" | ORG entity |
| Location | "based in downtown Toronto" | GPE entity |

### Recommended Redaction Pipeline

```python
# Step 1: Regex patterns
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
url_pattern = r'https?://[^\s]+|www\.[^\s]+'

# Step 2: NER (spaCy or similar)
ner_entities = ['PERSON', 'ORG', 'GPE', 'EMAIL', 'PHONE']

# Step 3: Replace with placeholders
redacted = "[EMAIL]", "[PHONE]", "[PERSON]", "[ORG]", "[LOCATION]"
```

### Validation Using Ground Truth

```python
# Measure detection recall
detected_emails = detect_emails(text)
actual_emails = row['pii_email_count']
recall = len(detected_emails) / actual_emails if actual_emails > 0 else 1.0
```

---

## Distribution Highlights

| Metric | Value |
|--------|-------|
| Records with PII in text | 75.97% |
| USD currency transactions | 4.26% |
| Hourly pay frequency | 9.55% |
| Nonstandard/edge-case cities | 5.54% |
| Salary outliers | 2.06% |
| Duplicate snapshots per entity | 1-4 (mean 2.82) |

### Geographic Distribution

| Province | Percentage |
|----------|------------|
| Ontario | ~45% |
| British Columbia | ~20% |
| Quebec | ~15% |
| Alberta | ~10% |
| Other | ~10% |

### Industry Distribution

| Industry | Percentage |
|----------|------------|
| Technology/IT | ~50% |
| Finance | ~25% |
| Marketing | ~15% |
| Other | ~10% |

---

## Known Edge Cases

| Issue | Prevalence | Description |
|-------|------------|-------------|
| Nonstandard cities | ~2% | Invalid city names; lat/lng set to NaN |
| CSV injection | ~0.1% | Patterns like `=SUM(1,2)` in first_name |
| Mixed EN/FR text | Variable | Bilingual fragments in summaries |
| Emoji/special chars | Variable | In bio_note and recruiter_notes |
| Duplicate entities | ~45% | Same entity_id with multiple snapshots |

---

## Anonymization Pipeline

### Recommended Sequence

```
1. Data Cleaning
   ├── Fix nonstandard cities (map to 'Unknown' or drop)
   ├── Sanitize CSV injection patterns
   └── Normalize encodings

2. Direct Identifier Processing
   ├── Hash: entity_id, resume_id (SHA-256 + salt)
   ├── Drop: first_name, last_name, address_line1, fake_sin
   ├── Mask: email (keep domain type), phone (keep area code)
   └── Drop paths: linkedin_url, github_url, personal_website, twitter_handle

3. Quasi-Identifier Generalization
   ├── location_city -> Province or Region
   ├── geo_lat/lng -> 2 decimal places or city centroid
   ├── job_title_current -> Job family mapping
   ├── company_current -> Industry or pseudonym
   └── graduation_year -> 5-year bins

4. Sensitive Attribute Protection
   ├── Salary fields: Laplace noise (epsilon ~0.2) or buckets
   ├── Text fields: NER redaction for PII entities
   └── Dates: Coarsen to month (YYYY-MM)

5. Validation
   ├── k-anonymity check (target k >= 5)
   ├── PII detection recall using ground-truth flags
   ├── Distribution preservation (KS test for salaries)
   └── Utility preservation for ML tasks
```

---

## File Structure

```
hr_job_resume/
├── HR_RESUMES_CA_V2_10k.csv       # Main dataset (10K profiles)
├── HR_RESUMES_SAMPLE_CA.csv       # Small sample
├── HR_RESUMES_SAMPLE_CA_10k.csv   # Full sample variant
├── HR_JOB_SAMPLE.csv              # Job postings sample
├── HR_RESUMES_DICTIONARY_CA.csv   # Field definitions
├── HR_RESUMES_CA_V2_passport.json # Generation metadata
└── README.md                      # This documentation
```

---

## Usage Examples

### Load and Explore

```python
import pandas as pd

df = pd.read_csv("data/raw/hr_job_resume/HR_RESUMES_CA_V2_10k.csv")

# Check PII distribution
print(f"Records with PII: {df['has_pii_in_text'].mean():.1%}")
print(f"Total embedded emails: {df['pii_email_count'].sum()}")
print(f"Total embedded phones: {df['pii_phone_count'].sum()}")
```

### Test PII Detection

```python
from pamola_core.text import PIIDetector

detector = PIIDetector()

# Detect PII in text field
results = detector.scan(df['bio_note'])

# Compare with ground truth
precision, recall = detector.evaluate(
    predictions=results,
    ground_truth_emails=df['pii_email_count'],
    ground_truth_phones=df['pii_phone_count']
)
```

### Test Text Anonymization

```python
from pamola_core.text import TextRedactor

redactor = TextRedactor(
    entities=['EMAIL', 'PHONE', 'PERSON', 'ORG'],
    placeholder_format="[{entity}]"
)

df['bio_note_redacted'] = df['bio_note'].apply(redactor.redact)
```

---

## Known Limitations

1. **Template-based text** - Generated summaries may have recognizable patterns
2. **IT/Finance bias** - Overrepresentation of tech and finance roles
3. **Canadian focus** - Limited international representation
4. **Simplified career paths** - Linear progression without gaps
5. **No temporal dynamics** - Snapshot data without job change history

---

## License

Apache 2.0 - same as PAMOLA.CORE repository.

See [LICENSE](../../../LICENSE) for full terms.

---

**Maintainer:** [Realm Inveo Inc.](https://realmdata.io)  
**Repository:** [github.com/DGT-Network/PAMOLA](https://github.com/DGT-Network/PAMOLA)  
**Version:** 2.0.0 (Epic 2 Testing Suite)
