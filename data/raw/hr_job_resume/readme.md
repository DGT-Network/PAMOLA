# PAMOLA_HH_WESTERN_RESUMES_V2 – simulated only
## 1. Dataset Overview

* **Dataset Name**: PAMOLA_HH_WESTERN_RESUMES_V2
* **File**: `PAMOLA_HH_WESTERN_RESUMES_V2_10k.csv`
* **Type**: Synthetic (surrogate) resumes/profiles
* **Scope**: Canada-focused, IT-heavy + Finance/Marketing
* **Format**: CSV (UTF-8, comma, quoted strings)
* **Records**: 10,000
* **Header**: First row contains field names

## 2. Privacy Categories

* **Direct identifiers (D)**: address_line1, email, entity_id, fake_sin, first_name, github_url, last_name, linkedin_url, personal_website, phone, postal_code, resume_id, twitter_handle
* **Quasi-identifiers (Q)**: company_current, company_size_bucket, degree_major, education_level, employer_industry, geo_lat, geo_lng, graduation_year, job_level, job_title_current, languages, location_city, location_country, location_province, postal_code, relocation_preference, remote_preference, seniority_level, university_name, work_authorization
* **Sensitive (S)**: base_salary_cad, bio_note, bonus_cad, currency, current_salary_cad, equity_cad, experience_summary, hourly_rate, pay_frequency, recruiter_notes, summary, total_comp_cad
* **Indirect/utility (I)**: created_at, data_source, dup_group_size, has_pii_in_text, income_bracket, is_management, is_outlier_salary, member_of_training, pii_email_count, pii_phone_count, preferences_json, role_end_date, role_start_date, row_version, salary_period, skills, snapshot_date, specialization, split, updated_at, years_of_experience

## 3. Field Specifications

| Field | Type | Category | Missing % | Unique Ratio | Description | Anonymization |
| --- | --- | --- | --- | --- | --- | --- |
| `entity_id` | string | Direct | 0.00% | 0.45 | Stable person identifier (duplicates exist across snapshots). | hash(SHA-256 + project salt) — consistent pseudonymization |
| `resume_id` | string | Direct | 0.00% | 1.00 | Snapshot/record id (unique per row). | hash(SHA-256 + project salt) — consistent pseudonymization |
| `snapshot_date` | date | Indirect | 0.00% | 0.16 | Snapshot date for row versioning. | coarsen to month (YYYY-MM) or quarter |
| `row_version` | number | Indirect | 0.00% | 0.00 | Monotonic per entity. | keep |
| `created_at` | date | Indirect | 0.00% | 0.03 | Approximate creation date. | coarsen to month (YYYY-MM) or quarter |
| `updated_at` | date | Indirect | 0.00% | 0.00 | Last update date. | coarsen to month (YYYY-MM) or quarter |
| `data_source` | string | Indirect | 0.00% | 0.00 | 'generator_v2'. | keep |
| `first_name` | string | Direct | 0.00% | 0.00 | Given name. | drop or replace with synthetic names from fixed dictionary |
| `last_name` | string | Direct | 0.00% | 0.00 | Family name. | drop or replace with synthetic names from fixed dictionary |
| `email` | string | Direct | 12.09% | 0.42 | Email; ~12% missing. | mask username; keep domain group only (e.g., 'gmail', 'outlook') |
| `phone` | string | Direct | 22.56% | 0.53 | Phone in various formats; ~22% missing. | mask middle & last digits; keep country/area code |
| `address_line1` | string | Direct | 0.00% | 0.45 | Street address. | drop |
| `postal_code` | string | Direct | 0.00% | 0.45 | Canadian postal code A1A 1A1 format. | generalize to FSA (first 3 chars) |
| `linkedin_url` | string | Direct | 0.00% | 0.43 | LinkedIn profile URL. | drop path/handle; keep provider only |
| `github_url` | string | Direct | 24.68% | 0.05 | GitHub URL (IT mostly). | drop path/handle; keep provider only |
| `personal_website` | string | Direct | 86.26% | 0.25 | Optional website. | drop path/handle; keep provider only |
| `twitter_handle` | string | Direct | 88.23% | 0.71 | Optional handle. | drop path/handle; keep provider only |
| `location_city` | string | Quasi | 0.00% | 0.00 | City (incl. odd values). | hierarchical generalization City→Province→Country |
| `location_province` | string | Quasi | 0.00% | 0.00 | Province/territory code. | hierarchical generalization City→Province→Country |
| `location_country` | string | Quasi | 0.00% | 0.00 | Country (Canada). | hierarchical generalization City→Province→Country |
| `geo_lat` | number | Quasi | 28.89% | 0.59 | Latitude (jittered); NaN for unknowns. | coarsen (2 decimals) or map to city centroid; remove where city is nonstandard |
| `geo_lng` | number | Quasi | 28.89% | 0.59 | Longitude (jittered); NaN for unknowns. | coarsen (2 decimals) or map to city centroid; remove where city is nonstandard |
| `job_title_current` | string | Quasi | 0.00% | 0.00 | Current/most recent title. | map to normalized job families (e.g., Software Engineer→Engineer/IT) |
| `company_current` | string | Quasi | 0.00% | 0.10 | Employer name (fictional). | pseudonymize to 'Company A/B/…' or generalize to employer_industry |
| `years_of_experience` | number | Indirect | 0.00% | 0.00 | Total YOE. | keep |
| `work_authorization` | string | Quasi | 4.54% | 0.00 | Citizen/PR/Work Permit/Student Visa. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `remote_preference` | string | Quasi | 0.00% | 0.00 | Remote/Hybrid/Onsite. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `relocation_preference` | string | Quasi | 0.00% | 0.00 | Yes/No/Open to discuss. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `education_level` | string | Quasi | 0.00% | 0.00 | Highest level. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `degree_major` | string | Quasi | 14.63% | 0.00 | Major/field of study. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `university_name` | string | Quasi | 12.43% | 0.00 | Institution. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `graduation_year` | string | Quasi | 7.72% | 0.00 | Graduation year. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `languages` | string | Quasi | 0.00% | 0.02 | 'English; French; ...' | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `company_size_bucket` | string | Quasi | 0.00% | 0.00 | Company size bucket. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `employer_industry` | string | Quasi | 0.00% | 0.00 | Industry label. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `seniority_level` | string | Quasi | 0.00% | 0.00 | Junior/Mid/Senior/Lead/Manager/Director. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `job_level` | string | Quasi | 0.00% | 0.00 | IC/Manager/Executive. | keep; or generalize (group rare categories, bin years to 5-year buckets) |
| `current_salary_cad` | number | Sensitive | 7.90% | 0.84 | Annual salary (CAD). | add Laplace noise (ε≈0.2) or bucket to ranges; preserve order statistics |
| `salary_period` | string | Indirect | 0.00% | 0.00 | 'annual'. | keep |
| `base_salary_cad` | number | Sensitive | 0.00% | 0.83 | Base salary. | add Laplace noise (ε≈0.2) or bucket to ranges; preserve order statistics |
| `bonus_cad` | number | Sensitive | 0.00% | 0.77 | Bonus. | add Laplace noise (ε≈0.2) or bucket to ranges; preserve order statistics |
| `equity_cad` | number | Sensitive | 0.00% | 0.66 | Equity. | add Laplace noise (ε≈0.2) or bucket to ranges; preserve order statistics |
| `total_comp_cad` | number | Sensitive | 0.00% | 0.84 | Comp total. | add Laplace noise (ε≈0.2) or bucket to ranges; preserve order statistics |
| `currency` | string | Sensitive | 0.00% | 0.00 | Currency (CAD/USD). | keep categories; bucket hourly_rate |
| `pay_frequency` | string | Sensitive | 0.00% | 0.00 | annual/hourly. | keep categories; bucket hourly_rate |
| `hourly_rate` | number | Sensitive | 90.45% | 0.46 | If hourly. | keep categories; bucket hourly_rate |
| `role_start_date` | date | Indirect | 0.00% | 0.20 | Role start. | coarsen to month (YYYY-MM) or quarter |
| `role_end_date` | date | Indirect | 49.48% | 0.24 | Role end (nullable). | coarsen to month (YYYY-MM) or quarter |
| `skills` | string | Indirect | 0.00% | 0.63 | Skill list. | regex + NER redaction: EMAIL, PHONE, URL, PERSON, ORG, GPE → placeholders; optional term-preserving masking for tech keywords |
| `summary` | string | Sensitive | 2.95% | 0.42 | 100–420 chars, tech/company mentions. | regex + NER redaction: EMAIL, PHONE, URL, PERSON, ORG, GPE → placeholders; optional term-preserving masking for tech keywords |
| `experience_summary` | string | Sensitive | 3.17% | 0.45 | 180–600 chars, tech/process/regulatory. | regex + NER redaction: EMAIL, PHONE, URL, PERSON, ORG, GPE → placeholders; optional term-preserving masking for tech keywords |
| `bio_note` | string | Sensitive | 0.00% | 0.96 | 100–600 chars with possible disguised PII. | regex + NER redaction: EMAIL, PHONE, URL, PERSON, ORG, GPE → placeholders; optional term-preserving masking for tech keywords |
| `recruiter_notes` | string | Sensitive | 0.00% | 1.00 | 100–600 chars with possible disguised PII. | regex + NER redaction: EMAIL, PHONE, URL, PERSON, ORG, GPE → placeholders; optional term-preserving masking for tech keywords |
| `has_pii_in_text` | boolean | Indirect | 0.00% | 0.00 | True if PII injected. | keep |
| `pii_email_count` | number | Indirect | 0.00% | 0.00 | Ground-truth email count inside bio/recruiter. | keep |
| `pii_phone_count` | number | Indirect | 0.00% | 0.00 | Ground-truth phone count inside bio/recruiter. | keep |
| `specialization` | string | Indirect | 0.00% | 0.00 |     | keep |
| `preferences_json` | string | Indirect | 0.00% | 0.28 | Nested settings. | keep |
| `fake_sin` | string | Direct | 98.97% | 0.42 | Synthetic Canadian SIN (≈1%). | drop (synthetic but high-risk pattern) |
| `split` | string | Indirect | 0.00% | 0.00 | train/val/test. | keep |
| `member_of_training` | boolean | Indirect | 0.00% | 0.00 | Ground-truth for MIA tests. | keep |
| `is_management` | boolean | Indirect | 0.00% | 0.00 | Manager/Director etc. | keep |
| `income_bracket` | string | Indirect | 0.00% | 0.00 | L/M/H from salary. | keep |
| `is_outlier_salary` | boolean | Indirect | 0.00% | 0.00 | Outlier flag. | keep |
| `dup_group_size` | number | Indirect | 0.00% | 0.00 | Snapshots per entity. | keep |

## 4. Long Text & PII Notes

* `summary`, `experience_summary`, `bio_note`, `recruiter_notes` contain natural text (100–600 chars).
* PII may appear in `bio_note`/`recruiter_notes`; use regex + NER redaction (EMAIL, PHONE, URL, PERSON, ORG, GPE).
* Ground-truth helpers: `has_pii_in_text`, `pii_email_count`, `pii_phone_count`.

## 5. Anonymization Strategy (Pipeline)

1. **Data Cleaning**: fix weird cities, trim whitespace, normalize encodings.
2. **Direct IDs**: hash (`entity_id`, `resume_id`), drop names & address, mask `email`/`phone`, keep postal FSA only.
3. **Quasi IDs**: City→Province generalization; coarsen lat/lng; titles→families; companies→industry or pseudonyms.
4. **Sensitive**: add Laplace noise/bucket salaries; redact PII in long texts; bucket `hourly_rate`; coarsen dates to month.
5. **Validation**: k-anonymity (target k≥5), distribution tests (KS/χ²), utility checks for ML tasks.

## 6. Distribution Highlights

* Rows with PII in text: **75.97%**
* USD currency: **4.26%**; hourly pay: **9.55%**
* Weird/edge-case cities: **5.54%**
* Salary outliers: **2.06%**
* Duplicate snapshot group size (per entity) – min/median/mean/max: 1 / 3 / 2.82 / 4

## 7. Known Edge Cases

* ≈2% nonstandard cities/provinces; lat/lng set to NaN.
* ~0.1% CSV injection patterns in `first_name` (e.g., `=SUM(1,2)`).
* Mixed EN/FR fragments in long text; emojis/punctuation variety possible.

## 8. Suggested QA Checks

1. Verify no duplicate `resume_id` after hashing; ensure `entity_id` linkage preserved.
2. Validate generalization: postal → FSA only; lat/lng rounding; dates to month.
3. Redaction recall/precision using `pii_*` counters as partial ground truth.
4. KS/χ² tests to ensure distribution preservation for compensation & geography.

_Generated 2025-08-11T23:40:03.215326Z_
