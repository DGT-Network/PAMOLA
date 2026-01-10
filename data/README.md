# Synthetic Sample Data (No PII)

This directory contains **fully synthetic datasets** generated специально для демонстраций, тестирования и разработки PAMOLA.

## Key statement
- **No real personal data (PII/PHI) is included.**
- Records are **artificially generated** and do not correspond to real individuals.
- Any similarity to real persons is purely coincidental.

## Purpose
These datasets exist to:
- demonstrate ingestion/profiling/anonymization/synthesis workflows,
- support automated tests and examples,
- provide consistent sample inputs for the team.

## Structure
Each subfolder contains a synthetic dataset family, e.g.:
- `bank_*` — simulated banking / transactions / fraud scenarios
- `churn_*` — synthetic churn datasets
- `hr_job_resume` — synthetic HR profile / resume-like records (synthetic only)
- `med_*` — synthetic medical/EHR-style records (synthetic only)

## Policy (IMPORTANT)
- **Do not commit real customer/employee/patient data** into this repository.
- If you need to test with real data, use the approved secure storage and follow internal access procedures.

Questions: contact@realmdata.io
