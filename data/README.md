# Synthetic Sample Data (No PII)

This directory contains **fully synthetic datasets** generated specifically for PAMOLA demonstrations, testing, and development.

---

## Key Statement

- **No real personal data (PII/PHI) is included.**
- All records are **artificially generated** and do not correspond to real individuals.
- Any resemblance to actual persons is purely coincidental.

---

## Purpose

These datasets exist to:

- Demonstrate ingestion, profiling, anonymization, and synthesis workflows
- Support automated tests and provide working examples
- Provide consistent sample inputs for the development team

---

## Structure

Each subfolder contains a synthetic dataset family:

| Prefix | Description |
|--------|-------------|
| `bank_*` | Simulated banking transactions and fraud detection scenarios |
| `churn_*` | Synthetic customer churn datasets |
| `hr_job_resume` | Synthetic HR profiles and resume-like records |
| `med_*` | Synthetic medical and EHR-style records |

---

## Policy

> ⚠️ **Important:** Do not commit real customer, employee, or patient data to this repository.

If you need to test with real data, use the approved secure storage and follow internal data access procedures.

---

## Generation

Sample data is programmatically generated using the Python [Faker](https://faker.readthedocs.io/) library with custom domain-specific extensions.

---

**Questions:** info@realmdata.io
