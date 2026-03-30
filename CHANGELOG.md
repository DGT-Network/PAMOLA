# Changelog

All notable changes to `pamola-core` will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

## [1.0.0a1] - 2026-03-30

First alpha release. Epic 3 (Core Library) in progress.

### Added

- **CLI:** `pamola` CLI with 4 commands: `list-ops`, `run`, `schema`, `validate` via Typer (#96)
- **Sphinx docs:** API reference for all public modules with autosummary, napoleon, viewcode (#94)
- **Python compat:** Enforce Python 3.10-3.12 compatibility, remove 3.9 walrus operators (FR-EP3-CORE-003, FR-EP3-CORE-004)
- **Tests:** 127 test files, 3,527 test cases covering all Tier 1+2 modules (100% pass)
- **Docs:** 221 documentation pages in `docs/en/core/` synced with source code
- **Privacy models:** k-anonymity, l-diversity, t-closeness, differential privacy processors
- **Attacks:** Linkage, membership inference, attribute inference, DCR, NNDR metrics
- **Analysis:** 5 public functions (dataset_summary, privacy_risk, descriptive_stats, distribution, correlation)
- **Error system:** `BasePamolaError` hierarchy, `auto_exception` decorator, `ErrorHandler`, 80+ error codes

### Changed

- **Public API:** Restructure exports in `pamola_core/__init__.py` — flat mega-export of ~100 symbols (#93)
- **Error handling:** Centralize all exceptions under `pamola_core.errors.exceptions` (#93)
- **API surface:** Remove `common` and `errors` from top-level public API re-exports
- **Docstrings:** Fix 239 files with hybrid section headers (Google+NumPy) for Sphinx compatibility (#94)
- **Dependencies:** Pin to `torch>=2.8`, `spacy>=3.8`, `dask>=2025.11` in pyproject.toml

### Fixed

- **Sphinx build:** Achieve 0-warning clean build — fix duplicate object warnings, RST formatting, docstring issues (#94)
- **Op cache:** Remove Python 3.9-incompatible walrus operators from `op_cache.py`
- **Cell suppression:** Fix missing `operation_name` attribute initialization order in `cell_op.py`
- **Currency profiling:** Fix infinite recursion in `_get_cache_parameters()` (call `super()` instead of `self`)

## [0.0.1] - 2025-10-27

Initial development release.

### Added

- Core anonymization operations (10 ops: masking, generalization, noise, pseudonymization, suppression)
- Profiling analyzers (14 analyzers: anonymity, attribute, categorical, correlation, currency, date, email, group, identity, mvf, numeric, phone, text)
- Transformation operations (8 ops: cleaning, field ops, grouping, imputation, merging, splitting)
- Metric operations (fidelity, privacy, utility metrics)
- Fake data generation (email, name, organization, phone)
- I/O readers (CSV, JSON, Excel, Parquet)
- BaseOperation / BaseTask framework
- NLP subsystem (tokenization, entity extraction, LLM integration)

[Unreleased]: https://github.com/DGT-Network/PAMOLA/compare/v1.0.0a1...HEAD
[1.0.0a1]: https://github.com/DGT-Network/PAMOLA/compare/v0.0.1...v1.0.0a1
[0.0.1]: https://github.com/DGT-Network/PAMOLA/releases/tag/v0.0.1
