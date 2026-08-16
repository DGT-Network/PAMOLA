# Changelog

All notable changes to `pamola-core` will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

## [1.0.0.dev4] - 2026-08-16

> **Scope:** Packaging and installability only. **No library code changed** —
> `pamola_core/` is byte-identical to `1.0.0.dev3`. This release exists so that
> installing `pamola-core` produces a correct, honest, and far smaller
> environment.

### Added

- **Eight previously undeclared runtime dependencies.** Each is imported at
  **module scope** by `pamola_core` but was never listed in
  `[project.dependencies]`; every one of them arrived only as a *transitive*
  dependency of some other package. The build was therefore one upstream
  dependency-drop away from `import pamola_core` failing outright:

  | Added | Imported at |
  |---|---|
  | `python-dotenv` | `utils/env.py:7` |
  | `filelock` | `utils/tasks/context_manager.py:37` |
  | `joblib` | `anonymization/commons/processing_utils.py:23` (+18 more) |
  | `rich` | `cli/commands/list_ops.py:12` |
  | `tqdm` | `utils/progress.py:67` |
  | `packaging` | `utils/ops/op_registry.py:37` |
  | `jinja2` | `utils/reporting/template_engine.py:14` |
  | `regex` | `utils/nlp/minhash.py:30` |

  Two of these were supplied **exclusively** by packages removed below —
  `python-dotenv` by `uvicorn[standard]`, `filelock` by `torch` — so the
  removals and these additions must ship together. `requests`, `transformers`,
  `graphviz`, `pyage` and `git` are also imported but only lazily, inside
  guarded optional-capability paths, and remain deliberately undeclared.

### Removed

- **Eight unused runtime dependencies**, none of which are imported anywhere in
  `pamola_core/` or `tests/`: `torch`, `sdv`, `uvicorn[standard]`, `rstr`,
  `deepdiff`, `diff-match-patch`, `multidict`, `bcrypt`.
  `torch` occurred only as a keyword string in
  `resources/entities/skills_en.json` and as an *optional* capability probe
  (`DependencyManager.check_dependency('torch')`, written to work when torch is
  absent) — while costing every installation roughly 2.5 GB. `tensorflow` and
  `transformers` are probed the same way and were already correctly absent.

  **Migration:** if you relied on `pamola-core` pulling in `torch` or `sdv` as a
  side effect, add them to your own dependencies.

### Fixed

- **Package metadata.** `authors` / `maintainers` were placeholder values
  (`Author <author@example.com>`, `Maintainer <maintainer@example.com>`) and
  were published as such to PyPI for `dev1` … `dev3`. Now
  `REALM Inveo Inc. <info@realminveo.com>`.
- **Package URLs.** Added `[project.urls]` — Homepage, Repository, Changelog,
  Bug Tracker. The PyPI page previously had no link back to the project.
- **Discoverability.** Added `keywords` and `Development Status`,
  `Intended Audience`, and `Topic` classifiers.
- **Missing package data.** `utils/ops/templates/README.md`,
  `utils/ops/templates/config_example.jsonc`, and
  `utils/tasks/templates/README.md` were present in the repository but absent
  from the built wheel, so installed users got the operation skeleton without
  its documentation.
- **CI dependency install.** Both workflows ran
  `pip install -e ".[dev]" 2>/dev/null || pip install -e .`, but the `dev` extra
  was removed in `1.0.0.dev2`; the failure was silently swallowed and the
  project's own pinned test tooling was then overridden by an unpinned
  `pip install pytest pytest-cov`. Both now use `pip install -e ".[test]"`.

### Notes

- `pip install pamola-core` **without** `--pre` still does not install this
  library: a `0.0.1` name-reservation stub from 2026-02-23 is the only *final*
  release on PyPI, and pip prefers it. Use `pip install --pre pamola-core` or
  pin the version. Resolution of the stub is tracked separately.
- Relaxing the remaining `==` pins and splitting heavy dependencies (spacy,
  nltk, faiss, dask, the plotting stack) into optional extras is deferred to a
  later release; it needs coordination with downstream consumers.

## [1.0.0.dev3] - 2026-06-01

> **Scope:** This release is a **CORE pseudonymization & anonymization hardening** release. It does **not** introduce, modify, or imply any formal differential-privacy (DP-SGD) or DP-based synthetic-data-generation capability — those capabilities live in the separate BEST / SYNT packages and follow an independent roadmap.

### Security

- **CRITICAL:** Stop persisting AES-256-GCM mapping encryption key to disk via `save_config()`. Added `OperationConfig.SENSITIVE_KEYS` + `to_safe_dict()` infrastructure; `ConsistentMappingPseudonymizationConfig` declares `mapping_encryption_key` as sensitive so it is replaced with `*REDACTED*` in `{task_dir}/config.json`.
- **Hash-based pseudonymization:** Reject all-zero/empty default salt when `use_pepper=False` via `_is_weak_salt_value()` guard in `_validate_configuration()` — prevents deterministic pseudonyms across installs.
- **Hash-based pseudonymization:** Invalidate stale disk cache when `use_pepper=True` by including a per-run `_session_id` in the cache key — previous-run pseudonyms can no longer be served back.

### Added

- **Pseudonymization ops:** New `HashBasedPseudonymizationOperation` (SHA3-256/512 with salt+pepper) and `ConsistentMappingPseudonymizationOperation` (AES-256-GCM encrypted reversible mapping) refactored to PAMOLA 7-step lifecycle.
- **Schemas:** 4-file schema pattern (`*_core_schema.py`, `*_schema_exclude.py`, `*_tooltip.py`, `*_ui_schema.py`) for hash-based and mapping pseudonymization ops; same pattern applied to metrics fidelity/privacy/utility ops.
- **Examples:** 4 Jupyter notebooks under `examples/anonymization/pseudonymization/` (simple + advanced for each op).
- **Tests:** 41 tests for pseudonymization ops covering ENRICH/REPLACE modes, compound identifiers, pickle round-trip, reverse mapping, sequential/random_string pseudonyms, weak-salt rejection, pepper cache invalidation, `config.json` secret-leak prevention.
- **OperationConfig:** `SENSITIVE_KEYS: ClassVar[frozenset]` + `to_safe_dict()` helper for declaring secrets that must never reach disk.
- **Anonymization base:** `_is_pseudonymization` class-level marker for ops that need `*REDACTED*` null-handling placeholder.

### Changed

- **Metric naming:** Renamed `values_pseudonymized` (misleadingly counted rows) to `rows_processed`; added `unique_values_hashed`; fixed `pseudonymization_rate` from boolean (1.0/0.0) to true fraction `changed_non_null / total_non_null`.
- **Metrics schemas:** Reorganized `pamola_core/metrics/schemas/` — renamed `*_ops_config.py` → `*_op_core_schema.py`, split into 4-file pattern.
- **Profiling analyzers:** Standardized `if reporter:` / `if progress_tracker:` guards across all analyzers (anonymity, attribute, currency, date, email, identity, mvf, numeric, phone, text).
- **Correlation utils:** Local `log` variable replaces module-level `logger` mutation; added NaN-filter and zero-variance guards around scipy correlation calls.
- **Schema builder:** Reorganized `_build_all_op_configs()` by section with comments; loads 41 op configs.
- **Cache log:** `_check_cache` log/reporter messages use `self.operation_name` instead of hardcoded "generalization".

### Fixed

- **Hash-based pseudonymization:** `process_batch` now uses `list(self.additional_fields or [])` defensive guard — safe to call independently with reloaded configs.
- **Consistent mapping pseudonymization:** Same defensive guard plus normalized `additional_fields`/`quasi_identifiers` (None → `[]`) at constructor entry; removed redundant post-`setattr` reassignment.
- **Pseudonymization null-handling:** Replaced fragile `getattr(self, "algorithm", None)` heuristic with explicit `_is_pseudonymization` class marker for selecting the `*REDACTED*` placeholder.

## [1.0.0.dev2] - 2026-04-06

### Changed

- Mark attacks/ module as experimental in README and `__init__.py`
- Remove empty synthesis/ placeholder package (out of scope for CORE)
- Move pytest from production dependencies to `[project.optional-dependencies]`
- Fix PyPI classifier for BSD license

## [1.0.0.dev1] - 2026-03-30

First alpha release. Epic 3 (Core Library) in progress.

### Added

- **CI/CD:** GitHub Actions pipelines for lint (ruff), test (pytest), build (sdist+wheel), and PyPI release (#FR-EP3-CORE-002)
- **CI artifacts:** JUnit XML + Coverage HTML reports uploaded as artifacts on every CI run
- **CI matrix:** Conditional Python version matrix — 3.10/3.11/3.12 on main, 3.11 on develop
- **Tests:** 5,436 tests across 172 files reaching 85% Public API coverage
- **Coverage config:** `.coveragerc` defining 73-file Public API scope (18,896 statements)
- **CLI:** `pamola` CLI with 4 commands: `list-ops`, `run`, `schema`, `validate` via Typer (#96)
- **Sphinx docs:** API reference for all public modules with autosummary, napoleon, viewcode (#94)
- **Python compat:** Enforce Python 3.10-3.12 compatibility, remove 3.9 walrus operators (FR-EP3-CORE-003, FR-EP3-CORE-004)
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

- **CI:** Resolve 25 CI test failures, add missing `jsonschema` dependency
- **Tests:** Fix sys.modules pollution, keyword-only `run()` args, assertion mismatches across 40+ test files
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

[Unreleased]: https://github.com/DGT-Network/PAMOLA/compare/v1.0.0.dev3...HEAD
[1.0.0.dev3]: https://github.com/DGT-Network/PAMOLA/compare/v1.0.0.dev2...v1.0.0.dev3
[1.0.0.dev2]: https://github.com/DGT-Network/PAMOLA/compare/v1.0.0.dev1...v1.0.0.dev2
[1.0.0.dev1]: https://github.com/DGT-Network/PAMOLA/compare/v0.0.1...v1.0.0.dev1
[0.0.1]: https://github.com/DGT-Network/PAMOLA/releases/tag/v0.0.1