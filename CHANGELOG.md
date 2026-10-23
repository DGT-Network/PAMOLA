# Changelog

All notable changes to `pamola-core` will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Removed

- **`pamola_core.pipeline`, `pamola_core.interface`, `pamola_core.crypto`** —
  empty placeholder packages (15 lines each: a licence header and
  `__all__ = []`) that shipped in the wheel and created three public names with
  nothing behind them. Zero references anywhere in the codebase.
- **`pamola_core.attacks` narrowed to the anonymization-facing attacks.**
  The package now carries `LinkageAttack` (including the CVPL variant) and
  `AttributeInference` only.
  - `MembershipInference` removed — membership inference is a model-privacy
    question, not an anonymization one.
  - `DistanceToClosestRecord` and `NearestNeighborDistanceRatio` removed from
    `attacks/`: they duplicated `pamola_core.metrics.privacy.distance` and
    `pamola_core.metrics.privacy.neighbor`, which are the live
    implementations — `PrivacyMetricOperation` imports the metrics versions and
    never imported these. **Both metrics remain fully available from
    `pamola_core.metrics.privacy`; nothing is lost.**
  - `attacks/attacks_test.py` removed — a 641-line smoke-test harness that
    shipped to PyPI, was never collected by pytest, and duplicated the 98 real
    tests in `tests/attacks/`.

  **Migration:** import DCR and NNDR from `pamola_core.metrics.privacy` instead
  of `pamola_core.attacks`.

  Note: **singling-out is not implemented** in this package. The docstring of
  `pamola_core/attacks/__init__.py` states this explicitly so that no report
  generated from it claims coverage of that criterion.

### Added

- **`tests/conftest.py`** — an autouse fixture that clears ambient `PAMOLA_*`
  environment variables, making the suite hermetic. Previously a developer
  machine with `PAMOLA_PROJECT_ROOT` set failed five tests in
  `tests/utils/tasks/` while CI stayed green.
- **`tests/test_public_api_coverage_scope.py`** — asserts every name in
  `pamola_core.__all__` resolves to a module inside the `.coveragerc` scope.
  On its first run it caught `HashBasedPseudonymizationOperation` and
  `ConsistentMappingPseudonymizationOperation` being exported but unmeasured.

### Changed

- **BREAKING: an existing ENRICH output column is no longer overwritten
  silently** (DATA-04, SRS 6.8). In ENRICH mode every operation used to log a
  WARNING and overwrite a column that already existed. It now fails with the new
  `OutputFieldExistsError` (a `ValidationError`, naming the column only, never
  values) before any work is done. **Typical break:** running the same ENRICH
  operation twice over a frame that already holds its output (same
  `column_prefix`/`output_field_name`), or an `output_field_name` equal to an
  existing column. **Migration:** choose another `output_field_name` /
  `column_prefix`, or pass the new `overwrite_existing_output=True` (config key of
  the same name, default `False`) to replace the column deliberately. The check
  sits in the anonymization, transformation and fake-data bases
  (`BaseOperation._check_enrich_target`), so it applies to every operation built
  on them.
- **Quality gates now block instead of merely reporting.**
  `[tool.ruff.lint]` in `pyproject.toml` defines a narrow blocking set
  (`E9` + `F`); `.coveragerc` `fail_under` moves from `0` to `83` against a
  measured 84.327 %. CI's ruff step loses its `|| true`; a non-blocking mypy
  job is added.

- **Visualization is opt-in: `generate_visualization` now defaults to `False`**
  (VIZ-01, VIZ-02). One default, in `BaseOperation`; the core config schema
  agrees. A default run no longer starts the render thread, writes plot files or
  emits any visualization log line. The Section 15 baseline measured that thread
  at 28-87 % of wall time across the operations tested.
  **Migration:** pass `generate_visualization=True` (or set it in the operation
  config) wherever charts were relied on.
- **Explicit `generate_visualization=True` without the `viz` extra returns one
  `ERROR` result** with an actionable message (`pip install 'pamola-core[viz]'`)
  before any data is touched (VIZ-04). Previously the operation ran, logged about
  twelve WARNING/ERROR lines with tracebacks from every layer of the plotting
  stack, and still reported `SUCCESS` without charts. The check lives in
  `BaseOperation` (`check_visualization_dependencies`, applied to every
  subclass's `execute`), so no operation needs its own.

### Fixed

- **Output-encryption keys no longer reach the cache metadata** (CDX-AC-8).
  `_get_base_parameters()` carried the raw `encryption_key`, and every cache
  writer stored those parameters in an unencrypted JSON file. The key now
  influences the cache identity only through `encryption_key_id`, a truncated
  BLAKE2b digest. Persisted parameters are redacted centrally
  (`redact_sensitive_parameters`, applied in `build_base_cache` and again in
  `OperationCache.save_cache`) for secret-named keys (`*_key`, `password`,
  `token`, `secret`, `salt`, `pepper`...), and `_log_operation_start` uses the same
  rule plus the config's declared `SENSITIVE_KEYS`. Existing cache entries are not
  reused. Any entry written by an earlier version may hold a key in plaintext:
  delete old `cache/` directories.
- **The cache fingerprint now covers all of the data** (CDX-AC-6).
  `generate_data_hash` hashed 5,000 sampled rows plus column sums and fell back to
  "type + length" on any error, so changing unsampled rows or reversing index
  labels kept the cache key and a stale result was served. It now hashes the
  schema, the index and every value, streaming in 200K-row chunks (1M x 15
  columns, 353 MB: 1.7 s, no measurable memory growth). The fallback is removed:
  if hashing fails, the caller's cache lookup and save are skipped for that call
  (they already catch the error). The `sample_size` argument is kept but ignored.
  Affects every operation's cache key: existing cache entries are not reused, and
  cache keys cost a full pass over the data.
- **Duplicate column names raise a typed, actionable error** (DATA-03). Every
  tested operation failed on a frame with a repeated column name with an
  unrelated pandas message (`'DataFrame' object has no attribute 'dtype'`,
  "truth value of a DataFrame is ambiguous"). `load_data_operation` - the loader
  every operation uses - now raises the new `DuplicateColumnError` (a
  `ValidationError`) naming the repeated columns and saying to rename or drop
  them. Frames with unique names are unaffected.
- **Three plotting modules now go through `require_optional()`** (VIZ-04):
  `privacy_models/k_anonymity/ka_visualization.py` and
  `privacy_models/l_diversity/{visualization,advanced_visualization}.py` imported
  matplotlib/seaborn directly, so a base install got a bare
  `ModuleNotFoundError`. They now raise `DependencyMissingError` naming
  `pip install 'pamola-core[viz]'`. `tests/test_viz_import_boundary.py` covers
  them, and asserts that importing `pamola_core.privacy_models` pulls in no
  plotting backend.
- **`UtilityMetricOperation` has one vocabulary and fails early on a bad
  request** (UTIL-01, UTIL-02). The families `classification` / `regression`
  (schema = factory) go in `utility_metrics`; the target column is the required
  `metric_params[family]["value_field"]`; model and metric names are declared by
  the implementations (`SUPPORTED_MODELS` / `SUPPORTED_METRICS`). A missing or
  unknown target column, or an unsupported model/metric name (the tooltip showed
  `"lr"` and `target`, which never worked), is now rejected with the supported
  names before any model is trained, instead of a `KeyError` mid-run. An empty
  `utility_metrics` says what to pass. The `UtilityMetricsType` enum is unchanged
  and documented as the measures reported inside a family. Docs and the
  tooltip show a call that runs.
- **DCR no longer builds the full n x n distance matrix** (DCR-01, DCR-02,
  DCR-03). `DistanceToClosestRecord` defaulted to the dense sklearn path:
  ~80 GB at 100K x 100K (49 GB RSS observed, or `MemoryError`), ~7 TiB at 1M.
  - The exact search now runs in row blocks (`pairwise_distances_chunked`): the
    values are those of the dense matrix, the peak memory is a fixed budget
    (625 MiB RSS at 100K x 100K, 10 features).
  - `use_faiss` is now `None` by default (automatic). Above 10K x 10K record pairs
    FAISS is chosen **only** where it returns the same values as the exact search
    (euclidean/cosine `min`, cosine `mean_k`; equal to float32 rounding, ~2e-6
    relative). 100K x 100K: ~6.5 s, 370 MiB RSS. `use_faiss=True/False` still
    force a backend.
  - New `max_comparisons` guard: input beyond 2e10 pairs for the exact search
    (1e12 for FAISS) raises a typed `ResourceError` with the sizes and what to do,
    before any distance is computed. `PrivacyMetricOperation` no longer wraps it
    in a generic `ValidationError`.
  - **Known, not changed:** an explicit `use_faiss=True` with euclidean `mean_k`
    (averages squared distances before the root) or with `percentile` (10th
    percentile of at most 10 neighbours, not of all distances) returns different
    numbers from the exact search. Those combinations are never chosen
    automatically; aligning them is a metric-definition decision.
- **A cached metric no longer returns the score of a different transformed
  dataset** (METR-CACHE-01). With `use_cache=True`, `MetricsOperation`
  (fidelity, utility, privacy including DCR) checked the cache before loading the
  transformed dataset and keyed it on the original DataFrame alone, so the same
  original and configuration with new transformed data returned the previous
  result. The key now covers both DataFrames, the dataset names, `columns`,
  `column_mapping`, `sample_size`, `seed` and the operation config. The change is
  confined to `metrics/base_metrics_op.py`; `BaseOperation._generate_cache_key` is
  unchanged. Existing metric cache entries are not reused (the key differs).
- **Cell and record suppression on a 0-row DataFrame** (DATA-02).
  `CellSuppressionOperation` and `RecordSuppressionOperation` returned `ERROR`
  with a raw `No objects to concatenate` from `pd.concat([])` - a normal shape
  after an upstream filter. They now return `SUCCESS` with a header-only output
  that keeps the input's columns (plus the ENRICH column), zero counts in the
  metrics, and one WARNING saying there was nothing to suppress. This matches
  `AttributeSuppressionOperation` and the profilers, which already succeeded on
  empty input.
- **Operations no longer modify the caller's DataFrame** (DATA-01).
  `NumericGeneralizationOperation`, `UniformNumericNoiseOperation` and
  `AddOrModifyFieldsOperation` assigned columns on the object they were given
  (`DataSource` stores and returns the caller's frame itself): the caller's dtypes
  changed, and `AddOrModifyFields` added columns and rewrote values in it. The
  fix is one contract rather than three patches: `_validate_and_get_dataframe` of
  the anonymization, transformation and fake-data bases now returns a private
  deep copy. Cost: one transient copy of the frame per `execute()` (about 0.25 s
  and the frame's size again for a 1M-row, 13-column, 254 MB frame); Dask frames
  are lazy and are not copied. There is no in-place mode.
- **Ambiguous date formats were silently resolved to US convention.**
  In `common/regex/patterns.py` the key `\d{2}/\d{2}/\d{4}` was declared twice —
  once mapping to `%d/%m/%Y` and once to `%m/%d/%Y` — and a dict literal keeps
  the last value, so the DMY entries were dead and every ambiguous date parsed
  as MDY. Six such keys. The duplicates are removed rather than reordered,
  which preserves behaviour exactly while making the resolution explicit.
- Three exact duplicate dictionary keys removed (`full_masking_op.py`,
  `profiling/analyzers/anonymity.py`, `profiling/commons/phone_utils.py`).

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