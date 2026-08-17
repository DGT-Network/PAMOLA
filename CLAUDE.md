# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Full agent contract: `docs/prompts/20260816_SYSTEM_CC_PAMOLA_CORE.md` — read it before a working session.
Project state and debt register: `DEVLOG.md`.

## Commands

```bash
# Install (the extra is `test`, not `dev`)
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[test]"

# Run all tests
pytest tests/ -q

# Run one package / file / test
pytest tests/anonymization -q
pytest tests/anonymization/test_partial_masking.py::test_basic -v

# Coverage (public-API scope defined by .coveragerc)
python -m pytest tests/ --cov --cov-report=term

# Lint (CI gate is errors-only; the full run is advisory)
ruff check pamola_core/ --select E9,F63,F7,F82 --no-fix
ruff check pamola_core/ --statistics

# Build / CLI
python -m build
pamola-core --help
```

Installing pulls `torch`, `sdv`, `spacy`, `faiss-cpu`, `dask[complete]` — expect a multi-GB install.

## What this is

`pamola-core` is a **library** (PyPI: `pamola-core`), not a service: file-oriented anonymization
operations that read a `DataSource` and write a `task_dir`. It is the open-source core of PAMOLA,
evolved from the earlier `HHR` package. It is **independently usable** — nothing downstream is
required to run it.

The sibling project **BEST** (`D:\VK\_DEVEL\best`) is the governed execution system. It is *not*
a runtime consumer of this library: it retired the `pamola-core` bridge on 2026-06-14 and its
operations are native polars. What the two share is conceptual — DSL terms, privacy semantics,
artifact vocabulary, metric definitions. See ADR-PC-05 in `DEVLOG.md`.

Authoritative framework docs live in BOX: `C:\Users\valer\Box\DEVBOX\PAMOLA\core\` (`00_INDEX.md` …).
Read BOX for intent, the repo for reality; when they disagree, that gap is a finding for `docs/output/`.

## Architecture

| Package | Role |
|---|---|
| `pamola_core/anonymization` | generalization, masking, noise, suppression, pseudonymization |
| `pamola_core/profiling` | `analyzers/` + `commons/` + `schemas/` — dataset profiling ops |
| `pamola_core/transformations` | cleaning, field_ops, grouping, imputation, merging, splitting |
| `pamola_core/metrics` | fidelity / privacy / utility ops + scoring and schema commons |
| `pamola_core/fake_data` | synthetic replacement ops (name, email, phone, organization) |
| `pamola_core/privacy_models`, `attacks`, `analysis` | k-anonymity, re-identification attacks, stats |
| `pamola_core/io` | csv / excel / json / parquet readers |
| `pamola_core/utils` | largest package (147 files): `ops/`, `tasks/`, `io_helpers/`, `crypto_helpers/`, `nlp/` |
| `pamola_core/errors` | structured error framework + `context/*.yaml` |
| `pamola_core/cli` | typer app, entry point `pamola-core` |

`tests/` mirrors this layout.

## Operation contract

Every operation subclasses `BaseOperation` (`pamola_core/utils/ops/op_base.py`) and follows the
PAMOLA 7-step lifecycle:

- input is a `DataSource` (`op_data_source.py`); output is an `OperationResult` + `OperationStatus`
- every op writes a `task_dir` — `output/`, metrics JSON, `config.json`
- modes: `REPLACE` (overwrite the field) and `ENRICH` (new field, honouring `column_prefix`)
- config subclasses `OperationConfig`; **secrets must be listed in `SENSITIVE_KEYS`** so
  `to_safe_dict()` redacts them before `config.json` is written (regression guard for the
  1.0.0.dev3 CRITICAL key-leak fix)
- ops ship the 4-file schema pattern: `*_core_schema.py`, `*_schema_exclude.py`, `*_tooltip.py`, `*_ui_schema.py`

**Adding a public operation requires all five:** op module + 4 schema files + tests +
export in `pamola_core/__init__.py` `__all__` + entry in `.coveragerc` `[run] include` +
`CHANGELOG.md` entry. `__all__` and `.coveragerc` together define the public API (SRS 4.1.11).

**Public signatures and artifact filenames are a contract with downstream OSS users** — the
surface defined by `__all__` and `.coveragerc`. Breaking them needs an explicit decision
recorded in `DEVLOG.md`. Do not justify or block a change with "BEST depends on it" (ADR-PC-05).

## Branching and release

- Default branch `develop`; stable branch `main`. Never commit directly to either.
- Branch names `feat/…`, `fix/…`, `docs/…`; conventional commit subjects.
- Releases are tag-driven: `v*dev*` tags on `develop`, stable `v*` on `main`.
  `.github/workflows/release.yml` enforces tag↔branch, tag↔`pyproject.toml`, and version↔CHANGELOG.
  Do not hand-edit `version` outside a deliberate release-prep commit.
- Do not add runtime dependencies to `pyproject.toml` without approval, and do not add files
  to `data/raw/` (the repo already tracks ~120 MB of CSV).

## Reports

CC reports go to `docs/output/` as `YYYYMMDD_CC_<TOPIC>.md`, ending with a GCA Evidence Package
table (`FR-*/AC-*` | Status | `file:line` + test name). Narrative progress goes to `DEVLOG.md`.
Agent system prompts live in `docs/prompts/`.
