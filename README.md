# PAMOLA.CORE

<p align="center">
  <img src="https://realmdata.io/assets/img/logos/pamola-logo.png" alt="PAMOLA Logo" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSD%203--Clause-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10--3.12-blue.svg"></a>
  <a href="https://pypi.org/project/pamola-core/"><img alt="PyPI" src="https://img.shields.io/badge/pypi-1.0.0.dev4-orange.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange.svg">
</p>

---

## Privacy Engineering for Python. Finally.

**PAMOLA.CORE** is the open-source foundation of the PAMOLA platform — a Python library for **privacy-preserving data operations** with reproducible pipelines, structured artifacts, and an audit trail per operation.

Developed by **[Realm Inveo Inc.](https://realmdata.io)**

---

## The Problem

You need to anonymize sensitive data. You've tried:

- **ARX:** Powerful, but Java, GUI-focused, opaque operations.
- **Faker + Presidio + custom scripts:** Fragmented, no pipeline, no proof.
- **DP libraries:** Great math, but narrow scope.

You're still missing:

- Direct operations (mask, generalize, pseudonymize, suppress) — not just "achieve k-anonymity".
- Risk measurement (privacy/fidelity/utility metrics on real outputs).
- Reproducibility (config + metrics + artifacts written to disk per task).

---

## The Solution

PAMOLA.CORE: operations-first privacy engineering with a per-operation lifecycle (validate → load → process → save → metrics → visualize → cache).

```python
from pathlib import Path

import pandas as pd

from pamola_core import HashBasedPseudonymizationOperation, FullMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.tasks.task_reporting import TaskReporter
from pamola_core.utils.progress import HierarchicalProgressTracker

df = pd.read_csv("customers.csv")
data_source = DataSource(dataframes={"main": df})
task_dir = Path("./anonymize_customers")
reporter = TaskReporter(task_dir=task_dir, task_name="anonymize")
tracker = HierarchicalProgressTracker(total=2, description="Anonymize")

# 1) Irreversible hash-based pseudonymization of the email column
HashBasedPseudonymizationOperation(
    field_name="email",
    algorithm="sha3_256",
    salt_config={"source": "parameter", "value": "ab" * 32},
    use_pepper=True,
).execute(data_source, task_dir, reporter, tracker)

# 2) Mask the phone column entirely
FullMaskingOperation(
    field_name="phone",
    mask_char="*",
).execute(data_source, task_dir, reporter, tracker)
```

**Output structure (`task_dir/`):**

```
anonymize_customers/
├── config.json           # Operation configuration (secrets redacted)
├── output/               # Anonymized data (csv/parquet)
├── metrics/              # Privacy & quality metrics (JSON)
├── visualizations/       # Generated charts (PNG)
└── logs/                 # Per-task execution log
```

> **Security note (since 1.0.0.dev3):** `config.json` redacts sensitive parameters (e.g. AES-256 mapping encryption keys) via `OperationConfig.SENSITIVE_KEYS`. See [CHANGELOG.md](CHANGELOG.md) for the full release notes.

---

## PAMOLA Ecosystem

PAMOLA.CORE is part of a comprehensive privacy engineering stack:

| Component | Description | Availability |
|-----------|-------------|--------------|
| **PAMOLA.CORE** | Anonymization, profiling, transformation, metrics, and shared op framework | **Open Source (this repo)** |
| **PAMOLA.STUDIO** | Visual environment for data transformation and privacy management | Commercial |
| **PAMOLA.SYNT** | Synthetic data generation, including formal DP-SGD-based generators | Commercial |
| **PAMOLA.BEST** | Best-practice policy modules and DP accounting | Commercial |
| **PAMOLA.TEXT** | Long text and document anonymization (NLP/LLM-based) | Commercial |
| **PAMOLA.INSIGHT** | Agent modules for LLM integration | Commercial |

> **Scope of CORE:** This package provides anonymization and pseudonymization primitives, classical privacy metrics, and reproducibility plumbing. It does **not** implement formal differential-privacy synthetic-data generation — that lives in PAMOLA.SYNT / PAMOLA.BEST.

---

## What's In CORE

All classes below are exported from the top-level `pamola_core` package.

| Category | Operations |
|----------|------------|
| **Anonymization — Masking** | `FullMaskingOperation`, `PartialMaskingOperation` |
| **Anonymization — Generalization** | `CategoricalGeneralizationOperation`, `NumericGeneralizationOperation`, `DateTimeGeneralizationOperation` |
| **Anonymization — Suppression** | `AttributeSuppressionOperation`, `CellSuppressionOperation`, `RecordSuppressionOperation` |
| **Anonymization — Pseudonymization** | `HashBasedPseudonymizationOperation` (SHA3-256/512 + salt + pepper), `ConsistentMappingPseudonymizationOperation` (AES-256-GCM reversible mapping) |
| **Anonymization — Noise** | `UniformNumericNoiseOperation`, `UniformTemporalNoiseOperation` |
| **Profiling — Field analyzers** | `CategoricalOperation`, `CorrelationOperation`, `CorrelationMatrixOperation`, `CurrencyOperation`, `DateOperation`, `EmailOperation`, `GroupAnalyzerOperation`, `IdentityAnalysisOperation`, `MVFOperation`, `NumericOperation`, `PhoneOperation`, `TextSemanticCategorizerOperation` |
| **Profiling — Dataset-level** | `KAnonymityProfilerOperation`, `DataAttributeProfilerOperation` |
| **Transformation** | `AddOrModifyFieldsOperation`, `RemoveFieldsOperation`, `CleanInvalidValuesOperation`, `ImputeMissingValuesOperation`, `AggregateRecordsOperation`, `MergeDatasetsOperation`, `SplitByIDValuesOperation`, `SplitFieldsOperation` |
| **Fake Data** | `FakeNameOperation`, `FakeEmailOperation`, `FakePhoneOperation`, `FakeOrganizationOperation` |
| **Metrics** | `FidelityOperation` (KS, KL-divergence), `PrivacyMetricOperation` (DCR, NNDR, uniqueness, k-anonymity, l-diversity), `UtilityMetricOperation` (classification, regression) |
| **Analysis helpers** | `analyze_dataset_summary`, `analyze_descriptive_stats`, `analyze_correlation`, `visualize_distribution_df`, `calculate_full_risk` |

> **Note on attack simulation:** `pamola_core/attacks/` is an **internal** package with no stability guarantee, and it is deliberately narrow — it carries only the attacks that measure *anonymization* quality: `LinkageAttack` (including the CVPL variant) and `AttributeInference`. There are no public, registered `Attack*Operation` classes.
>
> Two corrections to earlier releases of this note. These modules are **not** used by `PrivacyMetricOperation` — that operation imports `pamola_core.metrics.privacy`. And DCR/NNDR live in `pamola_core.metrics.privacy.distance` and `.neighbor`; duplicate copies that had accumulated under `attacks/` were removed, as was membership inference, which measures model privacy rather than anonymization.
>
> **Singling-out is not implemented.** It is the third anonymization-facing criterion, and no report produced from this package covers it.

---

## Pseudonymization Spotlight

The `1.0.0.dev3` release hardened the pseudonymization stack:

```python
from pamola_core import ConsistentMappingPseudonymizationOperation

op = ConsistentMappingPseudonymizationOperation(
    field_name="customer_id",
    mapping_encryption_key="ab" * 32,         # 256-bit hex key
    pseudonym_type="uuid",                    # or "sequential" / "random_string"
    mapping_format="csv",                     # encrypted at rest with AES-256-GCM
    persist_frequency=1000,
)

# Outputs:
#  - {task_dir}/output/         anonymized data
#  - {task_dir}/output/         encrypted mapping file (re-identification key)
#  - {task_dir}/metrics/        operation metrics
#  - {task_dir}/config.json     mapping_encryption_key is replaced with "*REDACTED*"
```

Highlights:

- **AES-256-GCM mapping encryption keys are never persisted to disk.** A new `OperationConfig.SENSITIVE_KEYS` declaration + `to_safe_dict()` redacts secrets before any `save_config()` call.
- **Hash-based op rejects weak salts** (all-zero or empty) when `use_pepper=False`.
- **Per-run session id** invalidates stale disk cache when `use_pepper=True`, so previous-run pseudonyms cannot be served back.
- **Compound identifiers**, **ENRICH/REPLACE modes**, **reverse mapping**, and **Dask pickle safety** are covered by 41 dedicated tests.

---

## Metrics

Metric operations write JSON artifacts under `{task_dir}/metrics/`:

```python
from pamola_core import FidelityOperation, PrivacyMetricOperation

FidelityOperation(
    fidelity_metrics=["ks", "kl"],
    columns=["age", "income"],
).execute(data_source, task_dir, reporter, tracker)

PrivacyMetricOperation(
    privacy_metrics=["dcr", "nndr", "uniqueness"],
    quasi_identifiers=["age", "gender", "zipcode"],
).execute(data_source, task_dir, reporter, tracker)
```

For dataset-level utility scoring (classification / regression downstream models) use `UtilityMetricOperation`.

---

## Installation

**From PyPI:**

```bash
pip install --pre pamola-core
```

> **The `--pre` flag is required.** Every `1.0.0` release so far is a
> pre-release, and `pip` skips pre-releases by default — so a plain
> `pip install pamola-core` will **not** give you this library. You can also
> pin the exact version instead: `pip install pamola-core==1.0.0.dev4`.
> The flag stops being necessary once `1.0.0` final ships.

**From source:**

```bash
git clone https://github.com/DGT-Network/PAMOLA.git
cd PAMOLA
pip install -e .
```

**Test extras:**

```bash
pip install -e ".[test]"   # adds pytest, pytest-cov
```

> **Heads-up:** All scientific dependencies (numpy, pandas, scikit-learn, scipy, dask, spacy, faker, cryptography, etc.) are pinned in the main `[project.dependencies]` table — no separate `[fast]/[ner]/[dp]` extras in this release.
>
> As of `1.0.0.dev4`, `torch` and `sdv` are **no longer installed**: nothing in
> the library imported them. If you relied on them being pulled in as a side
> effect, install them explicitly. Splitting the remaining heavy dependencies
> (spacy, nltk, faiss, dask, the plotting stack) into optional extras is
> planned for a later release.

---

## Supported Python Versions

PAMOLA.CORE supports Python **3.10, 3.11, and 3.12** (`requires-python = ">=3.10,<3.13"`).

| Python Version | Supported |
|---|---|
| 3.10 | ✅ |
| 3.11 | ✅ |
| 3.12 | ✅ |
| 3.9 and below | ❌ |
| 3.13 and above | ❌ (blocked by third-party dependencies) |

---

## Core Dependencies

A non-exhaustive view of the heaviest third-party packages (full list in `pyproject.toml`):

| Package | Pin | Purpose |
|---|---|---|
| **numpy** | `1.26.4` | Numerical computation across metrics, attacks, statistical analysis |
| **pandas** | `2.2.2` | The DataFrame container for every CORE operation |
| **scikit-learn** | `1.7.2` | Classification/regression metrics, nearest-neighbor distance, model-based utility |
| **scipy** | `1.15.3` | Statistical tests (KS, KL divergence) used by `FidelityOperation` |
| **cryptography** | `46.0.3` | AES-256-GCM mapping encryption for `ConsistentMappingPseudonymizationOperation` |
| **dask[complete]** | `2025.11.0` | Optional out-of-core / distributed execution path |
| **pyarrow** | `14.0.2` | Parquet I/O |
| **typer** | `0.24.1` | CLI entry point |

---

## Versioning

PAMOLA.CORE follows [Semantic Versioning](https://semver.org/) and [PEP 440](https://peps.python.org/pep-0440/).

```python
import pamola_core
print(pamola_core.__version__)   # e.g. "1.0.0.dev4"
```

| Phase | Version | Branch | Tag | Install |
|-------|---------|--------|-----|---------|
| Dev (current) | `1.0.0.dev4` | `develop` | `v1.0.0.dev4` | `pip install --pre pamola-core` |
| Stable (planned) | `1.0.0` | `main` | `v1.0.0` | `pip install pamola-core` |

- **Source of truth:** `pyproject.toml` → `version`
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **CI/CD:** GitHub Actions — lint (ruff), test (3.10/3.11/3.12, pytest), build (sdist+wheel), PyPI publish on tag `v*`
- **Release rules:** Dev tags (`v*dev*`) must be on `develop`; stable tags on `main`.

---

## CLI

The `pamola-core` console script is installed automatically:

```bash
pamola-core --version
pamola-core list-ops                              # discover registered operations
pamola-core run --task task.json                  # run a task definition
pamola-core run --op FullMaskingOperation --config config.json --input data.csv
pamola-core schema FullMaskingOperation           # show parameter schema
pamola-core validate-config --config config.json  # validate a config file
```

Run `pamola-core --help` for the full command list.

---

## Examples

Hands-on notebooks live under [`examples/`](https://github.com/DGT-Network/PAMOLA/tree/main/examples):

- `examples/anonymization/pseudonymization/` — simple + advanced for hash-based and consistent-mapping pseudonymization
- `examples/anonymization/` — masking, generalization, noise, suppression
- `examples/profiling/` — field-level and dataset-level profilers
- `examples/transformations/` — merge, split, aggregate, clean, impute
- `examples/fake_data/` — synthetic identity, email, phone, organization
- `examples/metrics/` — fidelity, privacy, utility metrics
- `examples/data_examples/sample.csv` — non-PII synthetic sample used by the notebooks

> **No real personal data** is included in this repository. All sample records are artificially generated.

---

## Philosophy

- **Operations-first:** Direct transforms with a well-defined 7-step lifecycle, not constraint optimization.
- **Measure everything:** Privacy, fidelity, and utility metrics persisted as JSON artifacts.
- **Reproducibility by default:** Each operation writes a `config.json` (with secrets redacted) alongside its output.
- **Secret hygiene:** `OperationConfig.SENSITIVE_KEYS` provides a single place to declare parameters that must never reach disk.

---

## API Documentation

The project uses **Sphinx** to generate API reference documentation from Python docstrings.

```bash
cd docs
make html        # output: docs/_build/html/index.html
```

---

## Documentation

| Resource | Link |
|----------|------|
| **PET Knowledge Base** | [realmdata.io/kb](https://realmdata.io/kb/index.html) |
| **Technical Documentation** | [docs/en/index.md](https://github.com/DGT-Network/PAMOLA/blob/main/docs/en/index.md) |
| **Glossary** | [realmdata.io/glossary](https://realmdata.io/pages/glossary.html) |
| **Examples** | [`examples/`](https://github.com/DGT-Network/PAMOLA/tree/main/examples) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |

---

## Use Cases

- **Data Engineering:** Prepare privacy-safe datasets for ML training.
- **Healthcare:** HIPAA-oriented de-identification workflows (Safe Harbor support).
- **Finance:** Privacy engineering aligned with PCI/GDPR considerations.
- **Compliance:** Audit-ready evidence with structured per-operation artifacts.
- **Data Sharing:** Risk-assessed data exchange between organizations.

---

## Regulatory Context

PAMOLA.CORE provides technical building blocks for privacy compliance programs:

| Regulation | Relevant Capabilities |
|------------|----------------------|
| **GDPR** | Pseudonymization (reversible / irreversible), data minimization (Art. 25, 32) |
| **HIPAA** | Safe Harbor de-identification support |
| **CCPA/CPRA** | Data suppression, masking, anonymization workflows |

> **Important:** PAMOLA.CORE provides technical capabilities only. Legal compliance requires organizational policies, procedures, and legal guidance beyond software tools.

---

## Contributing

```bash
git clone https://github.com/DGT-Network/PAMOLA.git
cd PAMOLA
pip install -e ".[test]"
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Ownership & Licensing

**PAMOLA.CORE** is developed and owned exclusively by **[Realm Inveo Inc.](https://realmdata.io)**

This repository is hosted under the DGT-Network GitHub organization, which provides shared development infrastructure for Realm Inveo projects. **DGT-Network does not claim ownership of this intellectual property.** All IP rights belong exclusively to Realm Inveo Inc.

**License:** BSD 3-Clause — see [LICENSE](https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE).

---

## Contact

| Purpose | Contact |
|---------|---------|
| **General inquiries** | [contact@realmdata.io](mailto:contact@realmdata.io) |
| **Commercial / Sales** | [sales@realmdata.io](mailto:sales@realmdata.io) |
| **Due diligence / Legal** | [legal@realmdata.io](mailto:legal@realmdata.io) |
| **Website** | [realmdata.io](https://realmdata.io) |

---

<p align="center">
  <sub>Built by <a href="https://realmdata.io">Realm Inveo Inc.</a></sub>
</p>
