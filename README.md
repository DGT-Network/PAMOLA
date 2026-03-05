# PAMOLA.CORE

<p align="center">
  <img src="https://realmdata.io/assets/img/logos/pamola-logo.png" alt="PAMOLA Logo" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10--3.12-blue.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange.svg">
</p>

---

## Privacy Engineering for Python. Finally.

**PAMOLA.CORE** is the open-source foundation of the PAMOLA platform - a Python library for **privacy-preserving data operations** with pipeline workflow, reproducibility, and audit trail.

Developed by **[Realm Inveo Inc.](https://realmdata.io)**

---

## The Problem

You need to anonymize sensitive data. You've tried:

- **ARX:** Powerful, but Java, GUI-focused, opaque operations
- **Faker + Presidio + custom scripts:** Fragmented, no pipeline, no proof
- **DP libraries:** Great math, but narrow scope

You're still missing:

- Direct operations (mask, drop, fake - not just "achieve k-anonymity")
- Risk testing (can someone actually re-identify records?)
- Short text handling (job titles, comments - without LLM)
- Reproducibility (what exactly was done?)

---

## The Solution

PAMOLA.CORE: operations-first privacy engineering with full audit trail.

```python
from pamola_core.tasks import TaskRunner
from pamola_core.profiling import ProfileOperation
from pamola_core.anonymization import MaskingOperation, GeneralizationOperation
from pamola_core.noise import LaplaceNoiseOperation
from pamola_core.metrics import PrivacyProxyMetricsOperation
from pamola_core.attacks import AttackSuiteOperation

# Define pipeline with reproducible seed
task = TaskRunner(task_dir="./anonymize_customers", seed=42)

task.run([
    ProfileOperation(params={"analyzers": ["all"]}),
    MaskingOperation(params={"fields": ["name", "email", "phone"], "strategy": "partial"}),
    GeneralizationOperation(params={"field": "age", "bins": [0, 18, 35, 50, 65, 100]}),
    LaplaceNoiseOperation(params={"fields": ["salary"], "epsilon": 1.0, "sensitivity": 1000}),
    PrivacyProxyMetricsOperation(params={"metrics": ["k_anonymity", "l_diversity"]}),
    AttackSuiteOperation(params={"policy": "standard"}),
], input_data="customers.csv")

# Result: task_dir/ with data, metrics, attack results, manifest.json
```

**Output structure:**

```
anonymize_customers/
├── manifest.json          # Full reproducibility record
├── output/                # Anonymized data (csv/parquet)
│   └── anonymized.csv
├── metrics/               # Privacy & utility metrics (JSON)
│   ├── metrics_summary.json
│   └── metrics_detail.json
├── attacks/               # Attack simulation results
│   └── suite_report.json
├── plots/                 # Generated visualizations
├── dictionaries/          # Extracted mappings
└── logs/                  # Execution logs
```

---

## PAMOLA Ecosystem

PAMOLA.CORE is part of a comprehensive privacy engineering stack:

| Component | Description | Availability |
|-----------|-------------|--------------|
| **PAMOLA.CORE** | Operations, metrics, attacks, pipeline runtime | **Open Source** |
| **PAMOLA.STUDIO** | Visual environment for data transformation and privacy management | Commercial |
| **PAMOLA.SYNT** | Synthetic data generation with DP guarantees (CTGAN, TVAE) | Commercial |
| **PAMOLA.TEXT** | Long text and document anonymization (NLP/LLM-based) | Commercial |
| **PAMOLA.INSIGHT** | Agent modules for LLM integration | Commercial |

---

## What's In CORE

| Category | Operations |
|----------|------------|
| **Profiling** | `ProfileOperation`, `CorrelationOperation`, `ShortTextProfileOperation` |
| **Anonymization** | `MaskingOperation`, `GeneralizationOperation`, `SuppressionOperation`, `PseudonymizationOperation` |
| **Transformation** | `CleaningOperation`, `MergeOperation`, `SplitOperation`, `AggregateOperation` |
| **Noise (DP-semantics)** | `LaplaceNoiseOperation`, `GaussianNoiseOperation`, `DateTimeJitterOperation`, `RandomizedResponseOperation` |
| **Short Text** | `ShortTextProfileOperation`, `ShortTextCategorizerOperation`, `ShortTextMaskOperation`, `ShortTextNEROperation` |
| **Fake Data** | `FakeNameOperation`, `FakeEmailOperation`, `FakePhoneOperation`, `FakeOrgOperation` |
| **Metrics** | `QualityMetricsOperation`, `PrivacyProxyMetricsOperation`, `AttackBasedMetricsOperation`, `CompositeScoreOperation` |
| **Attacks** | `CVPLAttackOperation`, `LinkageAttackOperation`, `SinglingOutOperation`, `AttributeInferenceOperation`, `AttackSuiteOperation` |

---

## Data-Release Attacks

PAMOLA.CORE tests practical re-identification risk on your **data**:

| Attack | Question |
|--------|----------|
| **CVPL** | How much information leaks between releases? (PAMOLA signature) |
| **Fellegi-Sunter Linkage** | Can records be matched to external data? |
| **Singling-out** | Are any records uniquely identifiable? |
| **Attribute inference** | Can sensitive attributes be guessed from QI? |

```python
from pamola_core.tasks import TaskRunner
from pamola_core.attacks import AttackSuiteOperation

task = TaskRunner(task_dir="./risk_assessment", seed=42)
task.run([
    AttackSuiteOperation(params={
        "policy": "standard",  # or "minimal", "comprehensive"
        "quasi_identifiers": ["age", "gender", "zipcode"],
        "sensitive_columns": ["diagnosis"]
    })
], input_data="anonymized.csv")

# Result: attacks/suite_report.json with risk scores and verdicts
```

> **Note:** Model-centric attacks (MIA on generators) belong to PAMOLA.SYNT

---

## Metrics with Verdicts

Metrics produce actionable signals, not just numbers:

```python
from pamola_core.metrics import CompositeScoreOperation

# Aggregate metrics with weighted scoring
CompositeScoreOperation(params={
    "weights": {
        "quality": 0.3,
        "privacy_proxy": 0.2,
        "privacy_attack": 0.4,  # Attack-based metrics weighted higher
        "utility": 0.1
    }
})
# Output: metrics_summary.json with verdict (PASS/WARN/FAIL)
```

**Output example (`metrics_summary.json`):**

```json
{
  "overall": {
    "quality_score": 0.85,
    "privacy_score": 0.78,
    "composite_score": 0.84,
    "verdict": "PASS"
  },
  "metrics": {
    "k_anonymity": {"value": 5, "verdict": "PASS"},
    "linkage_rate": {"value": 0.02, "verdict": "PASS"}
  }
}
```

---

## DP-Semantics Noise

Add calibrated noise with differential privacy semantics:

```python
from pamola_core.noise import LaplaceNoiseOperation

LaplaceNoiseOperation(params={
    "fields": ["salary", "age"],
    "epsilon": 1.0,
    "sensitivity": {"salary": 1000, "age": 1},
    "seed": 42,
    "clip_bounds": {"salary": [0, None]}  # No negative values
})
# Output includes noise_report.json documenting exactly what was applied
```

> **Note:** This provides DP-like noise but NOT formal DP guarantees without external accountant. For formal guarantees, use `pamola-core[dp]` with OpenDP adapter.

---

## What's NOT in CORE

| Feature | Package | Why separate |
|---------|---------|--------------|
| Long text + LLM anonymization | `pamola-core[text]` | Heavy deps (torch, transformers) |
| Formal DP with accountant | `pamola-core[dp]` | Use OpenDP/diffprivlib adapters |
| Synthetic data generation | `pamola-synt` | Different concern, ML models |
| Model-centric attacks (MIA) | `pamola-synt` | Requires trained model access |

---

## Installation

**From source (current):**

```bash
git clone https://github.com/DGT-Network/PAMOLA.git
cd PAMOLA
pip install -e .
```

**With optional extras:**

```bash
pip install -e ".[fast]"       # + Polars, ConnectorX, DuckDB
pip install -e ".[profiling]"  # + YData-profiling, Presidio
pip install -e ".[ner]"        # + spaCy for short text NER
pip install -e ".[dp]"         # + OpenDP for formal DP guarantees
pip install -e ".[dev]"        # + pytest, coverage, black, ruff
```

**PyPI (coming soon):**

```bash
pip install pamola-core
pip install pamola-core[fast,ner]
```

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

These packages are declared in `pyproject.toml` under `[project.dependencies]` and are automatically installed with `pip install pamola-core`.

| Package | Version Range | Purpose |
|---|---|---|
| **numpy** | `==1.26.4` | Numerical computation and array operations used throughout privacy metrics, attack simulations, and statistical analysis |
| **pandas** | `==2.2.2` | Tabular data structures and DataFrame processing; the primary data container for all PAMOLA operations |
| **scikit-learn** | `==1.7.2` | Machine learning utilities used by core operations including nearest-neighbor attacks, classification metrics, and model-based privacy risk assessment |

---

## CLI

```bash
pamola-core --version
pamola-core list-ops
pamola-core run --task task.json --output ./task_dir
pamola-core run --op MaskingOperation --config config.json --input data.csv
pamola-core schema MaskingOperation
```

---

## Sample Data

> **Note:** Synthetic test datasets are available in [`data/raw/`](https://github.com/DGT-Network/PAMOLA/tree/main/data/raw) for development and testing purposes only.
>
> **No real personal data (PII/PHI) is included.** All records are artificially generated.

---

## Philosophy

- **Operations-first:** Direct transforms, not constraint optimization
- **Measure everything:** Quality, privacy, utility - with verdicts
- **Test before release:** Practical risk via data-release attacks
- **Noise with transparency:** DP-semantics + detailed reports
- **Reproducibility by default:** manifest.json tracks everything

---

## Documentation

| Resource | Link |
|----------|------|
| **PET Knowledge Base** | [realmdata.io/kb](https://realmdata.io/kb/index.html) |
| **Technical Documentation** | [docs/en/index.md](https://github.com/DGT-Network/PAMOLA/blob/main/docs/en/index.md) |
| **Glossary** | [realmdata.io/glossary](https://realmdata.io/pages/glossary.html) |
| **Examples** | [`examples/`](https://github.com/DGT-Network/PAMOLA/tree/main/examples) |

---

## Use Cases

- **Data Engineering:** Prepare privacy-safe datasets for ML training
- **Healthcare:** HIPAA-oriented de-identification workflows (Safe Harbor support)
- **Finance:** Privacy engineering aligned with PCI/GDPR considerations
- **Compliance:** Audit-ready evidence with manifest.json and attack reports
- **Data Sharing:** Risk-assessed data exchange between organizations

---

## Regulatory Context

PAMOLA.CORE provides technical building blocks for privacy compliance programs:

| Regulation | Relevant Capabilities |
|------------|----------------------|
| **GDPR** | Pseudonymization, data minimization (Art. 25, 32) |
| **HIPAA** | Safe Harbor de-identification support |
| **CCPA/CPRA** | Data suppression, anonymization workflows |

> **Important:** PAMOLA.CORE provides technical capabilities only. Legal compliance requires organizational policies, procedures, and legal guidance beyond software tools.

---

## Contributing

```bash
git clone https://github.com/DGT-Network/PAMOLA.git
cd PAMOLA
pip install -e ".[dev]"
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Ownership & Licensing

**PAMOLA.CORE** is developed and owned exclusively by **[Realm Inveo Inc.](https://realmdata.io)**

This repository is hosted under DGT-Network GitHub organization, which provides shared development infrastructure for Realm Inveo projects. **DGT-Network does not claim ownership of this intellectual property.** All IP rights belong exclusively to Realm Inveo Inc.

**License:** Apache 2.0 - see [LICENSE](https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE)

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
