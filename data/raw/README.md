# Raw Sample Data - Fully Synthetic Test Datasets

> **⚠️ Legal Notice**
> 
> All datasets in this directory are **artificially generated** for testing and demonstration purposes.
> **No real personal information (PII/PHI) is included.**
> 
> - Not collected from any external sources.
> - Not derived from any real customer, employee, or patient datasets.
> - Any resemblance to actual persons, organizations, or events is purely coincidental.
>
> *This notice is provided for transparency, not as legal advice.*

---

## Purpose

These synthetic datasets serve three primary functions in PAMOLA development:

| Purpose | Description |
|---------|-------------|
| **Unit & Integration Testing** | Provide consistent, reproducible inputs for automated tests |
| **Anonymization Validation** | Measure effectiveness of de-identification operations against known ground truth |
| **Demo Scenarios** | Showcase PAMOLA capabilities without handling sensitive data |

### Why Synthetic Data?

Privacy engineering tools must be tested on data that *resembles* real-world patterns without exposing actual individuals. Synthetic data provides:

- **Known ground truth** - we control what is "identifiable" for measuring re-identification risk
- **Domain coverage** - banking, healthcare, HR, telecom scenarios
- **Reduced compliance exposure** - enables safe demos and testing without using regulated real-world data

---

## Datasets

| Directory | Domain | Records | Description |
|-----------|--------|---------|-------------|
| `bank_credit_ca/` | Banking | ~10K | Credit application profiles |
| `bank_fraud/` | Banking | ~50K | Transaction data with fraud indicators |
| `bank_txs/` | Banking | ~100K | General transaction records |
| `churn_ca/` | Telecom | ~5K | Customer churn prediction features |
| `hr_job_resume/` | HR | ~2K | Job postings and candidate profiles |
| `med_ehr/` | Healthcare | ~8K | Electronic health records (synthetic) |
| `med_rare_lab/` | Healthcare | ~1K | Rare disease lab results |

**Formats:** CSV and JSON. Schemas documented in per-dataset README files.

> **Note:** Record counts are approximate. See individual dataset README files for exact specifications.

---

## Data Provenance & Fingerprinting

### Generation Methods

All datasets are programmatically generated using:

- **[Faker](https://faker.readthedocs.io/)** - names, addresses, emails, dates
- **Domain-specific generators** - realistic medical codes, transaction patterns
- **Statistical distributions** - matching real-world patterns without real data

### Synthetic Fingerprints

These datasets contain intentional markers (fingerprints) embedded during generation.

**What fingerprints are:**
- Synthetic patterns in value distributions
- Metadata markers in identifier fields
- Statistical signatures for lineage tracking

**What fingerprints are NOT:**
- They do not contain any real personal data
- They do not track users or system activity
- They do not transmit or expose any information externally

**Purpose of fingerprints:**

1. **Prevent misrepresentation** - detect if synthetic data is incorrectly claimed as real
2. **Enable lineage tracking** - trace data origin through processing pipelines
3. **Support audit scenarios** - demonstrate data provenance in compliance reviews

These markers do not affect testing validity.

---

## Usage

> **Note:** API may change; examples reflect the current development snapshot.

### Loading Data
```python
import pandas as pd
from pamola_core.tasks import TaskRunner
from pamola_core.profiling import ProfileOperation

# Direct pandas load
df = pd.read_csv("data/raw/bank_credit_ca/credit_applications.csv")

# Via PAMOLA TaskRunner
task = TaskRunner(task_dir="./test_run", seed=42)
result = task.run(
    [ProfileOperation(params={"analyzers": ["all"]})],
    input_data="data/raw/bank_credit_ca/credit_applications.csv"
)
```

### Testing Anonymization Effectiveness
```python
from pamola_core.attacks import LinkageAttack
from pamola_core.synthesis import SynthesisOperation

# Generate synthetic version
synth_result = task.run([SynthesisOperation(params={...})], input_data=original_path)

# Measure re-identification risk against known original
attack = LinkageAttack(original=original_path, synthetic=synth_result.output_path)
risk_score = attack.evaluate()
```

---

## Policy

> **🚫 Do not commit real customer, employee, or patient data to this repository.**

This is a public open-source repository. For testing with real data:
- Use approved secure environments
- Follow internal data handling procedures
- Never push to public branches

---

## Adding New Datasets

1. Create directory: `data/raw/{domain}_{name}/`
2. Generate data with documented method
3. Add fingerprints per generation standards
4. Create `README.md` with schema and provenance
5. Update this file's dataset table

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

---

## License

Apache 2.0 - same as PAMOLA.CORE repository.

See [LICENSE](../../LICENSE) for full terms.

---

**Maintainer:** [Realm Inveo Inc.](https://realmdata.io) - info@realmdata.io  
**Repository:** [github.com/DGT-Network/PAMOLA](https://github.com/DGT-Network/PAMOLA)
