# PAMOLA.CORE Test Suite

**Last Updated:** 2026-04-03
**Total Files:** 172
**Total Tests:** 5,436
**Pass Rate:** 100%
**Public API Coverage:** 85%
**Python:** 3.10–3.12
**Framework:** pytest + pytest-cov

---

## Quick Run

```bash
# Run all tests
python -m pytest tests/ -v

# Run all tests (quiet)
python -m pytest tests/ --tb=no -q

# Run specific module
python -m pytest tests/anonymization/ -v

# Run with coverage (uses .coveragerc for Public API scope)
python -m pytest tests/ --cov --cov-report=term-missing

# Run with JUnit XML + HTML reports (same as CI)
python -m pytest tests/ --tb=short -q \
  --junitxml=reports/junit.xml \
  --cov --cov-report=html:reports/coverage-html \
  --cov-report=term

# Run per-module summary
for dir in analysis anonymization attacks catalogs cli common configs errors fake_data io_readers metrics privacy_models profiling transformations utils; do
  echo "=== $dir ==="
  python -m pytest tests/$dir/ --tb=no -q --no-header 2>&1 | tail -1
done
```

---

## Strategy: Tiered Testing

| Tier | Scope | Target | Tests |
|------|-------|--------|-------|
| **Tier 1** | Public API (39 ops + 17 functions) | >=85% coverage ✓ | ~3,500 |
| **Tier 2** | Framework core (BaseOperation, BaseTask, privacy_models, attacks, cli) | >=85% coverage ✓ | ~1,936 |
| **Tier 3** | Internal helpers | NOT tested individually — covered indirectly | 0 |

---

## Test Inventory

### Tier 1 — Public API (~106 files, ~3,500 tests)

| Directory | Files | Tests | What it tests |
|-----------|-------|-------|---------------|
| `analysis/` | 9 | ~380 | `analyze_dataset_summary`, `calculate_full_risk`, `analyze_descriptive_stats`, `visualize_distribution_df`, `analyze_correlation` + edge/deep coverage |
| `anonymization/` | 24 | ~700 | 10 anonymization ops + dask/enrich/progress/suppression deep coverage |
| `profiling/` | 18 | ~600 | 14 profiling analyzers + attribute_utils + currency deep + progress tracker |
| `transformations/` | 14 | ~550 | 8 transformation ops + cleaning/imputation/field_ops extended + deep coverage |
| `metrics/` | 12 | ~280 | 3 metric ops + `BaseMetricsOp` + risk_scoring + predicted_utility_scoring |
| `fake_data/` | 4 | 101 | `FakeEmailOperation`, `FakeNameOperation`, `FakeOrganizationOperation`, `FakePhoneOperation` |
| `io_readers/` | 4 | 172 | `read_csv`, `read_json`, `read_excel`, `read_parquet` (via DataCSV/JSON/Excel/Parquet) |
| `errors/` | 4 | 160 | `BasePamolaError`, `auto_exception`, `ErrorHandler`, `ErrorCode` registry |

### Tier 2 — Framework Core (~66 files, ~1,936 tests)

| Directory | Files | Tests | What it tests |
|-----------|-------|-------|---------------|
| `utils/ops/` | 13 | ~450 | `BaseOperation` (run branches, save_config, scope, field/dataframe ops), `OperationConfig`, `DataSource` (classmethods, internal paths, file reading, gaps), `DataWriter`, `OperationResult` |
| `utils/tasks/` | 14 | ~500 | `BaseTask`, `TaskConfig` + extended coverage |
| `utils/io_helpers/` | 6 | ~250 | `crypto_utils`, `directory_utils`, `crypto_router` + extended/gaps coverage |
| `utils/crypto_helpers/` | 3 | ~80 | `key_store`, `crypto_router` gaps |
| `utils/` (io) | 4 | ~150 | `io.py` facade, dask/progress paths, extended paths |
| `utils/nlp/` | 19 | ~200 | NLP base, cache, tokenization, entity extraction |
| `privacy_models/` | 5 | 173 | `KAnonymityProcessor`, `LDiversityCalculator`, `TCloseness`, `DifferentialPrivacyProcessor` |
| `attacks/` | 7 | 181 | `LinkageAttack`, `MembershipInference`, `AttributeInference`, `DistanceToClosestRecord`, `NearestNeighborDistanceRatio` |
| `cli/` | 5 | 156 | 4 CLI commands (`list-ops`, `run`, `schema`, `validate`) via `CliRunner` |
| `common/` | 3 | 131 | `EncryptionMode`, enums, `DataHelper` type detection |
| `configs/` | 3 | 135 | `Settings`, `config_variables`, `field_definitions` |
| `catalogs/` | 1 | 24 | `get_operations_catalog()`, `get_operation_entry()` |

---

## CI Artifacts

CI pipeline (`.github/workflows/ci.yml`) generates artifacts on every run:

| Artifact | Format | Path | Description |
|----------|--------|------|-------------|
| JUnit XML | `.xml` | `reports/junit.xml` | Test results for CI dashboards, PR annotations |
| Coverage HTML | `.html` | `reports/coverage-html/` | Browsable coverage report per file/line |
| Coverage term | terminal | stdout | Summary table printed in CI logs |

Artifacts are uploaded as `test-reports-py{version}` with 30-day retention.
Download from GitHub Actions → workflow run → Artifacts section.

**Coverage scope** is defined in `.coveragerc` — 73 Public API files (18,896 statements).

---

## Test Conventions

### Naming
- **File:** `test_<source_module_name>.py`
- **Class:** `TestInit`, `TestExecution`, `TestErrors`, `TestEdgeCases`
- **Function:** `test_<feature>_<scenario>`

### Required Test Categories
1. **Initialization** — default params, custom params, invalid params
2. **Success path** — normal execution with valid data
3. **Error handling** — invalid input, missing fields, data source errors
4. **Edge cases** — empty df, single row, null values, special chars
5. **Metrics/artifacts** — verify metrics collected, artifacts generated

### Mocking Rules
- **DO mock:** External I/O, visualization (matplotlib/plotly), cache, network
- **DO NOT mock:** Internal business logic, operation execution, DataFrame operations
- Use `monkeypatch` or `@patch` for external deps only

### Common Fixtures

```python
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
    })

@pytest.fixture
def task_dir(tmp_path):
    return tmp_path

@pytest.fixture
def dummy_data_source(sample_df):
    class DummyDataSource:
        def __init__(self, df):
            self.df = df
            self.encryption_keys = {}
            self.encryption_modes = {}
            self.settings = {}
            self.data_source_name = "main"
        def get_dataframe(self, dataset_name, **kwargs):
            return self.df, None
        def apply_data_types(self, df, dataset_name=None, **kwargs):
            return df
    return DummyDataSource(df=sample_df)
```

---

## Known Patterns

| Issue | Solution |
|-------|----------|
| `numpy.bool_` vs `bool` | Use `== True` not `is True` |
| `numpy.int64` vs `int` | Use `isinstance(x, (int, np.integer))` |
| `dtype_backend='numpy_nullable'` | Expect `Int64/Float64/StringDtype` not `int64/float64/object` |
| Rich Console not captured by CliRunner | Only assert exit codes + `typer.echo()` output |
| `DataCSV/DataExcel.write()` default `index=True` | Use `index=False` in roundtrip tests |
| Privacy model errors caught internally | Don't `pytest.raises()`, assert return dict |
| `execution_time` can be 0.0 on fast machines | Assert `>= 0` not `> 0` |
| `DummyDataSource` needs `apply_data_types` | Add method returning df unchanged |
| Schema validation catches before custom errors | Use `pytest.raises((CustomError, Exception))` |

---

## DO NOT

- Create tests for Tier 3 internal helpers
- Use `unittest.TestCase` (use pytest style)
- Skip tests without documented reason
- Use hardcoded file paths (use `tmp_path`)
- Mock internal logic
- Ignore failing tests to pass CI
- Leave auto-generated `.md` files in test directories

---

## Sync Prompt

Run after code changes to keep tests in sync:

```
Please run the tests sync prompt at plans/prompts/prompt-update-tests-pamola-core.md
```
