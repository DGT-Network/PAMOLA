# CC Note — Test & Coverage Baseline

**Date:** 2026-08-16
**Author:** CC (Claude Code)
**Purpose:** establish the numbers that audit finding F-06 needs before quality
gates can be made blocking. Closes F-13 (environment not provisioned).

---

## Headline

| | |
|---|---|
| Tests collected | **5 477** |
| Result (clean environment) | **5 477 passed, 0 failed** |
| Result (this machine, as-is) | 5 472 passed, **5 failed** — environment leakage, see §2 |
| Wall time | 42 min 51 s (`dev3` deps) / 41 min 07 s (`dev4` deps) — single process, Windows, Python 3.11.0 |
| Verified on `1.0.0.dev4` dependency set | **5 477 passed, 0 failed** — see §7 |
| Coverage (public-API scope) | **85.13 %** — 18 962 statements, 2 820 missed |
| Files in coverage scope | 63 |

**The project's advertised "85 % test coverage" is accurate.** It was worth
verifying rather than assuming, and it holds. Note precisely what it measures:
the 63 files enumerated in `.coveragerc` `[run] include` — the declared public
API per SRS 4.1.11 — not the 537-module, ~202 k-LOC tree. Both numbers are
real; they answer different questions. See §4.

---

## 1. Environment

```
Python 3.11.0 (MSC v.1933 64-bit)
python -m venv .venv && pip install -e ".[test]"
```

Install size: **2 363 MB**. `torch` alone accounts for roughly 2.5 GB of
download — and is removed in `1.0.0.dev4` (PR #101) because nothing imports it.

Command:

```bash
python -m pytest tests/ -q --tb=line --cov --cov-report=term --cov-report=json
```

---

## 2. The 5 failures are environment leakage, not defects

All five failures share one cause:

```
tests/utils/tasks/test_project_config_loader.py::TestFindProjectRoot::test_find_project_root_gitpython
tests/utils/tasks/test_task_config.py::TestTaskConfigInitNoProgress::test_project_root_set
tests/utils/tasks/test_task_config.py::TestPathApiMethods::test_get_project_root
tests/utils/tasks/test_task_config.py::TestResolveLegacyPath::test_relative_path_resolves_to_project_root
tests/utils/tasks/test_task_registry.py::TestTaskConfig::test_path_resolution
```

```
AssertionError: assert 'D:\\VK\\_DEVEL\\HHR' == WindowsPath('.../pytest-175/test_get_project_root0')
```

The tests build a config under `tmp_path`, but resolution returned the path of
an **unrelated project on the developer's machine**. Cause: the ambient
environment variable

```
PAMOLA_PROJECT_ROOT=D:\VK\_DEVEL\HHR
```

left over from the predecessor HHR project. `pamola_core/utils/paths.py:45`
gives that variable top priority — which is correct, documented product
behaviour.

Proof it is the whole cause — same five tests, same machine, variable cleared:

```
$ Remove-Item env:PAMOLA_PROJECT_ROOT
$ pytest <the five tests> -q
5 passed in 11.65s
```

### The real finding: the suite is not hermetic

The product behaviour is right. The **tests** are wrong to permit it: a test
that constructs its own config in `tmp_path` must not be overridable by
machine-level state. Consequences:

- CI is green (the variable is unset there), so this never surfaces upstream;
- any developer who has ever worked on HHR — or who sets `PAMOLA_PROJECT_ROOT`
  for real work, which is its intended use — gets five confusing failures that
  look like code defects;
- it is silent: nothing tells you the environment is interfering.

**Proposed fix** (small, foundational, belongs before any coverage push):
a root `tests/conftest.py` with an autouse fixture that neutralises ambient
PAMOLA configuration for every test —

```python
@pytest.fixture(autouse=True)
def _isolate_pamola_env(monkeypatch):
    monkeypatch.delenv("PAMOLA_PROJECT_ROOT", raising=False)
```

Tests that *want* to exercise the variable set it explicitly. Note there is
currently **no root `conftest.py`** at all — only `tests/cli/conftest.py` and
`tests/privacy_models/conftest.py` — so there is no shared place where such an
invariant could live. That gap is worth closing on its own merits.

Registered as **TD-PC-11**.

---

## 3. Coverage within the declared public-API scope

85.13 % over 63 files. Weakest members:

| Coverage | Statements | File |
|---:|---:|---|
| 74.5 % | 415 | `anonymization/generalization/categorical_op.py` |
| 75.1 % | 610 | `anonymization/suppression/cell_op.py` |
| 75.3 % | 733 | `profiling/analyzers/currency.py` |
| 77.6 % | 375 | `profiling/analyzers/email.py` |
| 78.0 % | 123 | `metrics/operations/utility_ops.py` |
| 78.2 % | 449 | `anonymization/generalization/numeric_op.py` |
| 78.8 % | 364 | `anonymization/suppression/attribute_op.py` |
| 79.9 % | 536 | `utils/tasks/base_task.py` |
| 80.2 % | 430 | `profiling/analyzers/correlation.py` |
| 81.0 % | 374 | `profiling/analyzers/date.py` |

### Recommendation for F-06 (`fail_under`)

Set **`fail_under = 84`** now. One point below the measured 85.13 %: tight
enough to catch a real regression, loose enough to absorb ordinary noise
without blocking unrelated work. Raise it as coverage rises — a ratchet, not a
target negotiated once.

Setting it *at* 85 would make the very next merge a coin flip; setting it at
70 would defend nothing.

---

## 4. The other number: whole-tree gaps

The public-API scope is 63 files. The tree is 537 non-`__init__` modules.
Counting modules that no test references by import:

| Package | Modules | Untested | LOC | Untested LOC | Gap |
|---|---:|---:|---:|---:|---:|
| utils | 132 | 71 | 68 308 | 33 617 | 49 % |
| anonymization | 85 | 65 | 36 385 | 19 844 | 55 % |
| profiling | 93 | 75 | 34 688 | 14 815 | 43 % |
| fake_data | 42 | 32 | 15 298 | 9 119 | 60 % |
| privacy_models | 18 | 12 | 8 756 | 6 527 | **75 %** |
| metrics | 44 | 26 | 10 580 | 4 207 | 40 % |
| transformations | 48 | 34 | 16 553 | 3 307 | 20 % |
| common | 27 | 19 | 2 161 | 1 232 | 57 % |
| attacks | 9 | 2 | 1 470 | 742 | 50 % |
| errors | 15 | 6 | 3 783 | 581 | 15 % |
| cli | 6 | 3 | 618 | 436 | 71 % |
| analysis | 6 | 1 | 1 418 | 79 | 6 % |
| io | 7 | 2 | 1 108 | 44 | 4 % |
| **total** | **537** | **349** | **202 479** | **94 555** | **47 %** |

(Method: a module counts as referenced if its dotted path or module stem
appears in an import in any test file. This over-counts — a referenced module
is not necessarily a *tested* one — so the true gap is wider, not narrower.)

**"Total coverage" is a programme of work, not a task.** 349 modules and
~95 k LOC currently have no test that so much as imports them. Stating that
plainly is more useful than a plan that implies it can be closed in a sprint.

### Proposed order

1. **`tests/conftest.py` isolation fixture** (§2). Without it, added tests
   inherit the same non-hermeticity.
2. **`privacy_models`** — 75 % gap, and it is the package that carries the
   actual privacy guarantees (k-anonymity, l-diversity, t-closeness, DP
   calculation). Highest risk-per-line in the tree.
3. **`utils/ops`** — `op_base`, `op_config`, `op_data_source`, `op_result`.
   Everything else depends on it, so a defect here is systemic. Already
   partially covered (`op_base` 96 %, `op_result` 93 %); finish it.
4. **Integration layer on `_DATARAW`** — see §5.
5. Then breadth: `fake_data`, `profiling/commons`, `anonymization/commons`.

Deliberately *last*: `utils/vis_helpers` (~10 k LOC of plotting) — high LOC,
low defect consequence, expensive to assert meaningfully.

---

## 5. Where `_DATARAW` fits

Current tests build their DataFrames inline and read nothing from `data/`.
**For unit tests that is the correct design** and should not change.

The synthetic corpora are valuable for a *second, separate* layer:
every dataset ships a `*_passport.json` declaring the
`direct` / `quasi` / `sensitive` categorisation of every column — i.e. machine-
readable **ground truth**. That is exactly what is needed to assert things unit
tests cannot:

- k-anonymity/l-diversity computed over the declared quasi-identifiers matches
  the passport's expectation;
- an anonymization op actually removes re-identifiability, measured by the
  `attacks/` package rather than asserted;
- profiling correctly classifies a column whose true category is known.

This is an **addition** to the unit suite, not a route to "total coverage" —
integration tests are slow and few by nature. It should be marked
(`@pytest.mark.integration`), excluded from the default run, and pointed at the
small fixtures rather than the `*_10k` files.

---

## 6. The suite writes into the repository working tree

Running `pytest tests/` leaves 43 untracked files behind, in ten locations —
including one **inside the installed package directory**:

```
?? config.json                      (repo root)
?? my_task/                         2 files
?? output/                          8 files
?? pamola_core/utils/resources/     1 file   ← inside the package
?? pamola_datasets/                 2 files
?? task/                            2 files
?? temp_task_dir/                   1 file
?? test_task_dir/                  22 files
?? test_tmp_dir/                    3 files
?? test_vis_dir/                    2 files
?? configs/execution_log.json                ← inside a tracked directory
?? configs/test_task_001.json                ← inside a tracked directory
```

Contents are ordinary operation artifacts written to a relative path instead of
`tmp_path`, e.g.
`test_task_dir/output/mvf_field_MVFOperation_values_dictionary_output_20260816_145316.csv`.

Three consequences, in increasing seriousness:

1. **`git status` is dirty after every run**, so real changes are easy to miss
   and easy to commit by accident. None of these paths is in `.gitignore`.
2. **Timestamped filenames accumulate** — two runs produced two copies of the
   same dictionary. Nothing cleans up.
3. **Two tracked directories are written into.** `pamola_core/utils/resources/`
   is created inside the package, and `configs/` — which holds the committed
   `prj_config.json` / `prj_config.yaml` / task configs — receives
   `execution_log.json` and `test_task_001.json`. Both are worse than stray
   files in the root:
   - a test writing into the *code* tree can mask a packaging bug, because a
     resource the wheel fails to ship still resolves locally from what a
     previous run created. Given that this very release fixes missing package
     data (P-01), that is not hypothetical;
   - a test writing into `configs/` is one filename collision away from
     overwriting a committed project configuration.

Note also that the run is single-process and takes **43 minutes**; writing to
fixed relative paths is precisely what blocks `pytest -n auto`, since parallel
workers would collide in `test_task_dir/`.

**Proposed fix:** route these through `tmp_path` / `tmp_path_factory`. As an
interim guard, add the ten paths to `.gitignore` so nothing gets committed by
accident — but the guard is not the fix, and adding it without the fix risks
making the problem permanent.

Registered as **TD-PC-12**.

---

## 7. Verification of the `1.0.0.dev4` dependency set — and what it caught

PR #101 removes eight unused dependencies. The first test run above does **not**
verify that change: the virtualenv was built before the edit and still contained
`torch` and the rest. A second, clean virtualenv was built from the edited
`pyproject.toml`.

**It failed immediately — 170 collection errors:**

```
pamola_core/utils/env.py:7: from dotenv import load_dotenv
ModuleNotFoundError: No module named 'dotenv'
```

`import pamola_core` did not work at all.

### Root cause: undeclared dependencies, not the removals

The static reference count behind the removals was correct — nothing imports
`uvicorn`. What it could not see is what `uvicorn` *brought with it*.
`uvicorn[standard]` depends on `python-dotenv`, and `torch` depends on
`filelock`. Both are imported at **module scope** by `pamola_core`, and neither
was ever declared.

An AST audit of every import in the package against the declared list found
**thirteen** undeclared third-party modules, eight of them at module scope:

| Module | Imported at | Was supplied by |
|---|---|---|
| `python-dotenv` | `utils/env.py:7` | `uvicorn[standard]` **only** |
| `filelock` | `utils/tasks/context_manager.py:37` | `torch` **only** |
| `joblib` | `anonymization/commons/processing_utils.py:23` (+18) | scikit-learn |
| `rich` | `cli/commands/list_ops.py:12` | typer |
| `tqdm` | `utils/progress.py:67` | nltk / spacy |
| `packaging` | `utils/ops/op_registry.py:37` | many |
| `jinja2` | `utils/reporting/template_engine.py:14` | bokeh / dask |
| `regex` | `utils/nlp/minhash.py:30` | spacy / nltk |

This is a larger finding than the one that exposed it. The package's
importability rested on transitive luck: any upstream dropping one of these
would have broken `import pamola_core`, and `pyproject.toml` gave no signal.
The removals did not create the fragility — they revealed it.

`requests`, `transformers`, `graphviz`, `pyage` and `git` are also imported but
only lazily, inside guarded optional-capability branches, and are correctly left
undeclared.

All eight module-scope modules are now declared explicitly in PR #101, each with
its import site in a comment. Dependency count: 46 → 36 (removals) → **44**.

### Result after the fix

```
$ pytest tests/ -q          # clean venv, PAMOLA_PROJECT_ROOT unset
5477 passed, 114 warnings in 2467.61s (0:41:07)
```

Install size: **954 MB**, against **2 363 MB** for the `dev3` set — a 1 409 MB
(60 %) reduction, with no test regression.

**Method note for future dependency work:** counting references to a package
answers "does anything import it", which is not the question that matters when
removing it. The question is "what did it supply". Only a clean-environment
install and a full run answer that.

---

## 8. Warning-filter caveat

`pytest.ini` suppresses 15 warning categories, including the blanket
`ignore::RuntimeWarning:numpy.*`. In a numerical privacy library, NumPy
`RuntimeWarning` is the channel through which overflow, divide-by-zero, and
invalid-value conditions announce themselves in a metric computation.
Suppressing the whole category trades a noisy suite for silently wrong numbers.
Narrow it to the specific messages that are genuinely noise.

The run produced 114 warnings *after* those filters.

---

## GCA Evidence Package

| Item | Status | Evidence |
|---|---|---|
| Test suite executes end to end | PASS | 5 477 collected; 5 472 passed / 5 failed on this machine; 5 477 passed with `PAMOLA_PROJECT_ROOT` unset |
| Public-API coverage = 85 % as advertised | PASS | `--cov-report=json`: 18 962 statements, 2 820 missing, 85.13 % over the 63 `.coveragerc` files |
| 5 failures are environment, not code | PASS | `pamola_core/utils/paths.py:45`; five named tests pass in 11.65 s with the variable cleared |
| Baseline exists for `fail_under` | PASS | recommendation: 84 (§3) |

*Refs: `docs/output/20260816_CC_REPO_AUDIT.md` F-06, F-13; `DEVLOG.md` TD-PC-05.*
