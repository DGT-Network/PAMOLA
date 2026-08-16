# CC Repository Audit — PAMOLA.CORE

**Date:** 2026-08-16
**Author:** CC (Claude Code)
**Audience:** Val, CW, Titan team
**Scope:** repository hygiene, packaging, CI/CD, quality gates, security posture,
documentation. **Not** a functional audit of anonymization algorithms.
**Method:** static inspection of the working copy. **No tests were run** — the
local environment is not provisioned (see F-13). Every claim below is anchored
to a file, line, or git object.

**Snapshot:**

| | |
|---|---|
| Remote | `https://github.com/DGT-Network/PAMOLA` (public) |
| Default branch | `develop` @ `79a9f1c` (2026-06-01) |
| Stable branch | `main` @ `5fd64cc` (2026-07-09) |
| Version | `1.0.0.dev3` (PyPI `pamola-core`) |
| Tags | `v1.0.0.dev1`, `v1.0.0.dev2`, `v1.0.0.dev3` |
| Package files | 634 under `pamola_core/` (618 `.py`) |
| Test modules | 221 |
| Python LOC (incl. tests) | ~286 000 |
| Working tree | ~152 MB |
| Contributors | 12; top: `thuan.dang` 160, `Hoang Ha Huy` 115, `lannguyen53` 58, `ValKhv` 52 |
| Open PRs / issues | 0 / 0 (issues appear disabled) |

**Verdict:** the library itself is substantial and well-structured — a clear
package taxonomy, a real operation lifecycle, a structured error framework, a
CLI, and a test suite that mirrors the package layout. The problems are
**around** the code: branch hygiene, packaging metadata, dependency policy,
and quality gates that measure without enforcing. None of the findings block
day-to-day development; several block a credible `1.0.0` stable release.

---

## Findings

Severity: **High** = blocks a stable release or actively costs the team.
**Medium** = real risk, schedulable. **Low** = hygiene.

---

### F-01 · High · `main` and `develop` have diverged and were never reconciled

> **ACTIONED 2026-08-16.** Crucial detail found after the first revision:
> `git diff --stat origin/main origin/develop` produces **no output** — the two
> trees are byte-identical. The divergence is purely *historical* (a squash
> merge into `main` that was never merged back), not a content split. That
> makes reconciliation a no-content-change merge commit rather than a conflict
> resolution. Done via PR — see the *Actions Taken* section.
> The stale-branch half of this finding remains open as F-01b / Q-04.

`git rev-list --left-right --count origin/main...origin/develop` → `1  26`.

- `main`-only: `5fd64cc` — *"feat: pamola-core 1.0.0.dev3 — pseudonymization
  refactor & security fixes (#98)"* (2026-07-09). This is the squash-merge of
  PR #98 into `main`.
- `develop` is 26 commits ahead of the merge base, including
  `79a9f1c` *docs(readme): rewrite…* and `33ca9b6`-era work.

So the two branches hold the same *content* by different *history*, and
nothing has merged `main` back into `develop`. Every future release PR will
show a spurious conflict surface, and `git branch --contains` checks in
`release.yml` will keep drifting.

There are also **26 remote branches**, many long-lived and stale
(`json_dynamic`, `json_dynamic_2`, `json_dynamic_pamola_core`,
`json_dynamic_pamola_core_2`, `json_dynamic_check_update_schema`,
`json_dynamic_fix_transformation`, `json_dynamic_transformation`,
`happypath-20260518`, `hot_fix_run_operation`, two `claude/*` agent branches).

**Impact:** release process fragility; unclear which branch is authoritative
for a given fix.

**Proposal:** merge `main` into `develop` once (`-s ours` if content is
already identical, with a message explaining why), then prune the `json_dynamic*`
family after confirming with their authors. Decide and document whether the
project is really running git-flow or should collapse to trunk + release tags.
**Owner:** Val + Titan. **Registered as TD-PC-01.**

---

### F-02 · Low (was High) · ~120 MB of synthetic CSV datasets are tracked in git

> **RESOLVED 2026-08-16 — the data is fully synthetic, and provenance is
> documented in-repo.** The first revision of this finding asked whether the
> datasets were real and stated that `data/raw/` had no provenance file. That
> was wrong: `data/raw/README.md`, a per-dataset `readme.md`, and a
> `*_passport.json` per file are all present and tracked. See the *Provenance*
> subsection below. Severity drops from High to Low; the finding is now purely
> about **published surface area**, not legal or privacy exposure.

#### Provenance (evidence)

Every dataset carries a machine-readable passport, e.g.
`data/raw/med_ehr/EHR_FHIR_LITE_OUTPATIENT_passport.json`:

```json
{
  "dataset_name": "PAMOLA_EHR_FHIR_LITE_OUTPATIENT",
  "records": 10000,
  "generated_at": "2025-08-12T00:05:56.671925Z",
  "privacy_categories": { "direct": [...], "quasi": [...], "sensitive": [...] }
}
```

`data/raw/README.md` carries an explicit legal notice: *"All datasets in this
directory are artificially generated… No real personal information (PII/PHI) is
included. Not collected from any external sources. Not derived from any real
customer, employee, or patient datasets."*

Generation is programmatic and partly reproducible in-repo —
`data/raw/med_rare_lab/generate_med_rare_lab.py` and
`data/raw/bank_txs/generate_bank_txs_v2.py` are committed generators. The
upstream generation workspace is `D:\VK\_DEVEL\_DATARAW` (not published),
whose `README.md` documents the method (Faker + domain-specific generators +
statistical distributions matched to real-world shapes without real data) and
the **synthetic fingerprinting** scheme — intentional distribution and
identifier markers embedded at generation time so that a dataset cannot be
misrepresented as real and so lineage can be traced through pipelines.

`_DATARAW` also holds genuinely third-party corpora (`MIMIC-III`,
`ENRON_EMAIL_DATASET`, `NYC_Taxi_Trips_Dataset`, `UCI_Adult_Dataset`,
`ieee-cis-fraud-detection`, `ai4privacy`, `nvidia_NEMOTRON_PII`, …).
**None of these are in the public repository** — only the PAMOLA-generated
families were copied in. That separation is correct and should be stated as a
rule, not left as a happy accident (see proposal below).

#### What remains open: surface area

The provenance question is closed; the size and exposure questions are not.

| File | Size |
|---|---|
| `data/raw/hr_job_resume/HR_RESUMES_CA_V2_10k.csv` | 23.6 MB |
| `data/raw/crossparty_anti-fraud/BANK_ANTIFRAUD.csv` | 14.9 MB |
| `data/raw/crossparty_anti-fraud/PAYMENT_ANTIFRAUD.csv` | 12.8 MB |
| `data/raw/med_ehr/EHR_FHIR_LITE_OUTPATIENT_10k.csv` | 11.0 MB |
| `data/raw/hr_job_resume/HR_RESUMES_SAMPLE_CA_10k.csv` | 10.5 MB |
| `data/raw/crossparty_anti-fraud/TELECOM_ANTIFRAUD.csv` | 9.8 MB |
| … 6 more ≥ 2.9 MB | |

Three reasons to reduce the published set even though the data is safe:

1. **Clone cost.** Every contributor and every CI checkout pays ~150 MB for
   fixtures that unit tests do not need at 10k rows.
2. **Signal.** A 10k-row synthetic EHR is a *demo* asset, not a *test* asset.
   Tests want 50–200 rows with known ground truth; demos want the full file.
   Shipping both through the same channel serves neither well.
3. **Misuse risk.** The datasets are deliberately realistic and deliberately
   fingerprinted. Fingerprinting exists precisely because someone might
   present them as real — a public, frictionless download makes that easier.
   The fingerprints detect misrepresentation after the fact; reducing surface
   prevents some of it.

Note also `data/raw/crossparty_anti-fraud/LICENSE` is **Apache-2.0**, sitting
inside a BSD-3-Clause repository, with no explanation of which terms govern
the datasets overall.

**Proposal (needs Val's decision, nothing done yet):**

- Keep in-repo: the small fixtures the test suite actually loads
  (`S_CHURN_BANK_CANADA_50.csv/.json`, `HR_RESUMES_SAMPLE_CA.csv`,
  the `*_dictionary.csv` files), **all** `*_passport.json`, all `readme.md`,
  and the committed generator scripts. These are cheap and are the part that
  makes the claim "synthetic" auditable.
- Move out of the working tree: the `*_10k` / full-size CSVs → GitHub Release
  assets or a `pamola-data` companion repo, referenced by a downloader in
  `data/raw/README.md`. History rewrite is **not** proposed — the blobs stay in
  history; only future clones of the working tree get lighter, and the
  decision can be revisited.
- Add `data/raw/POLICY.md` stating the rule that made the current state safe:
  *only PAMOLA-generated synthetic families are published; third-party corpora
  stay in `_DATARAW` and are never committed.*
- Resolve the Apache-2.0 / BSD-3 licence question for `data/`.

**Registered as TD-PC-02 (severity Low, decision pending).**

---

### F-02b · Withdrawn · Original "unknown dataset provenance" concern

The first revision of this report asked whether `data/raw/` might contain real
personal data and asserted that no provenance file existed. **Both parts were
wrong**, and the error was mine: I inspected file sizes and inferred risk from
directory names (`HR_RESUMES_CA`, `EHR_FHIR_LITE_OUTPATIENT`, `BANK_ANTIFRAUD`)
without opening `data/README.md`, `data/raw/README.md`, the per-dataset
`readme.md` files, or the `*_passport.json` passports — all of which are
tracked and all of which state the answer plainly.

**Standing answer, for the record:** the datasets are **fully synthetic**,
programmatically generated (Faker + domain generators + statistical
distributions), fingerprinted for lineage, and documented per file. No real
PII/PHI. See F-02 for the evidence.

Two observations from the original finding do survive, and are folded into
F-02 above: the ~120 MB clone cost, and the fact that the datasets entered the
repo on 2026-01-10 via an unrelated commit
(`git log -1 -- data/raw` → `ValKhv`, *"Revise PSI/VFL Protocols diagram in
README"*) rather than a deliberate data-publication commit — which is why the
provenance material, though present, is easy to miss.

`MANIFEST.in` does not ship `data/`, so the PyPI wheel is unaffected either
way: this is a repository cost, never a package cost.

---

### F-03 · High · Every runtime dependency is hard-pinned with `==`

`pyproject.toml:27-72` — 46 dependencies, all `==`:

```
torch==2.8.0            sdv==1.18.0           spacy==3.8.9
dask[complete]==2025.11.0   faiss-cpu==1.12.0     numpy==1.26.4
pandas==2.2.2           scikit-learn==1.7.2   nltk==3.9.2
uvicorn[standard]==0.38.0   cryptography==46.0.3  …
```

Three separate problems:

1. **`==` pins in a library.** Applications pin; libraries constrain. Any
   consumer that also depends on `numpy`, `pandas`, `torch`, or `scikit-learn`
   at a different version gets an unsatisfiable resolution. **BEST is exactly
   such a consumer** — it installs `pamola-core` alongside Ray, Polars, and its
   own pinned stack. `numpy==1.26.4` in particular locks consumers out of the
   NumPy 2.x line entirely.
2. **Weight.** `torch`, `sdv`, `spacy`, `faiss-cpu`, `dask[complete]`,
   `matplotlib`, `seaborn`, `plotly`, `kaleido`, `wordcloud`, `nltk`,
   `fasttext-wheel` are all **mandatory**. `pip install pamola-core` is a
   multi-GB, multi-minute download for a user who wants to mask one column.
3. **`uvicorn[standard]` is a runtime dependency of a file-processing
   library** (`pyproject.toml:40`). Nothing in a library that has no server
   should need an ASGI server. This looks like a leftover.

**Proposal:** split into a small mandatory core (`pandas`, `numpy`, `pyarrow`,
`pydantic`, `PyYAML`, `typer`, `cryptography`, `jsonschema`, `psutil`,
`chardet`) plus extras — `[nlp]` (spacy, nltk, fasttext, langdetect),
`[synth]` (sdv, torch), `[viz]` (matplotlib, seaborn, plotly, kaleido,
wordcloud, matplotlib-venn), `[bigdata]` (dask), `[all]`. Relax `==` to
`>=x.y,<next-major`. Drop `uvicorn` unless a caller is found.
This is a coordinated change with BEST — CW should own the split.
**Registered as TD-PC-03.**

---

### F-04 · Medium · Placeholder identity is published to PyPI

`pyproject.toml:6-11`:

```toml
authors    = [{ name = "Author",     email = "author@example.com" }]
maintainers = [{ name = "Maintainer", email = "maintainer@example.com" }]
```

Meanwhile `pamola_core/__init__.py:7` correctly states
*"(C) 2024 Realm Inveo Inc. and DGT Network Inc."*

Also missing from `[project]`: `keywords`, `[project.urls]` (Homepage,
Repository, Documentation, Changelog), `Development Status` and
`Intended Audience` / `Topic` classifiers. The PyPI page for a
privacy-engineering library currently has no link back to the repo or docs.

**Impact:** cosmetic but public — this is the first thing an evaluator sees.
**Proposal:** one-line fix, bundle into the next release prep.
**Registered as TD-PC-04.**

---

### F-05 · Medium · CI installs a dependency extra that does not exist

`.github/workflows/ci.yml:49` and `release.yml` (same line in the test job):

```bash
pip install -e ".[dev]" 2>/dev/null || pip install -e .
```

`pyproject.toml:74-78` defines only a `test` extra. There is no `dev` extra —
it was deliberately removed (`chore: remove dev optional-dependencies group,
keep test only`, 2026-04-06), but the workflow was not updated.

So the first command always fails, its error is discarded by `2>/dev/null`,
and the fallback runs. `pytest`/`pytest-cov` are then installed *unpinned* on
the next line, ignoring the `pytest==8.4.2` pin in the `test` extra.

**Impact:** CI silently ignores the project's own test-tool pins; a future
breaking pytest release lands in CI unannounced. The `|| true`-style pattern
also hides genuine install failures.

**Proposal:** replace both occurrences with `pip install -e ".[test]"` and
delete the follow-up `pip install pytest pytest-cov`.
This is a two-line, zero-risk fix — **recommend doing it first.**

---

### F-06 · Medium · Quality gates measure but do not enforce

Three independent instances of the same pattern:

| Gate | Location | Behaviour |
|---|---|---|
| Lint | `ci.yml:31,34` | Blocking run is `--select E9,F63,F7,F82` (syntax errors + undefined names only). The full run is `\|\| true`. |
| Coverage | `.coveragerc:[report] fail_under = 0` | Coverage is computed and reported; nothing ever fails. |
| Types | — | No mypy job, no `mypy` in any extra, despite type hints throughout the codebase. |

There is also no `[tool.ruff]` section anywhere, so the "full" ruff run uses
defaults rather than a project style, which is why it has to be non-blocking.

The dev2 commit message advertises *"85% test coverage"* — that number is
currently unverifiable from CI config alone and is not defended against
regression.

**Impact:** the repo *looks* gated and is not. A refactor that drops public-API
coverage from 85% to 40% passes CI green.

**Proposal, in order of value per unit of effort:**
1. Set `fail_under` to the current measured number minus a small margin
   (run coverage once locally to establish the real baseline — see F-13).
2. Add `[tool.ruff]` with the rule set the team actually agrees to, then move
   the full ruff run to blocking.
3. Add a `mypy` job in non-blocking mode first, tighten later.

**Registered as TD-PC-05.**

---

### F-07 · Medium · `.coveragerc` `include` list is hand-maintained public-API scope

`.coveragerc` enumerates ~55 files as *"Public API only (per SRS 4.1.11)"*,
grouped by package with hand-written counts in comments
(*"anonymization: 10 operation classes"*, *"profiling: 14 operation classes"*).

This is a thoughtful design — coverage is measured against declared public
surface, not against 286k LOC. But it is a **second copy** of the export list
in `pamola_core/__init__.py` `__all__`, maintained by hand, with no check that
the two agree.

**Failure mode:** a new operation is exported from `__init__.py` but not added
to `.coveragerc` → it is public, untested, and invisible to the coverage
number, which may even *rise*.

**Proposal:** add a single test — `tests/test_public_api_coverage_scope.py` —
that resolves every name in `__all__` to its defining module and asserts the
module path appears in `.coveragerc` `[run] include`. ~30 lines, closes the
drift permanently. This is a good first CC task.

---

### F-08 · Medium · Unsandboxed `eval()` on configuration-supplied strings

Three call sites, of differing severity:

1. **`pamola_core/transformations/field_ops/add_modify_fields.py:570`** —
   ```python
   batch[output_field_name] = batch[base_on_column].apply(
       lambda x: eval(expression.replace(expression_character, str(x)))
   )
   ```
   `expression` comes straight from `field_config` (the task config file).
   No globals restriction, no AST validation. A task JSON can execute
   arbitrary Python. Worse, the value is interpolated by **string
   replacement** into the expression before evaluation, so *data* can inject
   code too if `base_on_column` holds attacker-influenced text.

2. **`pamola_core/profiling/commons/correlation_utils.py:90`** —
   `mvf_lamdable = eval(mvf_parser) if isinstance(mvf_parser, str) else …`
   `mvf_parser` is a caller-supplied string eval'd with full builtins.

3. **`pamola_core/transformations/commons/aggregation_utils.py:670,693`** —
   `eval(expr, Constants.SAFE_GLOBALS, {"row": row})`.
   This one **does** restrict globals. Much better; still worth reviewing what
   `SAFE_GLOBALS` contains, since Python's globals sandboxing is famously
   escapable via `__class__`/`__subclasses__` traversal unless `__builtins__`
   is explicitly emptied.

Related: **`pamola_core/fake_data/commons/mapping_store.py:679`** —
`pickle.load(f)` on a mapping-store file path. Loading a mapping store from an
untrusted location is arbitrary code execution by design of the format.

**Framing:** the threat model here is *config and artifact files*, not
network input. In a single-user notebook this is low risk. But BEST executes
pamola-core operations from **user-submitted task configurations** in a
multi-tenant control plane — there, sites 1 and 2 are a tenant-to-host escape.

**Proposal:**
- Sites 1 & 2: route through the `SAFE_GLOBALS` mechanism already present in
  `aggregation_utils`, or better, an AST allow-list evaluator. Stop
  string-interpolating values into the expression — bind them as names.
- `mapping_store`: document that pickle load is trusted-input-only, and add a
  JSON/Parquet serialisation path as the default for untrusted sources.
- Confirm with CW whether BEST ever exposes `modify_expression` or
  `mvf_parser` to tenant-supplied config. **If it does, this is High, not
  Medium.**

**Registered as TD-PC-06.**

---

### F-09 · Medium · MD5 and SHA-1 offered for identity and pseudonym hashing

Roughly 25 MD5 call sites. Most are legitimate non-security uses — cache
keys, artifact filenames, group keys — and one is correctly annotated
(`pamola_core/utils/ops/op_cache.py:383`, `# nosec B324`).

Two are not clearly benign:

- **`pamola_core/profiling/commons/identity_utils.py:25`** —
  `calculate_hash(values, algorithm="md5")`: MD5 is the **default** for
  identity-record hashing.
- **`pamola_core/fake_data/commons/utils.py:113-148`** — the pseudonym hash
  helper accepts `"md5"` and `"sha1"` alongside SHA-256/512.
- **`pamola_core/utils/ops/op_field_utils.py:663,725-726`** — same menu for
  field hashing (default is SHA-256, which is right).

In a privacy library, a *pseudonymisation* function that can be configured to
MD5 is a compliance finding waiting to happen — MD5 pseudonyms are trivially
reversible by rainbow table for low-entropy identifiers.

Note the 1.0.0.dev3 release already hardened this area well
(weak-salt rejection, pepper cache invalidation, SHA3-256/512 for the new
`HashBasedPseudonymizationOperation`). This finding is about the **older**
helpers that were not swept.

**Proposal:** keep MD5 where it is a cache/filename key and annotate each with
`usedforsecurity=False` (Python 3.9+) or a `# nosec` comment stating why.
Remove `md5`/`sha1` from the *pseudonymisation* and *identity* menus, or gate
them behind an explicit `allow_weak_hash=True`. Flip
`identity_utils.calculate_hash` default to `sha256`.

---

### F-10 · Medium · Missing OSS governance; PyPI publishes with a long-lived token

Absent from a public repository: `SECURITY.md`, `CODEOWNERS`,
`.github/dependabot.yml`, `.github/PULL_REQUEST_TEMPLATE.md`,
`CODE_OF_CONDUCT.md`. (`CONTRIBUTING.md` and `LICENSE` are present.)
GitHub Issues appear to be disabled — `gh issue list` errors, so there is no
public channel for a vulnerability report on a *security* library.

`.github/workflows/release.yml` publish job:

```yaml
env:
  TWINE_USERNAME: __token__
  TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
run: twine upload dist/*
```

- Long-lived PyPI API token rather than **Trusted Publishing (OIDC)**, which
  PyPI now recommends and which removes the stored secret entirely.
- The `publish` job has no GitHub **environment** with required reviewers, so
  anyone able to push a `v*` tag can publish to PyPI.
- No `twine check dist/*` before upload (catches broken README rendering).
- Actions are pinned by tag (`actions/checkout@v4`), not by SHA.

The release workflow's *validation* job, by contrast, is genuinely good —
tag↔branch, tag↔version, and version↔CHANGELOG are all enforced. Credit where
due.

**Proposal:** add `SECURITY.md` with a disclosure address and enable Issues or
name an alternative channel; migrate to Trusted Publishing; add a protected
`pypi` environment; add `twine check`.
**Registered as TD-PC-07.**

---

### F-11 · Low · IDE and editor state is tracked despite `.gitignore`

- `.idea/workspace.xml` is **tracked** (added 2025-02-27, *"Updated project
  structure"*) even though `.gitignore:60` ignores `.idea/`. `.gitignore` does
  not untrack files already in the index. *(Checked: the file contains no
  local absolute paths or usernames, so no information leak — just noise.)*
- `docs/.obsidian/` — Obsidian vault configuration, tracked.
- `site/` and `scripts/` contain nothing but a `__placeholder__` file each.
  `site/` is the conventional MkDocs *build output* directory; an empty
  tracked one invites a future accidental commit of generated HTML.

**Proposal:** `git rm --cached -r .idea docs/.obsidian`; add `site/` to
`.gitignore` and drop the placeholder, or delete `site/` and `scripts/` if
nothing is planned for them.

---

### F-12 · Low · Two documentation toolchains and three doc trees

The repo ships **both**:
- Sphinx — `docs/conf.py`, `docs/index.rst`, `docs/Makefile`, `docs/make.bat`,
  `docs/_static/`, `docs/_templates/`
- MkDocs — `mkdocs.yml`, `site/`

and three parallel content trees: `docs/en/`, `docs/ru/`, `docs/specs/`
(the latter with lettered folders `a.concept` … `o.visualization` plus
`CORE SPEC.md`), for 393 tracked files under `docs/`.

Meanwhile the *authoritative* framework documentation lives outside the repo,
in BOX (`C:\Users\valer\Box\DEVBOX\PAMOLA\core\` — `00_INDEX.md` through
`13_BEST_migration_strategy.md`).

**Impact:** unclear which tree a contributor should update; high odds that
`docs/en` and `docs/ru` have already drifted apart and from BOX.

**Proposal:** pick one builder (Sphinx is already wired into the dev2 release
per the CHANGELOG) and delete the other. Declare in `CONTRIBUTING.md` which
tree is source of truth and what the BOX↔repo relationship is.
**Registered as TD-PC-08.**

---

### F-13 · Low · Local environment not provisioned; test suite unverified

No `.venv` exists in the working copy and no test run has been performed
locally. Provisioning is non-trivial precisely because of F-03 — a full
install pulls `torch`, `sdv`, `spacy`, `faiss-cpu` and `dask[complete]`.

Consequently this audit **cannot state** the current test pass rate or
coverage percentage. Those numbers are needed before F-06 can be actioned
(you cannot set `fail_under` without a baseline).

Related observation — `pytest.ini` suppresses 15 warning categories, including
the broad `ignore::RuntimeWarning:numpy.*`. In a numerical privacy library,
NumPy `RuntimeWarning` is how you find out about overflow, divide-by-zero, and
invalid values in a metric computation. Suppressing the whole category
globally trades a noisy suite for silent wrong numbers.

**Proposal:** provision `.venv` and capture a baseline
(`pytest tests/ -q --cov --cov-report=term`) as the first CC task; narrow the
NumPy warning filter to the specific messages that are actually noise.

---

## Priority Ranking

| # | Finding | Severity | Effort | Owner | Status |
|---|---|---|---|---|---|
| — | F-02 dataset provenance | ~~High~~ | — | Val | **CLOSED — synthetic, documented** |
| — | F-01 reconcile `main`/`develop` | High | small | CC | **DONE 2026-08-16** (PR opened) |
| 1 | F-05 CI installs non-existent `[dev]` extra | Medium | trivial | CC | open |
| 2 | F-13 provision env, capture test/coverage baseline | Low | small | CC | open |
| 3 | F-08 `eval()` sandboxing (severity depends on BEST exposure) | Medium | medium | CC + CW | open, Q-02 |
| 4 | F-03 dependency split into extras | High | large | CW | open, Q-03 |
| 5 | F-06 make quality gates blocking | Medium | medium | CC | open |
| 6 | F-04 packaging metadata | Medium | trivial | CC | open |
| 7 | F-07 public-API ↔ coverage-scope drift test | Medium | small | CC | open |
| 8 | F-10 governance files + Trusted Publishing | Medium | small | Val | open |
| 9 | F-09 weak-hash menus in pseudonymisation helpers | Medium | small | CC | open |
| 10 | F-02 reduce published dataset surface | Low | medium | Val | open, Q-01 |
| 11 | F-01b prune 26 stale remote branches | Low | small | Val/Titan | open, Q-04 |
| 12 | F-11 untrack IDE state | Low | trivial | CC | open |
| 13 | F-12 consolidate documentation toolchain | Low | medium | Val | open, Q-05 |

---

## Question Register

Answered questions stay in the table with their answer and date — this is the
record, not a scratchpad.

| ID | Question | Status | Answer |
|---|---|---|---|
| **Q-00** | Are the `data/raw/` datasets real or synthetic? | **ANSWERED 2026-08-16 (Val)** | **Fully synthetic.** Generated in the `_DATARAW` workspace with Faker + domain generators + statistical distributions; fingerprinted for lineage; passports and legal notice committed alongside. No real PII/PHI. Third-party corpora (MIMIC-III, Enron, NYC Taxi, UCI Adult, …) live only in `_DATARAW` and are never published. |
| **Q-01** | Should the *published* dataset surface be reduced even though the data is safe? | **OPEN — leaning yes (Val)** | Val's position: *"не уверен, что сами данные, даже синтетические, стоит публиковать в публичном репозитории; возможно надо сократить поверхность."* Concrete proposal in F-02: keep small fixtures + passports + generators in-repo, move `*_10k`/full-size CSVs to release assets or a `pamola-data` companion repo, add `data/raw/POLICY.md`. **No files removed pending decision.** |
| **Q-02** | Does BEST's bridge pass tenant-supplied `expression` / `mvf_parser` / `modify_expression` into pamola-core? | **OPEN — for CW** | Determines whether F-08 (`eval()` on config strings) is a Medium hygiene item or a High multi-tenant escape. Until answered, F-08 is treated as High-if-exposed. |
| **Q-03** | Dependency policy: split the 46 `==` pins into a small core + extras? Who owns it, on what timeline? | **OPEN — for CW + Val** | Largest single obstacle to adoption outside the PAMOLA ecosystem, and a live resolution hazard for BEST (`numpy==1.26.4` blocks NumPy 2.x). Must be designed jointly with BEST. |
| **Q-04** | Branching model: keep git-flow (`main` + `develop`), or collapse to trunk + release tags? | **PARTIALLY ANSWERED 2026-08-16** | git-flow retained for now; `main` → `develop` reconciliation done (see F-01). Remaining sub-question: prune the 26 remote branches, especially the seven-strong `json_dynamic*` family — needs their authors' sign-off. |
| **Q-05** | Documentation source of truth: BOX, `docs/specs/`, `docs/en/`, or Sphinx output? | **OPEN — for Val** | Two builders (Sphinx + MkDocs) and three content trees currently coexist across 393 tracked files. |
| **Q-06** | CC's mandate: open PRs directly against `develop`, or route proposals through Val/Titan? | **ANSWERED 2026-08-16 (Val)** | CC opens PRs. This audit still changed nothing outside `DEVLOG.md`, `CLAUDE.md`, `docs/prompts/`, and `docs/output/` — code and CI changes remain proposals until scheduled. |

---

## What Was Not Audited

- Correctness of anonymization, metrics, profiling, or attack algorithms
  against their specs (that is a CD task, and requires the BOX chapters
  05–10 as the reference)
- Runtime performance and memory behaviour on the 10k-row sample datasets
- `pamola_core/utils/` internals — 147 files, the largest package, and the one
  every operation depends on; it deserves its own dedicated review
- The `errors/` framework and its `context/*.yaml` catalogue
- Actual test pass rate and coverage (blocked by F-13)
- Conformance of the BEST bridge against this library's current signatures

---

## GCA Evidence Package

Not applicable — this is an observational audit, not a block delivery. No
requirement statuses changed. No source file was modified.

---

*Report produced by CC on 2026-08-16. Findings are proposals, not decisions.*
*Debt items TD-PC-01 … TD-PC-08 registered in `DEVLOG.md`.*
