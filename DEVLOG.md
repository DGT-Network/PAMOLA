# PAMOLA.CORE DEVLOG

Narrative development log for `pamola-core`. Newest entry first.
Format follows the BEST PRW DEVLOG: dated entries, then ADRs, then the
technical-debt register.

Companion documents:
- `docs/prompts/20260816_SYSTEM_CC_PAMOLA_CORE.md` — CC agent system prompt
- `docs/output/` — CC / CD / GCA reports
- `CHANGELOG.md` — user-facing, release-scoped (Keep a Changelog / PEP 440)

DEVLOG is for *why and how*; CHANGELOG is for *what shipped*.

---

## 2026-08-17 — 1.0 cleanup: ~12 600 lines removed, singling-out added

Sections A and B of `docs/output/20260816_CC_RUNBOOK_1_0.md` executed, plus the
decisions Val took on top of them. Session report:
`docs/output/20260817_CC_SESSION_REPORT.md`.

**Positioning changed, and it invalidated an ADR.** CD established that BEST
retired the pamola-core bridge on **2026-06-14** (`BRIDGED_OP_CODES = 0`,
`best/src/best/operations/constants.py:3`); BEST's core operations are native
polars and `pamola-core` is not among its dependencies. ADR-PC-02 — "the BEST
bridge is a public contract" — was therefore false when written, two months
after the fact. Superseded by **ADR-PC-05**: *BEST is the governed execution
system; CORE is the open privacy-engineering toolkit.* No decision in this
repository may be justified by BEST depending on it.

**Removed, ~12 600 lines, no functionality lost:**

| What | Lines | Why |
|---|---:|---|
| LLM contour (`utils/nlp/llm/` + `text_transformer.py`) | 8 376 | Val's decision; a leaf of the dependency tree |
| `attacks/attacks_test.py` | 641 | smoke harness that shipped to PyPI and pytest never collected |
| `membership_inference` + two dead duplicates of DCR/NNDR | 326 | MIA is model privacy, not anonymization; the duplicates shadowed the live `metrics/privacy` implementations |
| `op_test_helpers.py` → `tests/` | 827 | test scaffolding inside the wheel, neither exported nor covered |
| `metrics/quality/` | 249 | zero consumers; `FidelityOperation` registers only `fidelity/distribution` |
| `pipeline/`, `interface/`, `crypto/`, duplicate enum | ~60 | empty placeholder packages in the wheel |

**Added:** `SinglingOutAttack` — the EDPB criterion the package did not cover.
Ported, not written: three PAMOLA implementations were compared and the best was
**not** BEST's. `spikes/i_piper/roles/msu.py` (MSU + SUDA2, Elliot) is already
pandas and clean-room; BEST's contributes the worst-case sweep and the
independent-marginals baseline, without which a uniqueness rate is
uninterpretable. Both engines are kept because they answer different questions —
*how exposed* versus *which columns cause it*.

**Contract:** `supports_batch` declared on `BaseOperation`, so all 42 exported
operations answer whether they can be batch-processed (19 True, 23 False)
instead of raising `FeatureNotImplementedError` when tried. Narrowing was chosen
over inventing batch semantics for record suppression; `False → True` later is
backwards compatible, the reverse is not.

**Gates now block:** ruff `E9 + F` over `pamola_core/` *and* `tests/`,
`fail_under = 83`, public-API↔coverage drift test, batch-contract test.
Twelve real defects surfaced the moment the lint gate stopped being advisory —
the sharpest being six duplicated regex keys in `common/regex/patterns.py` that
made every ambiguous date parse as US MDY, with the DMY entries dead on arrival.

**Honesty fixes:** `FidelityMetricsType` dropped `JS`/`WASSERSTEIN`, which were
declared but never registered, so they could not be selected; the test now
compares the enum against the operation registry so the two cannot drift again.
README moved to `dev4` and its attack note corrected — it had claimed modules
were used by `PrivacyMetricOperation` that never were.

**Numbers:** 5 551 tests, coverage 84.55 % (was 84.33 %), install 954 MB (was
2 363 before dev4).

**`docs/prompts/` is now git-ignored.** Agent working prompts are development
scaffolding, not project documentation, and on a public repository they invite
confusion about what the project promises. The public record is DEVLOG plus
`docs/output/`.

---

## 2026-08-16 (later) — `1.0.0.dev4` published; `0.0.1` stub yanked

PRs #99, #100 and #101 merged into `develop`; `main` is now an ancestor of
`develop` (`main...develop` = `0 34`, was `1 26`).

Tagged `v1.0.0.dev4` and published. `release.yml` run 31976061553: validate,
pytest on **3.10 / 3.11 / 3.12**, and publish — all green. Published artifact
verified independently: version `1.0.0.dev4`, author `REALM Inveo Inc.`,
4 `Project-URL`s, 46 `Requires-Dist` (all eight additions present, none of the
eight removals), template docs shipped, 71 public symbols, install size 973 MB.

Val then yanked `0.0.1` via the PyPI web UI. Post-conditions measured:

| Command | Before | After |
|---|---|---|
| `pip install pamola-core` | installs the 3.9 kB stub, silently | **hard error** naming all versions |
| `pip install --pre pamola-core` | `1.0.0.dev3` | `1.0.0.dev4` |
| `pip install pamola-core==0.0.1` | installs | installs, with a yank WARNING + reason |

**The open question is closed: pip does *not* fall back to pre-releases when
the only final release is yanked.** So the README `--pre` note stays until
`1.0.0` final ships — that, not the yank, is what makes plain `pip install`
correct. What the yank bought is the change from a *silent, misleading* failure
(an empty package that imports fine, then `AttributeError`s on every symbol) to
a *loud, actionable* one.

TD-PC-09 resolved. Full record and sign-off in
`docs/output/20260816_CC_PYPI_YANK_RUNBOOK.md` §8.

---

## Current State (2026-08-16)

- **Version:** `1.0.0.dev3` — published to PyPI as
  [`pamola-core`](https://pypi.org/project/pamola-core/)
- **Default branch:** `develop` @ `79a9f1c` (2026-06-01)
- **Stable branch:** `main` @ `5fd64cc` — the squash-merged dev3 release
  PR (#98). `develop` was 26 commits ahead of the merge base while `main`
  carried 1 commit absent from `develop`, **although the two trees were
  byte-identical** — a purely historical split. Reconciled 2026-08-16 via
  PR #99 (no content change).
- **Tag `v1.0.0.dev3`** points at a commit on `develop` (per release policy
  for `*dev*` tags).
- **Scale:** 634 tracked files under `pamola_core/` (618 `.py`), 221 test
  modules, ~286k Python LOC including tests.
- **Working tree:** ~152 MB, of which ~120 MB is tracked raw CSV in `data/raw/`.
- **Local environment:** not provisioned. No `.venv`, no local test run yet.
- **Upstream ownership:** the Titan team authors most commits
  (`thuan.dang` 136 + 24, `Hoang Ha Huy` 115, `lannguyen53` 58,
  `ValKhv` 52, others). CC is a *co-worker on a public repo*, not the owner.
- **Open findings:** `docs/output/20260816_CC_REPO_AUDIT.md`

---

## 2026-08-16 — Working copy bootstrapped; repository audit

Set up `D:\VK\_DEVEL\pamola_core` as a working copy of
`https://github.com/DGT-Network/PAMOLA` and established the CC working
conventions, mirroring the BEST PRW setup.

- `git init` + `origin` → full fetch → checked out `develop` (default branch).
  `D:/VK/_DEVEL/pamola_core` added to git `safe.directory` (the drive does not
  record ownership).
- Created `docs/output/` (CC/CD/GCA reports) and `docs/prompts/` (agent system
  prompts), each with a `.gitkeep`.
- Wrote `docs/prompts/20260816_SYSTEM_CC_PAMOLA_CORE.md` — the CC system
  prompt, modelled on BEST's `docs/prompts/20260322_SYSTEM_GCA.md`. It records
  the operation contract, the BEST-bridge compatibility constraint, the
  branching/release policy, and the report format.
- Wrote this DEVLOG and `docs/output/20260816_CC_REPO_AUDIT.md`.

**Nothing in `pamola_core/`, `tests/`, or CI was modified.** The audit is
observation only; every finding is a proposal awaiting Val's decision.

**Headline audit findings** (detail in the report):
1. `main`/`develop` divergence is unreconciled — a release commit lives only
   on `main`. **→ fixed same day, PR #99.**
2. ~120 MB of synthetic CSV datasets tracked in git since 2026-01-10.
   **→ provenance question closed (see below); surface reduction pending.**
3. `pyproject.toml` ships placeholder author/maintainer identities
   (`author@example.com`) — these are live on PyPI.
4. All 46 runtime dependencies are hard-pinned with `==`, including `torch`,
   `sdv`, `spacy`, `faiss-cpu`, `dask[complete]`. For a *library* this is a
   dependency-resolution hazard, and it directly affects BEST, which installs
   pamola-core alongside its own pinned stack.
5. CI installs `-e ".[dev]"` but no `dev` extra exists; the failure is
   swallowed by `2>/dev/null || pip install -e .`.
6. The lint gate is effectively advisory (`--select E9,F63,F7,F82`), coverage
   is measured but never enforced (`fail_under = 0`), and there is no type
   check.
7. Unsandboxed `eval()` on config-supplied strings in two production paths,
   plus `pickle.load()` of a mapping store file.
8. No `SECURITY.md`, `CODEOWNERS`, dependabot config, or PR template on a
   public privacy-engineering repo. PyPI publish uses a long-lived API token
   rather than Trusted Publishing.

**Debt registered below as TD-PC-01 … TD-PC-08.**

### Correction, same day — `data/raw/` is fully synthetic

The audit's first revision asked whether `data/raw/` might contain real
personal data and claimed no provenance file existed. **Both parts were
wrong.** Recorded here so the question is never reopened from scratch:

> **All datasets under `data/raw/` are fully synthetic.** They are
> programmatically generated (Faker + domain-specific generators +
> statistical distributions matched to real-world shapes without real data),
> carry intentional **synthetic fingerprints** for lineage tracking and
> misrepresentation detection, and contain **no real PII or PHI**.

The evidence was in the repo the whole time, in three tracked layers:
`data/README.md` and `data/raw/README.md` (explicit legal notice), a
per-dataset `readme.md`, and a `*_passport.json` per file recording
`records`, `generated_at`, and the `direct` / `quasi` / `sensitive` privacy
categorisation of every column. Two generators are committed as well —
`data/raw/med_rare_lab/generate_med_rare_lab.py` and
`data/raw/bank_txs/generate_bank_txs_v2.py`.

The upstream generation workspace is `D:\VK\_DEVEL\_DATARAW` (local, not
published). It also holds genuinely third-party corpora — MIMIC-III, Enron,
NYC Taxi, UCI Adult, IEEE-CIS, ai4privacy, NVIDIA Nemotron-PII and others —
**none of which were ever copied into the public repo.** Only the
PAMOLA-generated families were. That separation is the rule that keeps the
public repo safe, and it should be written down rather than left implicit:
proposed as `data/raw/POLICY.md`.

**What remains open (Q-01):** whether to publish this data at all, at this
size. Val's position is that even synthetic data may not belong in a public
repo and the surface should probably shrink. The proposal in the audit is to
keep small fixtures, all passports, all readmes and the generators in-repo,
and move the `*_10k` / full-size CSVs to release assets or a `pamola-data`
companion repo. **Nothing has been deleted; no history rewrite is proposed.**

### Later the same day — test baseline, and a dependency graph held together by luck

Provisioned the environment (closing F-13) and ran the suite end to end.

**Baseline:** 5 477 tests, **5 477 passed** in a clean environment, 41–43 min
single-process. Public-API coverage **85.13 %** (18 962 statements, 2 820
missed, 63 files) — **the project's advertised 85 % is accurate.** Recommended
`fail_under = 84`.

Five tests fail on this machine only: ambient `PAMOLA_PROJECT_ROOT=D:\VK\_DEVEL\HHR`
(left over from the predecessor project) overrides configs the tests build in
`tmp_path`. Product behaviour is correct (`utils/paths.py:45`); the **tests** are
not hermetic, and there is no root `conftest.py` where such an invariant could
live. TD-PC-11.

Running the suite also leaves 43 untracked files in the working tree, including
inside `pamola_core/utils/resources/` and `configs/`. TD-PC-12.

**The significant finding.** PR #101 removes eight unused dependencies. The
static reference count behind that was correct — but it answers "does anything
import this package", not "what did this package supply". A clean virtualenv
built from the edited `pyproject.toml` could not import `pamola_core` at all:
170 collection errors, `ModuleNotFoundError: No module named 'dotenv'`.

An AST audit of every import against the declared list found **thirteen
undeclared third-party modules, eight at module scope** — `python-dotenv`,
`filelock`, `joblib`, `rich`, `tqdm`, `packaging`, `jinja2`, `regex`. All eight
arrived only as *transitive* dependencies; two of them (`python-dotenv` via
`uvicorn[standard]`, `filelock` via `torch`) were supplied **exclusively** by
packages being removed. The package's importability rested on transitive luck,
with no signal in `pyproject.toml`. The removals did not create that fragility —
they revealed it. TD-PC-13, fixed in PR #101.

After declaring all eight: **5 477 passed, 0 failed**; install size **954 MB**
against 2 363 MB for the `dev3` set (−60 %).

### Actions taken

- **PR #99** — `chore: merge main into develop (reconcile release history)`.
  Zero file changes: `git diff --stat origin/main origin/develop` was already
  empty, so the split was historical only and the merge is a no-op on content.
- **PR #100 (this branch)** — CC project setup and reports: `CLAUDE.md`,
  `DEVLOG.md`, `data/raw/POLICY.md`, `docs/prompts/`, `docs/output/`.
- **PR #101** — `1.0.0.dev4` release prep: packaging metadata, package-data,
  CI `[dev]`→`[test]`, README `--pre` note, and the dependency corrections
  above. Verified on a clean environment before requesting merge.
- **Yank runbook** — `docs/output/20260816_CC_PYPI_YANK_RUNBOOK.md`. Strict
  ordering: publish `1.0.0.dev4` **first**, yank `0.0.1` second. Yanking first
  would leave plain `pip install` failing outright rather than serving the stub.
- Note: pushing `chore/sync-main-into-develop` reported
  `remote: Bypassed rule violations` — a repository ruleset is configured and
  was bypassed by the pushing account's permissions. Worth a look; CC did not
  change any ruleset.

---

## ADR-PC-01: pamola-core is a library, not a service

**Status:** Accepted (de facto, recorded here 2026-08-16)
**Rule:** the deliverable is a PyPI wheel. No server, no scheduler, no
persistent state. Operations read a `DataSource` and write a `task_dir`.
**Consequence:** anything requiring orchestration, a control plane, budget
enforcement, or distributed execution belongs in BEST, not here.

## ADR-PC-02: BEST bridge is a public contract

**Status:** **SUPERSEDED by ADR-PC-05** (2026-08-17). Recorded 2026-08-16 on a
premise that was already false: BEST retired the bridge on 2026-06-14
(`BRIDGED_OP_CODES = 0`, `best/src/best/operations/constants.py:3`), its core
operations are native polars, and `pamola-core` is not among its dependencies.
Kept here for the record; **do not cite this ADR to justify any decision.**
**Original rule (void):** public operation signatures, parameter names, and
artifact filenames are a contract with BEST's `_bridge_impl/` mixins.

## ADR-PC-03: Public API is defined by `__init__.py` + `.coveragerc`

**Status:** Accepted (de facto, recorded here 2026-08-16)
**Rule:** `pamola_core/__init__.py` `__all__` is the canonical export list;
`.coveragerc` `[run] include` is the canonical *public-API coverage scope*
(per SRS 4.1.11). The two must stay in sync — a symbol exported but absent
from `.coveragerc` is untracked public surface.
**Weakness:** both lists are hand-maintained. See TD-PC-05.

## ADR-PC-04: Release is tag-driven, branch-gated

**Status:** Accepted (implemented in `.github/workflows/release.yml`)
**Rule:** `v*dev*` tags must be on `develop`; other `v*` tags on `main`.
The workflow verifies tag ↔ branch, tag ↔ `pyproject.toml` version, and that
the version appears in `CHANGELOG.md` before publishing.
**Consequence:** never hand-edit `version` in `pyproject.toml` outside a
deliberate release preparation commit.

## ADR-PC-05: CORE is an independent OSS library; BEST is not its consumer

**Status:** Accepted 2026-08-17 (CD). **Supersedes ADR-PC-02.**

**Context.** BEST retired the `pamola-core` bridge on 2026-06-14. It does not
import this library, does not depend on it, and does not execute it. The two
projects are siblings in one ecosystem, not a stack.

**Rule.**
- `pamola-core` is a standalone OSS Python library: operations, profiling,
  metrics, reproducible artifacts, and a synthetic benchmark/demo layer. It is
  designed to be installed and run on its own.
- BEST owns runtime, orchestration, gates, budgets, evidence packs, and the
  distributed substrate (Ray/S3/polars). None of that belongs here.
- What the two share is **conceptual, not runtime**: DSL terms, privacy
  semantics, artifact vocabulary, metric definitions — the checkable
  definitions, not the call signatures.

**Consequence.**
- No change in this repository may be justified, blocked, or scoped by "BEST
  depends on it." That premise is void.
- Public-facing text (README, docs, reports) states *"usable independently;
  shares privacy model and vocabulary with the PAMOLA ecosystem"* — never
  *"compatible with the BEST bridge."*
- Breaking-change policy still applies, but its beneficiary is **downstream OSS
  users**, defined by ADR-PC-03's `__all__`, not by BEST.

---

## Technical Debt Register

| ID | Title | Severity | Owner | Status |
|---|---|---|---|---|
| TD-PC-01 | `main`/`develop` divergence unreconciled | High | CC | **RESOLVED** — PR #99 |
| TD-PC-01b | 26 remote branches, many stale (`json_dynamic*` ×7) | Low | Val/Titan | OPEN (Q-04) |
| TD-PC-02 | ~120 MB synthetic CSV published in a public repo | Low | Val | OPEN (Q-01) — provenance closed, surface pending |
| TD-PC-02b | `data/raw/crossparty_anti-fraud/LICENSE` is Apache-2.0 inside a BSD-3 repo | Low | Val | OPEN |
| TD-PC-03 | Hard-pinned `==` runtime deps incl. torch/sdv/spacy | High | Val/CW | OPEN |
| TD-PC-04 | Placeholder author/maintainer identity on PyPI | Medium | Val | OPEN |
| TD-PC-05 | Quality gates advisory only (lint, coverage, no mypy) | Medium | CC | OPEN |
| TD-PC-06 | `eval()` / `pickle.load()` on config-supplied input | Medium | CC | OPEN |
| TD-PC-07 | Missing OSS governance files; PyPI long-lived token | Medium | Val | OPEN |
| TD-PC-08 | Two documentation toolchains (Sphinx + MkDocs) + stale `site/` | Low | Val | OPEN |
| TD-PC-09 | `pip install pamola-core` resolves to an empty, proprietary-licensed `0.0.1` stub | **High** | Val | **RESOLVED** 2026-08-16 — `1.0.0.dev4` published, `0.0.1` yanked |
| TD-PC-10 | 3 template docs in repo not packaged in the wheel | Low | CC | **RESOLVED** — PR #101 |
| TD-PC-11 | Test suite is not hermetic — ambient `PAMOLA_PROJECT_ROOT` breaks 5 tests; no root `conftest.py` exists | Medium | CC | **RESOLVED** 2026-08-17 |
| TD-PC-12 | Test suite writes 43 files into the working tree, incl. `pamola_core/utils/resources/` and `configs/` | **High** | CC | OPEN — `.gitignore` is a stopgap |
| TD-PC-14 | `TestDataWriter` fails a different test each run | Medium | CC | OPEN — cause found: `_tkinter.TclError`, matplotlib picks the Tk backend. Fix is `Agg` in `tests/conftest.py` |
| TD-PC-15 | Six tests in `tests/` are shadowed by F811 and never execute | Medium | CC | OPEN — quarantined per-file in `pyproject.toml` with each case analysed |
| TD-PC-16 | `FidelityMetricsType` lost `JS`; a working `_jensen_shannon_divergence` helper exists in `statistical_fidelity.py` but is unreachable from `FidelityOperation` | Low | CC | OPEN |
| TD-PC-17 | Ambiguous `dd/mm` vs `mm/dd` dates resolve to US convention; previously silent, now explicit in `common/regex/patterns.py` | Low | Val | OPEN — product decision |
| TD-PC-13 | 8 module-scope imports were undeclared, arriving only transitively | **High** | CC | **RESOLVED** — PR #101 |

Detail, evidence, and proposed remediation for each: see
`docs/output/20260816_CC_REPO_AUDIT.md`.

---

## Open Questions

The authoritative register — with answers, dates, and who answered — is the
**Question Register** in `docs/output/20260816_CC_REPO_AUDIT.md`. Summary:

| ID | Question | Status |
|---|---|---|
| Q-00 | Are `data/raw/` datasets real or synthetic? | **ANSWERED** — fully synthetic (Val, 2026-08-16) |
| Q-01 | Reduce the *published* dataset surface anyway? | OPEN — leaning yes |
| Q-02 | Does BEST pass tenant-supplied `expression`/`mvf_parser` through? | OPEN — for CW; gates F-08 severity |
| Q-03 | Split the 46 `==` pins into core + extras? Owner, timeline? | OPEN — for CW + Val |
| Q-04 | Keep git-flow, and prune the 26 remote branches? | PARTIAL — reconciled; pruning open |
| Q-05 | Documentation source of truth: BOX / `docs/specs` / `docs/en` / Sphinx? | OPEN — for Val |
| Q-06 | Does CC open PRs directly against `develop`? | **ANSWERED** — yes (Val, 2026-08-16) |
