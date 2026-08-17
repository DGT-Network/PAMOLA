# TD-PC-15 — the six quarantined F811 tests: case-by-case analysis

**Date:** 2026-08-17 · **Author:** CC · **Status:** analysis only, no code changed
**Scope:** the five per-file `F811` quarantine entries in `pyproject.toml`
(`[tool.ruff.lint.per-file-ignores]`), covering eight redefinition sites.

This document decides, for each site, **what stays, what is deleted, and what is
renamed** — so the fix that follows the merge of PR #105 is mechanical and its
risk is known in advance. It does not change code.

---

## 1. Why this is not cosmetic

Ruff reports all eight as one rule, but they are **three different defects** with
three different consequences. Collapsing them into "duplicate definitions" is
what let them survive.

| Kind | Sites | What actually happens |
|---|---|---|
| **A. Dead test** | 4 | The later definition wins; the earlier test body never executes. The suite reports a pass for a test that never ran. |
| **B. Stub substitution** | 3 | The duplicates are *helpers*, not tests. Name lookup inside a test method happens at **call** time, so tests defined at line 86 run against a stub defined at line 823. |
| **C. Dead class** | 1 | An entire `unittest.TestCase` is shadowed; two tests never run. |

Kind **B** is the one worth slowing down for. In kinds A and C the loss is
coverage we thought we had. In kind B the *live* tests are silently exercising a
different fixture than the one written next to them — which means a passing
result is evidence about the wrong object.

---

## 2. Case A1 — `test_datetime_op_extended.py:436 / 452`

`test_process_value_rounding_day`, both inside the same class, identical setup.
Only the assertion differs:

| Line | Assertion | Fate |
|---|---|---|
| 436 | `result == pd.Timestamp("2023-06-15 00:00:00")` — exact | **dead** |
| 452 | `isinstance(result, (pd.Timestamp, str))` — a tautology | runs |

The stronger test is the dead one, and the surviving test asserts almost
nothing: `process_value` returning the *unrounded* input would still pass it.

**Decision: delete 452, keep 436.**
**Risk: real but small.** Line 436 has never executed. If `rounding_unit="day"`
returns a string rather than a `Timestamp`, un-shadowing turns a green test red.
That is the correct outcome — but it must be run before the PR is opened, not
discovered in CI.

## 3. Case A2/A3 — `test_kl_divergence.py:115 / 262` and `134 / 244`

Both pairs sit in `TestKLDivergence` (single class, confirmed — no other class
boundary between the lines). Unlike A1 these are **not duplicates**: each pair is
a public-path test and a private-helper test that happen to share a name.

| Name | Dead half | Live half |
|---|---|---|
| `..._jensen_shannon_distance` | 115 — `calculate_metric(...)["jensen_shannon_distance"]` | 262 — `kl._jensen_shannon_distance(p, q)` |
| `..._statistical_significance` | 134 — `calculate_metric(...)["statistical_significance"]` | 244 — `kl._is_statistically_significant(v)` |

In both pairs the **public** test is the dead one. That is the wrong half to
lose: the private helpers are covered, while the contract that the returned dict
actually carries these keys is not tested at all.

**Decision: rename, do not delete.** Both halves are legitimate tests.

- `115` → `test_kl_divergence_jensen_shannon_distance_via_calculate_metric`
- `134` → `test_kl_divergence_statistical_significance_via_calculate_metric`

**Risk: low, and pre-verified.** Both keys are genuinely produced —
`kl_divergence.py:148` (`"statistical_significance"`) and `:150`
(`"jensen_shannon_distance"`), populated at `:139` and `:133`. The dead tests
should pass on being woken.

**Note for TD-PC-16.** This is the same `jensen_shannon` helper that PR #105
found unreachable from `FidelityOperation`. Reviving test 115 documents that the
computation works and is exposed *by `KLDivergence`* — it does not close
TD-PC-16, which is about `FidelityMetricsType` not offering it. Do not let a
green test here be read as closing that debt.

## 4. Case A4 — `test_distance.py:90 / 229`

`test_sample_size` in `TestDistanceToClosestRecord`, **byte-identical** including
the comment. Pure copy-paste.

**Decision: delete 229, keep 90. Zero risk** — the surviving body is the one
already running.

**Separate finding, not part of this fix.** The test is misnamed and vacuous: its
own comment says `DistanceToClosestRecord does not support sample_size`, and the
body only asserts `"dcr_statistics" in result` — making it a duplicate of
`test_single_column`. There is no test of sampling behaviour anywhere. Deleting
the F811 twin does not fix that; it makes it visible. Logging as a follow-up
rather than widening this change.

## 5. Case B — `test_categorical.py:17/778, 33/823, 46/838`

Three helpers — `DummyDataSource`, `Reporter`, `Progress` — each defined twice at
module scope, ~760 lines apart. The first block serves the `unittest` classes;
the second was appended later with the note `# Add pytest-based tests for full
coverage`.

Because the `unittest` classes resolve these names **when a test method runs**,
not when the class body is read, `TestCategoricalOperation.setUp` at lines 86–87
instantiates the **second** definitions.

| Helper | First def | Second def | Equivalent? |
|---|---|---|---|
| `DummyDataSource` | 17 | 778 | **Yes** — identical bodies. Harmless. |
| `Reporter` | 33 | 823 | **Yes** — identical bodies. Harmless. |
| `Progress` | 46 | 838 | **No.** |

`Progress` is the actual defect:

- **First (46):** `update(*args, **kwargs)` — accepts anything. No `__init__`,
  no `create_subtask`.
- **Second (838):** `update(self, step, info)` — **two required positional
  args** — plus `__init__` and `create_subtask`.

So every `unittest` test in this file has been running against a *stricter*
progress stub than the one written beside it, and against one that supports
`create_subtask` where the intended stub would have raised `AttributeError`.

**Decision: keep one definition per helper, placed before first use.** Delete the
second `DummyDataSource` and `Reporter` (proven equivalent). For `Progress`,
**keep the second (838) body** — it is the one that has actually been exercised,
so keeping it preserves current behaviour — and delete the stub at 46.

**Risk: this is the case that needs a real run.** `categorical.py` calls
`progress_tracker.update(...)` at seven sites (`:266, :292, :331, :372, :428,
:456, :486`). Consolidating onto the two-arg signature is behaviour-preserving
*only if* every one of those calls already passes two arguments — which is
implied by the tests currently passing, but must be confirmed by execution, not
by reading. Do this case in its own commit.

## 6. Case C — `test_correlation.py:109 / 902`

`TestCorrelationAnalyzer` is declared twice. The class at **109 is entirely
dead** — `test_analyze` and `test_analyze_matrix` never run.

Comparing the two:

| | Dead class (109) | Live class (902) |
|---|---|---|
| `analyze` delegation | yes | yes, **plus** asserts `extra_param` passthrough |
| `analyze_matrix` delegation | yes | yes, **plus** asserts `min_threshold` passthrough |
| `estimate_resources` | — | yes |

The live class is a strict superset: same two behaviours, stronger assertions
(kwargs passthrough), plus a third test. The dead class contributes nothing the
live one lacks.

**Decision: delete the class at 109 outright. Risk: none** — no executed test
changes, and no assertion is lost.

---

## 7. Execution plan and expected effect

Four commits, in this order — cheapest and safest first, the one that can move
behaviour last:

1. **C** — delete dead `TestCorrelationAnalyzer` (109). No behaviour change.
2. **A4** — delete duplicate `test_sample_size` (229). No behaviour change.
3. **A2/A3** — rename the two dead public KL tests. **+2 executed tests.**
4. **A1** — delete the vacuous `rounding_day` (452). May turn red; that is the point.
5. **B** — consolidate the three `test_categorical.py` helpers. Own commit, own run.

Then drop the five `per-file-ignores` entries and the quarantine comment block
from `pyproject.toml`, so ruff `F` blocks on `tests/` with no exemptions.

**Expected test count: 5 551 → ~5 553.** Two revived, two duplicates removed,
one dead class removed (it was contributing zero). A count that lands far from
5 553 means something else was shadowed too, and the analysis was incomplete.

**Definition of done:** `ruff check tests/ --select F811` is clean with **no**
`per-file-ignores`, and the suite is green at the expected count.

---

## Evidence Package

| Claim | Status | Evidence |
|---|---|---|
| Later definition wins; earlier test never executes | CONFIRMED | Python name binding; 8 sites listed in `pyproject.toml:183-193` |
| A1: dead half carries the stronger assertion | CONFIRMED | `tests/anonymization/generalization/test_datetime_op_extended.py:436` (exact) vs `:452` (`isinstance`) |
| A2/A3: pairs are distinct tests, public half is dead | CONFIRMED | `tests/metrics/fidelity/distribution/test_kl_divergence.py:115,134` (public) vs `:244,262` (private); single class from `:30` |
| A2/A3: both dict keys are really produced | CONFIRMED | `pamola_core/metrics/fidelity/distribution/kl_divergence.py:148,150`; computed `:139,133` |
| A4: the two `test_sample_size` bodies are identical | CONFIRMED | `tests/metrics/privacy/test_distance.py:90` vs `:229` |
| A4: the test does not test sampling | CONFIRMED | same file, in-test comment + sole assertion `"dcr_statistics" in result` |
| B: live tests bind the *later* stubs | CONFIRMED | use at `tests/profiling/analyzers/test_categorical.py:86-87`; defs at `:838` |
| B: only `Progress` differs between the pairs | CONFIRMED | `:46` `update(*args, **kwargs)` vs `:838` `update(self, step, info)` + `create_subtask` |
| B: seven call sites must be checked by execution | OPEN | `pamola_core/profiling/analyzers/categorical.py:266,292,331,372,428,456,486` |
| C: live class is a strict superset of the dead one | CONFIRMED | `tests/profiling/analyzers/test_correlation.py:109-183` vs `:902-935` |
| A1 outcome (may fail on waking) | OPEN | requires execution after PR #105 merges |
