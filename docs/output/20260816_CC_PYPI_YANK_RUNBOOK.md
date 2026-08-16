# Runbook — Yanking `pamola-core` 0.0.1 on PyPI

**Date:** 2026-08-16
**Author:** CC (Claude Code)
**Operator:** Val (requires PyPI maintainer rights on the `pamola-core` project)
**Estimated time:** 5 minutes for the yank itself; the prerequisite release is longer.
**Refs:** `docs/output/20260816_CC_PYPI_PARITY.md` P-03, `DEVLOG.md` TD-PC-09

---

## 0. Why

`pip install pamola-core` currently installs a **3.9 kB stub**, not this library.

Verified, not inferred:

```
$ pip download pamola-core --no-deps
  Downloading pamola_core-0.0.1-py3-none-any.whl (3.9 kB)
```

`0.0.1` (uploaded 2026-02-23) is a name-reservation placeholder containing one
`__init__.py` with a docstring and five dunder assignments. Everything released
since is a *pre-release* (`1.0.0.dev1..dev3`), and pip skips pre-releases by
default — so `0.0.1`, being the only *final* release, wins.

`import pamola_core` then succeeds and every documented symbol raises
`AttributeError`. The failure mode is not "not installed", it is
**"installed and silently empty."**

The stub additionally declares `License-Expression: LicenseRef-Proprietary`
against the project's BSD-3-Clause, and points its `Project-URL` entries at
`github.com/realminveo/pamola-core` rather than `github.com/DGT-Network/PAMOLA`.

---

## 1. ORDER OF OPERATIONS — read this before touching anything

**Do NOT yank `0.0.1` before `1.0.0.dev4` is live on PyPI.**

Yanking first leaves the project in a worse state than today for anyone using
a plain install: pip will refuse outright rather than serve *something*.
Publish first, yank second.

```
  ①  park / merge open branches
  ②  merge PR #101  (release prep 1.0.0.dev4)
  ③  tag + push  →  release.yml publishes 1.0.0.dev4 to PyPI
  ④  VERIFY dev4 is installable
  ⑤  yank 0.0.1
  ⑥  VERIFY the new resolution behaviour
```

Steps ① – ④ are prerequisites. **Step ⑤ is the only irreversible-looking one,
and it is in fact reversible** (§6).

---

## 2. Prerequisites — confirm ALL of these before step ⑤

| # | Check | How | Expected |
|---|---|---|---|
| 2.1 | PR #101 merged into `develop` | GitHub | merged |
| 2.2 | `pyproject.toml` version | `python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"` | `1.0.0.dev4` |
| 2.3 | CHANGELOG contains the version | `grep 1.0.0.dev4 CHANGELOG.md` | a match |
| 2.4 | Tag is on `develop` | required by `release.yml` for `*dev*` tags | yes |
| 2.5 | `1.0.0.dev4` visible on PyPI | https://pypi.org/project/pamola-core/#history | listed |
| 2.6 | dev4 actually installs | `pip install --pre pamola-core` in a scratch venv, then `python -c "import pamola_core; print(pamola_core.__version__)"` | `1.0.0.dev4` |

**If 2.6 fails, stop.** Do not proceed to the yank.

### Publishing dev4 (step ③)

```bash
git checkout develop && git pull
git tag v1.0.0.dev4
git push origin v1.0.0.dev4
```

`.github/workflows/release.yml` then runs three gates before publishing —
tag↔branch, tag↔`pyproject.toml`, version↔`CHANGELOG.md` — and uploads via
`twine`. Watch the run to completion; do not assume success.

---

## 3. The yank

1. Go to **https://pypi.org/manage/project/pamola-core/releases/**
   (requires Owner or Maintainer role on the project).
2. Find **0.0.1** in the release list.
3. Open its **Options** menu (⋮) → **Yank**.
4. In the reason field enter exactly:

   ```
   Name-reservation placeholder. Contains no library code and declares a
   licence that does not match the project. Use 1.0.0.dev4 or later.
   ```

   The reason is shown to anyone who pins the version, so it must be
   actionable, not just descriptive.
5. Confirm.

**Do NOT use "Delete".** Deletion is permanent, frees nothing useful, and PyPI
will not let the version number be reused — which forecloses options. Yank is
the correct instrument here.

---

## 4. What yank does and does not do

| | |
|---|---|
| Removes the files | **No.** `pip install pamola-core==0.0.1` still works — by design, so nothing already pinned breaks. |
| Removes it from the resolver's default choices | **Yes.** This is the point. |
| Deletes the version page | No. The proprietary licence string and the `realminveo` URLs stay visible on that version's page. |
| Is reversible | **Yes** — see §6. |
| Affects `1.0.0.dev*` | No. |

---

## 5. Verification after the yank (step ⑥) — do not skip

Run in a **fresh** virtualenv, not one that has any cached install:

```bash
python -m venv /tmp/verify && source /tmp/verify/bin/activate    # Windows: \Scripts\activate
pip cache purge

# A — plain install must NOT silently produce an empty package
pip install pamola-core

# B — the documented path must work
pip install --pre pamola-core
python -c "import pamola_core; print(pamola_core.__version__)"
```

**Expected:**

- **B** installs `1.0.0.dev4` and prints it. *This is the check that matters.*
- **A** does one of two things, and **both are acceptable outcomes**:
  - errors with `Could not find a version that satisfies the requirement`
    while listing the available pre-releases, or
  - installs `1.0.0.dev4` directly.

  Which one occurs depends on how this pip version orders yank-filtering
  against pre-release-filtering. I verified the *exclusion* proxy
  (`pip download "pamola-core!=0.0.1"` → hard error, pip does **not** fall back
  to pre-releases) but I have **not** been able to verify the *yanked* case
  without performing the yank. **Record which of the two you observe** and add
  it to this file — it determines whether the README still needs the `--pre`
  instruction.

- **Unacceptable:** A installs `0.0.1`. That means the yank did not take
  effect — recheck the PyPI page before doing anything else.

---

## 6. Rollback

If anything downstream breaks:

1. https://pypi.org/manage/project/pamola-core/releases/ → **0.0.1** →
   **Options** → **Un-yank**.
2. State returns exactly to today's behaviour. No republish, no version bump,
   no data loss.

Rollback is complete and immediate. This is why yank, not delete.

---

## 7. Follow-ups this unblocks

| Item | Note |
|---|---|
| README `--pre` note | Added in PR #101. Remove it once `1.0.0` **final** ships — that is the point at which plain `pip install pamola-core` becomes correct with no caveats. |
| Licence consistency | The `0.0.1` page will still read `LicenseRef-Proprietary`. If that is unacceptable for a public BSD-3 project, deletion is the only removal — decide deliberately, not as part of this runbook. |
| `1.0.0` final | The real fix for plain-install correctness. Gated on the audit's High findings (dependency policy, coverage enforcement, `eval()` hardening). |

---

## 8. Sign-off

Fill this in after execution so the next reader knows the actual outcome:

```
Executed by:      ____________________
Date/time (UTC):  ____________________
dev4 on PyPI:     [ ] confirmed
0.0.1 yanked:     [ ] confirmed
Check A result:   [ ] hard error   [ ] installed 1.0.0.dev4   [ ] installed 0.0.1 (FAILURE)
Check B result:   [ ] 1.0.0.dev4   [ ] other: ____________
Rolled back:      [ ] no   [ ] yes, reason: ____________
```
