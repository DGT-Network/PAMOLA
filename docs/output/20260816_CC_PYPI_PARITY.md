# CC Note — PyPI Release Parity Check

**Date:** 2026-08-16
**Author:** CC (Claude Code)
**Question:** does the published `pamola-core` on PyPI match this repository?
**Answer:** **yes — exactly.** No code drift.

---

## Method

```bash
pip download pamola-core==1.0.0.dev3 --no-deps -d <tmp>     # 2.3 MB wheel
```

Unpacked the wheel and compared every file under `pamola_core/` against the
working tree at `develop` HEAD (`79a9f1c`), hashing with SHA-256 after
normalising CRLF→LF (the Windows checkout uses CRLF; the wheel was built on
Linux, so a raw byte comparison reports 100% "differences" and is meaningless).

Script: `scratchpad/diffwheel.py`.

## Result

```
wheel files: 631   repo files: 634

=== only in WHEEL (0) ===   (none)
=== only in REPO  (3) ===
utils/ops/templates/README.md
utils/ops/templates/config_example.jsonc
utils/tasks/templates/README.md
=== CONTENT DIFFERS (0) === (none)
```

**All 620 `.py` modules and all packaged data files are byte-identical** to
what is on PyPI. The published `1.0.0.dev3` is exactly `develop` HEAD.

This is a genuinely good result and worth stating plainly: the release
pipeline is trustworthy. Combined with the tag↔branch↔version↔CHANGELOG
validation in `release.yml`, artifact provenance for this project is sound.

Timeline is consistent too: `main` @ `5fd64cc` landed 2026-07-09, the wheel
was uploaded 2026-07-10, and tag `v1.0.0.dev3` sits on `develop` as the
release policy requires for `*dev*` tags.

---

## Findings from the comparison

### P-01 · Low · Three template files ship in the repo but not in the wheel

| Missing from wheel |
|---|
| `pamola_core/utils/ops/templates/README.md` |
| `pamola_core/utils/ops/templates/config_example.jsonc` |
| `pamola_core/utils/tasks/templates/README.md` |

`pamola_core/utils/ops/templates/operation_skeleton.py` and the six
`utils/tasks/templates/*.py` **are** packaged (they are `.py`, so setuptools
picks them up automatically). Their documentation and example config are not,
because `[tool.setuptools.package-data]` in `pyproject.toml:88-91` lists only
`errors.context/*.yaml`, `resources/**/*`, and `catalogs/*.yaml`.

So a user who `pip install`s the package gets the operation skeleton but not
the README explaining how to use it.

**Fix:** add `"pamola_core.utils.ops.templates" = ["*.md", "*.jsonc"]` and
`"pamola_core.utils.tasks.templates" = ["*.md"]` to `package-data`. Trivial.

### P-02 · Medium · Placeholder identity is confirmed live on PyPI

This closes the loop on audit finding F-04 — it is not merely a `pyproject.toml`
issue, it is what the world sees. From the published `METADATA`:

```
Name: pamola-core
Version: 1.0.0.dev3
Author-email: Author <author@example.com>
Maintainer-email: Maintainer <maintainer@example.com>
```

No `Project-URL` entries either, so the PyPI page has no link to the GitHub
repository, the documentation, or the changelog. For a privacy-engineering
library asking to be trusted with sensitive data, an anonymous-looking package
page is a real adoption obstacle, not a cosmetic one.

The in-code copyright is correct and specific
(`pamola_core/__init__.py:7` — *"(C) 2024 Realm Inveo Inc. and DGT Network
Inc."*), which makes the packaging metadata the odd one out.

### P-03 · **High** · `pip install pamola-core` installs an empty, proprietary-licensed stub

Full release history from the PyPI JSON API:

| Version | Files | Size | Uploaded |
|---|---|---|---|
| `0.0.1` | wheel + sdist | ~0 MB | 2026-02-23 |
| `1.0.0.dev1` | wheel + sdist | 2.14 / 1.74 MB | 2026-04-03 |
| `1.0.0.dev2` | wheel + sdist | 2.14 / 1.74 MB | 2026-04-06 |
| `1.0.0.dev3` | wheel + sdist | 2.18 / 1.76 MB | 2026-07-10 |

`0.0.1` is a name-reservation upload from 2026-02-23. Because `0.0.1` is a
*final* release while everything since is a *pre-release*, and pip excludes
pre-releases by default, **the plain command installs the stub.** Verified,
not inferred:

```
$ pip download pamola-core --no-deps
  Downloading pamola_core-0.0.1-py3-none-any.whl (3.9 kB)
```

3.9 kB. Its entire contents:

```
pamola_core/__init__.py     904 b   (docstring + 5 dunder assignments)
pamola_core/py.typed          0 b
```

`import pamola_core` succeeds. Every documented symbol —
`NumericGeneralizationOperation`, `FakeNameOperation`, all of it — raises
`AttributeError`. The failure mode for a new user is not "package not found",
it is "the library appears installed and is silently empty."

**Three separate problems, in increasing order of seriousness:**

1. **Wrong artifact by default.** Anyone following the PyPI link in the README
   badge gets the stub.
2. **Contradictory licence on the same name.** The stub declares
   `License-Expression: LicenseRef-Proprietary` and `__license__ = "Proprietary"`.
   Versions `1.0.0.dev1..dev3` declare **BSD-3-Clause**. The same PyPI package
   name currently serves both a proprietary and an open-source licence
   depending on which version resolves. For a project whose README leads with
   *"open-source foundation"*, this is the kind of inconsistency that ends a
   procurement review early.
3. **Points at a different repository.** The stub's `Project-URL` entries
   reference `github.com/realminveo/pamola-core` and `docs.realmdata.io`,
   not `github.com/DGT-Network/PAMOLA`. So the default install advertises a
   repo that is not this one.

There is an irony worth noting: **the stub's packaging metadata is better than
the real release's.** It has a genuine author email
(`info@realminveo.com`), keywords, `Intended Audience` classifiers, and five
`Project-URL` entries — everything P-02 says `1.0.0.dev3` is missing. The
metadata to fix F-04 already exists; it is attached to the wrong artifact.

**Proposal, in priority order:**
1. **Yank `0.0.1` on PyPI** (`pip` will then fall back correctly and refuse to
   install a yanked version by default). This is a one-click fix and it is the
   single highest-value action in either report.
2. Reconcile the licence declaration — the name must consistently mean
   BSD-3-Clause.
3. Port the stub's good metadata (author, URLs, keywords, classifiers) into
   `pyproject.toml`, closing F-04 / P-02 at the same time.
4. Until `1.0.0` final ships, state `pip install --pre pamola-core` explicitly
   in the README install section.

### P-04 · Informational · Dependency pins are baked into the published metadata

The wheel's `Requires-Dist` reproduces all 46 `==` pins, including
`torch==2.8.0`, `sdv==1.18.0`, `spacy==3.8.9`, `numpy==1.26.4`. This is audit
finding F-03 seen from the consumer side: any project installing
`pamola-core` inherits every one of those exact pins as a hard constraint.
Nothing new, but it confirms the finding is live in the wild rather than
theoretical.

---

## Conclusion

| Question | Answer |
|---|---|
| Does published code differ from the repo? | **No.** 631/631 files identical. |
| Is anything in the repo missing from the package? | 3 template docs (P-01). |
| Is the packaging metadata correct? | No — placeholder identity, no URLs (P-02). |
| Can a user `pip install pamola-core` and get this? | **No.** They get an empty proprietary stub (P-03). |

**The code pipeline is clean; the distribution surface is broken.** That
inversion is the headline of this note. `develop` HEAD and PyPI `1.0.0.dev3`
agree byte-for-byte — release engineering is doing its job — but the package
a new user actually receives from `pip install pamola-core` is a 3.9 kB
proprietary-licensed placeholder pointing at a different GitHub org.

Yanking `0.0.1` is a single action on PyPI and fixes the worst of it today.
P-01, P-02 and the metadata half of P-03 are together about an hour of work
and belong in the next release-prep commit.

**Registered as TD-PC-09 (P-03, High) and TD-PC-10 (P-01, Low); P-02 merges
into the existing TD-PC-04.**

---

*Refs: `docs/output/20260816_CC_REPO_AUDIT.md` F-03, F-04.*
