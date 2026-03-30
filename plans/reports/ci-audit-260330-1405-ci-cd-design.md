# CI/CD Audit & Design Report — pamola-core

**Date:** 2026-03-30
**FR:** FR-EP3-CORE-002 (Semantic Versioning)
**Branch:** develop
**Current version:** 1.0.0a1

---

## 1. Repository Audit

### Current State

| Item | Status | Detail |
|------|--------|--------|
| pyproject.toml version | 1.0.0a1 | PEP 440 compliant |
| `pamola_core.__version__` | via `importlib.metadata` | Falls back to `0.0.0+unknown` if not installed |
| CHANGELOG.md | Exists | Covers 0.0.1 → 1.0.0a1 |
| tests/test_version.py | Exists | PEP 440 format + CHANGELOG check |
| Tests | 3,529 pass, 0 fail | 127 test files across 15 modules |
| Package build | Builds correctly | `pamola_core-1.0.0a1-py3-none-any.whl` |
| CI workflows | **None** | No `.github/workflows/` existed |
| Ruff lint | 183 non-blocking errors | 73 E722 (bare-except), 61 F841 (unused-var) — cosmetic |
| Python compat | 3.10-3.12 | Enforced in pyproject.toml |
| Build system | setuptools + wheel | Standard, works |

### Issues Found

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | No CI/CD pipeline | High | Created `ci.yml` + `release.yml` |
| 2 | No automated release to PyPI | High | Created `release.yml` with Trusted Publishing |
| 3 | 183 ruff lint errors (non-fatal) | Low | CI runs ruff non-blocking; fatal errors (E9,F63,F7,F82) block |
| 4 | `lid.176.bin` (126MB) not in .gitignore | Medium | Added to .gitignore (uncommitted) |

---

## 2. Versioning Strategy

### Source of Truth

```
pyproject.toml → version = "X.Y.Z"  (single source)
pamola_core/_version.py → importlib.metadata reads it at runtime
pamola_core/__init__.py → re-exports __version__
```

### Version Format (PEP 440)

```
1.0.0a1     Alpha (feature development)
1.0.0b1     Beta (feature-complete, testing)
1.0.0rc1    Release Candidate (QA pass)
1.0.0       Stable release
1.0.1       Patch (bugfix)
1.1.0       Minor (new feature, backwards-compatible)
2.0.0       Major (breaking changes)
```

### Version Bump Process

1. Edit `pyproject.toml` → `version = "X.Y.Z"`
2. Update `CHANGELOG.md` → add `## [X.Y.Z] - YYYY-MM-DD` section
3. Commit: `chore: bump version to X.Y.Z`
4. Merge to main
5. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
6. CI validates tag == pyproject.toml version → publishes to PyPI

### CI Version Validation

- `ci.yml` build job: verifies wheel filename contains pyproject.toml version
- `release.yml` validate job: verifies git tag == pyproject.toml version == CHANGELOG entry
- `tests/test_version.py`: verifies PEP 440 format + CHANGELOG presence

---

## 3. CI Pipeline Architecture

```
Push/PR to main or develop
         │
         ├─► Lint (ruff)
         │     ├─ Fatal errors (E9,F63,F7,F82) → BLOCK
         │     └─ Other errors → REPORT (non-blocking)
         │
         ├─► Test (matrix: 3.10, 3.11, 3.12)
         │     └─ pytest tests/ → MUST PASS ALL
         │
         └─► Build (needs: lint + test)
               ├─ python -m build
               ├─ Verify version consistency
               └─ Upload artifacts
```

### Triggers

| Event | Branches | What runs |
|-------|----------|-----------|
| `push` | main, develop | lint → test → build |
| `pull_request` | main | lint → test → build |
| `push tag v*` | any | validate → publish to PyPI |

### Concurrency

- `cancel-in-progress: true` — new pushes cancel running CI for same branch

---

## 4. Release Pipeline

```
Developer creates tag: git tag v1.0.0a1
         │
         ├─► Validate
         │     ├─ Tag version == pyproject.toml version
         │     ├─ Version in CHANGELOG.md
         │     └─ All tests pass
         │
         └─► Publish (needs: validate)
               ├─ python -m build
               └─ PyPI Trusted Publishing (OIDC, no API tokens)
```

### PyPI Trusted Publishing Setup

Requires one-time configuration on PyPI:
1. Go to https://pypi.org/manage/project/pamola-core/settings/publishing/
2. Add GitHub as trusted publisher:
   - Owner: `DGT-Network`
   - Repository: `PAMOLA`
   - Workflow: `release.yml`
   - Environment: `pypi`

---

## 5. Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `.github/workflows/ci.yml` | **Created** | CI: lint + test (3.10-3.12) + build on push/PR |
| `.github/workflows/release.yml` | **Created** | Release: validate + publish to PyPI on tag v* |
| `CHANGELOG.md` | Already exists | Release history (created in pre-CI setup) |
| `tests/test_version.py` | Already exists | PEP 440 + CHANGELOG validation (created in pre-CI setup) |
| `pyproject.toml` | Already correct | version = 1.0.0a1 |
| `pamola_core/_version.py` | Already correct | importlib.metadata |

---

## 6. Developer Release Workflow

### For feature development (alpha bumps)

```bash
# 1. Implement feature on branch
git checkout -b feat/my-feature

# 2. Write code + tests
# 3. Run local checks
python -m pytest tests/ --tb=short -q
ruff check pamola_core/

# 4. Create PR to develop
# 5. CI runs lint + test + build → must pass
# 6. Merge PR

# 7. (Optional) Bump alpha for significant changes
# Edit pyproject.toml: 1.0.0a1 → 1.0.0a2
# Update CHANGELOG.md
# Commit + push
```

### For stable release

```bash
# 1. Ensure develop is stable (all tests pass)
git checkout develop
python -m pytest tests/ --tb=short -q

# 2. Bump version
# Edit pyproject.toml: 1.0.0a1 → 1.0.0
# Update CHANGELOG.md: move [Unreleased] items to [1.0.0]

# 3. Merge to main
git checkout main
git merge develop
git push origin main

# 4. Tag and push
git tag v1.0.0
git push origin v1.0.0

# 5. CI automatically:
#    - Validates tag == pyproject.toml == CHANGELOG
#    - Runs tests
#    - Publishes to PyPI
```

### Version bump cheatsheet

| Scenario | Version | Tag |
|----------|---------|-----|
| New alpha feature | 1.0.0a2 | v1.0.0a2 |
| Feature-complete beta | 1.0.0b1 | v1.0.0b1 |
| Release candidate | 1.0.0rc1 | v1.0.0rc1 |
| Stable release | 1.0.0 | v1.0.0 |
| Bugfix | 1.0.1 | v1.0.1 |
| New feature | 1.1.0 | v1.1.0 |
| Breaking change | 2.0.0 | v2.0.0 |

---

## 7. PyPI Trusted Publishing Setup (One-time)

**Required before first release:**

1. Login to https://pypi.org with the `pamola-core` project owner account
2. Navigate to: Project → Settings → Publishing
3. Add new publisher:
   - **Publisher:** GitHub Actions
   - **Owner:** `DGT-Network`
   - **Repository:** `PAMOLA`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
4. On GitHub, create environment `pypi` at:
   - Settings → Environments → New environment → Name: `pypi`
   - (Optional) Add protection rules: require approval for production releases

---

## Unresolved Questions

1. **PyPI Trusted Publishing** needs one-time setup on PyPI web UI — cannot be automated
2. **Ruff 183 non-blocking errors** — CI reports but doesn't block. Consider fixing E722 (bare-except) in a future PR
3. **Test duration** (~3.5 min) — may need optimization or parallel execution for CI matrix
