# Releasing pamola-core

Releases are cut by pushing a `v*` tag. Everything after that is automated by
`.github/workflows/release.yml`; nothing is built or uploaded by hand.

## Flow

```text
tag v<version>
  validate   tag == pyproject version, tag is on the right branch, CHANGELOG mentions it
  build      wheel + sdist, built ONCE; filename version check; twine check --strict;
             uploaded as the `dist` artifact
  smoke      the EXACT wheel from `dist` in a fresh venv (no -e, no extras):
             pip check, import, `pamola-core --version`, `list-ops`, 9-case operation smoke
             ubuntu py3.10 / 3.11 / 3.12, windows py3.11, macos py3.12
  test       the test suite with coverage on py3.10 / 3.11 / 3.12 (source install)
  publish    consumes the same `dist` artifact, never rebuilds:
               pre-release tag (a, b, rc, .dev) or manual run -> TestPyPI  (environment: testpypi)
               stable tag                                      -> PyPI      (environment: pypi)
```

Pull requests run the same wheel build and smoke on Linux / Python 3.11
(`ci.yml`, job `Build: sdist + wheel`), through the same script,
`scripts/wheel_smoke.sh`, so a packaging break is caught before merge.

## Cutting a release

1. Set `version` in `pyproject.toml`; add a section for it to `CHANGELOG.md`.
2. Merge to `develop` (pre-releases) or `main` (stable).
3. Tag that commit: `git tag v1.0.0 && git push origin v1.0.0`.
4. Stable only: approve the `pypi` environment deployment when GitHub asks.
5. Check the published files and their attestations on the project page.

A manual run of the workflow ("Run workflow") builds, smokes and tests the selected
ref and publishes to TestPyPI only. Use it to rehearse before the first stable tag:
`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pamola-core==<version>`.

## Authentication: Trusted Publishing

There is no API token. The two publish jobs have `id-token: write` (and nothing else
beyond `contents: read`), request an OIDC token, and `pypa/gh-action-pypi-publish`
(pinned by commit SHA) exchanges it for a short-lived upload credential. Digital
attestations are generated and uploaded with the files.

## One-time setup (repository / PyPI administrators)

- GitHub environments:
  - `pypi`: required reviewers, deployment restricted to tags `v*`.
  - `testpypi`: no reviewers needed.
- Trusted publishers (PyPI project settings, Publishing):
  - pypi.org: owner `DGT-Network`, repository `PAMOLA`, workflow `release.yml`, environment `pypi`.
  - test.pypi.org: the same, environment `testpypi`.
- Branch protection: the required check `Build: sdist + wheel` keeps its name.
- Remove the `PYPI_API_TOKEN` repository secret once the first Trusted Publishing release has succeeded.

## Reproducing the smoke locally

```bash
python -m build
bash scripts/wheel_smoke.sh dist <version>      # needs exactly one wheel in dist/
```
