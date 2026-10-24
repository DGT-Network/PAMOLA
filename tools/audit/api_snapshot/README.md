# Public API snapshot (QA-03)

`generate_snapshot.py` records every name exported in `__all__` by `pamola_core` and its public
packages (kind, defining module, qualname, normalised signature) and the list of importable
public modules. `tests/test_api_snapshot.py` compares the installed package with
`api_snapshot.json` and fails with a readable diff when they differ.

| File | What it is |
|---|---|
| `api_snapshot.json` | the snapshot the test checks (the ASSEMBLED 1.0 tree, including CVPL) - regenerate it with `--write` in the same commit as any intended API change |
| `snapshots/develop_9623aee.json` | baseline: the public API of `develop` at 9623aee, before the 1.0 stabilization |
| `snapshots/final_tree_without_cvpl.json` | the integrated stabilization tree (ops + privacy + scope + release), without CVPL; replace `api_snapshot.json` with a fresh snapshot of the assembled tree before the release |

Human-readable diff: `python tools/audit/api_snapshot/generate_snapshot.py --diff OLD.json NEW.json --labels OLD NEW`.

The snapshot sees only what the packages export in `__all__` plus module names; a changed
default of a method that is not exported, or a changed result field, is not an API change here and
belongs in the CHANGELOG.
