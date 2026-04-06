# PAMOLA.CORE Documentation

This directory contains the Sphinx-based API reference documentation for the **pamola-core** library.

---

## Quick Start

### 1. Install documentation dependencies

```bash
pip install sphinx sphinx-autodoc-typehints
```

> `alabaster` (the HTML theme) is bundled with Sphinx — no extra package needed.

### 2. Build HTML documentation

From the **repository root**:

```bash
sphinx-build -b html docs docs/_build
```

Or using the bundled Makefile (Linux / macOS):

```bash
cd docs
make html
```

Or on Windows (PowerShell):

```powershell
cd docs
.\make.bat html
```

### 3. Build with strict warning checking

```bash
sphinx-build -W docs docs/_build
```

The `-W` flag turns all warnings into errors. The build is expected to complete with **0 warnings**.

### 4. Open the documentation

Open `docs/_build/index.html` in your browser.

---

## Directory Structure

```
docs/
├── conf.py                    ← Sphinx configuration (auto-generates index.rst on every build)
├── index.rst                  ← Auto-generated — do NOT edit manually (see conf.py)
├── README.md                  ← This file
├── Makefile                   ← Build shortcut (Linux/macOS)
├── make.bat                   ← Build shortcut (Windows)
│
├── _static/                   ← Static assets (CSS, images)
│   └── .gitkeep
│
├── _templates/
│   └── autosummary/
│       └── module.rst         ← Custom autosummary stub template (:no-imported-members:)
│
└── _build/                    ← HTML build output (git-ignored)
    └── generated/             ← Auto-generated RST stubs (git-ignored)
```

The `docs/` directory also contains Obsidian markdown vaults (`en/`, `ru/`, `specs/`, etc.)
that coexist with Sphinx without interference — they are listed in `exclude_patterns` in `conf.py`.

---

## How `index.rst` Is Generated

`conf.py` parses `pamola_core/__init__.py` using the Python `ast` module and automatically
writes `docs/index.rst` before every Sphinx build. The generated file contains an
`.. autosummary::` directive that lists `pamola_core` and every public top-level submodule
discovered from `__init__.py` import statements.

This means:

- **Never edit `docs/index.rst` manually** — your changes will be overwritten on the next build.
- When a new public submodule is added to `pamola_core/__init__.py`, it is automatically
  picked up by the next `sphinx-build` run with no further action needed.

---

## Sphinx Extensions Used

| Extension | Purpose |
|---|---|
| `sphinx.ext.autodoc` | Generate API docs from Python docstrings |
| `sphinx.ext.napoleon` | Support Google- and NumPy-style docstrings |
| `sphinx.ext.viewcode` | Add `[source]` links to every documented item |
| `sphinx.ext.autosummary` | Auto-generate stub pages for all submodules |
| `sphinx_autodoc_typehints` | Generate param/return docs from type hints |

---

## Building Without Full Dependencies

`conf.py` configures `autodoc_mock_imports` for all heavy optional dependencies
(PyTorch, spaCy, Dask, pandas, NumPy, etc.), so the documentation build succeeds
in a minimal environment:

```bash
pip install sphinx sphinx-autodoc-typehints
sphinx-build -b html docs docs/_build
```

---

## Docstring Style

Source code uses **NumPy-style** docstrings throughout. The `sphinx.ext.napoleon`
extension renders them automatically.

Correct format:

```python
def my_function(df, fields, **kwargs):
    """Short one-line description.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    fields : list of str
        Column names to process.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    pandas.DataFrame
        Processed dataset.

    Raises
    ------
    FieldNotFoundError
        If any column in ``fields`` is not present in ``df``.

    Examples
    --------
    >>> result = my_function(df, ["age", "name"])
    """
```

Key rules:

- Section headers (`Parameters`, `Returns`, `Raises`, `Examples`, `Notes`) must have **no trailing colon**.
- The dash underline must immediately follow the section header.
- Bullet lists inside parameter descriptions must be preceded by a **blank line**.
- Do not use Markdown code fences (`` ``` ``) — use RST `.. code-block::` directives instead.
- Module-level docstrings must NOT use `----` dash underlines as decorative separators — these are parsed as RST section titles and cause build errors.

> **Note for `conf.py` contributors:** `import re` is a required import used by the
> docstring-sanitizing hook near the end of the file. Some linters may flag it as
> unused (because the `re.compile()` calls are far from the import). Add `# noqa: F401`
> if your linter removes it automatically.

---

## Rebuilding After Code Changes

`autodoc` reflects source changes automatically. After modifying docstrings or adding
new public symbols, re-run:

```bash
sphinx-build -b html docs docs/_build
```

For a full clean rebuild (clears all cached files):

```bash
sphinx-build -b html docs docs/_build -E
```

For live auto-reloading during development (requires `sphinx-autobuild`):

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```
