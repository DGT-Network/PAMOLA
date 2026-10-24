#!/usr/bin/env python
"""Snapshot of the public API of ``pamola_core`` (SRS v1.1 section 5.3, QA-03).

The public surface is what the package declares in ``__all__``: ``pamola_core.__all__`` and the
``__all__`` of every module under the public packages (SRS 3.1). For each exported name the
snapshot records ``kind``, defining ``module``, ``qualname`` and ``signature``.

The signature is normalised on purpose: parameter names, kinds and default values only - no
annotations and no return annotation - so the snapshot does not change between Python versions
(``Optional[str]`` prints differently from 3.10 to 3.14) or with memory addresses.

Usage::

    python tools/audit/api_snapshot/generate_snapshot.py --write tools/audit/api_snapshot/api_snapshot.json
    python tools/audit/api_snapshot/generate_snapshot.py --diff OLD.json NEW.json > api_diff.md

A module that cannot be imported in the current environment (a missing optional extra) is listed
under ``import_errors`` and contributes no names.
"""

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import json
import pkgutil
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: Public packages of SRS v1.1 section 3.1 (``pamola_core.utils`` is a curated surface; its
#: ``__all__`` lists are what counts).
PUBLIC_PACKAGES = (
    "pamola_core.anonymization",
    "pamola_core.profiling",
    "pamola_core.transformations",
    "pamola_core.metrics",
    "pamola_core.fake_data",
    "pamola_core.analysis",
    "pamola_core.io",
    "pamola_core.utils",
    "pamola_core.errors",
    "pamola_core.cli",
    # not listed in SRS 3.1, but importable modules a user may depend on: removing them is an
    # API change that must show up in the diff
    "pamola_core.privacy_models",
    "pamola_core.attacks",
)

_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")


def _default(value: Any) -> str:
    return _ADDRESS.sub("", repr(value))


def normalised_signature(obj: Any) -> str:
    """``(a, b=1, *args, c=None, **kw)`` - names, kinds and defaults only."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(?)"
    parts: List[str] = []
    seen_keyword_only = False
    for p in signature.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            seen_keyword_only = True
            continue
        if p.kind is p.KEYWORD_ONLY and not seen_keyword_only:
            parts.append("*")
            seen_keyword_only = True
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
            continue
        parts.append(p.name if p.default is p.empty else f"{p.name}={_default(p.default)}")
    return "(" + ", ".join(parts) + ")"


def describe(name: str, obj: Any) -> Dict[str, str]:
    if inspect.isclass(obj):
        kind = "class"
    elif inspect.ismodule(obj):
        return {"kind": "module", "module": getattr(obj, "__name__", ""), "qualname": "", "signature": ""}
    elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
        kind = "function"
    elif callable(obj):
        kind = "callable"
    else:
        return {"kind": "constant", "module": "", "qualname": "", "signature": type(obj).__name__}
    if kind == "class" and issubclass(obj, enum.Enum):
        # An Enum's constructor signature comes from the interpreter (3.11 adds
        # ``boundary``), not from us; its public surface is its members.
        signature = "members: [" + ", ".join(obj.__members__) + "]"
    else:
        signature = normalised_signature(obj)
    return {
        "kind": kind,
        "module": getattr(obj, "__module__", "") or "",
        "qualname": getattr(obj, "__qualname__", name),
        "signature": signature,
    }


def _iter_modules(package_name: str, errors: Dict[str, str]) -> Iterable[str]:
    try:
        package = importlib.import_module(package_name)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        errors[package_name] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return
    yield package_name
    path = getattr(package, "__path__", None)
    if not path:
        return

    def on_error(name: str) -> None:
        errors[name] = "import failed while walking the package"

    for info in pkgutil.walk_packages(path, prefix=package_name + ".", onerror=on_error):
        yield info.name


def build_snapshot(packages: Iterable[str] = PUBLIC_PACKAGES) -> Dict[str, Any]:
    warnings.filterwarnings("ignore")
    packages = tuple(packages)
    errors: Dict[str, str] = {}
    api: Dict[str, Dict[str, str]] = {}
    modules: set = set()

    def collect(module_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            errors[module_name] = f"{type(exc).__name__}: {str(exc)[:120]}"
            return
        if not any(part.startswith("_") for part in module_name.split(".")):
            modules.add(module_name)  # importable public module (removal of a module is an API change)
        names = getattr(module, "__all__", None)
        if not names:
            return
        for name in names:
            if not hasattr(module, name):
                api[f"{module_name}:{name}"] = {"kind": "missing", "module": "", "qualname": "", "signature": ""}
                continue
            api[f"{module_name}:{name}"] = describe(name, getattr(module, name))

    collect("pamola_core")
    for package in packages:
        for module_name in _iter_modules(package, errors):
            collect(module_name)
    return {
        "format": 1,
        "packages": list(packages),
        "modules": sorted(modules),
        "api": dict(sorted(api.items())),
        "import_errors": dict(sorted(errors.items())),
    }


def diff_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, List[str]]:
    a, b = old["api"], new["api"]
    ma, mb = set(old.get("modules", [])), set(new.get("modules", []))
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k]),
        "modules_added": sorted(mb - ma),
        "modules_removed": sorted(ma - mb),
    }


def render_markdown(old: Dict[str, Any], new: Dict[str, Any], old_label: str, new_label: str) -> str:
    d = diff_snapshots(old, new)
    lines = [
        f"# Public API diff: {old_label} -> {new_label}",
        "",
        f"{len(old['api'])} exported names before, {len(new['api'])} after: "
        f"**{len(d['added'])} added, {len(d['removed'])} removed, {len(d['changed'])} changed**.",
        "",
    ]
    for title, keys, source in (("Added", d["added"], new["api"]), ("Removed", d["removed"], old["api"])):
        lines += [f"## {title} ({len(keys)})", ""]
        for key in keys:
            e = source[key]
            shown = e["signature"] if e["kind"] in ("class", "function", "callable") else ""
            lines.append(f"- `{key}` - {e['kind']} `{e['qualname']}{shown}`")
        lines.append("")
    for title, keys in (("Modules added", d["modules_added"]), ("Modules removed", d["modules_removed"])):
        lines += [f"## {title} ({len(keys)})", ""]
        lines += [f"- `{m}`" for m in keys]
        lines.append("")
    lines += [f"## Changed ({len(d['changed'])})", ""]
    for key in d["changed"]:
        o, n = old["api"][key], new["api"][key]
        lines.append(f"- `{key}`")
        for field in ("kind", "module", "qualname", "signature"):
            if o[field] != n[field]:
                lines.append(f"    - {field}: `{o[field]}` -> `{n[field]}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", metavar="PATH", help="write the snapshot of the installed package")
    parser.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"), help="print a markdown diff of two snapshots")
    parser.add_argument("--labels", nargs=2, default=("old", "new"))
    args = parser.parse_args(argv)
    if args.diff:
        old, new = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.diff)
        sys.stdout.write(render_markdown(old, new, *args.labels))
        return 0
    snapshot = build_snapshot()
    text = json.dumps(snapshot, indent=1, sort_keys=True) + "\n"
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8", newline="\n")
        print(f"{len(snapshot['api'])} names, {len(snapshot['import_errors'])} import errors -> {args.write}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
