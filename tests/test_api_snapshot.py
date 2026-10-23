"""QA-03: the public API only changes together with its snapshot (SRS v1.1 section 5.3).

``tools/audit/api_snapshot/api_snapshot.json`` records every name the public packages export in
``__all__`` (kind, module, qualname, normalised signature). This test builds the same snapshot
from the installed package and fails with a readable diff when they differ - a removed name, an
accidentally added one, or a changed signature. After an *intended* change:

    python tools/audit/api_snapshot/generate_snapshot.py --write tools/audit/api_snapshot/api_snapshot.json

then review the diff of the JSON, add a CHANGELOG entry (and a migration note for a breaking
change) and commit both.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit" / "api_snapshot" / "generate_snapshot.py"
SNAPSHOT = ROOT / "tools" / "audit" / "api_snapshot" / "api_snapshot.json"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="tools/ is not part of this checkout")


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("pamola_api_snapshot_generator", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass/typing helpers resolve the module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def live(generator):
    return generator.build_snapshot()


def test_public_api_matches_the_committed_snapshot(generator, live):
    committed = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    # A module that cannot be imported here (a missing optional extra) is ignored on both sides.
    unavailable = set(live["import_errors"]) | set(committed["import_errors"])

    def visible(api):
        return {k: v for k, v in api.items() if k.split(":")[0] not in unavailable}

    def visible_modules(snapshot):
        return [m for m in snapshot.get("modules", []) if not any(m == u or m.startswith(u + ".") for u in unavailable)]

    old = dict(committed, api=visible(committed["api"]), modules=visible_modules(committed))
    new = dict(live, api=visible(live["api"]), modules=visible_modules(live))
    diff = generator.diff_snapshots(old, new)
    if any(diff.values()):
        report = generator.render_markdown(old, new, "committed snapshot", "installed package")
        pytest.fail(
            "The public API differs from tools/audit/api_snapshot/api_snapshot.json.\n"
            "If the change is intended, regenerate the snapshot (see the module docstring), "
            "update the CHANGELOG and commit both.\n\n" + report[:6000]
        )


def test_the_snapshot_is_not_empty_and_covers_the_root_package(live):
    names = [k for k in live["api"] if k.startswith("pamola_core:")]
    assert len(live["api"]) > 100
    assert "pamola_core:FullMaskingOperation" in names


def test_signature_normalisation_ignores_annotations_and_addresses(generator):
    def sample(a: "list[int]", b=object(), *args, c: int = 3, **kw) -> "str":  # noqa: B008
        return ""

    assert generator.normalised_signature(sample) == "(a, b=<object object>, *args, c=3, **kw)"

    def keyword_only(a, *, flag=True):
        return a

    assert generator.normalised_signature(keyword_only) == "(a, *, flag=True)"


def test_diff_reports_added_removed_and_changed_names(generator):
    entry = {"kind": "function", "module": "m", "qualname": "f", "signature": "(a)"}
    old = {"api": {"m:f": entry, "m:gone": entry}}
    new = {"api": {"m:f": dict(entry, signature="(a, b=1)"), "m:new": entry}}
    diff = generator.diff_snapshots(old, new)
    assert diff["added"] == ["m:new"] and diff["removed"] == ["m:gone"] and diff["changed"] == ["m:f"]
    assert diff["modules_added"] == [] and diff["modules_removed"] == []
    gone = generator.diff_snapshots({"api": {}, "modules": ["a.b", "a.c"]}, {"api": {}, "modules": ["a.b"]})
    assert gone["modules_removed"] == ["a.c"]
    markdown = generator.render_markdown(old, new, "before", "after")
    assert "1 added, 1 removed, 1 changed" in markdown and "(a, b=1)" in markdown
