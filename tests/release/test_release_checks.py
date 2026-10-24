"""
CDX-R2-14 / 15 / 16 and the CRLF defect: release-workflow checks.

The workflow delegates its decisions to scripts that can be tested here:
version classification (PEP 440), digest comparison for TestPyPI, and the
branch-ancestry rule (reproduced against a scratch git repository).
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_version = _load("release_version")
digests = _load("check_testpypi_digests")


# --- CDX-R2-15: PEP 440 classification ----------------------------------------


@pytest.mark.parametrize(
    "version,prerelease",
    [
        ("1.0.0", False),
        ("1.0.0.post1", False),
        ("1.0.0rc1", True),
        ("1.0.0a1", True),
        ("1.0.0b2", True),
        ("1.0.0.dev4", True),
        ("1.0.0rc1.post1", True),  # the regex called these stable
        ("1.0.0a1.post2", True),
        ("1.0.0rc1.dev3", True),
    ],
)
def test_prerelease_classification_follows_pep440(version, prerelease):
    assert release_version.classify(version) == (version, prerelease)


@pytest.mark.parametrize("bad", ["1.0.0+local", "1.0.0rc1+abc.1", "not-a-version", ""])
def test_local_and_invalid_versions_are_rejected(bad):
    with pytest.raises(ValueError):
        release_version.classify(bad)


def test_tag_must_match_the_package_version():
    release_version.check_tag("v1.0.0rc1", "1.0.0rc1")
    with pytest.raises(ValueError):
        release_version.check_tag("v1.0.0", "1.0.0rc1")


def test_noncanonical_pyproject_version_fails_the_script(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0.0-rc1"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("tomllib")
    assert release_version.main(["x", "branch", "main"]) == 1
    assert "canonical" in capsys.readouterr().err


# --- CDX-R2-16: TestPyPI byte comparison ----------------------------------------


def _release(files):
    return {"urls": [{"filename": n, "digests": {"sha256": d}} for n, d in files.items()]}


def test_identical_existing_file_may_be_skipped():
    assert digests.mismatches({"a.whl": "aa"}, _release({"a.whl": "aa"})) == []


def test_different_bytes_under_the_same_name_fail():
    assert digests.mismatches({"a.whl": "aa", "a.tar.gz": "bb"}, _release({"a.whl": "zz"})) == ["a.whl"]


def test_nothing_published_yet_is_fine():
    assert digests.mismatches({"a.whl": "aa"}, None) == []
    assert digests.mismatches({"a.whl": "aa"}, {}) == []


def test_script_fails_on_mismatch(tmp_path, monkeypatch, capsys):
    (tmp_path / "p-1.0.0.whl").write_bytes(b"new bytes")
    remote = _release({"p-1.0.0.whl": "0" * 64})
    monkeypatch.setattr(digests, "fetch_release", lambda package, version: remote)
    assert digests.main(["x", str(tmp_path), "p", "1.0.0"]) == 1
    assert "different bytes" in capsys.readouterr().err


# --- CDX-R2-14: branch ancestry ----------------------------------------------------


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"],
             "HOME": str(cwd)},
    ).stdout.strip()


@pytest.fixture
def unreviewed_commit(tmp_path):
    """A commit that exists only on `origin/main-unreviewed`, never on `origin/main`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f").write_text("1")
    _git(repo, "add", "f")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "main-unreviewed")
    (repo / "f").write_text("2")
    _git(repo, "commit", "-qam", "unreviewed")
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    _git(repo, "update-ref", "refs/remotes/origin/main-unreviewed", commit)
    return repo, commit


def test_old_predicate_accepted_an_unmerged_commit_new_rule_rejects_it(unreviewed_commit):
    repo, commit = unreviewed_commit
    contains = _git(repo, "branch", "-r", "--contains", commit)
    assert "origin/main" in contains  # prefix match on origin/main-unreviewed: the old bug

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
        cwd=repo,
    )
    assert result.returncode != 0


def test_workflow_uses_the_ancestry_rule_and_the_digest_check():
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "git merge-base --is-ancestor" in text
    assert "branch -r --contains" not in text
    assert "scripts/release_version.py" in text
    assert "scripts/check_testpypi_digests.py" in text
    assert text.index("check_testpypi_digests.py") < text.index("skip-existing")


# --- line endings -------------------------------------------------------------------


def test_shell_scripts_use_lf_and_attributes_enforce_it():
    assert b"\r\n" not in (ROOT / "scripts/wheel_smoke.sh").read_bytes()
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes and "*.py text eol=lf" in attributes


# --- ASB-5: CHANGELOG heading, version rules, no-network smoke, supply-chain job ---------------

check_changelog = _load("check_changelog")


@pytest.mark.parametrize(
    "text,version,ok",
    [
        ("## [1.0.0.dev4] - 2026-08-16\n", "1.0.0.dev4", True),
        ("## [1.0.0.dev4] - 2026-08-16\n", "1.0.0", False),  # a substring of the heading is not enough
        ("see 1.0.0 in prose\n## [Unreleased]\n", "1.0.0", False),
        ("## [1.0.0]\n", "1.0.0", True),
        ("## [1.0.0.post1] - x\n", "1.0.0", False),
        ("## [10.0.0] - x\n", "1.0.0", False),
        ("## [1x0x0] - x\n", "1.0.0", False),  # the dots are literal
    ],
)
def test_changelog_needs_the_exact_version_heading(text, version, ok):
    assert check_changelog.has_release_heading(text, version) is ok


def test_release_workflow_uses_the_exact_changelog_check():
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "scripts/check_changelog.py" in text
    assert 'grep -q "$VERSION" CHANGELOG.md' not in text


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0.post1", "1.0.0rc1", "1.0.0.dev4", "1.0.0rc1.post1"])
def test_test_version_and_release_script_accept_the_same_versions(version):
    from packaging.version import Version

    canonical, _ = release_version.classify(version)
    assert str(Version(version)) == canonical == version  # what tests/test_version.py checks


def test_the_no_network_smoke_guard_refuses_internet_sockets():
    clean = _load("clean_wheel_smoke")
    code = (
        "import sys; sys.path.insert(0, %r); import clean_wheel_smoke as c, socket, urllib.request\n"
        "c.block_network()\n"
        "for fn in (lambda: socket.create_connection(('example.com', 80)),\n"
        "           lambda: urllib.request.urlopen('http://example.com', timeout=1),\n"
        "           lambda: socket.socket().connect(('127.0.0.1', 9))):\n"
        "    try:\n"
        "        fn()\n"
        "    except c.NetworkBlocked:\n"
        "        continue\n"
        "    except Exception as e:\n"
        "        # urllib wraps the OSError in URLError\n"
        "        if isinstance(getattr(e, 'reason', None), c.NetworkBlocked):\n"
        "            continue\n"
        "        raise SystemExit('not blocked: ' + repr(e))\n"
        "    raise SystemExit('connected')\n"
        "print('blocked')\n" % str(ROOT / "scripts")
    )
    assert clean.NetworkBlocked
    out = subprocess.run([__import__("sys").executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "blocked" in out.stdout, out.stdout + out.stderr


def test_smoke_script_runs_the_cases_with_the_network_refused_and_ci_audits_the_wheel():
    assert "--no-network" in (ROOT / "scripts/wheel_smoke.sh").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pip-audit" in ci and "--path" in ci and "pip-licenses" in ci
    assert "dependency-licenses" in ci
