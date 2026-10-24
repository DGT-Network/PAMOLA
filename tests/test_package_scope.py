"""
PKG-01 / PKG-02 / PKG-03 / PKG-04 / TYPE-01: what ships matches the 1.0 scope.

Formal differential privacy and singling-out are not CORE 1.0 capabilities, so
neither the code nor the PyPI keywords may suggest them; the dead modules found
by the ground verification must stay gone; no ``py.typed`` marker may claim
PEP 561 support.
"""

import importlib
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
PACKAGE = ROOT / "pamola_core"


def _keywords():
    block = re.search(r"^keywords\s*=\s*\[(.*?)\]", PYPROJECT, re.S | re.M).group(1)
    return {k.lower() for k in re.findall(r'"([^"]+)"', block)}


def test_pypi_keywords_do_not_advertise_dp_or_synthetic_data():
    kw = _keywords()
    assert kw, "keywords list not found"
    assert not kw & {"differential-privacy", "synthetic-data"}


@pytest.mark.parametrize(
    "module",
    [
        "pamola_core.privacy_models.differential_privacy",
        "pamola_core.privacy_models.differential_privacy.calculation",
        "pamola_core.attacks.singling_out",
        "pamola_core.profiling.analyzers.email_dask",
        "pamola_core.profiling.analyzers.longtext",
        "pamola_core.privacy_models.l_diversity.report_generators",
    ],
)
def test_out_of_scope_and_dead_modules_are_not_importable(module):
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:  # parent package is gone as well
        spec = None
    assert spec is None


def test_no_out_of_scope_source_files_remain():
    names = {p.name for p in PACKAGE.rglob("*.py")}
    assert not {"singling_out.py", "email_dask.py", "longtext.py"} & names
    assert not (PACKAGE / "privacy_models" / "differential_privacy").exists()


def test_public_names_do_not_export_dp_or_singling_out():
    privacy_models = importlib.import_module("pamola_core.privacy_models")
    attacks = importlib.import_module("pamola_core.attacks")
    for module in (privacy_models, attacks):
        leaked = [
            n
            for n in module.__all__
            if "differentialprivacy" in n.lower() or "singling" in n.lower()
        ]
        assert leaked == [], leaked
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__} lists missing {name}"


def test_l_diversity_report_methods_are_not_shadowed():
    """Mis-indented 'convenience' methods used to replace the real ones."""
    from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport

    for name in ("generate_threshold_report", "generate_technical_report"):
        params = list(inspect.signature(getattr(LDiversityReport, name)).parameters)
        assert params[0] == "self", (name, params)


def test_sample_large_dataset_is_defined_once_and_honours_the_default():
    source = (PACKAGE / "anonymization/commons/visualization_utils.py").read_text(
        encoding="utf-8"
    )
    assert len(re.findall(r"^def sample_large_dataset\(", source, re.M)) == 1

    from pamola_core.anonymization.commons import visualization_utils as vu

    default = inspect.signature(vu.sample_large_dataset).parameters["max_samples"].default
    assert default == vu.DEFAULT_MAX_SAMPLES


def test_requirements_txt_is_gone():
    assert not (ROOT / "requirements.txt").exists()


# --- TYPE-01 ---------------------------------------------------------------


def test_no_py_typed_marker_and_no_typed_classifier():
    assert not list(PACKAGE.rglob("py.typed"))
    assert "Typing :: Typed" not in PYPROJECT
    assert not re.search(r'^\s*"?[^#\n]*py\.typed', PYPROJECT, re.M)
