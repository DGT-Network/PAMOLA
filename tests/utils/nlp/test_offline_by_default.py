"""
SEC-01 / SEC-02: pamola-core is offline by default.

No default execution may open a network connection. Remote acquisition of NLP
resources (FastText model, spaCy models, NLTK corpora, Hugging Face models) is
allowed only after an explicit opt-in (``PAMOLA_ALLOW_REMOTE_DOWNLOADS=1`` or
``configure_remote_downloads(True)``). With the opt-in enabled and the network
unavailable the caller gets one actionable degraded-mode signal, not a silent
success.

The tests install a network interceptor at every level a download can start:
sockets, ``urllib``, ``requests`` and ``subprocess`` (``spacy.cli.download``
shells out to ``pip``).
"""

import logging
import socket
import subprocess
import sys
import types
import urllib.request

import pandas as pd
import pytest

from pamola_core.errors.exceptions import ModelLoadError, ModelNotAvailableError
from pamola_core.utils.nlp import language as language_mod
from pamola_core.utils.nlp import remote_resources as rr
from pamola_core.utils.nlp.model_manager import NLPModelManager


class NetworkBlocked(OSError):
    """Raised by the interceptor in place of a real connection."""


@pytest.fixture
def network_calls(monkeypatch):
    """Record (and refuse) every outbound network attempt."""
    calls = []

    def record(kind):
        def _blocked(*args, **kwargs):
            calls.append((kind, str(args[:2])))
            raise NetworkBlocked(f"network access blocked by test ({kind})")

        return _blocked

    monkeypatch.setattr(socket.socket, "connect", record("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", record("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", record("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", record("getaddrinfo"))
    monkeypatch.setattr(urllib.request, "urlopen", record("urlopen"))
    monkeypatch.setattr(urllib.request, "urlretrieve", record("urlretrieve"))
    monkeypatch.setattr(subprocess, "Popen", record("subprocess.Popen"))
    monkeypatch.setattr(subprocess, "run", record("subprocess.run"))
    try:
        import requests

        monkeypatch.setattr(requests.sessions.Session, "request", record("requests"))
    except ImportError:  # pragma: no cover - requests is a transitive dependency
        pass
    return calls


@pytest.fixture
def fake_fasttext(monkeypatch):
    """Make the FastText branch reachable without the real 125 MB model."""
    loaded = []
    module = types.ModuleType("fasttext")
    module.load_model = lambda path: loaded.append(path) or object()
    monkeypatch.setitem(sys.modules, "fasttext", module)
    monkeypatch.setattr(language_mod, "_FASTTEXT_AVAILABLE", True)
    # The model cache would otherwise hand back a model from an earlier test.
    from pamola_core.utils.nlp.cache import get_cache

    get_cache("model").clear()
    yield loaded
    get_cache("model").clear()


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    path = tmp_path / "models"
    monkeypatch.setenv("PAMOLA_MODEL_DIR", str(path))
    return path


def _fake_spacy(download_hits):
    """A spaCy stand-in: nothing installed, ``cli.download`` goes to the network."""
    spacy = types.ModuleType("spacy")

    def load(name, **kwargs):
        raise OSError(f"[E050] Can't find model '{name}'")

    def download(name, *args, **kwargs):
        download_hits.append(name)
        # What the real spacy.cli.download does first: fetch the compatibility
        # table from raw.githubusercontent.com.
        urllib.request.urlopen(
            "https://raw.githubusercontent.com/explosion/spacy-models/master/"
            "compatibility.json"
        )

    spacy.load = load
    spacy.cli = types.SimpleNamespace(download=download)
    return spacy


# --------------------------------------------------------------------------
# Opt-in switch
# --------------------------------------------------------------------------


class TestOptInSwitch:
    def test_default_is_offline(self):
        assert rr.remote_downloads_allowed() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_env_opt_in(self, monkeypatch, value):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, value)
        assert rr.remote_downloads_allowed() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "garbage"])
    def test_env_values_that_do_not_opt_in(self, monkeypatch, value):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, value)
        assert rr.remote_downloads_allowed() is False

    def test_programmatic_override_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        rr.configure_remote_downloads(False)
        assert rr.remote_downloads_allowed() is False
        rr.configure_remote_downloads(True)
        monkeypatch.delenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS)
        assert rr.remote_downloads_allowed() is True


# --------------------------------------------------------------------------
# Vector 1: FastText language-identification model
# --------------------------------------------------------------------------


class TestFastTextVector:
    def test_default_makes_no_request_and_falls_back(
        self, network_calls, fake_fasttext, model_dir
    ):
        assert language_mod._get_fasttext_model() is None

        assert network_calls == []
        assert fake_fasttext == []  # nothing to load
        assert not model_dir.exists()  # no stray directory either
        reasons = rr.get_degraded_resources()
        assert list(reasons) == ["fasttext:lid.176"]
        assert "PAMOLA_ALLOW_REMOTE_DOWNLOADS" in reasons["fasttext:lid.176"]

    def test_language_detection_still_works_offline(
        self, network_calls, fake_fasttext, model_dir
    ):
        lang, _ = language_mod.detect_language_with_confidence(
            "Привет, как дела у вас сегодня?"
        )
        assert lang == "ru"
        assert network_calls == []

    def test_opt_in_with_network_down_gives_one_actionable_signal(
        self, network_calls, fake_fasttext, model_dir, monkeypatch, caplog
    ):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        with caplog.at_level(logging.WARNING):
            first = language_mod._get_fasttext_model()
            second = language_mod._get_fasttext_model()

        assert first is None and second is None
        # Exactly one attempt: a failed download is remembered, not retried per call.
        assert [c[0] for c in network_calls] == ["urlretrieve"]
        reasons = rr.get_degraded_resources()
        assert list(reasons) == ["fasttext:lid.176"]
        assert "network access" in reasons["fasttext:lid.176"]
        assert "lid.176.bin" in reasons["fasttext:lid.176"]
        warnings = [
            r for r in caplog.records if "fasttext:lid.176" in r.getMessage()
        ]
        assert len(warnings) == 1
        # A failed attempt leaves nothing behind.
        assert not model_dir.exists()

    def test_opt_in_downloads_only_through_the_guarded_path(
        self, fake_fasttext, model_dir, monkeypatch
    ):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        requested = []

        def fake_urlretrieve(url, dest):
            requested.append(url)
            with open(dest, "wb") as f:
                f.write(b"model")

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

        assert language_mod._get_fasttext_model() is not None
        assert requested == [
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        ]
        assert rr.get_degraded_resources() == {}

    def test_existing_local_model_needs_no_network_and_no_opt_in(
        self, network_calls, fake_fasttext, model_dir
    ):
        model_dir.mkdir()
        (model_dir / "lid.176.bin").write_bytes(b"model")

        assert language_mod._get_fasttext_model() is not None
        assert network_calls == []
        assert rr.get_degraded_resources() == {}


# --------------------------------------------------------------------------
# Vector 2: spaCy / model manager / NLTK
# --------------------------------------------------------------------------


class TestSpacyVector:
    @pytest.fixture
    def manager(self, monkeypatch):
        return NLPModelManager()

    def _patch_spacy(self, monkeypatch, manager, hits):
        from pamola_core.utils.nlp.base import DependencyManager

        fake = _fake_spacy(hits)
        real_get_module = DependencyManager.get_module
        monkeypatch.setattr(
            DependencyManager,
            "get_module",
            staticmethod(
                lambda name, *a, **k: fake
                if name == "spacy"
                else real_get_module(name, *a, **k)
            ),
        )
        monkeypatch.setattr(manager, "_nlp_libraries", {"spacy"})
        manager._model_cache.clear()
        return fake

    def test_default_never_calls_spacy_download(
        self, network_calls, manager, monkeypatch
    ):
        hits = []
        self._patch_spacy(monkeypatch, manager, hits)

        with pytest.raises(ModelNotAvailableError) as exc:
            manager.get_model("spacy", "en")

        assert hits == []
        assert network_calls == []
        assert "PAMOLA_ALLOW_REMOTE_DOWNLOADS" in str(exc.value)
        assert "spacy:en_core_web_sm" in rr.get_degraded_resources()

    def test_opt_in_with_network_down_is_typed_and_attempted_once(
        self, network_calls, manager, monkeypatch
    ):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        hits = []
        self._patch_spacy(monkeypatch, manager, hits)

        for _ in range(3):
            with pytest.raises(ModelLoadError) as exc:
                manager.get_model("spacy", "en")

        assert hits == ["en_core_web_sm"]  # not retried per call
        assert [c[0] for c in network_calls] == ["urlopen"]
        reason = rr.get_degraded_resources()["spacy:en_core_web_sm"]
        assert "download of spaCy model 'en_core_web_sm' failed" in reason
        assert "check network access" in reason
        assert "check network access" in str(exc.value)

    def test_spacy_cli_sys_exit_is_contained(self, monkeypatch):
        """spacy.cli.download calls sys.exit() when pip fails."""
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        spacy = types.SimpleNamespace(
            cli=types.SimpleNamespace(download=lambda name: sys.exit(1))
        )
        assert rr.download_spacy_model(spacy, "xx_model") is False
        assert "spacy:xx_model" in rr.get_degraded_resources()

    def test_disabled_download_helpers_do_not_touch_the_library(self):
        boom = types.SimpleNamespace(
            cli=types.SimpleNamespace(download=lambda *a, **k: pytest.fail("called")),
            download=lambda *a, **k: pytest.fail("called"),
        )
        assert rr.download_spacy_model(boom, "m") is False
        assert rr.download_nltk_resource(boom, "stopwords") is False
        assert set(rr.get_degraded_resources()) == {"spacy:m", "nltk:stopwords"}


class TestNltkVector:
    def test_opt_in_download_returning_false_is_a_failure(self, monkeypatch):
        """nltk.download reports network errors by returning False, not raising."""
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        nltk = types.SimpleNamespace(download=lambda *a, **k: False)
        assert rr.download_nltk_resource(nltk, "wordnet") is False
        assert "nltk:wordnet" in rr.get_degraded_resources()

    def test_opt_in_download_success(self, monkeypatch):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        nltk = types.SimpleNamespace(download=lambda *a, **k: True)
        assert rr.download_nltk_resource(nltk, "wordnet") is True
        assert rr.get_degraded_resources() == {}

    def test_setup_nlp_resources_offline_by_default(self, network_calls):
        from pamola_core.utils.nlp.compatibility import setup_nlp_resources

        status = setup_nlp_resources(download_if_missing=True)

        assert network_calls == []
        # Whatever is missing stays missing, and says why.
        for name, available in status.items():
            if not available:
                kind, _, resource = name.partition("_")
                assert f"{kind}:{resource}" in rr.get_degraded_resources()


class TestTransformersVector:
    def test_hub_download_requires_opt_in(self, monkeypatch):
        seen = []

        def from_pretrained(name, **kwargs):
            seen.append(kwargs)
            raise OSError("not cached")

        fake = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained),
            AutoModel=types.SimpleNamespace(from_pretrained=from_pretrained),
        )
        manager = NLPModelManager()
        monkeypatch.setattr(manager, "_nlp_libraries", {"transformers"})
        from pamola_core.utils.nlp.base import DependencyManager

        monkeypatch.setattr(
            DependencyManager, "get_module", staticmethod(lambda *a, **k: fake)
        )

        with pytest.raises(ModelLoadError) as exc:
            manager.get_model("transformers", "en")
        assert seen == [{"local_files_only": True}]
        assert "PAMOLA_ALLOW_REMOTE_DOWNLOADS" in str(exc.value)

        seen.clear()
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        with pytest.raises(ModelLoadError):
            manager.get_model("transformers", "en", task="other")
        assert seen == [{"local_files_only": False}]


# --------------------------------------------------------------------------
# End to end: the flagship text-profiling operation with default settings
# --------------------------------------------------------------------------


class TestDefaultTextProfilingIsOffline:
    def _run(self, tmp_path):
        from pamola_core.profiling.analyzers.text import (
            TextSemanticCategorizerOperation,
        )
        from pamola_core.utils.ops.op_data_source import DataSource
        from unittest.mock import MagicMock

        df = pd.DataFrame(
            {
                "bio": [
                    "Senior data engineer at Acme Corporation",
                    "Marketing manager, Globex Inc",
                    "Привет, я работаю учителем в школе",
                ]
            }
        )
        op = TextSemanticCategorizerOperation(
            field_name="bio", use_cache=False, generate_visualization=False
        )
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        return op.execute(
            data_source=DataSource(dataframes={"main": df}),
            task_dir=task_dir,
            reporter=MagicMock(),
        )

    def test_default_run_makes_zero_network_calls_and_reports_degradation(
        self, network_calls, tmp_path
    ):
        result = self._run(tmp_path)

        assert network_calls == [], f"unexpected outbound attempts: {network_calls}"
        # The run is allowed to succeed (weaker fallbacks), but it must say so.
        degraded = result.metrics.get("nlp_resources_degraded")
        assert degraded, "offline fallback must be visible in the result"
        assert any(k.startswith(("spacy:", "fasttext:")) for k in degraded)
        for reason in degraded.values():
            assert "PAMOLA_ALLOW_REMOTE_DOWNLOADS" in reason

    def test_opt_in_with_network_down_attempts_once_and_reports(
        self, network_calls, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        result = self._run(tmp_path)

        assert network_calls, "opt-in must actually try the network"
        degraded = result.metrics.get("nlp_resources_degraded")
        assert degraded
        assert all("failed" in reason for reason in degraded.values())
        # No per-record retry storm: one attempt per failed resource (the
        # pre-fix run made one attempt per record and per call site).
        assert len(network_calls) == len(degraded)


# --------------------------------------------------------------------------
# CDX-AC-9: the standalone TransformersTokenizer must obey the same policy
# --------------------------------------------------------------------------


class TestTransformersTokenizerPolicy:
    @pytest.fixture
    def tokenizer_env(self, monkeypatch):
        from pamola_core.utils.nlp import tokenization
        from pamola_core.utils.nlp.base import DependencyManager

        seen = []

        def from_pretrained(name, **kwargs):
            seen.append((name, kwargs))
            raise OSError("not in the local cache")

        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "transformers", fake)
        monkeypatch.setattr(tokenization, "_TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(
            DependencyManager,
            "get_module",
            staticmethod(lambda name, *a, **k: fake if name == "transformers" else None),
        )
        return tokenization, seen

    def test_default_loads_local_files_only_and_reports_degradation(self, tokenizer_env):
        tokenization, seen = tokenizer_env
        tok = tokenization.TransformersTokenizer(model_name="some/model")
        assert tok._init_tokenizer() is None

        assert seen == [("some/model", {"local_files_only": True})]
        reason = rr.get_degraded_resources()["transformers:some/model"]
        assert "PAMOLA_ALLOW_REMOTE_DOWNLOADS" in reason

    def test_opt_in_may_reach_the_hub(self, tokenizer_env, monkeypatch):
        tokenization, seen = tokenizer_env
        monkeypatch.setenv(rr.ENV_ALLOW_REMOTE_DOWNLOADS, "1")
        tokenization.TransformersTokenizer(model_name="some/model")._init_tokenizer()

        assert seen == [("some/model", {"local_files_only": False})]
        assert "check network access" in rr.get_degraded_resources()["transformers:some/model"]

    def test_no_from_pretrained_call_in_the_package_bypasses_the_policy(self):
        """Every Hub entry point must pass the policy kwargs (grep-level guard)."""
        import pathlib
        import re

        root = pathlib.Path(rr.__file__).resolve().parents[2]
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"from_pretrained\(", text):
                call = text[match.start() : match.start() + 200]
                if "hub_kwargs" not in call and "hub_load_kwargs" not in call:
                    offenders.append(f"{path.name}:{text[: match.start()].count(chr(10)) + 1}")
        assert offenders == []
