"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Remote NLP Resource Policy
Package:       pamola_core.utils.nlp
Status:        stable
License:       BSD 3-Clause

Description:
Single choke point for every code path that may fetch a model or corpus from
the network (FastText language-identification model, spaCy models, NLTK
corpora, Hugging Face Hub models).

pamola-core is offline by default: no NLP resource is downloaded unless the
caller opts in explicitly, either with the environment variable
``PAMOLA_ALLOW_REMOTE_DOWNLOADS=1`` or programmatically with
:func:`configure_remote_downloads`.

When a resource cannot be obtained (downloads disabled, or opted in but the
network is unavailable) the failure is recorded once per resource in a
process-wide registry (:func:`get_degraded_resources`). Operations surface
that registry in their results, so a run that silently fell back to a weaker
method is visible to the caller instead of looking like an ordinary success.
"""

import logging
import os
import threading
from typing import Dict, Optional

from pamola_core.errors.exceptions.nlp import ModelNotAvailableError

logger = logging.getLogger(__name__)

#: Environment variable that opts in to remote model/corpus acquisition.
ENV_ALLOW_REMOTE_DOWNLOADS = "PAMOLA_ALLOW_REMOTE_DOWNLOADS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_lock = threading.Lock()
_override: Optional[bool] = None
_degraded: Dict[str, str] = {}
# Resources whose download was actually attempted and failed in this process.
_failed_downloads: set = set()


def configure_remote_downloads(enabled: Optional[bool]) -> None:
    """
    Override the environment setting for remote resource acquisition.

    Parameters
    ----------
    enabled : bool or None
        ``True`` allows downloads, ``False`` forbids them regardless of the
        environment, ``None`` removes the override (environment decides).
    """
    global _override
    _override = enabled


def remote_downloads_allowed() -> bool:
    """Return True only if the caller explicitly opted in to remote downloads."""
    if _override is not None:
        return _override
    return os.environ.get(ENV_ALLOW_REMOTE_DOWNLOADS, "").strip().lower() in _TRUTHY


def record_degraded(resource: str, reason: str) -> bool:
    """
    Record that ``resource`` is unavailable; log the first occurrence only.

    Returns
    -------
    bool
        True if this is the first report for ``resource`` in this process.
    """
    with _lock:
        first = resource not in _degraded
        _degraded[resource] = reason
    if first:
        logger.warning("NLP resource '%s' unavailable: %s", resource, reason)
    return first


def is_degraded(resource: str) -> bool:
    """Return True if ``resource`` was already reported unavailable."""
    with _lock:
        return resource in _degraded


def get_degraded_resources() -> Dict[str, str]:
    """Return a copy of the ``{resource: reason}`` registry."""
    with _lock:
        return dict(_degraded)


def reset_degraded_resources() -> None:
    """Clear the registry (new session, or after the user fixed the setup)."""
    with _lock:
        _degraded.clear()
        _failed_downloads.clear()


def mark_download_failed(resource: str, reason: str) -> None:
    with _lock:
        _failed_downloads.add(resource)
    record_degraded(resource, reason)


def download_already_failed(resource: str) -> bool:
    with _lock:
        return resource in _failed_downloads


def disabled_reason(what: str) -> str:
    """Actionable message for a resource that is missing while downloads are off."""
    return (
        f"{what} is not installed locally and remote downloads are disabled "
        f"(offline by default). Install it manually, or opt in to downloads "
        f"with {ENV_ALLOW_REMOTE_DOWNLOADS}=1."
    )


def require_remote_downloads(resource: str, what: Optional[str] = None) -> None:
    """
    Raise a typed error unless remote downloads were explicitly enabled.

    Raises
    ------
    ModelNotAvailableError
        If downloads are disabled. The failure is also recorded in the
        degraded-resource registry.
    """
    if remote_downloads_allowed():
        return
    reason = disabled_reason(what or resource)
    record_degraded(resource, reason)
    raise ModelNotAvailableError(model_name=resource, reason=reason)


def download_spacy_model(spacy_module, model_name: str) -> bool:
    """
    Download a spaCy model if (and only if) remote downloads are enabled.

    Returns True on success. On refusal or failure the resource is recorded
    as degraded and False is returned; nothing is raised (``spacy.cli.download``
    calls ``sys.exit`` on failure, which is also contained here).
    """
    resource = f"spacy:{model_name}"
    if not remote_downloads_allowed():
        record_degraded(resource, disabled_reason(f"spaCy model '{model_name}'"))
        return False
    if download_already_failed(resource):
        # Already tried and failed in this process; do not hammer the network.
        return False
    logger.info("Downloading spaCy model '%s' (remote downloads enabled)", model_name)
    try:
        spacy_module.cli.download(model_name)
        return True
    except (Exception, SystemExit) as e:
        mark_download_failed(
            resource,
            f"download of spaCy model '{model_name}' failed ({e!r}); "
            "check network access or install the model manually.",
        )
        return False


def download_nltk_resource(nltk_module, name: str, force: bool = False) -> bool:
    """
    Download an NLTK resource if (and only if) remote downloads are enabled.

    ``nltk.download`` reports most network failures by returning False rather
    than raising; both forms are treated as failure here.
    """
    resource = f"nltk:{name}"
    if not remote_downloads_allowed():
        record_degraded(resource, disabled_reason(f"NLTK resource '{name}'"))
        return False
    if download_already_failed(resource):
        return False
    logger.info("Downloading NLTK resource '%s' (remote downloads enabled)", name)
    try:
        ok = nltk_module.download(name, quiet=True, force=force)
    except Exception as e:
        mark_download_failed(
            resource,
            f"download of NLTK resource '{name}' failed ({e!r}); "
            "check network access or install the corpus manually.",
        )
        return False
    if ok is False:
        mark_download_failed(
            resource,
            f"download of NLTK resource '{name}' failed; "
            "check network access or install the corpus manually.",
        )
        return False
    return True
