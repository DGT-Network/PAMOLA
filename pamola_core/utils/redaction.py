"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Secret redaction for diagnostics
Package:       pamola_core.utils
License:       BSD 3-Clause

ONE definition of "this parameter is a secret" and ONE way to remove it, used at every
boundary where parameters are serialized for people or for disk: ``config.json``
(``OperationConfig.to_safe_dict``), reporter events, operation logs, cache metadata and
task reports. A secret is a parameter whose name says so (``*_key``, ``password``,
``token``, ``secret``, ``salt``, ``pepper``...) or that the configuration declares in
``SENSITIVE_KEYS``.

No heavy imports on purpose: this module is imported by the configuration layer.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional

#: Placeholder written instead of a secret value.
REDACTED = "*REDACTED*"

_SENSITIVE_NAMES = frozenset(
    {
        "encryption_key", "password", "token", "secret", "api_key", "salt",
        "pepper", "passphrase", "private_key", "key",
    }
)
_SENSITIVE_SUFFIXES = (
    "_key", "_password", "_secret", "_token", "_passphrase", "_salt", "_pepper",
)


def is_sensitive_parameter(name: Any) -> bool:
    """True if a parameter called ``name`` carries a secret."""
    lowered = str(name).lower()
    return lowered in _SENSITIVE_NAMES or lowered.endswith(_SENSITIVE_SUFFIXES)


def redact_sensitive_parameters(value: Any, extra_keys: Iterable[str] = ()) -> Any:
    """Copy of ``value`` with the values of secret-named keys replaced.

    Works recursively through mappings, lists and tuples. ``extra_keys`` adds the names a
    configuration declares as sensitive (``OperationConfig.SENSITIVE_KEYS``). ``None``
    stays ``None`` (an absent secret is not a leak).
    """
    extra = set(extra_keys)
    if isinstance(value, dict):
        return {
            k: (
                REDACTED
                if (k in extra or is_sensitive_parameter(k)) and v is not None
                else redact_sensitive_parameters(v, extra)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_parameters(v, extra) for v in value]
    return value


def secret_identity(secret: Any) -> Optional[str]:
    """Non-secret stand-in for a secret that must still influence a cache key.

    A truncated BLAKE2b digest: stable for the same secret, different for another, and
    not the secret. ``None`` stays ``None``.
    """
    if secret is None:
        return None
    return hashlib.blake2b(
        str(secret).encode("utf-8"), digest_size=16, person=b"pamola-cache"
    ).hexdigest()
