"""
Opt-in gate for loading formats that can execute code (pickle).

``pickle.load`` and ``pandas.read_pickle`` run arbitrary code embedded in the file.
A file that did not come from a source you fully control must never be loaded that
way. PAMOLA therefore never loads a pickle implicitly: every call that would do so
must pass ``allow_unsafe_pickle=True``, and every such load logs a warning.

Supported formats for files of unknown origin: CSV, JSON, Parquet, Excel, and the
encrypted PAMOLA mapping format (authenticated AES-GCM frames, no code execution).
See ``docs/en/core/security_loading_files.md``.
"""

import logging
import warnings
from pathlib import Path
from typing import Union

from pamola_core.errors.exceptions import ValidationError

logger = logging.getLogger(__name__)


class UnsafeDeserializationError(ValidationError):
    """Raised when a format that can execute code is loaded without an explicit opt-in."""


def require_unsafe_pickle_opt_in(allow_unsafe_pickle: bool, path: Union[str, Path]) -> None:
    """Refuse to unpickle ``path`` unless the caller opted in explicitly.

    Raises
    ------
    UnsafeDeserializationError
        When ``allow_unsafe_pickle`` is not ``True``.
    """
    if allow_unsafe_pickle is not True:
        raise UnsafeDeserializationError(
            f"Refusing to load '{path}' as pickle: unpickling can execute arbitrary code. "
            "Use JSON, CSV, Parquet or the encrypted mapping format instead. If the file "
            "was produced by you or a source you fully trust, pass allow_unsafe_pickle=True."
        )
    message = (
        f"Loading '{path}' as pickle (allow_unsafe_pickle=True): this executes code from "
        "the file. Only do so for files from a fully trusted source."
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)
