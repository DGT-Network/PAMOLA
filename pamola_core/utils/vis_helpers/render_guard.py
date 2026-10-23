"""
Bounded, cancellable rendering for the operations' visualization step.

Rendering runs in a worker thread (matplotlib / plotly state is not thread-safe, and the step
must not block the operation forever). A thread cannot be killed, so the guarantee is built
from three parts:

* the wait is **finite** (``wait_for_render``): after ``timeout`` seconds the operation records
  an explicit failure and returns;
* every write of a figure goes through ``guarded_write`` (called by ``save_visualization``).
  After the timeout the run is *cancelled*: a write that has not started is refused, and a
  write already in flight is waited for, so no file appears after the operation returned;
* files the abandoned worker managed to write before the cancellation are removed, and the
  caller registers artifacts only for runs that completed.

The worker is a daemon thread, so an abandoned renderer cannot keep the process alive.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Used when an operation passes no (or a non-positive) timeout: waiting is never unbounded.
DEFAULT_VISUALIZATION_TIMEOUT = 120.0

#: How long ``cancel`` waits for a write that is already in flight.
_CANCEL_GRACE_SECONDS = 30.0


class VisualizationTimeoutError(TimeoutError):
    """The rendering worker did not finish within the visualization timeout."""


class VisualizationCancelledError(RuntimeError):
    """Raised inside an abandoned worker that tries to write after the run was cancelled."""


class RenderRun:
    """State shared by one rendering worker and the operation waiting for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._written: List[Path] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def written(self) -> Tuple[Path, ...]:
        return tuple(self._written)

    def cancel(self) -> List[Path]:
        """Refuse further writes, wait out one in flight, remove what was written."""
        acquired = self._lock.acquire(timeout=_CANCEL_GRACE_SECONDS)
        self._cancelled = True
        removed: List[Path] = []
        try:
            for path in self._written:
                try:
                    path.unlink()
                    removed.append(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("Could not remove abandoned visualization %s: %s", path.name, exc)
            self._written = []
        finally:
            if acquired:
                self._lock.release()
        return removed


_current_run: contextvars.ContextVar[Optional[RenderRun]] = contextvars.ContextVar(
    "pamola_render_run", default=None
)


def prepare_render_context() -> Tuple[contextvars.Context, RenderRun]:
    """Copy the current context and bind a fresh ``RenderRun`` inside the copy."""
    run = RenderRun()
    ctx = contextvars.copy_context()
    ctx.run(_current_run.set, run)
    return ctx, run


@contextlib.contextmanager
def guarded_write(path: Any) -> Iterator[None]:
    """Wrap the actual write of one visualization file.

    Outside a rendering worker this is a no-op. Inside one, a cancelled run refuses the write
    and a successful write is remembered so a later cancellation can remove it.
    """
    run = _current_run.get()
    if run is None:
        yield
        return
    with run._lock:  # noqa: SLF001 - the guard is the run's own collaborator
        if run.cancelled:
            raise VisualizationCancelledError(
                "visualization run was cancelled after its timeout; nothing was written"
            )
        yield
        run._written.append(Path(path))  # noqa: SLF001


def wait_for_render(
    thread: threading.Thread,
    run: RenderRun,
    timeout: Optional[float],
    log: Optional[logging.Logger] = None,
    interval: float = 5.0,
) -> bool:
    """Wait for the rendering thread for at most ``timeout`` seconds.

    Returns ``True`` if it finished. On timeout the run is cancelled and cleaned up and
    ``False`` is returned; the caller records ``VisualizationTimeoutError``.
    """
    log = log or logger
    limit = float(timeout) if timeout and float(timeout) > 0 else DEFAULT_VISUALIZATION_TIMEOUT
    waited = 0.0
    while thread.is_alive() and waited < limit:
        step = min(interval, limit - waited)
        thread.join(timeout=step)
        waited += step
        if thread.is_alive() and waited < limit:
            log.info("[DIAG] Visualization thread still running after %.1fs...", waited)
    if not thread.is_alive():
        return True
    removed = run.cancel()
    log.error(
        "[DIAG] Visualization did not finish within %.1fs; run cancelled, %d partial file(s) removed",
        limit,
        len(removed),
    )
    return False
