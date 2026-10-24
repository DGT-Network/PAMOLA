"""Importing the package must not take over the host application's logging.

The four ``profiling/commons`` modules used to call ``configure_logging()`` at import
time. With its default name that reconfigured the package-root logger ``pamola_core``:
a stdout handler was added and ``propagate`` was set to ``False``, so every
``pamola_core.*`` record stopped there and never reached the application's handlers
(or ``caplog``). Run in a fresh interpreter so the test sees import-time effects only.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

_PROBE = textwrap.dedent(
    """
    import logging
    import pamola_core
    import pamola_core.profiling.commons.currency_analysis
    import pamola_core.profiling.commons.data_processing
    import pamola_core.profiling.commons.report_generation
    import pamola_core.profiling.commons.statistical_analysis
    import pamola_core.profiling.analyzers.numeric
    import pamola_core.analysis.dataset_summary

    bad = []
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if not (name == "pamola_core" or name.startswith("pamola_core.")):
            continue
        if not isinstance(lg, logging.Logger):
            continue
        if lg.propagate is False:
            bad.append(name + ": propagate is False")
        if lg.handlers and not all(isinstance(h, logging.NullHandler) for h in lg.handlers):
            bad.append(name + ": has handlers " + repr(lg.handlers))
    if logging.root.manager.disable:
        bad.append("logging.disable is set")
    print("BAD:" + "|".join(bad))
    """
)


def test_importing_the_package_leaves_package_loggers_to_the_application():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=Path(__file__).resolve().parents[2],  # the suite runs inside a tmp working dir
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("BAD:")][-1]
    assert line == "BAD:", line


# --- the autouse fixture in tests/conftest.py (restore_logging_state) -------------------
# Two tests in definition order: the first leaks, the second must not see it.


def test_fixture_part_1_a_test_may_leak_logging_state():
    import logging

    logging.getLogger("pamola_core").propagate = False
    logging.getLogger("pamola_core.utils.ops").setLevel(logging.CRITICAL)
    logging.getLogger("pamola_core").addHandler(logging.StreamHandler())
    logging.disable(logging.CRITICAL)


def test_fixture_part_2_the_next_test_sees_the_original_state(caplog):
    import logging

    root = logging.getLogger("pamola_core")
    assert root.propagate is True
    assert root.handlers == []
    assert logging.getLogger("pamola_core.utils.ops").level == logging.NOTSET
    assert logging.root.manager.disable == 0
    with caplog.at_level(logging.WARNING):
        logging.getLogger("pamola_core.utils.ops.anything").warning("visible")
    assert "visible" in caplog.text


_EXERCISE = textwrap.dedent(
    """
    import logging, os, tempfile
    from pamola_core.privacy_models.l_diversity.calculation import LDiversityCalculator
    from pamola_core.utils.progress import SimpleProgressBar, get_progress_logger
    os.chdir(tempfile.mkdtemp())
    before = list(logging.root.handlers)
    pamola_before = {n for n in logging.root.manager.loggerDict if n.startswith("pamola_core")}

    # l-diversity used to call logging.basicConfig() in its constructor
    LDiversityCalculator(l=2, log_level="DEBUG")

    # the progress tracker used to configure a stdout handler and a log file on first use
    with SimpleProgressBar(total=3, description="probe") as bar:
        for _ in range(3):
            bar.update(1)
    get_progress_logger().info("probe")

    bad = []
    if list(logging.root.handlers) != before:
        bad.append("root handlers changed: " + repr(logging.root.handlers))
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if not (name == "pamola_core" or name.startswith("pamola_core.")):
            continue
        if not isinstance(lg, logging.Logger):
            continue
        if lg.propagate is False:
            bad.append(name + ": propagate is False")
        if lg.handlers and not all(isinstance(h, logging.NullHandler) for h in lg.handlers):
            bad.append(name + ": has handlers " + repr(lg.handlers))
    if os.listdir("."):
        bad.append("files written to the working directory: " + repr(os.listdir(".")))
    print("BAD:" + "|".join(bad))
    """
)


def test_using_l_diversity_and_progress_adds_no_handlers_and_writes_no_files():
    out = subprocess.run(
        [sys.executable, "-c", _EXERCISE],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("BAD:")][-1]
    assert line == "BAD:", line
