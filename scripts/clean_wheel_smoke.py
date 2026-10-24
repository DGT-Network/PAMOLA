"""Clean-wheel smoke test: does the INSTALLED pamola-core wheel actually work?

Run it with the interpreter of a fresh virtualenv that holds only the built wheel
(no ``-e``, no extras, no checkout on ``sys.path``)::

    python scripts/clean_wheel_smoke.py --expect-version 1.0.0

Nine cases: import, masking, generalization, pseudonymization, suppression,
profiling, metrics (fidelity and privacy), CSV I/O, Parquet I/O. Every case runs
on the base install, so none may need the ``viz`` extra: operations are used with
their defaults (visualization off). Exit code 0 only if all nine pass.
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path


class NetworkBlocked(OSError):
    """Raised by the smoke test's network guard."""


def block_network():
    """Make any attempt to reach the network fail loudly (the base install needs none).

    Patches name resolution and ``connect`` of internet sockets in this process (UNIX-domain
    sockets stay possible). Everything the smoke cases do runs under it, on every OS.
    """
    import socket

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(*_a, **_k):
        raise NetworkBlocked("network access attempted during the no-network smoke test")

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            refuse()
        return real_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            refuse()
        return real_connect_ex(self, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = refuse
    socket.gethostbyname = refuse
    socket.create_connection = refuse


def _frame():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave"] * (n // 4),
            "email": [f"user{i}@example.com" for i in range(n)],
            "age": rng.integers(18, 90, n).astype(float),
            "salary": rng.normal(50_000, 8_000, n),
        }
    )


def _run(op, df, tmp, **kwargs):
    from pamola_core.utils.ops.op_data_source import DataSource
    from pamola_core.utils.ops.op_result import OperationStatus

    result = op.execute(DataSource(dataframes={"main": df}), Path(tmp), None, **kwargs)
    assert result.status == OperationStatus.SUCCESS, (
        f"{type(op).__name__} returned {result.status}: {result.error_message}"
    )
    return result


def case_import(expect_version, **_):
    import pamola_core

    location = Path(pamola_core.__file__).resolve()
    assert "site-packages" in location.parts, (
        f"pamola_core imported from {location}, not from an installed wheel"
    )
    if expect_version:
        assert pamola_core.__version__ == expect_version, (
            f"installed version {pamola_core.__version__} != expected {expect_version}"
        )
    for backend in ("matplotlib", "plotly", "seaborn"):
        assert backend not in sys.modules, f"import pamola_core pulled in {backend}"


def case_masking(tmp, **_):
    from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation

    _run(FullMaskingOperation(field_name="name"), _frame(), tmp)


def case_generalization(tmp, **_):
    from pamola_core.anonymization.generalization.numeric_op import (
        NumericGeneralizationOperation,
    )

    _run(
        NumericGeneralizationOperation(field_name="age", strategy="binning", bin_count=5),
        _frame(),
        tmp,
    )


def case_pseudonymization(tmp, **_):
    from pamola_core.anonymization.pseudonymization.hash_based_op import (
        HashBasedPseudonymizationOperation,
    )

    _run(HashBasedPseudonymizationOperation(field_name="email"), _frame(), tmp)


def case_suppression(tmp, **_):
    from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation

    _run(CellSuppressionOperation(field_name="name", suppression_strategy="null"), _frame(), tmp)


def case_profiling(tmp, **_):
    from pamola_core.profiling.analyzers.numeric import NumericOperation

    _run(NumericOperation("salary", use_cache=False), _frame(), tmp)


def case_metrics(tmp, **_):
    from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
    from pamola_core.metrics.operations.privacy_ops import PrivacyMetricOperation
    from pamola_core.utils.ops.op_data_source import DataSource
    from pamola_core.utils.ops.op_result import OperationStatus

    original = _frame()[["age", "salary"]]
    transformed = original.copy()
    transformed["salary"] = transformed["salary"] * 1.05
    source = DataSource(dataframes={"main": original, "synth": transformed})
    names = {"original_dataset_name": "main", "transformed_dataset_name": "synth"}

    for label, op in (
        ("fidelity", FidelityOperation(fidelity_metrics=["ks"], columns=["age", "salary"])),
        ("privacy", PrivacyMetricOperation(privacy_metrics=["dcr"], columns=["age", "salary"])),
    ):
        out = Path(tmp) / label
        out.mkdir()
        result = op.execute(source, out, None, **names)
        assert result.status != OperationStatus.ERROR, f"{label}: {result.error_message}"
        assert any(a.path.exists() for a in result.artifacts), f"{label}: no artifact written"


def case_csv_io(tmp, **_):
    import pandas as pd

    from pamola_core.utils.io import read_full_csv, write_dataframe_to_csv

    df = _frame()
    path = Path(tmp) / "roundtrip.csv"
    write_dataframe_to_csv(df, path)
    back = read_full_csv(path)
    assert isinstance(back, pd.DataFrame) and len(back) == len(df)
    assert list(back.columns) == list(df.columns)


def case_parquet_io(tmp, **_):
    import pandas as pd

    from pamola_core.utils.io import read_parquet, write_parquet

    df = _frame()
    path = Path(tmp) / "roundtrip.parquet"
    write_parquet(df, path)
    back = read_parquet(path)
    assert isinstance(back, pd.DataFrame) and len(back) == len(df)
    assert list(back.columns) == list(df.columns)


CASES = [
    ("import", case_import),
    ("masking", case_masking),
    ("generalization", case_generalization),
    ("pseudonymization", case_pseudonymization),
    ("suppression", case_suppression),
    ("profiling", case_profiling),
    ("metrics (fidelity + privacy)", case_metrics),
    ("csv i/o", case_csv_io),
    ("parquet i/o", case_parquet_io),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect-version", default=None)
    parser.add_argument("--no-network", action="store_true",
                        help="refuse every network access while the cases run")
    args = parser.parse_args()
    if args.no_network:
        block_network()
        print("network guard: ON")

    # A checkout on sys.path would make the test pass against the source tree.
    cwd = Path.cwd().resolve()
    if (cwd / "pamola_core").is_dir():
        print(f"refusing to run inside a checkout ({cwd}); cd elsewhere", file=sys.stderr)
        return 2

    failures = 0
    for label, case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                case(tmp=tmp, expect_version=args.expect_version)
            except Exception:
                failures += 1
                print(f"FAIL  {label}")
                traceback.print_exc()
            else:
                print(f"PASS  {label}")
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
