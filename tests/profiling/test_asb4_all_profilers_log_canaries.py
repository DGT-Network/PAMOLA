"""ASB-4: the log-canary check covers ALL 14 profiling operations (CDX-R2-12 covered 3).

Every column carries source values that contain a unique marker; at DEBUG level no log
record of any profiler may contain it. Operations that cannot be driven by the shared frame
(a profile needs a recognisable value type) get a column of the right type whose digits carry
the numeric marker.
"""

import logging

import pandas as pd
import pytest

from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation
from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.profiling.analyzers.correlation import CorrelationMatrixOperation, CorrelationOperation
from pamola_core.profiling.analyzers.currency import CurrencyOperation
from pamola_core.profiling.analyzers.date import DateOperation
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.profiling.analyzers.group import GroupAnalyzerOperation
from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
from pamola_core.profiling.analyzers.mvf import MVFOperation
from pamola_core.profiling.analyzers.numeric import NumericOperation
from pamola_core.profiling.analyzers.phone import PhoneOperation
from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

CANARY = "ZQCANARY"
NUM_MARK = "913377"  # digits of the marker in numeric / phone / currency / date values


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _df():
    n = 12
    return pd.DataFrame({
        "uid": [f"{CANARY}-u{i % 4}" for i in range(n)],
        "rid": [f"r{i}" for i in range(n)],
        "first": [f"{CANARY}Alice", f"{CANARY}Bob", f"{CANARY}Cy"] * 4,
        "last": [f"{CANARY}Doe", f"{CANARY}Roe", f"{CANARY}Poe"] * 4,
        "tags": [f'["{CANARY}a", "{CANARY}b"]', f'["{CANARY}a"]', "not json {{ " + CANARY] * 4,
        "cat": [f"{CANARY}x", f"{CANARY}y", f"{CANARY}x", f"{CANARY}z"] * 3,
        "num": [float(f"{NUM_MARK}{i}.5") for i in range(n)],
        "num2": [float(f"{NUM_MARK}{i % 5}.25") * 3 for i in range(n)],
        "email": [f"{CANARY}.user{i}@{CANARY}mail.example" for i in range(n)],
        "phone": [f"+1 555 {NUM_MARK[:3]} {NUM_MARK[3:]}{i % 10}" for i in range(n)],
        "money": [f"${NUM_MARK}{i}.99" for i in range(n)],
        "when": [f"2020-{(i % 12) + 1:02d}-1{i % 9}" for i in range(n)],
        "text": [f"{CANARY} the patient lives in {CANARY}town and works at {CANARY}Corp {i}" for i in range(n)],
        "grp": [f"g{i % 3}" for i in range(n)],
        "qi": [f"{CANARY}q{i % 3}" for i in range(n)],
    })


OPERATIONS = {
    "categorical": lambda: CategoricalOperation(field_name="cat"),
    "numeric": lambda: NumericOperation("num", use_cache=False),
    "date": lambda: DateOperation(field_name="when"),
    "email": lambda: EmailOperation(field_name="email"),
    "phone": lambda: PhoneOperation(field_name="phone"),
    "currency": lambda: CurrencyOperation(field_name="money"),
    "mvf": lambda: MVFOperation(field_name="tags", format_type="json"),
    "text": lambda: TextSemanticCategorizerOperation(field_name="text"),
    "correlation": lambda: CorrelationOperation(field1="num", field2="num2"),
    "correlation_matrix": lambda: CorrelationMatrixOperation(fields=["num", "num2"]),
    "identity": lambda: IdentityAnalysisOperation(uid_field="uid", reference_fields=["first", "last"], id_field="rid"),
    "group": lambda: GroupAnalyzerOperation(field_name="grp", fields_config={"first": 1, "last": 1}),
    "attribute": lambda: DataAttributeProfilerOperation(),
    "k_anonymity": lambda: KAnonymityProfilerOperation(quasi_identifiers=["qi"], id_fields=["rid"]),
}


def test_asb4_the_matrix_covers_every_profiling_operation():
    assert len(OPERATIONS) == 14


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_asb4_no_profiler_logs_a_source_value(name, tmp_path, caplog):
    op = OPERATIONS[name]()
    with caplog.at_level(logging.DEBUG):
        result = op.execute(DataSource(dataframes={"main": _df()}), tmp_path / "t", _Reporter())
    assert result.status == OperationStatus.SUCCESS, result.error_message
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert CANARY not in logged, [line for line in logged.splitlines() if CANARY in line][:3]
    assert NUM_MARK not in logged, [line for line in logged.splitlines() if NUM_MARK in line][:3]
