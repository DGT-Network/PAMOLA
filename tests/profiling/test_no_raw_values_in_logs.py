"""
CDX-R2-12: profiling operations must not write source values to logs.

DEBUG logging is the level that used to dump complete source rows
(`df.head(5).to_dict(...)` in the identity and MVF profilers) and raw values in
parse warnings. Structure and counts are fine; values are not.
"""

import logging

import pandas as pd
import pytest

from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
from pamola_core.profiling.analyzers.mvf import MVFOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

CANARY = "ZQCANARY"


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _df():
    return pd.DataFrame(
        {
            "uid": [f"{CANARY}-u1", f"{CANARY}-u1", f"{CANARY}-u2", f"{CANARY}-u3"],
            "rid": ["r1", "r2", "r3", "r4"],
            "first": [f"{CANARY}Alice", f"{CANARY}Alice", f"{CANARY}Bob", f"{CANARY}Cy"],
            "last": [f"{CANARY}Doe", f"{CANARY}Doe", f"{CANARY}Roe", f"{CANARY}Poe"],
            "tags": [
                f'["{CANARY}a", "{CANARY}b"]',
                f'["{CANARY}a"]',
                "not json {{ " + CANARY,
                f'["{CANARY}c"]',
            ],
            "cat": [f"{CANARY}x", f"{CANARY}y", f"{CANARY}x", f"{CANARY}z"],
        }
    )


OPERATIONS = {
    "identity": lambda: IdentityAnalysisOperation(
        uid_field="uid", reference_fields=["first", "last"], id_field="rid"
    ),
    "mvf": lambda: MVFOperation(field_name="tags", format_type="json"),
    "categorical": lambda: CategoricalOperation(field_name="cat"),
}


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_profiling_debug_logs_contain_no_source_values(name, tmp_path, caplog):
    op = OPERATIONS[name]()
    with caplog.at_level(logging.DEBUG):
        result = op.execute(
            DataSource(dataframes={"main": _df()}), tmp_path / "t", _Reporter()
        )

    assert result.status == OperationStatus.SUCCESS, result.error_message
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert CANARY not in logged, [
        line for line in logged.splitlines() if CANARY in line
    ][:3]


def test_malformed_mvf_values_are_not_echoed_in_warnings(caplog):
    from pamola_core.profiling.commons.mvf_utils import parse_mvf

    with caplog.at_level(logging.DEBUG):
        parse_mvf(f"[broken {CANARY}", format_type="json")
        parse_mvf(f"[broken {CANARY}", format_type="array_string")
    assert CANARY not in "\n".join(r.getMessage() for r in caplog.records)
