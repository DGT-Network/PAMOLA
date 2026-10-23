"""CDX-R2-8: an encrypted request is never answered with plaintext part files.

Before: encrypted Dask output in parquet (always partitioned) or CSV with more than 10
partitions logged "not fully implemented" and wrote unencrypted parts.
"""

import dask.dataframe as dd
import pandas as pd
import pytest

from pamola_core.errors.exceptions import DataWriteError
from pamola_core.utils.ops.op_data_writer import DataWriter

KEY = "k" * 64


def _frame(partitions):
    return dd.from_pandas(pd.DataFrame({"x": range(60), "name": ["secret-value"] * 60}), npartitions=partitions)


def _written_files(task_dir):
    return [p for p in task_dir.rglob("*") if p.is_file()]


@pytest.mark.parametrize(("fmt", "partitions"), [("parquet", 3), ("csv", 11)])
def test_encrypted_partitioned_output_is_refused_before_writing(fmt, partitions, tmp_path):
    writer = DataWriter(task_dir=tmp_path)
    with pytest.raises(DataWriteError, match="not supported"):
        writer.write_dataframe(_frame(partitions), "out", format=fmt, encryption_key=KEY)
    assert _written_files(tmp_path) == []
    assert not any(p.is_dir() and p.name == "out" for p in tmp_path.rglob("*"))


def test_unencrypted_partitioned_output_still_works(tmp_path):
    result = DataWriter(task_dir=tmp_path).write_dataframe(_frame(3), "out", format="parquet")
    assert result.path.exists()


def test_encrypted_small_csv_is_materialized_and_encrypted(tmp_path):
    result = DataWriter(task_dir=tmp_path).write_dataframe(_frame(2), "out", format="csv", encryption_key=KEY)
    assert b"secret-value" not in result.path.read_bytes()
