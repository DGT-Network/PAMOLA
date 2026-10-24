"""The category-generalization mapping lists original categories: a PROTECTED_MAPPING, written
encrypted with the task key only; without a key no mapping file is written at all."""

import pandas as pd

from pamola_core.anonymization.generalization.categorical_op import (
    CategoricalGeneralizationOperation,
)
from pamola_core.utils.artifact_policy import ArtifactClass
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

CANARY = "ZQCATEGORY"


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _run(tmp_path, **kwargs):
    cats = [f"{CANARY}{i}" for i in range(6)]
    df = pd.DataFrame({"category": [c for c in cats for _ in range(5)] + [f"{CANARY}rare"]})
    op = CategoricalGeneralizationOperation(
        field_name="category",
        strategy="merge_low_freq",
        min_group_size=6,
        generate_visualization=False,
        use_cache=False,
        **kwargs,
    )
    out = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert out.status == OperationStatus.SUCCESS, out.error_message
    return out, tmp_path / "t"


def test_without_a_key_no_mapping_file_is_written(tmp_path):
    out, task_dir = _run(tmp_path)
    assert not list((task_dir / "mappings").glob("*")) if (task_dir / "mappings").exists() else True
    assert not [a for a in out.artifacts if a.category == "mapping"]


def test_with_the_task_key_the_mapping_is_encrypted_and_classified(tmp_path):
    out, task_dir = _run(tmp_path, use_encryption=True, encryption_key="unit-test-task-key")
    files = list((task_dir / "mappings").glob("*"))
    assert files, "the encrypted mapping was not written"
    for path in files:
        assert CANARY.encode() not in path.read_bytes()
    mapping = [a for a in out.artifacts if a.category == "mapping"]
    assert len(mapping) == 1
    assert mapping[0].artifact_class is ArtifactClass.PROTECTED_MAPPING
    assert mapping[0].contains_source_values is True
