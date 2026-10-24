"""
P1 canary test: with default settings every artifact carries a class, source values appear
only in DATA / PROTECTED-MAPPING-free / DIAGNOSTIC artifacts, never in an AGGREGATE_METRIC
or EVIDENCE artifact, and no mapping file is in plain text anywhere.

Rules checked for each operation (default policy, a unique canary in the data):

* every registered artifact has an ``ArtifactClass`` and ``to_dict()`` reports it;
* a file that contains the canary is registered and is DATA or DIAGNOSTIC; a DIAGNOSTIC one
  says ``contains_source_values`` and ``not_for_release``;
* an AGGREGATE_METRIC / EVIDENCE / PROTECTED_MAPPING file never contains the canary;
* every file under ``maps/`` or ``mappings/`` is free of the canary (encrypted or absent).
"""

import pandas as pd
import pytest

from pamola_core.anonymization.generalization.categorical_op import (
    CategoricalGeneralizationOperation,
)
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
from pamola_core.anonymization.pseudonymization.hash_based_op import (
    HashBasedPseudonymizationOperation,
)
from pamola_core.anonymization.pseudonymization.mapping_op import (
    ConsistentMappingPseudonymizationOperation,
)
from pamola_core.fake_data.operations.email_op import FakeEmailOperation
from pamola_core.profiling.analyzers.categorical import CategoricalOperation
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.utils.artifact_policy import ArtifactClass
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

CANARY = "zqcanary"
KEY = "6b2dcb5b58d1ff0f6b2dcb5b58cf0fd16b2dcb5b58d1ff0f6b2dcb5b58cf0fd1"
EMAILS = [f"{CANARY}.{n}@pii-canary.test" for n in ("alice", "bob", "carol")]
CATEGORIES = [f"{CANARY}cat{i}" for i in range(6)]


class _Reporter:
    def __init__(self):
        self.artifacts = []

    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _email_frame():
    return pd.DataFrame({"id": range(6), "email": EMAILS * 2})


def _category_frame():
    return pd.DataFrame(
        {"category": [c for c in CATEGORIES for _ in range(5)] + [f"{CANARY}rare"]}
    )


CASES = {
    "full_masking": (lambda: FullMaskingOperation(field_name="email"), _email_frame),
    "partial_masking": (
        lambda: PartialMaskingOperation(
            field_name="email", mask_strategy="fixed", unmasked_prefix=3, unmasked_suffix=0
        ),
        _email_frame,
    ),
    "hash_pseudonymization": (
        lambda: HashBasedPseudonymizationOperation(
            field_name="email",
            salt_config={"source": "parameter", "value": KEY},
            use_pepper=False,
        ),
        _email_frame,
    ),
    "mapping_pseudonymization": (
        lambda: ConsistentMappingPseudonymizationOperation(
            field_name="email", mapping_encryption_key=KEY, pseudonym_type="uuid"
        ),
        _email_frame,
    ),
    "fake_email_default": (
        lambda: FakeEmailOperation(field_name="email", consistency_mechanism="mapping", key="k"),
        _email_frame,
    ),
    "fake_email_save_mapping": (
        lambda: FakeEmailOperation(
            field_name="email",
            consistency_mechanism="mapping",
            key="k",
            save_mapping=True,
            use_encryption=True,
            encryption_key="unit-test-task-key",
        ),
        _email_frame,
    ),
    "category_generalization": (
        lambda: CategoricalGeneralizationOperation(
            field_name="category",
            strategy="merge_low_freq",
            min_group_size=6,
            generate_visualization=False,
            use_cache=False,
        ),
        _category_frame,
    ),
    "category_generalization_encrypted": (
        lambda: CategoricalGeneralizationOperation(
            field_name="category",
            strategy="merge_low_freq",
            min_group_size=6,
            generate_visualization=False,
            use_cache=False,
            use_encryption=True,
            encryption_key="unit-test-task-key",
        ),
        _category_frame,
    ),
    "profiling_email": (lambda: EmailOperation(field_name="email"), _email_frame),
    "profiling_categorical": (
        lambda: CategoricalOperation(field_name="category"),
        _category_frame,
    ),
}


def _holds_canary(path):
    return CANARY.encode() in path.read_bytes()


@pytest.mark.parametrize("name", list(CASES))
def test_default_settings_classify_every_artifact_and_keep_source_values_out_of_metrics_evidence_and_mappings(
    name, tmp_path
):
    build, frame = CASES[name]
    task_dir = tmp_path / "task"
    result = build().execute(DataSource(dataframes={"main": frame()}), task_dir, _Reporter())
    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert result.artifacts, "the operation registered no artifact"

    registered = {}
    for artifact in result.artifacts:
        assert isinstance(artifact.artifact_class, ArtifactClass)
        assert artifact.to_dict()["artifact_class"] == artifact.artifact_class.value
        registered[artifact.path.resolve()] = artifact
        if not artifact.path.is_file():
            continue
        has_canary = _holds_canary(artifact.path)
        if artifact.artifact_class in (
            ArtifactClass.AGGREGATE_METRIC,
            ArtifactClass.EVIDENCE,
            ArtifactClass.PROTECTED_MAPPING,
        ):
            assert not has_canary, (
                f"{artifact.artifact_class.value} artifact {artifact.path.name} "
                "contains source values"
            )
        if has_canary and artifact.artifact_class is ArtifactClass.DIAGNOSTIC:
            d = artifact.to_dict()
            assert d["contains_source_values"] is True and d["not_for_release"] is True

    output_dir = task_dir / "output"
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file() or output_dir in path.parents:
            continue
        if path.parent.name in ("maps", "mappings"):
            assert not _holds_canary(path), f"plain-text mapping file: {path.name}"
        if _holds_canary(path):
            art = registered.get(path.resolve())
            assert art is not None, f"unclassified file with source values: {path.name}"
            assert art.artifact_class in (ArtifactClass.DATA, ArtifactClass.DIAGNOSTIC), (
                path.name,
                art.artifact_class,
            )


def test_profiling_value_artifacts_are_diagnostic_and_not_for_release(tmp_path):
    result = EmailOperation(field_name="email").execute(
        DataSource(dataframes={"main": _email_frame()}), tmp_path / "t", _Reporter()
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    classes = {a.artifact_class for a in result.artifacts}
    assert classes == {ArtifactClass.DIAGNOSTIC}
    with_values = [a for a in result.artifacts if a.contains_source_values]
    assert with_values and all(a.to_dict()["not_for_release"] for a in with_values)
