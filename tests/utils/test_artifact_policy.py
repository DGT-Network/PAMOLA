"""Artifact classes: derivation, content rule, carriage in to_dict()."""

import pytest

from pamola_core.utils.artifact_policy import (
    ArtifactClass,
    ArtifactPolicyError,
    check_class_and_content,
    class_for_category,
    require_encryption_key,
)
from pamola_core.utils.ops.op_result import OperationResult


@pytest.mark.parametrize(
    "category,expected",
    [
        ("output", ArtifactClass.DATA),
        ("metrics", ArtifactClass.AGGREGATE_METRIC),
        ("dictionary", ArtifactClass.DIAGNOSTIC),
        ("visualization", ArtifactClass.DIAGNOSTIC),
        ("mapping", ArtifactClass.PROTECTED_MAPPING),
        ("NER_SKILL", ArtifactClass.DIAGNOSTIC),  # unknown -> conservative
    ],
)
def test_class_is_derived_from_the_legacy_category(category, expected):
    assert class_for_category(category) is expected


def test_every_artifact_carries_its_class_and_to_dict_reports_it(tmp_path):
    result = OperationResult()
    a = result.add_artifact("csv", tmp_path / "o.csv", category="output")
    b = result.add_artifact(
        "json",
        tmp_path / "m.json",
        category="mapping",
        artifact_class=ArtifactClass.PROTECTED_MAPPING,
        contains_source_values=True,
    )
    assert a.artifact_class is ArtifactClass.DATA
    assert a.to_dict()["artifact_class"] == "data"
    assert b.to_dict()["artifact_class"] == "protected_mapping"
    assert b.to_dict()["contains_source_values"] is True
    assert b.to_dict()["not_for_release"] is False  # an encrypted mapping is internal


def test_a_dictionary_artifact_is_diagnostic_with_source_values_and_not_for_release(tmp_path):
    result = OperationResult()
    d = result.add_artifact("csv", tmp_path / "d.csv", category="dictionary").to_dict()
    assert d["artifact_class"] == "diagnostic"
    assert d["contains_source_values"] is True
    assert d["not_for_release"] is True
    assert result.add_artifact("csv", tmp_path / "o.csv").to_dict()["not_for_release"] is False


@pytest.mark.parametrize("cls", [ArtifactClass.AGGREGATE_METRIC, ArtifactClass.EVIDENCE])
def test_source_values_cannot_be_a_metric_or_evidence(cls, tmp_path):
    with pytest.raises(ArtifactPolicyError):
        check_class_and_content(cls, True)
    result = OperationResult()
    with pytest.raises(ArtifactPolicyError):
        result.add_artifact(
            "json", tmp_path / "m.json", artifact_class=cls, contains_source_values=True
        )
    for ok in (ArtifactClass.DATA, ArtifactClass.PROTECTED_MAPPING, ArtifactClass.DIAGNOSTIC):
        assert check_class_and_content(ok, True) is ok


def test_protected_mapping_needs_a_key():
    with pytest.raises(ArtifactPolicyError, match="encrypted"):
        require_encryption_key(None, "the fake_data mapping")
    assert require_encryption_key("k", "x") == "k"
