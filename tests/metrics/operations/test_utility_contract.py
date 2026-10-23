"""UTIL-01, UTIL-02: one vocabulary for UtilityMetricOperation, and a working path.

The pieces used to disagree: the documented ``UtilityMetricsType`` enum named
individual measures (``r2``, ``auroc``...), the schema and the factory accepted
only the two families ``classification`` / ``regression``, the tooltip showed
``target``/``"lr"`` where the code needs ``value_field``/``"logistic"``, and a
request that got past all of that failed mid-run with a ``KeyError``. The
vocabulary is now: families in ``utility_metrics``; the target in
``metric_params[family]["value_field"]``; model and metric names as declared by
the implementation (``SUPPORTED_MODELS`` / ``SUPPORTED_METRICS``).
"""

import json

import numpy as np
import pandas as pd
import pytest

from pamola_core.common.enum.utility_metrics_type import UtilityMetricsType
from pamola_core.errors.exceptions import InvalidParameterError, ValidationError
from pamola_core.metrics.operations.utility_ops import (
    UTILITY_METRIC_FACTORY,
    UtilityMetricOperation,
)
from pamola_core.metrics.schemas.utility_op_core_schema import UtilityMetricConfig
from pamola_core.metrics.utility.classification import ClassificationUtility
from pamola_core.metrics.utility.regression import RegressionUtility
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

FAMILIES = {"classification": ClassificationUtility, "regression": RegressionUtility}


def _frames(n=240):
    rng = np.random.default_rng(0)
    original = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    original["amount"] = original["a"] * 2 + rng.normal(size=n)
    original["label"] = (original["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    transformed = original.copy()
    transformed["a"] += rng.normal(scale=0.1, size=n)
    return original, transformed


def _schema_families():
    items = UtilityMetricConfig.schema["allOf"][1]["properties"]["utility_metrics"]["items"]
    return {entry["const"] for entry in items["oneOf"]}


# --- UTIL-01 -------------------------------------------------------------------


def test_schema_and_factory_name_the_same_families():
    assert _schema_families() == set(FAMILIES)
    assert set(UTILITY_METRIC_FACTORY) == set(FAMILIES)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_schema_accepted_family_maps_to_an_implementation(family):
    op = UtilityMetricOperation(utility_metrics=[family])  # schema accepts it
    assert op.utility_metrics == [family]
    implementation = FAMILIES[family]
    assert callable(getattr(implementation, "calculate_metric"))


@pytest.mark.parametrize("measure", [m.value for m in UtilityMetricsType])
def test_individual_measure_names_are_not_family_names(measure):
    """The enum's names are measures reported *inside* a family, not families."""
    with pytest.raises(Exception):
        UtilityMetricOperation(utility_metrics=[measure])


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_declared_vocabulary_matches_the_implementation(family):
    implementation = FAMILIES[family]
    instance = implementation()
    assert set(implementation.SUPPORTED_MODELS) == set(instance.model_dict)
    # Defaults are drawn from the declared vocabulary.
    assert set(instance.models) <= set(implementation.SUPPORTED_MODELS)
    assert set(instance.metrics) <= set(implementation.SUPPORTED_METRICS)


_TARGET = {"classification": "label", "regression": "amount"}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_declared_metric_is_produced(family):
    original, transformed = _frames()
    drop = "amount" if family == "classification" else "label"
    original, transformed = original.drop(columns=drop), transformed.drop(columns=drop)
    produced = {}
    for metric in FAMILIES[family].SUPPORTED_METRICS:
        result = FAMILIES[family](cv_folds=3, models=[FAMILIES[family].SUPPORTED_MODELS[0]],
                                  metrics=[metric]).calculate_metric(
            original, transformed, value_field=_TARGET[family]
        )
        produced[metric] = result
        model_result = next(iter(result.values()))
        expected_key = "precision_recall_tradeoff" if metric == "precision_recall_tradeoff" else metric
        assert expected_key in model_result, f"{family}: {metric} produced {model_result.keys()}"


# --- UTIL-02: a working normal invocation ------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_documented_invocation_returns_success_with_documented_output(family, tmp_path):
    original, transformed = _frames()
    drop = "amount" if family == "classification" else "label"
    source = DataSource(
        dataframes={
            "main": original.drop(columns=drop),
            "synth": transformed.drop(columns=drop),
        }
    )
    op = UtilityMetricOperation(
        utility_metrics=[family],
        metric_params={family: {"value_field": _TARGET[family], "cv_folds": 3}},
    )

    result = op.execute(
        source,
        tmp_path,
        None,
        original_dataset_name="main",
        transformed_dataset_name="synth",
    )

    assert result.status == OperationStatus.SUCCESS, result.error_message
    artifact = next(a for a in result.artifacts if a.path.suffix == ".json")
    metrics = json.loads(artifact.path.read_text(encoding="utf-8"))["metrics"][family]
    assert set(metrics) == set(FAMILIES[family]().models)
    for model_metrics in metrics.values():
        for name in FAMILIES[family]().metrics:
            if name != "precision_recall_tradeoff":
                assert isinstance(model_metrics[name], float)


# --- explicit required configuration -------------------------------------------


def _calculate(family, params, original=None, transformed=None):
    o, t = _frames()
    op = UtilityMetricOperation(utility_metrics=[family])
    return op.calculate_metrics(
        original if original is not None else o,
        transformed if transformed is not None else t,
        utility_metrics=[family],
        metric_params={family: params},
    )


def test_missing_target_column_is_reported_with_the_fix():
    with pytest.raises(ValidationError, match="value_field"):
        _calculate("regression", {})


def test_unknown_target_column_names_the_dataset_and_columns():
    with pytest.raises(ValidationError, match="'nope' is not in the original"):
        _calculate("regression", {"value_field": "nope"})


@pytest.mark.parametrize(
    ("family", "key", "bad", "good"),
    [
        ("classification", "models", "lr", "logistic"),
        ("classification", "metrics", "auroc", "roc_auc"),
        ("regression", "models", "svm", "svr"),
        ("regression", "metrics", "precision", "r2"),
    ],
)
def test_unsupported_model_or_metric_is_rejected_before_training(family, key, bad, good):
    with pytest.raises(ValidationError) as excinfo:
        _calculate(family, {"value_field": _TARGET[family], key: [bad]})
    assert bad in str(excinfo.value)
    assert good in str(excinfo.value)  # the supported names are listed


def test_empty_metric_list_says_what_to_pass():
    op = UtilityMetricOperation()
    o, t = _frames()
    with pytest.raises(Exception, match="value_field"):
        op.calculate_metrics(o, t, utility_metrics=[], metric_params={})


def test_validator_is_typed_as_invalid_parameter():
    with pytest.raises(InvalidParameterError):
        UtilityMetricOperation._validate_metric_request(
            "regression", RegressionUtility, {}, *_frames()
        )
