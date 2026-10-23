"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL Linkage Operation
Package:       pamola_core.metrics.operations
License:       BSD 3-Clause

``CVPLLinkageOperation`` measures how far a protected release can be linked back to
its source with the CVPL operator chain ``B_lambda -> phi -> PCA -> cosine -> tau``.

It is a privacy *measurement* in the metrics architecture: two datasets in, one
metrics JSON out. It does not receive row-aligned pairs - the source row count is the
denominator and a suppressed row has no counterpart - so it opts out of the metrics
base class' index intersection (``requires_aligned_inputs = False``) instead of
bypassing it silently, and it never optimises dtypes of the inputs.

Artifacts and logs carry numbers and column names only: no values and no subject
identifiers (the ground-truth columns are read in memory and dropped).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from pamola_core.errors.exceptions import InvalidParameterError, ValidationError
from pamola_core.metrics.base_metrics_op import MetricsOperation
from pamola_core.metrics.privacy.cvpl import CVPLConfig, CVPLProjection, run_cvpl
from pamola_core.metrics.schemas.cvpl_op_core_schema import CVPLLinkageConfig
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.progress import HierarchicalProgressTracker

@register(version="1.0.0")
class CVPLLinkageOperation(MetricsOperation):
    """Linkage exposure of ``transformed`` (the release) against ``original`` (the source).

    Parameters
    ----------
    qids, auxiliary, blocking_ladder, tau, fit_strategy, ground_truth, ... :
        See :class:`pamola_core.metrics.privacy.cvpl.CVPLConfig`. Only the columns
        named in ``qids`` and ``auxiliary`` are ever used as the attacker's knowledge.
    frozen_projection : CVPLProjection, optional
        Required for ``fit_strategy='frozen'`` (a projection from an earlier run, so
        several releases of one source are measured in the same space). Not part of
        the cache identity; pass the same object to compare like with like.
    """

    requires_aligned_inputs = False

    def __init__(
        self,
        qids: Sequence[Any],
        auxiliary: Optional[Sequence[str]] = None,
        blocking_ladder: Optional[Sequence[Sequence[str]]] = None,
        tau: float = 0.9,
        fit_strategy: str = "joint",
        variance_retained: float = 0.90,
        min_components: int = 3,
        max_components: int = 50,
        max_categories: int = 50,
        qid_as_features: bool = True,
        null_policy: str = "equal_null",
        wildcard_tokens: Sequence[str] = ("*",),
        direction: str = "source_to_release",
        linkage_unit: str = "subject",
        ground_truth: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        max_comparisons: int = 20_000_000_000,
        max_group_pairs: int = 300_000_000,
        chunk_pairs: int = 1_000_000,
        tie_tolerance: float = 1e-9,
        zero_norm_tolerance: float = 1e-9,
        name: str = "cvpl_linkage",
        frozen_projection: Optional[CVPLProjection] = None,
        **kwargs: Any,
    ) -> None:
        params: Dict[str, Any] = dict(
            qids=qids, auxiliary=auxiliary or (), blocking_ladder=blocking_ladder, tau=tau,
            fit_strategy=fit_strategy, variance_retained=variance_retained,
            min_components=min_components, max_components=max_components,
            max_categories=max_categories, qid_as_features=qid_as_features,
            null_policy=null_policy, wildcard_tokens=wildcard_tokens, direction=direction,
            linkage_unit=linkage_unit, ground_truth=ground_truth, seed=seed,
            max_comparisons=max_comparisons, max_group_pairs=max_group_pairs,
            chunk_pairs=chunk_pairs, tie_tolerance=tie_tolerance,
            zero_norm_tolerance=zero_norm_tolerance,
        )
        for forbidden in ("columns", "column_mapping", "sample_size"):
            if kwargs.get(forbidden):
                raise InvalidParameterError(
                    forbidden, kwargs[forbidden],
                    "not used by CVPL: declare the attacker's attributes with 'qids' and "
                    "'auxiliary'; the source row count is always the denominator",
                )
            kwargs.pop(forbidden, None)
        self._cvpl = CVPLConfig.from_dict(params)  # typed validation, before anything runs
        if self._cvpl.fit_strategy == "frozen" and frozen_projection is None:
            raise InvalidParameterError(
                "frozen_projection", None, "fit_strategy='frozen' needs frozen_projection"
            )
        self._frozen_projection = frozen_projection

        kwargs.setdefault("description", "CVPL linkage exposure measurement")
        kwargs["optimize_memory"] = False  # the inputs are measured as they are
        if self._cvpl.seed is not None:
            kwargs["seed"] = self._cvpl.seed
        kwargs["config"] = CVPLLinkageConfig(
            **self._cvpl.to_params(), **{k: v for k, v in kwargs.items() if k == "description"}
        )
        super().__init__(name=name, **kwargs)
        self.operation_name = self.__class__.__name__

    # -- base-class hooks -------------------------------------------------------------
    def _validate_inputs(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame) -> None:
        if not isinstance(original_df, pd.DataFrame) or not isinstance(transformed_df, pd.DataFrame):
            raise ValidationError("Both original and transformed inputs must be pandas DataFrames")

    def _optimize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _calculate_metrics_with_config(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
    ) -> Dict[str, Any]:
        # no 'empty input => {}' shortcut: an empty release is a measured 0, an empty
        # source is reported as not measured, both by the engine
        return self.calculate_metrics(original_df, transformed_df, progress_tracker=progress_tracker)

    def calculate_metrics(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if progress_tracker:
            progress_tracker.update(1, {"step": "CVPL linkage measurement"})
        result = run_cvpl(
            original_df, transformed_df, self._cvpl, frozen_projection=self._frozen_projection
        )
        self.last_projection = result.projection
        return result.to_dict()

    def _generate_visualizations(
        self, metrics: Dict[str, Any], task_dir: Path, **kwargs: Any
    ) -> Dict[str, Path]:
        return {}

    def _get_cache_parameters(self) -> Dict[str, Any]:
        return self._cvpl.to_params()
