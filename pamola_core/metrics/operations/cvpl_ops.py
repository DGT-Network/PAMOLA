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
from typing import Any, Dict, Optional

import pandas as pd

from pamola_core.errors.exceptions import InvalidParameterError, ValidationError
from pamola_core.metrics.base_metrics_op import MetricsOperation
from pamola_core.metrics.privacy.cvpl import CVPLConfig, CVPLProjection, run_cvpl
from pamola_core.metrics.schemas.cvpl_op_core_schema import CVPLLinkageConfig
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.progress import HierarchicalProgressTracker

_CONFIG_KEYS = frozenset(CVPLConfig.__dataclass_fields__)


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
        name: str = "cvpl_linkage",
        frozen_projection: Optional[CVPLProjection] = None,
        **kwargs: Any,
    ) -> None:
        params: Dict[str, Any] = {k: kwargs.pop(k) for k in list(kwargs) if k in _CONFIG_KEYS}
        for forbidden in ("columns", "column_mapping", "sample_size"):
            if kwargs.get(forbidden):
                raise InvalidParameterError(
                    forbidden, kwargs[forbidden],
                    "not used by CVPL: declare the attacker's attributes with 'qids' and "
                    "'auxiliary'; the source row count is always the denominator",
                )
            kwargs.pop(forbidden, None)
        if "qids" not in params:
            raise InvalidParameterError("qids", None, "at least one quasi-identifier is required")
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
        config = CVPLLinkageConfig(**self._cvpl.to_params(), **{
            k: v for k, v in kwargs.items() if k in ("name", "description")
        })
        kwargs["config"] = config
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
