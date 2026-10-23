"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL configuration
Package:       pamola_core.metrics.privacy.cvpl
License:       BSD 3-Clause

Typed, validated configuration of the CVPL linkage measurement
(``B_lambda -> phi -> PCA -> cosine -> tau``). Every parameter that changes a
number is explicit here; nothing is added to the attacker's knowledge implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pamola_core.errors.exceptions import InvalidParameterError

#: Version of the measured quantity. Bump when the meaning of a number changes.
ALGORITHM_ID = "cvpl-operator-chain"
ALGORITHM_VERSION = "1.0"

FIT_STRATEGIES = ("joint", "source_only", "frozen")
NULL_POLICIES = ("equal_null", "wildcard", "incompatible")
DIRECTIONS = ("source_to_release",)
LINKAGE_UNITS = ("subject",)
GT_SCOPES = ("complete", "partial")
QID_KINDS = ("auto", "numeric", "categorical")
MAX_LADDER_RUNGS = 8


@dataclass(frozen=True)
class QIDSpec:
    """One quasi-identifier the attacker can use to restrict candidates.

    Parameters
    ----------
    name : column present in both the source and the released dataset.
    kind : ``numeric`` (values, intervals), ``categorical`` (labels, hierarchy) or
        ``auto`` (numeric if the source column has a numeric dtype).
    bin_width : numeric only. Point values are compared by the bin
        ``[floor(v/w)*w, floor(v/w)*w + w)`` (the "age decade" of the original CVPL);
        ``None`` compares exact values.
    hierarchy : categorical only. ``{child: parent}``; a released value that is an
        ancestor of the source value (or the other way round) is compatible.
    generalized : ``None`` detects generalized released values automatically;
        ``False`` declares the QID exact-only (unmatched released values are simply
        incompatible); ``True`` declares it generalized, which requires a hierarchy
        for a categorical QID.
    """

    name: str
    kind: str = "auto"
    bin_width: Optional[float] = None
    hierarchy: Optional[Mapping[str, str]] = None
    generalized: Optional[bool] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidParameterError("qids[].name", self.name, "must be a non-empty column name")
        if self.kind not in QID_KINDS:
            raise InvalidParameterError(f"qids[{self.name}].kind", self.kind, f"one of {QID_KINDS}")
        if self.bin_width is not None:
            if self.kind == "categorical" or not (float(self.bin_width) > 0):
                raise InvalidParameterError(
                    f"qids[{self.name}].bin_width", self.bin_width,
                    "must be > 0 and only applies to numeric QIDs",
                )
        if self.hierarchy is not None:
            if self.kind == "numeric":
                raise InvalidParameterError(
                    f"qids[{self.name}].hierarchy", "...", "only applies to categorical QIDs"
                )
            if not isinstance(self.hierarchy, Mapping):
                raise InvalidParameterError(
                    f"qids[{self.name}].hierarchy", type(self.hierarchy).__name__,
                    "must be a mapping {child: parent}",
                )


@dataclass(frozen=True)
class GroundTruthSpec:
    """Evaluation-only identity link between the two datasets.

    The columns hold subject identifiers: a source row and a released row refer to
    the same subject when the identifiers are equal. They are read by the auditor's
    evaluation only: they are never features, never blocking keys and never leave
    the process (artifacts carry counts, not identifiers).

    ``scope='complete'`` asserts that every source subject is listed and every
    released row carries an identifier, so a source subject with no released row
    really has no counterpart (suppression) and scores 0. ``scope='partial'`` says
    the links are only partly known: the numbers are then reported for the known
    subset only, never as a population measure.
    """

    source_id_column: str
    release_id_column: str
    scope: str = "complete"

    def __post_init__(self) -> None:
        if not self.source_id_column or not self.release_id_column:
            raise InvalidParameterError(
                "ground_truth", "...", "source_id_column and release_id_column are required"
            )
        if self.scope not in GT_SCOPES:
            raise InvalidParameterError("ground_truth.scope", self.scope, f"one of {GT_SCOPES}")


@dataclass(frozen=True)
class CVPLConfig:
    """Complete CVPL measurement configuration."""

    qids: Tuple[QIDSpec, ...]
    auxiliary: Tuple[str, ...] = ()
    blocking_ladder: Optional[Tuple[Tuple[str, ...], ...]] = None
    tau: float = 0.9
    fit_strategy: str = "joint"
    variance_retained: float = 0.90
    min_components: int = 3
    max_components: int = 50
    max_categories: int = 50
    qid_as_features: bool = True
    null_policy: str = "equal_null"
    wildcard_tokens: Tuple[str, ...] = ("*",)
    direction: str = "source_to_release"
    linkage_unit: str = "subject"
    ground_truth: Optional[GroundTruthSpec] = None
    seed: Optional[int] = None
    #: pre-flight ceiling on compared (source x candidate) pairs per ladder rung
    max_comparisons: int = 20_000_000_000
    #: ceiling on group-level compatibility checks (wildcard / interval / hierarchy)
    max_group_pairs: int = 300_000_000
    #: pairs held in memory at once by the blocked scan
    chunk_pairs: int = 4_000_000
    tie_tolerance: float = 1e-9
    zero_norm_tolerance: float = 1e-9

    # -- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.qids:
            raise InvalidParameterError("qids", [], "at least one quasi-identifier is required")
        names = [q.name for q in self.qids]
        if len(set(names)) != len(names):
            raise InvalidParameterError("qids", names, "quasi-identifier names must be unique")
        if len(set(self.auxiliary)) != len(self.auxiliary):
            raise InvalidParameterError("auxiliary", list(self.auxiliary), "names must be unique")
        overlap = set(names) & set(self.auxiliary)
        if overlap:
            raise InvalidParameterError(
                "auxiliary", sorted(overlap), "a column cannot be both a QID and auxiliary"
            )
        if not (0.0 < float(self.tau) <= 1.0):
            raise InvalidParameterError("tau", self.tau, "must be in (0, 1]")
        if self.fit_strategy not in FIT_STRATEGIES:
            raise InvalidParameterError("fit_strategy", self.fit_strategy, f"one of {FIT_STRATEGIES}")
        if not (0.0 < float(self.variance_retained) <= 1.0):
            raise InvalidParameterError("variance_retained", self.variance_retained, "must be in (0, 1]")
        if self.min_components < 1 or self.max_components < self.min_components:
            raise InvalidParameterError(
                "min_components/max_components", (self.min_components, self.max_components),
                "need 1 <= min_components <= max_components",
            )
        if self.max_categories < 2:
            raise InvalidParameterError("max_categories", self.max_categories, "must be >= 2")
        if self.null_policy not in NULL_POLICIES:
            raise InvalidParameterError("null_policy", self.null_policy, f"one of {NULL_POLICIES}")
        if self.direction not in DIRECTIONS:
            raise InvalidParameterError(
                "direction", self.direction,
                f"only {DIRECTIONS} is supported in 1.0 (release_to_reference needs a reference "
                "population and is not implemented)",
            )
        if self.linkage_unit not in LINKAGE_UNITS:
            raise InvalidParameterError("linkage_unit", self.linkage_unit, f"one of {LINKAGE_UNITS}")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise InvalidParameterError("seed", self.seed, "must be an integer or None (0 is valid)")
        for name, value in (("max_comparisons", self.max_comparisons),
                            ("max_group_pairs", self.max_group_pairs),
                            ("chunk_pairs", self.chunk_pairs)):
            if int(value) < 1:
                raise InvalidParameterError(name, value, "must be >= 1")
        if self.ground_truth is not None:
            used = set(names) | set(self.auxiliary)
            clash = used & {self.ground_truth.source_id_column, self.ground_truth.release_id_column}
            if clash:
                raise InvalidParameterError(
                    "ground_truth", sorted(clash),
                    "ground-truth columns are evaluation-only and cannot be QIDs or features",
                )
        for q in self.qids:
            if q.generalized and q.kind != "numeric" and not q.hierarchy:
                # kind 'auto' may still turn out numeric; the engine re-checks then
                if q.kind == "categorical":
                    raise InvalidParameterError(
                        f"qids[{q.name}].generalized", True,
                        "a generalized categorical QID needs a hierarchy",
                    )
        if self.blocking_ladder is not None:
            ladder = self.blocking_ladder
            if not (1 <= len(ladder) <= MAX_LADDER_RUNGS):
                raise InvalidParameterError(
                    "blocking_ladder", len(ladder), f"1 to {MAX_LADDER_RUNGS} rungs"
                )
            known = set(names)
            seen = set()
            for rung in ladder:
                unknown = set(rung) - known
                if unknown:
                    raise InvalidParameterError(
                        "blocking_ladder", sorted(unknown), "rungs may only use declared QIDs"
                    )
                if len(set(rung)) != len(rung):
                    raise InvalidParameterError("blocking_ladder", list(rung), "repeated QID in a rung")
                key = frozenset(rung)
                if key in seen:
                    raise InvalidParameterError("blocking_ladder", list(rung), "duplicate rung")
                seen.add(key)

    # -- derived ------------------------------------------------------------
    @property
    def qid_names(self) -> Tuple[str, ...]:
        return tuple(q.name for q in self.qids)

    @property
    def ladder(self) -> Tuple[Tuple[str, ...], ...]:
        """The effective ladder: the declared one, or one rung with every QID."""
        return self.blocking_ladder if self.blocking_ladder is not None else (self.qid_names,)

    @property
    def feature_columns(self) -> Tuple[str, ...]:
        """Attacker-visible attributes the representation is built from: Q (if
        ``qid_as_features``) then A. Nothing else is ever added."""
        cols: List[str] = list(self.qid_names) if self.qid_as_features else []
        cols.extend(self.auxiliary)
        return tuple(cols)

    @property
    def effective_seed(self) -> int:
        return 0 if self.seed is None else int(self.seed)

    # -- (de)serialisation ----------------------------------------------------
    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> "CVPLConfig":
        """Build from a plain dict (JSON/YAML friendly). Unknown keys are rejected."""
        data = dict(params)
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - allowed
        if unknown:
            raise InvalidParameterError("cvpl", sorted(unknown), f"unknown parameter(s); allowed: {sorted(allowed)}")
        qids = []
        for q in data.get("qids", ()):
            if isinstance(q, QIDSpec):
                qids.append(q)
            elif isinstance(q, str):
                qids.append(QIDSpec(name=q))
            elif isinstance(q, Mapping):
                qids.append(QIDSpec(**dict(q)))
            else:
                raise InvalidParameterError("qids", type(q).__name__, "entries are names or mappings")
        data["qids"] = tuple(qids)
        data["auxiliary"] = tuple(data.get("auxiliary", ()) or ())
        if data.get("blocking_ladder") is not None:
            data["blocking_ladder"] = tuple(tuple(r) for r in data["blocking_ladder"])
        if data.get("wildcard_tokens") is not None:
            data["wildcard_tokens"] = tuple(data["wildcard_tokens"])
        gt = data.get("ground_truth")
        if isinstance(gt, Mapping):
            data["ground_truth"] = GroundTruthSpec(**dict(gt))
        return cls(**data)

    def to_params(self) -> Dict[str, Any]:
        """Plain JSON-able dict that :meth:`from_dict` turns back into this config."""
        data: Dict[str, Any] = {
            "qids": [
                {k: v for k, v in {
                    "name": q.name, "kind": q.kind, "bin_width": q.bin_width,
                    "hierarchy": None if q.hierarchy is None else dict(q.hierarchy),
                    "generalized": q.generalized}.items() if v is not None}
                for q in self.qids
            ],
            "auxiliary": list(self.auxiliary),
            "blocking_ladder": None if self.blocking_ladder is None else [list(r) for r in self.blocking_ladder],
            "wildcard_tokens": list(self.wildcard_tokens),
            "ground_truth": None if self.ground_truth is None else {
                "source_id_column": self.ground_truth.source_id_column,
                "release_id_column": self.ground_truth.release_id_column,
                "scope": self.ground_truth.scope},
        }
        for key in ("tau", "fit_strategy", "variance_retained", "min_components", "max_components",
                    "max_categories", "qid_as_features", "null_policy", "direction", "linkage_unit",
                    "seed", "max_comparisons", "max_group_pairs", "chunk_pairs", "tie_tolerance",
                    "zero_norm_tolerance"):
            data[key] = getattr(self, key)
        return data

    def describe(self) -> Dict[str, Any]:
        """Parameters for the result (names and numbers only; never data values)."""
        return {
            "qids": [
                {"name": q.name, "kind": q.kind, "bin_width": q.bin_width,
                 "has_hierarchy": q.hierarchy is not None, "generalized": q.generalized}
                for q in self.qids
            ],
            "auxiliary": list(self.auxiliary),
            "blocking_ladder": [list(r) for r in self.ladder],
            "tau": float(self.tau),
            "fit_strategy": self.fit_strategy,
            "variance_retained": float(self.variance_retained),
            "min_components": self.min_components,
            "max_components": self.max_components,
            "max_categories": self.max_categories,
            "qid_as_features": self.qid_as_features,
            "null_policy": self.null_policy,
            "wildcard_tokens": list(self.wildcard_tokens),
            "direction": self.direction,
            "linkage_unit": self.linkage_unit,
            "ground_truth_declared": self.ground_truth is not None,
            "ground_truth_scope": None if self.ground_truth is None else self.ground_truth.scope,
            "seed": self.seed,
            "max_comparisons": self.max_comparisons,
        }
