"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL vectorization (phi) and projection (psi)
Package:       pamola_core.metrics.privacy.cvpl
License:       BSD 3-Clause

``phi``: declared attributes -> numeric vectors (numeric: mean-imputed z-score;
categorical: one-hot with a bounded width). ``psi``: PCA keeping the smallest number
of components that reaches ``variance_retained`` (within [min, max] components).

The PCA is computed from streamed sufficient statistics (mean and scatter matrix
merged chunk by chunk), so fitting needs ``O(F^2)`` memory whatever the number of
rows, and it is exact and deterministic: the eigen-decomposition of the sample
covariance matrix. Component signs are canonicalised (the first of the largest-magnitude loadings is positive;
cosine similarity does not depend on them anyway).

Stability: the retained subspace is determined by the eigenvalues only where they are separated.
Rotating the components INSIDE a retained eigenspace leaves every cosine similarity unchanged
(an orthogonal map), but when the cut-off between retained and dropped components falls inside a
(near-)degenerate eigenspace the retained subspace is arbitrary. ``ProjectionInfo`` reports
``degenerate_eigenspace`` (any examined eigenvalue pair closer than ``DEGENERACY_REL_TOL``),
``cutoff_in_degenerate_eigenspace`` (the harmful case) and ``min_relative_eigengap``.

``fit_strategy``: ``joint`` fits on source + release (the original CVPL), ``source_only``
on the source alone, ``frozen`` reuses a projection fitted earlier so that several
releases of one source are measured in the same space.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pamola_core.errors.exceptions import InvalidParameterError, ResourceError

from .config import ALGORITHM_VERSION, CVPLConfig

#: Internal category tags. Every user value is encoded as ``"v:" + text``; the missing-value
#: and overflow buckets are ``"m:"`` and ``"o:"``, which no user value can equal.
VALUE_PREFIX = "v:"
MISSING_LABEL = "m:"
OTHER_LABEL = "o:"
_MAX_FEATURES = 5000

#: ``name -> (kind, values)``; kind is ``numeric`` (float array, NaN = unknown) or
#: ``categorical`` (object array of str, nulls already mapped to MISSING_LABEL)
FeatureColumns = Dict[str, Tuple[str, np.ndarray]]


def numeric_values(series: pd.Series) -> np.ndarray:
    """Float values of a column; anything that is not a finite number becomes NaN (imputed)."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = series.to_numpy(dtype=float, na_value=np.nan)
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def categorical_values(series: pd.Series) -> np.ndarray:
    """Tagged category labels: a user value can never equal the missing/other bucket."""
    null = series.isna().to_numpy()
    text = series.astype(object).where(~null, "").astype(str)
    # tag the DISTINCT labels once and index them: rows share the label objects instead of
    # each holding its own concatenated string (that was +350 MB at 1M rows)
    codes, uniques = pd.factorize(text)
    labels = np.empty(len(uniques), dtype=object)
    labels[:] = [VALUE_PREFIX + str(u) for u in uniques]
    tagged = labels[codes]
    tagged[null] = MISSING_LABEL
    return tagged


@dataclass
class ProjectionInfo:
    """What the projection looks like - names and numbers only."""

    n_features: int
    n_components: int
    variance_retained: Optional[float]
    n_fit_rows: int
    fit_strategy: str
    feature_columns: List[str]
    encoding: Dict[str, dict]
    imputed_values: Dict[str, Dict[str, int]]
    degenerate: bool = False
    #: Two eigenvalues examined (adjacent retained components, and the last retained one against
    #: the first dropped one) differ by less than ``DEGENERACY_REL_TOL`` of the larger one.
    degenerate_eigenspace: bool = False
    #: The cut-off between retained and dropped components falls INSIDE a (near-)degenerate
    #: eigenspace: which directions of that eigenspace are retained is then arbitrary (it
    #: depends on rounding, row order and the linear-algebra library), and so can the result.
    cutoff_in_degenerate_eigenspace: bool = False
    #: Smallest relative gap ``(l_j - l_{j+1}) / l_j`` among the examined eigenvalue pairs.
    min_relative_eigengap: Optional[float] = None


@dataclass
class CVPLProjection:
    """A fitted ``psi . phi``; reusable (``fit_strategy='frozen'``)."""

    kinds: Dict[str, str]
    num_mean: Dict[str, float]
    num_scale: Dict[str, float]
    categories: Dict[str, List[str]]
    has_other: Dict[str, bool]
    pca_mean: np.ndarray
    components: np.ndarray  # (F, k)
    info: ProjectionInfo
    _cat_index: Dict[str, Dict[str, int]] = field(default_factory=dict, repr=False)

    def fingerprint(self) -> str:
        """Digest of the whole fitted representation: the ORDERED encoded feature layout
        (the order ``phi`` concatenates its blocks in), the scaling and category lists,
        the centering and the components with their shapes. Two fits that select different
        physical attributes never share an identity, even if their component arrays look
        alike."""
        layout = []
        for name, kind in self.kinds.items():  # insertion order == phi's block order
            if kind == "numeric":
                layout.append([name, kind, float(self.num_mean[name]).hex(), float(self.num_scale[name]).hex()])
            else:
                layout.append([name, kind, list(self.categories[name])])
        h = hashlib.sha256()
        h.update(ALGORITHM_VERSION.encode("utf-8"))
        h.update(b"\x00")
        h.update(json.dumps(layout, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
        for arr in (self.pca_mean, self.components):
            a64 = np.ascontiguousarray(arr, dtype=np.float64)
            h.update(b"\x00" + json.dumps(list(a64.shape)).encode("ascii") + b"\x00")
            h.update(a64.tobytes())
        return h.hexdigest()

    # -- phi -------------------------------------------------------------------
    def _phi(self, cols: FeatureColumns, start: int, stop: int) -> np.ndarray:
        blocks: List[np.ndarray] = []
        for name, kind in self.kinds.items():
            _, values = cols[name]
            chunk = values[start:stop]
            if kind == "numeric":
                x = np.where(np.isnan(chunk), self.num_mean[name], chunk)
                blocks.append(((x - self.num_mean[name]) / self.num_scale[name])[:, None])
            else:
                width = len(self.categories[name])
                code = self._cat_codes(name, chunk)
                onehot = np.zeros((len(chunk), width))
                valid = code >= 0
                onehot[np.flatnonzero(valid), code[valid]] = 1.0
                blocks.append(onehot)
        return np.hstack(blocks) if blocks else np.zeros((stop - start, 0))

    def _cat_codes(self, name: str, chunk: np.ndarray) -> np.ndarray:
        """Category index of every label (the shared bucket for unseen/rare ones, -1 for a
        label that has no block position at all)."""
        cats = self.categories[name]
        index = self._cat_index.get(name)
        if index is None:
            index = {c: i for i, c in enumerate(cats)}
            self._cat_index[name] = index
        other = index.get(OTHER_LABEL)
        codes, uniques = pd.factorize(chunk)
        mapped = np.array(
            [index.get(u, other if other is not None else -1) for u in uniques] + [-1],
            dtype=np.int64,
        )
        return mapped[np.where(codes < 0, len(uniques), codes)]

    def column_codes(self, name: str, values: np.ndarray) -> np.ndarray:
        """What fixes the ``phi`` block of one feature column, one number per row: the
        standardised value of a numeric column, the category index of a categorical one.
        Two rows have the same encoded block iff they have the same code, so exact equality of
        encoded vectors is equality of these codes - without ever building the dense block."""
        if self.kinds[name] == "numeric":
            x = np.where(np.isnan(values), self.num_mean[name], values)
            return (x - self.num_mean[name]) / self.num_scale[name] + 0.0  # canonical zero
        return self._cat_codes(name, values)

    def phi_rows(self, cols: FeatureColumns, rows: np.ndarray) -> np.ndarray:
        """phi for an arbitrary subset of rows (used by the exact-match fallback)."""
        sub = {n: (k, v[rows]) for n, (k, v) in cols.items()}
        return self._phi(sub, 0, len(rows))

    # -- psi -------------------------------------------------------------------
    def project(self, cols: FeatureColumns, n_rows: int, chunk_rows: Optional[int] = None) -> np.ndarray:
        """Projected coordinates of every row, computed chunk by chunk."""
        k = self.components.shape[1]
        out = np.empty((n_rows, k))
        step = chunk_rows or max(1000, 2_000_000 // max(1, self.info.n_features))
        for start in range(0, n_rows, step):
            stop = min(start + step, n_rows)
            out[start:stop] = (self._phi(cols, start, stop) - self.pca_mean) @ self.components
        return out


#: Relative gap ``(l_j - l_{j+1}) / l_j`` below which two eigenvalues are treated as equal.
DEGENERACY_REL_TOL = 1e-6


def _eigengap_diagnostics(evals: np.ndarray, k: int):
    """(degenerate_eigenspace, cutoff_in_degenerate_eigenspace, min_relative_eigengap).

    Examines the adjacent eigenvalue pairs among the ``k`` retained components and the pair
    formed by the last retained and the first dropped one (when there is one).
    """
    top = float(evals[0]) if len(evals) else 0.0
    if top <= 0.0:
        return False, False, None
    stop = min(k + 1, len(evals))
    gaps = []
    for j in range(stop - 1):
        lj = float(evals[j])
        gaps.append((lj - float(evals[j + 1])) / lj if lj > 0.0 else 0.0)
    if not gaps:
        return False, False, None
    cutoff = k < len(evals) and gaps[k - 1] < DEGENERACY_REL_TOL
    return bool(min(gaps) < DEGENERACY_REL_TOL), bool(cutoff), float(min(gaps))


def _row_chunks(n: int, width: int) -> Iterator[Tuple[int, int]]:
    step = max(1000, 2_000_000 // max(1, width))
    for start in range(0, n, step):
        yield start, min(start + step, n)


def fit_projection(
    cfg: CVPLConfig,
    source: FeatureColumns,
    n_source: int,
    release: FeatureColumns,
    n_release: int,
) -> CVPLProjection:
    """Fit phi and psi on the rows chosen by ``cfg.fit_strategy`` (not ``frozen``)."""
    if cfg.fit_strategy == "joint":
        fit_sets = [(source, n_source), (release, n_release)]
    elif cfg.fit_strategy == "source_only":
        fit_sets = [(source, n_source)]
    else:
        raise InvalidParameterError("fit_strategy", cfg.fit_strategy, "fit_projection does not fit 'frozen'")
    n_fit = sum(n for _, n in fit_sets)
    kinds = {name: kind for name, (kind, _) in source.items()}

    num_mean: Dict[str, float] = {}
    num_scale: Dict[str, float] = {}
    categories: Dict[str, List[str]] = {}
    has_other: Dict[str, bool] = {}
    encoding: Dict[str, dict] = {}
    imputed: Dict[str, Dict[str, int]] = {}

    for name, kind in kinds.items():
        if kind == "numeric":
            parts = [cols[name][1] for cols, _ in fit_sets]
            allv = np.concatenate(parts)
            finite = allv[~np.isnan(allv)]
            mean = float(finite.mean()) if len(finite) else 0.0
            filled = np.where(np.isnan(allv), mean, allv)
            std = float(filled.std()) if len(filled) else 0.0
            num_mean[name] = mean
            num_scale[name] = std if std > 0 else 1.0
            imputed[name] = {
                "source": int(np.isnan(source[name][1]).sum()),
                "release": int(np.isnan(release[name][1]).sum()),
            }
        else:
            allv = np.concatenate([cols[name][1] for cols, _ in fit_sets])
            vals, counts = np.unique(allv.astype(str), return_counts=True)
            n_distinct = len(vals)
            bounded = n_distinct > cfg.max_categories
            bucketed: List[int] = []
            # Rarest values go to the shared bucket. Equally frequent values: the
            # lexicographically earlier ones are bucketed first (a stable sort on count
            # over the sorted values) - the rule scikit-learn's OneHotEncoder applies,
            # kept so the historical CVPL numbers are reproducible.
            if bounded:
                n_keep = cfg.max_categories - 1
                asc = sorted(range(n_distinct), key=lambda i: int(counts[i]))  # stable: sorted-value order
                bucketed = asc[: n_distinct - n_keep]
                keep = sorted(str(vals[i]) for i in asc[n_distinct - n_keep:])
                cats = keep + [OTHER_LABEL]
            else:
                cats = sorted(str(v) for v in vals)
                # unseen values are possible when the fit set is not the whole data
                if cfg.fit_strategy != "joint":
                    cats = cats + [OTHER_LABEL]
            categories[name] = cats
            has_other[name] = OTHER_LABEL in cats
            encoding[name] = {
                "distinct_values": int(n_distinct),
                "encoded_width": len(cats),
                "bounded": bool(bounded),
                "values_in_other_bucket": int(sum(int(counts[i]) for i in bucketed)) if bounded else 0,
                "distinct_values_in_other_bucket": len(bucketed),
            }

    proj = CVPLProjection(
        kinds=kinds, num_mean=num_mean, num_scale=num_scale, categories=categories,
        has_other=has_other, pca_mean=np.zeros(0), components=np.zeros((0, 0)),
        info=ProjectionInfo(0, 0, None, n_fit, cfg.fit_strategy, list(kinds), encoding, imputed),
    )
    n_features = sum(1 if k == "numeric" else len(categories[n]) for n, k in kinds.items())
    if n_features > _MAX_FEATURES:
        raise ResourceError(
            operation="CVPL",
            reason=(f"the representation has {n_features:,} features (limit {_MAX_FEATURES:,}); "
                    "lower max_categories or declare fewer categorical attributes"),
        )
    if n_features == 0:
        raise InvalidParameterError("auxiliary/qids", [], "no feature columns to build a representation from")

    # streamed sufficient statistics: n, mean, scatter (Chan et al. merge)
    n_seen = 0
    mean = np.zeros(n_features)
    scatter = np.zeros((n_features, n_features))
    for cols, n in fit_sets:
        for start, stop in _row_chunks(n, n_features):
            x = proj._phi(cols, start, stop)
            m = x.shape[0]
            if m == 0:
                continue
            cm = x.mean(axis=0)
            xc = x - cm
            cs = xc.T @ xc
            delta = cm - mean
            tot = n_seen + m
            scatter += cs + np.outer(delta, delta) * (n_seen * m / tot)
            mean += delta * (m / tot)
            n_seen = tot
    if n_seen < 2:
        cov = np.zeros((n_features, n_features))
    else:
        cov = scatter / (n_seen - 1)
    evals, evecs = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    total = float(evals.sum())
    max_k = max(1, min(n_seen, n_features, cfg.max_components))
    if total <= 0.0:
        k = 1
        retained = None
        components = np.zeros((n_features, 1))
        degenerate = True
        eig_deg, eig_cut, eig_gap = False, False, None
    else:
        cum = np.cumsum(evals[:max_k] / total)
        k_var = min(int(np.searchsorted(cum, cfg.variance_retained)) + 1, max_k)
        k = max(1, min(max(cfg.min_components, k_var), max_k))
        retained = float(cum[k - 1])
        components = evecs[:, :k].copy()
        # canonical sign: the largest-magnitude loading of each component is positive
        for j in range(k):
            mags = np.abs(components[:, j])
            # first loading within rounding of the largest one: a tie between equal-magnitude
            # loadings must not be decided by the last bits
            lead = int(np.argmax(mags >= mags.max() * (1.0 - 1e-9)))
            if components[lead, j] < 0:
                components[:, j] = -components[:, j]
        degenerate = False
        eig_deg, eig_cut, eig_gap = _eigengap_diagnostics(evals, k)
    proj.pca_mean = mean
    proj.components = components
    proj.info = ProjectionInfo(
        n_features=n_features, n_components=int(k), variance_retained=retained, n_fit_rows=int(n_seen),
        fit_strategy=cfg.fit_strategy, feature_columns=list(kinds), encoding=encoding,
        imputed_values=imputed, degenerate=degenerate, degenerate_eigenspace=eig_deg,
        cutoff_in_degenerate_eigenspace=eig_cut, min_relative_eigengap=eig_gap,
    )
    return proj
