"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL measurement engine
Package:       pamola_core.metrics.privacy.cvpl
License:       BSD 3-Clause

The CVPL operator chain ``B_lambda -> phi -> psi -> s -> tau`` and the quantities
measured with it. See ``docs`` ("CVPL linkage metric") for what each number means.

For source record ``x_i`` and a ladder rung ``lambda``:

* ``C_i`` = released rows that are blocking-compatible with ``x_i`` and whose cosine
  similarity to ``x_i`` in the projected space is at least ``tau``;
* ``cvpl_lr`` = share of the ``N`` source records with ``|C_i| > 0`` (existence of a
  plausible link; *not* a re-identification probability);
* with a complete ground truth ``T_i`` (the subject's own released rows):
  ``cvpl_id_uniform_gt`` = mean of ``|subjects(C_i) & {s_i}| / |subjects(C_i)|``
  (0 when ``C_i`` is empty) - the expected success of a uniform choice among the
  candidates; ``cvpl_top1_success_gt`` = mean success of the maximum-similarity
  candidate (ties broken uniformly, in expectation); ``precision_at_tau`` = the same
  success over the records that *declare* a link.

Nothing here materialises a ``|D| x |D'|`` matrix: candidates are generated from
blocks, similarities are computed in bounded chunks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pamola_core.errors.exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    ResourceError,
)

from .compat import (
    QIDEncoding,
    classify_transition,
    encode_qid,
    plan_rung,
    resolve_kind,
)
from .config import ALGORITHM_ID, ALGORITHM_VERSION, CVPLConfig
from .projection import (
    CVPLProjection,
    FeatureColumns,
    categorical_values,
    fit_projection,
    numeric_values,
)

#: tolerance on ``cos >= tau`` so that identical vectors pass ``tau = 1``
_TAU_EPS = 1e-12


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------
@dataclass
class PreparedGroundTruth:
    status: str  # complete | partial | absent | invalid
    reason: Optional[str]
    n_known: int
    src_code: Optional[np.ndarray] = None  # subject code per source row, -1 unknown
    rel_code: Optional[np.ndarray] = None  # subject code per released row
    rel_unique: bool = True
    known_mask: Optional[np.ndarray] = None  # source rows the measures are computed on
    true_row: Optional[np.ndarray] = None  # unique-release fast path: row of own subject
    n_subjects: int = 0


def prepare_ground_truth(cfg: CVPLConfig, source: pd.DataFrame, release: pd.DataFrame) -> PreparedGroundTruth:
    n = len(source)
    spec = cfg.ground_truth
    if spec is None:
        return PreparedGroundTruth("absent", "no_ground_truth_declared", 0)
    src = source[spec.source_id_column]
    rel = release[spec.release_id_column]
    src_null = src.isna().to_numpy()
    rel_null = rel.isna().to_numpy()
    n_known_ids = int((~src_null).sum())
    if n_known_ids == 0:
        return PreparedGroundTruth("absent", "ground_truth_has_no_source_identifiers", 0)
    # one code space for both id columns
    both = pd.concat([src[~src_null].astype(str), rel[~rel_null].astype(str)], ignore_index=True)
    codes, uniques = pd.factorize(both)
    src_code = np.full(n, -1, dtype=np.int64)
    src_code[~src_null] = codes[: n_known_ids]
    rel_code = np.empty(len(release), dtype=np.int64)
    rel_code[~rel_null] = codes[n_known_ids:]
    n_anon = int(rel_null.sum())
    rel_code[rel_null] = len(uniques) + np.arange(n_anon)  # each unidentified row is its own subject
    known_src_codes = src_code[src_code >= 0]
    if len(np.unique(known_src_codes)) != len(known_src_codes):
        return PreparedGroundTruth(
            "invalid", "source_subject_identifiers_not_unique", n_known_ids,
        )
    rel_unique = len(np.unique(rel_code)) == len(rel_code)
    in_release = np.isin(src_code, rel_code) & (src_code >= 0)
    if (spec.scope == "complete" and n_known_ids == n and n_anon == 0):
        status, reason = "complete", None
        known = np.ones(n, dtype=bool)
    else:
        status = "partial"
        if spec.scope == "partial":
            reason = "scope_declared_partial"
            known = in_release  # only subjects whose release rows are confirmed
        elif n_anon:
            reason = "released_rows_without_identifier"
            known = src_code >= 0
        else:
            reason = "source_records_without_identifier"
            known = src_code >= 0
    true_row = None
    if rel_unique:
        row_of = np.full(len(uniques) + n_anon, -1, dtype=np.int64)
        row_of[rel_code] = np.arange(len(release))
        true_row = np.where(src_code >= 0, row_of[np.where(src_code >= 0, src_code, 0)], -1)
    return PreparedGroundTruth(
        status, reason, int(known.sum()), src_code, rel_code, rel_unique, known, true_row,
        n_subjects=len(uniques) + n_anon,
    )


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
@dataclass
class RungScan:
    n_cand: np.ndarray
    linked: np.ndarray
    best: np.ndarray
    undefined: np.ndarray
    fallback_any: np.ndarray
    fallback_only: np.ndarray
    cnt: Optional[np.ndarray] = None  # distinct subjects among passing candidates
    hit: Optional[np.ndarray] = None  # own subject among them
    tie_cnt: Optional[np.ndarray] = None
    tie_hit: Optional[np.ndarray] = None
    fallback_pairs: int = 0
    pairs_compared: int = 0


class _Scanner:
    def __init__(
        self, cfg: CVPLConfig, zs: np.ndarray, zr: np.ndarray, deg_s: np.ndarray, deg_r: np.ndarray,
        gt: Optional[PreparedGroundTruth], projection: CVPLProjection,
        cols_s: FeatureColumns, cols_r: FeatureColumns,
    ) -> None:
        self.cfg, self.zs, self.zr = cfg, zs, zr
        self.deg_s, self.deg_r = deg_s, deg_r
        self.gt, self.projection, self.cols_s, self.cols_r = gt, projection, cols_s, cols_r
        self.any_deg = bool(deg_s.any() or deg_r.any())

    # one scan: mode 'link' (threshold tau) then optionally 'top1' (ties of the maximum)
    def scan(self, plan, with_gt: bool) -> RungScan:
        n = len(self.zs)
        out = RungScan(
            n_cand=np.zeros(n, dtype=np.int64), linked=np.zeros(n, dtype=bool),
            best=np.full(n, -np.inf), undefined=np.zeros(n, dtype=bool),
            fallback_any=np.zeros(n, dtype=bool), fallback_only=np.zeros(n, dtype=bool),
        )
        if with_gt:
            out.cnt = np.zeros(n, dtype=np.int64)
            out.hit = np.zeros(n, dtype=bool)
        fb_pairs: Dict[int, List[int]] = {}  # source row -> candidate rows matched exactly
        for src_rows, cand_rows in plan.blocks():
            out.n_cand[src_rows] = len(cand_rows)
            self._block(src_rows, cand_rows, out, mode="link", fb=fb_pairs)
        # exact-equality fallback links count as passing candidates with similarity 1
        for row, cands in fb_pairs.items():
            out.fallback_any[row] = True
            out.fallback_pairs += len(cands)
            if not out.linked[row]:
                out.fallback_only[row] = True
            out.linked[row] = True
            out.best[row] = max(out.best[row], 1.0)
            if with_gt:
                self._add_subjects(out.cnt, out.hit, row, np.asarray(cands))
        out.undefined &= ~out.linked  # an undefined pair matters only if no link was found
        if with_gt:
            out.tie_cnt = np.zeros(n, dtype=np.int64)
            out.tie_hit = np.zeros(n, dtype=bool)
            for src_rows, cand_rows in plan.blocks():
                self._block(src_rows, cand_rows, out, mode="top1", fb=None)
            for row, cands in fb_pairs.items():
                # a degenerate source's tied top candidates are exactly its exact-equal rows
                self._add_subjects(out.tie_cnt, out.tie_hit, row, np.asarray(cands))
        return out

    # -- helpers -----------------------------------------------------------
    def _add_subjects(self, cnt: np.ndarray, hit: np.ndarray, row: int, cand_rows: np.ndarray) -> None:
        subj = np.unique(self.gt.rel_code[cand_rows])
        cnt[row] += len(subj)
        if self.gt.src_code[row] >= 0 and np.any(subj == self.gt.src_code[row]):
            hit[row] = True

    def _block(self, src_rows, cand_rows, out: RungScan, mode: str, fb: Optional[dict]) -> None:
        cfg = self.cfg
        src_deg = self.deg_s[src_rows]
        cand_deg = self.deg_r[cand_rows]
        vs = src_rows[~src_deg]
        vc = cand_rows[~cand_deg]
        if mode == "link":
            if cand_deg.any():
                out.undefined[vs] = True
            if src_deg.any():
                ds = src_rows[src_deg]
                out.undefined[ds] = True
                if cand_deg.any():
                    self._exact_fallback(ds, cand_rows[cand_deg], fb)
        if len(vs) == 0 or len(vc) == 0:
            return
        width = int(min(len(vc), 8192))
        step = max(1, cfg.chunk_pairs // width)
        tau = cfg.tau - _TAU_EPS
        with_gt = self.gt is not None and out.cnt is not None
        for a in range(0, len(vs), step):
            rows = vs[a: a + step]
            zq = self.zs[rows]
            if mode == "top1":
                keep = out.linked[rows]
                if not keep.any():
                    continue
                thr = np.where(keep, out.best[rows] - cfg.tie_tolerance, np.inf)
            else:
                thr = np.full(len(rows), tau)
            acc = _SubjectAcc(len(rows), self.gt) if (with_gt and not self.gt.rel_unique) else None
            best = np.full(len(rows), -np.inf)
            any_pass = np.zeros(len(rows), dtype=bool)
            cnt = np.zeros(len(rows), dtype=np.int64)
            hit = np.zeros(len(rows), dtype=bool)
            for b in range(0, len(vc), width):
                cc = vc[b: b + width]
                sim = zq @ self.zr[cc].T
                out.pairs_compared += sim.size if mode == "link" else 0
                if mode == "link":
                    best = np.maximum(best, sim.max(axis=1))
                pas = sim >= thr[:, None]
                if not with_gt:
                    any_pass |= pas.any(axis=1)
                    continue
                if acc is None:  # every released row is its own subject
                    cnt += pas.sum(axis=1)
                    tr = self.gt.true_row[rows]
                    ok = tr >= 0
                    if ok.any():
                        colpos = _lookup(cc, tr)
                        sel = ok & (colpos >= 0)
                        if sel.any():
                            idx = np.flatnonzero(sel)
                            hit[idx] |= pas[idx, colpos[idx]]
                else:
                    r_idx, c_idx = np.nonzero(pas)
                    if len(r_idx):
                        acc.add(r_idx, self.gt.rel_code[cc[c_idx]])
            if mode == "link":
                out.best[rows] = np.maximum(out.best[rows], best)
                if with_gt:
                    if acc is not None:
                        cnt, hit = acc.result(self.gt.src_code[rows])
                    out.cnt[rows] += cnt
                    out.hit[rows] |= hit
                    out.linked[rows] |= cnt > 0
                else:
                    out.linked[rows] |= any_pass
            else:
                if acc is not None:
                    cnt, hit = acc.result(self.gt.src_code[rows])
                out.tie_cnt[rows] += cnt
                out.tie_hit[rows] |= hit

    def _exact_fallback(self, deg_src_rows: np.ndarray, deg_cand_rows: np.ndarray, fb: dict) -> None:
        """Pairs whose projected vectors are both zero: link iff the *encoded feature
        vectors* are exactly equal (the projection destroyed the evidence, not the match)."""
        xs = self.projection.phi_rows(self.cols_s, deg_src_rows)
        xr = self.projection.phi_rows(self.cols_r, deg_cand_rows)
        joint = pd.factorize(
            np.array([r.tobytes() for r in np.ascontiguousarray(np.vstack([xs, xr]))], dtype=object)
        )[0]
        ks, kr = joint[: len(xs)], joint[len(xs):]
        by_key: Dict[int, List[int]] = {}
        for j, k in enumerate(kr):
            by_key.setdefault(int(k), []).append(int(deg_cand_rows[j]))
        for i, k in enumerate(ks):
            matches = by_key.get(int(k))
            if matches:
                fb.setdefault(int(deg_src_rows[i]), []).extend(matches)


def _lookup(sorted_or_not_cols: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Position of each target release row inside ``cols`` (-1 if absent)."""
    order = np.argsort(sorted_or_not_cols, kind="stable")
    srt = sorted_or_not_cols[order]
    pos = np.searchsorted(srt, targets)
    pos = np.clip(pos, 0, len(srt) - 1)
    found = (srt[pos] == targets) & (targets >= 0)
    return np.where(found, order[pos], -1)


class _SubjectAcc:
    """Exact distinct-subject counts per source row when subjects repeat in the release."""

    def __init__(self, n_rows: int, gt: PreparedGroundTruth) -> None:
        self.n = n_rows
        self.s = max(1, gt.n_subjects)
        self.keys = np.empty(0, dtype=np.int64)

    def add(self, r_idx: np.ndarray, subj: np.ndarray) -> None:
        k = r_idx.astype(np.int64) * self.s + subj.astype(np.int64)
        self.keys = np.unique(np.concatenate([self.keys, k]))

    def result(self, true_codes: np.ndarray):
        cnt = np.bincount(self.keys // self.s, minlength=self.n).astype(np.int64)
        hit = np.zeros(self.n, dtype=bool)
        if len(self.keys):
            rows = self.keys // self.s
            subj = self.keys % self.s
            m = subj == true_codes[rows]
            hit[rows[m]] = True
        return cnt, hit


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class CVPLResult:
    """Typed, identifier-free result; ``metrics`` is JSON-serialisable."""

    metrics: Dict[str, Any]
    projection: Optional[CVPLProjection] = field(default=None, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]

    def to_dict(self) -> Dict[str, Any]:
        return self.metrics


def _status(value: Optional[float], status: str, reason: Optional[str] = None, **extra) -> Dict[str, Any]:
    d = {"value": value, "status": status, "reason": reason}
    d.update(extra)
    return d


def _summary(n_cand: np.ndarray) -> Dict[str, Any]:
    if len(n_cand) == 0:
        return {"min": None, "median": None, "p95": None, "max": None, "zero_fraction": None}
    return {
        "min": int(n_cand.min()),
        "median": float(np.median(n_cand)),
        "p95": float(np.percentile(n_cand, 95)),
        "max": int(n_cand.max()),
        "zero_fraction": float((n_cand == 0).mean()),
    }


def _rung_measures(scan: RungScan, n: int, gt: Optional[PreparedGroundTruth], gt_ok_full: bool) -> Dict[str, Any]:
    linked = scan.linked
    n_linked = int(linked.sum())
    n_undet = int((scan.undefined & ~linked).sum())
    m: Dict[str, Any] = {
        "cvpl_lr": n_linked / n if n else None,
        "cvpl_lr_lower": n_linked / n if n else None,
        "cvpl_lr_upper": (n_linked + n_undet) / n if n else None,
        "n_linked": n_linked,
        "n_undetermined": n_undet,
        "n_exact_fallback": int(scan.fallback_only.sum()),
        "n_exact_fallback_records": int(scan.fallback_any.sum()),
        "n_exact_fallback_pairs": int(scan.fallback_pairs),
        "n_with_candidates": int((scan.n_cand > 0).sum()),
        "n_compared": int(scan.pairs_compared),
        "candidate_count_summary": _summary(scan.n_cand),
    }
    elig = scan.n_cand > 0
    if elig.any():
        w = 1.0 / scan.n_cand[elig]
        m["cvpl_weighted_lr_legacy"] = float((w * linked[elig]).sum() / w.sum())
        m["legacy_n_eligible"] = int(elig.sum())
    else:
        m["cvpl_weighted_lr_legacy"] = None
        m["legacy_n_eligible"] = 0
    if gt is not None and scan.cnt is not None:
        known = gt.known_mask
        cnt, hit = scan.cnt, scan.hit
        p = np.where(cnt > 0, hit / np.maximum(cnt, 1), 0.0)
        tcnt, thit = scan.tie_cnt, scan.tie_hit
        s = np.where(linked & (tcnt > 0), thit / np.maximum(tcnt, 1), 0.0)
        denom_known = int(known.sum())
        m["_gt"] = {
            "uniform_full": float(p.sum() / n) if n else None,
            "top1_full": float(s.sum() / n) if n else None,
            "uniform_known": float(p[known].sum() / denom_known) if denom_known else None,
            "top1_known": float(s[known].sum() / denom_known) if denom_known else None,
            "n_declared_full": n_linked,
            "success_declared_full": float(s[linked].sum()),
            "n_declared_known": int((linked & known).sum()),
            "success_declared_known": float(s[linked & known].sum()),
            "n_known": denom_known,
        }
    return m


def _gt_fields(sel: Dict[str, Any], gt: Optional[PreparedGroundTruth], n: int, status_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The three GT measures with their statuses (never a silent zero)."""
    out: Dict[str, Any] = {}
    g = sel.get("_gt")
    if gt is None or gt.status in ("absent", "invalid") or g is None:
        reason = "no_ground_truth" if gt is None or gt.status == "absent" else gt.reason
        why = reason if reason else "ground_truth_unavailable"
        for key in ("cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau"):
            out[key] = None
            out[f"{key}_status"] = _status(None, "not_measured", why)
        return out
    if gt.status == "complete":
        out["cvpl_id_uniform_gt"] = g["uniform_full"]
        out["cvpl_id_uniform_gt_status"] = _status(g["uniform_full"], "measured", denominator=n)
        out["cvpl_top1_success_gt"] = g["top1_full"]
        out["cvpl_top1_success_gt_status"] = _status(g["top1_full"], "measured", denominator=n)
        if g["n_declared_full"] > 0:
            val = g["success_declared_full"] / g["n_declared_full"]
            out["precision_at_tau"] = val
            out["precision_at_tau_status"] = _status(val, "measured", denominator=g["n_declared_full"])
        else:
            out["precision_at_tau"] = None
            out["precision_at_tau_status"] = _status(None, "not_measured", "no_declared_links", denominator=0)
        return out
    # partial: population measures are NOT available; the known subset is reported separately
    why = "ground_truth_incomplete:" + str(gt.reason)
    for key in ("cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau"):
        out[key] = None
    out["cvpl_id_uniform_gt_status"] = _status(
        None, "not_measured", why, known_subset_value=g["uniform_known"], known_subset_denominator=g["n_known"])
    out["cvpl_top1_success_gt_status"] = _status(
        None, "not_measured", why, known_subset_value=g["top1_known"], known_subset_denominator=g["n_known"])
    val = (g["success_declared_known"] / g["n_declared_known"]) if g["n_declared_known"] else None
    out["precision_at_tau_status"] = _status(
        None, "not_measured", why, known_subset_value=val, known_subset_denominator=g["n_declared_known"])
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _validate_inputs(cfg: CVPLConfig, source: pd.DataFrame, release: pd.DataFrame) -> None:
    if not isinstance(source, pd.DataFrame) or not isinstance(release, pd.DataFrame):
        raise InvalidParameterError("data", type(source).__name__, "source and release must be pandas DataFrames")
    for side, df, extra in (
        ("source", source, [cfg.ground_truth.source_id_column] if cfg.ground_truth else []),
        ("release", release, [cfg.ground_truth.release_id_column] if cfg.ground_truth else []),
    ):
        needed = list(cfg.qid_names) + list(cfg.auxiliary) + extra
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ColumnNotFoundError(column_name=missing, available_columns=list(df.columns), dataset_name=side)
        if not df.columns.is_unique:
            raise InvalidParameterError(f"{side}.columns", "duplicated", "duplicate column names are ambiguous")


def run_cvpl(
    source: pd.DataFrame,
    release: pd.DataFrame,
    config: CVPLConfig,
    frozen_projection: Optional[CVPLProjection] = None,
) -> CVPLResult:
    """Measure CVPL linkage of ``release`` against ``source`` (direction source -> release).

    The two frames keep their own, unrelated indices; the source row count ``N`` is the
    denominator of every population measure, whatever the release contains.
    """
    cfg = config
    t0 = time.perf_counter()
    _validate_inputs(cfg, source, release)
    n, m_rel = len(source), len(release)
    base: Dict[str, Any] = {
        "algorithm_id": ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "direction": cfg.direction,
        "linkage_unit": cfg.linkage_unit,
        "seed": cfg.seed,
        "n_original": n,
        "n_released": m_rel,
        "parameters": cfg.describe(),
        "metric_semantics": {
            "cvpl_lr": "existential_candidate_exposure",
            "cvpl_id_uniform_gt": "uniform_choice_correct_identity_success",
            "cvpl_top1_success_gt": "similarity_top1_correct_identity_success",
            "precision_at_tau": "correct_share_among_declared_top1_links",
            "cvpl_weighted_lr_legacy": "weighted_existential_linkage_v1",
        },
        "interpretation": (
            "Measurements of one specified attacker (declared attributes, blocking, projection, "
            "threshold). Not a probability of re-identification in general and not a verdict "
            "on anonymity."
        ),
    }
    if n == 0:
        base.update(_all_not_measured("empty_source"))
        return CVPLResult(base)

    # -- encodings ---------------------------------------------------------------
    encs: Dict[str, QIDEncoding] = {}
    for spec in cfg.qids:
        kind = resolve_kind(spec, source[spec.name])
        if kind == "categorical" and spec.generalized and not spec.hierarchy:
            raise InvalidParameterError(
                f"qids[{spec.name}].generalized", True, "a generalized categorical QID needs a hierarchy")
        encs[spec.name] = encode_qid(spec, source[spec.name], release[spec.name], cfg)
    base["qid_status"] = {
        q: {"kind": e.kind, "applicable": e.applicable, "limitation": e.limitation, **e.stats}
        for q, e in encs.items()
    }

    # -- phi / psi ---------------------------------------------------------------
    cols_s: FeatureColumns = {}
    cols_r: FeatureColumns = {}
    for name in cfg.feature_columns:
        if name in encs:
            e = encs[name]
            if e.kind == "numeric":
                cols_s[name] = ("numeric", e.mid_s)
                cols_r[name] = ("numeric", e.mid_r)
                continue
            cols_s[name] = ("categorical", categorical_values(source[name]))
            cols_r[name] = ("categorical", categorical_values(release[name]))
            continue
        if pd.api.types.is_numeric_dtype(source[name]) and not pd.api.types.is_bool_dtype(source[name]):
            cols_s[name] = ("numeric", numeric_values(source[name]))
            cols_r[name] = ("numeric", numeric_values(release[name]))
        else:
            cols_s[name] = ("categorical", categorical_values(source[name]))
            cols_r[name] = ("categorical", categorical_values(release[name]))
    if cfg.fit_strategy == "frozen":
        if frozen_projection is None:
            raise InvalidParameterError("frozen_projection", None, "fit_strategy='frozen' needs a fitted projection")
        proj = frozen_projection
        if set(proj.kinds) != set(cols_s) or any(proj.kinds[k] != cols_s[k][0] for k in cols_s):
            raise InvalidParameterError(
                "frozen_projection", sorted(proj.kinds), "feature columns/kinds differ from the configuration")
    else:
        proj = fit_projection(cfg, cols_s, n, cols_r, m_rel)
    info = proj.info
    base["projection"] = {
        "fit_strategy": cfg.fit_strategy, "n_features": info.n_features, "n_components": info.n_components,
        "variance_retained": info.variance_retained, "n_fit_rows": info.n_fit_rows,
        "feature_columns": info.feature_columns, "encoding": info.encoding,
        "imputed_values": info.imputed_values, "degenerate": info.degenerate,
    }
    zs = proj.project(cols_s, n)
    zr = proj.project(cols_r, m_rel)
    tol = cfg.zero_norm_tolerance
    ns = np.linalg.norm(zs, axis=1)
    nr = np.linalg.norm(zr, axis=1)
    deg_s, deg_r = ns <= tol, nr <= tol
    zs = np.where(deg_s[:, None], 0.0, zs / np.where(deg_s, 1.0, ns)[:, None])
    zr = np.where(deg_r[:, None], 0.0, zr / np.where(deg_r, 1.0, nr)[:, None])
    base["degenerate_vectors"] = {"source": int(deg_s.sum()), "release": int(deg_r.sum())}

    # -- ground truth ----------------------------------------------------------------
    gt = prepare_ground_truth(cfg, source, release)
    base["ground_truth"] = {
        "status": gt.status, "reason": gt.reason, "scope_declared": None if cfg.ground_truth is None else cfg.ground_truth.scope,
        "gt_coverage": (gt.n_known / n) if cfg.ground_truth is not None and gt.status != "invalid" else None,
        "n_known": gt.n_known if cfg.ground_truth is not None else 0,
        "release_subject_duplicates": (not gt.rel_unique) if gt.rel_code is not None else None,
    }
    base["gt_coverage"] = base["ground_truth"]["gt_coverage"]
    use_gt = gt.status in ("complete", "partial")

    scanner = _Scanner(cfg, zs, zr, deg_s, deg_r, gt if use_gt else None, proj, cols_s, cols_r)

    # -- ladder ----------------------------------------------------------------------
    ladder = cfg.ladder
    rungs: List[Dict[str, Any]] = []
    cumulative = np.zeros(n, dtype=bool)
    cum_with_cand = np.zeros(n, dtype=bool)
    last_measures: Optional[Dict[str, Any]] = None
    last_status: Dict[str, Any] = {}
    transitions: List[str] = []
    for r, qids in enumerate(ladder):
        entry: Dict[str, Any] = {"rung": r, "qids": list(qids)}
        if r:
            tr = classify_transition(ladder[r - 1], qids)
            entry["transition"] = tr
            transitions.append(tr)
        try:
            plan = plan_rung(qids, encs, n, m_rel, cfg)
        except ResourceError as exc:
            entry.update(status="not_measured", reason="resource_limit", detail=str(exc))
            rungs.append(entry)
            last_measures, last_status = None, {"status": "not_measured", "reason": "resource_limit"}
            continue
        entry["n_blocks"] = plan.n_blocks
        entry["planned_comparisons"] = int(plan.comparisons)
        if not plan.applicable:
            entry.update(status="not_applicable", reason=plan.reason)
            rungs.append(entry)
            last_measures, last_status = None, {"status": "not_applicable", "reason": plan.reason}
            continue
        if plan.comparisons > cfg.max_comparisons:
            entry.update(
                status="not_measured", reason="resource_limit",
                detail=f"{plan.comparisons:,} comparisons planned, limit {cfg.max_comparisons:,}",
            )
            rungs.append(entry)
            last_measures, last_status = None, {"status": "not_measured", "reason": "resource_limit"}
            continue
        scan = scanner.scan(plan, with_gt=use_gt)
        meas = _rung_measures(scan, n, gt if use_gt else None, gt.status == "complete")
        cumulative |= scan.linked
        cum_with_cand |= scan.n_cand > 0
        entry.update(status="measured", reason=None, cumulative_lr=float(cumulative.sum() / n))
        public = {k: v for k, v in meas.items() if k != "_gt"}
        entry["measures"] = public
        if use_gt:
            entry["measures"].update({k: v for k, v in _gt_fields(meas, gt, n, {}).items() if not k.endswith("_status")})
        rungs.append(entry)
        last_measures, last_status = meas, {"status": "measured", "reason": None}

    nested = all(t in ("relaxing", "equal") for t in transitions)
    tightening_only = bool(transitions) and all(t in ("tightening", "equal") for t in transitions)
    base["blocking_ladder_trajectory"] = rungs
    base["ladder"] = {
        "n_rungs": len(ladder),
        "transitions": transitions,
        "nested": nested,
        "monotone_lr_theorem_applies": bool(nested and len(ladder) > 1),
        "note": (
            "cvpl_lr cannot decrease along a nested relaxing ladder (fixed phi, psi, tau). "
            "cvpl_id_uniform_gt has no monotonicity guarantee."
            if nested else
            ("rungs are not ordered by inclusion (non_nested or tightening): see candidate_union_trajectory"
             if len(ladder) > 1 else "single rung")
        ),
    }
    base["candidate_union_trajectory"] = [
        {"rung": e["rung"], "cvpl_lr_union": e.get("cumulative_lr"), "qids": e["qids"]}
        for e in rungs
    ]
    base["cvpl_lr_union"] = float(cumulative.sum() / n) if any(e["status"] == "measured" for e in rungs) else None
    base["n_with_candidates_union"] = int(cum_with_cand.sum())

    # -- headline = the last declared rung --------------------------------------------
    base["headline_rung"] = len(ladder) - 1
    if last_measures is None:
        base.update(_all_not_measured(last_status.get("reason") or "not_measured",
                                      status=last_status.get("status", "not_measured")))
    else:
        head = {k: v for k, v in last_measures.items() if k != "_gt"}
        base.update(head)
        proj_degenerate = info.degenerate or (deg_s.all() and deg_r.all())
        if proj_degenerate and head["n_linked"] == 0:
            base.update(_all_not_measured("projection_degenerate"))
        else:
            base["cvpl_lr_status"] = _status(head["cvpl_lr"], "measured", denominator=n,
                                             lower=head["cvpl_lr_lower"], upper=head["cvpl_lr_upper"])
            base.update(_gt_fields(last_measures, gt, n, {}))
        base["undefined_policy"] = (
            "a source record whose compatible candidates include a pair with undefined cosine "
            "(zero projected vector) and that has no link is counted in n_undetermined; it stays in the "
            "denominator N, contributes 0 to the point value cvpl_lr and 1 to cvpl_lr_upper"
        )
    base["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    return CVPLResult(base, proj)


def _all_not_measured(reason: str, status: str = "not_measured") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("cvpl_lr", "cvpl_id_uniform_gt", "cvpl_top1_success_gt", "precision_at_tau",
                "cvpl_weighted_lr_legacy"):
        out[key] = None
        out[f"{key}_status"] = _status(None, status, reason)
    return out
