"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL measurement engine
Provenance:    method after DGT-Network/CVPL @bb25aa9 (MIT); implemented independently for CORE
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
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # scikit-learn depends on it; guard anyway so the metric never needs it to import
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

from pamola_core.errors.exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    ResourceError,
)

from .compat import (
    QIDEncoding,
    _group_compat,
    classify_transition,
    encode_qid,
    limitation_message,
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


class _SubjectAcc:
    """Exact distinct-subject counts per source row when subjects repeat in the release.

    Keys are ``source_row * n_subjects + subject``; they are merged and de-duplicated as
    they arrive, so a (row, subject) pair seen through two released rows - or through two
    ladder rungs - counts once.
    """

    _FLUSH = 4_000_000

    def __init__(self, n_rows: int, gt: PreparedGroundTruth) -> None:
        self.n = n_rows
        self.s = max(1, gt.n_subjects)
        self.keys = np.empty(0, dtype=np.int64)
        self._pending: List[np.ndarray] = []
        self._pending_len = 0

    def add(self, rows: np.ndarray, subj: np.ndarray) -> None:
        k = rows.astype(np.int64) * self.s + subj.astype(np.int64)
        self._pending.append(k)
        self._pending_len += len(k)
        if self._pending_len >= self._FLUSH:
            self._flush()

    def _flush(self) -> None:
        if self._pending:
            self.keys = np.unique(np.concatenate([self.keys] + self._pending))
            self._pending, self._pending_len = [], 0

    def result(self, true_codes: np.ndarray):
        self._flush()
        cnt = np.bincount(self.keys // self.s, minlength=self.n).astype(np.int64)
        hit = np.zeros(self.n, dtype=bool)
        if len(self.keys):
            rows = self.keys // self.s
            subj = self.keys % self.s
            hit[rows[subj == true_codes[rows]]] = True
        return cnt, hit


class _Scanner:
    """Blocked similarity scan. ``scan`` evaluates one rung (or, with ``cover``, the part of
    a rung not already explored by earlier rungs, so that several rungs together are
    evaluated over the UNION of their candidate sets with every pair counted once)."""

    def __init__(
        self, cfg: CVPLConfig, zs: np.ndarray, zr: np.ndarray, deg_s: np.ndarray, deg_r: np.ndarray,
        gt: Optional[PreparedGroundTruth], projection: CVPLProjection,
        cols_s: FeatureColumns, cols_r: FeatureColumns, encs: Dict[str, QIDEncoding],
    ) -> None:
        self.cfg, self.zs, self.zr = cfg, zs, zr
        self.deg_s, self.deg_r = deg_s, deg_r
        self.gt, self.projection, self.cols_s, self.cols_r = gt, projection, cols_s, cols_r
        self.encs = encs
        self._acc: Optional[_SubjectAcc] = None

    def scan(self, plans: Sequence[Tuple[Any, Sequence[Sequence[str]]]], with_gt: bool) -> RungScan:
        """``plans``: ``[(RungPlan, qid_lists_of_earlier_rungs_to_exclude), ...]``."""
        # The blocks are many and the matrices thin (k <= 50 columns): a multi-threaded BLAS
        # spends more time coordinating threads than multiplying (measured 10-20x slower on
        # a shared 12-core host). One BLAS thread per scan; the work is memory-bound anyway.
        limit = threadpool_limits(limits=1, user_api="blas") if threadpool_limits else nullcontext()
        with limit:
            return self._scan(plans, with_gt)

    def _scan(self, plans, with_gt: bool) -> RungScan:
        n = len(self.zs)
        gt = self.gt
        out = RungScan(
            n_cand=np.zeros(n, dtype=np.int64), linked=np.zeros(n, dtype=bool),
            best=np.full(n, -np.inf), undefined=np.zeros(n, dtype=bool),
            fallback_any=np.zeros(n, dtype=bool), fallback_only=np.zeros(n, dtype=bool),
        )
        use_acc = with_gt and not gt.rel_unique
        if with_gt:
            out.cnt = np.zeros(n, dtype=np.int64)
            out.hit = np.zeros(n, dtype=bool)
        fb: Dict[int, set] = {}  # source row -> released rows matched by the exact fallback
        self._acc = _SubjectAcc(n, gt) if use_acc else None
        for plan, cover in plans:
            for src_rows, cand_rows in plan.blocks():
                self._count_candidates(src_rows, cand_rows, cover, out)
                self._block(src_rows, cand_rows, cover, out, mode="link", fb=fb)
        for row, cands in fb.items():
            out.fallback_any[row] = True
            out.fallback_pairs += len(cands)
            if not out.linked[row]:
                out.fallback_only[row] = True
            out.linked[row] = True
            out.best[row] = max(out.best[row], 1.0)
            if with_gt:
                self._add_fallback(row, np.fromiter(cands, dtype=np.int64), out.cnt, out.hit)
        if use_acc:
            out.cnt, out.hit = self._acc.result(gt.src_code)
            out.linked |= out.cnt > 0
        out.undefined &= ~out.linked  # an undefined pair matters only if no link was found
        if with_gt:
            out.tie_cnt = np.zeros(n, dtype=np.int64)
            out.tie_hit = np.zeros(n, dtype=bool)
            self._acc = _SubjectAcc(n, gt) if use_acc else None
            for plan, cover in plans:
                for src_rows, cand_rows in plan.blocks():
                    self._block(src_rows, cand_rows, cover, out, mode="top1", fb=None)
            for row, cands in fb.items():
                # a degenerate source's tied top candidates are exactly its exact-equal rows
                self._add_fallback(row, np.fromiter(cands, dtype=np.int64), out.tie_cnt, out.tie_hit)
            if use_acc:
                out.tie_cnt, out.tie_hit = self._acc.result(gt.src_code)
        return out

    # -- helpers -----------------------------------------------------------------
    def _add_fallback(self, row: int, cand_rows: np.ndarray, cnt: np.ndarray, hit: np.ndarray) -> None:
        gt = self.gt
        if self._acc is not None:
            self._acc.add(np.full(len(cand_rows), row), gt.rel_code[cand_rows])
            return
        subj = np.unique(gt.rel_code[cand_rows])
        cnt[row] += len(subj)
        if gt.src_code[row] >= 0 and np.any(subj == gt.src_code[row]):
            hit[row] = True

    def _covered(self, rows: np.ndarray, cols: np.ndarray, cover: Sequence[Sequence[str]]) -> np.ndarray:
        """Pairs already compatible under an earlier rung (any of them)."""
        covered = np.zeros((len(rows), len(cols)), dtype=bool)
        for qids in cover:
            m = np.ones_like(covered)
            for q in qids:
                m &= _group_compat(self.encs[q], rows, cols, self.cfg.null_policy)
            covered |= m
        return covered

    def _count_candidates(self, src_rows, cand_rows, cover, out: RungScan) -> None:
        if not cover:
            out.n_cand[src_rows] += len(cand_rows)
            return
        width = int(min(len(cand_rows), 8192))
        step = max(1, self.cfg.chunk_pairs // width)
        for a in range(0, len(src_rows), step):
            rows = src_rows[a: a + step]
            for b in range(0, len(cand_rows), width):
                cc = cand_rows[b: b + width]
                out.n_cand[rows] += (~self._covered(rows, cc, cover)).sum(axis=1)

    def _block(self, src_rows, cand_rows, cover, out: RungScan, mode: str, fb: Optional[dict]) -> None:
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
        gt = self.gt
        with_gt = gt is not None and out.cnt is not None
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
            best = np.full(len(rows), -np.inf)
            any_pass = np.zeros(len(rows), dtype=bool)
            cnt = np.zeros(len(rows), dtype=np.int64)
            hit = np.zeros(len(rows), dtype=bool)
            for b in range(0, len(vc), width):
                cc = vc[b: b + width]
                sim = zq @ self.zr[cc].T
                if cover:
                    covered = self._covered(rows, cc, cover)
                    sim[covered] = -2.0  # explored by an earlier rung: not a new pair
                    n_new = sim.size - int(covered.sum())
                else:
                    n_new = sim.size
                if mode == "link":
                    out.pairs_compared += n_new
                    best = np.maximum(best, sim.max(axis=1))
                pas = sim >= thr[:, None]
                if not with_gt:
                    any_pass |= pas.any(axis=1)
                    continue
                if self._acc is None:  # every released row is its own subject
                    cnt += pas.sum(axis=1)
                    tr = gt.true_row[rows]
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
                        self._acc.add(rows[r_idx], gt.rel_code[cc[c_idx]])
            if mode == "link":
                out.best[rows] = np.maximum(out.best[rows], best)
                if with_gt:
                    if self._acc is None:
                        out.cnt[rows] += cnt
                        out.hit[rows] |= hit
                        out.linked[rows] |= cnt > 0
                else:
                    out.linked[rows] |= any_pass
            elif self._acc is None:
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
                fb.setdefault(int(deg_src_rows[i]), set()).update(matches)


def _lookup(sorted_or_not_cols: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Position of each target release row inside ``cols`` (-1 if absent)."""
    order = np.argsort(sorted_or_not_cols, kind="stable")
    srt = sorted_or_not_cols[order]
    pos = np.searchsorted(srt, targets)
    pos = np.clip(pos, 0, len(srt) - 1)
    found = (srt[pos] == targets) & (targets >= 0)
    return np.where(found, order[pos], -1)


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
    force_union: bool = False,
) -> CVPLResult:
    """Measure CVPL linkage of ``release`` against ``source`` (direction source -> release).

    The two frames keep their own, unrelated indices; the source row count ``N`` is the
    denominator of every population measure, whatever the release contains.

    ``force_union`` evaluates the headline over the union of all rungs' candidate sets even
    for a nested ladder (where it must equal the last rung, Theorem 1); it exists to verify
    that equality.
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
        q: {"kind": e.kind, "applicable": e.applicable, "limitation": e.limitation,
            "fix": None if e.applicable else limitation_message(q, e.limitation), **e.stats}
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
    # bucketing is never silent: which attributes were collapsed, and by how much
    bucketed = [
        {"feature": name, "distinct_values": e["distinct_values"], "encoded_width": e["encoded_width"],
         "distinct_values_collapsed": e["distinct_values_in_other_bucket"],
         "rows_in_other_bucket": e["values_in_other_bucket"]}
        for name, e in info.encoding.items() if e["bounded"]
    ]
    base["bucketed_features"] = bucketed
    base["warnings"] = [
        f"feature '{b['feature']}': {b['distinct_values']} distinct values encoded in {b['encoded_width']} "
        f"columns; {b['distinct_values_collapsed']} rare values ({b['rows_in_other_bucket']} rows) share one "
        "bucket (raise max_categories for finer detail)"
        for b in bucketed
    ]
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

    scanner = _Scanner(cfg, zs, zr, deg_s, deg_r, gt if use_gt else None, proj, cols_s, cols_r, encs)

    # -- ladder ----------------------------------------------------------------------
    ladder = cfg.ladder
    rungs: List[Dict[str, Any]] = []
    plans: List[Any] = []
    rung_meas: List[Optional[Dict[str, Any]]] = []
    rung_state: List[Dict[str, Any]] = []
    cumulative = np.zeros(n, dtype=bool)
    cum_with_cand = np.zeros(n, dtype=bool)
    transitions: List[str] = []
    for r, qids in enumerate(ladder):
        entry: Dict[str, Any] = {"rung": r, "qids": list(qids)}
        if r:
            tr = classify_transition(ladder[r - 1], qids)
            entry["transition"] = tr
            transitions.append(tr)
        plan = None
        meas = None
        state = {"status": "not_measured", "reason": None}
        try:
            plan = plan_rung(qids, encs, n, m_rel, cfg)
        except ResourceError as exc:
            entry.update(status="not_measured", reason="resource_limit", detail=str(exc))
            state = {"status": "not_measured", "reason": "resource_limit"}
        else:
            entry["n_blocks"] = plan.n_blocks
            entry["planned_comparisons"] = int(plan.comparisons)
            if not plan.applicable:
                entry.update(status="not_applicable", reason=plan.reason)
                state = {"status": "not_applicable", "reason": plan.reason}
            elif plan.comparisons > cfg.max_comparisons:
                entry.update(
                    status="not_measured", reason="resource_limit",
                    detail=f"{plan.comparisons:,} comparisons planned, limit {cfg.max_comparisons:,}",
                )
                state = {"status": "not_measured", "reason": "resource_limit"}
            else:
                scan = scanner.scan([(plan, [])], use_gt)
                meas = _rung_measures(scan, n, gt if use_gt else None, gt.status == "complete")
                cumulative |= scan.linked
                cum_with_cand |= scan.n_cand > 0
                entry.update(status="measured", reason=None, cumulative_lr=float(cumulative.sum() / n))
                entry["measures"] = {k: v for k, v in meas.items() if k != "_gt"}
                if use_gt:
                    entry["measures"].update(
                        {k: v for k, v in _gt_fields(meas, gt, n, {}).items() if not k.endswith("_status")}
                    )
                state = {"status": "measured", "reason": None}
        rungs.append(entry)
        plans.append(plan)
        rung_meas.append(meas)
        rung_state.append(state)

    nested = all(t in ("relaxing", "equal") for t in transitions)
    base["blocking_ladder_trajectory"] = rungs
    base["cvpl_lr_by_rung"] = [
        {"rung": e["rung"], "qids": e["qids"], "status": e["status"],
         "cvpl_lr": e.get("measures", {}).get("cvpl_lr")}
        for e in rungs
    ]
    base["candidate_union_trajectory"] = [
        {"rung": e["rung"], "cvpl_lr_union": e.get("cumulative_lr"), "qids": e["qids"]} for e in rungs
    ]
    base["cvpl_lr_union"] = float(cumulative.sum() / n) if any(e["status"] == "measured" for e in rungs) else None
    base["n_with_candidates_union"] = int(cum_with_cand.sum())

    # -- headline -------------------------------------------------------------------------
    # One rung, or a nested relaxing ladder: the last rung IS the union of the candidate sets
    # (Theorem 1's premise). Otherwise the attacker who tries every declared profile is
    # evaluated over the union of their candidate sets - every pair counted once.
    union_basis = (len(ladder) > 1 and not nested) or force_union
    base["headline_basis"] = "union_of_rungs" if union_basis else "last_rung"
    base["headline_rung"] = None if union_basis else len(ladder) - 1
    base["ladder"] = {
        "n_rungs": len(ladder),
        "transitions": transitions,
        "nested": nested,
        "non_nested": bool(len(ladder) > 1 and not nested),
        "monotone_lr_theorem_applies": bool(nested and len(ladder) > 1),
        "note": (
            "nested relaxing ladder: cvpl_lr cannot decrease along it (fixed phi, psi, tau) and the "
            "last rung equals the union of all rungs; cvpl_id_uniform_gt has no monotonicity guarantee."
            if nested and len(ladder) > 1 else
            ("rungs are not ordered by inclusion: no monotonicity statement; the headline is the "
             "attacker trying every declared profile (union of candidate sets)"
             if len(ladder) > 1 else "single rung")
        ),
    }
    head_meas: Optional[Dict[str, Any]]
    head_state: Dict[str, Any]
    if not union_basis:
        head_meas, head_state = rung_meas[-1], rung_state[-1]
    else:
        bad = [st for st in rung_state if st["status"] != "measured"]
        total = sum(int(p.comparisons) for p in plans if p is not None)
        if bad:
            na = [st for st in bad if st["status"] == "not_applicable"]
            head_meas = None
            head_state = (
                {"status": "not_applicable", "reason": "union_needs_every_rung:" + str(na[0]["reason"])}
                if na else {"status": "not_measured", "reason": "resource_limit"}
            )
        elif total > cfg.max_comparisons:
            head_meas = None
            head_state = {"status": "not_measured", "reason": "resource_limit"}
            base["union_detail"] = f"{total:,} comparisons planned over all rungs, limit {cfg.max_comparisons:,}"
        else:
            cover = [(plans[r], [list(q) for q in ladder[:r]]) for r in range(len(ladder))]
            uscan = scanner.scan(cover, use_gt)
            head_meas = _rung_measures(uscan, n, gt if use_gt else None, gt.status == "complete")
            head_state = {"status": "measured", "reason": None}
    if head_meas is None:
        base.update(_all_not_measured(head_state.get("reason") or "not_measured",
                                      status=head_state.get("status", "not_measured")))
    else:
        head = {k: v for k, v in head_meas.items() if k != "_gt"}
        base.update(head)
        proj_degenerate = info.degenerate or (deg_s.all() and deg_r.all())
        if proj_degenerate and head["n_linked"] == 0:
            base.update(_all_not_measured("projection_degenerate"))
        else:
            base["cvpl_lr_status"] = _status(head["cvpl_lr"], "measured", denominator=n,
                                             lower=head["cvpl_lr_lower"], upper=head["cvpl_lr_upper"])
            base.update(_gt_fields(head_meas, gt, n, {}))
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
