"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL measurement engine
Provenance:    method after DGT-Network/CVPL @bb25aa9 (MIT); implemented independently for CORE
Package:       pamola_core.metrics.privacy.cvpl
License:       BSD 3-Clause

The CVPL operator chain ``B_lambda -> phi -> psi -> s -> tau`` and the quantities
measured with it. See ``docs`` ("CVPL linkage metric") for what each number means.

CVPL is a **white-box evaluation**: the auditor holds the source data, the release and
(for the GT measures) the ground truth, and chooses the attacker's strategy with full
access. Every published measure is therefore an UPPER, worst-case-oriented estimate of
what the declared attacker achieves, not an average over attackers and not a
Fellegi-Sunter style match probability.

For source record ``x_i``:

* ``C_i`` = admissible released rows (blocking-compatible) whose cosine similarity to
  ``x_i`` in the projected space is at least ``tau`` (an exact-equal pair of encoded
  vectors also counts when the projection is degenerate);
* ``cvpl_lr`` = share of the ``N`` source records with ``|C_i| > 0`` (existence of a
  plausible link; *not* a re-identification probability);
* with a complete ground truth ``T_i``: ``cvpl_id_uniform_subjects_gt`` (uniform choice
  among the distinct candidate SUBJECTS), ``cvpl_id_uniform_rows_gt`` (uniform choice
  among the candidate released ROWS), the headline ``cvpl_id_uniform_gt`` = the larger of
  the two population values, ``cvpl_top1_success_gt`` and ``precision_at_tau``.

A pair whose cosine is undefined (a zero projected vector that is not an exact-equal
pair) is *unknown*. Every measure that such a pair can influence is published with
bounds (status ``bounded``, point value ``null``), never as a measured point value.

Nothing here materialises a ``|D| x |D'|`` matrix: candidates are generated from blocks,
similarities are computed in bounded tiles, and every per-record accumulator lives only
for the chunk of source records being processed.
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
_EXACT_ID_LIMIT = 2**53
_TILE_WIDTH = 8192


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
    known_mask: Optional[np.ndarray] = None  # source rows whose correspondence is fully known
    true_row: Optional[np.ndarray] = None  # unique-release fast path: row of own subject
    n_subjects: int = 0
    anonymous_release: bool = False  # released rows without an identifier: membership unknown


class _IdentifierError(ValueError):
    """An identifier column that cannot be compared without guessing."""


def _canonical_ids(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, set]:
    """-> (null mask, keys of the non-null entries, namespaces present).

    Equality of identifiers is *value* equality inside one namespace: numbers compare
    as numbers (``1 == 1.0``), strings as strings, and the two never compare equal
    (``"1" != 1``). Booleans, non-finite numbers and any other type are rejected: a silent
    coercion would turn "same person" into "different person" (or the reverse).
    """
    null = series.isna().to_numpy()
    present = series[~null]
    if pd.api.types.is_bool_dtype(series):
        raise _IdentifierError("boolean identifiers are not supported")
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            arr = present.to_numpy()
            big = bool(arr.size) and (int(arr.max()) > _EXACT_ID_LIMIT or int(arr.min()) < -_EXACT_ID_LIMIT)
            keys = np.array([int(v) for v in arr.tolist()], dtype=object) if big else arr.astype(float)
        else:
            arr = present.to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                raise _IdentifierError("non-finite numeric identifiers are not supported")
            keys = arr + 0.0
        return null, keys, ({"n"} if len(keys) else set())
    keys = []
    spaces = set()
    for v in present.tolist():
        if isinstance(v, (bool, np.bool_)):
            raise _IdentifierError("boolean identifiers are not supported")
        if isinstance(v, (int, np.integer)):
            keys.append(int(v))
            spaces.add("n")
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if not np.isfinite(fv):
                raise _IdentifierError("non-finite numeric identifiers are not supported")
            keys.append(int(fv) if fv.is_integer() else fv)
            spaces.add("n")
        elif isinstance(v, str):
            keys.append(v)
            spaces.add("s")
        else:
            raise _IdentifierError(f"identifier type {type(v).__name__} is not supported")
    out = np.empty(len(keys), dtype=object)
    out[:] = keys
    return null, out, spaces


def _unify_keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concatenate two key arrays; floats that are integral become ints when objects are
    involved, so that 3 and 3.0 are one identifier."""
    if a.dtype != object and b.dtype != object:
        return np.concatenate([a, b])

    def to_obj(x: np.ndarray) -> np.ndarray:
        if x.dtype == object:
            return x
        out = np.empty(len(x), dtype=object)
        out[:] = [int(v) if float(v).is_integer() else float(v) for v in x.tolist()]
        return out

    return np.concatenate([to_obj(a), to_obj(b)])


def prepare_ground_truth(cfg: CVPLConfig, source: pd.DataFrame, release: pd.DataFrame) -> PreparedGroundTruth:
    n = len(source)
    spec = cfg.ground_truth
    if spec is None:
        return PreparedGroundTruth("absent", "no_ground_truth_declared", 0)
    src = source[spec.source_id_column]
    rel = release[spec.release_id_column]
    try:
        src_null, src_keys, src_spaces = _canonical_ids(src)
        rel_null, rel_keys, rel_spaces = _canonical_ids(rel)
    except _IdentifierError as exc:
        return PreparedGroundTruth("invalid", f"unsupported_identifier:{exc}", 0)
    n_known_ids = int((~src_null).sum())
    if n_known_ids == 0:
        return PreparedGroundTruth("absent", "ground_truth_has_no_source_identifiers", 0)
    if src_spaces and rel_spaces and not (src_spaces & rel_spaces):
        return PreparedGroundTruth(
            "invalid", "identifier_types_incompatible: the source identifiers are "
            f"{'numbers' if 'n' in src_spaces else 'text'} and the release identifiers are "
            f"{'numbers' if 'n' in rel_spaces else 'text'}; convert one side explicitly", 0)
    # one code space for both id columns
    both = _unify_keys(src_keys, rel_keys)
    codes, uniques = pd.factorize(both)
    n_src_keys = len(src_keys)
    src_code = np.full(n, -1, dtype=np.int64)
    src_code[~src_null] = codes[:n_src_keys]
    rel_code = np.empty(len(release), dtype=np.int64)
    rel_code[~rel_null] = codes[n_src_keys:]
    n_anon = int(rel_null.sum())
    rel_code[rel_null] = len(uniques) + np.arange(n_anon)  # each unidentified row is its own code
    known_src_codes = src_code[src_code >= 0]
    if len(np.unique(known_src_codes)) != len(known_src_codes):
        return PreparedGroundTruth(
            "invalid", "source_subject_identifiers_not_unique", n_known_ids,
        )
    rel_unique = len(np.unique(rel_code)) == len(rel_code)
    in_release = np.isin(src_code, rel_code) & (src_code >= 0)
    anonymous = n_anon > 0
    # A source record's correspondence set T_i is KNOWN only if every released row's
    # membership is known (all carry an identifier) and the record itself has one.
    # An anonymous released row might belong to anyone: nothing about any T_i is then known.
    if anonymous:
        known = np.zeros(n, dtype=bool)
    elif spec.scope == "partial":
        known = in_release  # absence from the release proves nothing when links are partly known
    else:
        known = src_code >= 0
    if spec.scope == "complete" and n_known_ids == n and not anonymous:
        status, reason = "complete", None
        known = np.ones(n, dtype=bool)
    else:
        status = "partial"
        if spec.scope == "partial":
            reason = "scope_declared_partial"
        elif anonymous:
            reason = "released_rows_without_identifier"
        else:
            reason = "source_records_without_identifier"
    true_row = None
    if rel_unique:
        row_of = np.full(len(uniques) + n_anon, -1, dtype=np.int64)
        row_of[rel_code] = np.arange(len(release))
        true_row = np.where(src_code >= 0, row_of[np.where(src_code >= 0, src_code, 0)], -1)
    return PreparedGroundTruth(
        status, reason, int(known.sum()), src_code, rel_code, rel_unique, known, true_row,
        n_subjects=len(uniques) + n_anon, anonymous_release=anonymous,
    )


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
@dataclass
class RungScan:
    """Per source record results of scanning one rung (or the union of several)."""

    n_cand: np.ndarray  # admissible candidate pairs (a pair is counted once in a union)
    linked: np.ndarray  # a KNOWN accepted pair exists (a witness)
    best: np.ndarray  # best defined similarity (1.0 for an exact-equal fallback)
    unk_rows: np.ndarray  # admissible pairs whose cosine is undefined and not exact-equal
    fallback_any: np.ndarray
    fallback_only: np.ndarray
    fallback_pairs: int = 0
    pairs_compared: int = 0
    # ground-truth arrays (None without ground truth)
    kcnt_s: Optional[np.ndarray] = None  # distinct accepted subjects
    khit_s: Optional[np.ndarray] = None  # own subject among them
    kcnt_r: Optional[np.ndarray] = None  # accepted released rows
    khit_r: Optional[np.ndarray] = None  # accepted rows of the own subject
    unew_s: Optional[np.ndarray] = None  # distinct unknown subjects not among the accepted ones
    uhit_new: Optional[np.ndarray] = None  # own subject among those
    uhit_rows: Optional[np.ndarray] = None  # unknown rows of the own subject
    tcnt_s: Optional[np.ndarray] = None  # distinct subjects tied for the top
    thit_s: Optional[np.ndarray] = None  # own subject among them


class _ChunkState:
    """Accumulators of ONE chunk of source records; discarded when the chunk is done."""

    def __init__(self, m: int, gt: Optional[PreparedGroundTruth], rows: np.ndarray) -> None:
        self.m = m
        self.n_cand = np.zeros(m, dtype=np.int64)
        self.best = np.full(m, -np.inf)
        self.kcnt = np.zeros(m, dtype=np.int64)  # accepted rows (known + fallback)
        self.kpass = np.zeros(m, dtype=np.int64)  # accepted by cosine
        self.fb = np.zeros(m, dtype=np.int64)  # accepted by exact equality
        self.urows = np.zeros(m, dtype=np.int64)
        self.compared = 0
        self.gt = gt
        if gt is not None:
            self.S = max(1, gt.n_subjects)
            self.own = gt.src_code[rows]
            self.khit = np.zeros(m, dtype=np.int64)
            self.uhit = np.zeros(m, dtype=np.int64)
            self.tcnt = np.zeros(m, dtype=np.int64)
            self.thit = np.zeros(m, dtype=np.int64)
            self.K: List[np.ndarray] = []
            self.U: List[np.ndarray] = []
            self.T: List[np.ndarray] = []


def _distinct(keys: List[np.ndarray], m: int, S: int, own: np.ndarray):
    """Distinct (record, subject) keys -> per-record subject count, own-subject flag, keys."""
    if not keys:
        return np.zeros(m, dtype=np.int64), np.zeros(m, dtype=bool), np.empty(0, dtype=np.int64)
    k = np.unique(np.concatenate(keys))
    loc = k // S
    subj = k % S
    cnt = np.bincount(loc, minlength=m).astype(np.int64)
    hit = np.zeros(m, dtype=bool)
    hit[loc[subj == own[loc]]] = True
    return cnt, hit, k


class _Scanner:
    """Blocked similarity scan, one chunk of source records at a time.

    ``scan`` evaluates one rung, or - with ``cover`` lists - the union of several rungs:
    a pair admissible under an earlier rung is not examined again, so every pair is
    counted once.
    """

    def __init__(
        self, cfg: CVPLConfig, zs: np.ndarray, zr: np.ndarray, deg_s: np.ndarray, deg_r: np.ndarray,
        gt: Optional[PreparedGroundTruth], cls_s: np.ndarray, cls_r: np.ndarray,
        encs: Dict[str, QIDEncoding],
    ) -> None:
        self.cfg, self.zs, self.zr = cfg, zs, zr
        self.deg_s, self.deg_r = deg_s, deg_r
        self.gt = gt
        self.cls_s, self.cls_r = cls_s, cls_r
        self.encs = encs

    def scan(self, plans: Sequence[Tuple[Any, Sequence[Sequence[str]]]], with_gt: bool) -> RungScan:
        """``plans``: ``[(RungPlan, qid_lists_of_earlier_rungs_to_exclude), ...]``."""
        # The blocks are many and the matrices thin (k <= 50 columns): a multi-threaded BLAS
        # spends more time coordinating threads than multiplying (measured 10-20x slower on
        # a shared 12-core host). One BLAS thread per scan; the work is memory-bound anyway.
        limit = threadpool_limits(limits=1, user_api="blas") if threadpool_limits else nullcontext()
        with limit:
            return self._scan(plans, with_gt and self.gt is not None)

    # -- driver -----------------------------------------------------------------
    def _scan(self, plans, with_gt: bool) -> RungScan:
        n = len(self.zs)
        out = RungScan(
            n_cand=np.zeros(n, dtype=np.int64), linked=np.zeros(n, dtype=bool),
            best=np.full(n, -np.inf), unk_rows=np.zeros(n, dtype=np.int64),
            fallback_any=np.zeros(n, dtype=bool), fallback_only=np.zeros(n, dtype=bool),
        )
        if with_gt:
            for name in ("kcnt_s", "kcnt_r", "unew_s", "uhit_rows", "tcnt_s", "khit_r"):
                setattr(out, name, np.zeros(n, dtype=np.int64))
            for name in ("khit_s", "uhit_new", "thit_s"):
                setattr(out, name, np.zeros(n, dtype=bool))
        cost = np.zeros(n, dtype=np.int64)
        for plan, _ in plans:
            cost += plan.row_ncand
        active = np.flatnonzero(cost > 0)
        if active.size == 0:
            return out
        driver = plans[0][0]
        order = active[np.lexsort((driver.row_grp[active], driver.row_part[active]))]
        # Chunks hold at most ~chunk_pairs planned pairs, so every accumulator below is bounded
        # by that, whatever N is. (One huge block gives one-record chunks.) Only the
        # distinct-subject keys grow with the pairs; when the release has one row per subject
        # (or there is no ground truth) the chunk state is O(records) and chunks can be larger.
        keys_needed = with_gt and not self.gt.rel_unique
        budget = int(self.cfg.chunk_pairs) * (1 if keys_needed else 256)
        bucket = (np.cumsum(np.maximum(cost[order], 1)) - 1) // budget
        cuts = np.flatnonzero(np.diff(bucket)) + 1
        for part in np.split(order, cuts):
            self._chunk(np.sort(part), plans, with_gt, out)
        return out

    def _chunk(self, rows: np.ndarray, plans, with_gt: bool, out: RungScan) -> None:
        gt = self.gt if with_gt else None
        st = _ChunkState(len(rows), gt, rows)
        cache: List[Tuple[np.ndarray, np.ndarray, Any]] = []
        for plan, cover in plans:
            pp = plan.row_part[rows]
            gg = plan.row_grp[rows]
            idx = np.flatnonzero(pp >= 0)
            if idx.size == 0:
                continue
            o = idx[np.lexsort((gg[idx], pp[idx]))]
            cut = np.flatnonzero((np.diff(pp[o]) != 0) | (np.diff(gg[o]) != 0)) + 1
            for seg in np.split(o, cut):
                cand = plan.candidates(int(pp[seg[0]]), int(gg[seg[0]]))
                if len(cand) == 0:
                    continue
                self._link_pairs(seg, rows[seg], cand, cover, st)
                cache.append((seg, cand, cover))
        linked = st.kcnt > 0
        if gt is not None and linked.any():
            for loc, cand, cover in cache:
                self._top_pairs(loc, rows[loc], cand, cover, st, linked)
        self._finalize(rows, st, out)

    # -- pair evaluation ----------------------------------------------------------
    def _covered(self, rows: np.ndarray, cols: np.ndarray, cover: Sequence[Sequence[str]]) -> np.ndarray:
        """Pairs already admissible under an earlier rung (any of them)."""
        covered = np.zeros((len(rows), len(cols)), dtype=bool)
        for qids in cover:
            m = np.ones_like(covered)
            for q in qids:
                m &= _group_compat(self.encs[q], rows, cols, self.cfg.null_policy)
            covered |= m
        return covered

    def _tiles(self, rows: np.ndarray, loc: np.ndarray, cands: np.ndarray, cover):
        width = int(min(len(cands), _TILE_WIDTH))
        step = max(1, int(self.cfg.chunk_pairs) // width)
        for a in range(0, len(rows), step):
            r, l = rows[a: a + step], loc[a: a + step]
            for b in range(0, len(cands), width):
                c = cands[b: b + width]
                valid = None if not cover else ~self._covered(r, c, cover)
                yield r, l, c, valid

    def _link_pairs(self, loc: np.ndarray, src_rows: np.ndarray, cand: np.ndarray, cover, st: _ChunkState) -> None:
        tau = self.cfg.tau - _TAU_EPS
        sdeg = self.deg_s[src_rows]
        cdeg = self.deg_r[cand]
        if not cover:
            st.n_cand[loc] += len(cand)
        nd_s, d_s = ~sdeg, sdeg
        nd_c, d_c = ~cdeg, cdeg
        # 1. both vectors defined: cosine
        if nd_s.any() and nd_c.any():
            for r, l, c, valid in self._tiles(src_rows[nd_s], loc[nd_s], cand[nd_c], cover):
                sim = self.zs[r] @ self.zr[c].T
                if valid is not None:
                    sim = np.where(valid, sim, -np.inf)
                    st.n_cand[l] += valid.sum(axis=1)
                    st.compared += int(valid.sum())
                else:
                    st.compared += sim.size
                st.best[l] = np.maximum(st.best[l], sim.max(axis=1))
                acc = sim >= tau
                self._accept(l, c, acc, st, cosine=True)
        # 2./3. exactly one vector undefined: the cosine does not exist
        for sel_s, sel_c in ((nd_s, d_c), (d_s, nd_c)):
            if sel_s.any() and sel_c.any():
                for r, l, c, valid in self._tiles(src_rows[sel_s], loc[sel_s], cand[sel_c], cover):
                    unk = np.ones((len(r), len(c)), dtype=bool) if valid is None else valid
                    if valid is not None:
                        st.n_cand[l] += valid.sum(axis=1)
                    self._unknown(l, c, unk, st)
        # 4. both undefined: linked iff the ENCODED vectors are exactly equal, else unknown
        if d_s.any() and d_c.any():
            for r, l, c, valid in self._tiles(src_rows[d_s], loc[d_s], cand[d_c], cover):
                eq = self.cls_s[r][:, None] == self.cls_r[c][None, :]
                ok = np.ones_like(eq) if valid is None else valid
                if valid is not None:
                    st.n_cand[l] += valid.sum(axis=1)
                hit = eq & ok
                if hit.any():
                    st.best[l] = np.maximum(st.best[l], np.where(hit.any(axis=1), 1.0, -np.inf))
                    self._accept(l, c, hit, st, cosine=False)
                unk = ~eq & ok
                if unk.any():
                    self._unknown(l, c, unk, st)

    def _own(self, st: _ChunkState, loc_t: np.ndarray, cands_t: np.ndarray) -> np.ndarray:
        return self.gt.rel_code[cands_t][None, :] == st.own[loc_t][:, None]

    def _accept(self, loc_t, cands_t, acc: np.ndarray, st: _ChunkState, cosine: bool) -> None:
        cnt = acc.sum(axis=1)
        st.kcnt[loc_t] += cnt
        if cosine:
            st.kpass[loc_t] += cnt
        else:
            st.fb[loc_t] += cnt
        if st.gt is None:
            return
        hit_rows = (acc & self._own(st, loc_t, cands_t)).sum(axis=1)
        st.khit[loc_t] += hit_rows
        keys = None
        if not st.gt.rel_unique:
            r_i, c_i = np.nonzero(acc)
            if r_i.size:
                keys = loc_t[r_i].astype(np.int64) * st.S + st.gt.rel_code[cands_t[c_i]]
                st.K.append(keys)
        if not cosine:
            # a degenerate source's tied top candidates are exactly its exact-equal rows
            st.tcnt[loc_t] += cnt
            st.thit[loc_t] += hit_rows
            if keys is not None:
                st.T.append(keys)

    def _unknown(self, loc_t, cands_t, unk: np.ndarray, st: _ChunkState) -> None:
        st.urows[loc_t] += unk.sum(axis=1)
        if st.gt is None:
            return
        st.uhit[loc_t] += (unk & self._own(st, loc_t, cands_t)).sum(axis=1)
        if not st.gt.rel_unique:
            r_i, c_i = np.nonzero(unk)
            if r_i.size:
                st.U.append(loc_t[r_i].astype(np.int64) * st.S + st.gt.rel_code[cands_t[c_i]])

    def _top_pairs(self, loc, src_rows, cand, cover, st: _ChunkState, linked: np.ndarray) -> None:
        """Second pass: the top-similarity ties among the ACCEPTED candidates only.

        A candidate is a tie only if it passes the threshold AND is within the tie tolerance
        of the best similarity; tolerance never admits a pair the threshold rejected.
        """
        eligible = ~self.deg_s[src_rows] & linked[loc]
        cand_nd = cand[~self.deg_r[cand]]
        if not eligible.any() or len(cand_nd) == 0:
            return
        tau = self.cfg.tau - _TAU_EPS
        for r, l, c, valid in self._tiles(src_rows[eligible], loc[eligible], cand_nd, cover):
            sim = self.zs[r] @ self.zr[c].T
            if valid is not None:
                sim = np.where(valid, sim, -np.inf)
            thr = np.maximum(st.best[l] - self.cfg.tie_tolerance, tau)
            acc = sim >= thr[:, None]
            if not acc.any():
                continue
            cnt = acc.sum(axis=1)
            st.tcnt[l] += cnt
            st.thit[l] += (acc & self._own(st, l, c)).sum(axis=1)
            if not st.gt.rel_unique:
                r_i, c_i = np.nonzero(acc)
                st.T.append(l[r_i].astype(np.int64) * st.S + st.gt.rel_code[c[c_i]])

    # -- end of chunk -----------------------------------------------------------------
    def _finalize(self, rows: np.ndarray, st: _ChunkState, out: RungScan) -> None:
        out.n_cand[rows] = st.n_cand
        out.linked[rows] = st.kcnt > 0
        out.best[rows] = st.best
        out.unk_rows[rows] = st.urows
        out.fallback_any[rows] = st.fb > 0
        out.fallback_only[rows] = (st.fb > 0) & (st.kpass == 0)
        out.fallback_pairs += int(st.fb.sum())
        out.pairs_compared += st.compared
        if st.gt is None:
            return
        m = st.m
        if st.gt.rel_unique:
            kcnt_s, khit_s = st.kcnt, st.khit > 0
            unew, uhit_new = st.urows, st.uhit > 0
            tcnt_s, thit_s = st.tcnt, st.thit > 0
        else:
            kcnt_s, khit_s, kk = _distinct(st.K, m, st.S, st.own)
            tcnt_s, thit_s, _ = _distinct(st.T, m, st.S, st.own)
            _, _, uk = _distinct(st.U, m, st.S, st.own)
            only_u = np.setdiff1d(uk, kk, assume_unique=True)
            loc = only_u // st.S
            unew = np.bincount(loc, minlength=m).astype(np.int64)
            uhit_new = np.zeros(m, dtype=bool)
            uhit_new[loc[(only_u % st.S) == st.own[loc]]] = True
        out.kcnt_s[rows] = kcnt_s
        out.khit_s[rows] = khit_s
        out.kcnt_r[rows] = st.kcnt
        out.khit_r[rows] = st.khit
        out.unew_s[rows] = unew
        out.uhit_new[rows] = uhit_new
        out.uhit_rows[rows] = st.uhit
        out.tcnt_s[rows] = tcnt_s
        out.thit_s[rows] = thit_s


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


GT_MEASURES = (
    "cvpl_id_uniform_subjects_gt", "cvpl_id_uniform_rows_gt", "cvpl_id_uniform_gt",
    "cvpl_top1_success_gt", "precision_at_tau",
)
_UNC_EPS = 1e-12


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


def _record_arrays(scan: RungScan) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per record: (point value if every candidate were defined, lower bound, upper bound) of
    each GT measure. An undefined-cosine candidate may or may not be accepted, may tie or
    outrank; the bounds are the extremes over all of those possibilities (white-box: the
    auditor may pick the attacker's best reading, so the upper bound is the headline-relevant
    one)."""
    linked = scan.linked
    unk = scan.unk_rows > 0
    cnt, hit = scan.kcnt_s, scan.khit_s
    safe = np.maximum(cnt, 1)
    pt_s = np.where(cnt > 0, hit / safe, 0.0)
    hi_s = np.where(hit, 1.0 / safe, np.where(scan.uhit_new, 1.0 / (cnt + 1), 0.0))
    lo_s = np.where(hit, 1.0 / np.maximum(cnt + scan.unew_s, 1), 0.0)

    kc, kh, u, uh = scan.kcnt_r, scan.khit_r, scan.unk_rows, scan.uhit_rows
    pt_r = np.where(kc > 0, kh / np.maximum(kc, 1), 0.0)
    hi_r = np.where(kc + uh > 0, (kh + uh) / np.maximum(kc + uh, 1), 0.0)
    lo_r = np.where(kc + (u - uh) > 0, kh / np.maximum(kc + u - uh, 1), 0.0)

    tc = scan.tcnt_s
    pt_t = np.where(linked & (tc > 0), scan.thit_s / np.maximum(tc, 1), 0.0)
    hi_t = np.where(uh > 0, 1.0, pt_t)  # an unknown row of the own subject may be the top one
    lo_t = np.where(unk, 0.0, pt_t)  # an unknown row may outrank the true one
    return {
        "subjects": (pt_s, lo_s, hi_s), "rows": (pt_r, lo_r, hi_r), "top1": (pt_t, lo_t, hi_t),
    }


def _pop(arrs: Tuple[np.ndarray, np.ndarray, np.ndarray], mask: np.ndarray) -> Optional[Dict[str, Any]]:
    den = int(mask.sum())
    if den == 0:
        return None
    pt, lo, hi = (a[mask] for a in arrs)
    return {
        "point": float(pt.sum() / den), "lower": float(lo.sum() / den), "upper": float(hi.sum() / den),
        "n_uncertain": int((hi - lo > _UNC_EPS).sum()), "denominator": den,
    }


def _precision(rec, scan: RungScan, mask: np.ndarray) -> Dict[str, Any]:
    pt, lo, hi = rec["top1"]
    linked = scan.linked & mask
    unlinked_unknown = ~scan.linked & (scan.unk_rows > 0) & mask
    n_l, n_ul = int(linked.sum()), int(unlinked_unknown.sum())
    top_unc = int((hi[linked] - lo[linked] > _UNC_EPS).sum())
    if n_ul == 0 and top_unc == 0:
        return {"point": float(pt[linked].sum() / n_l) if n_l else None, "lower": None, "upper": None,
                "n_uncertain": 0, "denominator": n_l}
    num_lo = float(lo[linked].sum())
    num_hi = float(hi[linked].sum() + hi[unlinked_unknown].sum())
    den_hi = n_l + n_ul
    lower = num_lo / den_hi if den_hi else None
    upper = min(1.0, num_hi / n_l) if n_l else (1.0 if n_ul else None)
    return {"point": None, "lower": lower, "upper": upper, "n_uncertain": top_unc + n_ul,
            "denominator": n_l, "denominator_upper": den_hi}


def _gt_stats(rec, scan: RungScan, mask: np.ndarray) -> Dict[str, Any]:
    return {
        "subjects": _pop(rec["subjects"], mask), "rows": _pop(rec["rows"], mask),
        "top1": _pop(rec["top1"], mask), "precision": _precision(rec, scan, mask),
        "n": int(mask.sum()),
    }


def _pop_status(pop: Optional[Dict[str, Any]], **extra) -> Tuple[Optional[float], Dict[str, Any]]:
    if pop is None:
        return None, _status(None, "not_measured", "empty_population", **extra)
    if pop["n_uncertain"] == 0 and pop["point"] is not None:
        return pop["point"], _status(pop["point"], "measured", None, denominator=pop["denominator"],
                                      n_uncertain=0, **extra)
    if pop["n_uncertain"] == 0:  # defined by an empty set of declared links
        return None, _status(None, "not_measured", "no_declared_links", denominator=0, **extra)
    return None, _status(
        None, "bounded", "undefined_cosine_candidates", denominator=pop["denominator"],
        lower=pop["lower"], upper=pop["upper"], n_uncertain=pop["n_uncertain"], **extra)


def _headline(sub: Optional[Dict[str, Any]], rows: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The headline uniform-choice value: the larger of the two POPULATION values, so it
    never falls below what either declared simple strategy achieves (upper-oriented)."""
    if sub is None or rows is None:
        return None, None
    if sub["n_uncertain"] == 0 and rows["n_uncertain"] == 0:
        basis = "subjects" if sub["point"] >= rows["point"] else "rows"
        best = sub if basis == "subjects" else rows
        return dict(best), basis
    lower, upper = max(sub["lower"], rows["lower"]), max(sub["upper"], rows["upper"])
    n_unc = int(max(sub["n_uncertain"], rows["n_uncertain"]))
    basis = "subjects" if sub["upper"] >= rows["upper"] else "rows"
    if upper - lower <= _UNC_EPS:
        return {"point": upper, "lower": lower, "upper": upper, "n_uncertain": 0,
                "denominator": sub["denominator"]}, basis
    return {"point": None, "lower": lower, "upper": upper, "n_uncertain": n_unc,
            "denominator": sub["denominator"]}, basis


def _rung_measures(scan: RungScan, n: int, gt: Optional[PreparedGroundTruth]) -> Dict[str, Any]:
    linked = scan.linked
    n_linked = int(linked.sum())
    n_undet = int(((scan.unk_rows > 0) & ~linked).sum())
    m: Dict[str, Any] = {
        # a point value only if no undefined-cosine pair can matter; else bounds only
        "cvpl_lr": (n_linked / n if n else None) if n_undet == 0 else None,
        "cvpl_lr_lower": n_linked / n if n else None,
        "cvpl_lr_upper": (n_linked + n_undet) / n if n else None,
        "n_linked": n_linked,
        "n_undetermined": n_undet,
        "n_exact_fallback": int(scan.fallback_only.sum()),
        "n_exact_fallback_records": int(scan.fallback_any.sum()),
        "n_exact_fallback_pairs": int(scan.fallback_pairs),
        "n_with_candidates": int((scan.n_cand > 0).sum()),
        # cosine evaluations only; exact-fallback and unknown pairs are counted separately
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
    if gt is not None and scan.kcnt_s is not None:
        rec = _record_arrays(scan)
        m["_gt"] = {
            "full": _gt_stats(rec, scan, np.ones(n, dtype=bool)),
            "known": _gt_stats(rec, scan, gt.known_mask),
        }
    return m


def _gt_fields(sel: Dict[str, Any], gt: Optional[PreparedGroundTruth], n: int) -> Dict[str, Any]:
    """The GT measures with their statuses (never a silent zero, never a bound as a point)."""
    out: Dict[str, Any] = {"cvpl_id_uniform_basis": None}
    g = sel.get("_gt")
    if gt is None or gt.status in ("absent", "invalid") or g is None:
        reason = "no_ground_truth" if gt is None or gt.status == "absent" else gt.reason
        why = reason if reason else "ground_truth_unavailable"
        for key in GT_MEASURES:
            out[key] = None
            out[f"{key}_status"] = _status(None, "not_measured", why)
        return out
    if gt.status == "complete":
        s = g["full"]
        head, basis = _headline(s["subjects"], s["rows"])
        pops = {
            "cvpl_id_uniform_subjects_gt": s["subjects"], "cvpl_id_uniform_rows_gt": s["rows"],
            "cvpl_id_uniform_gt": head, "cvpl_top1_success_gt": s["top1"], "precision_at_tau": s["precision"],
        }
        for key in GT_MEASURES:
            extra = {"basis": basis} if key == "cvpl_id_uniform_gt" else {}
            out[key], out[f"{key}_status"] = _pop_status(pops[key], **extra)
        out["cvpl_id_uniform_basis"] = basis
        return out
    # partial: population measures are NOT available; the subset whose correspondence is
    # fully known is reported separately, with its own denominator
    why = "ground_truth_incomplete:" + str(gt.reason)
    k = g["known"]
    head, basis = _headline(k["subjects"], k["rows"])
    pops = {
        "cvpl_id_uniform_subjects_gt": k["subjects"], "cvpl_id_uniform_rows_gt": k["rows"],
        "cvpl_id_uniform_gt": head, "cvpl_top1_success_gt": k["top1"], "precision_at_tau": k["precision"],
    }
    for key in GT_MEASURES:
        pop = pops[key]
        out[key] = None
        extra: Dict[str, Any] = {"known_subset_denominator": pop["denominator"] if pop else 0}
        if pop is None:
            extra.update(known_subset_value=None, known_subset_reason=(
                "release_membership_unknown" if gt.anonymous_release else "no_source_with_known_correspondence"))
        elif pop["n_uncertain"] == 0:
            extra.update(known_subset_value=pop["point"])
        else:
            extra.update(known_subset_value=None, known_subset_lower=pop["lower"],
                         known_subset_upper=pop["upper"])
        out[f"{key}_status"] = _status(None, "not_measured", why, **extra)
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


def _encoded_classes(
    proj: CVPLProjection, cols_s: FeatureColumns, cols_r: FeatureColumns,
    deg_s: np.ndarray, deg_r: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalence classes of the encoded feature vectors of the degenerate records of both
    sides (-1 for a record with a defined projection). Two records are an exact-equal pair
    iff they share a class; aggregating by class replaces any per-pair bookkeeping."""
    cls_s = np.full(len(deg_s), -1, dtype=np.int64)
    cls_r = np.full(len(deg_r), -1, dtype=np.int64)
    rs, rr = np.flatnonzero(deg_s), np.flatnonzero(deg_r)
    if rs.size == 0 or rr.size == 0:
        return cls_s, cls_r
    xs = proj.phi_rows(cols_s, rs) + 0.0
    xr = proj.phi_rows(cols_r, rr) + 0.0
    _, inverse = np.unique(np.vstack([xs, xr]), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    cls_s[rs] = inverse[: len(rs)]
    cls_r[rr] = inverse[len(rs):]
    return cls_s, cls_r


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
        "evaluation_model": "white_box_upper_estimate",
        "metric_semantics": {
            "cvpl_lr": "existential_candidate_exposure",
            "cvpl_id_uniform_subjects_gt": "uniform_choice_among_distinct_candidate_subjects",
            "cvpl_id_uniform_rows_gt": "uniform_choice_among_candidate_released_rows",
            "cvpl_id_uniform_gt": "larger_of_subjects_and_rows_population_values",
            "cvpl_top1_success_gt": "similarity_top1_correct_identity_success",
            "precision_at_tau": "correct_share_among_declared_top1_links",
            "cvpl_weighted_lr_legacy": "weighted_existential_linkage_v1",
        },
        "interpretation": (
            "White-box evaluation of one specified attacker (declared attributes, blocking, "
            "projection, threshold) with full auditor access: an upper, worst-case-oriented "
            "estimate, not an average over attackers. Not a probability of re-identification "
            "in general and not a verdict on anonymity."
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
        "fingerprint": proj.fingerprint(),
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
    if not (np.isfinite(zs).all() and np.isfinite(zr).all()):
        # a non-finite representation says nothing about linkage: it must never reach the
        # threshold, where NaN similarities would silently count as "no link"
        base.update(_all_not_measured("non_finite_representation"))
        base["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        return CVPLResult(base, proj)
    tol = cfg.zero_norm_tolerance
    ns = np.linalg.norm(zs, axis=1)
    nr = np.linalg.norm(zr, axis=1)
    deg_s, deg_r = ns <= tol, nr <= tol
    zs = np.where(deg_s[:, None], 0.0, zs / np.where(deg_s, 1.0, ns)[:, None])
    zr = np.where(deg_r[:, None], 0.0, zr / np.where(deg_r, 1.0, nr)[:, None])
    base["degenerate_vectors"] = {"source": int(deg_s.sum()), "release": int(deg_r.sum())}
    cls_s, cls_r = _encoded_classes(proj, cols_s, cols_r, deg_s, deg_r)

    # -- ground truth ----------------------------------------------------------------
    gt = prepare_ground_truth(cfg, source, release)
    base["ground_truth"] = {
        "status": gt.status, "reason": gt.reason, "scope_declared": None if cfg.ground_truth is None else cfg.ground_truth.scope,
        "gt_coverage": (gt.n_known / n) if cfg.ground_truth is not None and gt.status not in ("invalid",) else None,
        "n_known": gt.n_known if cfg.ground_truth is not None else 0,
        "release_subject_duplicates": (not gt.rel_unique) if gt.rel_code is not None else None,
        "release_membership_unknown": bool(gt.anonymous_release) if gt.rel_code is not None else None,
    }
    base["gt_coverage"] = base["ground_truth"]["gt_coverage"]
    use_gt = gt.status in ("complete", "partial")

    scanner = _Scanner(cfg, zs, zr, deg_s, deg_r, gt if use_gt else None, cls_s, cls_r, encs)

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
            elif plan.exceeded or plan.comparisons > cfg.max_comparisons:
                entry.update(
                    status="not_measured", reason="resource_limit",
                    detail=(f"{'at least ' if plan.exceeded else ''}{plan.comparisons:,} comparisons "
                            f"planned, limit {cfg.max_comparisons:,}"),
                )
                state = {"status": "not_measured", "reason": "resource_limit"}
            else:
                scan = scanner.scan([(plan, [])], use_gt)
                meas = _rung_measures(scan, n, gt if use_gt else None)
                cumulative |= scan.linked
                cum_with_cand |= scan.n_cand > 0
                entry.update(status="measured", reason=None, cumulative_lr=float(cumulative.sum() / n))
                entry["measures"] = {k: v for k, v in meas.items() if k != "_gt"}
                if use_gt:
                    entry["measures"].update(_gt_fields(meas, gt, n))
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
            head_meas = _rung_measures(uscan, n, gt if use_gt else None)
            head_state = {"status": "measured", "reason": None}
    if head_meas is None:
        base.update(_all_not_measured(head_state.get("reason") or "not_measured",
                                      status=head_state.get("status", "not_measured")))
    else:
        head = {k: v for k, v in head_meas.items() if k != "_gt"}
        base.update(head)
        proj_degenerate = info.degenerate or (deg_s.all() and deg_r.all())
        # a degenerate representation only invalidates a conclusion when there ARE candidates
        # to compare; no admissible candidate is a settled zero whatever the geometry
        if proj_degenerate and head["n_linked"] == 0 and head["n_with_candidates"] > 0:
            base.update(_all_not_measured("projection_degenerate"))
        else:
            lr_status = "measured" if head["n_undetermined"] == 0 else "bounded"
            base["cvpl_lr_status"] = _status(
                head["cvpl_lr"], lr_status,
                None if lr_status == "measured" else "undefined_cosine_candidates", denominator=n,
                lower=head["cvpl_lr_lower"], upper=head["cvpl_lr_upper"],
                n_uncertain=head["n_undetermined"])
            base.update(_gt_fields(head_meas, gt, n))
        base["undefined_policy"] = (
            "a source record with an admissible candidate whose cosine is undefined (a zero "
            "projected vector that is not an exact-equal pair) is unknown: it stays in the "
            "denominator N and every measure it can influence is published with bounds "
            "(status 'bounded', point value null) instead of a point value; n_undetermined counts "
            "the records with no known link and at least one unknown candidate"
        )
    base["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    return CVPLResult(base, proj)


def _all_not_measured(reason: str, status: str = "not_measured") -> Dict[str, Any]:
    out: Dict[str, Any] = {"cvpl_id_uniform_basis": None}
    for key in ("cvpl_lr", *GT_MEASURES, "cvpl_weighted_lr_legacy"):
        out[key] = None
        out[f"{key}_status"] = _status(None, status, reason)
    return out
