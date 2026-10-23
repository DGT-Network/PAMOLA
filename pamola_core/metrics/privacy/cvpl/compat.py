"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL blocking / candidate compatibility
Package:       pamola_core.metrics.privacy.cvpl
License:       BSD 3-Clause

``B_lambda``: which released rows are admissible candidates for a source row.
Compatibility is decided per quasi-identifier from the *declared* rules:

* equal values, or the bin of a numeric point value (``bin_width``);
* the wildcard token (``*``): suppression, compatible with anything;
* a numeric interval (``[20-29]``, ``20-29``, ``pandas.Interval``) contains or
  overlaps the other side's value;
* an ancestor in a user-supplied categorical hierarchy;
* ``null`` according to ``null_policy``.

A released value that none of these rules explains (a categorical label outside
the source domain with no hierarchy, a numeric string that is not an interval)
makes the QID *not applicable*: it is reported, never silently treated as an exact
mismatch that would look like protection.

Scalability: QIDs that need no pairwise reasoning ("plain") become hash keys that
partition both datasets; only the "special" QIDs (wildcards, intervals,
hierarchies) are compared pairwise, between *groups* of rows with identical tokens
inside a partition. Nothing here builds a ``|D| x |D'|`` structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pamola_core.errors.exceptions import InvalidParameterError, ResourceError

from .config import CVPLConfig, QIDSpec

STATE_VALUE, STATE_NULL, STATE_WILD = 0, 1, 2
_NULL_KEY = np.iinfo(np.int64).min + 1
_MAX_HIERARCHY_DOMAIN = 8000

_NUM = r"[-+]?\d+(?:\.\d+)?"
_INTERVAL_RE = re.compile(
    rf"^\s*(?P<lb>[\[\(]?)\s*(?P<lo>{_NUM})\s*(?:,|;|\.\.|–|—|\bto\b|-)\s*"
    rf"(?P<hi>{_NUM})\s*(?P<rb>[\]\)]?)\s*$",
    re.IGNORECASE,
)
_OPEN_ENDED_RE = re.compile(rf"^\s*[\[\(]?\s*(?P<lo>{_NUM})\s*\+\s*[\]\)]?\s*$")


#: limitation code -> (what is wrong, how to fix it); shown in the result, names only
LIMITATION_HELP = {
    "generalized_values_without_hierarchy": (
        "the release holds values outside the source's domain that look like generalized levels, "
        "and no hierarchy was declared",
        "declare qids[].hierarchy ({child: parent}), or set qids[].generalized=False if this QID is "
        "exact-only and new labels simply mean different people",
    ),
    "release_values_not_numeric_or_interval": (
        "the release holds values that are neither numbers, intervals nor the wildcard",
        "provide numeric or interval values (e.g. '[20-29]'), or set qids[].generalized=False to treat "
        "them as non-matching",
    ),
}


def limitation_message(qid: str, code: Optional[str]) -> str:
    what, fix = LIMITATION_HELP.get(code or "", (code or "unknown limitation", "see the documentation"))
    return f"QID '{qid}' is not applicable: {what}. Fix: {fix}."


@dataclass
class QIDEncoding:
    """Both sides of one QID, encoded for compatibility tests."""

    name: str
    kind: str
    state_s: np.ndarray
    state_r: np.ndarray
    applicable: bool = True
    limitation: Optional[str] = None
    stats: Dict[str, int] = field(default_factory=dict)
    # categorical
    code_s: Optional[np.ndarray] = None
    code_r: Optional[np.ndarray] = None
    related: Optional[np.ndarray] = None  # K x K bool, or None for plain equality
    # numeric (closed/open interval per row)
    lo_s: Optional[np.ndarray] = None
    hi_s: Optional[np.ndarray] = None
    loi_s: Optional[np.ndarray] = None
    hii_s: Optional[np.ndarray] = None
    lo_r: Optional[np.ndarray] = None
    hi_r: Optional[np.ndarray] = None
    loi_r: Optional[np.ndarray] = None
    hii_r: Optional[np.ndarray] = None
    has_interval: bool = False
    #: numeric midpoint per side, for the feature representation (NaN where unknown)
    mid_s: Optional[np.ndarray] = None
    mid_r: Optional[np.ndarray] = None

    @property
    def any_wild(self) -> bool:
        return bool((self.state_s == STATE_WILD).any() or (self.state_r == STATE_WILD).any())

    @property
    def any_null(self) -> bool:
        return bool((self.state_s == STATE_NULL).any() or (self.state_r == STATE_NULL).any())


# ---------------------------------------------------------------------------
# encoding of one QID
# ---------------------------------------------------------------------------
def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def resolve_kind(spec: QIDSpec, source: pd.Series) -> str:
    if spec.kind != "auto":
        return spec.kind
    return "numeric" if _is_numeric_series(source) else "categorical"


def _parse_numeric_object(value, wildcard_tokens):
    """-> (state, lo, hi, lo_incl, hi_incl, kind) with kind in value/interval/bad."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value is pd.NaT:
        return STATE_NULL, np.nan, np.nan, True, True, "value"
    if isinstance(value, pd.Interval):
        return (STATE_VALUE, float(value.left), float(value.right),
                value.closed in ("left", "both"), value.closed in ("right", "both"), "interval")
    if isinstance(value, (bool, np.bool_)):
        return STATE_VALUE, np.nan, np.nan, True, True, "bad"
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return STATE_VALUE, v, v, True, True, "value"
    text = str(value).strip()
    if text in wildcard_tokens:
        return STATE_WILD, np.nan, np.nan, True, True, "value"
    try:
        v = float(text)
        return STATE_VALUE, v, v, True, True, "value"
    except ValueError:
        pass
    m = _OPEN_ENDED_RE.match(text)
    if m:
        return STATE_VALUE, float(m.group("lo")), np.inf, True, False, "interval"
    m = _INTERVAL_RE.match(text)
    if m:
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if lo <= hi:
            return (STATE_VALUE, lo, hi, m.group("lb") != "(", m.group("rb") != ")", "interval")
    return STATE_VALUE, np.nan, np.nan, True, True, "bad"


def _numeric_arrays(series: pd.Series, wildcard_tokens, bin_width):
    """Parse one numeric side. Returns state, lo, hi, loi, hii, n_interval, n_bad."""
    n = len(series)
    if isinstance(series.dtype, pd.IntervalDtype):
        closed = series.array.closed
        lo = series.array.left.to_numpy(dtype=float)
        hi = series.array.right.to_numpy(dtype=float)
        state = np.where(series.isna().to_numpy(), STATE_NULL, STATE_VALUE).astype(np.int8)
        loi = np.full(n, closed in ("left", "both"))
        hii = np.full(n, closed in ("right", "both"))
        mid = np.where(state == STATE_VALUE, (lo + hi) / 2.0, np.nan)
        return state, lo, hi, loi, hii, int((state == STATE_VALUE).sum()), 0, mid
    if _is_numeric_series(series):
        vals = series.to_numpy(dtype=float, na_value=np.nan)
        state = np.where(np.isnan(vals), STATE_NULL, STATE_VALUE).astype(np.int8)
        lo, hi = vals.copy(), vals.copy()
        loi, hii = np.ones(n, dtype=bool), np.ones(n, dtype=bool)
        n_int = n_bad = 0
    else:
        codes, uniques = pd.factorize(series.astype(object), use_na_sentinel=True)
        parsed = [_parse_numeric_object(u, wildcard_tokens) for u in uniques]
        st = np.array([p[0] for p in parsed] + [STATE_NULL], dtype=np.int8)
        plo = np.array([p[1] for p in parsed] + [np.nan])
        phi = np.array([p[2] for p in parsed] + [np.nan])
        pli = np.array([p[3] for p in parsed] + [True])
        phi_i = np.array([p[4] for p in parsed] + [True])
        pk = np.array([p[5] for p in parsed] + ["value"], dtype=object)
        idx = np.where(codes < 0, len(uniques), codes)
        state, lo, hi, loi, hii = st[idx], plo[idx], phi[idx], pli[idx], phi_i[idx]
        kind = pk[idx]
        n_int = int((kind == "interval").sum())
        bad = (kind == "bad")
        n_bad = int(bad.sum())
        # unparsable text carries no usable value on this side
        state = np.where(bad, STATE_NULL, state).astype(np.int8)
    mid = np.where(state == STATE_VALUE, (lo + hi) / 2.0, np.nan)
    if bin_width:
        w = float(bin_width)
        point = (state == STATE_VALUE) & (lo == hi) & loi & hii
        base = np.floor(lo / w) * w
        lo = np.where(point, base, lo)
        hi = np.where(point, base + w, hi)
        hii = np.where(point, False, hii)
    return state, lo, hi, loi, hii, n_int, n_bad, mid


def encode_qid(
    spec: QIDSpec, source: pd.Series, release: pd.Series, cfg: CVPLConfig
) -> QIDEncoding:
    """Encode one QID on both sides; decides ``applicable`` (see module doc)."""
    kind = resolve_kind(spec, source)
    wild = set(cfg.wildcard_tokens)
    if kind == "numeric":
        s_state, s_lo, s_hi, s_loi, s_hii, s_int, s_bad, s_mid = _numeric_arrays(source, wild, spec.bin_width)
        r_state, r_lo, r_hi, r_loi, r_hii, r_int, r_bad, r_mid = _numeric_arrays(release, wild, spec.bin_width)
        enc = QIDEncoding(
            name=spec.name, kind="numeric", state_s=s_state, state_r=r_state,
            lo_s=s_lo, hi_s=s_hi, loi_s=s_loi, hii_s=s_hii,
            lo_r=r_lo, hi_r=r_hi, loi_r=r_loi, hii_r=r_hii,
            has_interval=bool(s_int or r_int),
        )
        enc.mid_s, enc.mid_r = s_mid, r_mid
        enc.stats = {
            "source_wildcard": int((s_state == STATE_WILD).sum()),
            "release_wildcard": int((r_state == STATE_WILD).sum()),
            "source_null": int((s_state == STATE_NULL).sum()),
            "release_null": int((r_state == STATE_NULL).sum()),
            "release_intervals": int(r_int),
            "source_intervals": int(s_int),
            "release_unparsable": int(r_bad),
        }
        if r_bad and spec.generalized is not False:
            enc.applicable = False
            enc.limitation = "release_values_not_numeric_or_interval"
        return enc

    # categorical -----------------------------------------------------------
    def to_str(series: pd.Series):
        null = series.isna().to_numpy()
        text = series.astype(object).where(~null, "").astype(str).to_numpy(dtype=object)
        return text, null

    t_s, null_s = to_str(source)
    t_r, null_r = to_str(release)
    wild_s = np.isin(t_s, list(wild)) & ~null_s
    wild_r = np.isin(t_r, list(wild)) & ~null_r
    state_s = np.where(null_s, STATE_NULL, np.where(wild_s, STATE_WILD, STATE_VALUE)).astype(np.int8)
    state_r = np.where(null_r, STATE_NULL, np.where(wild_r, STATE_WILD, STATE_VALUE)).astype(np.int8)
    val_s, val_r = state_s == STATE_VALUE, state_r == STATE_VALUE
    codes, domain = pd.factorize(np.concatenate([t_s[val_s], t_r[val_r]]))
    code_s = np.full(len(t_s), -1, dtype=np.int64)
    code_r = np.full(len(t_r), -1, dtype=np.int64)
    code_s[val_s] = codes[: int(val_s.sum())]
    code_r[val_r] = codes[int(val_s.sum()):]
    source_vals = set(t_s[val_s].tolist())
    unmatched_rows = int(np.sum(val_r & ~np.isin(t_r, list(source_vals)))) if len(source_vals) else int(val_r.sum())
    enc = QIDEncoding(
        name=spec.name, kind="categorical", state_s=state_s, state_r=state_r,
        code_s=code_s, code_r=code_r,
    )
    enc.stats = {
        "source_wildcard": int(wild_s.sum()), "release_wildcard": int(wild_r.sum()),
        "source_null": int(null_s.sum()), "release_null": int(null_r.sum()),
        "release_values_outside_source_domain": unmatched_rows,
    }
    if spec.hierarchy:
        k = len(domain)
        if k > _MAX_HIERARCHY_DOMAIN:
            raise ResourceError(
                operation="CVPL", reason=(
                    f"QID '{spec.name}' has {k} distinct values with a hierarchy; the hierarchy "
                    f"matrix is limited to {_MAX_HIERARCHY_DOMAIN}"),
            )
        parent = {str(c): str(p) for c, p in spec.hierarchy.items()}
        index = {str(v): i for i, v in enumerate(domain)}
        related = np.eye(k, dtype=bool)
        for value, i in index.items():
            seen, cur = {value}, value
            while cur in parent and parent[cur] not in seen:
                cur = parent[cur]
                seen.add(cur)
                j = index.get(cur)
                if j is not None:
                    related[i, j] = related[j, i] = True
        enc.related = related if related.sum() > k else None
    elif unmatched_rows and spec.generalized is not False:
        enc.applicable = False
        enc.limitation = "generalized_values_without_hierarchy"
    return enc


# ---------------------------------------------------------------------------
# rung planning
# ---------------------------------------------------------------------------
def _combine(parts: List[np.ndarray]) -> np.ndarray:
    """Factorize several int64 key arrays into one dense id array."""
    comb = np.zeros(len(parts[0]), dtype=np.int64)
    for p in parts:
        f, uniq = pd.factorize(p)
        comb = comb * (len(uniq) + 1) + f
        comb = pd.factorize(comb)[0].astype(np.int64)
    return comb


def _plain_key(enc: QIDEncoding, side: str) -> np.ndarray:
    state = enc.state_s if side == "s" else enc.state_r
    if enc.kind == "categorical":
        key = (enc.code_s if side == "s" else enc.code_r).copy()
        key[state == STATE_NULL] = -1
        key[state == STATE_WILD] = -2
        return key
    lo = enc.lo_s if side == "s" else enc.lo_r
    key = np.nan_to_num(lo, nan=0.0).view(np.int64).copy()
    key[state == STATE_NULL] = _NULL_KEY
    key[state == STATE_WILD] = _NULL_KEY + 1
    return key


def _is_special(enc: QIDEncoding, null_policy: str) -> bool:
    if enc.any_wild:
        return True
    if null_policy == "wildcard" and enc.any_null:
        return True
    if enc.kind == "categorical":
        return enc.related is not None
    return enc.has_interval


def _special_token(enc: QIDEncoding, side: str) -> np.ndarray:
    """One int64 token per row identifying its special-QID value (for grouping)."""
    state = enc.state_s if side == "s" else enc.state_r
    if enc.kind == "categorical":
        code = enc.code_s if side == "s" else enc.code_r
        return np.where(state == STATE_VALUE, code, -1 - state.astype(np.int64))
    lo, hi, loi, hii = (
        (enc.lo_s, enc.hi_s, enc.loi_s, enc.hii_s) if side == "s"
        else (enc.lo_r, enc.hi_r, enc.loi_r, enc.hii_r)
    )
    parts = [
        np.nan_to_num(lo, nan=0.0, posinf=1e308, neginf=-1e308).view(np.int64),
        np.nan_to_num(hi, nan=0.0, posinf=1e308, neginf=-1e308).view(np.int64),
        loi.astype(np.int64) * 2 + hii.astype(np.int64),
        state.astype(np.int64),
    ]
    return _combine(parts)


def _group_compat(enc: QIDEncoding, rep_s: np.ndarray, rep_r: np.ndarray, null_policy: str) -> np.ndarray:
    """Boolean ``(len(rep_s), len(rep_r))`` compatibility for one special QID."""
    ss = enc.state_s[rep_s][:, None]
    sr = enc.state_r[rep_r][None, :]
    wild = (ss == STATE_WILD) | (sr == STATE_WILD)
    if null_policy == "wildcard":
        wild = wild | (ss == STATE_NULL) | (sr == STATE_NULL)
        nulls_equal = np.zeros(wild.shape, dtype=bool)
    elif null_policy == "equal_null":
        nulls_equal = (ss == STATE_NULL) & (sr == STATE_NULL)
    else:
        nulls_equal = np.zeros(wild.shape, dtype=bool)
    both_value = (ss == STATE_VALUE) & (sr == STATE_VALUE)
    if enc.kind == "categorical":
        cs = enc.code_s[rep_s][:, None]
        cr = enc.code_r[rep_r][None, :]
        if enc.related is not None:
            safe_s = np.where(cs < 0, 0, cs)
            safe_r = np.where(cr < 0, 0, cr)
            rel = enc.related[safe_s, safe_r]
        else:
            rel = cs == cr
        return wild | nulls_equal | (both_value & rel)
    lo_s, hi_s = enc.lo_s[rep_s][:, None], enc.hi_s[rep_s][:, None]
    loi_s, hii_s = enc.loi_s[rep_s][:, None], enc.hii_s[rep_s][:, None]
    lo_r, hi_r = enc.lo_r[rep_r][None, :], enc.hi_r[rep_r][None, :]
    loi_r, hii_r = enc.loi_r[rep_r][None, :], enc.hii_r[rep_r][None, :]
    # [lo_s, hi_s] overlaps [lo_r, hi_r]: each lower end is below the other's upper end
    a = (lo_s < hi_r) | ((lo_s == hi_r) & loi_s & hii_r)
    b = (lo_r < hi_s) | ((lo_r == hi_s) & loi_r & hii_s)
    return wild | nulls_equal | (both_value & a & b)


@dataclass
class RungPlan:
    """Candidate structure of one ladder rung (no pair is materialised)."""

    qids: Tuple[str, ...]
    applicable: bool = True
    reason: Optional[str] = None
    n_source: int = 0
    n_release: int = 0
    n_special: int = 0
    n_plain: int = 0
    # per partition: (src rows sorted, src group offsets, rel rows sorted, rel group offsets, compat or None)
    _parts: List[tuple] = field(default_factory=list)
    comparisons: int = 0
    group_pairs: int = 0
    n_blocks: int = 0

    def blocks(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(source_rows, candidate_rows)`` blocks; every source row of a block
        shares the same candidate set."""
        for s_rows, s_off, r_rows, r_off, compat in self._parts:
            if compat is None:
                yield s_rows, r_rows
                continue
            for gi in range(len(s_off) - 1):
                cols = np.flatnonzero(compat[gi])
                if len(cols) == 0:
                    continue
                rows = s_rows[s_off[gi]:s_off[gi + 1]]
                if len(cols) == len(r_off) - 1:
                    cand = r_rows
                else:
                    cand = np.concatenate([r_rows[r_off[c]:r_off[c + 1]] for c in cols])
                yield rows, cand


def plan_rung(
    qids: Sequence[str], encs: Dict[str, QIDEncoding], n_source: int, n_release: int, cfg: CVPLConfig,
) -> RungPlan:
    plan = RungPlan(qids=tuple(qids), n_source=n_source, n_release=n_release)
    bad = [q for q in qids if not encs[q].applicable]
    if bad:
        plan.applicable = False
        plan.reason = " ".join(limitation_message(q, encs[q].limitation) for q in bad)
        return plan
    ok_s = np.ones(n_source, dtype=bool)
    ok_r = np.ones(n_release, dtype=bool)
    if cfg.null_policy == "incompatible":
        for q in qids:
            ok_s &= encs[q].state_s != STATE_NULL
            ok_r &= encs[q].state_r != STATE_NULL
    rows_s = np.flatnonzero(ok_s)
    rows_r = np.flatnonzero(ok_r)
    if len(rows_s) == 0 or len(rows_r) == 0:
        return plan

    plain = [q for q in qids if not _is_special(encs[q], cfg.null_policy)]
    special = [q for q in qids if _is_special(encs[q], cfg.null_policy)]
    plan.n_plain, plan.n_special = len(plain), len(special)

    if plain:
        keys = [np.concatenate([_plain_key(encs[q], "s")[rows_s], _plain_key(encs[q], "r")[rows_r]]) for q in plain]
        pid = _combine(keys)
    else:
        pid = np.zeros(len(rows_s) + len(rows_r), dtype=np.int64)
    pid_s, pid_r = pid[: len(rows_s)], pid[len(rows_s):]

    if special:
        tok_s = [_special_token(encs[q], "s")[rows_s] for q in special]
        tok_r = [_special_token(encs[q], "r")[rows_r] for q in special]
        sg_s = _combine([pid_s] + tok_s)
        sg_r = _combine([pid_r] + tok_r)
    else:
        sg_s, sg_r = pid_s, pid_r

    # partitions present on both sides
    common = np.intersect1d(np.unique(pid_s), np.unique(pid_r))
    if len(common) == 0:
        return plan
    keep_s = np.isin(pid_s, common)
    keep_r = np.isin(pid_r, common)
    rs, rr = rows_s[keep_s], rows_r[keep_r]
    pid_s, pid_r, sg_s, sg_r = pid_s[keep_s], pid_r[keep_r], sg_s[keep_s], sg_r[keep_r]

    # sort both sides by (partition, group); then walk partitions
    order_s = np.lexsort((sg_s, pid_s))
    order_r = np.lexsort((sg_r, pid_r))
    rs, pid_s, sg_s = rs[order_s], pid_s[order_s], sg_s[order_s]
    rr, pid_r, sg_r = rr[order_r], pid_r[order_r], sg_r[order_r]
    ps_b = np.concatenate([[0], np.flatnonzero(np.diff(pid_s)) + 1, [len(rs)]])
    pr_b = np.concatenate([[0], np.flatnonzero(np.diff(pid_r)) + 1, [len(rr)]])
    ps_ids, pr_ids = pid_s[ps_b[:-1]], pid_r[pr_b[:-1]]
    pr_pos = {int(p): i for i, p in enumerate(pr_ids)}

    total_cmp = 0
    total_gp = 0
    for i, p in enumerate(ps_ids):
        j = pr_pos[int(p)]
        s_rows = rs[ps_b[i]:ps_b[i + 1]]
        r_rows = rr[pr_b[j]:pr_b[j + 1]]
        if not special:
            total_cmp += len(s_rows) * len(r_rows)
            plan._parts.append((s_rows, np.array([0, len(s_rows)]), r_rows, np.array([0, len(r_rows)]), None))
            plan.n_blocks += 1
            continue
        gs = sg_s[ps_b[i]:ps_b[i + 1]]
        gr = sg_r[pr_b[j]:pr_b[j + 1]]
        s_off = np.concatenate([[0], np.flatnonzero(np.diff(gs)) + 1, [len(gs)]])
        r_off = np.concatenate([[0], np.flatnonzero(np.diff(gr)) + 1, [len(gr)]])
        n_sg, n_rg = len(s_off) - 1, len(r_off) - 1
        total_gp += n_sg * n_rg
        if total_gp > cfg.max_group_pairs:
            raise ResourceError(
                operation="CVPL",
                reason=(f"rung {list(qids)} needs more than {cfg.max_group_pairs:,} group-level "
                        "compatibility checks (wildcards / intervals / hierarchies on high-cardinality "
                        "QIDs); use a coarser ladder or raise max_group_pairs"),
            )
        rep_s = s_rows[s_off[:-1]]
        rep_r = r_rows[r_off[:-1]]
        compat = np.ones((n_sg, n_rg), dtype=bool)
        for q in special:
            compat &= _group_compat(encs[q], rep_s, rep_r, cfg.null_policy)
        r_sizes = np.diff(r_off)
        s_sizes = np.diff(s_off)
        total_cmp += int((compat @ r_sizes) @ s_sizes)
        plan._parts.append((s_rows, s_off, r_rows, r_off, compat))
        plan.n_blocks += n_sg
    plan.comparisons = total_cmp
    plan.group_pairs = total_gp
    return plan


def classify_transition(prev: Sequence[str], nxt: Sequence[str]) -> str:
    """How the candidate sets of two consecutive rungs relate, from the QID sets alone.

    ``relaxing``: the next rung constrains fewer QIDs (candidates only grow);
    ``tightening``: more QIDs (candidates only shrink); ``equal``; otherwise
    ``non_nested`` - the sets are not ordered by inclusion and no monotonicity
    statement applies. Valid because every QID is compared by the same rule in
    every rung.
    """
    a, b = frozenset(prev), frozenset(nxt)
    if a == b:
        return "equal"
    if b < a:
        return "relaxing"
    if a < b:
        return "tightening"
    return "non_nested"
