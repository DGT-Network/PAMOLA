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

STATE_VALUE, STATE_NULL, STATE_WILD, STATE_BAD = 0, 1, 2, 3
_NULL_KEY = np.iinfo(np.int64).min + 1
_MAX_HIERARCHY_DOMAIN = 8000
#: integers beyond this are not exactly representable as float64
_EXACT_LIMIT = 2**53
#: cells of a streamed group-compatibility tile
_TILE_CELLS = 4_000_000

_NUM = r"[-+]?\d+(?:\.\d+)?"
_INTERVAL_RE = re.compile(
    rf"^\s*(?P<lb>[\[\(]?)\s*(?P<lo>{_NUM})\s*(?:,|;|\.\.|–|—|\bto\b|-)\s*"
    rf"(?P<hi>{_NUM})\s*(?P<rb>[\]\)]?)\s*$",
    re.IGNORECASE,
)
_OPEN_ENDED_RE = re.compile(rf"^\s*[\[\(]?\s*(?P<lo>{_NUM})\s*\+\s*[\]\)]?\s*$")
_INT_RE = re.compile(r"^[-+]?\d+$")


#: limitation code -> (what is wrong, how to fix it); shown in the result, names only
LIMITATION_HELP = {
    "generalized_values_without_hierarchy": (
        "the release holds values outside the source's domain that look like generalized levels, "
        "and no hierarchy was declared",
        "declare qids[].hierarchy ({child: parent}), or set qids[].generalized=False if this QID is "
        "exact-only and new labels simply mean different people",
    ),
    "generalized_values_not_in_hierarchy": (
        "the release holds values outside the source's domain that the declared hierarchy does not "
        "mention (a hierarchy that does not cover them gives no interpretation of those values)",
        "extend qids[].hierarchy to cover every generalized value in the release, or set "
        "qids[].generalized=False if this QID is exact-only and new labels simply mean different people",
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
    # numeric (closed/open interval per row). ``lo2``/``hi2`` are exact remainders for
    # integers beyond 2**53 (all zero otherwise): a value is the pair (float, remainder).
    lo_s: Optional[np.ndarray] = None
    hi_s: Optional[np.ndarray] = None
    loi_s: Optional[np.ndarray] = None
    hii_s: Optional[np.ndarray] = None
    lo_r: Optional[np.ndarray] = None
    hi_r: Optional[np.ndarray] = None
    loi_r: Optional[np.ndarray] = None
    hii_r: Optional[np.ndarray] = None
    lo2_s: Optional[np.ndarray] = None
    hi2_s: Optional[np.ndarray] = None
    lo2_r: Optional[np.ndarray] = None
    hi2_r: Optional[np.ndarray] = None
    exact: bool = False
    has_interval: bool = False
    #: finite numeric representative per side for the feature representation (NaN where unknown)
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


def _split_exact(values: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Integers -> (float64 value, exact float64 remainder); ``int(v) == int(hi) + rem``."""
    hi = np.array([float(v) for v in values], dtype=float)
    rem = np.array([float(int(v) - int(h)) for v, h in zip(values, hi.tolist())], dtype=float)
    return hi, rem


def _int_pair(iv: int) -> Tuple[float, float]:
    """One integer -> ``(float, exact remainder)``; the remainder is 0 up to 2**53."""
    if abs(iv) > _EXACT_LIMIT:
        hi = float(iv)
        return hi, float(iv - int(hi))
    return float(iv), 0.0


def _is_integer_scalar(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _endpoint(v) -> Tuple[float, float]:
    """An interval endpoint object (``pd.Interval.left`` ...) -> ``(float, remainder)``."""
    if _is_integer_scalar(v):
        return _int_pair(int(v))
    return float(v) + 0.0, 0.0


def _token_pair(token: str) -> Tuple[float, float]:
    """A numeric token of an interval string -> ``(float, remainder)``; integers keep
    every digit (``9007199254740993`` is not ``9007199254740992``)."""
    if _INT_RE.match(token):
        return _int_pair(int(token))
    return float(token) + 0.0, 0.0


def _index_pairs(index: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    """Interval endpoints of an IntervalDtype column -> exact (float, remainder) arrays."""
    if pd.api.types.is_integer_dtype(index.dtype):
        present = np.asarray(index.notna())
        vals = index[present].to_numpy()
        hi = np.full(len(index), np.nan)
        rem = np.zeros(len(index))
        if vals.size and (int(vals.max()) > _EXACT_LIMIT or int(vals.min()) < -_EXACT_LIMIT):
            h, r = _split_exact(vals.tolist())
            hi[present] = h
            rem[present] = r
        else:
            hi[present] = vals.astype(float)
        return hi, rem
    return index.to_numpy(dtype=float), np.zeros(len(index))


def _finite_mid(lo: np.ndarray, hi: np.ndarray, state: np.ndarray) -> np.ndarray:
    """A finite representative for the feature space: the midpoint of a closed interval, the
    finite bound of an open-ended one (``60+`` -> 60), NaN (imputed) when nothing is finite."""
    both = np.isfinite(lo) & np.isfinite(hi)
    mid = np.where(both, lo / 2.0 + hi / 2.0, np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, np.nan)))
    return np.where(state == STATE_VALUE, mid, np.nan)


def _parse_numeric_object(value, wildcard_tokens):
    """-> (state, lo, hi, lo_incl, hi_incl, kind, lo2, hi2); kind in value/interval/bad."""
    nan = np.nan
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value is pd.NaT:
        return STATE_NULL, nan, nan, True, True, "value", 0.0, 0.0
    if isinstance(value, pd.Interval):
        (lo, lo2), (hi, hi2) = _endpoint(value.left), _endpoint(value.right)
        if np.isnan(lo) or np.isnan(hi):
            return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0
        return (STATE_VALUE, lo, hi, value.closed in ("left", "both"), value.closed in ("right", "both"),
                "interval", lo2, hi2)
    if isinstance(value, (bool, np.bool_)):
        return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0
    if isinstance(value, (int, np.integer)):
        v, rem = _int_pair(int(value))
        return STATE_VALUE, v, v, True, True, "value", rem, rem
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0
        v += 0.0  # canonical zero
        return STATE_VALUE, v, v, True, True, "value", 0.0, 0.0
    text = str(value).strip()
    if text in wildcard_tokens:
        return STATE_WILD, nan, nan, True, True, "value", 0.0, 0.0
    try:
        if _INT_RE.match(text):
            v, rem = _int_pair(int(text))
            return STATE_VALUE, v, v, True, True, "value", rem, rem
        v = float(text)
        if not np.isfinite(v):
            return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0
        v += 0.0
        return STATE_VALUE, v, v, True, True, "value", 0.0, 0.0
    except (ValueError, OverflowError):
        pass
    m = _OPEN_ENDED_RE.match(text)
    if m:
        try:
            lo, lo2 = _token_pair(m.group("lo"))
        except (ValueError, OverflowError):
            return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0
        return STATE_VALUE, lo, np.inf, True, False, "interval", lo2, 0.0
    m = _INTERVAL_RE.match(text)
    if m:
        try:
            lo_p, hi_p = _token_pair(m.group("lo")), _token_pair(m.group("hi"))
        except (ValueError, OverflowError):
            return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0
        if lo_p <= hi_p:  # lexicographic order of canonical (float, remainder) pairs
            return (STATE_VALUE, lo_p[0], hi_p[0], m.group("lb") != "(", m.group("rb") != ")", "interval",
                    lo_p[1], hi_p[1])
    return STATE_VALUE, nan, nan, True, True, "bad", 0.0, 0.0


def _bin_points(state, lo, hi, loi, hii, lo2, hi2, w):
    """Replace every point value by its half-open bin ``[base, base + w)``.

    Integers beyond 2**53 are binned EXACTLY (python integers, integral ``w``) and keep
    their remainder, so neighbouring integers fall into the bins they belong to. A bin
    whose float width vanishes (the value is too large for ``w`` to register) becomes the
    closed point itself, so that two equal values still share one bin.
    """
    point = (state == STATE_VALUE) & (lo == hi) & loi & hii & (lo2 == hi2)
    with np.errstate(invalid="ignore"):
        base = np.floor(lo / w) * w
        top = base + w
    collapsed = point & (top == base)
    lo_new = np.where(point, base, lo)
    hi_new = np.where(collapsed, base, np.where(point, top, hi))
    hii_new = np.where(collapsed, True, np.where(point, False, hii))
    lo2_new = np.where(point, 0.0, lo2)
    hi2_new = np.where(point, 0.0, hi2)
    if float(w).is_integer():
        wi = int(w)
        big = point & ((np.abs(lo) >= _EXACT_LIMIT) | (lo2 != 0))
        cache: Dict[Tuple[float, float], Tuple[float, float, float, float]] = {}
        for i in np.flatnonzero(big):
            key = (float(lo[i]), float(lo2[i]))
            got = cache.get(key)
            if got is None:
                v = int(key[0]) + int(key[1])
                b_int = (v // wi) * wi
                (bl, bl2), (bh, bh2) = _int_pair(b_int), _int_pair(b_int + wi)
                got = cache[key] = (bl, bl2, bh, bh2)
            lo_new[i], lo2_new[i], hi_new[i], hi2_new[i] = got
            hii_new[i] = False
    return lo_new, hi_new, hii_new, lo2_new, hi2_new


def _numeric_arrays(series: pd.Series, wildcard_tokens, bin_width):
    """Parse one numeric side. Returns state, lo, hi, loi, hii, lo2, hi2, n_interval, n_bad, mid.

    Three different things never merge: a *missing* value (state NULL), a *wildcard* (state
    WILD) and an *invalid* value that cannot be read as a number or interval (state BAD, which
    is compatible with nothing, not even another invalid value or a missing one).
    """
    n = len(series)
    lo2 = np.zeros(n)
    hi2 = np.zeros(n)
    if isinstance(series.dtype, pd.IntervalDtype):
        closed = series.array.closed
        lo, lo2 = _index_pairs(series.array.left)
        hi, hi2 = _index_pairs(series.array.right)
        null = series.isna().to_numpy()
        bad = ~null & (np.isnan(lo) | np.isnan(hi))
        state = np.where(null, STATE_NULL, np.where(bad, STATE_BAD, STATE_VALUE)).astype(np.int8)
        lo = lo + 0.0
        hi = hi + 0.0
        loi = np.full(n, closed in ("left", "both"))
        hii = np.full(n, closed in ("right", "both"))
        n_int = int((state == STATE_VALUE).sum())
        n_bad = int(bad.sum())
    elif _is_numeric_series(series):
        null = series.isna().to_numpy()
        if pd.api.types.is_integer_dtype(series):
            lo = np.full(n, np.nan)
            present = ~null
            ints = series[present].to_numpy()
            if ints.size and (int(ints.max()) > _EXACT_LIMIT or int(ints.min()) < -_EXACT_LIMIT):
                hi_vals, rem = _split_exact(ints.tolist())
                lo[present] = hi_vals
                lo2[present] = rem
            else:
                lo[present] = ints.astype(float)
            bad = np.zeros(n, dtype=bool)
        else:
            lo = series.to_numpy(dtype=float, na_value=np.nan)
            bad = ~null & ~np.isfinite(lo)
        state = np.where(null, STATE_NULL, np.where(bad, STATE_BAD, STATE_VALUE)).astype(np.int8)
        lo = np.where(state == STATE_VALUE, lo, np.nan) + 0.0
        hi = lo.copy()
        hi2 = lo2.copy()
        loi, hii = np.ones(n, dtype=bool), np.ones(n, dtype=bool)
        n_int = 0
        n_bad = int(bad.sum())
    else:
        codes, uniques = pd.factorize(series.astype(object), use_na_sentinel=True)
        parsed = [_parse_numeric_object(u, wildcard_tokens) for u in uniques]
        nan_row = (STATE_NULL, np.nan, np.nan, True, True, "value", 0.0, 0.0)
        parsed = parsed + [nan_row]
        st = np.array([p[0] for p in parsed], dtype=np.int8)
        plo = np.array([p[1] for p in parsed])
        phi = np.array([p[2] for p in parsed])
        pli = np.array([p[3] for p in parsed])
        phi_i = np.array([p[4] for p in parsed])
        pk = np.array([p[5] for p in parsed], dtype=object)
        plo2 = np.array([p[6] for p in parsed])
        phi2 = np.array([p[7] for p in parsed])
        idx = np.where(codes < 0, len(uniques), codes)
        state, lo, hi, loi, hii = st[idx], plo[idx], phi[idx], pli[idx], phi_i[idx]
        lo2, hi2 = plo2[idx], phi2[idx]
        kind = pk[idx]
        n_int = int((kind == "interval").sum())
        bad = kind == "bad"
        n_bad = int(bad.sum())
        # unparsable text is INVALID, which is not the same as missing
        state = np.where(bad, STATE_BAD, state).astype(np.int8)
    mid = _finite_mid(lo, hi, state)
    if bin_width:
        lo, hi, hii, lo2, hi2 = _bin_points(state, lo, hi, loi, hii, lo2, hi2, float(bin_width))
    return state, lo, hi, loi, hii, lo2, hi2, n_int, n_bad, mid


def encode_qid(
    spec: QIDSpec, source: pd.Series, release: pd.Series, cfg: CVPLConfig
) -> QIDEncoding:
    """Encode one QID on both sides; decides ``applicable`` (see module doc)."""
    kind = resolve_kind(spec, source)
    wild = set(cfg.wildcard_tokens)
    if kind == "numeric":
        s_state, s_lo, s_hi, s_loi, s_hii, s_lo2, s_hi2, s_int, s_bad, s_mid = _numeric_arrays(
            source, wild, spec.bin_width)
        r_state, r_lo, r_hi, r_loi, r_hii, r_lo2, r_hi2, r_int, r_bad, r_mid = _numeric_arrays(
            release, wild, spec.bin_width)
        enc = QIDEncoding(
            name=spec.name, kind="numeric", state_s=s_state, state_r=r_state,
            lo_s=s_lo, hi_s=s_hi, loi_s=s_loi, hii_s=s_hii,
            lo_r=r_lo, hi_r=r_hi, loi_r=r_loi, hii_r=r_hii,
            lo2_s=s_lo2, hi2_s=s_hi2, lo2_r=r_lo2, hi2_r=r_hi2,
            exact=bool(np.any(s_lo2) or np.any(s_hi2) or np.any(r_lo2) or np.any(r_hi2)),
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
            "source_unparsable": int(s_bad),
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
    outside = val_r & ~np.isin(t_r, list(source_vals)) if len(source_vals) else val_r.copy()
    unmatched_rows = int(outside.sum())
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
        nodes = set(parent) | set(parent.values())
        # A generalized value is interpretable only if the hierarchy names it. A non-empty
        # mapping that never mentions the released values explains nothing.
        uncovered = int(sum(1 for v in set(t_r[outside].tolist()) if v not in nodes)) if unmatched_rows else 0
        enc.stats["release_values_not_in_hierarchy"] = uncovered
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
        if uncovered and spec.generalized is not False:
            enc.applicable = False
            enc.limitation = "generalized_values_not_in_hierarchy"
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


def _bits(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=1e308, neginf=-1e308).view(np.int64)


def _plain_keys(enc: QIDEncoding, side: str) -> List[np.ndarray]:
    """Int64 key arrays (comparable across the two sides) identifying a plain QID value."""
    state = enc.state_s if side == "s" else enc.state_r
    if enc.kind == "categorical":
        key = (enc.code_s if side == "s" else enc.code_r).copy()
        key[state == STATE_NULL] = -1
        key[state == STATE_WILD] = -2
        return [key]
    lo = enc.lo_s if side == "s" else enc.lo_r
    key = _bits(lo).copy()
    key[state == STATE_NULL] = _NULL_KEY
    key[state == STATE_WILD] = _NULL_KEY + 1
    keys = [key]
    if enc.exact:
        lo2 = enc.lo2_s if side == "s" else enc.lo2_r
        keys.append(_bits(lo2).copy())
    return keys


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
        _bits(lo), _bits(hi),
        loi.astype(np.int64) * 2 + hii.astype(np.int64),
        state.astype(np.int64),
    ]
    if enc.exact:
        lo2, hi2 = (enc.lo2_s, enc.hi2_s) if side == "s" else (enc.lo2_r, enc.hi2_r)
        parts += [_bits(lo2), _bits(hi2)]
    return _combine(parts)


def _group_compat(enc: QIDEncoding, rep_s: np.ndarray, rep_r: np.ndarray, null_policy: str) -> np.ndarray:
    """Boolean ``(len(rep_s), len(rep_r))`` admissibility for one QID.

    This is THE admissibility predicate: planning, scanning and the union of rungs all use
    it (or the row filter that is exactly its single-row form), so they cannot disagree.
    """
    ss = enc.state_s[rep_s][:, None]
    sr = enc.state_r[rep_r][None, :]
    wild = (ss == STATE_WILD) | (sr == STATE_WILD)
    null_any = (ss == STATE_NULL) | (sr == STATE_NULL)
    if null_policy == "wildcard":
        wild = wild | null_any
        nulls_equal = np.zeros(wild.shape, dtype=bool)
    elif null_policy == "equal_null":
        nulls_equal = (ss == STATE_NULL) & (sr == STATE_NULL)
    else:  # incompatible: a missing value on either side admits nothing, wildcard or not
        nulls_equal = np.zeros(wild.shape, dtype=bool)
        wild = wild & ~null_any
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
    if enc.exact:
        lo2_s, hi2_s = enc.lo2_s[rep_s][:, None], enc.hi2_s[rep_s][:, None]
        lo2_r, hi2_r = enc.lo2_r[rep_r][None, :], enc.hi2_r[rep_r][None, :]

        def below(a, a2, b, b2, closed):  # a < b, or a == b when both ends are closed
            return (a < b) | ((a == b) & ((a2 < b2) | ((a2 == b2) & closed)))

        a = below(lo_s, lo2_s, hi_r, hi2_r, loi_s & hii_r)
        b = below(lo_r, lo2_r, hi_s, hi2_s, loi_r & hii_s)
    else:
        # [lo_s, hi_s] overlaps [lo_r, hi_r]: each lower end is below the other's upper end
        a = (lo_s < hi_r) | ((lo_s == hi_r) & loi_s & hii_r)
        b = (lo_r < hi_s) | ((lo_r == hi_s) & loi_r & hii_s)
    # an invalid value is admissible with nothing (it is not "missing" and not a wildcard match)
    not_bad = (ss != STATE_BAD) & (sr != STATE_BAD)
    return (wild | nulls_equal | (both_value & a & b)) & not_bad


@dataclass
class _Part:
    """One partition of a rung: source/release rows sorted by group, group offsets, and the
    special QIDs whose pairwise compatibility is computed on demand (never stored)."""

    s_rows: np.ndarray
    s_off: np.ndarray
    r_rows: np.ndarray
    r_off: np.ndarray
    rep_s: Optional[np.ndarray] = None
    rep_r: Optional[np.ndarray] = None
    special: Tuple[str, ...] = ()


@dataclass
class RungPlan:
    """Candidate structure of one ladder rung (no pair and no pairwise matrix is stored)."""

    qids: Tuple[str, ...]
    applicable: bool = True
    reason: Optional[str] = None
    n_source: int = 0
    n_release: int = 0
    n_special: int = 0
    n_plain: int = 0
    _parts: List[_Part] = field(default_factory=list)
    comparisons: int = 0
    group_pairs: int = 0
    n_blocks: int = 0
    #: planning stopped because the comparison budget was exceeded (``comparisons`` is then a
    #: lower bound of the real count)
    exceeded: bool = False
    #: per source row: the partition / group it belongs to (-1: no candidate group) and how
    #: many candidates its group has
    row_part: Optional[np.ndarray] = None
    row_grp: Optional[np.ndarray] = None
    row_ncand: Optional[np.ndarray] = None
    _encs: Optional[Dict[str, QIDEncoding]] = field(default=None, repr=False)
    _null_policy: str = "equal_null"

    def _compat_row(self, part: _Part, g0: int, g1: int) -> np.ndarray:
        """Compatibility of groups ``g0:g1`` of the source side with every release group."""
        compat = np.ones((g1 - g0, len(part.rep_r)), dtype=bool)
        for q in part.special:
            compat &= _group_compat(self._encs[q], part.rep_s[g0:g1], part.rep_r, self._null_policy)
        return compat

    def group_rows(self, p: int, g: int) -> np.ndarray:
        part = self._parts[p]
        return part.s_rows[part.s_off[g]:part.s_off[g + 1]]

    def candidates(self, p: int, g: int) -> np.ndarray:
        """Released rows admissible for every source row of group ``g`` of partition ``p``."""
        part = self._parts[p]
        if part.rep_s is None:
            return part.r_rows
        cols = np.flatnonzero(self._compat_row(part, g, g + 1)[0])
        if len(cols) == 0:
            return part.r_rows[:0]
        if len(cols) == len(part.r_off) - 1:
            return part.r_rows
        return np.concatenate([part.r_rows[part.r_off[c]:part.r_off[c + 1]] for c in cols])

    def blocks(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(source_rows, candidate_rows)`` blocks; every source row of a block
        shares the same candidate set."""
        for p, part in enumerate(self._parts):
            for g in range(len(part.s_off) - 1):
                cand = self.candidates(p, g)
                if len(cand):
                    yield self.group_rows(p, g), cand


def plan_rung(
    qids: Sequence[str], encs: Dict[str, QIDEncoding], n_source: int, n_release: int, cfg: CVPLConfig,
) -> RungPlan:
    plan = RungPlan(qids=tuple(qids), n_source=n_source, n_release=n_release)
    plan._encs, plan._null_policy = encs, cfg.null_policy
    bad = [q for q in qids if not encs[q].applicable]
    if bad:
        plan.applicable = False
        plan.reason = " ".join(limitation_message(q, encs[q].limitation) for q in bad)
        return plan
    plan.row_part = np.full(n_source, -1, dtype=np.int32)
    plan.row_grp = np.full(n_source, -1, dtype=np.int32)
    plan.row_ncand = np.zeros(n_source, dtype=np.int64)
    # rows that no pair can ever be admissible for: an invalid value is admissible with
    # nothing; a missing value with nothing under the 'incompatible' null policy
    ok_s = np.ones(n_source, dtype=bool)
    ok_r = np.ones(n_release, dtype=bool)
    for q in qids:
        ok_s &= encs[q].state_s != STATE_BAD
        ok_r &= encs[q].state_r != STATE_BAD
        if cfg.null_policy == "incompatible":
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
        keys_by_q = [(_plain_keys(encs[q], "s"), _plain_keys(encs[q], "r")) for q in plain]
        keys = []
        for ks, kr in keys_by_q:
            for a_, b_ in zip(ks, kr):
                keys.append(np.concatenate([a_[rows_s], b_[rows_r]]))
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
        p_idx = len(plan._parts)
        if not special:
            total_cmp += len(s_rows) * len(r_rows)
            plan._parts.append(_Part(s_rows, np.array([0, len(s_rows)]), r_rows, np.array([0, len(r_rows)])))
            plan.row_part[s_rows] = p_idx
            plan.row_grp[s_rows] = 0
            plan.row_ncand[s_rows] = len(r_rows)
            plan.n_blocks += 1
            if total_cmp > cfg.max_comparisons:
                plan.exceeded = True
                break
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
        part = _Part(
            s_rows, s_off, r_rows, r_off,
            rep_s=s_rows[s_off[:-1]], rep_r=r_rows[r_off[:-1]], special=tuple(special),
        )
        r_sizes = np.diff(r_off)
        s_sizes = np.diff(s_off)
        # Streamed: a tile of group rows at a time, never the whole n_sg x n_rg matrix, and the
        # comparison budget is checked as the tiles arrive.
        step = max(1, _TILE_CELLS // max(1, n_rg))
        for g0 in range(0, n_sg, step):
            g1 = min(n_sg, g0 + step)
            compat = plan._compat_row(part, g0, g1)
            per_group = compat @ r_sizes
            total_cmp += int(per_group @ s_sizes[g0:g1])
            for g in range(g0, g1):
                rows_g = s_rows[s_off[g]:s_off[g + 1]]
                plan.row_part[rows_g] = p_idx
                plan.row_grp[rows_g] = g
                plan.row_ncand[rows_g] = per_group[g - g0]
            if total_cmp > cfg.max_comparisons:
                plan.exceeded = True
                break
        plan._parts.append(part)
        plan.n_blocks += n_sg
        if plan.exceeded:
            break
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
