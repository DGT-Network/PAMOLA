"""CDX-CVPL-R2-9: an independent per-rung oracle (vendored, seeded, small) and two guards.

The oracle uses no production helper: explicit candidate-pair sets, scalar geometry and, for the
soundness check, an enumeration of EVERY completion of the undefined cosines (one score per pair,
shared by all rungs). It was written by the reviewer for the R2 norm (cvpl_lr on the union of
the rungs; identification measures per rung with the headline = largest over the rungs).
Reviewer result on 360 cases: 0 defects, all bounds sound; this file replays the first cases.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

GT = {"source_id_column": "id", "release_id_column": "id", "scope": "complete"}
BASE = dict(qids=[{"name": "g", "generalized": False}], auxiliary=["x"], qid_as_features=False,
            ground_truth=GT, seed=0, min_components=1, max_components=1)
EPS = 1e-10
ID = ["cvpl_id_uniform_subjects_gt", "cvpl_id_uniform_rows_gt", "cvpl_top1_success_gt", "precision_at_tau"]


def frame(g, x, ids):
    return pd.DataFrame(dict(g=g, x=x, id=ids))


_TRAIN = frame(["A", "A"], [-1.0, 1.0], ["a", "b"])
PROJ = run_cvpl(_TRAIN, _TRAIN, CVPLConfig.from_dict(BASE)).projection


def evaluate(s, r, params=None, force=False):
    cfg = CVPLConfig.from_dict({**BASE, "fit_strategy": "frozen", **(params or {})})
    return run_cvpl(s, r, cfg, frozen_projection=PROJ, force_union=force).metrics


def intervals(m, key):
    st = m.get(key + "_status", {})
    if st.get("status") == "measured":
        return m[key], m[key]
    return st.get("lower"), st.get("upper")


def atom(a, b, policy, kind):
    na, nb = a is None, b is None
    if policy == "incompatible" and (na or nb):
        return False
    if policy == "wildcard" and (na or nb):
        return True
    if a == "*" or b == "*":
        return True
    if na or nb:
        return na and nb
    return a == b if kind != "numeric" else (a == b)


def pair_state(s, r, i, j):
    a, b = float(s.x.iloc[i]), float(r.x.iloc[j])
    if a == 0 or b == 0:
        return 1.0 if a == b else None  # exact fallback / undefined cosine
    return 1.0 if a * b > 0 else -1.0


def explicit(s, r, sets, tau=0.9, tol=1e-9):
    """measure -> (lower, upper) over the completions, with the documented rules (loose top-1)."""
    rec = []
    for i, js in enumerate(sets):
        known, unknown = {}, []
        for j in js:
            v = pair_state(s, r, i, j)
            if v is None:
                unknown.append(j)
            else:
                known[j] = v
        passed = {j: v for j, v in known.items() if v >= tau}
        own = s.id.iloc[i]
        vals = []
        for bits in itertools.product([False, True], repeat=len(unknown)):
            chosen = set(passed) | {j for j, b in zip(unknown, bits) if b}
            subs = {r.id.iloc[j] for j in chosen}
            vals.append((float(bool(chosen)), float(own in subs) / len(subs) if subs else 0.0,
                         sum(r.id.iloc[j] == own for j in chosen) / len(chosen) if chosen else 0.0))
        vals = np.array(vals)
        tc = {r.id.iloc[j] for j, v in passed.items() if v >= max(passed.values()) - tol} if passed else set()
        top = float(own in tc) / len(tc) if tc else 0.0
        own_unk = any(r.id.iloc[j] == own for j in unknown)
        rec.append(dict(lo=vals.min(0), hi=vals.max(0), top=top, top_lo=0.0 if unknown else top,
                        top_hi=1.0 if (own_unk or (unknown and own in tc)) else top,
                        linked=bool(passed), unk=bool(unknown)))
    lo, hi = np.mean([v["lo"] for v in rec], axis=0), np.mean([v["hi"] for v in rec], axis=0)
    out = {"cvpl_lr": (lo[0], hi[0]), "cvpl_id_uniform_subjects_gt": (lo[1], hi[1]),
           "cvpl_id_uniform_rows_gt": (lo[2], hi[2]),
           "cvpl_top1_success_gt": (np.mean([v["top_lo"] for v in rec]), np.mean([v["top_hi"] for v in rec]))}
    linked = [v for v in rec if v["linked"]]
    unlinked_unknown = [v for v in rec if not v["linked"] and v["unk"]]
    unc = any(v["top_lo"] != v["top_hi"] for v in linked)
    if not unlinked_unknown and not unc:
        p = sum(v["top"] for v in linked) / len(linked) if linked else None
        out["precision_at_tau"] = (p, p)
    else:
        lower = sum(v["top_lo"] for v in linked) / (len(linked) + len(unlinked_unknown))
        upper = (min(1.0, (sum(v["top_hi"] for v in linked) + sum(v["top_hi"] for v in unlinked_unknown))
                     / len(linked)) if linked else 1.0)
        out["precision_at_tau"] = (lower, upper)
    return out


def headline(per_rung, cands):
    live = [(q, k, per_rung[q][k]) for q, k in cands if per_rung[q][k][0] is not None]
    if not live:
        return None
    return max(x[2][0] for x in live), max(x[2][1] for x in live)


def record_outcomes(s, r, i, families):
    """Reachable per-family (declared, subjects, rows, top1) tuples over every completion."""
    allj = sorted(set().union(*[f[i] for f in families]))
    own = s.id.iloc[i]
    known = {j: pair_state(s, r, i, j) for j in allj}
    unk = [j for j in allj if known[j] is None]
    base = {j: 0 for j in allj if known[j] == 1.0}
    outs = set()
    for assign in itertools.product([None] + list(range(len(unk) + 1)), repeat=len(unk)):
        score = dict(base)
        for j, lvl in zip(unk, assign):
            if lvl is not None:
                score[j] = lvl
        tup = []
        for f in families:
            acc = [j for j in f[i] if j in score]
            if not acc:
                tup.append((0, 0.0, 0.0, 0.0))
                continue
            subs = {r.id.iloc[j] for j in acc}
            best = min(score[j] for j in acc)
            tc = {r.id.iloc[j] for j in acc if score[j] == best}
            tup.append((1, float(own in subs) / len(subs), sum(r.id.iloc[j] == own for j in acc) / len(acc),
                        float(own in tc) / len(tc)))
        outs.add(tuple(tup))
    return outs


def exact_ranges(s, r, rungsets, union):
    n, R = len(s), len(rungsets)
    states = {tuple([0.0] * ((R + 1) * 4))}
    for i in range(n):
        outs = record_outcomes(s, r, i, rungsets + [union])
        states = {tuple(round(a + b, 12) for a, b in zip(st, [x for t in o for x in t]))
                  for st in states for o in outs}
    agg = {k: [math.inf, -math.inf] for k in ["cvpl_lr", *ID, "cvpl_id_uniform_gt"]}
    prec_undef = False
    for st in states:
        v = lambda q, c: st[q * 4 + c] / n  # noqa: E731
        vals = {"cvpl_lr": v(R, 0), "cvpl_id_uniform_subjects_gt": max(v(q, 1) for q in range(R)),
                "cvpl_id_uniform_rows_gt": max(v(q, 2) for q in range(R)),
                "cvpl_top1_success_gt": max(v(q, 3) for q in range(R))}
        vals["cvpl_id_uniform_gt"] = max(vals["cvpl_id_uniform_subjects_gt"], vals["cvpl_id_uniform_rows_gt"])
        pr = [st[q * 4 + 3] / st[q * 4] for q in range(R) if st[q * 4] > 0]
        if pr:
            vals["precision_at_tau"] = max(pr)
        else:
            prec_undef = True
        for key, x in vals.items():
            agg[key] = [min(agg[key][0], x), max(agg[key][1], x)]
    return agg, prec_undef


def close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= EPS)


@pytest.mark.parametrize("k", range(36))
def test_cdx_cvpl_r2_9_engine_matches_the_per_rung_oracle_and_its_bounds_are_sound(k):
    rng = np.random.default_rng(220192 + k)
    n, m = 4, 5
    policy = ["equal_null", "incompatible", "wildcard"][k % 3]
    kind = "numeric" if k % 2 else "categorical"
    domain = [None, "*", 0, 1, 2] if kind == "numeric" else [None, "*", "A", "B", "C"]
    a = [domain[x] for x in rng.integers(0, len(domain), n)]
    b = [domain[x] for x in rng.integers(0, len(domain), m)]
    s = frame(pd.Series(a, dtype=object), rng.choice([-1.0, 0.0, 1.0], n), np.arange(n))
    r = frame(pd.Series(b, dtype=object), rng.choice([-1.0, 0.0, 1.0], m),
              rng.integers(0, 6, m) if k % 4 else np.arange(m))
    s["h"], r["h"] = rng.choice(["X", "Y"], n), rng.choice(["X", "Y"], m)
    ladder = [["g"], ["h"]] if k % 3 == 0 else ([["g", "h"], ["g"]] if k % 3 == 1 else [["h"], ["g", "h"]])
    params = dict(qids=[dict(name="g", kind=kind, generalized=False), dict(name="h", generalized=False)],
                  blocking_ladder=ladder, null_policy=policy, chunk_pairs=[1, 3, 10000][k % 3])
    sets = [[{j for j in range(m) if all(
        atom(s[q].iloc[i], r[q].iloc[j], policy, kind if q == "g" else "categorical") for q in rung)}
        for i in range(n)] for rung in ladder]
    union = [set().union(*(rs[i] for rs in sets)) for i in range(n)]
    got = evaluate(s, r, params, force=k % 2 == 1)

    per_rung = [explicit(s, r, rs) for rs in sets]
    un = explicit(s, r, union)
    exp = {"cvpl_lr": un["cvpl_lr"]}
    R = len(sets)
    for key in ID:
        exp[key] = headline(per_rung, [(q, key) for q in range(R)])
    exp["cvpl_id_uniform_gt"] = headline(
        per_rung, [(q, "cvpl_id_uniform_subjects_gt") for q in range(R)]
        + [(q, "cvpl_id_uniform_rows_gt") for q in range(R)])
    for key, e in exp.items():
        gl, gh = (got["cvpl_lr_lower"], got["cvpl_lr_upper"]) if key == "cvpl_lr" else intervals(got, key)
        if e is None:
            assert gl is None and gh is None, key
            continue
        assert close(gl, e[0]) and close(gh, e[1]), (key, e, (gl, gh))
        if key != "cvpl_lr":
            want = "measured" if abs(e[1] - e[0]) <= EPS else "bounded"
            assert got[key + "_status"]["status"] == want, key
    for ix, ex in enumerate(per_rung):
        meas = got["blocking_ladder_trajectory"][ix]["measures"]
        for key, e in ex.items():
            gl, gh = (meas["cvpl_lr_lower"], meas["cvpl_lr_upper"]) if key == "cvpl_lr" else intervals(meas, key)
            assert close(gl, e[0]) and close(gh, e[1]), (ix, key, e, (gl, gh))

    exact, prec_undef = exact_ranges(s, r, sets, union)
    for key, (xlo, xhi) in exact.items():
        if xlo == math.inf:
            continue
        gl, gh = (got["cvpl_lr_lower"], got["cvpl_lr_upper"]) if key == "cvpl_lr" else intervals(got, key)
        assert gl is not None and gh is not None, key
        assert gl <= xlo + EPS and gh >= xhi - EPS, ("unsound", key, (xlo, xhi), (gl, gh))
    st = got["precision_at_tau_status"]
    if st["status"] == "bounded":
        assert bool(st.get("possibly_undefined", False)) is prec_undef, (st, prec_undef)


def test_cdx_cvpl_r2_9_precision_with_no_known_declared_link_has_lower_bound_zero():
    """The guard: a rung with no known declared link gets precision lower bound 0, not a bound
    taken only over the completions in which the precision is defined."""
    s, r = frame(["A"], [0.0], ["a"]), frame(["A"], [1.0], ["a"])  # the only pair is undefined
    m = evaluate(s, r)
    st = m["precision_at_tau_status"]
    assert st["status"] == "bounded" and st["lower"] == 0.0 and st["upper"] == 1.0
    assert st["possibly_undefined"] is True
    rung = m["blocking_ladder_trajectory"][0]["measures"]["precision_at_tau_status"]
    assert rung["lower"] == 0.0 and rung["possibly_undefined"] is True


def test_cdx_cvpl_r2_9_a_known_declared_link_makes_precision_always_defined():
    s, r = frame(["A", "B"], [-1.0, 0.0], ["a", "b"]), frame(["A", "B"], [-1.0, 1.0], ["a", "b"])
    m = evaluate(s, r)  # record a: known link; record b: undefined cosine
    st = m["precision_at_tau_status"]
    assert st["status"] == "bounded"
    assert "possibly_undefined" not in st
    # two rungs: the headline is undefined only if EVERY rung can be
    s2, r2 = s.copy(), r.copy()
    s2["h"], r2["h"] = ["X", "Y"], ["X", "Y"]
    only_unknown = evaluate(frame(["A"], [0.0], ["a"]).assign(h="X"), frame(["A"], [1.0], ["a"]).assign(h="X"),
                            dict(qids=[dict(name="g", generalized=False), dict(name="h", generalized=False)],
                                 blocking_ladder=[["g"], ["g", "h"]]))
    assert only_unknown["precision_at_tau_status"]["possibly_undefined"] is True
