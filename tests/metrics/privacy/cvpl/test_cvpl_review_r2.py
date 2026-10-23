"""Regression tests for the second adversarial review of the CVPL implementation (CDX-CVPL-R2-n).

Every test reproduces a reviewer fixture (``codex_scratch/cvpl_r2``: final_probes,
operation_probes, rewrite_oracle, extended_paths, memory_encoding, tiny_uncertainty) with the
reviewer's expected numbers. Where the reviewer accepted a deliberately loose bound the test
asserts containment of the reviewer's feasible value, not equality.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

GT = {"source_id_column": "id", "release_id_column": "id", "scope": "complete"}
BASE = dict(
    qids=[{"name": "g", "generalized": False}], auxiliary=["x"], qid_as_features=False,
    ground_truth=GT, seed=0, min_components=1, max_components=1,
)
B = 2 ** 53


def frame(g, x, ids):
    return pd.DataFrame(dict(g=g, x=x, id=ids))


_TRAIN = frame(["A", "A"], [-1.0, 1.0], ["a", "b"])
PROJ = run_cvpl(_TRAIN, _TRAIN, CVPLConfig.from_dict(BASE)).projection


def evaluate(s, r, params=None, projection=PROJ, **kw):
    cfg = CVPLConfig.from_dict({**BASE, "fit_strategy": "frozen", **(params or {})})
    return run_cvpl(s, r, cfg, frozen_projection=projection, **kw).metrics


def bounds(m, key):
    """(lower, upper) of a measure whatever its status."""
    st = m[key + "_status"]
    if st["status"] == "measured":
        return m[key], m[key]
    return st["lower"], st["upper"]


# ======================================================================================
# CDX-CVPL-R2-4: exact integer endpoints and bin identities everywhere
# ======================================================================================
@pytest.mark.parametrize(
    "source,release,extra,lr",
    [
        # Codex: source B, release interval [B+1, B+2] -> incompatible
        (B, f"[{B + 1},{B + 2}]", {}, 0.0),
        # Codex: source B+1, release [B+1, B+1] -> compatible
        (B + 1, f"[{B + 1},{B + 1}]", {}, 1.0),
        # Codex: source B+1, release pd.Interval(B, B+1, closed="both") -> compatible
        (B + 1, pd.Interval(B, B + 1, closed="both"), {}, 1.0),
        # Codex: source B, release B+1, bin_width=1 -> different bins
        (B, B + 1, {"bin_width": 1}, 0.0),
        # neighbours of the same width-2 bin / of different width-2 bins
        (B + 2, B + 3, {"bin_width": 2}, 1.0),
        (B + 1, B + 2, {"bin_width": 2}, 0.0),
        (B, B, {"bin_width": 1}, 1.0),
        # open-ended and half-open string intervals keep every digit
        (B + 1, f"{B + 1}+", {}, 1.0),
        (B, f"{B + 1}+", {}, 0.0),
        (B + 1, f"({B + 1},{B + 5}]", {}, 0.0),
        (B + 2, f"({B + 1},{B + 5}]", {}, 1.0),
    ],
)
def test_cdx_cvpl_r2_4_exact_integer_endpoints_intervals_and_bins(source, release, extra, lr):
    s, r = frame([source], [1.0], ["a"]), frame([release], [1.0], ["a"])
    m = evaluate(s, r, dict(qids=[dict(name="g", kind="numeric", **extra)]))
    assert m["cvpl_lr"] == lr
    assert m["cvpl_id_uniform_gt"] == lr  # the single candidate is the right person


def test_cdx_cvpl_r2_4_interval_dtype_column_with_large_integers():
    s = frame([B + 1], [1.0], ["a"])
    r = frame(pd.arrays.IntervalArray.from_arrays([B + 1], [B + 1], closed="both"), [1.0], ["a"])
    assert evaluate(s, r, dict(qids=[dict(name="g", kind="numeric")]))["cvpl_lr"] == 1.0
    r = frame(pd.arrays.IntervalArray.from_arrays([B + 2], [B + 3], closed="both"), [1.0], ["a"])
    assert evaluate(s, r, dict(qids=[dict(name="g", kind="numeric")]))["cvpl_lr"] == 0.0


def test_cdx_cvpl_r2_4_union_counts_a_pair_once_with_binned_large_integers():
    """Codex ``bin_same_value_union``: one source, one release, both g=B, h=A, width 1, ladder
    [[g],[h]]. The explicit union contains ONE pair; the bin's float upper end used to collapse
    onto its lower end, so the overlap predicate rejected the pair and rung 2 counted it again."""
    s, r = frame([B], [1.0], ["a"]), frame([B], [1.0], ["a"])
    s["h"], r["h"] = "A", "A"
    params = dict(
        qids=[dict(name="g", kind="numeric", bin_width=1), dict(name="h", generalized=False)],
        blocking_ladder=[["g"], ["h"]],
    )
    m = evaluate(s, r, params)
    assert m["headline_basis"] == "union_of_rungs"
    assert m["n_compared"] == 1  # the pair, once
    assert m["cvpl_lr"] == 1.0
    assert m["cvpl_id_uniform_subjects_gt"] == 1.0  # was 0.5
    assert m["cvpl_id_uniform_rows_gt"] == 1.0
    assert m["cvpl_top1_success_gt"] == 1.0  # was 0.5


@pytest.mark.parametrize("pair", [(B, B + 1, 1), (B + 1, B + 1, 1), (B + 1, B + 2, 2), (B + 3, B + 2, 2),
                                  (3, 4, 2), (3.5, 3.6, 1), (-5, -4, 3)])
def test_cdx_cvpl_r2_4_plain_key_partition_equals_the_shared_predicate(pair):
    """The optimised plain-key partition and the pairwise predicate used for the union
    exclusion must say the same thing about every pair."""
    from pamola_core.metrics.privacy.cvpl.compat import _group_compat, _plain_keys, encode_qid
    from pamola_core.metrics.privacy.cvpl.config import QIDSpec

    a, b, w = pair
    cfg = CVPLConfig.from_dict({**BASE, "qids": [dict(name="g", kind="numeric", bin_width=w)]})
    enc = encode_qid(cfg.qids[0], pd.Series([a, b, a], dtype=object), pd.Series([a, b, b], dtype=object), cfg)
    ks, kr = _plain_keys(enc, "s"), _plain_keys(enc, "r")
    rows = np.arange(3)
    same_key = np.ones((3, 3), dtype=bool)
    for x, y in zip(ks, kr):
        same_key &= x[:, None] == y[None, :]
    assert (same_key == _group_compat(enc, rows, rows, "equal_null")).all()


# ======================================================================================
# operation-level helpers
# ======================================================================================
def op_run(tmp_path, s, r, name="run", **params):
    from pamola_core.metrics import CVPLLinkageOperation
    from pamola_core.utils.ops.op_data_source import DataSource

    d = tmp_path / name
    d.mkdir(exist_ok=True)
    op = CVPLLinkageOperation(**{**BASE, **params})
    out = op.execute(DataSource(dataframes={"s": s, "r": r}), d, None,
                     original_dataset_name="s", transformed_dataset_name="r")
    return out, d


def _all_text(d, out):
    chunks = [json.dumps(out.metrics, default=str)]
    for p in d.rglob("*"):
        if p.is_file():
            chunks.append(p.read_bytes().decode("utf-8", errors="ignore"))
    return "\n".join(chunks)


# ======================================================================================
# CDX-CVPL-R2-3: the projection fingerprint covers the ORDERED encoded feature layout
# ======================================================================================
def test_cdx_cvpl_r2_3_feature_order_is_part_of_the_fingerprint_and_the_cache(tmp_path):
    t = pd.DataFrame(dict(g=["A"] * 4, x=[1.0, -1.0, 0.0, 0.0], y=[0.0, 0.0, 1.0, -1.0], id=list("abcd")))
    ps = [run_cvpl(t, t, CVPLConfig.from_dict({**BASE, "auxiliary": aux})).projection for aux in (["x", "y"], ["y", "x"])]
    assert list(ps[0].kinds) == ["x", "y"] and list(ps[1].kinds) == ["y", "x"]
    assert ps[0].components.tolist() == ps[1].components.tolist() == [[0.0], [1.0]]  # same arrays
    assert ps[0].fingerprint() != ps[1].fingerprint()  # different physical attributes
    s = pd.DataFrame(dict(g=["A"], x=[1.0], y=[1.0], id=["a"]))
    r = pd.DataFrame(dict(g=["A"], x=[1.0], y=[-1.0], id=["a"]))
    kw = dict(auxiliary=["x", "y"], fit_strategy="frozen")
    first, _ = op_run(tmp_path, s, r, "ord", frozen_projection=ps[0], use_cache=True, **kw)
    second, _ = op_run(tmp_path, s, r, "ord", frozen_projection=ps[1], use_cache=True, **kw)
    fresh, _ = op_run(tmp_path, s, r, "ord_fresh", frozen_projection=ps[1], use_cache=False, **kw)
    assert first.metrics["cvpl_lr"] == 0.0
    assert fresh.metrics["cvpl_lr"] == 1.0
    assert second.metrics["cvpl_lr"] == 1.0  # was the cached 0


def test_cdx_cvpl_r2_3_fingerprint_serialisation_is_unambiguous():
    from pamola_core.metrics.privacy.cvpl.projection import CVPLProjection, ProjectionInfo

    def proj(cats):
        info = ProjectionInfo(2, 1, None, 1, "frozen", ["c"], {}, {})
        return CVPLProjection(
            kinds={"c": "categorical"}, num_mean={}, num_scale={}, categories={"c": cats},
            has_other={"c": False}, pca_mean=np.zeros(len(cats)), components=np.ones((len(cats), 1)), info=info)

    # a joined string cannot tell ["ab", "c"] from ["a", "bc"]
    assert proj(["ab", "c"]).fingerprint() != proj(["a", "bc"]).fingerprint()
    assert proj(["a", "b"]).fingerprint() == proj(["a", "b"]).fingerprint()


# ======================================================================================
# CDX-CVPL-R2-5: custom wildcard tokens are input values and are never published
# ======================================================================================
CANARY_TOKEN = "RAW_WILDCARD_R2_BOB_1966"


def test_cdx_cvpl_r2_5_wildcard_token_is_absent_from_every_surface(tmp_path, caplog):
    s = frame([CANARY_TOKEN, "ordinary"], [-1.0, 1.0], ["RAW_GT_R2_PERSON_001", "another"])
    r = s.copy()
    with caplog.at_level("DEBUG"):
        out, d = op_run(tmp_path, s, r, "wild", wildcard_tokens=["*", CANARY_TOKEN], use_cache=True)
    assert str(out.status).endswith("SUCCESS")
    text = _all_text(d, out)
    assert CANARY_TOKEN not in text
    assert CANARY_TOKEN not in "\n".join(rec.getMessage() for rec in caplog.records)
    params = out.metrics["parameters"]
    assert "wildcard_tokens" not in params
    assert params["wildcard_token_count"] == 2 and len(params["wildcard_tokens_digest"]) == 64


def test_cdx_cvpl_r2_5_digest_keeps_the_cache_identity(tmp_path):
    s, r = frame(["X1"], [1.0], ["a"]), frame(["Y"], [1.0], ["a"])
    first, _ = op_run(tmp_path, s, r, "tok", wildcard_tokens=["*", "X1"], fit_strategy="frozen",
                      frozen_projection=PROJ, use_cache=True)
    second, _ = op_run(tmp_path, s, r, "tok", wildcard_tokens=["*", "X2"], fit_strategy="frozen",
                       frozen_projection=PROJ, use_cache=True)
    assert first.metrics["cvpl_lr"] == 1.0  # X1 is a wildcard: compatible with Y
    assert second.metrics["cvpl_lr"] == 0.0  # X1 is an ordinary label: not Y
