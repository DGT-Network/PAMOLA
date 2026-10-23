"""Shared fixtures for the CVPL tests.

The hand-checkable cases use ONE numeric feature (``x``) and blocking on ``g``
(``qid_as_features=False``). The representation is then one-dimensional, the
projected vector of a row is ``(x - mean)`` and the cosine of two rows is exactly
``+1`` (same side of the joint mean) or ``-1`` (opposite sides): similarity is a
design choice of the fixture, not an accident of the data. Filler rows in a block
nobody queries (``g='Z'``) move the joint mean to where the case needs it.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from pamola_core.metrics.privacy.cvpl import CVPLConfig, run_cvpl

GT = {"source_id_column": "sid", "release_id_column": "rid"}
#: an exact-only QID: released labels outside the source domain are simply different people
EXACT_G = {"name": "g", "generalized": False}


def source_frame(rows: Sequence[tuple], index_start: int = 0) -> pd.DataFrame:
    """rows: (g, x, subject_id)"""
    g, x, sid = zip(*rows)
    return pd.DataFrame(
        {"g": list(g), "x": [float(v) for v in x], "sid": list(sid)},
        index=range(index_start, index_start + len(rows)),
    )


def release_frame(rows: Sequence[tuple], index_start: int = 1000) -> pd.DataFrame:
    """rows: (g, x, subject_id or None)"""
    g, x, rid = zip(*rows)
    return pd.DataFrame(
        {"g": list(g), "x": [float(v) for v in x], "rid": list(rid)},
        index=range(index_start, index_start + len(rows)),
    )


def anchors(n: int = 6, value: float = -100.0, start: int = 900) -> List[tuple]:
    """Release-only filler in block 'Z' (never a candidate) that pulls the mean down."""
    return [("Z", value - i, f"anchor{start + i}") for i in range(n)]


def joint_mean(source: pd.DataFrame, release: pd.DataFrame, col: str = "x") -> float:
    return float(pd.concat([source[col], release[col]]).mean())


def measure(source: pd.DataFrame, release: pd.DataFrame, **overrides):
    """Run CVPL on the one-dimensional design (tau 0.5, GT columns sid / rid)."""
    params: Dict = dict(
        qids=[EXACT_G], auxiliary=["x"], qid_as_features=False, tau=0.5,
        ground_truth=dict(GT),
    )
    params.update(overrides)
    return run_cvpl(source, release, CVPLConfig.from_dict(params))


def make_df(n: int = 60, seed: int = 1) -> pd.DataFrame:
    """The generator behind the historical fixtures G1-G5 (BEST test suite)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "person_id": [f"p{i}" for i in range(n)],
        "age": rng.integers(20, 70, n),
        "gender": rng.choice(["M", "F"], n),
        "region_level1": rng.choice(["R1", "R2", "R3"], n),
        "income": rng.normal(50_000, 12_000, n).round(2),
    })
