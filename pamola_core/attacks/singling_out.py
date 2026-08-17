"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Singling-out risk (EDPB Criterion 1)
Package: pamola_core.attacks

Singling-out asks: can a single record be isolated from the rest using only
quasi-identifiers? It is the first of the three EDPB anonymisation criteria
(alongside linkability and inference) and, until this module, the one criterion
this package did not cover.

Two complementary views are provided, because they answer different questions.

**Sweep (how bad is it?)** - enumerate every quasi-identifier subset up to
``max_combo``, count records that are alone in their equivalence class, and take
the worst-case subset. Yields ``singling_out_rate``, the headline number, and
the subset that produced it.

**SUDA / MSU (what do I fix?)** - for each record, find the *minimal* attribute
combinations that make it unique (a Minimal Sample Unique). A record unique on
``{age, occupation}`` but not on either field alone carries an MSU of size 2;
smaller MSUs are sharper singling paths. Per-attribute contributions then say
which column to generalise first. A rate alone cannot tell you that.

**Baseline (is the rate meaningful?)** - a uniqueness rate is uninterpretable on
its own: a table with many high-cardinality columns is unique almost by
construction. ``independent_marginals_baseline`` shuffles each column of the
worst-case subset independently, destroying cross-column correlation while
preserving marginals, and reports the rate that arises from the marginals alone.
Risk is the *excess* over that baseline.

Provenance and licence
----------------------
The sweep, baseline and Wilson-interval logic are ported from the PAMOLA.BEST
implementation of ATK-SINGLING-001 (polars -> pandas here). The MSU/SUDA2 engine
is ported from the clean-room implementation in PAMOLA spikes
(``i_piper/roles/msu.py``), which follows the published SUDA2 semantics of
Elliot and Manning-Haglin-Keane. Both sources are PAMOLA-internal and
BSD-compatible; no GPL code (notably not R's ``sdcMicro::suda2``) was consulted
or copied.

NOTE: This module requires 'numpy' and 'pandas'.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from pamola_core.errors.exceptions import FieldNotFoundError, ValidationError

_Z_95 = 1.959963984540054

#: Default combination depth for both the sweep and the MSU search.
#:
#: This is a guard, not a tuning knob. An unbounded default (the column count)
#: is not a slow default, it is a non-terminating one: at 49 candidate columns
#: an exhaustive search is 2**49 subsets.
MAX_COMBO = 4


@dataclass
class RecordRisk:
    """Per-record singling-out risk derived from its Minimal Sample Uniques."""

    index: int
    #: Minimal attribute sets on which this record is unique in the sample.
    msus: List[frozenset] = field(default_factory=list)
    #: Size of the smallest MSU - the sharpest singling path for this record.
    min_msu_size: int = 0
    #: Elliot SUDA score: sum over MSUs of (ATT - |MSU|)!  Smaller MSUs score higher.
    suda_score: float = 0.0


def enumerate_subset_count(n_qid: int, max_combo: int) -> int:
    """Number of QID subsets of size 1..``max_combo`` over ``n_qid`` fields.

    Use this before a sweep to decide whether the search is affordable.
    """
    return sum(math.comb(n_qid, k) for k in range(1, max_combo + 1))


def wilson_ci(p_hat: float, n: int, z: float = _Z_95) -> List[float]:
    """95% Wilson score interval on a proportion, clamped to [0, 1].

    Reported alongside the rate so that a rate measured on 200 records is not
    read with the same confidence as one measured on 200 000.
    """
    if n <= 0:
        return [0.0, 0.0]
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _validate(df: pd.DataFrame, quasi_identifiers: Sequence[str]) -> List[str]:
    qi = list(quasi_identifiers)
    if not qi:
        raise ValidationError("At least one quasi-identifier is required.")
    missing = [c for c in qi if c not in df.columns]
    if missing:
        raise FieldNotFoundError(missing[0], list(df.columns), dataset_name="input data")
    return qi


def singling_out_sweep(
    df: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    max_combo: int = MAX_COMBO,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Sweep QID subsets of size 1..``max_combo`` and report the worst case.

    Worst-case, not average: an attacker uses the subset that works best, so the
    maximum rate is the honest summary. Averaging across subsets would dilute a
    single catastrophic combination with many harmless ones.

    Parameters
    ----------
    df : pd.DataFrame
        Data to assess.
    quasi_identifiers : Sequence[str]
        Columns an adversary is assumed to know.
    max_combo : int
        Maximum subset size. See :data:`MAX_COMBO`.
    threshold : float
        Subsets whose rate exceeds this are listed in ``vulnerable_combinations``.

    Returns
    -------
    dict
        ``singling_out_rate``, ``unique_count``, ``best_subset``,
        ``qid_combinations_tested``, ``vulnerable_combinations``,
        ``vulnerable_records``, ``wilson_ci_95``.
    """
    qi = _validate(df, quasi_identifiers)
    n = len(df)

    best_rate, best_subset, best_unique = -1.0, [], 0
    combos_tested = 0
    vulnerable_combinations: List[Dict[str, Any]] = []
    vulnerable_records: set = set()

    for k in range(1, min(max_combo, len(qi)) + 1):
        for combo in combinations(qi, k):
            subset = list(combo)
            combos_tested += 1
            # duplicated(keep=False) marks every member of a repeated group, so
            # its negation is exactly "alone in its equivalence class". This is
            # the vectorised form of a groupby size == 1 test, without
            # materialising a per-row broadcast of group sizes for every subset.
            singleton_mask = ~df.duplicated(subset=subset, keep=False)
            unique_p = int(singleton_mask.sum())
            rate_p = unique_p / n if n else 0.0

            if rate_p > threshold:
                vulnerable_combinations.append(
                    {
                        "qid_fields": subset,
                        "singling_out_rate": round(rate_p, 6),
                        "unique_records": unique_p,
                    }
                )
            if unique_p:
                vulnerable_records.update(int(i) for i in df.index[singleton_mask])
            if rate_p > best_rate:
                best_rate, best_subset, best_unique = rate_p, subset, unique_p

    rate = max(best_rate, 0.0)
    vulnerable_combinations.sort(key=lambda c: c["singling_out_rate"], reverse=True)
    return {
        "singling_out_rate": rate,
        "unique_count": best_unique,
        "best_subset": best_subset,
        "qid_combinations_tested": combos_tested,
        "vulnerable_combinations": vulnerable_combinations,
        "vulnerable_records": sorted(vulnerable_records),
        "wilson_ci_95": wilson_ci(rate, n),
    }


def independent_marginals_baseline(
    df: pd.DataFrame,
    subset: Sequence[str],
    draws: int = 10,
    seed: int = 42,
) -> float:
    """Expected uniqueness rate when ``subset`` columns are independent.

    Each column is shuffled with its own seed: cross-column correlation is
    destroyed, marginal distributions are preserved. The mean rate over
    ``draws`` permutations is how unique the table would be from its marginals
    alone. Deterministic given ``seed``.

    Compare the measured rate against this: exceeding it is what indicates
    structure an adversary can exploit, rather than mere column cardinality.
    """
    cols = list(subset)
    n = len(df)
    if not cols or n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    rates: List[float] = []
    for _ in range(draws):
        shuffled = pd.DataFrame(
            {c: rng.permutation(df[c].to_numpy()) for c in cols}
        )
        singleton = ~shuffled.duplicated(keep=False)
        rates.append(int(singleton.sum()) / n)
    return sum(rates) / len(rates)


def minimal_sample_uniques(
    df: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    max_size: Optional[int] = None,
) -> Dict[int, List[frozenset]]:
    """Minimal attribute subsets on which each record is unique.

    Returns ``{record_index: [MSU, ...]}``. A subset is *minimal* when no proper
    subset of it is already unique for that record.

    Only records that are unique on the FULL quasi-identifier set can carry an
    MSU, so the search is restricted to them up front. On a table where few
    records are unique that is a real prune, not a micro-optimisation.
    """
    cols = _validate(df, quasi_identifiers)
    att = len(cols)
    max_size = max_size or min(att, MAX_COMBO)

    full_unique = ~df.duplicated(subset=cols, keep=False)
    universe = set(df.index[full_unique])
    if not universe:
        return {}

    msus: Dict[int, List[frozenset]] = {}
    found: Dict[int, List[frozenset]] = {}
    for m in range(1, max_size + 1):
        for combo in combinations(cols, m):
            subset_set = frozenset(combo)
            unique_idx = df.index[~df.duplicated(subset=list(combo), keep=False)]
            for idx in unique_idx:
                if idx not in universe:
                    continue
                prior = found.get(idx)
                # A smaller already-unique subset means this one is not minimal.
                if prior and any(h <= subset_set for h in prior):
                    continue
                msus.setdefault(idx, []).append(subset_set)
                found.setdefault(idx, []).append(subset_set)
    return msus


def suda_scores(
    df: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    max_size: Optional[int] = None,
) -> tuple:
    """SUDA2 per-record risks and per-attribute contributions.

    The SUDA score of an MSU of size ``k`` over ``ATT`` attributes is
    ``(ATT - k)!`` (Elliot): the smaller the minimal unique combination, the
    higher the score, because fewer attributes suffice to isolate the record.
    An attribute's contribution is the summed score of the MSUs containing it -
    which is the practical output, since it ranks columns by how much they
    enable singling-out.

    Returns
    -------
    (List[RecordRisk], Dict[str, float])
        Records sorted by sharpest MSU first, then by descending SUDA score.
    """
    cols = _validate(df, quasi_identifiers)
    att = len(cols)
    msus = minimal_sample_uniques(df, cols, max_size)

    risks: List[RecordRisk] = []
    contribution: Dict[str, float] = {c: 0.0 for c in cols}
    for idx, sets in msus.items():
        score = 0.0
        for s in sets:
            sc = float(math.factorial(att - len(s)))
            score += sc
            for attribute in s:
                contribution[attribute] += sc
        risks.append(
            RecordRisk(
                index=int(idx),
                msus=sets,
                min_msu_size=min(len(s) for s in sets),
                suda_score=score,
            )
        )
    risks.sort(key=lambda r: (r.min_msu_size, -r.suda_score))
    return risks, contribution


def dis_risk(
    df: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    max_size: Optional[int] = None,
) -> float:
    """Fraction of records carrying at least one MSU.

    The population of at-risk records that SUDA surfaces. Distinct from
    ``singling_out_rate``: that is the worst *single subset*, this counts records
    reachable by *any* minimal combination.
    """
    msus = minimal_sample_uniques(df, quasi_identifiers, max_size)
    return len(msus) / len(df) if len(df) else 0.0


class SinglingOutAttack:
    """Singling-out risk assessment (EDPB Criterion 1).

    Combines the worst-case subset sweep with the SUDA2 minimal-unique engine so
    that one call answers both "how exposed is this dataset" and "which columns
    cause it".

    Example
    -------
    >>> attack = SinglingOutAttack(max_combo=3)                    # doctest: +SKIP
    >>> report = attack.evaluate(df, ["age", "sex", "postcode"])   # doctest: +SKIP
    >>> report["singling_out_rate"], report["excess_over_baseline"]  # doctest: +SKIP
    """

    def __init__(
        self,
        max_combo: int = MAX_COMBO,
        threshold: float = 0.0,
        baseline_draws: int = 10,
        seed: int = 42,
    ):
        self.max_combo = max_combo
        self.threshold = threshold
        self.baseline_draws = baseline_draws
        self.seed = seed

    def evaluate(
        self,
        df: pd.DataFrame,
        quasi_identifiers: Iterable[str],
        include_suda: bool = True,
    ) -> Dict[str, Any]:
        """Assess singling-out risk.

        Parameters
        ----------
        df : pd.DataFrame
            Data to assess.
        quasi_identifiers : Iterable[str]
            Columns an adversary is assumed to know. For passported PAMOLA
            datasets these are the ``quasi`` fields of the dataset passport.
        include_suda : bool
            Also run the MSU/SUDA engine. Set False for the rate alone on wide
            tables, where the minimal-unique search is the expensive half.

        Returns
        -------
        dict
            The sweep result, plus ``independent_marginals_baseline``,
            ``excess_over_baseline`` and - when ``include_suda`` - ``dis_risk``,
            ``attribute_contribution`` and ``top_risk_records``.
        """
        qi = list(quasi_identifiers)
        report = singling_out_sweep(df, qi, self.max_combo, self.threshold)

        baseline = independent_marginals_baseline(
            df, report["best_subset"], self.baseline_draws, self.seed
        )
        report["independent_marginals_baseline"] = round(baseline, 6)
        # Excess, not ratio: a ratio is unstable when the baseline approaches
        # zero, which is exactly the low-cardinality case.
        report["excess_over_baseline"] = round(
            report["singling_out_rate"] - baseline, 6
        )

        if include_suda:
            risks, contribution = suda_scores(df, qi, self.max_combo)
            report["dis_risk"] = round(len(risks) / len(df) if len(df) else 0.0, 6)
            report["attribute_contribution"] = {
                k: round(v, 6)
                for k, v in sorted(
                    contribution.items(), key=lambda kv: kv[1], reverse=True
                )
            }
            report["top_risk_records"] = [
                {
                    "index": r.index,
                    "min_msu_size": r.min_msu_size,
                    "suda_score": r.suda_score,
                    "msus": [sorted(s) for s in r.msus],
                }
                for r in risks[:10]
            ]
        return report
