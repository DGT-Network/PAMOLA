# Regulatory evidence map

**Purpose.** Privacy and data-protection assessments of a de-identified dataset are usually organized around three questions: can a person be *singled out* (individualization), can records be *linked* to each other or to other data (linkability / correlation), and can information about a person be *inferred* from the data (inference). This page maps what PAMOLA.CORE measures to those three questions, and, for each measurement, what it does **not** establish.

**What this page is not.** It is not a compliance statement. PAMOLA.CORE does not determine that a dataset is anonymous, that a processing activity complies with GDPR, PIPEDA, HIPAA, the Quebec anonymization regulation or any other regime, or that a release is safe. Those determinations belong to a regulated process that includes organizational controls, legal analysis, threat modelling for the specific release and, where required, expert determination. The measurements below are *technical evidence inputs* to that process.

## How to read a measurement

Every number in the table is a property of **the data you passed in, under the settings you chose** (quasi-identifier set, distance metric, blocking profile, sample, random seed). It says nothing about data you did not include (external datasets an attacker may hold, later releases, other fields), and nothing about an attacker stronger than the one the measurement models. A good result is the absence of a particular signal, not a proof of absence of risk.

## Individualization (singling out)

| Evidence | What it measures | Where | What it does **not** establish |
|---|---|---|---|
| k-anonymity profile (`KAnonymityProfilerOperation`; metrics `min_k`, `mean_k`, `unique_groups`, `unique_percent`, `vulnerable_count`, `vulnerable_percent`) | For each quasi-identifier (QI) combination you list, the size of the groups of records that share the same QI values. `min_k = 1` means at least one record is unique on those fields. | `docs/en/core/profiling/anonymity.md` | That the QI list is complete (an attribute you left out may still single a person out); that group size says anything about the sensitive value (see inference); population uniqueness: it counts uniqueness *in your sample*, not in the population the sample came from. |
| Uniqueness (`PrivacyMetricOperation` with `privacy_metrics=["uniqueness"]`; `identity_disclosure_rate`, `k_anonymity_stats`) | Share of records that are alone in their QI group (`identity_disclosure_rate`), and, for each requested `k`, how many groups are smaller than `k`. | `docs/en/core/metrics/privacy/identity.md` | The same limits as above; it is computed on the transformed dataset only and does not compare with the source. |

## Linkability and correlation

| Evidence | What it measures | Where | What it does **not** establish |
|---|---|---|---|
| Distance to Closest Record (`PrivacyMetricOperation`, `dcr`) | For every record of the transformed dataset, the distance to the nearest record of the source in an encoded feature space. Small distances mean a transformed record almost reproduces a source record. | `docs/en/core/metrics/privacy/distance.md` | That no linkage is possible: distance depends on the feature encoding (categories are encoded as ordered codes), on normalization and on the chosen columns; it does not model an attacker with external data. The text in `interpretation` is a heuristic, not a verdict. |
| Nearest-neighbour distance ratio (`PrivacyMetricOperation`, `nndr`) | Ratio of the distance to the nearest and to the second-nearest source record; values near 0 mean a record is much closer to one source record than to any other. | `docs/en/core/metrics/privacy/neighbor.md` | The same limits as DCR; the thresholds (`threshold`, `at_risk_threshold`, `realistic_threshold`) are yours to choose and are not regulatory limits. |
| Basic linkability (CVPL, `CVPLLinkageOperation`; `cvpl_lr` and its bounds) | The share of source records that have at least one release record that is *compatible* under the declared blocking profile (the quasi-identifiers you consider known to an attacker, with the declared tolerance for generalized values) *and* similar enough in the projected representation (cosine similarity at or above `tau`). Each value carries a status: `measured`, `bounded` (with lower and upper bound), `not_measured` or `not_applicable`. | `docs/en/core/metrics/privacy/cvpl.md` | That a linked pair is the same person (existence of a candidate is not identification); that no linkage exists outside the declared quasi-identifiers and threshold; anything about an optimal attacker: values are the maximum over the declared attacker profiles and strategies only. |
| CVPL identification measures against ground truth (`cvpl_id_uniform_gt`, `cvpl_top1_success_gt`, `precision_at_tau`) | When you supply a ground-truth mapping of release rows to source subjects (evaluation only), how often a declared attack strategy picks the right subject among the candidates. | `docs/en/core/metrics/privacy/cvpl.md` | Anything when the ground truth is partial or absent (the measure is then `not_measured` or reported only for the known subset); the real-world attacker rarely has ground truth. The mapping is used to *evaluate* risk, never as an input to the linkage. |

## Inference

| Evidence | What it measures | Where | What it does **not** establish |
|---|---|---|---|
| l-diversity statistics (`Uniqueness`; `min_l_diversity`, `avg_l_diversity`, `l_diversity_distribution`) | Number of distinct sensitive-attribute values inside each QI group. | `docs/en/core/metrics/privacy/identity.md` | That the values are semantically diverse (two distinct values may reveal the same thing, for example two neighbouring diagnoses); that an attacker cannot infer the value from other columns. |
| t-closeness statistics (`Uniqueness`; `t_closeness`) | Distance (Wasserstein) between the distribution of the sensitive attribute inside each QI group and its distribution in the whole dataset. | `docs/en/core/metrics/privacy/identity.md` | A bound on what an attacker learns; the distance is not scaled to a regulatory threshold, and categorical attributes rely on the encoding. |
| Privacy-model classes (`privacy_models`: k-anonymity, l-diversity, t-closeness) | Library implementations of the models for use in analyses. | `docs/en/core/privacy_models/` | That a dataset satisfying a model parameter is anonymous. |

## Evidence about the trade-off (not privacy evidence)

| Evidence | What it measures | What it does **not** establish |
|---|---|---|
| Fidelity (`FidelityOperation`: Kolmogorov-Smirnov, Kullback-Leibler) | Statistical similarity of distributions between source and transformed data. | Anything about privacy. High fidelity can coexist with high risk and the reverse. |
| Utility (`UtilityMetricOperation`: regression, classification) | How well models trained on the source perform on the transformed data (on a seeded sample above the row budget; the sample is reported in the result). | Anything about privacy; a high score is not evidence of leakage and not evidence of safety. |

## What the transformations are (and are not)

Generalization, suppression, masking, pseudonymization, uniform (non-DP) noise and fake-value substitution are *transformations*. They change the data; they are not evidence. Whether they achieved a particular protection level is answered only by measuring the *result* with the evidence above, for the release and threat model in question. Pseudonymized data remains linkable by design and is not anonymous. See the terminology table in the [README](../../../README.md).

## Putting the evidence together

A defensible assessment typically combines, for the same release:

1. a statement of the threat model (who the attacker is, what auxiliary data they may hold, which fields count as quasi-identifiers);
2. the profile of the *source* (what is risky before transformation) and of the *release* (what remains);
3. the measurements above, with their configuration saved (`config.json` of each operation records the parameters used, with secrets redacted);
4. the fidelity and utility cost of the transformations;
5. organizational and legal analysis outside this software.

PAMOLA.CORE produces items 2 to 4. It does not produce items 1 and 5, and it does not combine them into a conclusion.
