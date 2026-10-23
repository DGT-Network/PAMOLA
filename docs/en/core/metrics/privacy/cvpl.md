# CVPL linkage exposure (`CVPLLinkageOperation`)

**Module:** `pamola_core.metrics.privacy.cvpl` (engine), `pamola_core.metrics.CVPLLinkageOperation` (operation)
**Algorithm:** `cvpl-operator-chain`, version `1.0`

CVPL (Cluster-Vector-Projection Linkage) measures how far a protected release can be
linked back to its source by **one specified attacker**. It is a measurement, not a
verdict: it returns numbers and says what each of them counts. It does not decide
whether a release is "anonymous" or "safe", and there is no pass/fail threshold in it.

> **This is not a legal or regulatory conclusion.** A low number does not prove that
> individuals cannot be identified, and a high one does not prove that they can. The
> numbers describe a particular attacker model on this particular data. Interpreting
> them for a release decision, and the choice of any threshold, belongs to the data
> controller's own policy and context.

It is unrelated to the older `pamola_core.attacks.linkage_attack` (TF-IDF /
TruncatedSVD), which is kept unchanged and measures something else.

## Threat model: what kind of estimate this is

CVPL is a **white-box evaluation**. The auditor has full access: the source data, the
release, the attacker's declared knowledge and - for the ground-truth (GT) measures - the
true correspondence between them. The attacker strategy is chosen by the auditor and
evaluated with that access, so every published number is an **upper, worst-case-oriented
estimate** of what the declared attacker achieves. It is *not* the average over attackers
that a black-box tool such as Anonymeter reports, and it is *not* a Fellegi-Sunter style
match probability. Consequences:

* a number can only go up when the attacker is given more knowledge (more `qids` /
  `auxiliary`), a looser ladder or a lower `tau`; it does not describe a typical adversary;
* where several simple strategies are defined (uniform choice among subjects, among rows)
  the headline is the **larger** of them, so it never falls below what one of them achieves;
* an unknown (undefined cosine) is resolved in the attacker's favour in the *upper* bound
  and the number is published as a bound, never as a point value (see "Statuses");
* the numbers say what **this** attacker model reaches. They say nothing about attackers
  with other background knowledge or a better matcher.

## The attacker

The attacker holds the **source** dataset `D` (or the declared part of it) and sees the
**release** `D'`. For each source record `x_i` the attacker:

1. **Blocks** (`B`): keeps only the released rows whose quasi-identifiers (QIDs) are
   *compatible* with `x_i` (the **admissible** candidates - one predicate, used identically
   for planning, scanning and the union of ladder rungs).
2. **Vectorizes** (`phi`) the declared attributes: numeric columns are z-scored
   (unknown values filled with the mean), categorical columns are one-hot encoded
   (at most `max_categories` columns per attribute; the rarest values share one bucket).
3. **Projects** (`psi`) with PCA, keeping the smallest number of components that reach
   `variance_retained` (between `min_components` and `max_components`).
4. Compares by **cosine similarity** (`s`) and keeps the candidates with similarity
   `>= tau` (`tau` in `(0, 1]`).

The attacker's knowledge is exactly `qids` (used to block) plus `auxiliary` (extra
attributes), both declared by you. Nothing else is added: the **declared** ground-truth
columns are never features.

> **Undeclared identifier-like columns act as identifiers.** The engine excludes only the
> columns you declare as ground truth. A column that happens to hold a person identifier
> (a `person_id`, an e-mail, a record number) and that you list in `qids` or `auxiliary`
> is real attacker knowledge and will link records almost perfectly. Declare identifiers
> as ground truth, or leave them out of the attacker's attributes.

## What each number means

| Field | Counts | Needs ground truth |
|---|---|---|
| `cvpl_lr` | the share of the `N` source records for which **at least one** admissible released row survives `tau`. *Existence* of a plausible link. **Not** the probability of identifying the right person. | no |
| `cvpl_id_uniform_subjects_gt` | the expected probability of choosing the **right subject** if the attacker picks uniformly among the distinct candidate **subjects** (a subject released in several rows counts once). `1 of 5` candidate subjects correct gives `0.2`. | yes |
| `cvpl_id_uniform_rows_gt` | the expected probability of choosing the right subject if the attacker picks uniformly among the candidate released **rows** (`rows of the subject among the candidates / candidate rows`). | yes |
| `cvpl_id_uniform_gt` | the **headline** of the two: the larger of the two *population* values; `cvpl_id_uniform_basis` names which one it is (`subjects` or `rows`; equal values report `subjects`). Both are always published. | yes |
| `cvpl_top1_success_gt` | the share of the `N` source records whose **most similar** accepted candidate is the right subject (ties among accepted candidates: uniform, in expectation). | yes |
| `precision_at_tau` | the same success, over the records that **declare** a link only. Undefined (`null`, reason `no_declared_links`) when nobody declares one. | yes |
| `gt_coverage` | the share of source records whose correspondence set is **completely known** (see "Ground truth"). | yes |
| `cvpl_weighted_lr_legacy` | the older inverse-candidate-count weighted existential rate (`weighted_existential_linkage_v1`), kept for comparison with earlier results. **Not** a probability of identification. | no |

All population measures use the **source** record count `N` as the denominator, so a
subject that was suppressed from the release counts in `N` and contributes `0`.
Counts accompany the rates:

* `n_original` / `n_released` - rows of the two inputs;
* `n_with_candidates` - source records with at least one admissible candidate;
* `n_linked` - records with a known accepted pair; `n_undetermined` - records with no known
  link and at least one unknown candidate;
* `n_compared` - **cosine evaluations only** (pairs both of whose vectors are defined). Exact
  fallback matches and unknown pairs are *not* in it; see `n_exact_fallback_pairs`;
* `candidate_count_summary` (min / median / p95 / max / share with none).

With complete ground truth and a defined cosine for every candidate,
`cvpl_id_uniform_gt <= cvpl_lr` and `cvpl_top1_success_gt <= cvpl_lr` always hold: a record
that has no accepted candidate contributes `0` to both, and a record with one contributes at
most `1`. The identification measures can be much lower than `cvpl_lr` (many look-alikes,
few of them right); they are never higher. Never rename one measure into another.

Linking a dataset to itself is not evidence of discrimination: with a constant or
uninformative representation every record links to every other and `cvpl_lr` is `1` while no
subject is distinguishable. Read `cvpl_lr` together with the candidate counts and the
identification measures.

### Statuses

Every published measure has a `*_status` with `status`, a `reason` and, when measured, its
`denominator`; the same keys are in the operation's `OperationResult.metrics`, together with
the trajectories and bounds. A missing number is never a zero.

| `status` | Meaning |
|---|---|
| `measured` | a point value; no undefined-cosine candidate can influence this measure for any record |
| `bounded` | `value` is `null`; `lower` and `upper` bracket it (`n_uncertain` records are affected) |
| `not_measured` | `value` is `null`; the `reason` says why (`no_ground_truth`, `ground_truth_incomplete:...`, `resource_limit`, `projection_degenerate`, `non_finite_representation`, `empty_source`) |
| `not_applicable` | a rung needs a QID the release cannot be interpreted for (see below) |

**Undefined cosines.** A record or candidate whose projected vector is exactly zero has no
defined cosine. If the *encoded feature vectors* of the pair are exactly equal the pair is
accepted deterministically (`n_exact_fallback`, `n_exact_fallback_pairs`). Otherwise the
pair is *unknown*: it may or may not be accepted and, if accepted, may tie or outrank the
others. Each measure is bracketed over all of those possibilities, per record:

* `cvpl_lr`: a known accepted pair is a witness (the record is linked whatever else is
  unknown); with only unknown candidates the record contributes `[0, 1]`;
* the uniform measures: an unknown candidate can enter (diluting a correct one, or being the
  correct one) or stay out;
* `cvpl_top1_success_gt`: an unknown row may outrank the true candidate (lower bound `0`) and
  an unknown row of the right subject may be the top one (upper bound `1`);
* `precision_at_tau`: conservative bounds over which unknown records declare a link.

The population value is a point value (`measured`) only if *no* record's contribution is
uncertain; otherwise it is `bounded`. **A worst-case reader uses the `upper` bound** (for
`cvpl_lr` also `cvpl_lr_upper`): it is the value the white-box attacker can reach if every
unknown candidate is resolved in their favour, and it is never below the lower bound or a
point value. A known false link does **not** hide an unknown true
candidate: the existential `cvpl_lr` may be exact while the identification measures are
bounded. `undefined_policy` in the result restates this.

### Ground truth

Declare `ground_truth={"source_id_column": ..., "release_id_column": ..., "scope": "complete"}`.
Identifiers are equal when they are equal **as values inside one namespace**: numbers compare
as numbers (`1 == 1.0`, large integers exactly), text as text, and a number never equals a
text (`1 != "1"`). Numeric on one side and text on the other, booleans, non-finite numbers
and other types are rejected (`ground_truth.status = "invalid"`), never coerced. Identifiers
are used in memory only; they never appear in the result, the artifacts or the logs.

* Several released rows of one subject are **one candidate subject** for the subject strategy
  and **several candidate rows** for the row strategy.
* `scope="complete"` states that every source record has an identifier and every released row
  has one, so a source subject with no released row really has no counterpart (suppression)
  and scores `0`.
* An identifier being present is not the same as the correspondence being **known**. A
  released row without an identifier might belong to anyone, so with any such row the
  membership of the release is unknown: no source record has a known correspondence set,
  `gt_coverage` is `0` and no "known subset" figure exists
  (`known_subset_reason: release_membership_unknown`). When every released row is identified,
  the source records that carry an identifier have a known correspondence set and form the
  known subset, reported separately with its own denominator (`known_subset_value`,
  `known_subset_denominator`); it is never a population figure.
* Population GT measures need complete ground truth; with partial, invalid or absent GT they
  are `null` (`not_measured`) while `cvpl_lr` is still computed.

## Quasi-identifiers and generalization

Per QID (`{"name", "kind", "bin_width", "hierarchy", "generalized"}`):

* equal values are compatible; numeric points may be compared by bin (`bin_width=10` puts
  `21` and `27` in the same block). Numbers are compared exactly: integers beyond `2**53` stay
  distinct, `-0.0` equals `0.0`;
* the wildcard `*` (configurable `wildcard_tokens`) is compatible with any value that is not
  invalid, and under `null_policy="incompatible"` not with a missing one;
* numeric intervals (`[20-29]`, `20-29`, `[20, 30)`, `60+`, `pandas.Interval`) are compatible
  with the values they contain (and overlapping intervals). In the feature space an interval
  is represented by its midpoint and an open-ended one by its finite bound (`60+` -> `60`);
  an interval with no finite bound is imputed like a missing value;
* a categorical `hierarchy` `{child: parent}` makes an ancestor compatible with its descendants;
* a **missing** value follows `null_policy`: `equal_null` (default, matches only missing),
  `wildcard` (matches everything), `incompatible` (matches nothing);
* an **invalid** value (text that is neither a number, an interval nor the wildcard, a
  non-finite number) is a third thing: it is admissible with nothing, not even another invalid
  or a missing value. It is counted (`qid_status[<qid>].source_unparsable` /
  `release_unparsable`).

If the release contains values for a QID that none of these rules explains, that QID is
**`not_applicable`**: no number is reported for a rung that uses it. That includes categorical
labels outside the source domain with no hierarchy **and** with a hierarchy that does not name
them (a non-empty mapping that never mentions the released values explains nothing). The status
names the QID and the fix (`qid_status[<qid>].fix`). It is never treated as a plain mismatch,
which would look like protection. Fix it by supplying a hierarchy that covers the released
values, by using a ladder that does not need the QID, or, if the QID really is exact-only and
new labels just mean different people, by declaring `generalized=False`.

Hierarchies are *input values* and are never persisted: `config.json`, the cache metadata and
the logs carry a digest and a size (`hierarchy_digest`, `hierarchy_size`) instead of the mapping;
the digest still distinguishes two hierarchies, so they never share a cache entry.

## Blocking ladder

`blocking_ladder=[["age","gender","zip"], ["age","gender"], ["age"]]` evaluates the attacker
with progressively fewer constraints. The representation, `tau` and the fit are the same on
every rung. The result lists every rung (`blocking_ladder_trajectory`), whether each step
is `relaxing`, `tightening`, `equal` or `non_nested`, and `ladder.nested`.

On a nested relaxing ladder (each rung constrains fewer QIDs than the previous) `cvpl_lr`
cannot decrease, and the last rung **is** the union of all rungs' candidate sets (Theorem 1's
premise, for a fixed representation); the identification measures have no such guarantee,
more candidates can lower them.

The headline fields are always one well-defined attacker:

* one rung, or a nested relaxing ladder: the last rung (`headline_basis: "last_rung"`);
* any other ladder (`non_nested` or tightening steps): the attacker who tries **every**
  declared profile, i.e. all measures computed on the **union** of the rungs' admissible
  candidate sets, each pair counted once (`headline_basis: "union_of_rungs"`). The union uses
  the same admissibility predicate as the single rungs, and its `cvpl_lr` equals the last
  value of `candidate_union_trajectory`. A rung that cannot be measured makes the union
  `not_measured` / `not_applicable`; the union also has to fit `max_comparisons` as a whole.

Per-rung values stay available: `blocking_ladder_trajectory` (all measures per rung),
`cvpl_lr_by_rung`, and `candidate_union_trajectory` (cumulative `cvpl_lr_union` after each
rung). Choose the rungs before looking at ground-truth results; picking the rung that gives
the highest number with the ground truth in hand leaks the evaluation into the attacker.

## Projection fit and invariance

`fit_strategy="joint"` (default, the original CVPL) fits the scaler, encoder and PCA on
source and release together. `"source_only"` fits on the source. `"frozen"` reuses a fitted
projection (`result.projection`) so that several releases of one source are measured in the
same space - use it, or `source_only`, to compare releases.

**Duplicate release rows.** Subject-level counting (`cvpl_id_uniform_subjects_gt`, top-1) is
invariant to duplicating a released row of a subject **only under a fixed representation**
(`source_only`, `frozen`). In `joint` mode a duplicated row changes the joint mean, scale and
components, so it can change the geometry - and with it who passes `tau` - even if the
duplicate lies outside every candidate set. The row-uniform strategy changes with duplicates
of *candidate* rows in every mode (that is what it measures).

The result carries `projection.fingerprint`, a digest of the whole fitted representation
(encoding, scaling, centering, components). It and `algorithm_version` are part of the cache
identity; the operation's result cache never serves a result across different fitted states.

## Bounded encoding is visible

Categorical attributes are encoded in at most `max_categories` columns; the rarest values
share one bucket. Whenever that happens the result says so: `bucketed_features` lists, per
attribute, `distinct_values`, `encoded_width`, `distinct_values_collapsed` and
`rows_in_other_bucket`, and `warnings` carries the same in a sentence (names and counts only).
Missing values and the overflow bucket have internal tags: no user value, whatever its text,
can collide with them.

## Resources

Candidates are generated from blocks and similarities are computed in bounded tiles; a
`|D| x |D'|` matrix is never built, and neither is a group-level compatibility matrix:
compatibility between groups is streamed in tiles of at most 4M cells while planning, and
planning stops as soon as the comparison budget is exceeded. Per-record accumulators
(candidate subjects, exact-fallback matches, ties) are kept for one chunk of source records
(about `chunk_pairs` planned pairs) and discarded; exact-equal matches are aggregated by the
equivalence class of the encoded vector, never stored as pairs. `max_comparisons` (default
`2e10` per rung) is checked before a rung is scanned; above it the rung is `not_measured`
with reason `resource_limit` and the planned count, while other rungs are still measured.
`max_group_pairs` bounds the group-level checks needed by wildcards, intervals and
hierarchies. `n_compared` counts cosine evaluations only.

## Example

```python
from pamola_core.metrics import CVPLLinkageOperation

op = CVPLLinkageOperation(
    qids=[{"name": "age", "bin_width": 10}, "gender", "zip3"],
    auxiliary=["income"],
    blocking_ladder=[["age", "gender", "zip3"], ["age", "gender"]],
    tau=0.9,
    seed=0,
    ground_truth={"source_id_column": "patient_id", "release_id_column": "release_patient_id"},
)
result = op.execute(data_source, task_dir, reporter,
                    original_dataset_name="source", transformed_dataset_name="release")
```

The metrics JSON and `result.metrics` carry the fields above plus `algorithm_id`,
`algorithm_version`, `evaluation_model` (`white_box_upper_estimate`), `direction`
(`source_to_release`, the only direction in 1.0), `linkage_unit` (`subject`), `seed`,
`projection` and `qid_status`. Column names and counts only: no values, no identifiers.

## Legacy compatibility fixtures

The historical goldens G1, G3 and G5 reproduce exactly (existential rate, linked counts and the
legacy weighted rate). **G2 and G4 diverge, and the divergence is accepted as legacy-only.**
They leave `person_id` among the features (60 one-hot columns bounded to 50), so 49 of the PCA
variances are identical: the basis inside that degenerate eigenspace is arbitrary, cutting the
kept components through it rotates the cosine geometry, and a different, equally valid solver
legitimately links a different number of pairs. BEST linked 21 records (LR 0.35, weighted rate
0.319195); this implementation links 19 (LR 0.316667, weighted rate 0.276781) with the same 28
components and 0.903622 of the variance. The test pins the numbers this implementation
produces; they are not reference values and say nothing about the BEST ones.

## Limitations

* One specified attacker: the declared attributes, blocking, representation and threshold.
  Another attacker (other background knowledge, a better matcher, row order or index
  information) can do better or worse.
* The uniform *subjects* strategy assumes the attacker knows which candidate rows belong to
  the same subject (the grouping comes from the evaluation-only ground truth); the uniform
  *rows* strategy needs no such knowledge, which is why both are published and the larger one
  is the headline.
* Linear projection and cosine similarity only; no learned embedding, no optimal strategy.
* `source_to_release` only; the reverse question (which person does a released record
  belong to) needs a reference population and is not implemented.
* Ground truth is an auditor's input and must be trustworthy; this metric does not check it
  beyond uniqueness of source identifiers and identifier types.
* The joint fit depends on the whole data; compare releases with `source_only` or `frozen`.
* Categorical attributes are one-hot encoded. Very high-cardinality attributes are encoded in
  a bounded width (rare values share a bucket); a rare value's exact-match evidence then
  reaches the attacker through blocking only if the attribute is also a QID.
