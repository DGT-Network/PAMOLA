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

## The attacker

The attacker holds the **source** dataset `D` (or the declared part of it) and sees the
**release** `D'`. For each source record `x_i` the attacker:

1. **Blocks** (`B`): keeps only the released rows whose quasi-identifiers (QIDs) are
   *compatible* with `x_i`.
2. **Vectorizes** (`phi`) the declared attributes: numeric columns are z-scored
   (unknown values filled with the mean), categorical columns are one-hot encoded
   (at most `max_categories` columns per attribute; the rarest values share one bucket).
3. **Projects** (`psi`) with PCA, keeping the smallest number of components that reach
   `variance_retained` (between `min_components` and `max_components`).
4. Compares by **cosine similarity** (`s`) and keeps the candidates with similarity
   `>= tau` (`tau` in `(0, 1]`).

The attacker's knowledge is exactly `qids` (used to block) plus `auxiliary` (extra
attributes), both declared by you. Nothing else is added: the identifier and
ground-truth columns are never features.

## What each number means

| Field | Counts | Needs ground truth |
|---|---|---|
| `cvpl_lr` | the share of the `N` source records for which **at least one** released row survives blocking and `tau`. *Existence* of a plausible link. **Not** the probability of identifying the right person. | no |
| `cvpl_id_uniform_gt` | the expected probability of choosing the **right subject** if the attacker picks uniformly among the surviving candidates (subjects, not rows). `1 of 5` candidates correct gives `0.2`; none correct gives `0`; no candidates gives `0`. | yes |
| `cvpl_top1_success_gt` | the share of the `N` source records whose **most similar** surviving candidate is the right subject (ties: uniform, in expectation). | yes |
| `precision_at_tau` | the same success, over the records that **declare** a link only (denominator: records with at least one surviving candidate). | yes |
| `gt_coverage` | the share of source subjects whose ground truth is known. | yes |
| `cvpl_weighted_lr_legacy` | the older inverse-candidate-count weighted existential rate (`weighted_existential_linkage_v1`), kept for comparison with earlier results. **Not** a probability of identification. | no |

All population measures use the **source** record count `N` as the denominator, so a
subject that was suppressed from the release counts in `N` and contributes `0`.
Counts (`n_original`, `n_released`, `n_with_candidates`, `n_linked`, `n_compared`) and
`candidate_count_summary` (min / median / p95 / max / share with none) accompany the rates.

`cvpl_lr` can be high while `cvpl_id_uniform_gt` is low (many look-alikes, few of them
right) and the other way round; never rename one into the other.

### Statuses

Each measure has a `*_status` with `status` (`measured`, `not_measured`,
`not_applicable`), a `reason` and, when measured, its `denominator`. A missing number is
never a zero:

* no ground truth: the three GT measures are `null`, `not_measured`, reason
  `no_ground_truth`; `cvpl_lr` is still computed;
* partial ground truth: the population measures are `null` (`ground_truth_incomplete`);
  the value on the known subset is given separately with its own denominator
  (`known_subset_value`, `known_subset_denominator`) and must not be read as a population figure;
* a QID generalized in the release without a declared rule: `not_applicable` (see below);
* over the comparison budget: `not_measured`, reason `resource_limit`.

## Ground truth

Declare `ground_truth={"source_id_column": ..., "release_id_column": ..., "scope": "complete"}`.
Equal identifiers mean the same subject. Several released rows of one subject are one
candidate (the subject), so duplicating a row does not lower the subject's probability.
`scope="complete"` states that every source subject is listed and every released row has
an identifier; a source subject with no released row then scores `0` (suppression).
Identifiers are used in memory only; they never appear in the result, the artifacts or the logs.

## Quasi-identifiers and generalization

Per QID (`{"name", "kind", "bin_width", "hierarchy", "generalized"}`):

* equal values are compatible; numeric points may be compared by bin (`bin_width=10` puts
  `21` and `27` in the same block);
* the wildcard `*` (configurable `wildcard_tokens`) is compatible with any value;
* numeric intervals (`[20-29]`, `20-29`, `[20, 30)`, `60+`, `pandas.Interval`) are compatible
  with the values they contain (and overlapping intervals);
* a categorical `hierarchy` `{child: parent}` makes an ancestor compatible with its descendants;
* `null` follows `null_policy`: `equal_null` (default, matches only null), `wildcard`
  (matches everything), `incompatible` (matches nothing).

If the release contains values for a QID that none of these rules explains (for example
`"South"` for `Lyon` with no hierarchy), that QID is **`not_applicable`**: no number is
reported for a rung that uses it. The status names the QID and the fix (`qid_status[<qid>].fix`). It is never treated as a plain mismatch, which would look
like protection. Fix it by supplying the hierarchy, by using a ladder that does not need
the QID, or, if the QID really is exact-only and new labels just mean different people,
by declaring `generalized=False`.

## Blocking ladder

`blocking_ladder=[["age","gender","zip"], ["age","gender"], ["age"]]` evaluates the attacker
with progressively fewer constraints. The representation, `tau` and the fit are the same on
every rung. The result lists every rung (`blocking_ladder_trajectory`), whether each step
is `relaxing`, `tightening`, `equal` or `non_nested`, and `ladder.nested`.

On a nested relaxing ladder (each rung constrains fewer QIDs than the previous) `cvpl_lr`
cannot decrease, and the last rung **is** the union of all rungs' candidate sets (Theorem 1's
premise); `cvpl_id_uniform_gt` has no such guarantee, more candidates can lower it.

The headline fields (`cvpl_lr`, `cvpl_id_uniform_gt`, `cvpl_top1_success_gt`,
`precision_at_tau`, counts) are always one well-defined attacker:

* one rung, or a nested relaxing ladder: the last rung (`headline_basis: "last_rung"`);
* any other ladder (`non_nested` or tightening steps): the attacker who tries **every**
  declared profile, i.e. all measures computed on the **union** of the rungs' candidate sets,
  each pair counted once (`headline_basis: "union_of_rungs"`). A rung that cannot be measured
  makes the union `not_measured` / `not_applicable`; the union also has to fit
  `max_comparisons` as a whole.

Per-rung values stay available: `blocking_ladder_trajectory` (all measures per rung),
`cvpl_lr_by_rung`, and `candidate_union_trajectory` (cumulative `cvpl_lr_union` after each
rung; its last value equals the union headline). Choose the rungs before looking at
ground-truth results; picking the rung that gives the highest number with the ground truth in
hand leaks the evaluation into the attacker.

## Projection fit

`fit_strategy="joint"` (default, the original CVPL) fits the scaler, encoder and PCA on
source and release together; adding unrelated rows to either can move all distances.
`"source_only"` fits on the source. `"frozen"` reuses a fitted projection
(`result.projection`) so that several releases of one source are measured in the same
space - use it, or `source_only`, to compare releases.

## Bounded encoding is visible

Categorical attributes are encoded in at most `max_categories` columns; the rarest values
share one bucket. Whenever that happens the result says so: `bucketed_features` lists, per
attribute, `distinct_values`, `encoded_width`, `distinct_values_collapsed` and
`rows_in_other_bucket`, and `warnings` carries the same in a sentence (names and counts only).
`projection.encoding` has the per-attribute detail for every categorical feature.

## Degenerate cases

A record whose projected vector is exactly zero has no defined cosine. If the *encoded
feature vectors* of the pair are exactly equal, the link is counted deterministically
(`n_exact_fallback`). Otherwise the pair is undefined: the record stays in the denominator
`N`, adds `0` to the point value, and is counted in `n_undetermined`; `cvpl_lr_lower` /
`cvpl_lr_upper` give the bounds (`upper` treats every undetermined record as linked). If the
representation is degenerate for every record the measures are `not_measured`
(`projection_degenerate`).

## Resources

Candidates are generated from blocks and similarities are computed in bounded chunks; a
`|D| x |D'|` matrix is never built. `max_comparisons` (default `2e10` per rung) is checked
before a rung is scanned; above it the rung is `not_measured` with reason `resource_limit`
and the planned count, while other rungs are still measured. `max_group_pairs` bounds the
group-level checks needed by wildcards, intervals and hierarchies.

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

The metrics JSON carries the fields above plus `algorithm_id`, `algorithm_version`,
`direction` (`source_to_release`, the only direction in 1.0), `linkage_unit` (`subject`),
`seed`, `projection` (components, variance, encoding widths, imputed counts) and `qid_status`.
Column names and counts only: no values, no identifiers.

## Limitations

* One specified attacker: the declared attributes, blocking, representation and threshold.
  Another attacker (other background knowledge, a better matcher, row order or index
  information) can do better or worse.
* Linear projection and cosine similarity only; no learned embedding, no optimal strategy.
* `source_to_release` only; the reverse question (which person does a released record
  belong to) needs a reference population and is not implemented.
* Ground truth is an auditor's input and must be trustworthy; this metric does not check it
  beyond uniqueness of source identifiers.
* The joint fit depends on the whole data; compare releases with `source_only` or `frozen`.
* Categorical attributes are one-hot encoded. Very high-cardinality attributes are encoded in
  a bounded width (rare values share a bucket); a rare value's exact-match evidence then
  reaches the attacker through blocking only if the attribute is also a QID.
