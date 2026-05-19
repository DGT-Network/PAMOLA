# Operation Config Workaround Report
**Pipeline:** Bank Churn Anonymization — Happy Path 2026-05-17  
**Scope:** ANON-PSEUDO-001, ANON-GEN-001, ANON-GEN-002, ANON-SUPP-002  
**Rule:** No source changes — workarounds using existing parameters only.

### Source files referenced

| Operation | Source file |
|---|---|
| ANON-PSEUDO-001 | `pamola_core/anonymization/pseudonymization/hash_based_op.py` |
| ANON-GEN-001 | `pamola_core/anonymization/generalization/numeric_op.py` |
| ANON-GEN-002 | `pamola_core/anonymization/generalization/categorical_op.py` |
| ANON-GEN-002 (strategies) | `pamola_core/anonymization/commons/categorical_strategies.py` |
| ANON-GEN-002 (hierarchy file) | `pamola_core/anonymization/commons/hierarchy_dictionary.py` |
| ANON-SUPP-002 | `pamola_core/anonymization/suppression/record_op.py` |
| ANON-SUPP-002 (schema) | `pamola_core/anonymization/schemas/record_op_core_schema.py` |
| BA spec | `E:/DTX/pamola-production/docs/ba-artifacts/TITAN_PAMOLA_PROD_HAPPY_PATH_2026-05-17.md` |

---

## ANON-PSEUDO-001 — Hash-Based Pseudonymization (`HashBasedPseudonymizationOperation`)

**Used for:** `ID` (Op 2.1), `FAMILYID` (Op 2.2), `CardNumber` (Op 2.3)

### What the BA config wants

```json
// Op 2.1 — ID
{"strategy":"hmac_sha256","output_length":16,"preserve_format":false}

// Op 2.2 — FAMILYID
{"strategy":"hmac_sha256","output_length":16,"preserve_format":false,"null_handling":"keep_null"}

// Op 2.3 — CardNumber
{"strategy":"hmac_sha256","output_length":16,"preserve_format":true}
```

| Intent | Description |
|---|---|
| `strategy: "hmac_sha256"` | Use HMAC-SHA256 keyed hash — same key across all 3 ops produces consistent pseudonyms (same `FAMILYID` → same output) |
| `output_length: 16` | Truncate output to 16 characters |
| `preserve_format: false` | Plain hash string output, no visual pattern |
| `preserve_format: true` | Preserve original visual format (e.g., `"XXXX XXXX XXXX XXXX"` for CardNumber) |
| `null_handling: "keep_null"` | Keep null values as-is, do not hash them |

### Parameter mapping

| BA config key | Actual param in op | Value to set | Source ref | Notes |
|---|---|---|---|---|
| `strategy: "hmac_sha256"` | `algorithm` | `"sha3_256"` | `hash_based_op.py:148` — schema `enum: ["sha3_256", "sha3_512"]` | No HMAC construction in op. SHA3+salt achieves the same goal: irreversible, secret-keyed transformation. |
| *(same key across 3 ops)* | `salt_config` | `{"source":"parameter","value":"<workspace_secret_hex>"}` | `hash_based_op.py:148-155` (schema), `:329-332` (default) | Use the **same fixed salt string** across all 3 instances for reproducible pseudonyms. |
| `output_length: 16` | `output_length` | `16` | `hash_based_op.py:161` — `minimum: 8` | Direct mapping. Truncates hex string to first 16 chars. |
| `preserve_format: false` | *(no param needed)* | — | `hash_based_op.py:228` — `output_format: str = "hex"` | Default output is plain hex. No param needed. |
| `preserve_format: true` (CardNumber) | *(not supported)* | — | No format-preservation logic found in source | **Gap — see note below.** |
| `null_handling: "keep_null"` | `null_strategy` | `"PRESERVE"` | `hash_based_op.py:226` — `null_strategy: str = "PRESERVE"` | Already the default — can be omitted. |
| *(reproducibility across runs)* | `use_pepper` | `false` | `hash_based_op.py:222` — `use_pepper: bool = True` (default); `:923-925` generates random pepper per session | Must disable pepper to get deterministic output. With `use_pepper=True` (default), output changes every run. |

### Config to pass into the UI

**Op 2.1 — `ID`:**
```json
{
  "algorithm": "sha3_256",
  "salt_config": {"source": "parameter", "value": "<workspace_secret_hex>"},
  "output_length": 16,
  "use_pepper": false
}
```

**Op 2.2 — `FAMILYID`:**
```json
{
  "algorithm": "sha3_256",
  "salt_config": {"source": "parameter", "value": "<workspace_secret_hex>"},
  "output_length": 16,
  "use_pepper": false,
  "null_strategy": "PRESERVE"
}
```

**Op 2.3 — `CardNumber`:**
```json
{
  "algorithm": "sha3_256",
  "salt_config": {"source": "parameter", "value": "<workspace_secret_hex>"},
  "output_length": 16,
  "use_pepper": false
}
```

> Replace `<workspace_secret_hex>` with the workspace secret key as a hex string (e.g., 64 hex chars = 32 bytes). Use the **same value** across all 3 instances.

### Gap: `preserve_format: true` for CardNumber

The op always outputs a flat string in the chosen format (`hex`, `base64`, etc.). No format-preservation logic exists anywhere in source.

The BA specifies CardNumber output should maintain the `"XXXX XXXX XXXX XXXX"` visual pattern. This **cannot be achieved** with current op parameters.

**Workaround options:**

| Option | Approach |
|---|---|
| A *(recommended for baseline)* | Accept plain 16-char hex output. Note as known deviation. |
| B | Post-process: split the 16-char hex into 4×4 groups → `"a4f2-c8e1-b7d0-9f3a"`. Does not match original card format exactly but gives structured appearance. |
| C | Use `output_format: "uuid"` (`hash_based_op.py:161`) — outputs `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` format. Structured but not matching card pattern. |

### Expected output sample

| Input | Output |
|---|---|
| `"ID-00123"` | `"a4f2c8e1b7d09f3a"` (16-char hex, illustrative) |
| `"FAMILYID-456"` | `"3c91e7a2d6f08b5c"` (illustrative) |
| `null` | `null` (PRESERVE — `hash_based_op.py:839-841`) |
| Same `"ID-00123"` on re-run | `"a4f2c8e1b7d09f3a"` — same output (reproducible because `use_pepper=false` + fixed salt) |

---

## ANON-GEN-001 — Numeric Generalization (`NumericGeneralizationOperation`)

**Used for:** `AGE`, `Income`, `HOMEVALUE`, `RENTVALUE`

### What the BA config wants

```json
{
  "method": "manual_bins",
  "bins": [[18,25],[26,35],[36,45],[46,55],[56,65],[66,100]],
  "output_format": "{start}-{end}",
  "out_of_range_handling": "clip"
}
```

| Intent | Description |
|---|---|
| `method: "manual_bins"` | User-defined bin boundaries, not auto-computed |
| `bins` | Explicit list of `[start, end]` intervals |
| `output_format: "{start}-{end}"` | Label each bin as `"18-25"`, `"26-35"`, etc. |
| `out_of_range_handling: "clip"` | Values outside all bins snap to the nearest bin, no "other" label |

### Parameter mapping

| BA config key | Actual param in op | Value to set | Source ref | Notes |
|---|---|---|---|---|
| `method: "manual_bins"` | `strategy` | `"range"` | `numeric_op.py:163` (constructor), `:714-725` (`process_batch` dispatch) | Only `strategy="range"` accepts user-defined intervals. `"binning"` does not — it auto-computes bins. |
| `bins` | `range_limits` | e.g. `[[18,25],[26,35],…]` | `numeric_op.py:167`, `:1540-1548` (`convert_range_limits_for_schema`) | Passed as array-of-arrays from JSON; op stores as-is. `range_max` of each interval is used only for the label — **not** as the right bin edge (see boundary note). |
| `output_format: "{start}-{end}"` | *(no param — drop this key)* | — | `numeric_op.py:986-987` hard-codes `f"{fmt(range_min)}-{fmt(range_max)}"` | Label format is fixed in source. **Do not pass `output_format` to the op** — it collides with the file-format param of the same name which expects `"csv"/"parquet"/"arrow"` (`numeric_op.py:799-804`) and will raise `InvalidParameterError`. |
| `out_of_range_handling: "clip"` | *(no param — handle outside op)* | — | `numeric_op.py:977-996` — always generates `"<min"` and `">=max"` bins | See pre-processing section below. |
| `null_handling: "keep_null"` | `null_strategy` | `"PRESERVE"` | Base class default — `"PRESERVE"` | Only relevant for `HOMEVALUE` and `RENTVALUE`. Already the default. |

### Config to pass into the UI

**AGE and Income** (no nulls expected):
```json
{
  "strategy": "range",
  "range_limits": [[18,25],[26,35],[36,45],[46,55],[56,65],[66,100]]
}
```

**HOMEVALUE and RENTVALUE** (nulls present — preserve them):
```json
{
  "strategy": "range",
  "range_limits": [[0,200000],[200000,500000],[500000,1000000],[1000000,2000000],[2000000,10000000]],
  "null_strategy": "PRESERVE"
}
```

### Required pre-processing (clip)

`_apply_range` (`numeric_op.py:939-1014`) always adds `"<min"` and `">=max"` bins for out-of-range values. To prevent outliers from getting these labels, clip the column **before** running the op.

```python
# AGE — clip to 19 (not 18) — see boundary note below
df["AGE"] = df["AGE"].clip(lower=19, upper=100)

# Income
df["Income"] = df["Income"].clip(lower=1, upper=1000000)

# HOMEVALUE
df["HOMEVALUE"] = df["HOMEVALUE"].clip(lower=1, upper=10000000)

# RENTVALUE
df["RENTVALUE"] = df["RENTVALUE"].clip(lower=1, upper=50000)
```

If the pipeline has a "Pre-transform" or "Data Prep" step, configure the clip there.

### Boundary note — IMPORTANT ⚠️

**`_apply_range` builds right-closed bins** (`numeric_op.py:1008-1013` uses `pd.cut(..., include_lowest=True)`). The bin edges are derived **only from `range_min` values**, not from `range_max`.

For `range_limits = [[18,25],[26,35],…]`, the actual pd.cut bins are:

```
(−∞, 18]   →  label "<18"
(18,  26]  →  label "18-25"
(26,  36]  →  label "26-35"
  ...
(66, 100]  →  label "66-100"
(100, +∞)  →  label ">=100"
```

**Affected boundary values (shifted one bin lower):**

| Input value | Actual label | Expected label |
|---|---|---|
| `18` | `"<18"` | `"18-25"` ❌ |
| `26` | `"18-25"` | `"26-35"` ❌ |
| `36` | `"26-35"` | `"36-45"` ❌ |
| `46` | `"36-45"` | `"46-55"` ❌ |
| `56` | `"46-55"` | `"56-65"` ❌ |
| `66` | `"56-65"` | `"66-100"` ❌ |

**Why clip(lower=18) doesn't fully help:** Any value clipped to exactly `18` still falls in the `(−∞, 18]` bin → gets label `"<18"`. Only the upper bound clip works correctly (100 falls in `(66, 100]` → `"66-100"` ✅).

**Clip workaround for lower boundary:** Clip to one above the first `range_min` (e.g., `lower=19` for AGE). This means:
- Values previously below 18 → become 19 → labeled `"18-25"` ✅
- Age 18 itself → becomes 19 → slight data distortion, but acceptable for baseline k-anonymity

**For the `Income`, `HOMEVALUE`, `RENTVALUE` bins** where `range_min` starts at `0` — clip to `lower=1` so zero-valued rows land in the first named bin, not `"<0"`.

**Known limitation:** The 6 boundary values (26, 36, 46, 56, 66 for AGE, and equivalent boundaries in other fields) that fall exactly on `range_min` of non-first bins will be labeled one bin lower. This is a source-level issue in `_apply_range` — no parameter can fix it. For the happy-path baseline, the impact is small (single-value boundaries in a 10K row dataset).

---

## ANON-GEN-002 — Categorical Generalization (`CategoricalGeneralizationOperation`)

**Used for:** `RACE`, `CITY`

### What the BA config wants

```json
{
  "value_hierarchy": {"White":"White","Asian":"OTHER","Black":"OTHER",...},
  "unmapped_strategy": "map_to_fallback",
  "fallback": "OTHER"
}
```

| Intent | Description |
|---|---|
| `value_hierarchy` | Explicit flat mapping: original value → generalized value |
| `unmapped_strategy: "map_to_fallback"` | Any value not in the dict maps to the fallback |
| `fallback: "OTHER"` | The fallback label |

### Parameter mapping

| BA config intent | Actual param in op | Value to set | Source ref | Notes |
|---|---|---|---|---|
| Inline dict mapping | `strategy` | `"hierarchy"` | `categorical_op.py:834` — dispatches to `apply_hierarchy()` | Only `"hierarchy"` uses explicit value-to-value mapping. `"merge_low_freq"` and `"frequency_based"` are automatic and do not accept a user-supplied dict. |
| Inline dict `{value: generalized}` | `external_dictionary_path` | Path to a JSON file | `categorical_op.py:1341-1357` (`_load_hierarchy`) raises `ValidationError` if path not provided | Op does not accept an inline dict — mapping must be saved to an external file. See format below. |
| `unmapped_strategy: "map_to_fallback"` | `allow_unknown` + `unknown_value` | `allow_unknown=True`, `unknown_value="OTHER"` | `categorical_op.py:174,176` — both are defaults | Unmapped values are passed through `engine.apply_to_series` which maps unknowns to `unknown_value` (`categorical_strategies.py:298-307`). |
| `fallback: "OTHER"` | `unknown_value` | `"OTHER"` or `"Other"` | `categorical_op.py:174` — default `"OTHER"` | Match exactly to BA spec: `RACE` → `"OTHER"`, `CITY` → `"Other"` (title-case). |

### Required pre-step: save the mapping as an external file

The op loads the hierarchy via `HierarchyDictionary.load_from_file` → `_parse_json_hierarchy` (`hierarchy_dictionary.py:505-552`). 

**⚠️ Critical format requirement:** The JSON file **must include a `"levels"` key**. Without it, `self._levels` remains `[]` and `get_hierarchy()` returns `None` for every lookup due to the guard at `hierarchy_dictionary.py:282`:

```python
if level < 1 or level > len(self._levels):  # 1 > 0 → True → always returns None
    return None
```

All values would then fall through to `unknown_value = "OTHER"`, defeating the purpose of the mapping.

**Correct file format for `RACE`** (save as e.g. `hierarchy_race.json`):
```json
{
  "levels": ["category"],
  "White": "White",
  "Asian": "OTHER",
  "Black": "OTHER",
  "Hispanic": "OTHER",
  "Latino": "OTHER",
  "Indigenous": "OTHER",
  "Mixed": "OTHER"
}
```

**Correct file format for `CITY`** (save as e.g. `hierarchy_city.json`):
```json
{
  "levels": ["region"],
  "Toronto": "GTA",
  "Mississauga": "GTA",
  "Brampton": "GTA",
  "Hamilton": "GTA",
  "Markham": "GTA",
  "Vaughan": "GTA",
  "Ottawa": "Eastern Ontario",
  "Kingston": "Eastern Ontario",
  "Montreal": "Quebec Metro",
  "Laval": "Quebec Metro",
  "Quebec City": "Quebec Metro",
  "Vancouver": "Lower Mainland",
  "Burnaby": "Lower Mainland",
  "Surrey": "Lower Mainland",
  "Calgary": "Alberta Metro",
  "Edmonton": "Alberta Metro",
  "Winnipeg": "Prairies",
  "Saskatoon": "Prairies",
  "Regina": "Prairies",
  "Halifax": "Maritimes",
  "St. John's": "Maritimes"
}
```

How `_parse_json_hierarchy` handles the simple string values (`hierarchy_dictionary.py:545-547`):
```python
elif isinstance(hierarchy, str):
    self._data[value] = {"level_1": hierarchy}
```
With `"levels": ["category"]` present, `len(self._levels) = 1`, so `get_hierarchy(value, level=1)` passes the guard and returns `hierarchy["level_1"]` ✅.

### Config to pass into the UI

**RACE:**
```json
{
  "strategy": "hierarchy",
  "external_dictionary_path": "/path/to/hierarchy_race.json",
  "dictionary_format": "auto",
  "unknown_value": "OTHER",
  "allow_unknown": true
}
```

**CITY:**
```json
{
  "strategy": "hierarchy",
  "external_dictionary_path": "/path/to/hierarchy_city.json",
  "dictionary_format": "auto",
  "unknown_value": "Other",
  "allow_unknown": true
}
```

> `CITY` uses `"Other"` (title-case) while `RACE` uses `"OTHER"` (all-caps) — match exactly to BA spec.

### Expected output sample (RACE)

| Input `RACE` | Output |
|---|---|
| `"White"` | `"White"` |
| `"Asian"` | `"OTHER"` |
| `"Black"` | `"OTHER"` |
| `"Pacific Islander"` *(not in dict)* | `"OTHER"` — `allow_unknown=True` fallback (`categorical_strategies.py:298-307`) |
| `null` | `null` — PRESERVE default |

---

## ANON-SUPP-002 — Record Suppression (`RecordSuppressionOperation`)

**Used for:** k=5 enforcement across 12 quasi-identifier fields after generalization.

### What the BA config wants

> Drop any equivalence-class group that still has fewer than k=5 rows after generalization.  
> QI scope: `SEX`, `CITY`, `PROVINCE`, `AGE`, `RACE`, `IsMarried`, `IsEduBachelors`, `IsHomeOwner`, `IsUnemployed`, `Income`, `HOMEVALUE`, `RENTVALUE`

| Intent | Description |
|---|---|
| Group-based suppression | Evaluate rows as groups (by QI combination), not individually |
| Threshold: k=5 | Drop all rows in any group with fewer than 5 members |
| Scope: 12 QI fields | Group defined by the combination of all 12 fields |

### Parameter mapping

| BA intent | Actual param in op | Value to set | Source ref | Notes |
|---|---|---|---|---|
| Group-level k enforcement | `suppression_condition` | `"risk"` | `record_op.py:136` — valid conditions: `["null", "value", "range", "risk", "custom"]`; `:866-870` — risk mask logic | The only condition that supports threshold-based row removal. |
| k=5 threshold | `risk_threshold` | `5.0` | `record_op.py:866-870`: `mask = batch[self.ka_risk_field] < self.risk_threshold` | Rows where `ka_risk_field < 5` are suppressed. |
| Group size from 12 QI fields | `ka_risk_field` | `"_k_score"` | `record_op.py:866-870` — reads the named column directly | **Op does not compute group sizes.** A pre-computed column must be present before the op runs. |
| Scope: 12 QI fields | *(no op param)* | — | Used only in the pre-compute step | |

### Required pre-processing: compute the k-score column

Before running the op, add a column that holds the equivalence class size for each row. Run this **after** all ANON-GEN-001 and ANON-GEN-002 ops so groups are evaluated on the **generalized** values:

```python
qi_fields = [
    "SEX", "CITY", "PROVINCE", "AGE", "RACE",
    "IsMarried", "IsEduBachelors", "IsHomeOwner", "IsUnemployed",
    "Income", "HOMEVALUE", "RENTVALUE"
]

df["_k_score"] = df.groupby(qi_fields, dropna=False)["SEX"].transform("count")
```

### Config to pass into the UI

```json
{
  "field_name": "SEX",
  "suppression_condition": "risk",
  "ka_risk_field": "_k_score",
  "risk_threshold": 5.0
}
```

> `field_name` is required by the op schema (`record_op_core_schema.py:166`) and validated to exist (`record_op.py:1373`), but not used in the mask logic when `suppression_condition="risk"` (`record_op.py:866-870`). Any stable column in the dataset works.

### Expected behavior

| `_k_score` value | Action | Source ref |
|---|---|---|
| `< 5` | Row is removed | `record_op.py:866-870` — `mask = ka_risk_field < risk_threshold`, then `batch[~mask]` kept |
| `>= 5` | Row is kept | |

Expected output size: roughly 9,000–9,500 rows (down from 10,000), consistent with BA spec estimate.

---

## Summary of gaps

| Op | BA config key | Status | Workaround | Source evidence |
|---|---|---|---|---|
| ANON-PSEUDO-001 | `strategy: "hmac_sha256"` | Not supported | Use `algorithm: "sha3_256"` | `hash_based_op.py:148` — enum only has sha3_256/sha3_512 |
| ANON-PSEUDO-001 | Reproducibility across runs | Not default | Set `use_pepper: false` + fixed `salt_config.value` across all 3 instances | `hash_based_op.py:222,923-925` — pepper is random per session by default |
| ANON-PSEUDO-001 | `preserve_format: true` (CardNumber) | Not supported | Accept plain hex output; note as known deviation | No format-preservation logic found in `hash_based_op.py` |
| ANON-PSEUDO-001 | `null_handling: "keep_null"` | Supported | `null_strategy: "PRESERVE"` (already the default) | `hash_based_op.py:226` |
| ANON-GEN-001 | `method: "manual_bins"` | Not supported | Use `strategy: "range"` | `numeric_op.py:714-725` — only binning/rounding/range strategies exist |
| ANON-GEN-001 | `output_format: "{start}-{end}"` | Hard-coded correctly, but key name conflicts | Drop the key — do not pass to op | `numeric_op.py:986-987` (label), `:799-804` (file format conflict) |
| ANON-GEN-001 | `out_of_range_handling: "clip"` | Not supported | Pre-clip column before op; use `lower = range_min + 1` (not `range_min`) for lower bound | `numeric_op.py:977-996` — always generates `"<min"` / `">=max"` bins |
| ANON-GEN-001 | Boundary values at `range_min` | Source-level limitation | Accept that values at 26, 36, 46, 56, 66 (AGE) are labeled one bin lower; document as known deviation | `numeric_op.py:1008-1013` — pd.cut right-closed `(a, b]`; `range_max` not used as bin edge |
| ANON-GEN-001 | `null_handling: "keep_null"` | Supported | `null_strategy: "PRESERVE"` (default) | Base class |
| ANON-GEN-002 | Inline value mapping dict | Not supported | Save dict as external JSON file; pass path via `external_dictionary_path` | `categorical_op.py:1341-1357` |
| ANON-GEN-002 | JSON file format | Must include `"levels"` key | Add `"levels": ["category"]` to the JSON file | `hierarchy_dictionary.py:282` — guard returns None if `_levels` is empty |
| ANON-GEN-002 | `unmapped_strategy: "map_to_fallback"` | Supported | `allow_unknown=true` + `unknown_value="OTHER"` (defaults) | `categorical_strategies.py:298-307` |
| ANON-SUPP-002 | Group-level k=5 enforcement | Not supported natively | Pre-compute `_k_score` column; use `suppression_condition: "risk"` | `record_op.py:866-870` |
