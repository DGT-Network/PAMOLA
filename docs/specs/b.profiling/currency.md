Here is the **Software Requirements Specification (SRS)** for the `CurrencyFieldProfiler` module, to be implemented under `pamola_core/profiling/analyzers/currency.py`.

---

# 🧾 Software Requirements Specification (SRS)

## 📘 Title
**Currency Field Profiler Operation** – *Profiling of Financial Fields in Tabular Datasets*

## 📌 Identifier
`pamola_core.profiling.analyzers.currency.CurrencyOperation`

---

## 1. Purpose

This module defines an AI-assisted operation for profiling currency fields in tabular datasets. It is part of the **PAMOLA / PAMOLA.CORE** privacy risk analysis and anonymization pipeline. The operation extracts detailed statistical summaries, handles locale-aware parsing of currency values, and produces artifacts to support subsequent anonymization, data quality evaluation, and semantic transformation.

---

## 2. Scope

- Supports **Pandas** and optionally **Dask** for large-scale data.
- Accepts input via **DataFrame** or **CSV**.
- Extracts descriptive statistics, outlier detection, distribution shape, and normality.
- Robust against locale-specific formatting and common inconsistencies.
- Produces multiple artifacts: stats in JSON, visualizations (PNG), and samples (CSV).

---

## 3. Inputs

| Parameter       | Type            | Description |
|----------------|-----------------|-------------|
| `df` or `csv_path` | `pd.DataFrame` or `str` | Input dataset (provided by the caller). |
| `field_name`   | `str`           | Name of the column to analyze. |
| `sep`          | `str` (opt.)    | CSV separator (default `;`). |
| `encoding`     | `str` (opt.)    | Encoding of input file (default `UTF-8`). |
| `quotechar`    | `str` (opt.)    | Quotation character in CSV (default `"`). |
| `use_dask`     | `bool` (opt.)   | Enable Dask for large files. |
| `chunk_size`   | `int` (opt.)    | Size of chunks (default 10000). |

---

## 4. Functional Requirements

### 4.1 Data Parsing
- Strip symbols like `$`, `€`, `£`, whitespace.
- Convert values using locale-aware rules (`1.000,00` → `1000.00`).

### 4.2 Statistics Extraction
- `min`, `max`, `mean`, `median`, `std`, `iqr`, `percentiles`.
- Null ratio, valid count.

### 4.3 Distribution & Outliers
- Histogram and boxplot data.
- IQR-based outlier detection.
- Optional: Q-Q plot and Shapiro-Wilk/Anderson test.

### 4.4 Robust Handling
- Graceful fallback on:
  - Missing or non-parsable values.
  - Unexpected formats or mislabeling.
- Log issues, skip fields if required.

---

## 5. Output Artifacts (Saved to `task_dir`)

| Type        | Format | Location | Description |
|-------------|--------|----------|-------------|
| JSON        | `.json`| `{field_name}_stats.json` | Full analysis report. |
| PNG         | `.png` | `{field_name}_distribution.png`, `{field_name}_boxplot.png`, `{field_name}_qq_plot.png` (optional) | Visualization of results. |
| CSV Sample  | `.csv` | `dictionaries/{field_name}_sample.csv` | Sample records with `id_field` and original currency values. |

---

## 6. Architecture & Dependencies

### Class Structure
- `CurrencyAnalyzer` → logic layer
- `CurrencyOperation` → inheriting from `FieldOperation`

### Inherits from:
- `pamola_core.utils.ops.base_operation.FieldOperation`

### Utilizes:
- `pamola_core.utils.io.*`
- `pamola_core.utils.logging`
- `pamola_core.utils.visualization`
- `pamola_core.utils.progress`
- `pamola_core.profiling.commons.currency_utils` (if >700 LOC or reused)

---

## 7. Non-functional Requirements

| Property        | Value |
|-----------------|-------|
| Stateless       | ✅    |
| Thread-safe     | ✅    |
| Locale-safe     | ✅    |
| Logging         | ✅ `pamola_core.utils.logging` |
| Max file size   | Handled by Dask or chunking |
| Test coverage   | ≥ 90% |
| Code style      | PEP8 + PAMOLA template |

---

## 8. Extension Points

- Locale-specific rules (`locale="en_US"`, etc.)
- Multi-field batch profiling
- Deep semantic interpretation of transaction types (with NER)

---

Would you like me to also generate the **module docstring** and class skeletons (`CurrencyAnalyzer`, `CurrencyOperation`) or move on to unit test and Markdown prompt generation?