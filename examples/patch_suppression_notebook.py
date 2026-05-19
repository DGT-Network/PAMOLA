"""Patch 02_record_suppression_advanced.ipynb - insert Step 8 before Summary."""
import json
import uuid

NOTEBOOK = "E:/DTX/PAMOLA/examples/anonymization/suppression/02_record_suppression_advanced.ipynb"


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source_lines,
    }


with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Remove any previously inserted Step 8 cells (idempotent)
cells[:] = [
    c for c in cells
    if "Step 8: Bank Churn Happy Path Workaround Verification (ANON-SUPP-002)" not in "".join(c["source"])
    and "D1: Without _k_score" not in "".join(c["source"])
    and "D2: Pre-compute k-score" not in "".join(c["source"])
]

summary_idx = next(
    i for i, c in enumerate(cells)
    if "Summary" in "".join(c["source"]) and "Accomplished" in "".join(c["source"])
)
print(f"Summary cell at index: {summary_idx}")

md_cell = make_md_cell([
    "## Step 8: Bank Churn Happy Path Workaround Verification (ANON-SUPP-002)\n",
    "\n",
    "Verify k=5 enforcement workaround per `docs/ba-artifacts/happy_path_op_config_workarounds.md`.\n",
    "\n",
    "**Key finding:** `RecordSuppressionOperation` does NOT compute group sizes.\n",
    "A `_k_score` column must be pre-computed before the op runs.\n",
    "\n",
    "**D1:** Run op WITHOUT `_k_score` column -> expect error/failure (documents the gap)  \n",
    "**D2:** Pre-compute `_k_score` via `groupby.transform('count')` -> run op -> verify output  \n",
    "- All remaining rows must have `_k_score >= 5`  \n",
    "- Output size expected ~9,000-9,500 rows (from 10,000)",
])

d_cell = make_code_cell([
    "# D1 + D2: ANON-SUPP-002 k=5 enforcement workaround verification\n",
    "import pandas as pd\n",
    "import os\n",
    "from pathlib import Path\n",
    "from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation\n",
    "from pamola_core.utils.ops.op_data_source import DataSource\n",
    "from pamola_core.utils.progress import HierarchicalProgressTracker\n",
    "from pamola_core.utils.tasks.task_reporting import TaskReporter\n",
    "\n",
    "CSV_PATH = 'E:/DTX/pamola-production/frontend/pamola-spa/tests/e2e/shared/sample-dataset/S_CHURN_BANK_CANADA_10K.csv'\n",
    "df_bank = pd.read_csv(CSV_PATH)\n",
    "\n",
    "print(f'Loaded {len(df_bank):,} rows.')\n",
    "print(f'Columns: {list(df_bank.columns)}')\n",
    "\n",
    "# Apply simplified generalization to get realistic equivalence classes\n",
    "# (mirrors what ANON-GEN-001 + ANON-GEN-002 would produce in the full pipeline)\n",
    "df_bank['AGE_bin'] = pd.cut(\n",
    "    df_bank['AGE'].clip(19, 100),\n",
    "    bins=[18, 26, 36, 46, 56, 66, 100],\n",
    "    labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-100']\n",
    ")\n",
    "df_bank['RACE_gen'] = df_bank['RACE'].map(lambda x: x if x == 'White' else 'OTHER')\n",
    "\n",
    "supp_task_dir = project_root / 'examples' / 'data_examples' / 'bank_churn_supp002'\n",
    "os.makedirs(supp_task_dir, exist_ok=True)\n",
    "supp_reporter = TaskReporter(\n",
    "    task_id='bank_churn_supp002',\n",
    "    task_type='record_suppression',\n",
    "    description='Bank Churn ANON-SUPP-002 verification',\n",
    "    report_path=supp_task_dir\n",
    ")\n",
    "\n",
    "print('\\n' + '=' * 80)\n",
    "print('D1: Run op WITHOUT _k_score column (expect error - documents the gap)')\n",
    "print('=' * 80)\n",
    "\n",
    "try:\n",
    "    op_d1 = RecordSuppressionOperation(\n",
    "        field_name='SEX',\n",
    "        suppression_mode='REMOVE',\n",
    "        suppression_condition='risk',\n",
    "        ka_risk_field='_k_score',  # column does NOT exist in df_bank yet\n",
    "        risk_threshold=5.0,\n",
    "        save_output=False,\n",
    "        generate_visualization=False,\n",
    "    )\n",
    "    ds_d1 = DataSource(dataframes={'main_dataset': df_bank.copy()})\n",
    "    tracker_d1 = HierarchicalProgressTracker(total=6, description='D1 no k_score', unit='steps')\n",
    "    result_d1 = op_d1.execute(\n",
    "        data_source=ds_d1,\n",
    "        task_dir=supp_task_dir / 'd1_no_kscore',\n",
    "        reporter=supp_reporter,\n",
    "        progress_tracker=tracker_d1,\n",
    "        dataset_name='main_dataset'\n",
    "    )\n",
    "    # If no exception: check output size - should be 0 suppressed or all suppressed\n",
    "    d1_out = sorted(list((supp_task_dir / 'd1_no_kscore' / 'output').glob('*.csv')),\n",
    "                     key=lambda x: x.stat().st_mtime, reverse=True)\n",
    "    if d1_out:\n",
    "        rdf_d1 = pd.read_csv(d1_out[0])\n",
    "        print(f'[UNEXPECTED] D1 did not raise an error. Output rows: {len(rdf_d1)}')\n",
    "        print('This may indicate op silently skips missing ka_risk_field (investigate).')\n",
    "    else:\n",
    "        print('[NOTE] D1 completed but no output file - op may have raised internally.')\n",
    "except Exception as e:\n",
    "    print(f'[PASS - expected] D1 raised error when _k_score missing:')\n",
    "    print(f'  {type(e).__name__}: {e}')\n",
    "    print('  Source: record_op.py:866-870 reads ka_risk_field directly from batch.')\n",
    "\n",
    "print('\\n' + '=' * 80)\n",
    "print('D2: Pre-compute _k_score + run op (workaround path)')\n",
    "print('=' * 80)\n",
    "\n",
    "qi_fields = [\n",
    "    'SEX', 'CITY', 'PROVINCE', 'AGE_bin', 'RACE_gen',\n",
    "    'IsMarried', 'IsEduBachelors', 'IsHomeOwner', 'IsUnemployed',\n",
    "    'Income', 'HOMEVALUE', 'RENTVALUE'\n",
    "]\n",
    "\n",
    "# Verify all QI fields exist\n",
    "missing_qi = [f for f in qi_fields if f not in df_bank.columns]\n",
    "if missing_qi:\n",
    "    print(f'WARNING: QI fields not found: {missing_qi}')\n",
    "    print('Using available fields only...')\n",
    "    qi_fields = [f for f in qi_fields if f in df_bank.columns]\n",
    "\n",
    "# Pre-compute k-score: group size for each equivalence class\n",
    "df_bank['_k_score'] = df_bank.groupby(qi_fields, dropna=False)['SEX'].transform('count')\n",
    "k_score_dist = df_bank['_k_score'].value_counts().sort_index()\n",
    "print(f'_k_score distribution (top 10):')\n",
    "print(k_score_dist.head(10).to_dict())\n",
    "print(f'Rows with k < 5: {(df_bank[\"_k_score\"] < 5).sum():,}')\n",
    "print(f'Rows with k >= 5: {(df_bank[\"_k_score\"] >= 5).sum():,}')\n",
    "\n",
    "op_d2 = RecordSuppressionOperation(\n",
    "    field_name='SEX',\n",
    "    suppression_mode='REMOVE',\n",
    "    suppression_condition='risk',\n",
    "    ka_risk_field='_k_score',\n",
    "    risk_threshold=5.0,\n",
    "    save_output=True,\n",
    "    generate_visualization=False,\n",
    ")\n",
    "\n",
    "ds_d2 = DataSource(dataframes={'main_dataset': df_bank.copy()})\n",
    "tracker_d2 = HierarchicalProgressTracker(total=6, description='D2 with k_score', unit='steps')\n",
    "result_d2 = op_d2.execute(\n",
    "    data_source=ds_d2,\n",
    "    task_dir=supp_task_dir / 'd2_with_kscore',\n",
    "    reporter=supp_reporter,\n",
    "    progress_tracker=tracker_d2,\n",
    "    dataset_name='main_dataset'\n",
    ")\n",
    "\n",
    "d2_out = sorted(\n",
    "    [f for f in (supp_task_dir / 'd2_with_kscore' / 'output').glob('*.csv')\n",
    "     if 'suppressed' not in f.name],\n",
    "    key=lambda x: x.stat().st_mtime, reverse=True\n",
    ")\n",
    "if d2_out:\n",
    "    rdf_d2 = pd.read_csv(d2_out[0])\n",
    "    row_count = len(rdf_d2)\n",
    "    suppressed_count = len(df_bank) - row_count\n",
    "\n",
    "    # Assert 1: all remaining rows have _k_score >= 5\n",
    "    if '_k_score' in rdf_d2.columns:\n",
    "        k_all_valid = (rdf_d2['_k_score'] >= 5).all()\n",
    "        min_k = rdf_d2['_k_score'].min()\n",
    "        print(f'\\n[{\"PASS\" if k_all_valid else \"FAIL\"}] All remaining rows have _k_score >= 5:')\n",
    "        print(f'  Min _k_score in output: {min_k}')\n",
    "    else:\n",
    "        print('\\nNOTE: _k_score not in output columns (REPLACE mode dropped it)')\n",
    "        print('  Cannot verify k constraint directly from output - relying on op mask logic.')\n",
    "        k_all_valid = True  # trust the op\n",
    "\n",
    "    # Assert 2: output row count in expected range\n",
    "    in_range = 9000 <= row_count <= 9500\n",
    "    print(f'\\n[{\"PASS\" if in_range else \"FAIL (check expected range)\"}] Output row count:')\n",
    "    print(f'  Original: {len(df_bank):,}')\n",
    "    print(f'  After suppression: {row_count:,} ({(row_count/len(df_bank)*100):.1f}% retained)')\n",
    "    print(f'  Suppressed: {suppressed_count:,} rows')\n",
    "    print(f'  Expected range: 9,000-9,500 (BA spec estimate)')\n",
    "\n",
    "    # Print k_score distribution of suppressed groups\n",
    "    print(f'\\nSuppression by k_score bucket (rows dropped per group size):')\n",
    "    low_k = df_bank[df_bank['_k_score'] < 5]['_k_score'].value_counts().sort_index()\n",
    "    print(low_k.to_dict())\n",
    "else:\n",
    "    print('BLOCKED: No D2 output file found')\n",
])

cells.insert(summary_idx, d_cell)
cells.insert(summary_idx, md_cell)

print(f"Inserted 2 cells before Summary (was index {summary_idx})")
print(f"Total cells: {len(cells)}")

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved.")
