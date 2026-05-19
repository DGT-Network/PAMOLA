"""Patch 02_numeric_generalization_advanced.ipynb - insert Step 8 before Summary."""
import json
import uuid
import os

NOTEBOOK = "E:/DTX/PAMOLA/examples/anonymization/generalization/02_numeric_generalization_advanced.ipynb"


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

# Remove any previously inserted Step 8 cells to stay idempotent
cells[:] = [
    c for c in cells
    if "Step 8: Bank Churn Happy Path Workaround Verification (ANON-GEN-001)" not in "".join(c["source"])
    and "B1: Workaround verify" not in "".join(c["source"])
    and "B2: Boundary off-by-one gap" not in "".join(c["source"])
]

# Find Summary cell index
summary_idx = next(
    i for i, c in enumerate(cells)
    if "Summary" in "".join(c["source"]) and "Accomplished" in "".join(c["source"])
)
print(f"Summary cell at index: {summary_idx}")

md_cell = make_md_cell([
    "## Step 8: Bank Churn Happy Path Workaround Verification (ANON-GEN-001)\n",
    "\n",
    "Verify workaround `range_limits` + pre-clip pattern per `docs/ba-artifacts/happy_path_op_config_workarounds.md`.\n",
    "Per-field bins from BA spec `TITAN_PAMOLA_PROD_HAPPY_PATH_2026-05-17.md` Op 3.3-3.6.\n",
    "\n",
    "**B1:** Per-field bins on real data - verify no `\"<min\"`/`\">=max\"` labels after clip  \n",
    "**B2:** Boundary off-by-one micro-df - reproduce known gap at `numeric_op.py:1008-1013`",
])

b1_cell = make_code_cell([
    "# B1: Workaround verify - per-field bins on real Bank Churn data\n",
    "import pandas as pd\n",
    "import os, sys, time\n",
    "from pathlib import Path\n",
    "\n",
    "CSV_PATH = 'E:/DTX/pamola-production/frontend/pamola-spa/tests/e2e/shared/sample-dataset/S_CHURN_BANK_CANADA_10K.csv'\n",
    "df_bank = pd.read_csv(CSV_PATH)\n",
    "\n",
    "print(f'Loaded {len(df_bank):,} rows.')\n",
    "print(f'AGE range: {df_bank[\"AGE\"].min()} - {df_bank[\"AGE\"].max()}')\n",
    "print(f'Income range: {df_bank[\"Income\"].min()} - {df_bank[\"Income\"].max()}')\n",
    "print(f'HOMEVALUE nulls: {df_bank[\"HOMEVALUE\"].isnull().sum()}')\n",
    "print(f'RENTVALUE nulls: {df_bank[\"RENTVALUE\"].isnull().sum()}')\n",
    "\n",
    "# Pre-clip per workaround doc (lower=range_min+1 to avoid '<min' bin)\n",
    "df_bank['AGE'] = df_bank['AGE'].clip(lower=19, upper=100)\n",
    "df_bank['Income'] = df_bank['Income'].clip(lower=1, upper=1_000_000)\n",
    "df_bank['HOMEVALUE'] = df_bank['HOMEVALUE'].clip(lower=1, upper=10_000_000)\n",
    "df_bank['RENTVALUE'] = df_bank['RENTVALUE'].clip(lower=1, upper=50_000)\n",
    "\n",
    "# Per-field bins per BA spec\n",
    "BINS = {\n",
    "    'AGE':       [[18,25],[26,35],[36,45],[46,55],[56,65],[66,100]],\n",
    "    'Income':    [[0,30000],[30000,60000],[60000,100000],[100000,200000],[200000,1000000]],\n",
    "    'HOMEVALUE': [[0,200000],[200000,500000],[500000,1000000],[1000000,2000000],[2000000,10000000]],\n",
    "    'RENTVALUE': [[0,500],[500,1000],[1000,2000],[2000,5000],[5000,50000]],\n",
    "}\n",
    "\n",
    "from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation\n",
    "from pamola_core.utils.ops.op_data_source import DataSource\n",
    "from pamola_core.utils.progress import HierarchicalProgressTracker\n",
    "from pamola_core.utils.tasks.task_reporting import TaskReporter\n",
    "\n",
    "b1_task_dir = project_root / 'examples' / 'data_examples' / 'bank_churn_gen001'\n",
    "os.makedirs(b1_task_dir, exist_ok=True)\n",
    "b1_reporter = TaskReporter(\n",
    "    task_id='bank_churn_gen001',\n",
    "    task_type='numeric_generalization',\n",
    "    description='Bank Churn ANON-GEN-001 verification',\n",
    "    report_path=b1_task_dir\n",
    ")\n",
    "\n",
    "results_b1 = {}\n",
    "for field, bins in BINS.items():\n",
    "    valid_labels = {f'{b[0]}-{b[1]}' for b in bins}\n",
    "    df_field = df_bank[[field]].copy()\n",
    "    null_strat = 'PRESERVE' if field in ('HOMEVALUE', 'RENTVALUE') else 'EXCLUDE'\n",
    "    op = NumericGeneralizationOperation(\n",
    "        field_name=field,\n",
    "        mode='REPLACE',\n",
    "        strategy='range',\n",
    "        range_limits=bins,\n",
    "        null_strategy=null_strat,\n",
    "        save_output=True,\n",
    "        generate_visualization=False,\n",
    "    )\n",
    "    ds = DataSource(dataframes={'main_dataset': df_field})\n",
    "    tracker = HierarchicalProgressTracker(total=6, description=f'Gen {field}', unit='steps')\n",
    "    op.execute(\n",
    "        data_source=ds,\n",
    "        task_dir=b1_task_dir / f'field_{field.lower()}',\n",
    "        reporter=b1_reporter,\n",
    "        progress_tracker=tracker,\n",
    "        dataset_name='main_dataset'\n",
    "    )\n",
    "    out_files = sorted(\n",
    "        list((b1_task_dir / f'field_{field.lower()}' / 'output').glob('*.csv')),\n",
    "        key=lambda x: x.stat().st_mtime, reverse=True\n",
    "    )\n",
    "    if out_files:\n",
    "        rdf = pd.read_csv(out_files[0])\n",
    "        output_vals = rdf[field].dropna().unique().tolist()\n",
    "        bad_labels = [v for v in output_vals if str(v).startswith('<') or str(v).startswith('>=')]\n",
    "        results_b1[field] = {'unique_labels': sorted(str(v) for v in output_vals), 'bad_labels': bad_labels, 'label_counts': rdf[field].value_counts().to_dict()}\n",
    "        status = 'PASS' if not bad_labels else 'FAIL'\n",
    "        print(f'\\n[{status}] {field}:')\n",
    "        print(f'  Labels: {sorted(str(v) for v in output_vals)}')\n",
    "        if bad_labels:\n",
    "            print(f'  BAD labels: {bad_labels}')\n",
    "        print(f'  Distribution: {rdf[field].value_counts().to_dict()}')\n",
    "    else:\n",
    "        print(f'BLOCKED: No output for {field}')\n",
])

b2_cell = make_code_cell([
    "# B2: Boundary off-by-one gap demo (numeric_op.py:1008-1013)\n",
    "# Do NOT clip here - show raw behavior to prove the gap exists\n",
    "import pandas as pd, tempfile\n",
    "from pathlib import Path\n",
    "from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation\n",
    "from pamola_core.utils.ops.op_data_source import DataSource\n",
    "from pamola_core.utils.progress import HierarchicalProgressTracker\n",
    "\n",
    "print('=' * 80)\n",
    "print('B2: Boundary off-by-one gap - raw behavior without clip')\n",
    "print('=' * 80)\n",
    "\n",
    "micro_df = pd.DataFrame({'AGE': [18, 26, 36, 46, 56, 66, 100]})\n",
    "age_bins = [[18, 25], [26, 35], [36, 45], [46, 55], [56, 65], [66, 100]]\n",
    "\n",
    "op_b2 = NumericGeneralizationOperation(\n",
    "    field_name='AGE',\n",
    "    mode='REPLACE',\n",
    "    strategy='range',\n",
    "    range_limits=age_bins,\n",
    "    null_strategy='EXCLUDE',\n",
    "    save_output=True,\n",
    "    generate_visualization=False,\n",
    ")\n",
    "\n",
    "b2_tmp_dir = Path(tempfile.mkdtemp())\n",
    "b2_reporter_none = None\n",
    "ds_b2 = DataSource(dataframes={'main_dataset': micro_df})\n",
    "tracker_b2 = HierarchicalProgressTracker(total=6, description='B2 micro', unit='steps')\n",
    "try:\n",
    "    op_b2.execute(\n",
    "        data_source=ds_b2,\n",
    "        task_dir=b2_tmp_dir,\n",
    "        reporter=b2_reporter_none,\n",
    "        progress_tracker=tracker_b2,\n",
    "        dataset_name='main_dataset'\n",
    "    )\n",
    "    b2_out_files = sorted(list((b2_tmp_dir / 'output').glob('*.csv')),\n",
    "                           key=lambda x: x.stat().st_mtime, reverse=True)\n",
    "    if b2_out_files:\n",
    "        result_b2 = pd.read_csv(b2_out_files[0])\n",
    "        expected_map = {18: '18-25', 26: '26-35', 36: '36-45', 46: '46-55', 56: '56-65', 66: '66-100', 100: '66-100'}\n",
    "        age_vals = micro_df['AGE'].tolist()\n",
    "        output_labels = result_b2['AGE'].tolist()\n",
    "        print(f'{\"Input\":>8} | {\"Output\":>12} | {\"Expected\":>12} | Result')\n",
    "        print('-' * 52)\n",
    "        for age_val, label in zip(age_vals, output_labels):\n",
    "            expected = expected_map[age_val]\n",
    "            match = 'PASS' if str(label) == expected else 'FAIL (off-by-one)'\n",
    "            print(f'{age_val:>8} | {str(label):>12} | {expected:>12} | {match}')\n",
    "        print()\n",
    "        print('Root cause: numeric_op.py:1008-1013 builds right-closed pd.cut from range_min only.')\n",
    "        print('AGE=18 -> bin (-inf,18] -> label \"<18\" (not \"18-25\").')\n",
    "        print('AGE=100 -> bin (66,100] -> label \"66-100\" (correct - upper bound works).')\n",
    "        print('Workaround: clip(lower=19) avoids all lower-boundary values.')\n",
    "    else:\n",
    "        print('WARNING: No B2 output files found')\n",
    "except Exception as e:\n",
    "    print(f'ERROR in B2: {e}')\n",
    "    import traceback; traceback.print_exc()\n",
])

# Insert 3 cells before Summary
cells.insert(summary_idx, b2_cell)
cells.insert(summary_idx, b1_cell)
cells.insert(summary_idx, md_cell)

print(f"Inserted 3 cells before Summary (was index {summary_idx})")
print(f"Total cells: {len(cells)}")

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved.")
