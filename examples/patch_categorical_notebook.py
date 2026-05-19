"""Patch 02_categorical_generalization_advanced.ipynb - insert Step 9 before Summary."""
import json
import uuid

NOTEBOOK = "E:/DTX/PAMOLA/examples/anonymization/generalization/02_categorical_generalization_advanced.ipynb"
HIER_RACE = "E:/DTX/PAMOLA/examples/anonymization/generalization/hierarchy_race.json"
HIER_CITY = "E:/DTX/PAMOLA/examples/anonymization/generalization/hierarchy_city.json"


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

# Remove any previously inserted Step 9 cells (idempotent)
cells[:] = [
    c for c in cells
    if "Step 9: Bank Churn Happy Path Workaround Verification (ANON-GEN-002)" not in "".join(c["source"])
    and "C1: levels key experiment" not in "".join(c["source"])
    and "C2: Two-layer mapping on real Bank Churn data" not in "".join(c["source"])
]

summary_idx = next(
    i for i, c in enumerate(cells)
    if "Summary" in "".join(c["source"]) and "Accomplished" in "".join(c["source"])
)
print(f"Summary cell at index: {summary_idx}")

md_cell = make_md_cell([
    "## Step 9: Bank Churn Happy Path Workaround Verification (ANON-GEN-002)\n",
    "\n",
    "Verify workaround for `CategoricalGeneralizationOperation` per `docs/ba-artifacts/happy_path_op_config_workarounds.md`.\n",
    "\n",
    "**C1:** Critical `\"levels\"` key requirement in hierarchy JSON - broken vs correct file  \n",
    "**C2:** Two-layer mapping on real Bank Churn RACE + CITY data  \n",
    "- Layer 1: explicit hierarchy entries (White/Asian/Black/Latino)  \n",
    "- Layer 2: `allow_unknown` fallback (Native/Others/mwugja7p)  \n",
    "- CITY: by-design ~3,300 rows fall through `\"Other\"` (only 21 cities mapped)",
])

c1_cell = make_code_cell([
    "# C1: levels key experiment - broken vs correct hierarchy file\n",
    "import json, tempfile, os, sys\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation\n",
    "from pamola_core.utils.ops.op_data_source import DataSource\n",
    "from pamola_core.utils.progress import HierarchicalProgressTracker\n",
    "from pamola_core.utils.tasks.task_reporting import TaskReporter\n",
    "\n",
    "# Broken hierarchy - missing 'levels' key\n",
    "hierarchy_broken = {\n",
    "    'White': 'White', 'Asian': 'OTHER', 'Black': 'OTHER',\n",
    "    'Latino': 'OTHER', 'Hispanic': 'OTHER', 'Indigenous': 'OTHER', 'Mixed': 'OTHER'\n",
    "}\n",
    "\n",
    "# Correct hierarchy - has 'levels' key\n",
    "hierarchy_correct = {\n",
    "    'levels': ['category'],\n",
    "    'White': 'White', 'Asian': 'OTHER', 'Black': 'OTHER',\n",
    "    'Latino': 'OTHER', 'Hispanic': 'OTHER', 'Indigenous': 'OTHER', 'Mixed': 'OTHER'\n",
    "}\n",
    "\n",
    "# Save to temp files\n",
    "tmp_dir = Path(tempfile.mkdtemp())\n",
    "broken_path = tmp_dir / 'hierarchy_race_broken.json'\n",
    "correct_path = tmp_dir / 'hierarchy_race_correct.json'\n",
    "broken_path.write_text(json.dumps(hierarchy_broken))\n",
    "correct_path.write_text(json.dumps(hierarchy_correct))\n",
    "\n",
    "# Small test df with known RACE values\n",
    "df_race_test = pd.DataFrame({'RACE': ['White', 'Asian', 'Black', 'White', 'Latino', 'White']})\n",
    "print(f'Test input RACE values: {df_race_test[\"RACE\"].tolist()}')\n",
    "\n",
    "c1_task_dir = project_root / 'examples' / 'data_examples' / 'bank_churn_gen002_c1'\n",
    "os.makedirs(c1_task_dir, exist_ok=True)\n",
    "c1_reporter = TaskReporter(\n",
    "    task_id='gen002_c1',\n",
    "    task_type='categorical_generalization',\n",
    "    description='C1 levels key experiment',\n",
    "    report_path=c1_task_dir\n",
    ")\n",
    "\n",
    "def run_cat_op(hier_path, label, task_subdir):\n",
    "    op = CategoricalGeneralizationOperation(\n",
    "        field_name='RACE',\n",
    "        mode='ENRICH',\n",
    "        output_field_name='RACE_gen',\n",
    "        strategy='hierarchy',\n",
    "        external_dictionary_path=str(hier_path),\n",
    "        allow_unknown=True,\n",
    "        unknown_value='OTHER',\n",
    "        save_output=True,\n",
    "        generate_visualization=False,\n",
    "    )\n",
    "    ds = DataSource(dataframes={'main_dataset': df_race_test.copy()})\n",
    "    tracker = HierarchicalProgressTracker(total=6, description=f'C1 {label}', unit='steps')\n",
    "    sub = c1_task_dir / task_subdir\n",
    "    os.makedirs(sub, exist_ok=True)\n",
    "    op.execute(\n",
    "        data_source=ds,\n",
    "        task_dir=sub,\n",
    "        reporter=c1_reporter,\n",
    "        progress_tracker=tracker,\n",
    "        dataset_name='main_dataset'\n",
    "    )\n",
    "    out_files = sorted(list((sub / 'output').glob('*.csv')),\n",
    "                        key=lambda x: x.stat().st_mtime, reverse=True)\n",
    "    if out_files:\n",
    "        return pd.read_csv(out_files[0])\n",
    "    return None\n",
    "\n",
    "print('\\n' + '=' * 80)\n",
    "print('C1a: Broken hierarchy (no levels key)')\n",
    "print('=' * 80)\n",
    "try:\n",
    "    result_broken = run_cat_op(broken_path, 'broken', 'broken')\n",
    "    if result_broken is not None and 'RACE_gen' in result_broken.columns:\n",
    "        white_wrong = (result_broken.loc[result_broken['RACE'] == 'White', 'RACE_gen'] == 'OTHER').all()\n",
    "        print(f'Output RACE_gen: {result_broken[\"RACE_gen\"].tolist()}')\n",
    "        print(f'[{\"PASS\" if white_wrong else \"FAIL\"}] White mapped to OTHER with broken file: {white_wrong}')\n",
    "        print('Expected: ALL values -> OTHER because levels=[] guard fails at hierarchy_dictionary.py:282')\n",
    "    else:\n",
    "        print('WARNING: No RACE_gen column in broken output')\n",
    "except Exception as e:\n",
    "    print(f'BROKEN test error: {e}')\n",
    "\n",
    "print('\\n' + '=' * 80)\n",
    "print('C1b: Correct hierarchy (with levels key)')\n",
    "print('=' * 80)\n",
    "try:\n",
    "    result_correct = run_cat_op(correct_path, 'correct', 'correct')\n",
    "    if result_correct is not None and 'RACE_gen' in result_correct.columns:\n",
    "        white_preserved = (result_correct.loc[result_correct['RACE'] == 'White', 'RACE_gen'] == 'White').all()\n",
    "        others_mapped = (result_correct.loc[result_correct['RACE'] != 'White', 'RACE_gen'] == 'OTHER').all()\n",
    "        print(f'Output RACE_gen: {result_correct[\"RACE_gen\"].tolist()}')\n",
    "        print(f'[{\"PASS\" if white_preserved else \"FAIL\"}] White preserved as White: {white_preserved}')\n",
    "        print(f'[{\"PASS\" if others_mapped else \"FAIL\"}] Non-White mapped to OTHER: {others_mapped}')\n",
    "    else:\n",
    "        print('WARNING: No RACE_gen column in correct output')\n",
    "except Exception as e:\n",
    "    print(f'CORRECT test error: {e}')\n",
])

c2_cell = make_code_cell([
    "# C2: Two-layer mapping on real Bank Churn data (RACE + CITY)\n",
    "import pandas as pd\n",
    "import os\n",
    "from pathlib import Path\n",
    "from pamola_core.anonymization.generalization.categorical_op import CategoricalGeneralizationOperation\n",
    "from pamola_core.utils.ops.op_data_source import DataSource\n",
    "from pamola_core.utils.progress import HierarchicalProgressTracker\n",
    "from pamola_core.utils.tasks.task_reporting import TaskReporter\n",
    "\n",
    "CSV_PATH = 'E:/DTX/pamola-production/frontend/pamola-spa/tests/e2e/shared/sample-dataset/S_CHURN_BANK_CANADA_10K.csv'\n",
    "HIER_RACE = 'E:/DTX/PAMOLA/examples/anonymization/generalization/hierarchy_race.json'\n",
    "HIER_CITY = 'E:/DTX/PAMOLA/examples/anonymization/generalization/hierarchy_city.json'\n",
    "\n",
    "df_bank = pd.read_csv(CSV_PATH)\n",
    "print('RACE distribution:', df_bank['RACE'].value_counts().to_dict())\n",
    "print('Expected: White(5831), Asian(2635), Black(648), Native(315), Others(306), Latino(264), mwugja7p(1)')\n",
    "\n",
    "c2_task_dir = project_root / 'examples' / 'data_examples' / 'bank_churn_gen002_c2'\n",
    "os.makedirs(c2_task_dir, exist_ok=True)\n",
    "c2_reporter = TaskReporter(\n",
    "    task_id='gen002_c2',\n",
    "    task_type='categorical_generalization',\n",
    "    description='C2 two-layer mapping on real data',\n",
    "    report_path=c2_task_dir\n",
    ")\n",
    "\n",
    "# --- RACE ---\n",
    "print('\\n' + '=' * 80)\n",
    "print('C2a: RACE generalization (two-layer: explicit + unknown fallback)')\n",
    "print('=' * 80)\n",
    "op_race = CategoricalGeneralizationOperation(\n",
    "    field_name='RACE',\n",
    "    mode='ENRICH',\n",
    "    output_field_name='RACE_gen',\n",
    "    strategy='hierarchy',\n",
    "    external_dictionary_path=HIER_RACE,\n",
    "    allow_unknown=True,\n",
    "    unknown_value='OTHER',\n",
    "    save_output=True,\n",
    "    generate_visualization=False,\n",
    ")\n",
    "ds_race = DataSource(dataframes={'main_dataset': df_bank[['RACE']].copy()})\n",
    "tracker_race = HierarchicalProgressTracker(total=6, description='C2 RACE', unit='steps')\n",
    "op_race.execute(\n",
    "    data_source=ds_race,\n",
    "    task_dir=c2_task_dir / 'race',\n",
    "    reporter=c2_reporter,\n",
    "    progress_tracker=tracker_race,\n",
    "    dataset_name='main_dataset'\n",
    ")\n",
    "race_out = sorted(list((c2_task_dir / 'race' / 'output').glob('*.csv')),\n",
    "                   key=lambda x: x.stat().st_mtime, reverse=True)\n",
    "if race_out:\n",
    "    rdf_race = pd.read_csv(race_out[0])\n",
    "    race_dist = rdf_race['RACE_gen'].value_counts().to_dict()\n",
    "    print(f'RACE_gen distribution: {race_dist}')\n",
    "    white_count = race_dist.get('White', 0)\n",
    "    other_count = race_dist.get('OTHER', 0)\n",
    "    white_pass = white_count == 5831\n",
    "    other_pass = other_count == (10000 - 5831)  # ~4169\n",
    "    print(f'[{\"PASS\" if white_pass else \"FAIL\"}] White preserved: {white_count} (expected 5831)')\n",
    "    print(f'[{\"PASS\" if other_pass else \"FAIL\"}] All non-White -> OTHER: {other_count} (expected 4169)')\n",
    "    # Check Layer 2 fallback - Native/Others/mwugja7p not in hierarchy dict\n",
    "    unknown_in_hierarchy = {'Native', 'Others', 'mwugja7p'}\n",
    "    # Verify these unknown values all mapped to OTHER\n",
    "    for unknown_val in unknown_in_hierarchy:\n",
    "        rows = rdf_race[rdf_race['RACE'] == unknown_val]\n",
    "        if not rows.empty:\n",
    "            mapped = rows['RACE_gen'].unique().tolist()\n",
    "            print(f'  Layer 2 fallback: {unknown_val} ({len(rows)} rows) -> {mapped}')\n",
    "else:\n",
    "    print('WARNING: No RACE output')\n",
    "\n",
    "# --- CITY ---\n",
    "print('\\n' + '=' * 80)\n",
    "print('C2b: CITY generalization (by-design fallback coverage gap)')\n",
    "print('=' * 80)\n",
    "op_city = CategoricalGeneralizationOperation(\n",
    "    field_name='CITY',\n",
    "    mode='ENRICH',\n",
    "    output_field_name='CITY_region',\n",
    "    strategy='hierarchy',\n",
    "    external_dictionary_path=HIER_CITY,\n",
    "    allow_unknown=True,\n",
    "    unknown_value='Other',\n",
    "    save_output=True,\n",
    "    generate_visualization=False,\n",
    ")\n",
    "ds_city = DataSource(dataframes={'main_dataset': df_bank[['CITY']].copy()})\n",
    "tracker_city = HierarchicalProgressTracker(total=6, description='C2 CITY', unit='steps')\n",
    "op_city.execute(\n",
    "    data_source=ds_city,\n",
    "    task_dir=c2_task_dir / 'city',\n",
    "    reporter=c2_reporter,\n",
    "    progress_tracker=tracker_city,\n",
    "    dataset_name='main_dataset'\n",
    ")\n",
    "city_out = sorted(list((c2_task_dir / 'city' / 'output').glob('*.csv')),\n",
    "                   key=lambda x: x.stat().st_mtime, reverse=True)\n",
    "if city_out:\n",
    "    rdf_city = pd.read_csv(city_out[0])\n",
    "    city_dist = rdf_city['CITY_region'].value_counts().to_dict()\n",
    "    print(f'CITY_region distribution:')\n",
    "    for region, count in sorted(city_dist.items(), key=lambda x: -x[1]):\n",
    "        marker = ' <- by-design fallback' if region == 'Other' else ''\n",
    "        print(f'  {region:20s}: {count:,}{marker}')\n",
    "    other_count_city = city_dist.get('Other', 0)\n",
    "    other_pass_city = 2500 <= other_count_city <= 4000  # expected ~3300\n",
    "    mapped_count = sum(v for k, v in city_dist.items() if k != 'Other')\n",
    "    print(f'[{\"PASS\" if other_pass_city else \"FAIL\"}] Other fallback bucket: {other_count_city} rows (expected ~3300)')\n",
    "    print(f'  Mapped to explicit regions: {mapped_count} rows (~6700 top-21 cities)')\n",
    "    print(f'  Total unique cities in data: {df_bank[\"CITY\"].nunique()}')\n",
    "    print(f'  Cities in hierarchy: 21 (by design - remaining fall to Other)')\n",
    "else:\n",
    "    print('WARNING: No CITY output')\n",
])

cells.insert(summary_idx, c2_cell)
cells.insert(summary_idx, c1_cell)
cells.insert(summary_idx, md_cell)

print(f"Inserted 3 cells before Summary (was index {summary_idx})")
print(f"Total cells: {len(cells)}")

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved.")
