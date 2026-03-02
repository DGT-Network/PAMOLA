# PAMOLA Examples - Quick Start Guide

Welcome to PAMOLA.CORE examples! This folder contains interactive Jupyter notebooks demonstrating various data privacy operations and techniques.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Examples](#running-examples)
  - [VS Code (Recommended)](#running-in-vs-code)
  - [Jupyter Notebook UI](#running-in-jupyter-notebook-ui)
- [Available Examples](#available-examples)
- [Working with Examples](#working-with-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🔧 Prerequisites

### 1. Python Installation

**Required:** Python >= 3.8

Check your Python version:
```bash
python --version
```

**Installation:**
- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **macOS:** `brew install python3`
- **Linux:** `sudo apt-get install python3 python3-pip`

### 2. PAMOLA.CORE

Clone the repository:
```bash
git clone <repository-url>
cd PAMOLA
```

---

## 📦 Installation

### Step 1: Install Dependencies

From the PAMOLA root directory:

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 View full dependency list</summary>

```
python-decouple==3.8
pydantic[email]==2.12.4
matplotlib==3.8.2
seaborn==0.13.2
pandas==2.2.2
torch==2.8.0
scikit-learn==1.7.2
deepdiff==8.6.1
diff-match-patch==20241021
sdv==1.18.0
faker==33.3.1
scipy==1.16.3
uvicorn[standard]==0.38.0
openpyxl==3.1.5
xlrd==2.0.1
rstr==3.2.2
recordlinkage==0.16
plotly==6.4.0
kaleido==1.2.0
pillow==11.3.0
multidict==6.7.0
bcrypt==4.3.0
wordcloud==1.9.4
langdetect==1.0.9
fasttext-wheel==0.9.2
datasketch==1.7.0
nltk==3.9.2
dask[complete]==2025.11.0
pyarrow==14.0.2
numpy==1.26.4
pytest==8.4.2
matplotlib-venn==1.1.2
psutil==5.9.8
ijson==3.4.0.post0
chardet==5.2.0
rapidfuzz==3.14.3
cachetools==6.6.2
faiss-cpu==1.12.0
spacy==3.8.9
phonenumbers==9.0.10
base58==2.1.1
cryptography==46.0.3
PyYAML==6.0.2
```

</details>

### Step 2: Verify Installation

```bash
python -c "import pamola_core; print('✅ PAMOLA.CORE installed successfully')"
```

---

## 🚀 Running Examples

### Running in VS Code

**Recommended for:** Interactive development, debugging, and learning

#### Step 1: Install VS Code Extensions

Open **Visual Studio Code** and install:

**Required:**
- **Jupyter** (ms-toolsai.jupyter) by Microsoft
  - Enables running `.ipynb` notebooks in VS Code
  - Interactive cell execution
  - Variable explorer and debugging

**Recommended:**
- **Python** (ms-python.python) by Microsoft
  - IntelliSense and code completion
  - Linting and formatting
  - Debugging support

**Installation:**
1. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
2. Search for "Jupyter" and click Install
3. Search for "Python" and click Install
4. Restart VS Code

#### Step 2: Open Project

1. **File → Open Folder**
2. Select the **PAMOLA root folder** (not just examples/)
3. This ensures proper module imports

#### Step 3: Open a Notebook

1. Navigate to `examples/` folder in VS Code Explorer
2. Browse to the category you want (e.g., `anonymization/generalization/`)
3. Click on any `.ipynb` file

#### Step 4: Select Kernel

1. Click **"Select Kernel"** in top-right corner
2. Choose **"Python Environments"**
3. Select your Python interpreter (e.g., `Python 3.12.8`)

#### Step 5: Run Cells

**Methods:**
- **Run All:** Click "Run All" button or `Ctrl+Shift+Enter`
- **Run Cell:** Click ▶ button or `Shift+Enter`
- **Run & Insert Below:** `Alt+Enter`
- **Interactive:** Edit and re-run cells as needed

**VS Code Tips:**
- 🔍 **Variable Explorer:** View all variables in notebook
- 📝 **IntelliSense:** Auto-complete while typing
- 🐛 **Debug:** Set breakpoints in cells
- 📂 **Outline:** Navigate cells easily (`Ctrl+Shift+O`)

---

### Running in Jupyter Notebook UI

**Recommended for:** Traditional Jupyter experience

#### Step 1: Install Jupyter

```bash
pip install notebook
```

#### Step 2: Start Jupyter Server

```bash
cd /path/to/PAMOLA/examples
jupyter notebook
```

This will:
- Start server on `http://localhost:8888`
- Open browser automatically
- Show file explorer

#### Step 3: Navigate and Run

1. Browse folders in browser interface
2. Click on notebook file (`.ipynb`)
3. Run cells:
   - **Cell → Run All**
   - **Shift+Enter** to run current cell
   - **Alt+Enter** to run and insert below

#### Step 4: Stop Server

Press `Ctrl+C` in terminal and confirm with `y`

---

## 📚 Available Examples

### 🔐 Anonymization

Operations for protecting sensitive data while maintaining utility.

#### Generalization
**Location:** `examples/anonymization/generalization/`

**Notebooks:**

##### 1️⃣ Simple Example
**File:** `01_categorical_generalization_simple.ipynb`

- 🎯 **Level:** Beginner
- ⏱️ **Duration:** ~10 minutes
- 📖 **Topics:** 
  - Basic categorical generalization workflow
  - Using hierarchy dictionaries
  - Simple operation.execute() pattern
  - Single-field anonymization
- 🎓 **What you'll learn:**
  - Load and configure generalization operations
  - Apply hierarchy-based generalization
  - Understand privacy metrics
  - Export anonymized results

##### 2️⃣ Advanced Multi-Strategy Example
**File:** `02_categorical_generalization_advanced.ipynb`

- 🎯 **Level:** Advanced
- ⏱️ **Duration:** ~30 minutes
- 📖 **Topics:**
  - **Strategy 1:** Hierarchy-based generalization with external dictionaries
  - **Strategy 2:** Frequency-based category grouping
  - **Strategy 3:** Merge low-frequency categories
  - Calculate k-anonymity and l-diversity
  - Analyze information loss
  - Design multi-stage privacy pipelines
- 🎓 **What you'll learn:**
  - Compare different generalization strategies
  - Calculate and interpret privacy metrics
  - Balance privacy vs. utility trade-offs
  - Chain multiple operations in production pipelines
  - Export results from each strategy

**Use Cases:**
- Generalizing location data (City → State → Country)
- Occupation hierarchies (Job Title → Industry → Sector)
- Product categories (SKU → Category → Department)
- Age ranges (Exact Age → Age Group → Broad Range)
- Income brackets (Salary → Income Range → Income Level)

---

### 📊 Coming Soon

More example categories will be added in future releases:

- **Data Profiling** - Quality assessment and statistical analysis
- **Data Transformation** - Format conversions and standardization
- **Synthetic Data Generation** - Creating realistic test data
- **Record Linkage** - Entity matching and deduplication

---

## 📁 Example Datasets

Sample datasets are provided in `examples/data_examples/`:

| File | Description | Records | Use Case |
|------|-------------|---------|----------|
| `category_hierarchy_data.csv` | Categorical data with hierarchies | 1,000 | Generalization examples |
| `category_hierarchy.json` | Hierarchy dictionary | - | Generalization mapping |
| `sample.csv` | Simple test dataset | 100 | Quick testing |

**Data Format Example:**
```csv
id,category,region,risk_score,gender,location,group
1,Fruit > Apple > Fuji,North,3.2,Male,City A1,Group A
2,Fruit > Apple > Gala,North,4.8,Female,City A2,Group B
```

---

## 🛠️ Working with Examples

### Modifying Examples

Examples are designed to be modified and experimented with:

1. **Change parameters:**
   ```python
   operation = CategoricalGeneralizationOperation(
       field_name="your_field",  # ← Modify this
       hierarchy_level=2,         # ← Adjust level
       freq_threshold=0.05        # ← Tune threshold
   )
   ```

2. **Use your own data:**
   ```python
   df = pd.read_csv("your_data.csv")
   ```

3. **Combine operations:**
   ```python
   # Chain multiple operations
   result1 = operation1.execute(...)
   result2 = operation2.execute(result1, ...)
   ```

### Best Practices

✅ **Do:**
- Read through notebooks before running
- Experiment with different parameters
- Check output files generated
- Review visualizations and metrics

❌ **Don't:**
- Skip prerequisite notebooks
- Modify core library code without understanding
- Use production data without backup
- Ignore error messages

### Output Files

Notebooks generate outputs in subfolders:

```
examples/data_examples/
├── advanced_output/           # Multi-strategy outputs
│   ├── strategy1_hierarchy/
│   ├── strategy2_frequency/
│   └── strategy3_merge/
└── simple_output/             # Simple output
```

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><strong>❌ "Module 'pamola_core' not found"</strong></summary>

**Solution:**
```bash
# Install from root directory
cd /path/to/PAMOLA
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

**VS Code:** Make sure you opened the PAMOLA root folder, not just examples/
</details>

<details>
<summary><strong>❌ "No kernel found" in VS Code</strong></summary>

**Solution:**
```bash
# Install ipykernel
pip install ipykernel

# Register kernel
python -m ipykernel install --user --name=pamola_env

# Restart VS Code and select kernel
```
</details>

<details>
<summary><strong>❌ "Permission denied" when saving files</strong></summary>

**Solution:**
- Close files if open in other programs (Excel, etc.)
- Check folder permissions
- Run with appropriate privileges
</details>

<details>
<summary><strong>❌ "Circular reference detected" error</strong></summary>

**Solution:**

**Option 1:** Use correct hierarchy format (recommended)
- See `examples/data_examples/category_hierarchy.json`
- Use `level_1`, `level_2`, `level_3` keys
- Avoid key names in hierarchy values

**Option 2:** Disable check temporarily
```python
# In pamola_core/utils/hierarchy_dict.py
def _check_circular_reference(self, value, visited=None):
    return False  # Temporarily disable
```
</details>

<details>
<summary><strong>⚠️ Slow performance or memory errors</strong></summary>

**Solutions:**
```python
# Enable vectorization
operation = Operation(use_vectorization=True)

# Disable cache for large datasets
operation = Operation(use_cache=False)

# Skip unnecessary features
operation = Operation(
    generate_visualization=False,
    privacy_check_enabled=False
)

# Use categorical dtype
df['column'] = df['column'].astype('category')
```
</details>

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pamola_core; print('OK')"

# Check Python version
python --version

# Start Jupyter Notebook
cd examples
jupyter notebook

# Run single notebook (from command line)
jupyter nbconvert --execute --to notebook notebook.ipynb
```

### Project Structure

```
PAMOLA/
├── examples/                         # ← Example notebooks
│   ├── README.md                     # This file
│   ├── anonymization/               # Anonymization examples
│   │   └── generalization/
│   │       ├── 01_categorical_generalization_simple.ipynb
│   │       └── 02_categorical_generalization_advanced.ipynb
│   └── data_examples/               # Sample data & outputs
│       ├── category_hierarchy_data.csv
│       ├── category_hierarchy.json
│       ├── sample.csv
│       ├── simple_output/
│       └── advanced_output/
├── pamola_core/                     # Core library
├── docs/                            # Documentation
├── tests/                           # Unit tests
└── requirements.txt                 # Dependencies
```

### Useful Shortcuts (VS Code)

| Action | Shortcut |
|--------|----------|
| Run Cell | `Shift+Enter` |
| Run All | `Ctrl+Shift+Enter` |
| Insert Cell Below | `B` |
| Delete Cell | `D D` |
| Cell Mode | `Esc` |
| Edit Mode | `Enter` |
| Command Palette | `Ctrl+Shift+P` |

---

## 📖 Additional Resources

### Documentation
- **API Reference:** `../docs/api/`
- **Architecture:** `../docs/architecture/`
- **User Guide:** `../docs/user_guide/`

### Core Modules
- **Anonymization:** `../pamola_core/anonymization/`
- **Profiling:** `../pamola_core/profiling/`
- **Utilities:** `../pamola_core/utils/`

### Community
- **GitHub Issues:** Report bugs and request features
- **Discussions:** Ask questions and share ideas
- **Contributing:** See `CONTRIBUTING.md` for guidelines

---

## 🤝 Contributing

We welcome contributions! Ways to contribute:

1. **Add new examples:**
   - Create notebooks in appropriate category
   - Include clear documentation
   - Provide sample data
   - Follow existing notebook structure

2. **Improve existing examples:**
   - Fix errors or typos
   - Add explanations
   - Optimize code
   - Update for new features

3. **Share your use cases:**
   - Submit real-world examples
   - Document lessons learned
   - Contribute sample datasets (anonymized)

**Guidelines:**
- Follow PEP 8 style guide
- Add docstrings and comments
- Test notebooks before submitting
- Update README if adding new categories

---

## 🚀 Getting Started Checklist

- [ ] Install Python 3.8+
- [ ] Install PAMOLA dependencies
- [ ] Install VS Code + Jupyter extension (or Jupyter Notebook)
- [ ] Open PAMOLA root folder in VS Code
- [ ] Run simple generalization example
- [ ] Explore advanced multi-strategy example
- [ ] Try with your own data
- [ ] Share feedback or contribute

---

## 📝 Notes

- **Always open PAMOLA root folder** in VS Code for proper imports
- **Python 3.8+** required for all features
- **Sample data** provided in `data_examples/` - don't use production data directly
- **Output files** are generated in `data_examples/` subfolders
- **Notebooks are self-contained** - can run independently
- **Check troubleshooting section** if you encounter issues

---

## 🎓 Learning Path

**Beginner:**
1. Start with `anonymization/generalization/01_categorical_generalization_simple.ipynb`
2. Understand basic concepts and workflow
3. Experiment with provided sample data

**Intermediate:**
4. Move to `02_categorical_generalization_advanced.ipynb`
5. Learn multi-strategy approaches
6. Understand privacy metrics

**Advanced:**
7. Combine multiple operations
8. Use your own datasets
9. Customize for your use cases
10. Contribute back to the project

---

**Happy Learning! 🎉**

For questions or issues, please refer to our documentation or open an issue on GitHub.

---

*Last updated: November 2025*
*PAMOLA.CORE version: 1.0.0*