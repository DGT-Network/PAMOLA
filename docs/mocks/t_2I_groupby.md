
# Task Specification: `t_2I_GroupBy.py`

---

## Project Context

**Project:** PAMOLA.CORE (Resume Anonymization)  

**Task ID:** `t_2I` (must be explicitly set at the start; all folders, files, configs, and outputs derive from this ID)  
**Script Location:** `{project_root}\scripts\mock\t_2I_GroupBy.py`

This task defines the development of a Python script `t_2I_GroupBy.py` intended to perform grouping and aggregation operations on the output datasets produced by the previous task `t_1I_scissors.py`. It is part of the mock pipeline for controlled testing of specific functionalities in the PAMOLA.CORE project.

---

## Purpose

The script `t_2I_GroupBy.py` is designed to aggregate grouped datasets, compute detailed variation metrics, and prepare both short and long tables for downstream anonymization and reconstruction tasks. It ensures data readiness for later pseudonymization, generalization, suppression, and LLM rewriting steps.

---

## Clarified Details and Updated Algorithms

### 1️⃣ How is the `RESUME_VARIANCE` index calculated?

- For each group (by `resume_id`):
    
    - Compare each non-ID field across rows:
        
        - If numeric or date → assign difference score **3** if any value differs.
            
        - If list-type (detected by surrounding `[]`) → assign difference score **2** if any value differs.
            
        - If text → assign difference score **1** if any value differs.
            
    - Sum the difference scores **per row** (ignoring `ID` and `resume_id`).
        
    - Normalize by the total number of compared fields.
        
    - Aggregate normalized row scores to get the final `RESUME_VARIANCE` for the group (range 0–1: 0 = identical, 1 = fully varied).
        

### 2️⃣ How do we handle string vs. numeric fields?

- Apply weights:
    
    - Numeric/date → **3**
        
    - List fields → **2**
        
    - Text → **1**
        
- Always compare by equality, excluding `ID` and `resume_id`.
    

### 3️⃣ How are long tables (non-aggregated) handled?

- Tables like `ADDITIONAL_EDU`, `ATTESTATION`, `EXPIRIENCE`, `PRIMARY_EDU`, `SPECS`:
    
    - Compute `RESUME_VARIANCE` per group.
        
    - If **variance ≤ configurable threshold** (default: 0.2), aggregate by keeping representative rows (e.g., last non-null or most common).
        
    - If **variance > threshold**, leave all rows unaggregated.
        
    - Always save updated tables (with added metrics) to the output directory.
        
    - Ensure text qualifiers (`"`) are preserved when saving to CSV.
        

### 4️⃣ How are short tables grouped?

- Group by `resume_id`.
    
- **Do not embed long `IDS` lists**; instead, create and store a separate **mapping table** (`resume_id → [ID1; ID2; ID3]`).
    
- For all other fields, take the last non-null value in the group.
    
- If all values are null, retain as null or empty.
    

### 5️⃣ What is the output format?

- UTF-16 encoding.
    
- Comma `,` as the separator.
    
- Double-quote `"` as the text qualifier.
    
- All configurable via JSON.
    

### 6️⃣ What visualizations are needed?

- Heatmap: field-level variation per table.
    
- Bar chart: aggregated diversity metrics by table.
    
- Pie chart: distribution of grouped vs. ungrouped records.
    

### 7️⃣ How are short vs. long tables identified?

- Explicitly specified in the JSON config.
    
- Script should not hardcode names; follow config-driven patterns.
    

### 8️⃣ How should the script process input?

- Process each table independently:
    
    - Read → process → write.
        
    - No need to load all tables simultaneously.
        

---

## Updated Deliverables

✅ Python script `t_2I_GroupBy.py`  
✅ JSON configuration (auto-generated if missing)  
✅ Aggregated CSV outputs (short tables)  
✅ Conditionally aggregated long tables  
✅ Mapping table (`resume_id → IDS`) stored separately  
✅ Metrics JSON file  
✅ Visualizations (PNG)  
✅ Log file  
✅ Optional summary report (Markdown or text)

---

---

### 🛠 **Key Changes Required for Script Update**

✔ **Mapping Table**

- Generate a separate file (`MAPPING.csv` or `mapping.json`) storing `resume_id → [ID1; ID2; ID3]`.
    
- Remove `IDS` from grouped output tables.
    

✔ **Long Table Aggregation**

- Introduce a `variation_threshold_long_tables` parameter (default: 0.2) in the config.
    
- Aggregate long tables only if variance ≤ threshold.
    

✔ **Configuration Extension**

- Update JSON config to include the new threshold parameter.
    
- Ensure the script reads this value dynamically.
    

✔ **Documentation Update**

- Update docstring and comments to reflect separation of mapping storage and conditional aggregation logic.
    
