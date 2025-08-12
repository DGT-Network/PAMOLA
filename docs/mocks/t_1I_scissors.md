# Task Specification: `t_1p_scissors.py`

## Project Context

**Project:** PAMOLA.CORE (Resume Anonymization)  
**Task ID:** t_1P  
**Script Location:** `{project_root}\scripts\mock\t_1p_scissors.py`

This task defines the development of an AI-driven Python script called `t_1p_scissors.py`, intended to simulate (mock) specific data slicing functionality on a defined dataset as part of the PAMOLA.CORE project.

---

## Purpose

The script `t_1p_scissors.py` is designed to **vertically split a raw dataset into multiple subsets** by columns, keeping a shared identifier for rejoining later. It prepares data for further anonymization and processing by dividing it into thematic parts (identity, details, experience, etc.).

---

## Script Naming and Placement

- **Script name:** `t_1p_scissors.py`
    
- **Prefix logic:**  
    `m` (mandatory prefix) + `1` (sequence number) + `P` (action prefix for 'prepare') → `t_1P`
    
- **Directory:** `{project_root}\scripts\mock\`
    

---

## Input Data

- All data references are **relative to the `DATA` repository root**.
    
- Example input file: `DATA/raw/10k.csv` (passed as `raw/10k.csv`)
    
- Default input parameters:
    
    - Separator: `,`
        
    - Encoding: `UTF-16`
        
    - Text qualifier: `"`
        
- Input parameters and field split configuration are read from `{project_root}\configs\t_1P.json` (auto-created if missing with defaults).
    

### Example JSON Config (can be extended later):

```json
{
  "input_file": "raw/10k.csv",
  "encoding": "UTF-16",
  "separator": ",",
  "text_qualifier": "\"",
  "id_field_name": "ID",
  "field_groups": {
    "IDENT": ["first_name", "last_name", "middle_name", "birth_day", "gender"],
    "DETAILS": ["post", "education_level", "salary", "salary_currency", "area_name", "relocation", "metro_station_name", "road_time_type", "business_trip_readiness", "work_schedules", "employments", "driver_license_types", "has_vehicle"]
    // ... continued for other groups
  },
  "special_fields": {
    "file_as": ["last_name", "first_name", "middle_name"],
    "age": "birth_day"
  }
}
```

**Note:** The JSON config can be extended with additional parameters or overrides as needed.

---

## Functional Requirements

### 1️⃣ Add ID Column

- Add a new column named `ID` (default) assigning row numbers for joining later.
    

### 2️⃣ Split Into Subsets

- Split the dataset into the following predefined groups (with configuration stored in JSON):
    

|Subset Name|Included Fields|Special Operations|
|---|---|---|
|IDENT|ID, resume_id, first_name, last_name, middle_name, birth_day, gender|Add `file_as` (concat last_name, first_name, middle_name); compute `age` from `birth_day` (skip if empty or invalid date)|
|DETAILS|ID, resume_id, post, education_level, salary, salary_currency, area_name, relocation, metro_station_name, road_time_type, business_trip_readiness, work_schedules, employments, driver_license_types, has_vehicle|—|
|EXPIRIENCE|ID, resume_id, experience_start_dates, experience_end_dates, experience_organizations, experience_descriptions, experience_posts, experience_company_urls|—|
|PRIMARY_EDU|ID, resume_id, primary_education_names, primary_education_faculties, primary_education_diplomas, primary_education_end_dates|—|
|ADDITIONAL_EDU|ID, resume_id, additional_education_names, additional_education_organizations, additional_education_diplomas, additional_education_end_dates|—|
|ATTESTATION|ID, resume_id, attestation_education_names, attestation_education_results, attestation_education_organizations, attestation_education_end_dates|—|
|CONTACTS|ID, resume_id, email, home_phone, work_phone, cell_phone|—|
|SPECS|ID, resume_id, key_skill_names, specialization_names|—|

### 3️⃣ Output Management

- **Processed data directory:** `DATA/processed`
    
- **Task-specific directory:** `DATA/processed/t_1P`
    
- **Output files:** `DATA/processed/t_1P/output/{SUBSET_NAME}.csv`
    
- Ensure correct handling of text qualifiers, newlines, and optionally configurable output encoding and separator.
    

### 4️⃣ Metrics and Visualizations

- **Metrics file:** `DATA/processed/t_1P/task_metrics.json` (includes task description, date, used config, execution time, error counts, row counts, I/O parameters)
    
- **Visualization:** `DATA/processed/t_1P/data_distribution.png` (pie chart of processed subset sizes)
    
- **Log file:** `{project_root}\logs\t_1P.log`
    

### 5️⃣ Directory and File Handling

- Create missing directories automatically.
    
- Overwrite existing files.
    

### 6️⃣ Progress Reporting

- Provide clear progress bar and console messages showing the current stage.
    

### 7️⃣ Documentation

- Include a top-level docstring with:
    
    - Project name: `PAMOLA.CORE - RESUME ANONYMIZATION`
        
    - Script ID: `t_1p_scissors`
        
    - Purpose description
        
    - Key features overview
        
    - Comments in English
        

### 8️⃣ Code Modularity

- If the script exceeds **700 lines**, extract helper functions into `{project_root}\scripts\mock\utils\t_1p_scissors_helpers.py`.
    

### 9️⃣ Completion Message

- Print a clear console message on successful completion listing generated artifacts and their locations.
    

### 🔹 Recommended Libraries

- `pandas` for data processing
    
- `matplotlib` or `seaborn` for visualization
    
- `tqdm` for progress bars
    
- `json` for configuration handling
    
- `os` and `pathlib` for file management
    
- `logging` for log handling
    

### 🔹 Validation Considerations

- Ensure date fields are correctly validated when computing `age`.
    
- Include robust error handling for I/O and data transformation.
    

---

## Deliverables

✅ Python script `t_1p_scissors.py` in the specified directory  
✅ JSON configuration file (if absent, auto-generated)  
✅ Processed CSV outputs for each subset  
✅ Metrics JSON file  
✅ Pie chart visualization PNG  
✅ Log file recording process details

---

## Notes

- Use robust error handling for file I/O, data parsing, and splitting operations.
    
- Focus on modular, well-documented code with clear input/output pathways.
    
- Maintain consistency with the broader PAMOLA CORE project conventions.
    

Would you like me to draft the helper module or example config file next?