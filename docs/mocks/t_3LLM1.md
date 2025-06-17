# Software Requirements Specification (SRS)

## Project: PAMOLA.CORE Mock Task – LLM Anonymization with LM Studio

### Task ID: `t_3LLM1`

---

## 1. Introduction

This document describes the requirements for a mock task in the PAMOLA.CORE framework designed to demonstrate anonymization of resume fields using a locally running LLM (via LM Studio WebSocket API). The task does not currently rely on the pamola core operations and instead operates independently to facilitate end-to-end testing.

---

## 2. Scope and Objectives

The task reads a specified CSV dataset containing resume data, processes certain fields through LM Studio to anonymize personal identifiers, and writes the anonymized version to a separate directory. It also provides caching, logging, and metric tracking to evaluate task performance.

---

## 3. File Structure

* **Script location:** `scripts/mock/t_3llm1_experience.py`
* **Config file:** `{project_root}/configs/t_3LLM1.json`
* **Logs:** `{project_root}/log/t_3LLM1.log`
* **Reports:** `{data_repository}/reports/t_3LLM1.json`
* **Output directory:** `{task_dir}/output/{dataset_name}`

---

## 4. Configuration Parameters

If the config file does not exist, it is created using the following defaults:

```json
{
  "project_root": "D:/VK/_DEVEL/PAMOLA.CORE",
  "data_repository": "D:/VK/_DEVEL/PAMOLA.CORE/DATA",
  "dataset_path": "processed/t_2I/output/EXPIRIENCE.csv",
  "encoding": "utf-16",
  "separator": ",",
  "text_qualifier": "\"",
  "fields": ["experience_organizations", "experience_descriptions", "experience_posts"],
  "prompts": {
    "experience_organizations": "Перепиши название организации '{ORGANIZATION}', сохранив суть, но убрав точное название и правовую форму. Пример: 'Горэлектротранс' → 'транспортная компания'.",
    "experience_descriptions": "Опиши кратко и обобщённо, чем человек занимался в '{ORGANIZATION}'. Пример: 'создание UI-интерфейса' → 'дизайн интерфейса'.",
    "experience_posts": "Укажи аналог должности в более общем виде. Пример: 'Менеджер по продажам услуг' → 'продажи и маркетинг'."
  },
  "legal_forms": ["ООО", "ЗАО", "АО", "НПО"],
  "fallback_values": {
    "experience_organizations": ["производственная организация", "комбинат 23", "IT компания"],
    "experience_descriptions": ["работал по заданию", "согласно должностным обязанностям", "поддержка"],
    "experience_posts": ["сотрудник", "ответственный", "менеджер"]
  },
  "task_dir": "processed/3LLM",
  "cache_responses": true,
  "use_minhash_cache": false,
  "llm": {
    "server_ip": "127.0.0.1",
    "server_port": 1234,
    "timeout": 1,
    "model_name": "llama-4-scout-17b-16e-instruct-i1"
  },
  "max_rows": 100,
  "max_error_count": 10,
  "dry_run": false
}
```

---

## 5. Functional Requirements

### 5.1 Initialization

* On start, check for existence of configuration.
* Load configuration from JSON, or create default config if not found.

### 5.2 Data Loading

* Use `pamola_core/utils/io.py` to load CSV from `{data_repository}/{dataset_path}` with specified encoding, separator, and qualifier.

### 5.3 Record Processing

For each record (up to `max_rows`):

* Normalize text for each field:

  * Lowercase
  * Remove duplicate characters
  * Remove non-ASCII characters
  * Strip legal forms (ООО, АО, etc.) from organization names
  * Truncate to 100 characters
* Use MD5 hash caching to avoid duplicate LLM calls
* (Optional) MinHash similarity check using `pamola_core/utils/nlp/minhash.py`
* Construct prompt using placeholder substitution
* Query LLM via LM Studio WebSocket API
* Enforce timeout and skip record if it expires
* Normalize returned text as above and limit to 10 words or 100 characters

### 5.4 Output

* Save resulting DataFrame using `pamola_core/utils/io.py` to `{task_dir}/output/{dataset_name}`
* Log via `pamola_core/utils/logging.py` to `{project_root}/log/t_3LLM1.log`
* Display progress with `pamola_core/utils/progress.py` and `tqdm`
* Save metrics as JSON (`metrics_{dataset_name}.json`) in `{task_dir}`
* Save visual performance chart using `pamola_core/utils/visualization.py` to `{task_dir}/speed_plot_{dataset_name}.png`
* Save anonymization report to `{data_repository}/reports/t_3LLM1.json`

### 5.5 CLI Behavior

* Shows progress bar using `tqdm`
* Tolerates up to `max_error_count` errors before aborting
* Logs all issues to file and console

---

## 6. Non-Functional Requirements

* Must support UTF-16 encoding
* Ensure <1s response time per record where possible
* Avoid duplicate LLM queries using cache or similarity match
* Fail gracefully and log errors for later analysis

---

## 7. Metrics

Metrics will include:

* Total processed records
* Mean response time
* Number of empty fields replaced
* Error count
* Average input prompt length
* Response lengths (min, max, mean)
* Cached vs new queries ratio
* Visualization of processing speed over time

All stored in `{task_dir}` with prefixed filenames.

---

## 8. Integration and Future Compatibility

* Compatible with PAMOLA.CORE I/O, logging, and progress tracking modules
* Future integration planned with profile/anonymization operations in pamola core

---

## 9. Notes

* Prompts use placeholders (e.g., `{ORGANIZATION}`) and are optimized for brevity
* Task ID and filenames consistently reflect `t_3LLM1`
* All code comments must be in English

---

## End of Document
