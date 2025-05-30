
## 🧩 **Модуль: `DataAttributeProfilerOperation`**

### 📌 **Цель**

Реализовать операцию для автоматического профилирования атрибутов входного набора данных с целью категоризации каждого столбца по его роли в задачах анонимизации и синтеза. Поддерживаются как `pandas.DataFrame`, так и ссылка на внешний CSV-файл (с использованием `io.py`).
 Этот модуль - операция, которая получает данные от задачи, устанавливает типы, генерит артефакты в домашней директории задачи,  и возвращает список задаче в соответствии с требованиями к стандартной операции.

   Модуль выполняет следующую работу: получает списк атрибутов, их типов и катгорий (в части конфиденциальности), анализирует количество строк, непустых строк, процент уникальных строк (то есть значение не повторяется более заданного количества раз относительно передаваемого идентификатора - важно, поскольку могут быть дублируемые строки и нерегулярные операции, например, повторение имен относительно resume_id -ннесколько записей для одного резюме)
   Хотя модуль пишется с прицелом на набор данных PAMOLA.CORE, он также должен быть написан как общая функция - для других случаев анализа

---

## ⚙️ **Функциональные возможности**

### 1. 📋 **Анализ структуры**

- Извлечение списка столбцов
    
- Определение типа данных: `numeric`, `categorical`, `text`, `datetime`, `boolean`, `long_text`
    
- Отображение N (по умолчанию 10) **примеров значений** по каждому столбцу
    

### 2. 🧠 **Классификация атрибутов**

Каждому полю присваивается одна из следующих категорий:

|Категория|Описание|
|---|---|
|`DIRECT_IDENTIFIER`|Явные ID (имя, паспорт, email, UID и др.) — уникальны и однозначно идентифицируют|
|`QUASI_IDENTIFIER`|Не уникальны по отдельности, но позволяют идентификацию в комбинации (birth_day, region и др.)|
|`SENSITIVE_ATTRIBUTE`|Конфиденциальные или чувствительные поля (здоровье, финансы, поведение)|
|`INDIRECT_IDENTIFIER`|Длинные тексты, поведенческие профили, адреса — потенциально идентифицируют через анализ содержимого|
|`NON_SENSITIVE`|Остальные поля, не содержащие чувствительной информации|

> 🔄 Категоризация проводится на основе **семантики (ключевых слов)** и **статистики (энтропии, уникальности)**

---

## 🧠 **Алгоритм категоризации**

### 🔍 1. **Семантический анализ**

- Словарь ключевых слов по индустриям (HR, Health, Finance и др.)
    
- Проверка соответствия названия столбца и его значений (если строковые)
    

```python
# Пример логики
if 'email' in name or 'ssn' in name:
    → DIRECT_IDENTIFIER
elif 'birth' in name or 'region' in name:
    → QUASI_IDENTIFIER
elif 'salary' in name or 'diagnosis' in name:
    → SENSITIVE_ATTRIBUTE
elif is_text_column(col) and avg_len > threshold:
    → INDIRECT_IDENTIFIER
else:
    → NON_SENSITIVE
```


Вот расширенный **профессиональный список ключевых слов** для автоматической классификации столбцов данных (в названиях колонок) в контексте **анонимизации / синтетической генерации / профилирования рисков**. Список охватывает разные **индустрии** (HR, финансы, медицина, маркетинг, безопасность и др.), содержит **ожидаемые типы данных** и назначение — категорию столбца.

## 📋 **КЛЮЧЕВЫЕ СЛОВА ДЛЯ КЛАССИФИКАЦИИ СТОЛБЦОВ**

|🔑 Ключевые слова в названии|Ожидаемый тип данных|Категория профилирования|
|---|---|---|
|`id`, `uuid`, `uid`, `guid`, `identifier`, `record_id`|string (уникальные)|`DIRECT_IDENTIFIER`|
|`first_name`, `last_name`, `middle_name`, `surname`, `name_latin`, `full_name`|string|`DIRECT_IDENTIFIER`|
|`email`, `e-mail`, `mail_address`|string (в формате email)|`DIRECT_IDENTIFIER`|
|`phone`, `telephone`, `cell`, `mobile`, `contact_number`|string (numeric string)|`DIRECT_IDENTIFIER`|
|`passport`, `ssn`, `sin`, `nino`, `driver_license`, `id_number`, `national_id`|string|`DIRECT_IDENTIFIER`|
|`address`, `street`, `building`, `apt`, `flat`, `postal_code`, `zip`, `location_detail`|string|`QUASI_IDENTIFIER`|
|`region`, `province`, `state`, `district`, `oblast`, `kra`, `municipality`|string|`QUASI_IDENTIFIER`|
|`city`, `town`, `village`, `settlement`, `locality`, `metro_area`|string|`QUASI_IDENTIFIER`|
|`country`, `nation`|string|`QUASI_IDENTIFIER`|
|`birth`, `dob`, `birth_date`, `birth_day`|date|`QUASI_IDENTIFIER`|
|`gender`, `sex`|string|`QUASI_IDENTIFIER`|
|`ethnicity`, `race`, `nationality`|string|`SENSITIVE_ATTRIBUTE`|
|`religion`, `faith`, `belief`|string|`SENSITIVE_ATTRIBUTE`|
|`salary`, `income`, `earnings`, `revenue`, `pay`, `compensation`, `bonus`, `cost`|numeric|`SENSITIVE_ATTRIBUTE`|
|`job`, `position`, `occupation`, `profession`, `role`, `rank`|string|`QUASI_IDENTIFIER`|
|`company`, `employer`, `organization`, `department`|string|`QUASI_IDENTIFIER`|
|`experience`, `work_history`, `job_years`|string / numeric|`NON_SENSITIVE` или `QUASI_IDENTIFIER`|
|`education`, `degree`, `university`, `college`, `school`, `faculty`|string|`QUASI_IDENTIFIER`|
|`grade`, `gpa`, `score`, `assessment`|numeric|`NON_SENSITIVE` или `SENSITIVE_ATTRIBUTE` (в зависимости от контекста)|
|`diagnosis`, `disease`, `health_condition`, `illness`, `disorder`, `symptom`, `treatment`, `medication`, `therapy`, `mental`|string|`SENSITIVE_ATTRIBUTE`|
|`bank`, `iban`, `bic`, `account`, `credit_card`, `debit`, `payment_method`, `transaction`, `balance`, `loan`, `mortgage`|string / numeric|`SENSITIVE_ATTRIBUTE`|
|`geo`, `longitude`, `latitude`, `location`, `coordinates`, `gps`|float / string|`INDIRECT_IDENTIFIER`|
|`ip`, `mac`, `device_id`, `browser_fingerprint`, `hardware`, `imei`, `udid`|string|`INDIRECT_IDENTIFIER`|
|`photo`, `image`, `avatar`, `face_encoding`, `biometric`, `voice`, `fingerprint`|binary / text|`INDIRECT_IDENTIFIER`|
|`resume_text`, `description`, `comments`, `notes`, `feedback`, `free_text`, `bio`, `profile_summary`|string (длинные)|`INDIRECT_IDENTIFIER`|
|`timestamp`, `date`, `datetime`, `created_at`, `updated_at`, `log_time`, `event_time`|datetime|`NON_SENSITIVE`|
|`status`, `stage`, `state`, `approved`, `rejected`, `active`, `inactive`|categorical|`NON_SENSITIVE`|
|`campaign`, `source`, `utm`, `channel`, `referrer`, `click_id`, `session_id`|string|`QUASI_IDENTIFIER`|
|`device`, `browser`, `os`, `platform`, `screen`, `resolution`|string|`INDIRECT_IDENTIFIER`|
|`vote`, `opinion`, `choice`, `survey_answer`, `rating`, `reaction`|string / numeric|`SENSITIVE_ATTRIBUTE` (если персонализировано)|
|`event`, `log`, `action`, `activity`, `operation`|string|`NON_SENSITIVE`|
|`tag`, `label`, `category`, `cluster`|string|`NON_SENSITIVE`|


## 💼 **Индустриальные добавки (по контекстам)**

### 👨‍⚕️ Здравоохранение

- `icd_code`, `diagnosis`, `prescription`, `clinical_note` → `SENSITIVE_ATTRIBUTE`
    
- `hospital`, `doctor_name` → `QUASI_IDENTIFIER`
    

### 💳 Финансы

- `account_id`, `transaction_id`, `iban` → `DIRECT_IDENTIFIER`
    
- `purchase_amount`, `payment_method`, `spending_category` → `SENSITIVE_ATTRIBUTE`
    

### 🧑‍💼 HR / Рекрутинг

- `resume_id`, `candidate_id`, `cv_id` → `DIRECT_IDENTIFIER`
    
- `experience_level`, `skills`, `certifications`, `job_preferences` → `QUASI_IDENTIFIER`
    

### 📈 Маркетинг / Поведенческие данные

- `click_path`, `session_data`, `device_info`, `time_spent` → `INDIRECT_IDENTIFIER`
    
- `survey`, `opinion`, `motivation`, `interests` → `SENSITIVE_ATTRIBUTE`
    


## 🔧 **Настройки и расширения**

- **Поддержка нескольких языков**: добавление синонимов на русском и других языках
    
- **Поддержка регулярных выражений**: для идентификации email, IP, телефона
    
- **Конфигурируемые словари**: возможность загрузки/дополнения словарей пользователем
    

---

    
2. Предлагается часть разместить внутри модуля, а часть дополнить и вынести в json раполагаемый в DATA (репозиторий данных) в external_dictionaries - путь по умолчнию (пользователь может его менять_ Если словарь доступен -  использование его  в `attribute_utils.py` , если нет - внутренний список и энттропия
    

JSON сгенерить как часть работы
### 📈 2. **Статистический анализ**

- Расчет **энтропии**: `H(X) = -∑ p(x) log₂ p(x)`
    
- Оценка **процент уникальных значений**
    
- Учет средней длины строк (для текстовых столбцов)
    

> ⚙️ Настройки порогов (энтропия, уникальность, длина текста) — передаются через конфигурацию

---

## 🗂 **Архитектура и интеграция**

### ✅ **Имя класса**:

```python
class DataAttributeProfilerOperation(FieldOperation):
```

### 📁 **Расположение:**

`pamola_core.profiling.analyzers.attribute.py`

#### ✔ Использование инфраструктуры:

- Чтение данных: `data_source.get_dataframe("main")`
    
- Репорты: `reporter.log(...)`
    
- Артефакты: `result.add_artifact(...)`
    
- Прогресс: `progress_tracker.update(...)` (опционально)

### 📦 **Общие утилиты:**

- `pamola_core.profiling.commons.attribute_utils.py` — функции анализа и классификации
    
- `pamola_core.utils.io.py` — чтение CSV, сохранение JSON
    
- `pamola_core.utils.visualization.py` — визуализация распределений (pie, bar, wordcloud по частоте ключей)
    - pamola_core.utils.progress.py - работа с прогресс-баром 
    - pamola_core.utils.logging.py - логирование
    

### Принять во внимание
**Файлы, которые принять во внимание:**

1. `pamola_core/utils/ops/op_base.py` – определение `FieldOperation`, `OperationResult`
    
2. `pamola_core/utils/ops/op_data_source.py` – `DataSource` и `get_dataframe(...)`
    
3. `pamola_core/utils/ops/op_result.py` – `OperationResult`, `OperationArtifact`
    
4. `pamola_core/utils/ops/op_registry.py` – `@register`

`7 IDENT.md` – описание таблицы, логики UID, описание задачи
    
8. `index.md` – документация по Task/Operation API
    
9. `visualization.md`, `io.md` – документация API артефактов
---

## 📤 **Результаты и артефакты**

|Артефакт|Формат|Назначение|
|---|---|---|
|`attribute_roles.json`|JSON|Список всех столбцов и их классификация|
|`attribute_entropy.csv`|CSV|Энтропия и уникальность по каждому полю|
|`attribute_sample.json`|JSON|Примеры значений по каждому столбцу|
|`attribute_type_pie.png`|PNG|Круговая диаграмма распределения типов|
|`entropy_vs_uniqueness.png`|PNG|Сравнительная диаграмма полей по рискам|
Если задан идентификатор (или идентификаторы) - указать количество строк с учетом их уникальности
---

## 🧪 **Метрики (OperationResult)**

|Метрика|Описание|
|---|---|
|`num_columns`|Общее количество столбцов|
|`num_direct_identifiers`|Число прямых ID|
|`num_quasi_identifiers`|Число квазиидентификаторов|
|`num_sensitive_attributes`|Число чувствительных|
|`num_indirect_identifiers`|Число косвенных (например, длинные тексты)|
|`num_non_sensitive`|Не чувствительные поля|
|`avg_entropy`|Средняя энтропия|
|`avg_uniqueness`|Средний % уникальности|

---

## 🧩 **Порядок разработки / изменений**

### 🔨 Модули:

| Файл                 | Назначение                                                           |
| -------------------- | -------------------------------------------------------------------- |
| `attribute.py`       | Основная операция                                                    |
| `attribute_utils.py` | Семантические словари, энтропия, текстовые эвристики                 |
| `visualization.py`   | `create_pie_chart`, `entropy_vs_uniqueness_plot`, возможно wordcloud |
| `io.py`              | `save_json`, `save_dataframe` — переиспользуем                       |
| `op_registry.py`     | регистрация операции `@register(version="1.0.0")`                    |

---

## ❓Вопросы для уточнения

1. Нужно ли сохранять разметку типов полей в формате, пригодном для последующей передачи в модуль `PreKAnonymityProfilingOperation` (anonymity.py)?
Все операции работаю независимо, но могут быть обработаны задачей 
    
2. Есть ли потребность в **поддержке русскоязычных названий столбцов** (ключевые слова и эвристики на русском)?

На данном этапе нет, но расширение (json) может их добавлять, в этом случае надо читать с соответсвующей кодировкой
    
3. Следует ли поддерживать обработку `nested` структур (например, JSON внутри ячеек)?

Да, например, списки в формате ['Удаленная работа', 'Полный день'] - это MVF поля, особый тип категорийных полей
3. Надо ли выводить **оценку риска утечки** (на основе комбинации QUASI + ENTROPY)?

На данном этапе нет, для этого есть отдельный операции

3. Интересует ли вас конфигурация чувствительности классификатора (например, приоритет семантики или статистики)?
Не уверен, что стоит усложнять на данном этапе    




## 📦 **Вопросы об архитектуре и интеграции**

### 1. **Наследование от `BaseOperation`?**

✅ Да, **правильно**. `DataAttributeProfilerOperation` должен наследоваться от `BaseOperation`, поскольку:

- он обрабатывает **всю таблицу целиком**, а не одно поле;
    
- его задача — дать глобальную картину и категоризацию, а не манипулировать значениями одного столбца.
    

> `FieldOperation` применима к индивидуальному полю и ограничена scope, тогда как нам нужен модуль со сквозным обзором по столбцам.

---

### 2. **Использование в `BaseTask` и как самостоятельный модуль?**

✅ Да, модуль должен:

- Быть **самодостаточным** (можно вызвать отдельно, например, в dev/interactive анализе),
    
- И при этом **встраиваться в комплексную задачу профилирования** в рамках `BaseTask`.
    

> Использование через JSON-конфиг в задачах — основной сценарий. Возможен и REPL/CLI режим. ОДНАКО, ЭТО НЕ УРОВЕНЬ ОПЕРАЦИИ - ЭТО УРОВЕНЬ ЗАДАЧИ. ОПЕРАЦИЯ ПОЛУЧАЕТ ДАННЫЕ НА ВХОД ОТ ЗАДАЧИ (ИСТОЧНИК ДАННЫХ, ПУТЬ К ДОПОЛНИТЕЛЬНОМУ СЛОВАРЮ  - ЕСЛИ ЕСТЬ, КОЛЬЧЕСТВО ПРИМЕРОВ)

---

### 3. **Файл размещения: `pamola_core/profiling/analyzers/attribute.py`?**

✅ Да. Стандартное размещение:

```
pamola_core/
  profiling/
    analyzers/
      attribute.py  ← модуль DataAttributeProfilerOperation
```

> Общие утилиты: `pamola_core/profiling/commons/attribute_utils.py`

---

### 4. **Поддержка `progress_tracker`?**

🟡 ДА С ИСПОЛЬЗОВАНИЕМ  pamola_core.utils.progress.py:

- Да, поддержка должна быть предусмотрена (особенно при большом количестве полей/строк).
    
- Может использоваться шаговая прогрессия по полям, либо по этапам (`structure -> entropy -> category assignment`).
    

---

## 🧠 **Вопросы по функциональности и логике**

### 5. **Дополнительные метрики на поле**

📈 Можно добавить:

- `value_counts_topN` (частотный словарь) - сохраняем в {task_dir}\dictionaries
    
- `missing_rate` (доля пропущенных значений)
    
- `avg_text_length`, `max_text_length` — для текстов
    
- `data_type_inferred` — логический тип (помимо pandas типа)
    
- `entropy_normalized` — для сравнения разных полей
    

---

### 6. **Алгоритм разрешения конфликтов (email + высокая энтропия)?**

📌 Приоритетная схема:

1. **Семантика имеет наивысший приоритет** (если явный `email` — это всегда `DIRECT_IDENTIFIER`)
    
2. Если несколько семантических совпадений — берётся **наиболее чувствительный**
    
3. При отсутствии семантики — применяется **статистика**
    
4. Все конфликты — логгируются и сохраняются в `conflicts.json`
    

---

### 7. **Формулы для энтропии и уникальности**

✅ **Энтропия**:

```python
H(X) = -∑ p(x) * log2(p(x))
```

- `p(x)` — вероятность (доля) каждого уникального значения
    

✅ **Нормализованная энтропия**:

```python
H_norm(X) = H(X) / log2(n)
```

- где `n` — количество уникальных значений
    

✅ **Уникальность**:

```python
uniqueness_ratio = df[col].nunique() / len(df)
```

---

### 8. **История категоризации?**

🟡 Необязательно в MVP, но архитектурно:

- Желательно сохранять `attribute_roles_previous.json`, если запускается в рамках повторного анализа.
    
- Это позволит оценить drift/изменения и стабильность классификации.
    

---

### 9. **Уровень визуализации**

🎨 Для MVP:

- Достаточно базовых диаграмм: `pie`, `bar`, `entropy_vs_uniqueness` scatter
    
    

---

## 🔗 **Вопросы по интеграции с другими компонентами**

### 10. **Интеграция с PreKAnonymityProfilingOperation**

✅ Необходима передача:

- Списка `QUASI_IDENTIFIER` полей → напрямую как output JSON, но это опционально
    
- Возможно в виде `quasi_fields.json` или общего `attribute_roles.json`
    

---

### 11. **Формат хранения результатов**

✅ Стандартизированный JSON:

```json
{
  "column_name": {
    "role": "QUASI_IDENTIFIER",
    "type": "string",
    "entropy": 4.12,
    "uniqueness": 0.21,
    "missing_rate": 0.03
  },
  ...
}
```

---

### 12. **Словари ключевых слов**

✅ Да, лучше создать отдельный модуль:

```
pamola_core/profiling/dictionaries/keywords_attribute_roles.py
```

С возможностью редактирования / расширения пользователем:

- `general_keywords`
    
- `medical_keywords`
    
- `financial_keywords`
    
- и т.д.
    
+ и пользовательский json (путь к нему)
---

## 🔁 **Вопросы по обработке MVF (multi-valued fields)**

### 13. **Логика для MVF**

- Распознать поля, содержащие списки или строки с `,`, `;`, `|` → потенциально `MVF` , значения в квадратных скобках с разделителем
    
- Преобразовать в список, анализировать частоты и уникальность комбинаций
    
- Категоризировать по наличию ключевых слов в отдельных значениях списка
    

---

### 14. **Как отличать MVF от текстов?**

- `is_text = avg_token_count > threshold`
    
- `is_mvf = delimiter_count > N` и `values are short`
    

Пример: `"Удаленная работа; Полный день"` → MVF  
`"Я ищу работу в компании..."` → TEXT

---

## 📊 **Визуализация и отчетность**

### 15. **Использование `visualization.py`**

✅ В базовой версии — да:

- `create_pie_chart()`
    
- `create_bar_plot()`
    
- `create_scatter_plot()` (для entropy vs uniqueness)
    


    

---

### 16. **Финальный отчет: формат?**

✅ Должны быть:

- JSON и PNG артефакты
- csv для словарей
    

---

## 🧬 **Категоризация — приоритеты и алгоритмы**

### 17. **Приоритет: семантика или статистика?**

✅ **Семантика имеет приоритет**, особенно при прямом совпадении (email, ssn)

---

### 18. **Пороговые значения (по умолчанию):**

|Метрика|Значение|
|---|---|
|`entropy_high`|> 5.0|
|`entropy_mid`|3.0–5.0|
|`uniqueness_hi`|> 0.9|
|`uniqueness_lo`|< 0.2|
|`text_len_long`|> 100|

---

### 19. **Нужны ли сложные алгоритмы?**

🟡 В базовой версии — нет. Достаточно:

- Ключевых слов
    
- Статистики
    
- Простых эвристик
    

> Позже: возможна ML-модель на основе мета-фичей полей

---

### 20. **Работа с многоязычными данными**

✅ Да, желательно иметь **отдельные словари по языкам**:

- `keywords_en`, `keywords_ru`
    
- Автоопределение языка названий — опционально
    

---
