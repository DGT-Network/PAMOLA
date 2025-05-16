
# TextSemanticCategorizerOperation

**Specification Document**  
**Project**: PAMOLA / PAMOLA.CORE  
**Last Updated**: 2025-04-06

---

## 1. Overview

The `TextSemanticCategorizerOperation` is a profiling module for analyzing short text fields (typically under 255 characters) that do not fall under categorical types but contain meaningful phrases relevant for privacy analysis, such as job positions, institutions, payment descriptions, etc.

This module is intended for use in anonymization and synthesis pipelines, preparing replacement mappings and semantic categories based on three strategies:

- Dictionary-based recognition with alias mapping
    
- Named Entity Recognition (NER) using spaCy
    
- Clustering based on unresolved tokens or fragments
    

The module does not apply replacements itself but prepares structured outputs (JSON, CSV) for downstream anonymization operations.

---

## 2. Architecture & Integration

- **Inherits from**: `FieldOperation`
    
- **Module location**: `pamola_core/profiling/analyzers/text.py`
    
- **Helpers**: `pamola_core/profiling/commons/text_utils.py`
    
- **Uses utilities** from `pamola_core.utils`:
    
    - `io.py`: loading and saving files (CSV, JSON)
        
    - `visualization.py`: pie/bar plots and token distributions
        
    - `logging.py`: operation-level logging
        
    - `progress.py`: progress tracking if enabled
        
- **Comments and all code** should be written in **English**.
    

**Required for porting to another environment or chat:**

- `text.py` (operation definition)
    
- `text_utils.py` (tokenization, cleaning, NER helpers)
    
- `io.py`, `visualization.py`, `logging.py`, `progress.py`
    
- Sample dictionary file (`text_categories.json`)
    
- Test dataset with representative short text fields
    

This operation is invoked as part of a user-defined script (task), managed through a unified task controller that orchestrates various `op*` modules.

---

## 3. Input Conditions

A field qualifies for semantic categorization if:

- `inferred_type == "text"`
    
- `avg_text_length < 255`
    
- `uniqueness_ratio > 0.3`
    
- `not categorical`
    

Additional diagnostics:

- Missing rate
    
- Text length distribution (binned by: <50, 50–100, 100–150, 150–200, 200–250, >250)
    

---

## 4. Categorization Methods

### 4.1 Dictionary-Based Recognition

- Loads a dictionary from:
    
    - Explicit `dictionary_path`, or
        
    - Default path under `{task_dir}/dictionaries/text_categories.json`
        
- If not found, a built-in mini dictionary is used.
    
- Dictionary structure supports:
    
    - `category` name
        
    - `domain` (e.g., HR, Finance, Legal)
        
    - `seniority` level (e.g., Junior, Lead) — can be applied independently or jointly with category
        
    - `keywords`: list of exact or partial match terms
        
    - `alias`: replacement term used for all matches
        
    - `language` support for multilingual values
        

Priority matching:

- Match longest keyword first
    
- If multiple matches exist, use the one with most specific category (category > domain > fallback)
    
- Seniority may override alias when specified as a separate rule
    

**JSON Schema for Dictionary:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "patternProperties": {
    ".+": {
      "type": "object",
      "properties": {
        "alias": {"type": "string"},
        "domain": {"type": "string"},
        "seniority": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["alias", "keywords"]
    }
  }
}
```

### 4.2 NER-Based Matching

- Uses `spaCy` with support for multiple models:
    
    - `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg`
        
    - `ru_core_news_sm`, `ru_core_news_md`, `ru_core_news_lg`, `ru_core_news_trf`
        
    - Additional models for other supported languages (e.g., `xx_ent_wiki_sm` for multilingual)
        
- Recognizes entities like:
    
    - JOB, ORG, SKILL, TOOL, GPE, PERSON
        
- Maps recognized entities to dictionary categories when possible
    

### 4.3 Token Clustering (Fallback)

- For unmatched values:
    
    - Tokenize and clean
        
    - Group by shared tokens (Jaccard similarity or token overlap)
        
    - Assign temporary aliases (e.g., `CLUSTER_A`, `CLUSTER_B`)
        
- Random replacements from a predefined fallback list (e.g., "worker", "any job") can be suggested, not applied
    

---

## 5. Outputs and Artifacts

All outputs are stored under `{task_dir}/dictionaries/`

- `text_semantic_roles.json`: categorized field entries with match method (dictionary, ner, cluster)
    
- `unresolved_terms.csv`: entries not matched; includes record ID and field value
    
- `category_mappings.csv`: editable mapping of original → alias (human-curated or system-proposed)
    

### Visualizations (in `visualization.py`)

- Pie chart of top categories (by frequency)
    
- Bar chart of alias replacements
    
- Histogram of text lengths by bin
    
- Word clouds (optional, based on top tokens per category)
    

All diagrams saved as:

- `category_distribution.png`
    
- `replacement_frequency.png`
    
- `text_length_histogram.png`
    

---

## 6. Metrics

- `num_matched` (by dictionary)
    
- `num_ner_matched`
    
- `num_auto_clustered`
    
- `num_unresolved`
    
- `missing_rate`
    
- `avg_text_length`, `max_text_length`
    
- `top_replacements` (counts and categories)
    

---

## 7. Special Cases

- Texts like "any job", "работник" → categorized as `OTHER` or proposed random neutral replacement
    
- Embedded English in Cyrillic or other language texts → considered high-priority for token analysis
    
- Stop words removed using language-specific models (`text_utils`) + spaCy tokenization
    
- Longest match wins logic for keyword resolution
    

---

## 8. Dictionary Example (JSON)

```json
{
  "Project Management": {
    "alias": "project_manager",
    "domain": "IT",
    "seniority": "Middle",
    "keywords": [
      "Project manager", "Project director", "Manager проекта", "Руководитель проекта",
      "Project lead", "PM", "Delivery manager"
    ],
    "language": ["en", "ru"]
  },
  "Software Development": {
    "alias": "developer",
    "domain": "IT",
    "seniority": "Any",
    "keywords": ["Разработчик", "Developer", ".NET dev", "Fullstack", "Программист"],
    "language": ["en", "ru"]
  },
  "Finance": {
    "alias": "accountant",
    "domain": "Finance",
    "seniority": "Senior",
    "keywords": ["Бухгалтер", "финансист", "экономист", "financial analyst"],
    "language": ["en", "ru"]
  },
  "Other": {
    "alias": "any_job",
    "domain": "General",
    "seniority": "N/A",
    "keywords": ["Любая работа", "работник", "чтo угодно"],
    "language": ["ru"]
  }
}
```

---

## 9. Dictionary Integration Logic

1. If `dictionary_path` is explicitly provided → load from file
    
2. If not → check default in `{task_dir}/dictionaries/text_categories.json`
    
3. If not found → use built-in minimal fallback
    
4. Matching logic:
    
    - Normalize case, strip punctuation
        
    - Longest matching keyword takes precedence
        
    - If match is found → assign category, alias, domain, and optionally seniority
        
    - If not found → NER is applied
        
    - If still unmatched → fallback clustering
        

Dictionary matches are prioritized over NER, and NER over clustering. All assignments include method metadata for traceability.

---

## КОЛЛИЗИИ
### Проблема

Пользователь может указать:


`"Python разработчик", "Backend Developer", "Java Developer", "Fullstack разработчик"`

Все они **подходят** под `Разработчик`, но также — под более **специализированные категории** (Python, Java, Fullstack и т.д.).

Если обработать неправильно, мы получим:

- ❌ Потерю детализации (всё уходит в generic `Developer`)
    
- ❌ Коллизии в alias/заменах (разные технологии → одно и то же)
    


### Решение: Механизм приоритета и стратегии разрешения

#### 🔷 1. **Механизм приоритета (Longest Specific Match First)**

- Сначала проверяем более частные (level 2) категории
-  Внутри сначала проверяем ключевые слова по **длине** (например, `python разработчик` > `разработчик`)
    
- Затем проверяем **количество совпадений** токенов
    
- Затем по **специфичности домена**:
    
    - `Python`, `JavaScript`, `DevOps` и т.п. — приоритет выше, чем `General`
        

📌 **Пример логики:**

lua

CopyEdit

`match = find_all_matches(text) if match:     sorted_match = sort_by_length_then_domain_specificity(match)     return sorted_match[0]`

---

#### 🔷 2. **Метаданные при совпадении**

Для каждой подстановки фиксируем:


`{   "original": "python разработчик",   "matched_category": "Python Ecosystem Specialist",   "matched_alias": "python_dev",   "match_method": "dictionary",   "conflict_candidates": ["Developer"] }`

Это позволяет:

- 💡 Визуализировать, какие поля имели **альтернативные интерпретации**
    
- Дать возможность **пользователю переопределить** поведение
    

---

#### 🔷 3. **Стратегии разрешения:**

|Стратегия|Описание|
|---|---|
|`specific_first`|(По умолчанию) Узкоспециализированные категории важнее общих|
|`domain_prefer`|Приоритет по домену, если `Python` или `DevOps`, то выше `IT General`|
|`alias_only`|Используется заранее определенный alias (блокирует перезапись)|
|`user_override`|Позволяет использовать внешнюю карту переопределения|

---

### 🔄 Интеграция с текущей архитектурой

### 📌 В `TextSemanticCategorizerOperation`:

- Функция `get_best_match(text: str, dictionary: dict) -> MatchResult`
    
    - возвращает объект с основным совпадением и списком альтернатив
        
- Артефакт `text_semantic_roles.json` должен сохранять `conflict_candidates`

###  Редкие замены
если несмотря на категоризацию, замены редкие,  то используя ирерахию - делаем замену на более высокий уровень.  Но это задача анонимизации, за предалами данного модуля

## ВОПРОСЫ
### **1. Архитектура и размещение**

1. ✅ **Наследование:** правильно ли я понимаю, что новая операция будет наследоваться от `FieldOperation`, а не от `BaseOperation`, т.к. применяется к одному полю?
- Да, корректно
    
2. 📁 **Файл размещения:** предположительно `pamola_core/profiling/analyzers/text_semantic.py` — подтвердить или предложить структуру.
Давай использовать text.py ( частично уже имеется, но пока сломан) text.py в analyzers, вспомогательные данные в  `pamola_core/profiling/commons/text_utils.py

3. 🔁 **Интеграция с Task:** должна ли операция вызываться через `BaseTask`, как `TextOperation`, или может работать как вспомогательная (`helper`), вызываемая внутри другой операции?

Задача в терминах проекта - пользовательский скрипт, который объединяет несколько относительно независимых операций, передавая им все настройки, пути и получая от них артефакты, записанные в домашнюю диреткорию задачи.  Все операции базируются на op* модулях в пакете pamola_core/utils/op*

4. 🔗 **Использование `text_utils.py`:** можно ли переиспользовать такие функции как `tokenize_and_clean`, `detect_languages`, `extract_key_entities` и др.?
    Да, все функции из этого модуля можно использовать, в то же время Spacy имеет собственный токенизатор

---

### 🧠 **2. Функциональность и логика**

1. 🎯 **Цель:** правильно ли я понимаю, что задача — категоризовать **короткие текстовые поля**, которые не являются `categorical`, но содержат **ключевые слова**, отражающие _позиции, специализации, домены_?

В общем провести анализ текстовых полей и сформировтаь данные для использования их в анонимизации - csv таблицы и json. Кроме категоризации надо  оценить и визуализировтаь изменение длины (если вдруг больше 255 символов - то  > 250, а эо этого -  снекоорым шагом), количество пропушенных значений.

1. 📐 **Порог длины:** поле считается подходящим для анализа, если:
    
    - Тип поля — `text`
        
    - Средняя длина `avg_len < 255`
        
    - Кол-во уникальных значений выше определённого порога (например, >30%)
        
    - Не попало в `categorical` при профайлинге Подтвердить или предложить иные критерии?

		Да, основа подготовить данные дял замены. На этом этапе пользователь может вернуться и исправить словари что требует контентной работы, обучить специализированные модели или согласиться для будущих операций анонимизации с заменой. Только учти, что хотя мы все проигрываем на резюме - имеется более широкий спектр подобных полей
		
1. 🧩 **Алгоритмы категоризации:**
    
    - 1️⃣ **По словарю:** `POSITIONS.csv`, `posDomains.csv`, `posSeniority.csv`, `posCategories.csv` — использовать по аналогии с `attribute_utils.py`?
    Да, давай сформируем лучше json пользователя - лежит в той же директории, если словаря нет, используется встроенная минимальная версия.  Словарь - путь к нему, ищется автоматически, но пользователь может заменить словарь через указания пути.
    
    - 2️⃣ **По NER (например, spaCy):** выделение сущностей и сопоставление с известными категориями
    Да    
    - 3️⃣ **По ключевым словам:** кластеризация (например, по совпадающим словам), если не найдено явно — формировать временные группы?
        Да, и временные группы заменять одинаковмыи подменами (все подмены фиксировать для возможности анализа) как  с точки зрения карты, так и с точки зрения статистики
2. 🧪 **Уточнение:**
    
    - Какой механизм связывает синонимы: `Project manager` = `менеджер проекта` = `руководитель проекта` — используется поле `alias`?
        Через пользовательский json (лучше один, без дополнительных переходных таблиц как сейчас) - ведет к категории, которая также имеет поле FilaAs -  (можно другое название) - которое подставляется для всех замен данной катгории
    - Подтвердить формат хранения словарей и структуру категории (`name`, `domain`, `seniority`, `alias`)
        Для каждой категории /домена (имеется список ключевых слов) - прямо привязанных, подставляется alias, верно. Ключевые слова надо добавить как список, коорый может быть объемным. Сформировтаь для профессий на базе имеющегося списка как пример. Однако, существует еще проблема коллизий, когда одна категория или домен имеет пересечение с другой по ключевым словам. Например, менеджер и менеджер проекта, поэтому должен учитываться размер тега - в первую очередь применяются большие по размеру теги. Для этого должен обращаться словарь, находиться категория (если найдена - пропуск для следующей, затем для категрии подстановка и все в сочетании с другими способами кластеризации)

---

### 📤 **3. Артефакты и результаты**

1. 📦 **Результат:** какие артефакты сохранять?
    
    - `text_semantic_roles.json` — определённые категории, уровень, домен
        Да, возможно с учетом частотности замен, а также способа замены по словарю или через NER.частотность
    - `unresolved_terms.csv` — поля, которые не удалось категоризовать
        да, с указанием id (по полю переданному при  вызове операции), подумай какую еще информацию надо
    - `category_mappings.csv` — файл замен (можно править вручную)
        Да, обязательно
2. 📊 **Метрики:**
    
    - `num_matched`
        
    - `num_unresolved`
        
    - `num_auto_clustered`
    Да, а также по размеру текста, пропущенным значениям, а также  возможно топ замены с  визуализацией дапример круговыми диаграмми и другие (все визуализации через visualization.py)
3. 📝 **Нужен ли итоговый HTML-отчёт** по полю (см. `text.py`)?
    Нет, операция возвращает список артефактов задаче, которая формирует общий json, а только потом при необходимости строится html text.py - не совсем верный пример, хоть и какая-то реализация

---

### 🔄 **4. Работа со словарями**

1. 📂 Словари (категории, позиции, домены, уровни):
    
    - Загружаются через `kwargs` → `dictionary_path`?
        
    - Или отдельными файлами внутри `task_dir/dictionaries`?
    - Загрузка внешних словарей как в примере с attribute.py из внешней диреткории. А словари с частотностью и картой замен  ВЫГРУЖАЕТСЯ - в {task_dir}\dictionaries
    - {task_dir} передается в операцию внешней задачей
        
2. 📑 Нужно ли формировать комбинированный словарь на лету?
    Если используется пользовательский словарь, то втсроенный словарь игнорнируется
    
3. 🌐 Поддержка многоязычия: нужно ли обрабатывать русские и английские синонимы одновременно?
    да, весь текст с кодировкой UTF8/UTF-16, пользователь может указывтаь подряд и английскийе и русскийе названия и вьетнамские.

---

### 🧬 **5. Специальные кейсы**

1. 🧹 **Обработка исключений:**
    
    - Фразы вроде «любая работа», «работник» — рандомные или флаг `OTHER/UNKNOWN`?
    Если флаг замен для ненайденных категорий включен, NER также не сработал?  (или для него также)  используем заданный список замен (также перекрывается пользовательским json _ тем же самым) bp 20 -30 замен, которые выбираются произвольно. Сама операция не делает замен, только предлагает их
    - Очищать текст: `lowercase`, `normalize`, `strip punctuation`?
    Да, также удалять стоп слова - с использованием  унаследованного модуля NLP
2. 🧠 **Обработка англ. слов в русских названиях:** например, `Middle .Net developer` — использовать как признак?
Да, это также касается не только русского языка, если посредине используется английский - это признак и можно выделить, также тоенизировтаь если надо. Однако это надо делать только для нераспознанных  категорий (и соответсвенно подмен)    

---

### 🔗 **6. Связь с другими компонентами**

1. ✅ **Нужно ли передавать выход модуля в `PreKAnonymityProfilingOperation`** как "обогащённые квази-идентификаторы"?
Скорее нет, однако,  в дальнейшем карта замен используется соответсвующей операцией для замен

2. 📁 **Нужно ли сохранять маппинг `исходный текст → классифицированная категория` для использования в синтезе / анонимизации?
    Да, обязательно, в этом и есть смысл выделения - пользователь может переделать замены до их окончатльного применения
    
3. 💾 **Артефакт `category_mappings.csv`:**
    
    - Где он должен храниться?
        В директории задач в подкаталоге mapping или dictionaries. Архитектурно предусмотреть шифрование.
    - Должен ли он обновляться автоматически?
    Да, при повторном запуске, возможно стоит сначала проверять не нашли ли мы уже замену или не поправили ли руками, остальные артефакты перезаписываются

## Вопросы по спецификации:

1. **О словаре и иерархии категорий**:
    - В документе указано, что используется JSON-схема для словаря. Предполагается ли полное соответствие примеру словаря из раздела 8, или нам нужно поддерживать более сложную структуру, как в примере `text_semantic_map.json` (где есть иерархия категорий)?
- - да, следовать `text_semantic_map.json, который надо дополнить еще и он поддается обобщению
- 
    - Как должна обрабатываться иерархия категорий при поиске совпадений? Например, если у нас есть общая категория "Development" и более конкретная "Python Developer", как правильно приоритизировать?
-  Ставим сначала более частную категорию (уровень ниже), но при анонимизации  (не в этой операции), а в анонимизации при недостаточной  k-anonymity делаем замену на более общую
-
1. **О разрешении коллизий**:
    - В разделе "КОЛЛИЗИИ" описан механизм приоритета. Как этот механизм должен взаимодействовать с тремя основными методами анализа (словарный поиск, NER, кластеризация)?
- Это для словарей, оставшиесеся не обработанные - смотрим по NER и правилам. Там уже коллизий нет. Вообще не найденные обрабатываем оющими заменами, как указано в json
- 
    - Правильно ли я понимаю, что словарный поиск имеет высший приоритет, NER - средний, а кластеризация - низший?
- Совершенно верно
- 
1. **О параметрах операции**:
    - Какие параметры должны быть настраиваемыми для пользователя (через kwargs или аргументы конструктора)?
- Задача (пользовательский срикпт) передает в операцию директорию к репозиторию данных, ссылку или dataframe для данных и название поля, все дополнительные параметры, вклюач яиспользовтаь ли NER анализ. Операция обратно передает список созданных артефактов и т.п.
    - Нужно ли поддерживать различные стратегии разрешения коллизий как параметры?
- Думаю на данном этапе (для MVP) нужно упростить. В конце концов на выходе пользователь получает карту замен (включая ссылку на отдельно указываемое ID поле) и может поправить карту перед анонимизацией
1. **О функциональности NER**:
    - Спецификация упоминает spaCy для NER. Нужно ли поддерживать все перечисленные модели для разных языков, или достаточно русской и английской?
- На данном этапе только русский и английский
-
    - Какие конкретно типы сущностей (PERSON, ORG, и т.д.) считаются приоритетными при анализе?
- Для нашего примера приоритетны названия тезнологий, специальности. Но в другой задаче (внутри того же примера, для другого поля - выделение вузов, а также других организаций)
1. **О кластеризации "невыясненных" записей**:
    - Какой алгоритм или метод кластеризации предпочтителен для группировки неопознанных текстов?
    - По ключевым слловам, вместо алиаса категории поставляется ключевое слово. Можешь предложить более грамотный метод
    - 
    - Как определять пороговые значения для включения в кластер?
- Пожалуйста предложи значение по умолчанию, пользователь может его поправить
- 
1. **Об артефактах и визуализации**:
    - Для визуализаций требуется использовать существующие функции из `visualization.py`. Есть ли конкретные требования к типам или стилям визуализаций?
- Требований нет, но визуализация должна быть наглядной
    - Какую дополнительную метаинформацию нужно включать в JSON-артефакты помимо основных категорий и частот?
    -  Например 
    -{
  "record_id": 10293,
  "original_text": "python разработчик",
  "normalized_text": "python разработчик",
  "matched_alias": "python_dev",
  "matched_category": "Python Ecosystem Specialist",
  "matched_domain": "Python Ecosystem",
  "matched_seniority": "Any",
  "match_method": "dictionary",
  "match_score": 0.95,
  "conflict_candidates": ["Developer", "Software Development"],
  "token_count": 2,
  "language_detected": "ru",
  "source_field": "post"
}
1. **О взаимодействии с другими модулями**:
    - Как результаты операции должны интегрироваться с другими операциями профилирования?
-Другие операции при необходимости получают на вход извлеченную крату в csv
    -Планируется ли использовать результаты этой операции в operations для anonymization?
- Да, планируется, но аз пределами клнкретной задачи профайлинга
1. **О многоязычности**:
    - Как обрабатывать смешанные тексты, содержащие термины на разных языках?
    -Включение английского при наличии русского будет свидетельством тега, если он не извлечен словарем (где языки равны) то его можно попытаться извлечь.  Для примера мы используем приоритет на русский, но вообще - не установлено
    
    - Есть ли приоритет для определённого языка при анализе?
     -Нет
2. **О производительности**:
    - Ожидаемые объёмы данных для анализа?
    -Не большие - до 10 000, в дальнейшем - до миллиона. не забыть о прогрессе
    - Есть ли ограничения по времени выполнения или по памяти?
- - нет на данном этапе

