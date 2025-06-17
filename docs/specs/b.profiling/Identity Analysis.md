## **Реализация IdentityAnalysisOperation (pamola_core/profiling/analyzers/identity.py)**

### 📌 **Цель:**

Создать новый модуль `identity.py` с операцией `IdentityAnalysisOperation` для анализа уникальных идентификаторов пользователей (UID) и связанных с ними данных, в рамках профилирования таблицы IDENTIFICATION:

ID, resume_id, first_name, last_name, middle_name, birth_day, gender, file_as,  UID
8,242472597,Williams,Hartford,,2003-08-23,Мужчина,Hartford Williams,20487862DD407108131F5D1D645545B5
9,242472597,Williams,Hartford,,2003-08-23,Мужчина,Hartford Williams,20487862DD407108131F5D1D645545B5
10,242472615,Азат,Галеев,,,Мужчина,Галеев Азат,3875EFC2F9E260533046EFC8A06A1C1E
11,242472615,Азат,Галеев,,,Мужчина,Галеев Азат,3875EFC2F9E260533046EFC8A06A1C1E
12,242472615,Азат,Галеев,,,Мужчина,Галеев Азат,3875EFC2F9E260533046EFC8A06A1C1E
13,242472615,Азат,Галеев,,,Мужчина,Галеев Азат,3875EFC2F9E260533046EFC8A06A1C1E
14,242472615,Азат,Галеев,,,Мужчина,Галеев Азат,3875EFC2F9E260533046EFC8A06A1C1E
15,242472615,Азат,Галеев,,,Мужчина,Галеев Азат,3875EFC2F9E260533046EFC8A06A1C1E


Модуль представляет собой реализацию библиотечных функций (операций), операция не должна содержать детали относительно резюме (это ответсвенность задачи), только функции, пакет pamola_core.profiling пишется для более широкого контекста, не только для анализа резюме.

* Модуль располагается в pamola_core.profiling.analyzers (identity.py). при необходимости общие функции выносятся в pamola_core.profiling.commons.identity_utils.py 
*  Шапка и все комментарии по английски, аналогично, например, модулю categorical.py 
* Все вводы и выводы информации осуществляются с имеющимися пакетами visualization.py и io.py (документация в аттаче), причем все визуализации (обязательное использование visualization.py) в формате png и json метрки размещаются прямо в каталоге задачи, 
* Конфигурация не хранится в модуле операции, она передается задачей (пользовательским скриптом), как и настройки scope (поля или полей, к которым применяется операция. Документация на операции и задачи в аттаче 
* Мы не обрабатываем на данном этапе специфически очень большие файлы (более миллиона строк), но векторизация должна быть предусмотрена архитектурно
### ✅ **Основной класс**

`class IdentityAnalysisOperation(FieldOperation):     """     Operation for analyzing identity fields and their consistency.     """      def __init__(self, uid_field: str, reference_fields: List[str],                   id_field: Optional[str] = None, description: str = "")`

---

### 🧠 **Функциональные требования (MVP)**

#### 1. **Анализ консистентности UID:**
Представить как более общую задачу

- Проверить, соответствует ли `UID` хешу `md5(last_name + first_name + birth_day)`
    
- Поддержка возможного отсутствия одного из reference_fields (пропустить сравнение, логировать)
    
- Визуализировать процент совпадений и несоответствий
    
- Сохранить список top-N ошибок
    

#### 2. **Анализ количества резюме на UID:**
Это должно быть как общая задача (без акцента на резюме), вместо этого сравнивается вариация одного id (UID), относительно другого (resume_id)

- Подсчитать количество различных `resume_id` для каждого UID
    
- Построить распределение количества резюме на человека
    
- Визуализировать через bar plot или histogram
    
- Сохранить JSON с top-N по количеству резюме
    

#### 3. **Флаг пересечений между разными UID:**

- Найти случаи, где `first_name` + `last_name` совпадают, а `UID` — различен
    
- Сохранить таблицу пересечений как JSON
    

---

### 🗂 **Архитектура и интеграция**

#### ✔ Наследование

Класс должен наследоваться от `FieldOperation` из `pamola_core.utils.ops.op_base`.

#### ✔ Регистрация

Операция должна быть зарегистрирована через `@register` с указанием версии:

python

CopyEdit

`@register(version="1.0.0")`

#### ✔ Используемые пакеты:

- `hashlib` для MD5
    
- `pandas` для анализа
    
- `matplotlib.pyplot` или `visualization.py` для визуализаций
    
- `json`, `pathlib`, `datetime`, `collections`
    

#### ✔ Использование инфраструктуры:

- Чтение данных: `data_source.get_dataframe("main")`
    
- Репорты: `reporter.log(...)`
    
- Артефакты: `result.add_artifact(...)`
    
- Прогресс: `progress_tracker.update(...)` (опционально)
    

---

### 📂 **Файлы, которые принять во внимание:**

1. `pamola_core/utils/ops/op_base.py` – определение `FieldOperation`, `OperationResult`
    
2. `pamola_core/utils/ops/op_data_source.py` – `DataSource` и `get_dataframe(...)`
    
3. `pamola_core/utils/ops/op_result.py` – `OperationResult`, `OperationArtifact`
    
4. `pamola_core/utils/ops/op_registry.py` – `@register`
    
5. `pamola_core/utils/io.py` – для записи JSON
    
6. `pamola_core/utils/visualization.py` – для генерации графиков
    
7. `7 IDENT.md` – описание таблицы, логики UID, описание задачи
    
8. `index.md` – документация по Task/Operation API
    
9. `visualization.md`, `io.md` – документация API артефактов
    

---

### 📊 **Артефакты, которые должны быть сгенерированы:**

|Название|Тип|Назначение|
|---|---|---|
|`uid_consistency.json`|JSON|статистика по совпадениям UID|
|`uid_mismatch_examples.json`|JSON|примеры неправильных UID|
|`resume_counts_per_uid.json`|JSON|статистика по количеству резюме|
|`uid_cross_match.json`|JSON|случаи, где совпадают имена, но различный UID|
|`resume_count_distribution.png`|PNG|гистограмма по количеству резюме|
|`uid_match_pie.png`|PNG|круговая диаграмма: valid vs invalid UID|

---

### 📋 **Метрики (добавить в OperationResult):**

|Название|Описание|
|---|---|
|`valid_uid_count`|количество UID, соответствующих контрольной формуле|
|`invalid_uid_count`|количество UID, не соответствующих формуле|
|`max_resume_count`|максимум резюме на одного UID|
|`average_resume_count`|среднее количество резюме на UID|
|`cross_uid_match_count`|число случаев совпадений имен при разных UID|

---

### 📎 **Особенности реализации**

- Проверка наличия всех необходимых полей в DataFrame перед анализом
    
- Логгирование через `reporter.log(...)` на всех этапах
    
- Все визуализации сохраняются в `task_dir 
    
- Все JSON и примеры — в `task_dir 
    

---

### 🛑 **Что НЕ входит в MVP (можно позже):**

- Использование Levenshtein или fuzzy matching
    
- Распараллеливание обработки UID
    
- Кластеризация по профилю
    

---

### 📚 **Пример конфигурации задачи:**


`self.add_operation(data_source= "D:\VK__DEVEL\DATA\raw\IDENT.csv", task_dir="D:\VK__DEVEL\DATA\processed\profiling\identity",     "IdentityAnalysisOperation",     uid_field="UID",     reference_fields=["last_name", "first_name", "birth_day"],     id_field="resume_id" )`