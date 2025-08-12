## Мини-спецификация на разработку модуля анализа k-anonymity для PAMOLA.CORE


## 🧩 1. Название и цель модуля

### Предложенное имя модуля:

`KAnonymityProfilerOperation`

### Цель:

Предоставить **предварительную профилизацию** квази-идентификаторов на предмет устойчивости к деанонимизации через расчет **k-анонимности**, **энтропии**, и **уникальности**. Модуль **не реализует полноценную проверку на соответствие модели k-anonymity**, а служит для анализа рисков и подготовки к дальнейшим мерам защиты.

### Возможное разделение:

* **Базовая часть** (`PreKAProfiler` --> pamola_core.profiling.analyzers.anonymity.py): выполняет расчет всех метрик, 
    
* **Вспомогательные компоненты**: (вспомогательные функции реализуются в pamola_core.profiling.commons.anonymity_utils.py)
    
    * генератор индексов `KA_*`
        
    * расчет энтропии и уязвимости
        
    * визуализация
        
    * экспорт артефактов
        

* * *

## ⚙️ 2. Основные функциональные требования
Основной функционал операции предстаялеяет собой анализ уникальности полей, их энтропии, а также значений k-anonymity для сочетаний из группы полей (от одного поля до всех). При этом некоторые из осчетаний могут быть запрещены и также передаются в качестве параметров задачей (внешним скриптом)

### 2.1 Генерация KA-индексов
KA- индексы, это получаемые из названий полей аббревиатуры их сочетаний, генерируемые уникальн. 
#### Принцип именования:

`KA_` + конкатенация первых букв полей (уникальность обеспечивается увеличением длины префикса до 4 букв при коллизиях)

#### Алгоритм:

1. Каждому полю присваивается индекс из 2 первых символов наименования поля
    
2. Если конфликт — увеличить длину до 3 или 4
    
3. Объединить по `_` и добавить префикс `KA_`
    
4. Сохранить мапу `{KA_*: [fields]}` → `ka_index_map.csv` (в `dictionaries`)
    

* * *

### 2.2 Метрики по каждому индексу

| Метрика | Описание |
| --- | --- |
| `min_k` | минимальное значение k |
| `max_k` | максимальное значение |
| `mean_k` | среднее значение |
| `median_k` | медиана |
| `k=1_count` | количество уникальных записей |
| `k<treshold` | количество уязвимых записей |
| `%_k<treshold` | процент уязвимых записей |
| `entropy` | энтропия распределения групп |
| `num_groups` | общее число уникальных групп |
| `total_records` | общее число записей |

#### Дополнительно:

* оценка влияния добавления полей: снижение `mean_k` и рост `k=1` — в процентах
    
* визуализация по этим метрикам
    

* * *

### 2.3 Конфигурация

| Параметр              | Описание                                                                                                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fields_combinations` | список списков квази-идентификаторов                                                                                                                                                                                         |
| `id_fields`           | список уникальных идентификаторов (например `resume_id`) - для вывода словарей топ значений                                                                                                                                  |
| `treshold_k`          | пороговое значение (по умолчанию 5)                                                                                                                                                                                          |
| `task_dir`            | домашняя директория задачи                                                                                                                                                                                                   |
| `data_source`         | путь к данным или `DataSource` объект (также добавить возможность передачи данных для чтения через настраиваемый источник  разделитель, кодировка, текстовый квалификатор, если источник это ссылка на файл, а не dataframe) |

* * *

### 2.4 Результаты и артефакты

| Артефакт                      | Формат | Категория     | Пояснение                                                                                 |
| ----------------------------- | ------ | ------------- | ----------------------------------------------------------------------------------------- |
| `ka_index_map.csv`            | csv    | dictionary    | список индексов и полей                                                                   |
| `ka_metrics.csv`              | csv    | metrics       | таблица метрик по каждому индексу, включая уникальные значения, статистику по k, энтропию |
| `ka_vulnerable_records.json`  | json   | metrics       | список уязвимых записей по KA                                                             |
| `ka_entropy_distribution.png` | png    | visualization | гистограмма значений энтропии                                                             |
| `ka_k_distribution.png`       | png    | visualization | распределение k по диапазонам                                                             |
| `ka_vulnerable_curve.png`     | png    | visualization | % записей при k=2/5/20                                                                    |
| `ka_comparison_spider.png`    | png    | visualization | spider chart для всех KA                                                                  |
| `ka_field_uniqueness.png`     | png    | visualization | уникальность по отдельным полям                                                           |
СМ. по визуализации ниже
* * *

## 🧱 3. Архитектурные аспекты

### 3.1 Паттерны

* Класс должен наследовать `BaseOperation`, а также рабоатть с источником данных (Datasource)
    
* Использует `execute(data_source, task_dir, reporter, **kwargs)`
    
* Регистрируется через `@register(...)` с категорией `profiling`
    
* Описание, версия, и зависимости указываются в аннотациях
    

### 3.2 Утилиты

| Модуль | Используемые функции |
| --- | --- |
| `io.py` | `save_json`, `save_dataframe`, `ensure_dir` |
| `visualization.py` | `create_histogram`, `create_bar_plot`, `create_boxplot`, `create_line_plot` (для кривой соответствия), `plot_multiple_fields` |
| `progress.py` | `track_progress` при расчете большого количества индексов |
| `logging.py` | логирование шагов исполнения и ошибок |

### 3.3 Комментарии и документация

* Все docstring и inline-комментарии — **только на английском**
    
* Стиль соответствует стилю других операций
    

* * *

## 📊 4. Визуализация

### Необходимые типы диаграмм:

| Название | Реализован? | Требуется |
| --- | --- | --- |
| `histogram` | ✅ | для k |
| `bar_plot` | ✅ | для % соответствия по treshold |
| `boxplot` | ✅ | для значений k |
| `line_plot` | ✅ | для кривой соответствия по treshold |
| `spider_chart` | ❌ | НУЖНО ДОБАВИТЬ |
| `field_uniqueness bar` | ✅ | через `plot_multiple_fields` |

⚠️ **Необходимо расширить visualization.py и vis_helpers для spider chart**.

* * *

## ⚙️ 5. Пример конфигурации

```json
{
  "operation": "KAnonymityProfilerOperation",
  "params": {
    "fields_combinations": [
      ["birth_day", "gender", "first_name"],
      ["birth_day", "gender", "last_name"]
    ],
    "id_fields": ["resume_id"],
    "treshold_k": 5
  },
  "data_source": "data/resumes.csv",
  "task_dir": "output/tasks/ident/"
}
```

## УТОЧНЕНИЯ

### Part 1

1)For the spider/radar chart implementation, should it follow the same pattern as other charts with a Plotly primary implementation and a Matplotlib fallback, or is Plotly-only acceptable?

- Plotly primary implementation и только в крайне необходимых случаях Matplot
2) For the combined chart (Image 4), is this meant to be a new chart type or an extension of the existing bar chart that adds a line component with a secondary Y-axis?
Я бы предпочел обновление box с обратной совместимостью

3) Do you need internationalization (i18n) support for chart labels and titles, or just the ability to set non-English text?
just the ability to set non-English text

4) For the group visualization (Image 1), is this using the existing bar chart with specific formatting, or does it need additional customization?
Предложи дополнительную кастомизацию

5) Are there any specific adjustments needed for the existing line chart to support the k-threshold visualization (Image 2)?
Предложи
6) Should all new visualization functions be added to the main visualization.py module, following the pattern of existing functions like create_bar_plot(), create_histogram(), etc.?
Да, добавлены в visualization.py

7) Are there any specific color schemes that should be used for these visualizations?
Нет, но если необходимо, давай установим конфигурацию для visualization.py (если это не очень сложно), например в виде схемы по умолчанию, но при возможности - смены

Пожалуйста, все еще не пиши код - выведи дополнительные вопросы (если есть), короткий анализ и   последовательность модулей которые надо изменить или добавить. включи и pie charts

1)For spider charts, do you need the ability to adjust the number of axes dynamically, or will it always be the four metrics shown (uniqueness, vulnerability, normalized k, entropy)?
Мы пишем общий модуль, поэтому динамически - определяется операцией (функциональным модулем)

2) For the combined chart, do you need other combinations beyond bar+line (like line+area, etc.)?
Да, но давай создадим все же отдельнывй модуль, чтобы не усложнять имеющийся bar
3) Are there specific interactions needed for these charts (tooltips, zooming, etc.)?
На этом этапе нет, это статические картинки, сохраняющиеся в png

4) Would you like the pie chart to support subgroups (sunburst chart) as well?
Да

5) For the customized group visualization (Image 1), do you need the ability to sort by different criteria or filter specific groups?
Да, но предаются как необязательные параметры

### Part 2
---

### 1) **О расчете энтропии**

> **Формула энтропии**  
> ✅ **Используется стандартная формула Шеннона:**

H(X)=−∑i=1np(xi)⋅log⁡2p(xi)H(X) = - \sum_{i=1}^{n} p(x_i) \cdot \log_2 p(x_i)

- p(xi)p(x_i) – доля записей с конкретным значением в квазиидентификаторе.
    
- Рекомендуется использовать **base=2** (биты), т.к. значение удобно для восприятия в аналитике.
    

> **Нужно ли нормализовать энтропию для spider chart?**  
> ✅ **Да, нормализация обязательна**, чтобы сравнивать показатели по разным комбинациям.

- Формула нормализации:
    

Hnorm(X)=H(X)log⁡2nH_{norm}(X) = \frac{H(X)}{\log_2 n}

где nn — количество уникальных значений. Это ограничит результат в диапазоне [0,1].

---

### 2) **О формировании KA-индексов**

> **Есть ли специфические правила?**  
> ✅ Ваши правила логичны, дополним деталями:

- Пользователь задаёт список **исходных полей** (quasi-identifiers).
    
- Пробегаются **все сочетания (от 2 до N)**, по умолчанию N=4, но можно задать.
    
- Исключения (например, одинарные поля или конкретные комбинации) задаются в **исключающем списке**.
    
- **Индексация**:
    
    - Формируется индекс в формате `KA_<аббревиатура_поля1>_<аббревиатура_поля2>_...`
        
    - Аббревиатуры: первые 2–4 символа от каждого поля (можно настраивать).
        
    - Если возникает **дублирование индекса**, применяется следующая стратегия:
        
        1. Увеличение количества символов в названии поля.
            
        2. Добавление постфикса `_1`, `_2`, … `_n`.
            
        3. При невозможности устранить конфликт — логируем ошибку.
            

🟡 _Дополнительно: можно включить хэш как fallback-опцию (например, crc32 от строки полей), но это ухудшает читаемость._

---

### 3) **О выходных артефактах**

> **Файл `ka_metrics.csv` должен содержать:**

|Столбец|Описание|
|---|---|
|`#`|Порядковый номер|
|`KA_INDEX`|Название комбинации|
|`FIELDS`|Список полей (можно добавить)|
|`KA_MIN`|Минимальное k|
|`KA_MAX`|Максимальное k|
|`KA_MEAN`|Среднее k|
|`KA_MEDIAN`|Медиана k|
|`UNIQUE_VALUES (%)`|Доля уникальных комбинаций|
|`VULNERABLE_RECORDS (%)`|Доля записей с k<thresholdk < threshold, по умолчанию k=5k=5|
|`ENTROPY`|Энтропия (опционально – нормализованная)|

✅ **Файл должен называться**:  
`ka_{general_index}_metrics.csv` — где general_index формируется из полного списка анализируемых полей (можно хэшировать).

> **Файл `ka_vulnerable_records.json`** должен содержать для каждого индекса:

```json
[
  {
    "ka_index": "KA_FN_LN",
    "min_k": 1,
    "vulnerable_count": 573,
    "vulnerable_percent": 7.23,
    "top_10_vulnerable_ids": [ "resume123", "resume456", ... ]
  }
]
```

📌 _Можно расширить до top_N и включать пример значений квазиидентификаторов, но только если не нарушается безопасность._

---

### 4) **О расчете k-анонимности для больших датасетов**

> **Оптимизации:**

- Да, при работе с большим числом строк и полей:
    
    - Используйте **Dask** или **Polars** (при необходимости группировок по миллионам записей).
        
    - Ключевой шаг — `groupby + count`, его можно распараллелить.
        
    - Вариант с **SQLite/SQL-группировками** — тоже возможен, если данные на диске.
        
    - Для кластерных задач: **Spark DataFrame** или **DuckDB** с in-memory аналитикой.
        

---

### 5) **О выявлении и визуализации "проблемных" комбинаций**

> **Нужно ли выделять комбинации с k=1?**  
> ✅ Да, рекомендуется визуально **подсвечивать комбинации, где `VULNERABLE_RECORDS (%) > X%`**, особенно если среди них есть записи с `k = 1`.

- Для визуализации:
    
    - Spider chart: показать normalized entropy, unique values %, vulnerability %.
        
    - **Heatmap или Bar chart**: KA_INDICES по оси X, показатель риска (например, % уязвимых) — по Y.
        
    - Таблица ранжирования: `TOP-10 наиболее уязвимых комбинаций` по убыванию риска.
        

> **Дополнительная визуализация**:

- Диаграмма "Top risk combinations" (bar)
    
- Группировка по количеству атрибутов: 2, 3, 4 → какие наиболее рискованные
    
- Визуальное сравнение между entropy и % уникальных — может выявить слабые места
    
