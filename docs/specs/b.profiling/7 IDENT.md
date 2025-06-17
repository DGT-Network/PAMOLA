# Анализ профилирования таблицы IDENTIFICATION и план рефакторинга

## 1. Анализ текущей структуры и выявление проблем

### 1.1 Общее понимание данных и задачи

Таблица IDENTIFICATION содержит идентификационные данные людей:

- `ID` - первичный ключ записи
- `resume_id` - идентификатор резюме (один человек может иметь несколько резюме)
- `first_name`, `last_name`, `middle_name` - компоненты ФИО
- `birth_day` - дата рождения
- `gender` - пол
- `file_as` - форматированное имя (например, "Фамилия Имя")
- `UID` - уникальный идентификатор человека, рассчитанный как MD5 от last_name+first_name+birth_day

Задача профилирования этой таблицы включает:

1. Анализ полноты и качества данных
2. Анализ имен, фамилий и отчеств
3. Анализ полов и их распределения
4. Анализ дат рождения и выявление аномалий
5. Анализ количества резюме на человека
6. Проверка консистентности UID
7. Анализ вариабельности внутри групп резюме

### 1.2 Текущая реализация

Текущая реализация в `profile_ident.py`:

1. Содержит много бизнес-логики напрямую в задаче
2. Вызывает низкоуровневые функции анализа без использования операций
3. Не использует полностью возможности фреймворка операций
4. Имеет избыточный код для отслеживания прогресса и управления артефактами

### 1.3 Вопросы для дальнейшего анализа

1. **Интеграция существующих операций**:
    
    - Как именно интегрировать существующие операции CategoricalOperation, DateOperation, GroupOperation?
    - Как определить наборы полей для каждой операции?
2. **Недостающие операции**:
    
    - Необходимы ли специализированные операции для анализа UID и file_as?
    - Нужна ли специальная операция для анализа резюме?
3. **Управление зависимостями между операциями**:
    
    - Как передавать результаты одной операции другой?
    - Как избежать повторного анализа одних и тех же данных?
4. **Формирование отчетов**:
    
    - Какой формат отчета нужно сохранить?
    - Как интегрировать результаты операций в единый отчет?

## 2. Анализ необходимых модулей

### 2.1 Существующие модули в pamola_core.profiling.analyzers

1. **categorical.py** - подходит для анализа полей:
    
    - first_name, last_name, middle_name (ФИО)
    - gender (пол)
    - file_as (форматированное имя)
2. **date.py** - подходит для анализа поля:
    
    - birth_day (дата рождения)
3. **group.py** - подходит для:
    
    - анализа вариабельности внутри групп resume_id
    - выявления изменений в полях для одного resume_id
4. **correlation.py** - может использоваться для:
    
    - анализа связи между компонентами ФИО
    - проверки связи между resume_id и UID

### 2.2 Недостающие модули/операции

1. **IdentityAnalysisOperation**: Необходима операция для:
    
    - Анализа консистентности UID
    - Подсчета резюме на человека
    - Проверки генерации UID по полям
2. **CompletenessOperation**: Операция для общего анализа полноты данных, если ее нет в существующей структуре.
    
3. **UniquenessOperation**: Операция для анализа уникальности полей, если ее нет.
    
4. **DuplicatesAnalysisOperation**: Специализированная операция для анализа дубликатов по различным наборам полей.
    

## 3. План рефакторинга операций профилирования

### 3.1 Доработка CategoricalOperation

```python
class CategoricalOperation(FieldOperation):
    """
    Operation for analyzing categorical fields.
    """
    
    def __init__(self, field_name: str, top_n: int = 15, min_frequency: int = 1, description: str = ""):
        super().__init__(field_name, description or f"Analysis of categorical field '{field_name}'")
        self.top_n = top_n
        self.min_frequency = min_frequency
    
    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        # Текущая реализация должна быть адаптирована, чтобы:
        # 1. Использовать data_source.get_dataframe()
        # 2. Сохранять результаты в стандартизированных директориях
        # 3. Возвращать OperationResult с артефактами
        # 4. Обновлять progress_tracker
        # 5. Использовать reporter для логирования
```

Необходимые улучшения:

- Использование стандартных путей и форматов для артефактов
- Улучшение обработки пропущенных значений
- Интеграция с системой прогресса
- Адаптация для использования с reporter

### 3.2 Доработка DateOperation

```python
class DateOperation(FieldOperation):
    """
    Operation for analyzing date fields.
    """
    
    def __init__(self, field_name: str, min_year: int = 1940, max_year: int = 2005, 
                 id_column: Optional[str] = None, uid_column: Optional[str] = None,
                 description: str = ""):
        super().__init__(field_name, description or f"Analysis of date field '{field_name}'")
        self.min_year = min_year
        self.max_year = max_year
        self.id_column = id_column  # Для анализа изменений в пределах одного resume_id
        self.uid_column = uid_column  # Для анализа изменений в пределах одного UID
```

Необходимые улучшения:

- Адаптация для работы с birth_day
- Улучшение обнаружения аномалий в датах рождения
- Расчет возрастов и их распределения
- Визуализация распределения по годам

### 3.3 Доработка GroupOperation

```python
class GroupOperation(BaseOperation):
    """
    Operation for analyzing groups of records.
    """
    
    def __init__(self, group_field: str, fields_weights: Dict[str, float],
                 min_group_size: int = 2, set_name: str = "",
                 description: str = ""):
        super().__init__(description or f"Analysis of group variations by {group_field}")
        self.group_field = group_field
        self.fields_weights = fields_weights
        self.min_group_size = min_group_size
        self.set_name = set_name
```

Необходимые улучшения:

- Адаптация для работы с resume_id как группирующим полем
- Анализ вариабельности по полям ФИО и birth_day
- Улучшение обнаружения изменений внутри групп

### 3.4 Создание IdentityAnalysisOperation (новый модуль)

Необходимо создать новую операцию в файле `pamola_core/profiling/analyzers/identity.py`:

```python
class IdentityAnalysisOperation(FieldOperation):
    """
    Operation for analyzing identity fields and their consistency.
    """
    
    def __init__(self, uid_field: str, reference_fields: List[str], 
                 id_field: Optional[str] = None, description: str = ""):
        super().__init__(uid_field, description or f"Analysis of identity field '{uid_field}'")
        self.reference_fields = reference_fields  # Fields used to verify UID
        self.id_field = id_field  # Field for resume_id/person_id
    
    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        # Реализация должна:
        # 1. Анализировать консистентность UID
        # 2. Проверять соответствие UID и полей reference_fields
        # 3. Анализировать количество резюме на человека
        # 4. Визуализировать результаты
```

### 3.5 Создание DuplicatesAnalysisOperation (новый модуль)

Необходимо создать новую операцию в файле `pamola_core/profiling/analyzers/duplicates.py`:

```python
class DuplicatesAnalysisOperation(BaseOperation):
    """
    Operation for analyzing duplicates based on various field combinations.
    """
    
    def __init__(self, field_sets: List[List[str]], description: str = ""):
        super().__init__(description or "Analysis of duplicates")
        self.field_sets = field_sets  # List of field combinations to check for duplicates
    
    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        # Реализация должна:
        # 1. Анализировать дубликаты по каждому набору полей
        # 2. Генерировать статистику дубликатов
        # 3. Сохранять примеры дубликатов
```

### 3.6 Проверка/создание CompletenessOperation

Необходимо проверить наличие или создать операцию анализа полноты:

```python
class CompletenessOperation(BaseOperation):
    """
    Operation for analyzing data completeness.
    """
    
    def __init__(self, description: str = ""):
        super().__init__(description or "Analysis of data completeness")
    
    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        # Реализация должна:
        # 1. Анализировать заполненность полей
        # 2. Визуализировать распределение заполненности
```

### 3.7 Проверка/создание UniquenessOperation

Необходимо проверить наличие или создать операцию анализа уникальности:

```python
class UniquenessOperation(BaseOperation):
    """
    Operation for analyzing field uniqueness.
    """
    
    def __init__(self, description: str = ""):
        super().__init__(description or "Analysis of field uniqueness")
    
    def execute(self, data_source, task_dir, reporter, progress_tracker=None, **kwargs):
        # Реализация должна:
        # 1. Анализировать уникальность значений полей
        # 2. Визуализировать распределение уникальности
```

## 4. План реализации задачи profile_ident.py

```python
#!/usr/bin/env python
"""
Профайлинг таблицы IDENTIFICATION в рамках проекта анонимизации резюме PAMOLA.CORE (Privacy-Preserving AI Data Processors).

Этот скрипт выполняет анализ основных идентификационных данных из резюме, включая
имена, даты рождения, пол и другие идентифицирующие поля. Данная задача является
первой в цепочке профилирования и не имеет зависимостей.
"""

import argparse
import os
import time
from pathlib import Path

from pamola_core.utils.cli import create_profiling_parser
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import create_operation_instance

class IdentificationProfilerTask(BaseTask):
    """
    Задача профилирования таблицы IDENTIFICATION.
    """
    
    def __init__(self):
        super().__init__(
            task_id="identification",
            task_type="profiling",
            description="Profiling of IDENTIFICATION table",
            dependencies=[]  # Нет зависимостей
        )
    
    def configure_operations(self):
        """
        Настройка операций для выполнения.
        """
        self.logger.info("Configuring operations for IDENTIFICATION profiling")
        
        # 1. Анализ полноты и уникальности
        self.add_operation("CompletenessOperation")
        self.add_operation("UniquenessOperation")
        
        # 2. Анализ категориальных полей (имена, пол)
        name_fields = ["first_name", "last_name", "middle_name"]
        for field in name_fields:
            if field in self.dataframe.columns:
                self.add_operation("CategoricalOperation", field_name=field, top_n=20)
        
        if "gender" in self.dataframe.columns:
            self.add_operation("CategoricalOperation", field_name="gender")
            
        if "file_as" in self.dataframe.columns:
            self.add_operation("CategoricalOperation", field_name="file_as")
        
        # 3. Анализ дат рождения
        if "birth_day" in self.dataframe.columns:
            self.add_operation("DateOperation", 
                              field_name="birth_day", 
                              min_year=1940, max_year=2005,
                              id_column="resume_id" if "resume_id" in self.dataframe.columns else None,
                              uid_column="UID" if "UID" in self.dataframe.columns else None)
        
        # 4. Анализ идентификаторов
        if "UID" in self.dataframe.columns:
            # Определяем поля для проверки UID
            reference_fields = []
            if "first_name" in self.dataframe.columns:
                reference_fields.append("first_name")
            if "last_name" in self.dataframe.columns:
                reference_fields.append("last_name")
            if "birth_day" in self.dataframe.columns:
                reference_fields.append("birth_day")
                
            self.add_operation("IdentityAnalysisOperation", 
                              uid_field="UID",
                              reference_fields=reference_fields,
                              id_field="resume_id" if "resume_id" in self.dataframe.columns else None)
        
        # 5. Анализ дубликатов
        field_sets = []
        # Дубликаты по персональным данным
        person_fields = []
        if "first_name" in self.dataframe.columns:
            person_fields.append("first_name")
        if "last_name" in self.dataframe.columns:
            person_fields.append("last_name")
        if "birth_day" in self.dataframe.columns:
            person_fields.append("birth_day")
        if person_fields:
            field_sets.append(person_fields)
            
        # Дубликаты по UID
        if "UID" in self.dataframe.columns:
            field_sets.append(["UID"])
            
        # Дубликаты по resume_id
        if "resume_id" in self.dataframe.columns:
            field_sets.append(["resume_id"])
            
        if field_sets:
            self.add_operation("DuplicatesAnalysisOperation", field_sets=field_sets)
        
        # 6. Анализ вариабельности групп резюме
        if "resume_id" in self.dataframe.columns:
            # Определяем поля и веса для анализа вариабельности
            fields_weights = {}
            if "first_name" in self.dataframe.columns:
                fields_weights["first_name"] = 0.3
            if "last_name" in self.dataframe.columns:
                fields_weights["last_name"] = 0.4
            if "birth_day" in self.dataframe.columns:
                fields_weights["birth_day"] = 0.3
                
            if fields_weights:
                self.add_operation("GroupOperation", 
                                  group_field="resume_id",
                                  fields_weights=fields_weights,
                                  min_group_size=2)
    
    def execute(self):
        """
        Выполнение операций профилирования.
        """
        self.logger.info("Executing IDENTIFICATION profiling task")
        
        # Создаем источник данных
        data_source = DataSource.from_dataframe(self.dataframe, "main")
        
        # Выполняем операции
        for i, operation_info in enumerate(self.operations):
            operation = operation_info["instance"]
            operation_name = operation_info["name"]
            
            self.logger.info(f"Executing operation {i+1}/{len(self.operations)}: {operation_name}")
            
            # Запускаем операцию
            result = operation.run(
                data_source=data_source,
                task_dir=self.task_dir,
                reporter=self.reporter,
                track_progress=True
            )
            
            # Проверяем результат
            if result.status != "success":
                self.logger.error(f"Operation {operation_name} failed: {result.error_message}")
                if not self.config.continue_on_error:
                    return False
            
            # Добавляем артефакты в отчет
            for artifact in result.artifacts:
                self.reporter.add_artifact(
                    artifact_type=artifact.artifact_type,
                    path=artifact.path,
                    description=artifact.description
                )
        
        return True


def main():
    """
    Основная функция профайлинга таблицы IDENTIFICATION.
    """
    # Настройка аргументов командной строки
    parser = create_profiling_parser(
        'Profile IDENTIFICATION table',
        'data/raw/IDENT.csv',
        'logs/profile_ident.log'
    )
    args = parser.parse_args()
    
    # Создаем и выполняем задачу
    task = IdentificationProfilerTask()
    
    # Измерение времени выполнения
    start_time = time.time()
    
    # Инициализация и выполнение
    if task.initialize(args):
        success = task.execute()
        task.finalize(success)
    else:
        success = False
    
    # Вывод результата
    execution_time = time.time() - start_time
    print(f"Task completed {'successfully' if success else 'with errors'} in {execution_time:.2f} seconds")
    
    return 0 if success else 1


if __name__ == "__main__":
    main()
```

## 5. Спецификация необходимых дополнительных файлов

### 5.1 Новые модули операций

1. **pamola_core/profiling/analyzers/identity.py**
    
    - Операция `IdentityAnalysisOperation` для анализа консистентности UID
    - Функции для проверки соответствия UID и полей
    - Функции для анализа количества резюме на человека
2. **pamola_core/profiling/analyzers/duplicates.py**
    
    - Операция `DuplicatesAnalysisOperation` для анализа дубликатов
    - Функции для анализа дубликатов по наборам полей
    - Функции для генерации статистики и примеров дубликатов
3. **pamola_core/profiling/analyzers/completeness.py** (если отсутствует)
    
    - Операция `CompletenessOperation` для анализа полноты данных
    - Функции для расчета метрик заполненности
    - Функции для визуализации полноты
4. **pamola_core/profiling/analyzers/uniqueness.py** (если отсутствует)
    
    - Операция `UniquenessOperation` для анализа уникальности полей
    - Функции для расчета метрик уникальности
    - Функции для визуализации уникальности

### 5.2 Утилитарные модули

1. **pamola_core/utils/tasks/base_task.py**
    
    - Класс `BaseTask` для стандартизации задач
    - Методы для управления операциями и формирования отчетов
2. **pamola_core/utils/tasks/task_config.py**
    
    - Класс `TaskConfig` для управления конфигурацией
    - Функции для загрузки и объединения конфигураций
3. **pamola_core/utils/tasks/task_registry.py**
    
    - Функции для регистрации выполнения задач
    - Функции для проверки зависимостей
4. **pamola_core/utils/tasks/task_reporting.py**
    
    - Класс `TaskReporter` для формирования отчетов
    - Функции для добавления операций и артефактов в отчет
5. **pamola_core/utils/tasks/utils.py**
    
    - Вспомогательные функции для работы с задачами

## 6. Структура данных и задачи

### 6.1 Структура данных IDENTIFICATION

Таблица IDENTIFICATION содержит основные идентификационные данные людей:

|Поле|Тип|Описание|
|---|---|---|
|ID|Число|Первичный ключ записи|
|resume_id|Число|Идентификатор резюме (один человек может иметь несколько резюме)|
|first_name|Строка|Имя|
|last_name|Строка|Фамилия|
|middle_name|Строка|Отчество (может быть пустым)|
|birth_day|Дата|Дата рождения (может быть пустым)|
|gender|Строка|Пол ("Мужчина" или "Женщина")|
|file_as|Строка|Форматированное имя (обычно "Фамилия Имя")|
|UID|Строка|Уникальный идентификатор человека (MD5-хеш)|

### 6.2 Основные задачи профилирования

1. **Анализ полноты данных**:
    
    - Выявление пропущенных значений во всех полях
    - Визуализация полноты данных
2. **Анализ уникальности**:
    
    - Определение степени уникальности значений полей
    - Визуализация уникальности
3. **Анализ персональных данных**:
    
    - Анализ распределения имен, фамилий и отчеств
    - Создание словарей наиболее частых значений
    - Визуализация распределений
4. **Анализ гендерного распределения**:
    
    - Подсчет количества мужчин и женщин
    - Визуализация распределения
5. **Анализ дат рождения**:
    
    - Проверка валидности дат
    - Выявление аномальных дат (слишком старые или будущие)
    - Анализ распределения по годам
    - Расчет возрастов и их распределения
6. **Анализ UID**:
    
    - Проверка консистентности UID с полями, по которым он должен формироваться
    - Анализ случаев несоответствия
    - Проверка алгоритма генерации
7. **Анализ резюме**:
    
    - Подсчет количества резюме на одного человека
    - Визуализация распределения
    - Анализ случаев с большим количеством резюме
8. **Анализ поля file_as**:
    
    - Проверка соответствия file_as формату "Фамилия Имя"
    - Выявление несоответствий
9. **Анализ дубликатов**:
    
    - Поиск дубликатов по персональным данным
    - Поиск дубликатов по UID
    - Поиск дубликатов по resume_id
10. **Анализ вариабельности групп**:
    
    - Анализ изменений в полях в пределах одного resume_id
    - Выявление групп с высокой вариабельностью
    - Визуализация распределения вариабельности

## 7. Ожидаемые артефакты

### 7.1 Общие артефакты

1. **JSON-отчеты**:
    
    - completeness.json
    - uniqueness.json
    - *_stats.json для каждого поля/анализа
2. **CSV-словари**:
    
    - dictionaries/first_name_dictionary.csv
    - dictionaries/last_name_dictionary.csv
    - dictionaries/middle_name_dictionary.csv
3. **Визуализации**:
    
    - *_distribution.png для распределений
    - birth_year_distribution.png
    - resume_count_distribution.png
    - group_variation_distribution.png

### 7.2 Артефакты по этапам

|Этап|JSON-артефакты|CSV-артефакты|PNG-артефакты|
|---|---|---|---|
|Полнота|completeness.json|-|completeness_*.png|
|Уникальность|uniqueness.json|-|-|
|Имена|name_fields_analysis.json|first_name_dictionary.csv<br>last_name_dictionary.csv<br>middle_name_dictionary.csv|first_name_distribution__.png<br>last_name_distribution__.png<br>middle_name_distribution_*.png|
|Пол|gender_distribution.json|-|gender_distribution_*.png|
|Даты рождения|birth_date_stats.json|birth_date_anomalies_*.csv|birth_year_distribution_*.png|
|Резюме|resume_counts.json<br>name_based_resume_counts.json|-|resume_count_distribution__.png<br>name_based_resume_distribution__.png|
|UID|uid_consistency.json<br>uid_generation_check.json|-|-|
|file_as|file_as_analysis.json|-|-|
|Дубликаты|person_duplicates.json<br>uid_duplicates.json<br>resume_id_duplicates.json|-|-|
|Вариабельность|resume_group_variation_*.json|group_variation_details_*.csv|group_variation_distribution_*.png|

## 8. Заключение

Данное решение предполагает комплексный рефакторинг существующей задачи профилирования таблицы IDENTIFICATION с переходом на архитектуру операций. Ключевые компоненты решения:

1. **Создание и доработка операций** в pamola_core.profiling.analyzers, которые соответствуют архитектуре op_base/op_registry/op_result.
    
2. **Реализация новой задачи** на основе BaseTask, которая настраивает и выполняет операции.
    
3. **Создание новых модулей** для анализа идентичности и дубликатов.
    
4. **Стандартизация артефактов** и отчетов в соответствии с требованиями архитектуры.
    

Этот подход позволит:

- Уменьшить дублирование кода
- Стандартизировать способы обработки данных
- Упростить поддержку и расширение функциональности
- Обеспечить единый подход к формированию отчетов
- Сделать код более модульным и тестируемым