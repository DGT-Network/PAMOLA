# План рефакторинга Categorical Generalization Operation v3.0 (Финальный)

## Изменения в версии 3.0
- ✅ Исправлена загрузка иерархий через HierarchyDictionary напрямую
- ✅ Исправлены сигнатуры функций metric_utils (добавлен timing_info)
- ✅ Убраны несуществующие функции (get_process_summary)
- ✅ Добавлены проверки реальных API модулей
- ✅ Уточнены методы HierarchyDictionary
- ✅ Добавлена секция критических исправлений

## ⚠️ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ

### 1. Загрузка иерархий
```python
# ❌ НЕПРАВИЛЬНО - функции load_hierarchy_dictionary НЕТ в category_utils.py
hierarchy = load_hierarchy_dictionary(path)

# ✅ ПРАВИЛЬНО - использовать HierarchyDictionary напрямую
from pamola_core.anonymization.commons.hierarchy_dictionary import HierarchyDictionary
hierarchy = HierarchyDictionary()
hierarchy.load_from_file(path, format_type)
```

### 2. Вызов collect_operation_metrics
```python
# ❌ НЕПРАВИЛЬНО - отсутствует параметр timing_info
metrics = collect_operation_metrics(
    operation_type="categorical_generalization",
    original_data=original,
    processed_data=anonymized,
    operation_params=operation_params
)

# ✅ ПРАВИЛЬНО - с timing_info
metrics = collect_operation_metrics(
    operation_type="categorical_generalization",
    original_data=original,
    processed_data=anonymized,
    operation_params=operation_params,
    timing_info={"start_time": self.start_time, "end_time": self.end_time}
)
```

### 3. Privacy summary
```python
# ❌ НЕПРАВИЛЬНО - функции get_process_summary НЕТ
metrics['privacy_summary'] = get_process_summary(privacy_metrics)

# ✅ ПРАВИЛЬНО - использовать get_process_summary_message из metric_utils
metrics['privacy_summary'] = get_process_summary_message(privacy_metrics)
```

## 1. Карта соответствия функционала и модулей

### 1.1. Базовый жизненный цикл операции

| Функционал | Текущая реализация | Новый модуль | Использует из фреймворка |
|------------|-------------------|--------------|-------------------------|
| Инициализация операции | `__init__` в categorical.py | `categorical_op.py` | `base_anonymization_op.AnonymizationOperation` |
| Метод execute | Наследуется из базового класса | `categorical_op.py` (не переопределяется) | `base_anonymization_op.execute()` |
| Обработка батчей | `process_batch()` | `categorical_op.py` → делегирует в `categorical_strategies.py` | `op_data_processing.get_dataframe_chunks()` |
| **Обработка ошибок батчей** | `continue_on_error`, `error_batch_handling` | `categorical_op._handle_batch_errors()` | `op_data_writer.write_json()` |
| **Адаптивный размер батча** | `adaptive_batch_size` | `categorical_op._adjust_batch_size()` | `op_data_processing.get_memory_usage()` |
| **Переключение на Dask** | Отсутствует | `categorical_op._check_large_data_mode()` | `dask.dataframe` (при >1M записей) |
| Сохранение конфигурации | `save_config()` из базового | Наследуется | `op_base.save_config()` |
| Управление прогрессом | Через `progress_tracker` | `categorical_op.py` | `ProgressTracker` из `pamola_core.utils.progress` |
| **Сброс состояния** | Отсутствует | `categorical_op.reset_state()` | Очистка локальных кэшей |
| **Логирование с trace ID** | Стандартный logger | `categorical_op._get_logger()` | `pamola_core.utils.logging.get_logger()` |

### 1.2. Стратегии категоризации и обработка NULL

| Стратегия | Текущий метод | Новый модуль | Вспомогательные функции |
|-----------|---------------|--------------|------------------------|
| Hierarchy | `_apply_hierarchy_vectorized()` | `categorical_strategies.apply_hierarchy()` | • `HierarchyDictionary.load_from_file()`<br>• `text_processing_utils.normalize_text()`<br>• `text_processing_utils.find_closest_category()` |
| Merge Low Freq | `_apply_merge_low_freq()` | `categorical_strategies.apply_merge_low_freq()` | • `category_utils.identify_rare_categories()`<br>• `category_utils.group_rare_categories()` |
| Frequency Based | `_apply_frequency_based()` | `categorical_strategies.apply_frequency_based()` | • `category_utils.analyze_category_distribution()`<br>• `category_mapping.CategoryMappingEngine` |
| **NULL обработка** | В каждом методе | `categorical_strategies.apply_null_and_unknown_strategy()` | • `data_utils.process_nulls()` |
| **Шаблоны редких** | Hardcoded "OTHER" | `categorical_strategies.format_rare_value()` | Использует `rare_value_template` |
| **Dask версии** | Отсутствуют | `categorical_strategies.apply_*_dask()` | `dask.dataframe` API |

### 1.3. Работа со словарями и иерархиями

| Функционал | Текущая реализация | Новый модуль | Фреймворк |
|------------|-------------------|--------------|-----------|
| Загрузка словарей | `_load_hierarchy_via_reader()` | **Прямое использование HierarchyDictionary** | `op_data_reader.DataReader` |
| Управление иерархиями | Через `HierarchyDictionary` | `hierarchy_dictionary.py` (+ LRU кэш) | - |
| Валидация словарей | В `_load_hierarchy_via_reader()` | `hierarchy_dictionary.validate_structure()` | - |
| **Thread-safe кэширование** | Отсутствует | `hierarchy_dictionary.py` с `@lru_cache` + RLock | `functools.lru_cache` |
| **Маппинг категорий** | `CategoryMappingEngine` в categorical.py | `category_mapping.CategoryMappingEngine` с `cachetools.LRUCache` | `threading.RLock` |

### 1.4. Сбор и сохранение метрик (ОБНОВЛЕНО)

| Тип метрик | Текущий метод | Новый модуль/функция | Используемые утилиты |
|------------|---------------|---------------------|---------------------|
| **Базовые метрики** | `_collect_specific_metrics()` | `categorical_op._collect_specific_metrics()` | `metric_utils.collect_operation_metrics(..., timing_info)` |
| Эффективность анонимизации | В `_collect_specific_metrics()` | Включено в collect_operation_metrics | `metric_utils.calculate_anonymization_effectiveness()` |
| Производительность | В базовом классе | Наследуется | `metric_utils.calculate_process_performance()` |
| **Категориальная потеря информации** | В `_collect_specific_metrics()` | Передается через operation_params | `metric_utils.calculate_categorical_information_loss()` |
| **Высота генерализации** | Отсутствует | При strategy="hierarchy" | `metric_utils.calculate_generalization_height()` |
| **Privacy метрики** | В `_collect_specific_metrics()` | `categorical_op._collect_specific_metrics()` | • `privacy_metric_utils.calculate_batch_metrics()`<br>• `privacy_metric_utils.check_anonymization_thresholds()`<br>• `metric_utils.get_process_summary_message()` |
| **Min group size** | Используется базовым классом | Не переопределять | `privacy_metric_utils.calculate_min_group_size()` |
| **Vulnerable ratio** | Используется базовым классом | Не переопределять | `privacy_metric_utils.calculate_vulnerable_records_ratio()` |
| Сохранение метрик | Через `DataWriter` | `categorical_op.execute()` → `DataWriter` | • `op_data_writer.DataWriter.write_metrics()`<br>• `metric_utils.save_process_metrics()` (fallback) |

### 1.5. Визуализация

| Тип визуализации | Текущий метод | Новый модуль | Утилиты | Порядок |
|------------------|---------------|--------------|---------|---------|
| Сравнение распределений | `_generate_visualizations()` | `categorical_op._generate_visualizations()` | `visualization_utils.create_category_distribution_comparison()` | 1 |
| Иерархия (sunburst) | `_generate_visualizations()` | То же | `visualization_utils.create_hierarchy_sunburst()` | 2 |
| **Метрики (heatmap)** | Отсутствует | То же | `visualization_utils.create_metric_visualization()` | 3 |
| Общее сравнение | `_generate_visualizations()` | То же | `visualization_utils.create_comparison_visualization()` | 5 |
| Регистрация артефактов | В `_generate_visualizations()` | То же | `visualization_utils.register_visualization_artifact()` | - |

### 1.6. Валидация

| Тип валидации | Текущая реализация | Новый модуль | Валидаторы |
|---------------|-------------------|--------------|------------|
| Конфигурация операции | `_validate_configuration()` | `categorical_config.validate_strategy_params()` | `validation.strategy_validators.validate_strategy_parameters()` |
| Поля данных | В `process_batch()` | `categorical_strategies.py` (в каждой функции) | `validation.CategoricalFieldValidator` |
| Параметры стратегий | `_validate_configuration()` | `categorical_config.py` | `validation.create_field_validator()` |
| Схема данных | JSON Schema в классе | `categorical_config.schema` | `op_config.OperationConfig` валидация |
| **NULL стратегия** | Отсутствует | `categorical_config.validate_null_strategy()` | `validation.NullStrategyValidator` |
| **Шаблоны значений** | Отсутствует | `categorical_config.validate_templates()` | Regex валидация для `{n}` placeholder |

### 1.7. Обработка текста

| Функционал | Текущее использование | Новый модуль | Утилиты |
|------------|---------------------|--------------|---------|
| Нормализация текста | В `_apply_hierarchy_vectorized()` | `categorical_strategies.apply_hierarchy()` | `text_processing_utils.normalize_text()` |
| Fuzzy matching | В `_apply_hierarchy_vectorized()` | То же | `text_processing_utils.find_closest_category()` |
| Очистка имен категорий | Не используется явно | `category_utils.build_category_mapping()` | `text_processing_utils.clean_category_name()` |
| Разбиение составных значений | Не используется | Опционально в стратегиях | `text_processing_utils.split_composite_value()` |

### 1.8. Логирование и управление ошибками

| Тип | Текущая реализация | Новый модуль | Фреймворк |
|-----|-------------------|--------------|-----------|
| **Инициализация логгера** | `custom_logging.get_logger()` | `get_logger(__name__, op_id=self.operation_id)` | `pamola_core.utils.logging.get_logger()` |
| **Формат логов** | Не стандартизирован | Унифицированный формат | `[timestamp] [op_id] [batch_id] [level] message` |
| Логирование ошибок | `self.logger.error/warning()` | В каждом модуле | Стандартный logger |
| **Trace ID** | Отсутствует | `categorical_op._generate_trace_id()` | UUID для операции |
| Логирование прогресса | Через `progress_tracker` | `categorical_op.process_batch()` | `ProgressTracker.update()` |
| **Batch errors** | Не сохраняются | `categorical_op._save_batch_errors()` | `DataWriter.write_json()` |
| **Error indices** | Не отслеживаются | До 100 индексов в `context['error_indices']` | Для forensic analysis |

### 1.9. Чтение источников данных

| Функционал | Текущее использование | Новый модуль | Фреймворк |
|------------|---------------------|--------------|-----------|
| Чтение основных данных | Через `DataSource` в execute() | Наследуется из базового | `op_data_source.DataSource.get_dataframe()` |
| Чтение словарей | `DataReader` в `_load_hierarchy_via_reader()` | **Прямое использование HierarchyDictionary** | `HierarchyDictionary.load_from_file()` |
| Оптимизация памяти | В `process_batch()` | `categorical_op.process_batch()` | `op_data_processing.optimize_dataframe_dtypes()` |
| **Проверка больших данных** | Отсутствует | `categorical_op._check_large_data_mode()` | Проверка len(df) > max_rows_in_memory |

## 2. Структура модулей после рефакторинга

```
pamola_core/anonymization/generalization/
├── __init__.py                    # Re-export основных классов
├── categorical_op.py              # Основной класс операции (фасад)
├── categorical_config.py          # Конфигурация и валидация
├── categorical_strategies.py      # Реализация стратегий
└── categorical_legacy.py          # Обратная совместимость (deprecated)

pamola_core/anonymization/commons/
├── category_mapping.py            # НОВЫЙ: Универсальный движок маппинга
├── category_utils.py              # БЕЗ load_hierarchy_dictionary!
└── hierarchy_dictionary.py        # ДОРАБОТАН: +LRU кэш + thread-safety
```

## 2.5. Критические функциональные требования (НОВОЕ)

### Таблица функциональных упущений

| № | Функциональная возможность | Где реализовать | Ссылка на SRS |
|---|---------------------------|-----------------|---------------|
| 1 | **`null_strategy`** (`PRESERVE/EXCLUDE/ANONYMIZE/ERROR`) | `categorical_config.schema` + `categorical_strategies.apply_null_and_unknown_strategy()` | REQ-CATGEN-004 |
| 2 | **`continue_on_error`** и **`error_batch_handling`** | `categorical_config.schema` + `categorical_op._handle_batch_errors()` | REQ-CATGEN-011 |
| 3 | **`rare_value_template`** (напр. `OTHER_{n}`) | `categorical_config.schema` + `categorical_strategies.format_rare_value()` | SRS §4.6 |
| 4 | **`unknown_value`** placeholder | `categorical_config.schema` + все стратегии | REQ-CATGEN-003 |
| 5 | **Переключение на Dask** (`engine="dask"`, `max_rows_in_memory`) | `categorical_config.schema` + `categorical_op._check_large_data_mode()` | REQ-CATGEN-007 |
| 6 | **State reset** после execute | `categorical_op.reset_state()` | REQ-ANON-015 |
| 7 | **Privacy check когда нет quasi_identifiers** | `categorical_op._collect_specific_metrics()` - всегда включать секцию | SRS §5.6 |
| 8 | **`calculate_generalization_height`** в метриках | `categorical_op._collect_specific_metrics()` при strategy="hierarchy" | SRS §5.6 |
| 9 | **`dictionary_coverage`** в метриках | `categorical_op._collect_specific_metrics()` через hierarchy.get_coverage() | REQ-CATGEN-005 |
| 10 | **Сохранение dtype** | `categorical_strategies` - проверка и восстановление dtype | Базовый фреймворк |
| 11 | **Детерминизм** (seed) | `categorical_config.schema` + `random_seed` параметр | Воспроизводимость |
| 12 | **`similarity_threshold`** для fuzzy | Уже есть в config, но проверить использование | REQ-CATGEN-002 |
| 13 | **Локалезависимое сравнение** | `text_normalization` параметр + `normalize_text()` | Многоязычность |
| 14 | **Метрики редких категорий** | `categorical_op._collect_specific_metrics()` - rare_categories_count | REQ-CATGEN-005 |
| 15 | **Применение condition mask** | `categorical_op.process_batch()` - передать mask в context | Базовый op |
| 16 | **Экспорт category_mapping** | `categorical_op._save_mapping_artifact()` через DataWriter | REQ-CATGEN-006 |
| 17 | **Cache key с hash словаря** | `categorical_op._get_cache_parameters()` + file hash | Кэширование |
| 18 | **Dataset-level information_loss** | `categorical_op._collect_specific_metrics()` - агрегированный IL | KPI метрики |

## 3. Детальное описание новых/измененных модулей

### 3.1. categorical_op.py (упрощенный фасад)

```python
class CategoricalGeneralizationOperation(AnonymizationOperation):
    """
    Фасад для категориальной генерализации.
    
    :requirement: REQ-CATGEN-001 - Основная операция
    :requirement: REQ-CATGEN-002 - Интерфейс конструктора
    """
    
    __version__ = "3.0.0"
    
    def __init__(self, **kwargs):
        # 1. Создать конфигурацию
        self.config = CategoricalGeneralizationConfig(**kwargs)
        
        # 2. Инициализировать базовый класс
        super().__init__(
            field_name=self.config.field_name,
            mode=self.config.mode,
            continue_on_error=self.config.continue_on_error,
            error_batch_handling=self.config.error_batch_handling,
            adaptive_batch_size=self.config.adaptive_batch_size,
            # ... остальные параметры из config
        )
        
        # 3. Сохранить параметры стратегии
        self.strategy = self.config.strategy
        self.strategy_params = self.config.get_strategy_params()
        
        # 4. Инициализировать trace ID и logger
        self.operation_id = self._generate_trace_id()
        self.logger = get_logger(__name__, op_id=self.operation_id)
        
        # 5. Thread-safe состояние
        self._lock = threading.RLock()
        self._batch_errors = []
        self._hierarchy_cache = {}
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка батча данных.
        
        :requirement: REQ-CATGEN-003 - Поддержка стратегий
        :requirement: REQ-CATGEN-007 - Производительность
        """
        batch_id = f"batch_{self.operation_id}_{len(self._batch_errors)}"
        
        # 1. Проверка на большие данные
        if self._check_large_data_mode(batch):
            self.logger.info(f"[{batch_id}] Switching to Dask mode")
            return self._process_batch_dask(batch)
        
        # 2. Подготовка контекста
        context = self._prepare_context(batch, batch_id)
        
        try:
            # 3. Делегирование в стратегию
            if self.strategy == "hierarchy":
                # ВАЖНО: загрузка иерархии через HierarchyDictionary
                if 'hierarchy' not in context:
                    hierarchy = HierarchyDictionary()
                    hierarchy.load_from_file(
                        self.strategy_params['external_dictionary_path'],
                        self.strategy_params['dictionary_format']
                    )
                    context['hierarchy'] = hierarchy
                
                result = apply_hierarchy(
                    batch[self.field_name], 
                    self.strategy_params, 
                    context,
                    logger=self.logger
                )
            elif self.strategy == "merge_low_freq":
                result = apply_merge_low_freq(
                    batch[self.field_name], 
                    self.strategy_params, 
                    context,
                    logger=self.logger
                )
            else:  # frequency_based
                result = apply_frequency_based(
                    batch[self.field_name], 
                    self.strategy_params, 
                    context,
                    logger=self.logger
                )
            
            # 4. Применение NULL и unknown стратегии
            result = apply_null_and_unknown_strategy(
                result, 
                self.config.null_strategy, 
                self.config.unknown_value,
                self.config.rare_value_template
            )
            
            # 5. Обновление батча
            return self._update_batch(batch, result, context)
            
        except Exception as e:
            if self.continue_on_error:
                self._handle_batch_error(batch, e, context)
                return batch  # Возвращаем оригинал
            else:
                raise
    
    def _collect_specific_metrics(self, original: pd.Series, anonymized: pd.Series) -> Dict[str, Any]:
        """
        Сбор метрик операции с правильными сигнатурами.
        
        :requirement: REQ-CATGEN-005 - Метрики
        :requirement: REQ-CATGEN-010 - Privacy проверки
        """
        # Подготовка timing_info
        timing_info = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "batch_count": getattr(self, '_batch_count', 1)
        }
        
        # Подготовка параметров для метрик
        operation_params = {
            "strategy": self.strategy,
            **self.strategy_params
        }
        
        # Добавляем mapping и hierarchy info если есть
        if hasattr(self, '_category_mapping'):
            operation_params['category_mapping'] = self._category_mapping
        if hasattr(self, '_hierarchy_info'):
            operation_params['hierarchy_info'] = self._hierarchy_info
        
        # ВАЖНО: вызов с правильной сигнатурой
        metrics = collect_operation_metrics(
            operation_type="categorical_generalization",
            original_data=original,
            processed_data=anonymized,
            operation_params=operation_params,
            timing_info=timing_info  # <-- обязательный параметр!
        )
        
        # Дополнительные метрики для категориальных данных
        if self.strategy in ["hierarchy", "merge_low_freq", "frequency_based"]:
            # Эти метрики уже включены в collect_operation_metrics,
            # но можем добавить дополнительные
            pass
        
        # Privacy метрики
        if self.quasi_identifiers:
            temp_df = pd.DataFrame({
                self.field_name: original,
                f"{self.field_name}_anonymized": anonymized
            })
            
            privacy_metrics = calculate_batch_metrics(
                original_batch=temp_df[[self.field_name]],
                anonymized_batch=temp_df[[f"{self.field_name}_anonymized"]],
                field_name=self.field_name,
                quasi_identifiers=self.quasi_identifiers
            )
            metrics['privacy_metrics'] = privacy_metrics
            
            # Проверка порогов
            privacy_thresholds = {
                "min_k": self.config.min_acceptable_k,
                "max_suppression": 0.2,
                "min_coverage": 0.95,
                "max_vulnerable_ratio": self.config.max_acceptable_disclosure_risk
            }
            
            threshold_results = check_anonymization_thresholds(
                privacy_metrics,
                privacy_thresholds
            )
            metrics['privacy_threshold_checks'] = threshold_results
            
            # ВАЖНО: используем get_process_summary_message вместо несуществующего get_process_summary
            metrics['privacy_summary'] = get_process_summary_message(metrics)
        else:
            metrics['privacy_metrics'] = {'status': 'SKIPPED', 'reason': 'no_quasi_identifiers'}
            metrics['privacy_summary'] = "Privacy checks skipped: no quasi-identifiers specified"
        
        # Добавить метрики ошибок
        metrics['error_handling'] = {
            'batch_errors': len(self._batch_errors),
            'continue_on_error': self.continue_on_error,
            'error_batch_handling': self.error_batch_handling
        }
        
        # Семантическое разнообразие
        if anonymized.nunique() <= 1000:
            metrics['semantic_diversity'] = calculate_semantic_diversity_safe(
                list(anonymized.unique())
            )
        
        return metrics
```

### 3.2. categorical_strategies.py (ОБНОВЛЕНО)

```python
def apply_hierarchy(series: pd.Series, config: Dict[str, Any], 
                   context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Применить иерархическую генерализацию.
    
    ВАЖНО: Ожидает, что hierarchy уже загружена в context['hierarchy']
    """
    if logger:
        logger.debug(f"[{context.get('batch_id')}] Applying hierarchy strategy")
    
    # 1. Получить иерархию из контекста (уже загружена в categorical_op)
    hierarchy = context.get('hierarchy')
    if not hierarchy:
        raise ValueError("Hierarchy not loaded in context")
    
    # 2. Создать маппинг движок
    engine = CategoryMappingEngine(
        unknown_value=config['unknown_value'],
        cache_size=10000
    )
    
    # 3. Построить маппинг
    for value in series.dropna().unique():
        str_value = str(value)
        normalized = normalize_text(str_value, config.get('text_normalization', 'basic'))
        
        # Прямой поиск через методы HierarchyDictionary
        category = hierarchy.get_hierarchy(normalized, config['hierarchy_level'])
        
        if category:
            engine.add_mapping(str_value, category)
        elif config.get('fuzzy_matching', False):
            # Fuzzy matching
            all_values = hierarchy.get_all_values_at_level(0)
            closest = find_closest_category(
                normalized, 
                list(all_values), 
                config.get('similarity_threshold', 0.85)
            )
            if closest:
                category = hierarchy.get_hierarchy(closest, config['hierarchy_level'])
                if category:
                    engine.add_mapping(str_value, category)
                    context.setdefault('fuzzy_matches', 0)
                    context['fuzzy_matches'] += 1
        
    # 4. Применить маппинг
    result = engine.apply_to_series(series)
    
    # 5. Сохранить маппинг для метрик
    context['category_mapping'] = engine.get_mapping_dict()
    context['hierarchy_info'] = config.get('hierarchy_info', {})
    
    return result
```

## 4. Матрица ответственности модулей

| Функционал | categorical_op | categorical_config | categorical_strategies | category_mapping | category_utils | hierarchy_dictionary |
|------------|:--------------:|:------------------:|:---------------------:|:----------------:|:--------------:|:-------------------:|
| Жизненный цикл операции | ✓ | | | | | |
| Валидация конфигурации | | ✓ | | | | |
| Реализация стратегий | | | ✓ | | | |
| Thread-safe маппинг | | | | ✓ | | |
| Загрузка иерархий | ✓ | | | | | ✓ |
| Анализ категорий | | | | | ✓ | |
| Сбор метрик | ✓ | | | | | |
| Визуализация | ✓ | | | | | |
| Логирование с trace ID | ✓ | ✓ | ✓ | | | |
| Обработка ошибок | ✓ | | ✓ | | | |
| NULL/unknown стратегии | | ✓ | ✓ | | | |

## 5. Необходимые импорты для categorical_op.py (ОБНОВЛЕНО)

```python
# Базовые импорты
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Базовый класс
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Фреймворк
from pamola_core.utils.logging import get_logger
from pamola_core.utils.ops.op_data_processing import (
    optimize_dataframe_dtypes,
    get_memory_usage,
    force_garbage_collection
)
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker

# Commons - метрики (С ПРАВИЛЬНЫМИ СИГНАТУРАМИ!)
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,  # с параметром timing_info!
    calculate_categorical_information_loss,
    calculate_generalization_height,
    save_process_metrics,
    get_process_summary_message  # НЕ get_process_summary!
)

# Commons - privacy метрики
from pamola_core.anonymization.commons.privacy_metric_utils import (
    calculate_batch_metrics,
    check_anonymization_thresholds
)

# Commons - визуализация
from pamola_core.anonymization.commons.visualization_utils import (
    create_category_distribution_comparison,
    create_comparison_visualization,
    create_hierarchy_sunburst,
    create_metric_visualization,
    register_visualization_artifact
)

# Commons - категории и иерархии
from pamola_core.anonymization.commons.category_utils import (
    calculate_semantic_diversity_safe
    # НЕТ load_hierarchy_dictionary!
)
from pamola_core.anonymization.commons.hierarchy_dictionary import HierarchyDictionary

# Локальные модули
from .categorical_config import CategoricalGeneralizationConfig
from .categorical_strategies import (
    apply_hierarchy,
    apply_merge_low_freq,
    apply_frequency_based,
    apply_null_and_unknown_strategy,
    apply_hierarchy_dask,
    apply_merge_low_freq_dask,
    apply_frequency_based_dask
)

# Константы
MAX_ERROR_INDICES_TO_STORE = 100
```

## 6. Контрольный чек-лист перед реализацией (ОБНОВЛЕНО)

### Функциональные требования
- [ ] ✅ `null_strategy` обрабатывается во всех стратегиях через `apply_null_and_unknown_strategy()`
- [ ] ✅ `continue_on_error` и `error_batch_handling` в схеме и обработчике ошибок
- [ ] ✅ `rare_value_template` с проверкой `{n}` и функцией `format_rare_value()`
- [ ] ✅ `unknown_value` используется во всех стратегиях
- [ ] ✅ Переключение на Dask при `len(df) > max_rows_in_memory`
- [ ] ✅ `reset_state()` очищает все кэши и состояние
- [ ] ✅ Privacy метрики ВСЕГДА включены (со статусом SKIPPED если нет QI)
- [ ] ✅ `calculate_generalization_height()` вызывается для hierarchy
- [ ] ✅ `dictionary_coverage` рассчитывается через `hierarchy.get_coverage()`
- [ ] ✅ Сохранение и восстановление dtype во всех стратегиях
- [ ] ✅ `random_seed` для детерминизма в config и стратегиях
- [ ] ✅ `similarity_threshold` используется в fuzzy matching
- [ ] ✅ `text_normalization` и `normalize_unicode` в параметрах нормализации
- [ ] ✅ Метрики редких категорий (before/after/reduction_ratio)
- [ ] ✅ Условная маска передается в context и применяется в стратегиях
- [ ] ✅ `category_mapping` сохраняется как артефакт через `_save_mapping_artifact()`
- [ ] ✅ Hash словаря включается в cache key если `cache_include_dict_hash=True`
- [ ] ✅ Dataset-level IL рассчитывается в дополнение к field-level

### Технические требования
- [ ] JSON-schema валидирует `rare_value_template` с regex `.*\{n\}.*`
- [ ] Thread-safe реализация с `threading.RLock()` в CategoryMappingEngine
- [ ] LRU кэширование через `cachetools.LRUCache`
- [ ] Формат логов: `[timestamp] [op_id] [batch_id] [level] message`
- [ ] Все публичные методы имеют `:requirement:` теги в docstring

### Метрики и визуализации
- [ ] ✅ Используется `collect_operation_metrics()` с параметром `timing_info`
- [ ] ✅ НЕ вызывается несуществующий `get_process_summary()`
- [ ] ✅ Используется `get_process_summary_message()` для создания summary
- [ ] Визуализации создаются в правильном порядке (1-5)
- [ ] Privacy summary сохраняется как HTML артефакт

### Работа с иерархиями
- [ ] ✅ HierarchyDictionary загружается напрямую через `load_from_file()`
- [ ] ✅ НЕТ вызовов несуществующей `load_hierarchy_dictionary()` из category_utils
- [ ] ✅ Используются только публичные методы HierarchyDictionary
- [ ] Иерархия кэшируется в context для повторного использования

### Совместимость и версионирование
- [ ] `__version__ = "3.0.0"` в categorical_op.py
- [ ] Legacy модуль выдает `DeprecationWarning`
- [ ] Нет циклических импортов между модулями
- [ ] CHANGELOG.md обновлен с breaking changes

### Валидация
- [ ] ✅ Используется новая система валидации из validation/
- [ ] ✅ НЕ используются deprecated функции из validation_utils
- [ ] ✅ Все валидаторы возвращают ValidationResult
- [ ] ✅ Ошибки валидации правильно обрабатываются

## 7. Критические изменения от v2

1. **Загрузка иерархий** - теперь только через HierarchyDictionary напрямую
2. **Метрики** - обязательный параметр timing_info в collect_operation_metrics
3. **Privacy summary** - использовать get_process_summary_message вместо get_process_summary
4. **Методы HierarchyDictionary** - только публичные методы (load_from_file, get_hierarchy, etc.)
5. **category_utils** - НЕ содержит load_hierarchy_dictionary

## 8. Преимущества финальной версии

1. **Полное соответствие API** - все импорты и сигнатуры проверены против реальных модулей
2. **Thread-safety** - использование RLock и thread-safe кэшей
3. **Производительность** - LRU кэширование на всех уровнях
4. **Отказоустойчивость** - обработка ошибок с сохранением контекста
5. **Масштабируемость** - подготовка к Dask для больших данных
6. **Наблюдаемость** - trace ID и структурированное логирование
7. **Совместимость** - legacy wrapper для плавной миграции
8. **Тестируемость** - четкое разделение ответственности
9. **Расширяемость** - легко добавить новые стратегии
10. **Корректность** - все функции и методы существуют и используются правильно