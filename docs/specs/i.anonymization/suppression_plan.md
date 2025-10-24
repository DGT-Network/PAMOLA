SUPPRESSION REALIZATION PLAN

## Структура модулей

```
pamola_core/anonymization/suppression/
├── __init__.py
├── attribute_suppression.py    # Удаление колонок
├── record_suppression.py       # Удаление записей
└── cell_suppression.py         # Замена значений ячеек
```

## План реализации (MVP)

### 1. AttributeSuppressionOperation (attribute_suppression.py)
**Минимальный функционал:**
- Удаление одной или нескольких колонок
- Сохранение метаданных об удаленных колонках
- Базовые метрики (количество удаленных колонок)

**Интеграция:**
- Наследование от `BaseAnonymizationOperation`
- Использование `check_multiple_fields_exist` из commons.validation
- Использование `DataWriter` для сохранения результатов
- Простая визуализация через commons.viz_utils

### 2. RecordSuppressionOperation (record_suppression.py)
**Минимальный функционал:**
- Удаление записей по условиям: null, value, range
- Опциональное сохранение удаленных записей
- Базовые метрики (количество/процент удаленных записей)

**Интеграция:**
- Наследование от `BaseAnonymizationOperation`
- Использование `check_field_exists` из commons.validation
- Использование `DataWriter` для сохранения удаленных записей
- Использование `calculate_suppression_metrics` из commons.metric_utils

### 3. CellSuppressionOperation (cell_suppression.py)
**Минимальный функционал:**
- Замена значений на: null, mean, mode, constant
- Условная замена (по значениям или редкости)
- Базовые метрики (количество замененных ячеек)

**Интеграция:**
- Наследование от `BaseAnonymizationOperation`
- Использование `validate_numeric_field` для проверки типов
- Использование `collect_operation_metrics` из commons.metric_utils
- Обработка ошибок через стандартные исключения (FieldNotFoundError, FieldTypeError)

## Ключевые принципы MVP:

1. **Простота**: Каждая операция делает одну вещь хорошо
2. **Переиспользование**: Максимальное использование commons-утилит
3. **Единообразие**: Следование паттернам базового класса
4. **Фокус на базовом функционале**: Откладываем сложные features (multi-field conditions, outlier detection) на будущее

## Приоритет реализации:

1. **AttributeSuppressionOperation** - самая простая, хороший старт
2. **RecordSuppressionOperation** - средняя сложность, важная для privacy
3. **CellSuppressionOperation** - самая сложная, но гибкая

Каждая операция будет:
- Использовать стандартный `process_batch()` метод
- Собирать метрики через `_collect_specific_metrics()`
- Генерировать простые визуализации
- Обрабатывать ошибки через стандартные механизмы

Это позволит быстро создать рабочий MVP с возможностью последующего расширения функционала.