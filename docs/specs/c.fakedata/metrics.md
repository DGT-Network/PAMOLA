# Анализ необходимости отдельного модуля для расчета метрик при генерации фейковых данных

## Текущее состояние

сейчас система сбора метрик частично реализована внутри классов операций в `operations.py`, в частности:

1. Метод `_collect_metrics()` в классе `FieldOperation` собирает базовые метрики:
    
    - Общее количество записей
    - Количество ненулевых записей
    - Время выполнения
    - Информация о выходном поле (при режиме ENRICH)
2. Метод `_collect_metrics()` в классе `GeneratorOperation` расширяет базовые метрики:
    
    - Добавляет информацию о типе генератора
    - Добавляет информацию о механизме согласованности
    - Собирает статистику по маппингам (при использовании mapping)
3. Сохранение метрик происходит через метод `_save_metrics()`, который записывает их в JSON-файл.
    

Однако эта реализация имеет ограничения:

- Фокусируется на технических метриках, а не на качестве данных
- Не предоставляет специализированных метрик для разных типов данных
- Не позволяет сравнивать исходные и синтетические данные
- Не содержит функционала для визуализации метрик

## Обоснование необходимости отдельного модуля

Отдельный модуль для метрик необходим по следующим причинам:

1. **Разделение ответственности**: Модуль операций должен отвечать за процесс генерации, а не за анализ качества.
    
2. Визуализация
3. Функции сбора метрик

```python
{
    "original_data": {
        "total_records": int,
        "unique_values": int,
        "value_distribution": Dict[str, float],
        "length_stats": {
            "min": int,
            "max": int,
            "mean": float,
            "median": float
        }
    },
    "generated_data": {
        "total_records": int,
        "unique_values": int,
        "value_distribution": Dict[str, float],
        "length_stats": {
            "min": int,
            "max": int,
            "mean": float,
            "median": float
        }
    },
    "quality_metrics": {
        "distribution_similarity_score": float,
        "uniqueness_preservation": float,
        "format_compliance": float,
        "type_specific_metrics": Dict[str, Any]
    },
    "transformation_metrics": {
        "null_values_replaced": int,
        "total_replacements": int,
        "replacement_strategy": str,
        "mapping_collisions": int,
        "reversibility_rate": float
    },
    "performance": {
        "generation_time": float,
        "records_per_second": int,
        "memory_usage_mb": float,
        "dictionary_load_time": float
    },
    "dictionary_metrics": {
        "total_dictionary_entries": int,
        "language_variants": List[str],
        "last_update": str
    }
}
```
### Структура модуля

Рекомендуемая структура:

```python
"""
Metrics collection and analysis for fake data generation.

This module provides tools for measuring the quality and statistical 
properties of generated fake data, including distribution comparison,
format validation, and anonymization quality assessment.
"""

class MetricsCollector:
    """Base class for metrics collectors."""
    
    def collect_basic_stats(self, df, column):
        """Collects basic statistical metrics for a column."""
        pass
    
    def compare_distributions(self, orig_df, fake_df, column):
        """Compares distributions in original and synthetic columns."""
        pass
    
class TypedMetricsCollector(MetricsCollector):
    """Base class for type-specific metrics collectors."""
    
    def collect_type_specific_metrics(self, orig_df, fake_df, column):
        """Collects metrics specific to a data type."""
        pass

class NameMetricsCollector(TypedMetricsCollector):
    """Metrics collector for name data."""
    
    def collect_gender_metrics(self, orig_df, fake_df, name_column, gender_column=None):
        """Collects metrics on gender distribution."""
        pass
    
    def collect_linguistic_metrics(self, orig_df, fake_df, name_column):
        """Collects linguistic metrics."""
        pass

class EmailMetricsCollector(TypedMetricsCollector):
    """Metrics collector for email data."""
    
    def collect_domain_metrics(self, orig_df, fake_df, email_column):
        """Collects metrics on domain distribution."""
        pass

# Additional collectors for other data types...

def create_metrics_collector(data_type):
    """Factory function to create appropriate metrics collector based on data type."""
    pass

def generate_metrics_report(metrics, output_format="json"):
    """Generates a metrics report in the specified format."""
    pass

def visualize_metrics(metrics, output_path=None):
    """Creates visualizations for metrics data."""
    pass
```

### Интеграция с существующим кодом

1. В классе `GeneratorOperation` метод `_collect_metrics()` нужно модифицировать для использования специализированного сборщика метрик:

```python
def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
    """Collects metrics for the generator operation."""
    # Base metrics from parent
    metrics = super()._collect_metrics(df)
    
    # Add generator-specific metrics
    metrics["generator"] = {
        "type": self.generator.__class__.__name__,
        "consistency_mechanism": self.consistency_mechanism,
    }
    
    # Use specialized metrics collector if available
    data_type = getattr(self.generator, "data_type", None)
    if data_type:
        from pamola_core.fake_data.commons.metrics import create_metrics_collector
        
        collector = create_metrics_collector(data_type)
        if collector:
            # Get original data if available (for REPLACE mode)
            orig_df = getattr(self, "_original_df", None)
            if orig_df is not None:
                type_metrics = collector.collect_type_specific_metrics(
                    orig_df, df, self.field_name
                )
                metrics["type_specific"] = type_metrics
    
    # Add mapping metrics if using mapping mechanism
    if self.consistency_mechanism == "mapping":
        field_mappings = self.mapping_store.get_all_mappings_for_field(self.field_name)
        metrics["mapping"] = {
            "total_mappings": len(field_mappings),
        }
    
    return metrics
```

2. В методе `execute()` нужно сохранить оригинальные данные для последующего сравнения:

```python
def execute(self, data_source, task_dir, reporter, **kwargs):
    # ... existing code ...
    
    # Load data
    df = self._load_data(data_source)
    
    # Store original data for metrics comparison
    self._original_df = df.copy()
    
    # ... rest of the method ...
```

## Заключение

Создание отдельного модуля `pamola_core/fake_data/commons/metrics.py` является необходимым шагом для:

1. **Обеспечения качественной оценки** генерируемых фейковых данных
2. **Специализированного анализа** для разных типов данных
3. **Соблюдения принципа разделения ответственности** в архитектуре
4. **Возможности расширения** функциональности метрик в будущем
5. **Согласованности** с изначальной спецификацией проекта

Такой модуль значительно повысит ценность пакета `fake_data`, обеспечивая не только генерацию данных, но и инструменты для оценки их качества.