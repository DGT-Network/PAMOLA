## Спецификация справочников для категориального обобщения

### Сводная таблица характеристик

| Справочник | Формат | Кол-во записей | Уровни иерархии | Поддержка алиасов | Дополнительные поля | Тип структуры |
|------------|--------|----------------|------------------|-------------------|---------------------|---------------|
| **cities_regions_sample.json** | JSON | 110 | 4 (city → region → state_province → country) | ❌ | ❌ | Простая иерархия |
| **professions_it_sample.csv** | CSV | 120 | 4 (job_title → category → department → industry) + seniority_level | ✅ (строка с разделителями) | seniority_level | Плоская таблица |
| **products_electronics_sample.json** | JSON | 120 | 4 (product → subcategory → category → department) | ✅ (массив) | brand | Иерархия с метаданными |
| **professions_simple_sample.csv** | CSV | 130 | 2 (value → category) | ❌ | ❌ | Простое отображение |
| **medical_specialties_sample.json** | JSON | 120 | 4 (position → specialty → department → category) | ✅ (массив) | ❌ | Иерархия с алиасами |
| **medical_conditions_sample.csv** | CSV | 130 | 3 (condition → category → system) | ✅ (строка с разделителями) | severity, icd_group | Расширенная таблица |
| **financial_transactions_sample.json** | JSON | 130 | 4 (transaction → subcategory → category → type) | ✅ (массив) | ❌ | Иерархия с алиасами |

### Детальная спецификация по типам структур

#### 1. Простая иерархия (JSON)
**Пример**: `cities_regions_sample.json`
```json
{
  "format_version": "1.0",
  "hierarchy_type": "geographic",
  "levels": ["city", "region", "state_province", "country"],
  "data": {
    "Toronto": {
      "region": "Greater Toronto Area",
      "state_province": "Ontario",
      "country": "Canada"
    }
  }
}
```
**Характеристики**:
- Вложенная структура с именованными уровнями
- Нет алиасов
- Минимальная структура для иерархической категоризации

#### 2. Плоская таблица с алиасами (CSV)
**Пример**: `professions_it_sample.csv`
```csv
job_title,category,department,industry,seniority_level,aliases
Software Engineer,Engineering,Technology,Information Technology,Mid-level,"Developer;Programmer;Software Developer"
```
**Характеристики**:
- Все уровни в одной строке
- Алиасы как строка с разделителями
- Дополнительные атрибуты в отдельных колонках

#### 3. Иерархия с метаданными (JSON)
**Пример**: `products_electronics_sample.json`
```json
{
  "data": {
    "iPhone 15 Pro": {
      "subcategory": "Premium Smartphones",
      "category": "Mobile Devices",
      "department": "Electronics",
      "brand": "Apple",
      "aliases": ["iPhone 15 Pro Max", "Apple iPhone 15 Pro"]
    }
  }
}
```
**Характеристики**:
- Иерархия + дополнительные поля (brand)
- Алиасы как массив строк
- Расширяемая структура

#### 4. Простое отображение (CSV)
**Пример**: `professions_simple_sample.csv`
```csv
value,category
Software Developer,Technology
Data Scientist,Technology
```
**Характеристики**:
- Минимальная структура: ключ → значение
- Нет алиасов или дополнительных полей
- Подходит для базовой категоризации

#### 5. Расширенная таблица (CSV)
**Пример**: `medical_conditions_sample.csv`
```csv
condition,category,system,severity,icd_group,common_names
Hypertension,Cardiovascular Diseases,Circulatory System,Chronic,I10-I15,"High Blood Pressure;HTN;Arterial Hypertension"
```
**Характеристики**:
- Множество дополнительных атрибутов
- Алиасы в отдельном поле (common_names)
- Богатая метаинформация

### Требования к коду обработки справочников

Код должен уметь обрабатывать:

1. **Различные форматы файлов**:
   - JSON с вложенной структурой
   - CSV с плоской структурой

2. **Различные способы хранения алиасов**:
   - Отсутствие алиасов
   - Алиасы как массив JSON
   - Алиасы как строка с разделителями (`;` или `|`)

3. **Различную глубину иерархии**:
   - От 2 уровней (простое отображение)
   - До 4-5 уровней (полная иерархия)

4. **Дополнительные поля**:
   - Игнорировать или использовать как метаданные
   - Не влияют на основную логику категоризации

5. **Метаданные справочника**:
   - format_version, hierarchy_type, levels (опциональны)
   - description (опционально)

### Универсальная схема загрузки

```python
def load_dictionary(path: str) -> Dict[str, Any]:
    """Универсальная загрузка справочника"""
    
    if path.endswith('.json'):
        # JSON: проверить наличие 'data' или использовать весь объект
        data = load_json(path)
        if 'data' in data:
            return data['data']
        return data
        
    elif path.endswith('.csv'):
        # CSV: первая колонка - ключ, остальные - атрибуты
        df = pd.read_csv(path)
        result = {}
        for _, row in df.iterrows():
            key = row.iloc[0]
            if len(df.columns) == 2:
                # Простое отображение
                result[key] = row.iloc[1]
            else:
                # Сложная структура
                result[key] = row.to_dict()
        return result
```

Эта спецификация показывает, что все справочники следуют общим паттернам, что позволяет создать универсальный код для их обработки.