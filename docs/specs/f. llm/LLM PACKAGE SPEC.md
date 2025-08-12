# Детальная спецификация пакета pamola_core/llm для PAMOLA CORE Text Anonymization (обновленная)

## 1. Обзор системы

В рамках разработки ядра core (основная библиотека PAMOLA.CORE) и прикладного ядра необходимо выполнить обработку длинных тектовых полей. 
Пакет `pamola_core/llm` предоставляет функциональность для анонимизации текстовых данных с использованием языковых моделей через API LM Studio. Модуль интегрируется с существующей системой PAMOLA CORE, наследуя основной дизайн операций и используя вспомогательные модули для I/O, логирования, прогресс-трекинга и визуализации.

## 2. Архитектура пакета

```
pamola_core/
├── llm/                      # Новый пакет для LLM-функциональности
│   ├── __init__.py
│   ├── base.py               # Базовые интерфейсы LLM-клиентов
│   ├── api_client.py         # Универсальный HTTP-клиент для LLM API
│   ├── prompt.py             # Управление промптами и шаблонами
│   ├── utils/                # Вспомогательные утилиты
│   │   ├── __init__.py
│   │   ├── routing.py        # Маршрутизация запросов к LLM
│   │   └── checkpoint_manager.py  # Управление контрольными точками
│   └── providers/            # Реализации для различных провайдеров
│       ├── __init__.py
│       └── lm_studio.py      # Реализация клиента для LM Studio
└── utils/
    ├── nlp/
    │   ├── vectorizer.py     # Новый модуль векторизации (добавляется)
    │   └── anonymization/    # Подпакет для анонимизации
    │       ├── __init__.py
    │       ├── base.py       # Базовые классы анонимизаторов
    │       ├── llm_anonymizer.py  # Анонимизация через LLM
    │       ├── registry.py   # Регистрация анонимизаторов
    │       └── multi_level.py  # Комбинирование стратегий анонимизации
    └── ops/
        └── op_llm_anonymizer.py  # Основная операция анонимизации
```

## 3. Основные компоненты и их назначение

### 3.1. LLM-клиенты (pamola_core/llm/)

#### 3.1.1. base.py

- Определяет базовый интерфейс `LLMClient` для всех LLM-клиентов
- Содержит класс `LLMResponse` для представления ответов от LLM
- Все провайдеры должны реализовывать этот интерфейс

#### 3.1.2. api_client.py

- Реализует базовый HTTP-клиент `APIClient`
- Обеспечивает механизм повторных попыток с экспоненциальной задержкой
- Обрабатывает сетевые ошибки и проблемы с API
- Поддерживает таймауты и настраиваемые HTTP-заголовки

#### 3.1.3. providers/lm_studio.py

- Реализует клиент для LM Studio
- Поддерживает формат API LM Studio
- Обрабатывает специфичные для LM Studio параметры и ответы

### 3.2. Управление промптами (pamola_core/llm/prompt.py)

- Класс `PromptTemplate` для работы с шаблонами промптов
- Загрузка промптов из JSON-файлов
- Простой механизм подстановки переменных в шаблоны
- Поддержка системных промптов для LLM

### 3.3. Маршрутизация и балансировка (pamola_core/llm/utils/routing.py)

- Класс `LoadBalancer` для работы с несколькими LLM-клиентами
- Стратегии балансировки: round-robin, random
- Обработка ошибок с автоматическим переключением на другие клиенты
- Сбор статистики использования

### 3.4. Контрольные точки (pamola_core/llm/utils/checkpoint_manager.py)

- Класс `CheckpointManager` для управления контрольными точками
- Сохранение и загрузка состояния обработки
- Поддержка возобновления обработки с последней точки
- Сохранение промежуточных результатов

### 3.5. Векторизация (pamola_core/utils/nlp/vectorizer.py)

- Класс `TextVectorizer` для векторизации текста
- Поддержка различных моделей (FastText, SentenceTransformers)
- Резервный механизм для случаев отсутствия моделей
- Вычисление косинусного сходства между текстами

### 3.6. Анонимизация (pamola_core/utils/nlp/anonymization/)

#### 3.6.1. base.py

- Базовый класс `BaseAnonymizer` для всех анонимизаторов
- Класс `EntityAwareAnonymizer` для анонимизаторов, учитывающих сущности
- Стандартный интерфейс для всех реализаций

#### 3.6.2. llm_anonymizer.py

- Класс `LLMAnonymizer` для анонимизации через LLM
- Интеграция с LLM-клиентами
- Опциональная проверка сходства через векторизацию

#### 3.6.3. registry.py

- Реестр доступных анонимизаторов
- Фабрика для создания анонимизаторов
- Регистрация пользовательских анонимизаторов

#### 3.6.4. multi_level.py

- Класс `MultiLevelAnonymizer` для комбинирования стратегий
- Последовательное применение нескольких анонимизаторов

### 3.7. Операция анонимизации (pamola_core/utils/ops/op_llm_anonymizer.py)

- Класс `LLMAnonymizerOperation`, наследующий от `FieldOperation`
- Интеграция с системой операций PAMOLA CORE
- Обработка входных и выходных данных
- Журналирование, прогресс-трекинг и метрики

## 4. Детальные спецификации компонентов

### 4.1. LLMClient и LLMResponse (интерфейсы)

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union

@dataclass
class LLMResponse:
    """Представляет ответ от LLM API."""
    request_id: str          # Уникальный ID запроса
    model_name: str          # Имя модели
    prompt_tokens: int       # Токены в запросе
    completion_tokens: int   # Токены в ответе
    total_tokens: int        # Суммарно токенов
    latency: float           # Время ответа в секундах
    success: bool            # Успех запроса
    content: str             # Содержание ответа
    error: Optional[str] = None  # Ошибка (если есть)
    
class LLMClient(ABC):
    @abstractmethod
    def complete(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 512,
                 stop: Optional[List[str]] = None, 
                 stream: bool = False,
                 **kwargs) -> LLMResponse:
        """
        Отправляет запрос на завершение текста и возвращает ответ.
        
        Args:
            prompt: Основной текст промпта
            system_prompt: Системный промпт (инструкции для модели)
            temperature: Креативность (0.0-1.0)
            top_p: Порог вероятности для выборки (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
            stop: Список стоп-слов для завершения ответа
            stream: Включить потоковый режим
            **kwargs: Дополнительные параметры для API
            
        Returns:
            Ответ от модели в виде объекта LLMResponse
        """
        pass
    
    @abstractmethod
    def get_client_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о клиенте и модели.
        
        Returns:
            Словарь с информацией о клиенте
        """
        pass
```

### 4.2. PromptTemplate (интерфейс шаблонов)

```python
class PromptTemplate:
    def __init__(self, template: str, system: Optional[str] = None, lang: Optional[str] = None, **kwargs):
        """Инициализация с шаблоном, системным промптом и языком"""
        
    def render(self, **kwargs) -> Dict[str, str]:
        """
        Рендеринг шаблона с заменой переменных
        
        Возвращает:
            {"prompt": "...", "system": "..."}
        """

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> 'PromptTemplate':
        """Загрузка шаблона из JSON-файла"""
```

### 4.3. CheckpointManager (интерфейс контрольных точек)

```python
class CheckpointManager:
    def __init__(self, task_dir: Union[str, Path], operation_id: str, id_column: Optional[str] = None):
        """Инициализация с директорией задачи и идентификатором операции"""
    
    def save_checkpoint(self, current_position: int, last_id: Any = None, 
                        partial_results_file: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Сохранение контрольной точки"""
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Загрузка последней контрольной точки"""
    
    def get_resume_position(self) -> Tuple[int, Optional[Any]]:
        """Получение позиции для возобновления и ID последней записи"""
```

### 4.4. TextVectorizer (интерфейс векторизации)

```python
class TextVectorizer:
    def __init__(self, model_name: Optional[str] = None):
        """Инициализация с опциональным именем модели"""
    
    def vectorize_text(self, text: str) -> np.ndarray:
        """Векторизация текста в numpy-массив"""
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисление косинусного сходства между векторами"""
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Вычисление сходства между текстами"""
```

### 4.5. BaseAnonymizer (интерфейс анонимизаторов)

```python
class BaseAnonymizer(ABC):
    def __init__(self, name: str):
        """Инициализация с именем анонимизатора"""
    
    @abstractmethod
    def anonymize(self, text: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Анонимизирует текст
        
        Возвращает:
            (анонимизированный_текст, метаданные)
        """
    
    def get_info(self) -> Dict[str, Any]:
        """Информация об анонимизаторе"""
```

### 4.6. Интерфейсы ядра системы

#### 4.6.1. FieldOperation (базовый класс)

```python
class FieldOperation(BaseOperation):
    def __init__(self, field_name: str, description: str = ""):
        """
        Инициализирует операцию для поля.
        
        Args:
            field_name: Имя поля для обработки
            description: Описание операции
        """
        super().__init__(f"{field_name} operation", description or f"Operation for field {field_name}")
        self.field_name = field_name
    
    def run(self, 
            data_source: DataSource, 
            task_dir: Path, 
            reporter: Any, 
            track_progress: bool = True,
            **kwargs) -> OperationResult:
        """Выполняет операцию с отслеживанием прогресса"""
        pass
```

#### 4.6.2. ProgressTracker

```python
class ProgressTracker:
    def __init__(self, total: int, description: str, unit: str = "records"):
        """
        Инициализирует трекер прогресса.
        
        Args:
            total: Общее количество шагов
            description: Описание процесса
            unit: Единица измерения (records, steps, etc.)
        """
    
    def update(self, n: int = 1, postfix: Optional[Dict[str, Any]] = None) -> None:
        """
        Обновляет прогресс.
        
        Args:
            n: Количество выполненных шагов
            postfix: Дополнительная информация для отображения
        """
    
    def create_subtask(self, total: int, description: str, unit: str = "subtasks") -> 'ProgressTracker':
        """
        Создает подзадачу с собственным прогрессом.
        
        Args:
            total: Общее количество шагов в подзадаче
            description: Описание подзадачи
            unit: Единица измерения в подзадаче
            
        Returns:
            Новый трекер прогресса для подзадачи
        """
```

#### 4.6.3. DataSource

```python
class DataSource:
    @classmethod
    def from_file_path(cls, file_path: Union[str, Path], **kwargs) -> 'DataSource':
        """
        Создает источник данных из файла.
        
        Args:
            file_path: Путь к файлу (CSV, Parquet)
            **kwargs: Дополнительные параметры чтения
            
        Returns:
            Источник данных
        """
    
    def get_dataframe(self, name: str = "main", load_if_path: bool = True) -> pd.DataFrame:
        """
        Получает DataFrame по имени.
        
        Args:
            name: Имя датафрейма
            load_if_path: Загружать ли, если это путь к файлу
            
        Returns:
            DataFrame с данными
        """
    
    def get_file_path(self, name: str) -> Optional[Path]:
        """
        Получает путь к файлу по имени.
        
        Args:
            name: Имя файла
            
        Returns:
            Путь к файлу или None
        """
```

#### 4.6.4. OperationResult

```python
@dataclass
class OperationResult:
    """Результат выполнения операции."""
    status: OperationStatus             # Статус операции (SUCCESS, ERROR, etc.)
    artifacts: List[OperationArtifact]  # Артефакты (файлы, графики)
    metrics: Dict[str, Any]             # Метрики выполнения
    error_message: Optional[str] = None # Сообщение об ошибке
    execution_time: Optional[float] = None  # Время выполнения в секундах
    
    def add_artifact(self, artifact_type: str, path: Union[str, Path], description: str = ""):
        """Добавляет артефакт к результату операции"""
    
    def add_metric(self, name: str, value: Any):
        """Добавляет метрику к результату операции"""
```

### 4.7. LLMAnonymizerOperation (интерфейс операции)

```python
@register()
class LLMAnonymizerOperation(FieldOperation):
    def __init__(self, 
                 field_name: str, 
                 prompt_name: str, 
                 vectorize: bool = True, 
                 replace: bool = False,
                 llm_url: str = "http://localhost:1234/v1",
                 model_name: Optional[str] = None,
                 lang: Optional[str] = None,
                 threshold: float = 0.7,
                 **kwargs):
        """
        Инициализация операции анонимизации.
        
        Args:
            field_name: Имя поля для анонимизации
            prompt_name: Имя JSON-файла с шаблоном промпта
            vectorize: Включить проверку сходства
            replace: Заменять исходное поле (иначе создает новое)
            llm_url: URL-адрес API LLM
            model_name: Имя модели (если None, используется модель по умолчанию)
            lang: Язык текста (если None, определяется автоматически)
            threshold: Порог сходства для отсечения некачественных результатов
            **kwargs: Дополнительные параметры
        """
        super().__init__(field_name, f"LLM anonymization for field {field_name}")
        # Инициализация параметров
    
    def execute(self, 
                data_source: DataSource, 
                task_dir: Path, 
                reporter: Any, 
                progress_tracker: Optional[ProgressTracker] = None, 
                **kwargs) -> OperationResult:
        """
        Выполнение операции анонимизации.
        
        Args:
            data_source: Источник данных
            task_dir: Директория задачи
            reporter: Объект для отчетов
            progress_tracker: Трекер прогресса
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат операции
        """
        pass
```

## 5. Форматы данных

### 5.1. Формат JSON-шаблона промпта

```json
{
  "system": "Ты специалист по анонимизации...",
  "template": "Анонимизируй следующий текст: {{text}}",
  "lang": "ru"
}
```

- Допустимые переменные шаблона: `{{text}}`, `{{field}}`, `{{name}}`
- Расположение промптов: `DATA/external_dictionaries/llm/prompts/*.json`

### 5.2. Формат файла контрольной точки

```json
{
  "operation_id": "llm_anonymizer_field_name",
  "position": 1000,
  "id_column": "id",
  "last_id": "12345",
  "timestamp": 1712345678,
  "partial_results_file": "output/partial_results_1000.csv",
  "metadata": {
    "processed_count": 1000,
    "errors_count": 5
  }
}
```

### 5.3. Формат метрик операции

```json
{
  "total_records": 10000,
  "processed_records": 9850,
  "skipped_records": 150,
  "errors_count": 15,
  "llm_requests": 9850,
  "llm_response_time": {
    "avg": 1.25,
    "max": 4.32,
    "min": 0.78
  },
  "similarity_metrics": {
    "avg": 0.85,
    "std": 0.12,
    "min": 0.45,
    "max": 0.99
  },
  "execution_time": 345.67
}
```

### 5.4. Формат API LM Studio

#### 5.4.1. Запрос к LM Studio API

```json
{
  "model": "llama-2",
  "prompt": "Анонимизируй следующий текст: ...",
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 512,
  "stop": ["---"]
}
```

#### 5.4.2. Ответ от LM Studio API

```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1680000000,
  "model": "llama-2",
  "choices": [
    {
      "text": "Обезличенный текст...",
      "finish_reason": "stop",
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 46,
    "completion_tokens": 72,
    "total_tokens": 118
  }
}
```

## 6. Интеграция с существующей инфраструктурой

### 6.1. Интеграция с pamola_core.utils.ops

- Класс `LLMAnonymizerOperation` наследуется от `pamola_core.utils.ops.op_base.FieldOperation`
- Регистрируется через декоратор `@register` из `pamola_core.utils.ops.op_registry`
- Возвращает результаты в формате `OperationResult` из `pamola_core.utils.ops.op_result`
- Соблюдает контракт выполнения операций, определенный в базовых классах

### 6.2. Интеграция с pamola_core.utils.io

- Использует функции `read_full_csv`, `write_dataframe_to_csv` для работы с файлами
- Загружает промпты через стандартные функции для работы с JSON
- Соблюдает структуру директорий, определенную в инфраструктуре PAMOLA CORE
- Поддерживает шифрование данных при необходимости

### 6.3. Интеграция с pamola_core.utils.nlp

- Использует `detect_language` для определения языка текста
- Интегрируется с системой извлечения сущностей через `extract_entities`
- Расширяет NLP-возможности новым модулем векторизации

### 6.4. Интеграция с pamola_core.utils.progress

- Использует `ProgressTracker` для отслеживания прогресса
- Поддерживает функции `track_operation`, `process_dataframe_in_chunks`
- Обеспечивает видимость прогресса для пользователя

### 6.5. Интеграция с pamola_core.utils.logging

- Использует стандартные функции логирования
- Настраивает логирование через `configure_task_logging`
- Получает логгеры через `get_logger`

### 6.6. Интеграция с pamola_core.utils.visualization

- Создает визуализации с помощью функций из пакета visualization
- Генерирует графики сравнения до/после анонимизации
- Визуализирует метрики сходства и другие показатели

## 7. Структура task_dir и артефакты

Операции модуля LLM-анонимизации взаимодействуют с внешне заданной директорией задачи (`task_dir`). Это пространство используется для хранения всех артефактов обработки.

### 7.1. Структура директорий

```
{task_dir}/
├── input/                     # (опционально) входные данные
├── output/                    # сохранённые выходные таблицы
│   └── {dataset_name}.csv     # результат с анонимизированным полем
├── dictionaries/              # частотные словари, NER-словарь
├── checkpoints/               # контрольные точки для возобновления
├── prompt_used.json           # шаблон промпта для операции
├── metrics_{field}.json       # метрики выполнения по полю
├── diff_{field}.png           # визуализация изменений по полю
├── similarity_{field}.csv     # таблица сходства (если включено)
├── similarity_{field}.png     # гистограмма косинусных расстояний
├── op_{operation_id}_summary.json  # JSON-описание операции
└── logs/
    └── {task_name}.log        # лог выполнения задачи
```

### 7.2. Артефакты первого этапа

|Артефакт|Расположение|Условие появления|
|---|---|---|
|CSV результат|`output/{dataset}.csv`|Всегда при сохранении результата|
|JSON метрики|`metrics_{field}.json`|Всегда|
|PNG сравнение до/после|`diff_{field}.png`|Если включена визуализация|
|Cosine similarity CSV|`similarity_{field}.csv`|При векторизации|
|Cosine similarity гистограмма|`similarity_{field}.png`|При векторизации|
|Used prompt|`prompt_used.json`|Всегда|
|Описание операции|`op_{operation_id}_summary.json`|Всегда|
|Логи|`logs/{task_name}.log`|Автоматически|

## 8. Обработка ошибок и отказоустойчивость

### 8.1. Стратегии повторных попыток

- При ошибках HTTP-запросов:
    - Экспоненциальная задержка между попытками (1s, 2s, 4s...)
    - Ограниченное количество повторных попыток (3 по умолчанию)
    - Логирование всех ошибок и повторов

### 8.2. Возобновление обработки

- Сохранение состояния после каждого чанка или N записей
- Поддержка возобновления с последней успешной точки
- Проверка и восстановление частичных результатов

### 8.3. Проверка качества результатов

- При использовании векторизации:
    - Проверка косинусного сходства между исходным и анонимизированным текстом
    - Применение порогового значения для отсечения неприемлемых результатов
    - Возможность остановки при превышении допустимого процента ошибок

## 9. Последовательность разработки и приоритеты

### 9.1. Этап 1 (MVP)

1. `pamola_core/llm/base.py` - базовые интерфейсы
2. `pamola_core/llm/api_client.py` - базовый HTTP-клиент
3. `pamola_core/llm/providers/lm_studio.py` - реализация для LM Studio
4. `pamola_core/llm/prompt.py` - управление промптами
5. `pamola_core/utils/nlp/vectorizer.py` - модуль векторизации текста
6. `pamola_core/llm/utils/checkpoint_manager.py` - базовая версия контрольных точек
7. `pamola_core/utils/nlp/anonymization/base.py` - базовые классы анонимизаторов
8. `pamola_core/utils/nlp/anonymization/llm_anonymizer.py` - анонимизация через LLM
9. `pamola_core/utils/ops/op_llm_anonymizer.py` - основная операция анонимизации

### 9.2. Этап 2

10. `pamola_core/utils/nlp/anonymization/registry.py` - реестр анонимизаторов
11. `pamola_core/utils/nlp/anonymization/multi_level.py` - многоуровневая анонимизация
12. `pamola_core/llm/utils/routing.py` - маршрутизация запросов

### 9.3. Этап 3

13. Дополнительные провайдеры LLM (по необходимости)
14. Расширенные функции контрольных точек
15. Интеграция с другими типами анонимизаторов

## 10. Технические требования

### 10.1. Общие требования

- Python 3.11+
- Код и документация на английском языке
- Типизация с использованием аннотаций типов
- Логирование всех операций
- Строгая обработка ошибок
- Наличие docstrings для всех классов и методов

### 10.2. Зависимости

**Основные:**

- pandas
- numpy
- requests
- tqdm
- psutil

**Опциональные:**

- fasttext (для векторизации)
- sentence_transformers (для векторизации)
- jinja2 (для расширенных шаблонов)

### 10.3. Конвенции кода

- Строгое соблюдение PEP 8
- Использование docstrings в формате Google Style
- Логирование с использованием стандартного модуля logging
- Использование абстрактных базовых классов (ABC) для интерфейсов
- Последовательное использование аннотаций типов
- Обработка всех исключений с информативными сообщениями

## 11. Пример использования

```python
from pamola_core.utils.ops.op_llm_anonymizer import LLMAnonymizerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.io import ensure_directory
from pathlib import Path

# Создаем директории
task_dir = Path("tasks/anonymization_job")
ensure_directory(task_dir)

# Создаем операцию
operation = LLMAnonymizerOperation(
    field_name="experience",
    prompt_name="resume_ru",   # Ищет в DATA/external_dictionaries/llm/prompts/resume_ru.json
    vectorize=True,            # Включаем проверку сходства
    replace=False,             # Создавать новый столбец, а не заменять
    llm_url="http://localhost:1234/v1",
    threshold=0.7,             # Порог сходства
    lang="ru"                  # Предпочтительный язык
)

# Создаем источник данных
data_source = DataSource.from_file_path("data/cv_dataset.csv")

# Запускаем операцию
result = operation.run(data_source, task_dir, None, track_progress=True)

# Проверяем результат
if result.status == "success":
    print(f"Anonymization completed successfully")
    print(f"Output file: {result.artifacts[0].path}")
    print(f"Metrics: {result.metrics}")
else:
    print(f"Anonymization failed: {result.error_message}")
```

## 12. Обработка контекстных ограничений

При работе с LLM и текстовыми данными возможны ситуации, когда длина текста превышает максимальный размер контекста модели. Стратегии обработки:

1. **Автоматическое разделение**: Для длинных текстов - разделение на части и отдельная анонимизация каждой с последующим объединением.
2. **Логирование предупреждений**: При превышении порога длины - предупреждение и ограничение текста.
3. **Параметры контроля**: Возможность указать максимальную длину входного текста и стратегию обработки при превышении.
4. **Прогрессивная загрузка**: Для очень больших полей - частичная обработка с сохранением промежуточных результатов.

Теперь спецификация содержит все необходимые детали для начала разработки, включая:

- Точные форматы запросов/ответов для API LM Studio
- Детали обязательных полей и структуры данных
- Интерфейсы ключевых компонентов ядра системы
- Стратегии обработки специфических ситуаций, таких как ограничения контекста