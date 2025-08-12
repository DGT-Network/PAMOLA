# Software Requirements Specification (SRS)

## Проект: PAMOLA.CORE Mock Task – Оптимизированная LLM Анонимизация

### Task ID: `t_3LLM1`

---

## 1. Введение

Данный документ описывает обновленные требования для задачи анонимизации резюме с использованием локальной LLM. Основной фокус обновления - повышение производительности, облегчение экспериментирования и улучшение качества анонимизации через предварительную обработку и умное кэширование.

---

## 2. Цели и задачи

- Упростить процесс экспериментирования с различными моделями и параметрами
- Оптимизировать скорость обработки данных через предпроцессинг и кэширование
- Повысить качество анонимизации за счет использования внешних словарей и категорий
- Обеспечить возможность поэтапной обработки и возобновления прерванных задач
- Реализовать гибкое переключение между различными конфигурациями и режимами работы

---

## 3. Системная архитектура

### 3.1 Компоненты системы

- **Модуль конфигурации** - загрузка и объединение настроек из файла и аргументов командной строки
- **Модуль предпроцессинга** - очистка и подготовка текста, замена на основе словарей
- **LLM-модуль** - взаимодействие с локальной моделью через LM Studio или напрямую
- **Модуль кэширования** - хранение и поиск результатов, включая MinHash для похожих значений
- **Модуль метрик** - сбор и анализ статистики выполнения задачи

### 3.2 Файловая структура

- **Script:** `scripts/mock/t_3llm1_experience.py`
- **Конфигурации:** 
  - Основная: `{project_root}/configs/t_3LLM1.json`
  - Экспериментальная: `{project_root}/configs/experimental/t_3LLM1_exp.json`
- **Логи:** `{project_root}/log/t_3LLM1.log`
- **Метрики и отчеты:** 
  - `{task_dir}/metrics_{timestamp}.json`
  - `{data_repository}/reports/t_3LLM1_{timestamp}.json`
- **Выходные данные:** `{task_dir}/output/{dataset_name}_{timestamp}.csv`
- **Кэш:** `{task_dir}/cache/cache_{field}_{timestamp}.json`

---

## 4. Конфигурационная система

### 4.1 Базовая и экспериментальная конфигурации

Система поддерживает два типа конфигураций:

1. **Базовая конфигурация** (`t_3LLM1.json`) - содержит стандартные параметры для полноценной обработки
2. **Экспериментальная конфигурация** (`t_3LLM1_exp.json`) - содержит параметры для быстрых экспериментов

Принципиальные отличия экспериментальной конфигурации:
- Более легкая модель LLM (7B вместо 17B)
- Обработка только одного поля вместо всех
- Включение расширенного словаря для предварительной замены
- Более короткие промпты для ускорения обработки
- Сохранение промежуточных результатов
- Расширенная метрика по использованию словаря

### 4.2 Переключение между конфигурациями

- Через аргумент командной строки `--config-type [base|exp]`
- Через аргумент `--config-path` для указания произвольного пути
- Возможность объединения параметров из разных источников с приоритетом:
  1. Аргументы командной строки (высший приоритет)
  2. Указанный файл конфигурации
  3. Параметры по умолчанию (низший приоритет)

### 4.3 Ключевые параметры для быстрого переключения через CLI

```
python -m scripts.mock.t_3LLM1_Experience 
    --model "llama-4-scout-7b" 
    --field "experience_organizations" 
    --prompt-set "minimal" 
    --rows 10 
    --start-row 50
    --save-mode "intermediate"
    --use-dictionary true
    --dict-path "DATA/external_dictionaries/ner/jobs_extended.json"
```

---

## 5. Структура конфигурации

### 5.1 Базовая конфигурация

```json
{
  "project_root": "D:/VK/_DEVEL/PAMOLA.CORE",
  "data_repository": "D:/VK/_DEVEL/PAMOLA.CORE/DATA",
  "dataset_path": "processed/t_2I/output/EXPIRIENCE.csv",
  "encoding": "utf-16",
  "separator": ",",
  "text_qualifier": "\"",
  
  "fields": ["experience_organizations", "experience_descriptions", "experience_posts"],
  
  "prompts": {
    "standard": {
      "experience_organizations": "Anonymize: {VALUE}",
      "experience_descriptions": "Depersonalize work experience: {VALUE}",
      "experience_posts": "Generalize job title: {VALUE}"
    },
    "active_prompt_set": "standard"
  },
  
  "legal_forms": ["ООО", "АО", "ЗАО", "ПАО", "НПО", "ИП", "ОАО", "ФГУП", "МУП", "ГУП"],
  
  "fallback_values": {
    "experience_organizations": ["manufacturing company", "industrial plant", "IT company"],
    "experience_descriptions": ["performed assigned tasks", "fulfilled responsibilities", "provided support"],
    "experience_posts": ["employee", "specialist", "manager"]
  },
  
  "task_dir": "processed/3LLM",
  
  "caching": {
    "cache_responses": true,
    "use_minhash_cache": false,
    "minhash_similarity_threshold": 0.8
  },
  
  "llm": {
    "server_ip": "127.0.0.1",
    "server_port": 1234,
    "timeout": 1,
    "model_name": "llama-4-scout-17b-16e-instruct-i1"
  },
  
  "max_rows": 100,
  "max_errors": 10,
  "error_threshold": 0.2,
  
  "dry_run": false
}
```

### 5.2 Экспериментальная конфигурация

```json
{
  "experimental": true,
  
  "model_config": {
    "model_name": "llama-4-scout-7b-instruct-i1",
    "direct_mode": false,
    "model_path": "D:/VK/_DEVEL/Models/llama-4-scout-7b-q4_K_M.gguf",
    "server_check_method": "socket"
  },

  "process_config": {
    "active_field": "experience_organizations",
    "process_all_fields": false,
    "start_from_row": 0,
    "processing_marker": "~",
    "skip_processed": true,
    "prompt_set": "minimal",
    "enable_in_place_edit": true, // Возможность модификации исходного файла вместо создания нового
    "create_backup": true,        // Создание бэкапа исходного файла перед модификацией
    "retry_count": 4,             // Количество попыток запроса к LLM перед fallback
    "retry_delay_ms": 500         // Базовая задержка между повторными попытками в миллисекундах
  },
  
  "prompts": {
    "minimal": {
      "experience_organizations": "Generic company name for: {VALUE}",
      "experience_descriptions": "Generic work tasks for: {JOB_TITLE}",
      "experience_posts": "Generic job title for: {VALUE}"
    }
  },
  
  "save_config": {
    "save_intermediate": true,
    "final_output": false,
    "timestamp_output": true,
    "output_suffix": "_exp_{TIMESTAMP}"
  },
  
  "preprocessing": {
    "enabled": true,
    "max_words": 6,
    "text_cleaning": {
      "remove_stopwords": true,
      "stop_words_path": "DATA/external_dictionaries/stopwords/russian.txt",
      "english_stop_words_path": "DATA/external_dictionaries/stopwords/english.txt",
      "use_core_stopwords": true,  // Использовать существующий модуль pamola_core/utils/nlp/stopwords.py
      "preserve_key_terms": true,  // Сохранять ключевые термины даже если они в списке стоп-слов
      "key_terms": ["senior", "lead", "manager", "head", "chief", "директор", "руководитель", "начальник", "ведущий", "главный", "старший"]
    },
    "dictionary_substitution": {
      "enabled": true,
      "dictionaries": {
        "jobs": {
          "path": "DATA/external_dictionaries/ner/jobs.json",
          "match_confidence_threshold": 0.7,
          "fallback_to_category": true
        }
      }
    }
  },
  
  "caching": {
    "cache_responses": true,
    "use_minhash_cache": true,
    "persist_cache": true,
    "cache_file": "{task_dir}/dictionaries/cache_{TASK_ID}_{FIELD}_{TIMESTAMP}.json",
    "cache_key_components": ["field_type", "model_version", "input_hash"],  // Компоненты ключа кэша
    "prompt_version_tag": "v1"  // Тег версии промпта для возможности сброса кэша при изменении промптов
  },
  
  "metrics": {
    "detailed_dictionary_matches": true,
    "preprocessing_stats": true,
    "compare_with_original": true,
    "save_interim_metrics": true,
    "metrics_suffix": "_exp_{TIMESTAMP}",
    "include_config_in_metrics": true  // Включать конфигурацию в метрики для последующего анализа
  },
  
  "execution_limits": {
    "max_rows": 20,
    "max_runtime_minutes": 10,
    "max_errors": 10,
    "error_threshold": 0.2
  }
}
```

---

## 6. Функциональные требования

### 6.1 Предпроцессинг данных

#### 6.1.1 Ограничение длины входного текста

- Ограничение текста до указанного количества слов (по умолчанию 6)
- Удаление стоп-слов перед подсчетом оставшихся слов, используя словари из:
  - `DATA/external_dictionaries/stopwords/russian.txt`
  - `DATA/external_dictionaries/stopwords/english.txt`
  - Опционально модуль `pamola_core/utils/nlp/stopwords.py`
- Сохранение ключевых слов при ограничении длины, даже если они находятся в списке стоп-слов
- Сохранение значимых терминов и рангов (senior, lead, manager) даже при ограничении длины

#### 6.1.2 Очистка текста

- Использование библиотеки `clean-text` как наиболее простой и быстрой
- Настраиваемая последовательность операций с возможностью отключения каждой:
  1. Токенизация
  2. Удаление специальных символов
  3. Применение регулярных выражений
  4. Приведение к нижнему регистру
  5. Ограничение длины
- Удаление правовых форм организаций

#### 6.1.3 Словарная замена и категоризация

- Поиск ключевых слов из расширенного словаря `jobs_extended.json` в тексте
- Определение категории на основе найденных ключевых слов
- Замена текста на стандартизированное значение при превышении порога уверенности
- При замене названий организаций или должностей возможность использовать соответствующие описания деятельности из поля `activity_descriptions`
- Маркировка замененных значений для пропуска LLM-обработки (префикс `~`)
- Внесение вариативности путем случайного выбора из подходящих описаний деятельности

#### 6.1.4 Маркировка обработанных полей

- Добавление маркера `~` в начало поля для обозначения обработанных значений
- Пропуск полей с маркером при повторной обработке
- Опция удаления маркеров при финальном сохранении
- Идемпотентность: при рестарте обработанные, но поврежденные поля можно обрабатывать повторно через специальный параметр

### 6.2 Кэширование и оптимизация запросов к LLM

#### 6.2.1 Хеш-кэширование

- MD5-хеширование нормализованного текста и типа поля
- Сохранение и поиск в кэше перед обращением к LLM
- Возможность сохранения кэша между запусками в директории `{task_dir}/dictionaries/`
- Ключ кэша формируется из комбинации: тип поля, версия модели, хеш входного текста
- Отдельный тег версии промпта для возможности сброса кэша при изменении промптов
- TTL кэша не применяется, но возможен сброс при изменении версии промпта

#### 6.2.2 MinHash для похожих значений

- Использование алгоритма MinHash для определения схожих текстов
- Настраиваемый порог схожести (по умолчанию 0.8)
- Возврат кэшированного результата для схожих входных данных
- Отслеживание статистики повторного использования через MinHash

#### 6.2.3 Политика повторных попыток (retry)

- До 4 попыток отправки запроса к LLM перед использованием fallback
- Экспоненциальная задержка между попытками (начиная с 500 мс)
- Логирование причин ошибок без сохранения исходных данных
- При превышении числа попыток - использование fallback значений

### 6.3 Промпты и отправка запросов в LLM

#### 6.3.1 Наборы промптов

- Поддержка нескольких наборов промптов (стандартные и минимальные)
- Возможность быстрого переключения между наборами
- Шаблонизация промптов с подстановкой значений

#### 6.3.2 Взаимодействие с LLM

- Прямое взаимодействие через llama-cpp-python (опционально)
- Взаимодействие через LM Studio API
- Тайм-ауты и обработка ошибок
- Ограничение ответа LLM до 10 слов или 100 символов

### 6.4 Сохранение и обработка данных

#### 6.4.1 Режимы сохранения

- Промежуточное сохранение после обработки каждого поля
- Финальное сохранение с опциональной временной меткой
- Опция модификации исходного файла вместо создания нового (`enable_in_place_edit`)
- Автоматическое создание бэкапа исходного файла при его модификации
- Логирование только ID записей без исходных данных

#### 6.4.2 Частичная обработка

- Возможность указать конкретное поле для обработки
- Пропуск строк до указанного индекса
- Ограничение максимального количества обрабатываемых строк
- Пропуск полей с маркером `~` в первой позиции (уже обработанные)
- Возможность принудительной повторной обработки маркированных полей через параметр конфигурации

### 6.5 Метрики и отчетность

#### 6.5.1 Базовые метрики

- Общее время выполнения
- Количество обработанных записей
- Частота и типы ошибок
- Скорость обработки и трендовые изменения

#### 6.5.2 Расширенные метрики

- Статистика словарных замен по категориям
- Статистика использования кэша и MinHash
- Сравнение результатов с оригинальными данными
- Визуализация процесса обработки

#### 6.5.3 Формат отчетов

- JSON-отчет с метриками
- Опциональная визуализация результатов
- Временные метки для сравнения разных запусков

---

## 7. Проверка LLM-сервера

### 7.1 Методы проверки

#### 7.1.1 Проверка через сокет (основной метод)

```python
def check_llm_server_socket(config: Dict[str, Any]) -> bool:
    """
    Проверка доступности LLM-сервера через прямое подключение к сокету.
    
    Args:
        config: Словарь конфигурации с настройками LLM
        
    Returns:
        True если сервер доступен, False в противном случае
    """
    ip = config["llm"]["server_ip"]
    port = config["llm"]["server_port"]
    
    try:
        with socket.create_connection((ip, port), timeout=1):
            logger.info(f"LLM server is available at {ip}:{port}")
            return True
    except Exception as e:
        logger.warning(f"Cannot connect to LLM server at {ip}:{port}: {e}")
        return False
```

#### 7.1.2 Проверка через CLI (запасной метод)

```python
def check_llm_server_cli(config: Dict[str, Any]) -> bool:
    """
    Проверка доступности LLM-сервера через CLI-команду.
    
    Args:
        config: Словарь конфигурации с настройками LLM
        
    Returns:
        True если сервер доступен, False в противном случае
    """
    try:
        result = subprocess.run(
            ["lms", "status"], 
            capture_output=True, 
            encoding="utf-8",
            errors="ignore"
        )
        
        if "Server: ON" in result.stdout:
            logger.info("LLM server is running")
            return True
        else:
            logger.warning("LLM server is not running")
            return False
    except Exception as e:
        logger.warning(f"Failed to check LLM server status via CLI: {e}")
        return False
```

---

## 8. Обработка ключевых слов из словаря

### 8.1 Структура словаря

Система использует словарь категорий из `jobs.json` для анализа и замены полей. Расширенный словарь включает типовые описания деятельности для каждой категории:

```json
{
  "categories_hierarchy": {
    "IT": {
      "alias": "it",
      "domain": "General",
      "level": 0,
      "children": ["Development", "Engineering", "Data", "DevOps", "Design", "Management", "QA", "IT_Infrastructure"],
      "keywords": ["IT", "информационные технологии", "айти", "компьютеры", "цифровые технологии"],
      "activity_descriptions": [
        "участие в разработке и поддержке информационных систем предприятия",
        "развитие технологической инфраструктуры компании и создание цифровых решений",
        "администрирование и оптимизация IT-инфраструктуры организации"
      ]
    },
    "Development": {
      "alias": "разработка",
      "domain": "IT",
      "level": 1,
      "parent_category": "IT",
      "children": ["Frontend", "Backend", "Mobile", "Game_Development", "Embedded"],
      "keywords": ["разработка", "разработчик", "программист", "кодинг", "программирование", "development", "developer", "coder", "программный код"],
      "activity_descriptions": [
        "создание и сопровождение программного обеспечения для основных бизнес-процессов",
        "разработка и поддержка программных решений в соответствии с техническими требованиями",
        "проектирование и внедрение программных продуктов для автоматизации процессов"
      ]
    },
    "Frontend": {
      "alias": "фронтенд",
      "domain": "Development",
      "level": 2,
      "parent_category": "Development",
      "keywords": ["frontend", "front-end", "клиентская часть", "UI", "пользовательский интерфейс", "верстка", "HTML", "CSS", "JavaScript", "веб-интерфейс"],
      "activity_descriptions": [
        "разработка пользовательских интерфейсов и веб-компонентов с использованием современных технологий",
        "верстка и интеграция клиентской части приложений с серверными API",
        "оптимизация производительности и удобства использования веб-приложений"
      ]
    },
    "Backend": {
      "alias": "бэкенд",
      "domain": "Development",
      "level": 2,
      "parent_category": "Development",
      "keywords": ["backend", "back-end", "серверная часть", "серверная разработка", "API", "сервер", "базы данных", "микросервисы", "бэкенд"],
      "activity_descriptions": [
        "проектирование и реализация серверной логики и API для взаимодействия с клиентскими приложениями",
        "разработка и оптимизация алгоритмов обработки данных на стороне сервера",
        "создание и поддержка микросервисной архитектуры информационных систем"
      ]
    },
    "Data": {
      "alias": "данные",
      "domain": "IT",
      "level": 1,
      "parent_category": "IT",
      "children": ["Data_Science", "Data_Engineering", "Data_Analysis", "Database"],
      "keywords": ["data", "данные", "информация", "аналитика", "большие данные", "big data", "датасеты"],
      "activity_descriptions": [
        "построение систем хранения и анализа больших объемов данных",
        "обработка и интерпретация информации для принятия бизнес-решений",
        "разработка и внедрение инструментов для работы с корпоративными данными"
      ]
    },
    "Data_Science": {
      "alias": "data_science",
      "domain": "Data",
      "level": 2,
      "parent_category": "Data",
      "keywords": ["data science", "machine learning", "ML", "AI", "искусственный интеллект", "машинное обучение", "нейронные сети", "data scientist", "data mining", "предиктивная аналитика"],
      "activity_descriptions": [
        "разработка и внедрение моделей машинного обучения для решения бизнес-задач",
        "создание алгоритмов прогнозирования и классификации на основе собранных данных",
        "применение методов искусственного интеллекта для автоматизации процессов принятия решений"
      ]
    },
    "Engineering": {
      "alias": "инженерные_науки",
      "domain": "General",
      "level": 1,
      "parent_category": "IT",
      "keywords": ["инженер", "engineering", "технический специалист", "конструктор", "системотехник", "проектирование", "разработка систем"],
      "activity_descriptions": [
        "проектирование и внедрение технических решений для оптимизации производственных процессов",
        "разработка технической документации и управление инженерными проектами",
        "модернизация существующих систем для повышения их эффективности и надежности"
      ]
    },
    "QA": {
      "alias": "quality_assurance",
      "domain": "IT",
      "level": 1,
      "parent_category": "IT",
      "children": ["Manual_QA", "Automation_QA"],
      "keywords": ["qa", "quality assurance", "тестирование", "обеспечение качества", "тестировщик", "QA engineer", "контроль качества", "баги", "тесты"],
      "activity_descriptions": [
        "обеспечение высокого качества программных продуктов через систематическое тестирование",
        "разработка и выполнение тестовых сценариев для проверки функциональности систем",
        "выявление и документирование дефектов программного обеспечения"
      ]
    },
    "Manual_QA": {
      "alias": "manual_qa",
      "domain": "QA",
      "level": 2,
      "parent_category": "QA",
      "keywords": ["manual testing", "ручное тестирование", "test cases", "тест-кейсы", "ручные тесты", "функциональное тестирование", "регрессионное тестирование"],
      "activity_descriptions": [
        "проведение ручного тестирования функциональности программных продуктов",
        "создание и выполнение тест-кейсов для проверки работоспособности системы",
        "проверка пользовательского интерфейса и удобства использования программного обеспечения"
      ]
    },
    "Automation_QA": {
      "alias": "automation_qa",
      "domain": "QA",
      "level": 2,
      "parent_category": "QA",
      "keywords": ["test automation", "автоматизация тестирования", "автотесты", "selenium", "cypress", "автоматизированное тестирование", "CI/CD", "тестовый фреймворк"],
      "activity_descriptions": [
        "разработка и поддержка автоматизированных тестов для непрерывной проверки качества",
        "создание фреймворков и инструментов для автоматизации процессов тестирования",
        "интеграция автоматических тестов в конвейер непрерывной поставки программного обеспечения"
      ]
    },
    "DevOps": {
      "alias": "devops",
      "domain": "IT",
      "level": 1,
      "parent_category": "IT",
      "keywords": ["devops", "деплой", "CI/CD", "continuous integration", "автоматизация развертывания", "kubernetes", "docker", "контейнеризация", "инфраструктура как код"],
      "activity_descriptions": [
        "настройка и поддержка процессов непрерывной интеграции и доставки программного обеспечения",
        "автоматизация развертывания и масштабирования инфраструктуры для информационных систем",
        "обеспечение стабильной работы и мониторинг производственных сред"
      ]
    },
    "Marketing": {
      "alias": "marketing",
      "domain": "Non-IT",
      "level": 1,
      "parent_category": "Non-IT",
      "keywords": ["marketing", "маркетинг", "smm", "seo", "реклама", "продвижение", "брендинг", "digital marketing", "контент-маркетинг", "маркетолог"],
      "activity_descriptions": [
        "разработка и реализация маркетинговых стратегий для продвижения продуктов и услуг",
        "проведение маркетинговых исследований и анализ целевой аудитории",
        "управление рекламными кампаниями и оценка их эффективности"
      ]
    },
    "HR": {
      "alias": "hr",
      "domain": "Non-IT",
      "level": 1,
      "parent_category": "Non-IT",
      "keywords": ["hr", "human resources", "рекрутер", "кадры", "персонал", "управление персоналом", "hr-менеджер", "подбор персонала", "HR-специалист"],
      "activity_descriptions": [
        "поиск и подбор персонала в соответствии с потребностями компании",
        "организация процессов адаптации и обучения сотрудников",
        "разработка систем мотивации и управление корпоративной культурой"
      ]
    }
  },
  "Unclassified": {
    "alias": "unclassified",
    "domain": "General",
    "level": 0,
    "parent_category": null,
    "fallback_strategy": "uniform_distribution",
    "generic_substitutes": [
      "IT Specialist",
      "Technical Specialist",
      "Engineering Professional",
      "Digital Specialist",
      "Technology Expert",
      "Systems Professional",
      "Solutions Developer",
      "Technical Consultant",
      "Digital Professional",
      "IT Consultant",
      "Software Professional",
      "Technical Expert",
      "IT Engineer",
      "Digital Engineer",
      "Systems Specialist",
      "Solutions Specialist",
      "Technical Coordinator",
      "IT Coordinator",
      "Digital Solutions Expert",
      "Technology Specialist",
      "Инженер",
      "IT-специалист",
      "Технический специалист",
      "Разработчик",
      "Программист",
      "Технолог",
      "Аналитик",
      "Консультант",
      "Специалист по цифровым технологиям",
      "Эксперт по техническим вопросам"
    ],
    "activity_descriptions": [
      "решение технических задач и оптимизация рабочих процессов",
      "участие в разработке и внедрении инновационных решений",
      "обеспечение бесперебойного функционирования технических систем",
      "консультирование по вопросам оптимизации и автоматизации процессов",
      "анализ и внедрение современных технологических решений"
    ],
    "language": ["en", "ru"],
    "usage_rules": {
      "apply_when": "no_confidence_or_unknown",
      "selection_method": "random_uniform",
      "min_confidence_threshold": 0.2
    }
  }
}
```

### 8.2 Алгоритм сопоставления и замены

```python
def match_category_by_keywords(text: str, categories: Dict, 
                              confidence_threshold: float = 0.7, 
                              field_type: str = "experience_posts") -> Tuple[str, float, str, Optional[str]]:
    """
    Определяет категорию на основе ключевых слов и возвращает соответствующую замену.
    
    Args:
        text: Анализируемый текст
        categories: Словарь категорий и ключевых слов
        confidence_threshold: Порог уверенности для определения категории
        field_type: Тип обрабатываемого поля для выбора соответствующей замены
        
    Returns:
        Tuple[category_name, confidence, suggested_replacement, activity_description]
    """
    # Нормализация текста
    normalized_text = normalize_text(text.lower())
    
    best_match = None
    best_confidence = 0.0
    best_keywords_matched = []
    
    # Поиск совпадений по каждой категории
    for category, info in categories.items():
        if "keywords" not in info:
            continue
            
        keywords = info["keywords"]
        matched_keywords = [kw for kw in keywords if kw.lower() in normalized_text]
        matches = len(matched_keywords)
        
        if matches > 0:
            # Вычисление уверенности на основе количества совпадений и их значимости
            # Даем больший вес если ключевое слово находится в начале текста
            weighted_matches = sum(1.5 if normalized_text.startswith(kw.lower()) else 1.0 
                                   for kw in matched_keywords)
            
            # Также учитываем процент от всех возможных ключевых слов
            confidence = min(1.0, weighted_matches / max(len(keywords), 1))
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = category
                best_keywords_matched = matched_keywords
    
    if best_match and best_confidence >= confidence_threshold:
        # Выбор замены на основе типа поля
        category_info = categories[best_match]
        
        # Для должностей используем generic_substitutes
        if field_type == "experience_posts":
            substitutes = category_info.get("generic_substitutes", [])
            
            if not substitutes and "parent_category" in category_info:
                # Если у категории нет замен, используем родительскую категорию
                parent = category_info["parent_category"]
                if parent in categories:
                    substitutes = categories[parent].get("generic_substitutes", [])
            
            # Если все еще нет замен, используем некатегоризованные
            if not substitutes:
                substitutes = categories.get("Unclassified", {}).get("generic_substitutes", [])
                
            # Выбор случайной замены из списка
            replacement = np.random.choice(substitutes) if substitutes else best_match
            
        # Для описаний опыта используем activity_descriptions
        elif field_type == "experience_descriptions":
            activities = category_info.get("activity_descriptions", [])
            
            if not activities and "parent_category" in category_info:
                # Если у категории нет описаний, используем родительскую категорию
                parent = category_info["parent_category"]
                if parent in categories:
                    activities = categories[parent].get("activity_descriptions", [])
            
            # Если все еще нет описаний, используем некатегоризованные
            if not activities:
                activities = categories.get("Unclassified", {}).get("activity_descriptions", [])
                
            # Выбор случайного описания деятельности
            activity_desc = np.random.choice(activities) if activities else None
            
            # Для описаний возвращаем и категорию (для справки) и текст активности
            return best_match, best_confidence, best_match, activity_desc
            
        # Для организаций можем использовать шаблоны типа "компания в сфере {категория}"
        elif field_type == "experience_organizations":
            org_templates = [
                f"компания в сфере {category_info.get('alias', best_match)}",
                f"организация, специализирующаяся на {category_info.get('alias', best_match)}",
                f"{category_info.get('alias', best_match)} компания",
                f"предприятие в области {category_info.get('alias', best_match)}"
            ]
            replacement = np.random.choice(org_templates)
            
        # Выбор описания деятельности для всех типов полей (кроме experience_descriptions)
        activity_desc = None
        if field_type != "experience_descriptions" and "activity_descriptions" in category_info:
            activity_desc = np.random.choice(category_info["activity_descriptions"])
        
        return best_match, best_confidence, replacement, activity_desc
        
    # Если подходящей категории не найдено
    unclassified = categories.get("Unclassified", {})
    
    if field_type == "experience_descriptions":
        fallback_desc = np.random.choice(
            unclassified.get("activity_descriptions", ["выполнение профессиональных обязанностей"])
        )
        return "Unclassified", 0.0, "Unclassified", fallback_desc
    else:
        fallback_replacement = np.random.choice(
            unclassified.get("generic_substitutes", ["Specialist"])
        )
        return "Unclassified", 0.0, fallback_replacement, None
```

---

## 9. Метрики и сравнение запусков

### 9.1 Базовые метрики

- Общее время выполнения
- Среднее время обработки одной записи
- Количество ошибок и отказов
- Статистика использования кэша
- График скорости обработки во времени

### 9.2 Расширенные метрики для экспериментальной конфигурации

- Статистика по словарным заменам с разбивкой по категориям
- Количество и процент заменяемых значений через MinHash
- Оценка схожести анонимизированных данных с оригиналом
- Распределение длин текстов до и после обработки

### 10. Безопасность и логирование

### 10.1 Требования к безопасности данных

- Никогда не логировать оригинальные данные полей, содержащие персональную информацию
- Логировать только идентификаторы записей, названия полей и метрики обработки
- При сохранении промежуточных результатов обеспечивать их доступность только для авторизованных пользователей

### 10.2 Структура и уровни логирования

- INFO: информация о начале/завершении обработки, загрузке конфигурации, общие статистические данные
- WARNING: проблемы, не прерывающие обработку (временное недоступность сервера, повторные попытки)
- ERROR: критические ошибки, требующие вмешательства (превышение порога ошибок, недоступность ресурсов)
- Каждая запись лога должна содержать метку времени, уровень, модуль, сообщение

### 10.3 Контроль ошибок

- Ограничение максимального количества последовательных ошибок до 10 (настраиваемо)
- Ограничение общего количества ошибок до 20% от обрабатываемых записей
- При превышении порогов ошибок - остановка обработки с сохранением промежуточных результатов
- Для каждой ошибки фиксируется: тип, сообщение, ID записи, имя поля, время

### 10.4 Хранение логов и метрик

- Ротация логов не предусмотрена на этапе Mock-реализации
- Логи хранятся в директории `{project_root}/log/`
- Метрики хранятся в директории `{task_dir}` с добавлением временных меток
- Отчеты хранятся в `{data_repository}/reports/` с добавлением временных меток

## 11. Сравнение и анализ результатов

### 11.1 Метрики для сравнения подходов

- Сравнение "полной" и "упрощенной" анонимизации на основе метрик:
  - Скорость обработки (записей в секунду)
  - Использование ресурсов (память, CPU/GPU)
  - Статистика использования словаря
  - Количество и тип ошибок

### 11.2 Формат и хранение результатов

- Конфигурация, используемая для получения результатов, сохраняется вместе с метриками
- Временные метки добавляются к именам файлов для облегчения сравнения
- Результаты могут быть сравнены с использованием стандартных инструментов для работы с JSON

### 11.3 Будущая валидация

- Ручная валидация результатов анонимизации будет проведена на следующих этапах проекта
- Автоматическое тестирование качества анонимизации и оценка k-anonymity не входят в рамки текущей задачи

### 9.3 Метрики использования словаря

Пример метрик использования словаря:

```json
{
  "dictionary_metrics": {
    "total_processed": 1000,
    "replaced_by_dictionary": 652,
    "replaced_by_llm": 348,
    "categories_distribution": {
      "Development": 145,
      "Data": 98,
      "Engineering": 76,
      "QA": 112,
      "DevOps": 45,
      "Marketing": 31,
      "HR": 27,
      "Unclassified": 118
    },
    "confidence_distribution": {
      "high_confidence": 423,
      "medium_confidence": 147,
      "low_confidence": 82
    },
    "activity_descriptions_used": 412,
    "average_confidence": 0.72,
    "time_saved_seconds": 1248.5,
    "estimated_cost_saved": "$24.97"
  }
}
```

### 9.4 Информационные потери при предобработке

Для оценки потерь информации при ограничении текста предусмотрены следующие метрики:

```json
{
  "preprocessing_metrics": {
    "original_text_length": {
      "avg_characters": 128.6,
      "avg_words": 18.3,
      "max_characters": 756,
      "max_words": 112
    },
    "preprocessed_text_length": {
      "avg_characters": 42.1,
      "avg_words": 5.8,
      "max_characters": 48,
      "max_words": 6
    },
    "information_retention": {
      "avg_word_ratio": 0.31,
      "key_terms_preserved": 0.94,
      "job_indicators_preserved": 0.98
    },
    "stopwords_removed": {
      "total_count": 5842,
      "avg_per_record": 5.8,
      "most_frequent": ["и", "в", "на", "с", "по"]
    },
    "keywords_preserved": {
      "total_count": 2156,
      "avg_per_record": 2.2,
      "most_frequent": ["разработка", "программирование", "тестирование", "анализ", "управление"]
    }
  }
}
```

### 9.5 Сравнение между конфигурациями

Для сравнения эффективности различных конфигураций:

```json
{
  "comparative_metrics": {
    "timestamp": "2025-05-11T22:30:15",
    "configurations_compared": ["base", "experimental"],
    "dataset_size": 1000,
    "processing_speed": {
      "base": {
        "records_per_second": 1.2,
        "total_time_seconds": 833.5,
        "llm_calls": 952
      },
      "experimental": {
        "records_per_second": 4.8,
        "total_time_seconds": 208.3,
        "llm_calls": 348
      },
      "improvement_factor": 4.0
    },
    "resource_usage": {
      "base": {
        "max_memory_mb": 1248,
        "avg_cpu_percent": 32.5,
        "gpu_memory_mb": 12288
      },
      "experimental": {
        "max_memory_mb": 945,
        "avg_cpu_percent": 28.3,
        "gpu_memory_mb": 6144
      }
    },
    "error_rates": {
      "base": 0.015,
      "experimental": 0.018,
      "difference_significant": false
    },
    "quality_comparison": {
      "dictionary_match_rate": {
        "base": 0.0,
        "experimental": 0.65
      },
      "text_similarity_to_original": {
        "base": 0.18,
        "experimental": 0.22
      }
    },
    "conclusion": "Экспериментальная конфигурация демонстрирует 4x прирост скорости при сохранении сопоставимого качества анонимизации."
  }
}
```

---

## 11. Ограничения и допущения

### 11.1 Требования к производительности

- Задача предназначена для обработки наборов данных среднего размера (до 180 000 строк)
- Кэширование выполняется в оперативной памяти во время выполнения задачи
- При необходимости, кэш может быть сохранен в файл и загружен при следующем запуске

### 11.2 Системные требования

- Python 3.12
- Минимум 12 GB видеопамяти при использовании GPU (для моделей до 7B параметров)
- Архитектура поддерживает работу через API, что позволяет использовать распределенные ресурсы

### 11.3 Ограничения реализации

- Не предусмотрена пакетная обработка (batching) промптов - один запрос за раз
- Не реализовано шифрование данных и логов (это часть не для Mock)
- Не предусмотрены интеграционные тесты на данном этапе
- Не реализован мониторинг через Prometheus/Grafana

### 11.4 Предположения

- Предполагается, что сервер LM Studio будет запущен и доступен по указанному адресу и порту
- Качество анонимизации будет проверяться на последующих этапах, а не в рамках данной задачи
- Предполагается, что структура данных не будет существенно меняться во время выполнения задачи

## 13. Заключение

Обновленное решение для задачи анонимизации резюме с использованием LLM предоставляет значительные улучшения в производительности и гибкости экспериментирования. Основные улучшения включают:

1. Гибкая конфигурация с возможностью быстрого переключения между двумя режимами работы (базовый и экспериментальный)
2. Эффективный предпроцессинг с ограничением длины текста до 6 слов с сохранением ключевых терминов и удалением стоп-слов
3. Словарные замены на основе расширенного словаря категорий профессий для снижения нагрузки на LLM
4. Умное кэширование и обнаружение похожих значений через MinHash
5. Маркировка обработанных полей символом `~` для возможности поэтапной обработки
6. Расширенная метрика и отчетность для сравнения эффективности различных подходов
7. Возможность модификации исходного файла вместо создания нового
8. Политика повторных попыток при ошибках с ограничением их максимального количества

Данная реализация фокусируется на практической применимости и оптимизации производительности при сохранении качества анонимизации. Она не решает вопросы долгосрочного хранения, мониторинга и безопасности данных, которые будут рассмотрены в последующих версиях.

Реализация обеспечивает баланс между скоростью и качеством, предоставляя исследователям инструмент для быстрого экспериментирования с различными настройками и параметрами анонимизации.