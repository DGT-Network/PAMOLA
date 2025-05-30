### 📦 Обновленное задание на рефакторинг NumericGeneralizationOperation

---

### ✅ Основные цели
- Привести NumericGeneralizationOperation к полной совместимости с архитектурой `pamola_core/utils/ops`.
- Убрать дублирование логики между `run()` и `execute()`.
- Перевести обработку ошибок и валидацию на unified error reporting.
- Подключить современный progress tracking.
- Перевести сохранение артефактов (json и визуализаций) в единый `{task_dir}`.
- Внедрить простой механизм кэширования (`use_cache`).

---

### 🔨 Объем работ

#### **1️⃣ base_anonymization_op.py**
- Проверить соответствие `BaseOperation` (execute → process_batch).
- Перенести всё выполнение через `execute()`.
- Подключить `progress_tracker` для этапов: подготовка, обработка батчей, сохранение метрик, финализация.
- Унифицировать error reporting (через OperationResult + logger).

#### **2️⃣ commons (metric_utils, processing_utils, validation_utils)**
- **metric_utils.py** → подключить к `op_data_writer.py` → все json (метрики) сохраняются в `{task_dir}`.
- **processing_utils.py** → проверить совместимость с batch-процессингом (`process_in_chunks`), убрать дубли.
- **validation_utils.py** → перевести на error reporting через logger/OperationResult, убрать сырые ValueError.

#### **3️⃣ numeric_op.py**
- Убрать кастомный `run()`, использовать родительский `execute()`.
- Подключить современный `progress_tracker`.
- Обеспечить поддержку REPLACE/ENRICH с правильными именами выходных полей.
- Обеспечить корректную работу `process_value` для пустых/текстовых значений (возврат оригинала + логирование).
- Внедрить кеширование (task_dir/cache/, хэш параметров + данных).
- Согласовать пути всех артефактов (json, визуализаций): сохраняем прямо в `{task_dir}`.

#### **4️⃣ Кэширование (use_cache)**
- Реализовать простой механизм: если хэш параметров + данных совпадает, читаем готовый результат из `{task_dir}/cache`.

#### **5️⃣ Визуализация**
- Использовать pamola_core.utils.visualization.
- Все графики (png) сохраняем в `{task_dir}`.

---

### 📋 Чеклист изменений
| Блок                     | Изменения                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------|
| base_anonymization_op.py | execute(), progress, unified errors                                                             |
| numeric_op.py            | убрать run, progress, process_value, use_cache, артефакты в task_dir                           |
| metric_utils            | подключение к op_data_writer                                                                    |
| processing_utils        | проверка совместимости с batch                                                                 |
| validation_utils        | переход на unified error reporting                                                             |
| caching (use_cache)     | task_dir/cache, простой pickle/parquet                                                         |
| visualizations          | task_dir, png через visualization utils                                                        |

---

### ❓ Вопросы из дополнительного анализа

| Вопрос                                                                                       | Ответ                                                                                                                                                       |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Есть ли изменения в BaseOperation.run()?                                                    | Да, теперь нужно использовать execute() → process_batch(), без кастомного run().                                                                             |
| Как теперь работает DataSource?                                                            | DataSource возвращает DataFrame через get_dataframe("main"), модификации проводятся in-place или с копиями в зависимости от режима.                         |
| Были ли обновления в progress tracking?                                                    | Да, добавлен progress_tracker, который нужно подключить к операциям, обновляя шаги обработки.                                                               |
| Как теперь регистрировать артефакты?                                                      | Через unified artifact registration, всё сохраняется в {task_dir}, json — через op_data_writer, изображения — через visualization utils.                    |
| Есть ли новые стандарты для error handling?                                                | Да, ошибки больше не выбрасываются напрямую, а регистрируются через OperationResult и logger.                                                              |
| Нужно ли оставлять validation_utils или переносить проверку в ops?                         | Пока оставить validation_utils, но их ошибки нужно поднимать как OperationResult с unified reporting, а не ValueError.                                       |
| Следует ли сохранять direct DataFrame modification или перейти на абстрактный паттерн?      | Сохраняем direct modification, но следим, чтобы DataSource abstraction учитывал это (in-place vs. enriched).                                                |
| Есть ли стандартный подход к chunked processing?                                           | Да, использовать process_in_chunks из commons/processing_utils или pamola_core/utils/ops.                                                                          |
| Были ли изменения в системе визуализаций?                                                 | Сохраняем текущую систему (pamola_core.utils.visualization), но стандартизируем пути сохранения артефактов (всё в {task_dir}).                                     |
| Есть ли новые стандарты логирования?                                                      | Да, использовать централизованный logger из ops, все ошибки и предупреждения должны логироваться, а не просто выбрасываться или печататься в stdout.        |

Если хочешь, могу подготовить Mermaid-диаграмму архитектуры или подробное техническое задание для команды. Скажи, нужно ли это.

