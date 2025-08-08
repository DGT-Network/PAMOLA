после предыдущего рефакторинга я получил вывод
(.venv) PS D:\VK\_DEVEL\PAMOLA> python scripts\mock\t_3LLM2_Experience.py --reset-config --no-skip-processed --no-cache --max-records 10 --verbose --debug-llm
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
D:\VK\_DEVEL\PAMOLA\pamola_core\utils\nlp\llm\config.py:771: UserWarning: Model 'gemma-2-9b-it-russian-function-calling' has multiple aliases: LLM1 and QUALITY
  warnings.warn(
D:\VK\_DEVEL\PAMOLA\pamola_core\utils\nlp\llm\config.py:771: UserWarning: Model 'google/gemma-3-4b' has multiple aliases: LLM2 and BALANCED
  warnings.warn(
D:\VK\_DEVEL\PAMOLA\pamola_core\utils\nlp\llm\config.py:771: UserWarning: Model 'phi-3-mini-128k-it-russian-q4-k-m' has multiple aliases: FAST and LLM3
  warnings.warn(
 
....
=== LLM Performance Metrics ===
Total Operations: 10
Success Rate: 80.0%
Uptime: 0:01:04.565829

Latency:
  Mean: 4576.2ms
  P50: 4739.3ms
  P95: 8553.2ms
  P99: 9866.1ms

Throughput:
  Requests: 0.22 rps
  Tokens: 61.46 tps

Cache Performance:
  Hit Rate: 0.0%
  Avg Hit Time: 0.0ms
  Avg Miss Time: 4576.2ms
(.venv) PS D:\VK\_DEVEL\PAMOLA> 


Анонимизированный текст:
Это была стажировка в университете; где я был ведущим дизайнером и менеджером проекта в команде из четырех человек. Была разработана система база знани 
~Жду исходный текст для анонимизации
~После окончания университета я начал свою карьеру в компании в сфере IT, где занимался разработкой программного обеспечения для мобильных устройств. В течение двух лет я работал над различными проектами, такими как разработка приложений для клиентов и оптимизация существующих продуктов. Мои обязанности включали проектирование, программирование и тестирование кода, а также взаимодействие с командой разработки и клиентами. Затем я перешел в компанию в сфере финансов, где занимался разработкой веб-приложений для внутреннего пользования. В течение трех лет я работал над различными проектами, такими как разработка систем управления данными и автоматизация бизнес-процессов. Мои обязанности включали проектирование, программирование и тестирование кода, а также взаимодействие с командой разработки и заинтересованными сторонами. В настоящее время я работаю в компании в сфере здравоохранения, где занимаюсь разработкой программного обеспечения для электронных медицинских карт. Мои обязанности включают разработку новых функций, поддержку существующих систем и взаимодействие с медицинскими работниками.
~После получения высшего образования в области информационных технологий я начал свою карьеру в компании в сфере телекоммуникаций, где занимался разработкой и внедрением программного обеспечения для управления сетью. В течение следующих трех лет я работал над различными проектами, связанными с оптимизацией производительности сети и повышением ее надежности. Затем я перешел в компанию в сфере финансовых услуг, где моя работа была направлена на разработку и внедрение систем обработки платежей. В этой роли мне приходилось работать с большим объемом данных и обеспечивать безопасность транзакций. В течение пяти лет я занимался разработкой новых функций для существующих систем и участвовал в проектах по модернизации инфраструктуры компании. В настоящее время я работаю в компании в сфере электронной коммерции, где моя текущая должность связана с разработкой и поддержкой веб-приложений. Моя работа включает в себя разработку пользовательского интерфейса, интеграцию с внешними сервисами и обеспечение безопасности данных пользователей.
~В течение лет я работал в компании в сфере , где занимался разработкой и внедрением программного обеспечения для автоматизации бизнес-процессов. В своей роли мне приходилось тесно сотрудничать с командой разработчиков, тестировщиков и менеджеров проекта, чтобы обеспечить своевременное и качественное выполнение проектов. Мои обязанности включали в себя разработку алгоритмов, проектирование архитектуры программных систем, написание кода на различных языках программирования и проведение тестирования. Я также принимал участие в процессах внедрения новых технологий и инструментов для повышения эффективности работы команды. В компании в сфере я занимался разработкой мобильных приложений для клиентов. Моя работа включала в себя проектирование пользовательского интерфейса, разработку функциональности приложения и тестирование на различных платформах. Я также сотрудничал с дизайнерами и маркетологами, чтобы обеспечить соответствие дизайна и функциональности приложения требованиям рынка. В течение лет я работал в компании в сфере , где занимался разработкой веб-приложений для корпоративных клиентов. Мои обязанности включали в себя разработку пользовательского интерфейса, создание бэкенд-систем и интеграцию с внешними сервисами. Я также принимал участие в процессах внедрения новых технологий и инструментов для повышения эффективности работы команды.

Все ответы довольно длинные, почему-то некоторые без маркера ~, во второй строке почему-то служебный текст


Почему -то снова попытка работать с кэшем, читая строки за пределами 10 - как показывается в выборке из лога:
2025-06-07 22:48:40 - pamola - INFO - Started: Reading EXPERIENCE.csv (total: 278 MB (approx))
2025-06-07 22:48:42 - pamola - INFO - Completed: Reading EXPERIENCE.csv in 1.26s (peak memory: 241.2MB, delta: +6.4MB)
2025-06-07 22:48:42 - pamola_core.utils.nlp.text_transformer - INFO - Using prompt template: custom_inline_template
2025-06-07 22:48:42 - pamola_core.utils.nlp.llm.client - INFO - Connected to LM Studio: gemma-2-9b-it-russian-function-calling (streaming: False)
2025-06-07 22:48:42 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:48:43 - pamola_core.utils.nlp.text_transformer - INFO - Connection test successful, response length: 61
2025-06-07 22:48:43 - pamola_core.utils.nlp.text_transformer - INFO - Successfully connected to Provider.LMSTUDIO
2025-06-07 22:48:43 - pamola_core.utils.nlp.text_transformer - INFO - TextTransformer initialized with model: gemma-2-9b-it-russian-function-calling, task_dir: D:\VK\_DEVEL\PAMOLA\DATA\processed\t_3LLM2
2025-06-07 22:48:43 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:48:49 - pamola_core.utils.nlp.llm.client - WARNING - Slow LLM response detected: 5.98s (threshold: 5.0s)
2025-06-07 22:48:49 - __main__ - WARNING - Model returned error response during test
2025-06-07 22:48:49 - __main__ - WARNING - LLM connection test had warnings - continuing with processing
2025-06-07 22:48:49 - pamola_core.utils.nlp.text_transformer - INFO - Limiting to 10 records
2025-06-07 22:48:49 - pamola_core.utils.nlp.text_transformer - INFO - Processing 10 records
2025-06-07 22:48:49 - pamola - INFO - Started: Processing experience_descriptions (total: 10 records)
2025-06-07 22:48:49 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:48:50 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:48:50 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:48:55 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:48:59 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:49:04 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:49:08 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:49:14 - pamola_core.utils.nlp.llm.client - WARNING - Slow LLM response detected: 5.07s (threshold: 5.0s)
2025-06-07 22:49:14 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:49:24 - pamola_core.utils.nlp.llm.client - WARNING - Slow LLM response detected: 10.19s (threshold: 5.0s)
2025-06-07 22:49:24 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:49:30 - pamola_core.utils.nlp.llm.client - WARNING - Slow LLM response detected: 6.55s (threshold: 5.0s)
2025-06-07 22:49:30 - pamola_core.utils.nlp.llm.client - WARNING - Model doesn't accept generation parameters: LLM.respond() got an unexpected keyword argument 'temperature'
2025-06-07 22:49:35 - pamola - INFO - Completed: Processing experience_descriptions in 45.77s (peak memory: 332.9MB, delta: +2.5MB)
2025-06-07 22:49:35 - pamola_core.utils.nlp.text_transformer - INFO - Processing completed: 8/10 successful
2025-06-07 22:49:35 - pamola_core.utils.nlp.text_transformer - INFO - Errors: 2
2025-06-07 22:49:35 - pamola_core.utils.nlp.text_transformer - INFO - Cache hit rate: 0.0%
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=1: Processing failed, value unchanged
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=5: Processing failed, value unchanged
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=11: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=12: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=13: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=14: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=15: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=16: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=17: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=18: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=19: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=20: Processing failed, no result
2025-06-07 22:49:35 - __main__ - ERROR - Record ID=21: Processing failed, no result

### Почему «странное» поведение — краткая диагностика

| Cимптом                                                                                     | Корневая причина                                                                                                                                                                                                                             | Как исправить                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. «Жду исходный текст…» / «Send me original text…»**<br>Попадает в выход без `~`-маркера | *Служебный ответ не отфильтрован.*<br>`TextTransformer._postprocess()` возвращает `success=True`, потому что по паттернам (`detect_service_answer`) не ловится эта формулировка ⇒ строка записывается как обычный текст, маркер не ставится. | • Расширить паттерны в **post-filter** (см. пункт A).<br>• При `response_type==SERVICE` помечать `success=False`, чтобы попало в колонку error или NaN.                                       |
| **2. Запросов 10, а в логе error-ID до 46**                                                 | Блок «вывод ошибок» перебирает **все** строки датафрейма и сравнивает `initial_target_values` с текущими, даже если эти строки не проходили через LLM.                                                                                       | • Перед циклом собрать `processed_indices` из `transformer.processed_row_indices` (или вернуть из `process_dataframe`) и проверять **только их**.                                             |
| **3. Warnings «LLM.respond() got an unexpected keyword argument 'temperature'»**            | LM Studio REST-режим принимает температуру и т.д. только в поле `args`.  Класс `client.LLM.request()` передаёт их на верхний уровень, поэтому метод отбрасывает.                                                                             | • В `GenerationConfig.to_api_params()` для `Provider.LMSTUDIO` вложить sampling-параметры под ключ `"args"`. *Иначе модель всегда работает на дефолте (`T=0.7`) и отвечает более «болтливо».* |
| **4. Лимит 10 строк, а пишется весь CSV (165 450 rows)**                                    | Ограничение делается внутри `transformer.process_dataframe`, но **весь** исходный df передаётся дальше на `write_dataframe_to_csv`, поэтому сохраняется файл целиком.                                                                        | • После обработки берите `result_df.loc[processed_indices]` если нужен мини-вывод.<br>• Или добавьте флаг `truncate_output` (по-умолчанию False).                                             |
| **5. Длинные ответы (нет обрезки)**                                                         | В шаблоне нет «ограничения длины» и\или Post-процесс не применяет `max_tokens_output`.                                                                                                                                                       | • В `GenerationConfig.merge_with_model_defaults()` поставьте `max_tokens` ≤ 256 для всех alias-ов, кроме «Prod».<br>• В Post-процессоре добавьте `truncate_to_sentence(text, 3 sentences)`.   |
| **6. Cache hit 0 %, зато «чтение строк за пределами 10»**                                   | Кеш-ключ строится по `text[:100]` → для длинного df с повторяющимися фрагментами ключи коллидируют; когда запись не попала в кеш из-за ошибки, при следующем запуске она снова пойдёт в LLM.                                                 | • Генерируйте хэш SHA-1 от нормализованного текста (lower-strip-spaces).<br>• При `success=False` **не записывайте** значение в кеш.                                                          |

---

## A. Усилить фильтр служебных ответов

```python
# pamola_core/utils/nlp/postprocessor.py
SERVICE_PATTERNS += [
    r"^\s*(жду|отправьте|send|provide).*исходн(ый|ого)\s+текст",  # RU + EN
    r"^\s*i am ready.*but need.*text",                            # generic
]
EMPTY_PATTERNS += [
    r'(?i)^(анонимизирован[а-я]*|обезличен[а-я]*|result|text)\s*:\s*["\']?\s*$'
]
```

*Если `re.search()` сработал → `response_type=SERVICE`, `success=False`, `text=''`.*

---

## B. Правильная передача sampling-параметров для LM Studio

```python
# GenerationConfig.to_api_params
if provider == Provider.LMSTUDIO:
    params = {
        "model": self.model_name,
        "args": {                # <<<<<<
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "stop": self.stop_sequences,
        },
        "stream": self.stream,
    }
```

Иначе LM Studio использует дефолты, а API ругается на лишние поля.

---

## C. Исправить счётчик ошибок

```python
# process_with_error_handling
processed_mask = transformer.last_processed_mask   # верните из TextTransformer
for idx in processed_mask.index[processed_mask]:
    ...
```

или возвращайте `error_df` отдельно, чтобы не вычислять вручную.

---

## D. Срез для вывода

```python
if config['max_records']:
    save_df = result_df.head(config['max_records'])   # только новые
else:
    save_df = result_df
write_dataframe_to_csv(save_df, output_path, ...)
```

---

### TL;DR

* **Главное**: расширьте паттерны SERVICE/EMPTY и не ставьте `success=True` для таких ответов.
* Используйте `args:{…}` для LM Studio, чтобы уйти от warning-ов и болтовни.
* Логи ошибок нужно строить только по реально обработанным индексам.
* Ограничивайте вывод и `max_tokens`, иначе модель «заливает» текст.

После этих четырёх правок скрипт перестанет:

1. сохранять «Жду текст…» без маркера,
2. ругаться на `temperature`,
3. репортить сотни «failed» для строк, которые вообще не трогались,
4. выдавать 600-символьные «поэмы» вместо лаконичных описаний.

