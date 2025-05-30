# Спецификация модулей операций пакета fake_data

## 1. Введение

Данная спецификация определяет требования к реализации модулей операций и генераторов в рамках пакета `pamola_core.fake_data` для генерации синтетических данных с сохранением статистических свойств оригинальных данных.
### 1.1 Типовые операции и генераторы
Операция - модуль в составе pamola_core/fake_data/operations/, каждая операция  получает на вход  от задачи (пользовательского скрипта за пределами пакета pamola core) необходимые данные:
	- **источник данных** (ссылку на файл - в этом случае операция сама читает данные и ей нужны параметры - разделитель, кодировка, текстовый квалификатор; dataframe - если операция получает данные уже из  ранее сделанных операций или задач/пользовательский скрипт уже прочитали данные), а также имя набора данных -  если он будет сохраняться;
	- **{task_dir}** - ссылку на директорию задачи (пользовательского скрипта), куда выводится результат в случае его сохранения.
	- **поля**: во первых, если производится замена исходного поля, то поле с именем, во-вторых, вспомогательные поля - такие как указание пола. (как правило пол для генерации имен, для генерации e-mail - поле фамилия и имя пользователя, для адреса - регион) и т.п. Эти дополнительные данные уникальны для операции)
	- **флаги для управления режимом генерации**: надо ли вставлять данные в исходный dataframe или добавлять отдельным столбцом (в этом случае по умолчанию используется префикс _ и исходное наименование столбца с именем), во-вторых, надо ли сохранять  карту замен, использовать случайную генерацию или использовать псевдо-рандомную функци. - PRF, национальность (en,ru, vn), указания для регистра и последовательности имен (FirstName-MiddleName-LastName; LastName-Firstname, ...), использование пропущенных значений как самостоятельных значений, необходимость использвать режим 1-to-1 (все имена John заменяются на Bill стандартно, аналогично фамилии, при этом может указываться общая политика обработки транзитивности/коллизий), используемый префикс для столбца, если данные обогащаются
	- **ссылки на словарь данных для генерации**: набор путей к словарям для генерации имен, фамилий, отчеств и полных имен. Словари могут отсутствовать. В этом случае используется встроенный словарь или он ищется 
	- **ключи** для шифрования или использования PRF 
Таким образом  выбор режима  действия операции выбирается пользователем на этапе вызова скрипта.  При этом пользователь выбирает и саму операцию - генерация имен, генерация электронных адресов, генерация организаций, генерация телефонов и т.п.
Операция определенного типа напрямую связана с  одним или несколькими генераторами, задачи которого вернуть данные, которые операция вставляет в целевое поле.
Генераторы могут быть для одного поля:
   	- случайный выбор на основании словаря с параметром (внутреннего или внешнего). Генератор получает флаг принудительно использовать внутренний (встроенный словарь) для выбора значений или вынужден это сделать, если словарь не передан или не находится в стандартном месте. Если при этом задан флаг сохранения карты замен, то карта сохраняется в {task_dir}\maps\{operation_name}.json (надо учесть, что задача может иметь много операций). Если задан флаг и ключ шифрования, то такой вывод шифруется  для защиты промежуточных данных. При включенном флаге кроме сохранения возможности восстановления карта используется для гарантий, что одинаковые данные отражаются в одинаковые данные.
	- выбор на основании тех же словарей через PRNG - в этом случае  установлен соответствующий флаг использовать детерминированную генерацию. Операция получает от задачи и передает в генератор контекст (соль) и ключ для формирования псевдо случайного числа по одному из также установленному алгоритму, после чего диапазон значения словаря отражается  на значение  генератора, что приводит к постоянному  выбору одних значений на другое. В этом случае всегда одни данные будут отражаться в другие данные одинаковым образом.

	- Пустые значения могут по флагу обрабатываться как самостоятельные значения или игнорироваться - то есть генерация в данной строке не осуществляется.

### 1.2 Общая структура пакета

```
pamola_core/fake_data/
├── __init__.py
├── operations/                # <-- сюда записываем операторы (name_op.py,...)
├── commons/                   # Общие утилиты
│   ├── __init__.py
│   ├── base.py                # Базовые абстрактные классы (имеется)
│   ├── operations.py          # Базовые классы операций (имеется)
│   ├── dict_helpers.py        # Работа со словарями  (имеется)
│   ├── prgn.py                # Псевдо-случайная генерация
│   ├── metrics.py             # Сбор и анализ метрик
│   ├── validators.py          # Валидаторы разных типов данных
│   ├── string_utils.py        # Утилиты для работы со строками ( реализовтаь)
│   ├── language_utils.py      # Язык  и транслит <-(использ pamola_core.utils.nlp...)
│   ├── contact_utils.py       # Утилиты для контактных данных (реализовать)
│   └── geo_utils.py           # Географические утилиты (откладываем за MVP)
├── mapping_store.py           # Хранилище маппингов (реализовано)
├── generators/                # Генераторы разных типов данных (<-- реализовать)
│   ├── __init__.py
│   ├── base_generator.py      # Базовый класс генератора (<-- реализовать)
│   ├── name.py                # Генератор имен (<-- реализовать)
│   ├── email.py               # Генератор email (<-- реализовать)
│   ├── phone.py               # Генератор телефонов (<-- реализовать)
│   ├── address.py             # Генератор адресов (<-- реализовать)
│   └── organization.py        # Генератор организаций (<-- реализовать)
├── mappers/                   # Класcы маппинга (реализовано)
│   ├── __init__.py
│   ├── one_to_one.py          # Маппинг один-к-одному (реализовано)
│   └── transitivity_handler.py # Обработчик транзитивностей (реализовано)
└── dictionaries/              # Встроенные словари
    ├── __init__.py
    ├── names.py               # Словари имен (реализовано)
    ├── domains.py             # Словари email-доменов (реализовано)
    ├── phones.py              # Словари телефонных кодов (реализо (реализовано)
    ├── addresses.py           # Словари адресных компонентов (реализовано)
    └── organizations.py       # Словари организаций (реализовано)
```

## 2. Базовые генераторы

### 2.1. BaseGenerator (pamola_core/fake_data/generators/base_generator.py)

Абстрактный базовый класс, определяющий единый интерфейс для всех генераторов.

#### Методы:

- `__init__(self, config=None)` - инициализация с опциональной конфигурацией
- `generate(self, count, **params)` - генерация указанного количества значений
- `generate_like(self, original_value, **params)` - генерация значения, похожего на оригинальное
- `transform(self, values, **params)` - преобразование списка исходных значений в синтетические
- `validate(self, value)` - проверка корректности значения

#### Параметры `**params` для всех генераторов:

- `seed` - seed для детерминированной генерации (если применяется PRGN)
- `context_salt` - соль для детерминированной генерации
- `avoid_collision` - избегать значений, уже использованных для других оригиналов
- `preserve_format` - сохранять формат оригинального значения
- `language` - язык для генерации (en, ru, vn)

### 2.2. NameGenerator (pamola_core/fake_data/generators/names.py)

Генератор имен, фамилий и отчеств.

#### Параметры конфигурации:

- `language` - язык генерируемых имен (ru, en, vn)
- `gender` - пол (M, F, None для нейтрального)
- `format` - формат вывода имени:
    - `FML` - FirstName MiddleName LastName
    - `FL` - FirstName LastName
    - `LF` - LastName FirstName
    - `LFM` - LastName FirstName MiddleName
    - `F_L` - FirstName_LastName
    - Регистры: верхний регистр (_FML), нижний регистр (fml_), только первая буква заглавная (Fml)
- `use_faker` - использовать ли библиотеку faker (bool)
- `case` - регистр (upper, lower, title)
- `gender_from_name` - определять ли пол по имени (если не указан явно)
- `f_m_ratio` - соотношение мужчин/женщин при случайной генерации (например, 0.6)
- `dictionaries` - пути к внешним словарям (см. ниже)

#### Поиск словарей:

1. По указанным путям в `dictionaries`
2. По шаблону `{language}_{gender}_{name_type}.txt` в директории `{DATA}/external_dictionaries/fake/`
3. Встроенные словари из `pamola_core.fake_data.dictionaries.names`

#### Форматы имен в словарях:

- Текстовые файлы, одна строка - одно имя
- Кодировка UTF-8
- Словари могут быть разделены по полу: `ru_m_first_names.txt`, `ru_f_first_names.txt`
- Или общие: `ru_last_names.txt`

#### Специфические методы:

- `generate_first_name(self, gender=None, language=None)` - генерация имени
- `generate_last_name(self, gender=None, language=None)` - генерация фамилии
- `generate_middle_name(self, gender=None, language=None)` - генерация отчества
- `generate_full_name(self, gender=None, language=None, format=None)` - генерация полного имени
- `detect_gender(self, name, language=None)` - определение пола по имени
- `parse_full_name(self, full_name, language=None)` - разбор полного имени на компоненты

### 2.3. EmailGenerator (pamola_core/fake_data/generators/email.py)

Генератор email-адресов.

#### Параметры конфигурации:

- `domains` - список используемых доменов или путь к файлу с доменами
- `format` - формат локальной части email:
    - `name_surname` - Имя.Фамилия@домен
    - `surname_name` - Фамилия.Имя@домен
    - `nickname` - СлучайныйНикнейм[разделитель][число]@домен
    - `existing_domain` - Использует домен из исходного email, генерируя новую локальную часть
- `format_ratio` - соотношение форматов при смешанной генерации (словарь: `{'name_surname': 0.4, 'nickname': 0.6}`)
- `validate_source` - проверять ли исходный email на корректность (bool)
- `handle_invalid_email` - что делать с некорректными email:
    - `generate_new` - генерировать новый случайный email
    - `keep_empty` - оставлять поле пустым
    - `generate_with_default_domain` - генерировать с доменом по умолчанию
- `nicknames_dict` - путь к словарю никнеймов

#### Источники доменов:

1. Указанный в параметрах список
2. Указанный в параметрах файл
3. Встроенный список из `pamola_core.fake_data.dictionaries.domains`

#### Специфические методы:

- `generate_username(self, first_name=None, last_name=None, format=None)` - генерация имени пользователя
- `generate_domain(self, existing_domain=None)` - генерация или выбор домена
- `extract_domain(self, email)` - извлечение домена из email-адреса
- `validate_email(self, email)` - проверка корректности email
- `generate_nickname(self, length=None)` - генерация случайного никнейма

### 2.4. PhoneGenerator (pamola_core/fake_data/generators/phone.py)

Генератор телефонных номеров.

#### Параметры конфигурации:

- `country_codes` - используемые коды стран:
    - Список: `['+1', '+44', '+7']`
    - Словарь с весами: `{'+1': 0.6, '+44': 0.3, '+7': 0.1}`
- `operator_codes_dict` - путь к словарю кодов операторов
- `format` - формат вывода номера, например: `+CC (AAA) BBB-BB-BB` Где:
    - `CC` - код страны
    - `AAA` - код оператора
    - `BBB-BB-BB` - случайные цифры
- `validate_source` - проверять ли исходный номер
- `handle_invalid_phone` - обработка некорректных номеров:
    - `generate_new` - генерировать новый номер
    - `keep_empty` - оставлять поле пустым
    - `generate_with_default_country` - генерировать с кодом страны по умолчанию

#### Формат словаря кодов операторов:

```
+1,201,917,646
+44,74,77,78
+7,903,916,926
```

Где первое значение в строке - код страны, остальные - коды операторов.

#### Специфические методы:

- `generate_country_code(self)` - выбор кода страны
- `generate_operator_code(self, country_code)` - выбор кода оператора
- `generate_phone_number(self, country_code=None, operator_code=None)` - генерация полного номера
- `format_phone(self, digits, country_code, format=None)` - форматирование номера
- `validate_phone(self, phone)` - проверка корректности номера

### 2.5. AddressGenerator (pamola_core/fake_data/generators/address.py)

Генератор адресов.

#### Параметры конфигурации:

- `region` - код региона (US-CA, RU-MOS, CA-ON и т.д.)
- `format` - формат вывода адреса с плейсхолдерами:
    
    - `{city}` - название города
    - `{street_type}` - тип улицы (улица, проспект, бульвар)
    - `{street_name}` - название улицы
    - `{house_number}` - номер дома
    - `{apartment_number}` - номер квартиры
    - `{postal_code}` - почтовый индекс
    
    Пример: `{street_type} {street_name}, д. {house_number}, кв. {apartment_number}, {city}, {postal_code}`
- `use_postal_code` - включать ли почтовый индекс (bool)
- `generate_apartment` - генерировать ли номер квартиры (bool)
- `house_number_format` - формат номера дома (например: `##`, `#-#`)
- `apartment_range` - диапазон номеров квартир (например: `(1, 200)`)
- `dictionaries` - словарь путей к файлам для компонентов адреса:
    
    ```python
    {  'US-CA': {    'cities': 'path/to/us_ca_cities.txt',    'street_types': 'path/to/us_ca_street_types.txt',    'street_names': 'path/to/us_ca_street_names.txt',    'postal_codes': 'path/to/us_ca_postal_codes.txt'  },  'RU-MOS': {    'cities': 'path/to/ru_mos_cities.txt',    'street_types': 'path/to/ru_mos_street_types.txt',    'street_names': 'path/to/ru_mos_street_names.txt'  }}
    ```
    

#### Поиск словарей адресов:

1. По указанным путям в `dictionaries`
2. По шаблону `[REGION]_component.txt` в директории `{DATA}/external_dictionaries/fake/addresses/`
3. Встроенные словари из `pamola_core.fake_data.dictionaries.addresses`

#### Специфические методы:

- `generate_city(self, region)` - генерация названия города
- `generate_street_name(self, region)` - генерация названия улицы
- `generate_street_type(self, region)` - генерация типа улицы
- `generate_house_number(self, format)` - генерация номера дома
- `generate_apartment_number(self, range)` - генерация номера квартиры
- `generate_postal_code(self, region, city=None)` - генерация почтового индекса
- `format_address(self, components, format)` - форматирование адреса из компонентов

### 2.6. OrganizationGenerator (pamola_core/fake_data/generators/organization.py)

Генератор названий организаций.

#### Параметры конфигурации:

- `organization_type` - тип организации:
    - `general` - общий список названий
    - `educational` - учебные заведения
    - `manufacturing` - производственные компании
    - `government` - государственные учреждения
    - `industry` - по отраслям (IT, finance, healthcare и т.д.)
- `dictionaries` - словарь путей к файлам для разных типов организаций
    
    ```python
    {  'general': 'path/to/general_organizations.txt',  'educational': 'path/to/educational_institutions.txt',  'manufacturing': 'path/to/manufacturing_companies.txt',  'it': 'path/to/it_companies.txt'}
    ```
    
- `prefixes` - словарь путей к файлам с префиксами названий
- `suffixes` - словарь путей к файлам с суффиксами названий
- `add_prefix_probability` - вероятность добавления префикса (0.0-1.0)
- `add_suffix_probability` - вероятность добавления суффикса (0.0-1.0)

#### Поиск словарей организаций:

1. По указанным путям в `dictionaries`, `prefixes`, `suffixes`
2. По шаблону `organization_type.txt` в директории `{DATA}/external_dictionaries/fake/organizations/`
3. Встроенные словари из `pamola_core.fake_data.dictionaries.organizations`

#### Специфические методы:

- `generate_organization_name(self, org_type=None)` - генерация названия организации
- `add_prefix(self, name, org_type=None)` - добавление префикса к названию
- `add_suffix(self, name, org_type=None)` - добавление суффикса к названию
- `validate_organization_name(self, name)` - проверка корректности названия

## 3. Операции

### 3.1. Иерархия операций в пакете

```
pamola_core.utils.ops.op_base.BaseOperation
    └── pamola_core.fake_data.commons.operations.BaseOperation
        └── pamola_core.fake_data.commons.operations.FieldOperation
            └── pamola_core.fake_data.commons.operations.GeneratorOperation
                ├── pamola_core.fake_data.operations.name_operations.NameOperation
                ├── pamola_core.fake_data.operations.email_operations.EmailOperation
                ├── pamola_core.fake_data.operations.phone_operations.PhoneOperation
                ├── pamola_core.fake_data.operations.address_operations.AddressOperation
                └── pamola_core.fake_data.operations.organization_operations.OrganizationOperation
```

**Важно:** Базовые классы операций (`BaseOperation`, `FieldOperation`, `GeneratorOperation`) определены в модуле `pamola_core.fake_data.commons.operations`, а не в `pamola_core.fake_data.operations` как указано ранее.

### 3.2. BaseOperation (pamola_core/fake_data/commons/operations.py)

Базовый класс для всех операций пакета fake_data, наследуемый от pamola_core.utils.ops.op_base.BaseOperation.

#### Методы:

- `__init__(self, name="base_operation", description="Base operation for fake data generation")` - инициализация
- `execute(self, data_source, task_dir, reporter, **kwargs)` - выполнение операции
- `run(self)` - запуск операции (наследуется от op_base)

### 3.3. FieldOperation (pamola_core/fake_data/commons/operations.py)

Базовый класс для операций над полями данных.

#### Параметры:

- `field_name` - имя обрабатываемого поля
- `mode` - режим операции ("REPLACE" или "ENRICH"):
    - "REPLACE" - замена значений в исходном поле
    - "ENRICH" - создание нового поля с префиксом
- `output_field_name` - имя выходного поля (только для mode="ENRICH")
- `null_strategy` - стратегия обработки NULL-значений:
    - "PRESERVE" - сохранять NULL в выходных данных
    - "REPLACE" - заменять NULL на синтетические значения
    - "EXCLUDE" - исключать строки с NULL
    - "ERROR" - генерировать ошибку при обнаружении NULL
- `batch_size` - размер пакета обработки (по умолчанию 10000)
- `column_prefix` - префикс для создаваемой колонки (по умолчанию "_")

#### Методы:

- `__init__(self, field_name, mode="REPLACE", output_field_name=None, batch_size=10000, null_strategy="PRESERVE", column_prefix="_")` - инициализация
- `execute(self, data_source, task_dir, reporter, **kwargs)` - выполнение операции
- `_load_data(self, data_source)` - загрузка данных из источника
- `_save_result(self, df, output_dir)` - сохранение результата
- `_save_metrics(self, metrics, task_dir)` - сохранение метрик
- `_save_mapping(self, mapping, task_dir)` - сохранение карты соответствий
- `_collect_metrics(self, original_df, processed_df)` - сбор метрик операции
- `process_batch(self, batch)` - обработка одного пакета данных
- `preprocess_data(self, df)` - предварительная обработка данных
- `postprocess_data(self, df)` - постобработка данных
- `handle_null_values(self, df)` - обработка NULL-значений

### 3.4. GeneratorOperation (pamola_core/fake_data/commons/operations.py)

Операция с использованием генератора.

#### Параметры:

- Наследуемые от FieldOperation
- `generator` - экземпляр BaseGenerator
- `consistency_mechanism` - механизм согласованности:
    - "mapping" - явное хранение соответствий
    - "prgn" - псевдо-случайная генерация
- `mapping_store_path` - путь к хранилищу маппингов
- `id_field` - поле для идентификации записей (для mapping)
- `generator_params` - словарь параметров для генератора
- `collect_metrics` - собирать ли метрики
- `key` - ключ для шифрования/PRGN
- `context_salt` - соль для PRGN
- `save_mapping` - сохранять ли карту соответствий
- `dictionary_paths` - пути к словарям

#### Методы:

- `__init__(self, field_name, generator, mode="REPLACE", output_field_name=None, null_strategy="PRESERVE", batch_size=10000, consistency_mechanism="prgn", **kwargs)` - инициализация
- `process_batch(self, batch)` - обработка пакета данных с использованием генератора
- `process_value(self, value, **params)` - обработка одного значения
- `collect_metrics(self, original_df, processed_df)` - сбор метрик качества генерации
- `_initialize_consistency_mechanism(self)` - инициализация механизма согласованности
- `_setup_generator(self, generator_params)` - настройка генератора
- `_save_artifacts(self, task_dir)` - сохранение артефактов операции

### 3.5. NameOperation (pamola_core/fake_data/operations/name_op.py)

Операция для генерации имен.

#### Параметры:

- Наследуемые от GeneratorOperation
- `language` - язык генерируемых имен (ru, en, vn)
- `gender_field` - поле, определяющее пол
- `gender_from_name` - определять ли пол по имени (bool)
- `format` - формат вывода имени (FML, LF, LFM и т.д.)
- `f_m_ratio` - соотношение полов при отсутствии явного указания
- `use_faker` - использовать ли библиотеку faker (bool)
- `dictionaries` - список путей к словарям имен, фамилий, отчеств

#### Пример использования:

```python
name_op = NameOperation(
    field_name="full_name",
    mode="ENRICH",
    language="ru",
    gender_field="gender",
    format="LFM",
    consistency_mechanism="mapping",
    save_mapping=True,
    column_prefix="fake_"
)
result = name_op.execute(df, Path("./task_dir"), reporter)
```

### 3.6. EmailOperation (pamola_core/fake_data/operations/email_op.py)

Операция для генерации email-адресов.

#### Параметры:

- Наследуемые от GeneratorOperation
- `domains` - список доменов
- `format` - формат email-адреса
- `format_ratio` - соотношение форматов
- `validate_source` - проверять ли исходный email (bool)
- `handle_invalid_email` - стратегия обработки некорректных email
- `nicknames_dict` - путь к словарю никнеймов

#### Пример использования:

```python
email_op = EmailOperation(
    field_name="email",
    mode="REPLACE",
    domains=["example.com", "test.org", "mail.ru"],
    format="name_surname",
    validate_source=True,
    handle_invalid_email="generate_new",
    consistency_mechanism="prgn",
    key="my-secret-key",
    context_salt="email-salt"
)
result = email_op.execute(df, Path("./task_dir"), reporter)
```

### 3.7. PhoneOperation (pamola_core/fake_data/operations/phone_op.py)

Операция для генерации телефонных номеров.

#### Параметры:

- Наследуемые от GeneratorOperation
- `country_codes` - используемые коды стран
- `operator_codes_dict` - путь к словарю кодов операторов
- `format` - формат телефонного номера
- `validate_source` - проверять ли исходный номер (bool)
- `handle_invalid_phone` - стратегия обработки некорректных номеров

#### Пример использования:

```python
phone_op = PhoneOperation(
    field_name="phone",
    mode="ENRICH",
    country_codes={"+7": 0.8, "+1": 0.2},
    format="+CC (AAA) BBB-BB-BB",
    validate_source=True,
    handle_invalid_phone="generate_with_default_country",
    consistency_mechanism="mapping",
    output_field_name="fake_phone"
)
result = phone_op.execute(df, Path("./task_dir"), reporter)
```

### 3.8. AddressOperation (pamola_core/fake_data/operations/address_op.py)

Операция для генерации адресов.

#### Параметры:

- Наследуемые от GeneratorOperation
- `region` - код региона
- `format` - формат адреса
- `use_postal_code` - включать ли почтовый индекс (bool)
- `generate_apartment` - генерировать ли номер квартиры (bool)
- `house_number_format` - формат номера дома
- `apartment_range` - диапазон номеров квартир
- `dictionaries` - словарь путей к файлам для компонентов адреса

#### Пример использования:

```python
address_op = AddressOperation(
    field_name="address",
    mode="REPLACE",
    region="RU-MOS",
    format="{street_type} {street_name}, д. {house_number}, {city}",
    house_number_format="###",
    consistency_mechanism="prgn",
    key="address-key"
)
result = address_op.execute(df, Path("./task_dir"), reporter)
```

### 3.9. OrganizationOperation (pamola_core/fake_data/operations/organization_op.py)

Операция для генерации названий организаций.

#### Параметры:

- Наследуемые от GeneratorOperation
- `organization_type` - тип организации
- `dictionaries` - словарь путей к файлам для разных типов организаций
- `prefixes` - словарь путей к файлам с префиксами названий
- `suffixes` - словарь путей к файлам с суффиксами названий
- `add_prefix_probability` - вероятность добавления префикса (0.0-1.0)
- `add_suffix_probability` - вероятность добавления суффикса (0.0-1.0)

#### Пример использования:

```python
org_op = OrganizationOperation(
    field_name="company",
    mode="ENRICH",
    organization_type="educational",
    add_suffix_probability=0.5,
    add_prefix_probability=0.3,
    consistency_mechanism="mapping",
    output_field_name="fake_company"
)
result = org_op.execute(df, Path("./task_dir"), reporter)
```

## 4. Артефакты операций

### 4.1. Структура директории задачи

Каждая операция сохраняет артефакты в директорию задачи по следующей структуре:

```
task_dir/
├── output/
│   └── output_data.csv           # Выходные данные (если SAVE_DATA=True)
├── metrics/
│   └── {operation_name}.json     # Метрики операции
├── maps/
│   └── {operation_name}.json     # Карта соответствий (если save_mapping=True)
└── visualizations/
    ├── {operation_name}_dist.png # Визуализация распределения
    └── {operation_name}_stats.png # Визуализация статистик
```

### 4.2. Формат метрик

Метрики сохраняются в формате JSON со следующей структурой:

```json
{
  "original_data": {
    "total_records": 10000,
    "unique_values": 500,
    "null_count": 50,
    "value_counts": {"valueA": 100, "valueB": 200, ...},
    "length_stats": {"min": 2, "max": 30, "mean": 15.5, "median": 14}
  },
  "generated_data": {
    "total_records": 10000,
    "unique_values": 550,
    "null_count": 50,
    "value_counts": {"valueX": 90, "valueY": 180, ...},
    "length_stats": {"min": 3, "max": 32, "mean": 16.2, "median": 15}
  },
  "quality_metrics": {
    "distribution_similarity_score": 0.92,
    "uniqueness_preservation": 0.98,
    "format_compliance": 0.99,
    "type_specific_metrics": {...}
  },
  "transformation_metrics": {
    "null_values_replaced": 0,
    "total_replacements": 9950,
    "replacement_strategy": "mapping",
    "mapping_collisions": 5,
    "reversibility_rate": 1.0
  },
  "performance": {
    "generation_time": 2.5,
    "records_per_second": 4000,
    "memory_usage_mb": 85,
    "dictionary_load_time": 0.3
  }
}
```

### 4.3. Формат карты соответствий

Карты соответствий сохраняются в формате JSON:

```json
{
  "field_name": "full_name",
  "consistency_mechanism": "mapping",
  "created_at": "2023-05-01T10:30:45",
  "mappings": [
    {"id": 1, "original": "John Smith", "synthetic": "Robert Johnson"},
    {"id": 2, "original": "Jane Doe", "synthetic": "Mary Williams"},
    {"id": 3, "original": "Ivan Petrov", "synthetic": "Sergey Ivanov"}
  ],
  "statistics": {
    "total_mappings": 3,
    "unique_originals": 3,
    "unique_synthetics": 3
  }
}
```

## 5. Общие требования

### 5.1. Совместимость с инфраструктурой PAMOLA.CORE

- Все операции должны наследоваться от pamola_core.utils.ops.op_base.BaseOperation
- Все операции должны возвращать экземпляр pamola_core.utils.ops.op_result.OperationResult
- Для файловых операций использовать pamola_core.utils.io
- Для логирования использовать pamola_core.utils.logging
- Для отслеживания прогресса использовать pamola_core.utils.progress
- Для визуализации использовать pamola_core.utils.visualization

### 5.2. Параметры по умолчанию

- Размер пакета (batch_size): 10000 записей
- Стратегия для NULL: "PRESERVE" (сохранять NULL)
- Режим по умолчанию: "ENRICH" (а не "REPLACE")
- Язык по умолчанию: "en"
- Механизм согласованности: "prgn" (как более экономичный)
- Префикс колонки (column_prefix): "_"

### 5.3. Регистрация операций

Все операции должны быть зарегистрированы в системе с помощью декоратора `@register`:

```python
from pamola_core.utils.ops.op_registry import register

@register()
class NameOperation(GeneratorOperation):
    name = "name_generator"
    description = "Generates synthetic names"
    category = "fake_data"
    # ...
```

### 5.4. Обработка ошибок

- Критические ошибки (неверные параметры, недоступные ресурсы):
    - Вызов исключения с информативным сообщением
    - Логирование с уровнем ERROR
    - Возврат OperationResult с флагом success=False и сообщением об ошибке
- Некритические ошибки (ошибки в отдельных строках):
    - Логирование с уровнем WARNING
    - Продолжение обработки
    - Включение информации об ошибках в метрики
- Использовать стандартизированное логирование:
    - DEBUG - подробная информация для отладки
    - INFO - общая информация о ходе операции
    - WARNING - предупреждения, не останавливающие операцию
    - ERROR - ошибки, останавливающие операцию