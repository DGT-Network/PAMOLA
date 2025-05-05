"""
Модуль конфигурации для системы отчетности.

Функции для работы с конфигурацией проекта и настройками отчетов.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Настройка логирования
logger = logging.getLogger(__name__)

# Значения по умолчанию
DEFAULT_CONFIG = {
    "project": {
        "name": "HeadHunter Resume Anonymization",
        "description": "Анализ и анонимизация данных резюме для обезличивания",
        "version": "1.0.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "updated": datetime.now().strftime("%Y-%m-%d")
    },
    "reporting": {
        "default_template": "default",
        "auto_open": True,
        "include_resources": False
    },
    "task_flow": {
        "dependencies": {}
    }
}


def get_config_path() -> Path:
    """
    Получает путь к файлу конфигурации проекта.

    Returns:
    --------
    Path
        Путь к файлу конфигурации
    """
    # Сначала ищем в переменной окружения
    if os.environ.get("HHR_CONFIG_PATH"):
        return Path(os.environ["HHR_CONFIG_PATH"])

    # Ищем в стандартных местах
    config_search_paths = [
        Path("hhr_config.json"),
        Path("configs/hhr_config.json"),
        Path("config/hhr_config.json"),
        Path(os.path.expanduser("~")) / ".hhr" / "config.json"
    ]

    # Проверяем текущую директорию проекта
    try:
        from pamola_core.utils.io import get_data_repository
        repo_dir = get_data_repository()
        config_search_paths.insert(0, repo_dir / "configs" / "hhr_config.json")
    except (ImportError, Exception):
        logger.warning("Не удалось получить директорию репозитория данных")

    # Ищем конфигурацию в указанных местах
    for path in config_search_paths:
        if path.exists():
            return path

    # Возвращаем путь по умолчанию, если ничего не найдено
    return Path("hhr_config.json")


def load_config() -> Dict[str, Any]:
    """
    Загружает конфигурацию проекта.

    Returns:
    --------
    Dict[str, Any]
        Словарь с конфигурацией проекта
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.warning(f"Файл конфигурации не найден: {config_path}. Используются настройки по умолчанию.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Добавляем недостающие разделы из конфигурации по умолчанию
        for section, defaults in DEFAULT_CONFIG.items():
            if section not in config:
                config[section] = defaults
            elif isinstance(defaults, dict):
                for key, value in defaults.items():
                    if key not in config[section]:
                        config[section][key] = value

        return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        return DEFAULT_CONFIG


def save_config(config: Dict[str, Any]) -> bool:
    """
    Сохраняет конфигурацию проекта.

    Parameters:
    -----------
    config : Dict[str, Any]
        Конфигурация для сохранения

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    config_path = get_config_path()

    try:
        # Обновляем дату изменения
        if "project" in config:
            config["project"]["updated"] = datetime.now().strftime("%Y-%m-%d")

        # Создаем директорию, если она не существует
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2) # type: ignore

        logger.info(f"Конфигурация сохранена в {config_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении конфигурации: {e}")
        return False


def update_project_metadata(name: Optional[str] = None,
                            description: Optional[str] = None,
                            version: Optional[str] = None) -> bool:
    """
    Обновляет метаданные проекта в конфигурации.

    Parameters:
    -----------
    name : str, optional
        Название проекта
    description : str, optional
        Описание проекта
    version : str, optional
        Версия проекта

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    config = load_config()

    if "project" not in config:
        config["project"] = DEFAULT_CONFIG["project"]

    if name is not None:
        config["project"]["name"] = name

    if description is not None:
        config["project"]["description"] = description

    if version is not None:
        config["project"]["version"] = version

    # Обновляем дату изменения
    config["project"]["updated"] = datetime.now().strftime("%Y-%m-%d")

    return save_config(config)


def get_reporting_settings() -> Dict[str, Any]:
    """
    Получает настройки системы отчетности.

    Returns:
    --------
    Dict[str, Any]
        Словарь с настройками отчетности
    """
    config = load_config()

    if "reporting" not in config:
        return DEFAULT_CONFIG["reporting"]

    # Проверяем наличие всех необходимых настроек
    for key, value in DEFAULT_CONFIG["reporting"].items():
        if key not in config["reporting"]:
            config["reporting"][key] = value

    return config["reporting"]


def update_reporting_settings(template: Optional[str] = None,
                              auto_open: Optional[bool] = None,
                              include_resources: Optional[bool] = None) -> bool:
    """
    Обновляет настройки системы отчетности.

    Parameters:
    -----------
    template : str, optional
        Имя шаблона отчета по умолчанию
    auto_open : bool, optional
        Открывать ли отчет автоматически после создания
    include_resources : bool, optional
        Включать ли ресурсы в отчет

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    config = load_config()

    if "reporting" not in config:
        config["reporting"] = DEFAULT_CONFIG["reporting"]

    if template is not None:
        config["reporting"]["default_template"] = template

    if auto_open is not None:
        config["reporting"]["auto_open"] = auto_open

    if include_resources is not None:
        config["reporting"]["include_resources"] = include_resources

    return save_config(config)


def get_task_dependencies() -> Dict[str, List[str]]:
    """
    Получает зависимости между задачами из конфигурации.

    Returns:
    --------
    Dict[str, List[str]]
        Словарь {задача: [зависимости]}
    """
    config = load_config()

    if "task_flow" not in config or "dependencies" not in config["task_flow"]:
        return {}

    return config["task_flow"]["dependencies"]


def update_task_dependencies(dependencies: Dict[str, List[str]]) -> bool:
    """
    Обновляет зависимости между задачами в конфигурации.

    Parameters:
    -----------
    dependencies : Dict[str, List[str]]
        Словарь {задача: [зависимости]}

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    config = load_config()

    if "task_flow" not in config:
        config["task_flow"] = DEFAULT_CONFIG["task_flow"]

    config["task_flow"]["dependencies"] = dependencies

    return save_config(config)


def add_task_dependency(task_id: str, dependency_id: str) -> bool:
    """
    Добавляет зависимость между задачами.

    Parameters:
    -----------
    task_id : str
        Идентификатор зависимой задачи
    dependency_id : str
        Идентификатор задачи, от которой зависит

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    dependencies = get_task_dependencies()

    if task_id not in dependencies:
        dependencies[task_id] = []

    if dependency_id not in dependencies[task_id]:
        dependencies[task_id].append(dependency_id)

    return update_task_dependencies(dependencies)


def remove_task_dependency(task_id: str, dependency_id: str) -> bool:
    """
    Удаляет зависимость между задачами.

    Parameters:
    -----------
    task_id : str
        Идентификатор зависимой задачи
    dependency_id : str
        Идентификатор задачи, от которой зависит

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    dependencies = get_task_dependencies()

    if task_id not in dependencies:
        return True

    if dependency_id in dependencies[task_id]:
        dependencies[task_id].remove(dependency_id)

    return update_task_dependencies(dependencies)


def get_templates_dir() -> Path:
    """
    Получает директорию с шаблонами отчетов.

    Returns:
    --------
    Path
        Путь к директории с шаблонами
    """
    try:
        from pamola_core.utils.io import get_data_repository
        repo_dir = get_data_repository()
        templates_dir = repo_dir / "templates"
    except (ImportError, Exception):
        logger.warning("Не удалось получить директорию репозитория данных. Используем текущую директорию.")
        templates_dir = Path("data/templates")

    if not templates_dir.exists():
        templates_dir.mkdir(parents=True, exist_ok=True)

    return templates_dir


def get_template_path(template_name: Optional[str] = None) -> Path:
    """
    Получает путь к шаблону отчета.

    Parameters:
    -----------
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)

    Returns:
    --------
    Path
        Путь к директории с шаблоном
    """
    if template_name is None:
        settings = get_reporting_settings()
        template_name = settings["default_template"]

    template_dir = get_templates_dir() / template_name

    if not template_dir.exists():
        template_dir.mkdir(parents=True, exist_ok=True)

    return template_dir


def get_reports_dir() -> Path:
    """
    Получает директорию для хранения отчетов.

    Returns:
    --------
    Path
        Путь к директории с отчетами
    """
    try:
        from pamola_core.utils.io import get_data_repository
        repo_dir = get_data_repository()
        reports_dir = repo_dir / "reports" / "html"
    except (ImportError, Exception):
        logger.warning("Не удалось получить директорию репозитория данных. Используем текущую директорию.")
        reports_dir = Path("data/reports/html")

    if not reports_dir.exists():
        reports_dir.mkdir(parents=True, exist_ok=True)

    return reports_dir


def detect_task_type(task_id: str) -> str:
    """
    Определяет тип задачи на основе ее идентификатора.

    Parameters:
    -----------
    task_id : str
        Идентификатор задачи

    Returns:
    --------
    str
        Тип задачи (профилирование, очистка, анонимизация и т. д.)
    """
    task_types = {
        "profile": "Профилирование",
        "clean": "Очистка",
        "anonymize": "Анонимизация",
        "evaluate": "Оценка",
        "transform": "Трансформация",
        "export": "Экспорт",
        "import": "Импорт",
        "test": "Тестирование"
    }

    for prefix, task_type in task_types.items():
        if task_id.startswith(prefix) or f"_{prefix}" in task_id:
            return task_type

    # Специальные случаи
    if "ident" in task_id or "identification" in task_id:
        return "Профилирование"
    elif "details" in task_id:
        return "Профилирование"
    elif "contacts" in task_id:
        return "Профилирование"

    # По умолчанию
    return "Другое"


def categorize_tasks(task_ids: List[str]) -> Dict[str, List[str]]:
    """
    Категоризирует задачи по типам.

    Parameters:
    -----------
    task_ids : List[str]
        Список идентификаторов задач

    Returns:
    --------
    Dict[str, List[str]]
        Словарь {категория: [задачи]}
    """
    categories = {}

    for task_id in task_ids:
        task_type = detect_task_type(task_id)

        if task_type not in categories:
            categories[task_type] = []

        categories[task_type].append(task_id)

    return categories