"""
Модуль для работы с шаблонами отчетов.

Функции для загрузки, рендеринга и копирования шаблонов с использованием Jinja2.
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import jinja2

from pamola_core.utils.reporting.config import get_template_path, get_templates_dir

# Настройка логирования
logger = logging.getLogger(__name__)


def get_jinja_environment(template_name: Optional[str] = None) -> jinja2.Environment:
    """
    Создает и настраивает окружение Jinja2.

    Parameters:
    -----------
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)

    Returns:
    --------
    jinja2.Environment
        Настроенное окружение Jinja2
    """
    template_path = get_template_path(template_name)

    # Проверяем наличие директории шаблонов
    if not template_path.exists():
        logger.warning(f"Директория шаблона не найдена: {template_path}")
        template_path.mkdir(parents=True, exist_ok=True)

    # Создаем загрузчик файлов
    file_system_loader = jinja2.FileSystemLoader(template_path)

    # Создаем окружение
    env = jinja2.Environment(
        loader=file_system_loader,
        autoescape=jinja2.select_autoescape(['html', 'xml']),
        trim_blocks=True,
        lstrip_blocks=True
    )

    # Добавляем пользовательские фильтры
    env.filters['format_size'] = format_file_size
    env.filters['to_json'] = to_json

    return env


def render_template(template_name: str,
                    context: Dict[str, Any],
                    template_dir: Optional[str] = None) -> str:
    """
    Рендерит шаблон с заданным контекстом.

    Parameters:
    -----------
    template_name : str
        Имя файла шаблона (например, base.html)
    context : Dict[str, Any]
        Контекст для рендеринга
    template_dir : str, optional
        Имя директории шаблона (если не указано, используется шаблон по умолчанию)

    Returns:
    --------
    str
        Отрендеренный HTML-код
    """
    try:
        env = get_jinja_environment(template_dir)
        template = env.get_template(template_name)
        return template.render(**context)
    except jinja2.exceptions.TemplateNotFound:
        logger.error(f"Шаблон не найден: {template_name} в директории {template_dir}")
        return f"<p>Ошибка: шаблон {template_name} не найден</p>"
    except Exception as e:
        logger.error(f"Ошибка при рендеринге шаблона {template_name}: {e}")
        return f"<p>Ошибка при рендеринге шаблона: {str(e)}</p>"


def copy_static_resources(target_dir: Path, template_name: Optional[str] = None) -> bool:
    """
    Копирует статические ресурсы шаблона (CSS, JS) в директорию отчета.

    Parameters:
    -----------
    target_dir : Path
        Целевая директория
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    template_path = get_template_path(template_name)

    # Копируем CSS файлы
    styles_dir = template_path / "styles"
    if styles_dir.exists():
        target_styles_dir = target_dir / "styles"
        target_styles_dir.mkdir(exist_ok=True)

        for file in styles_dir.glob("*.css"):
            try:
                shutil.copy2(file, target_styles_dir / file.name)
            except Exception as e:
                logger.error(f"Ошибка при копировании {file.name}: {e}")
                return False

    # Копируем JS файлы
    scripts_dir = template_path / "scripts"
    if scripts_dir.exists():
        target_scripts_dir = target_dir / "scripts"
        target_scripts_dir.mkdir(exist_ok=True)

        for file in scripts_dir.glob("*.js"):
            try:
                shutil.copy2(file, target_scripts_dir / file.name)
            except Exception as e:
                logger.error(f"Ошибка при копировании {file.name}: {e}")
                return False

    # Копируем изображения
    images_dir = template_path / "images"
    if images_dir.exists():
        target_images_dir = target_dir / "images"
        target_images_dir.mkdir(exist_ok=True)

        for file in images_dir.glob("*.*"):
            try:
                shutil.copy2(file, target_images_dir / file.name)
            except Exception as e:
                logger.error(f"Ошибка при копировании {file.name}: {e}")
                return False

    return True


def include_external_resources(target_dir: Path) -> bool:
    """
    Загружает и сохраняет внешние ресурсы (Bootstrap, Chart.js и т. д.).

    Parameters:
    -----------
    target_dir : Path
        Целевая директория

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    try:
        import requests

        # Создаем директории для ресурсов
        styles_dir = target_dir / "styles"
        styles_dir.mkdir(exist_ok=True)

        scripts_dir = target_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Список внешних ресурсов для загрузки
        resources = [
            {
                "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css",
                "path": styles_dir / "bootstrap.min.css"
            },
            {
                "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js",
                "path": scripts_dir / "bootstrap.bundle.min.js"
            },
            {
                "url": "https://cdn.jsdelivr.net/npm/chart.js",
                "path": scripts_dir / "chart.min.js"
            },
            {
                "url": "https://cdnjs.cloudflare.com/ajax/libs/d3/7.0.0/d3.min.js",
                "path": scripts_dir / "d3.min.js"
            },
            {
                "url": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.4/dagre-d3.min.js",
                "path": scripts_dir / "dagre-d3.min.js"
            }
        ]

        # Загружаем каждый ресурс
        for resource in resources:
            response = requests.get(resource["url"], timeout=10)

            if response.status_code == 200:
                with open(resource["path"], 'wb') as f:
                    f.write(response.content)
                logger.info(f"Загружен ресурс: {resource['url']}")
            else:
                logger.error(f"Ошибка при загрузке ресурса {resource['url']}: {response.status_code}")
                return False

        return True
    except ImportError:
        logger.warning("Модуль requests не установлен. Включение внешних ресурсов невозможно.")
        return False
    except Exception as e:
        logger.error(f"Ошибка при включении внешних ресурсов: {e}")
        return False


def check_template_exists(template_name: Optional[str] = None) -> bool:
    """
    Проверяет наличие шаблона.

    Parameters:
    -----------
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)

    Returns:
    --------
    bool
        True, если шаблон существует и содержит необходимые файлы
    """
    template_path = get_template_path(template_name)

    # Проверяем наличие основных файлов шаблона
    required_files = [
        "base.html",
        "partials/header.html",
        "partials/sidebar.html"
    ]

    for file in required_files:
        if not (template_path / file).exists():
            return False

    return True


def initialize_template(template_name: str) -> bool:
    """
    Инициализирует шаблон - копирует стандартные файлы в директорию шаблона.

    Parameters:
    -----------
    template_name : str
        Имя нового шаблона

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    try:
        template_path = get_template_path(template_name)
        default_template_path = get_template_path("default")

        # Если директория уже существует и не пустая, уточняем
        if template_path.exists() and any(template_path.iterdir()):
            logger.warning(f"Директория шаблона {template_name} уже существует и не пустая")
            return False

        # Если директория по умолчанию не существует, создаем ее и базовые файлы
        if not default_template_path.exists():
            logger.warning("Директория шаблона по умолчанию не существует, создаем...")
            create_default_template()

        # Копируем файлы из шаблона по умолчанию
        for item in default_template_path.glob("**/*"):
            if item.is_file():
                # Создаем относительный путь
                rel_path = item.relative_to(default_template_path)
                # Создаем целевой путь
                target_path = template_path / rel_path
                # Создаем родительскую директорию, если она не существует
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Копируем файл
                shutil.copy2(item, target_path)

        logger.info(f"Шаблон {template_name} инициализирован")
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации шаблона {template_name}: {e}")
        return False


def list_available_templates() -> List[str]:
    """
    Получает список доступных шаблонов.

    Returns:
    --------
    List[str]
        Список имен доступных шаблонов
    """
    templates_dir = get_templates_dir()

    # Собираем список директорий в директории шаблонов
    templates = []
    for item in templates_dir.iterdir():
        if item.is_dir() and check_template_exists(item.name):
            templates.append(item.name)

    return templates


def create_default_template() -> bool:
    """
    Создает шаблон по умолчанию с базовыми файлами.

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    try:
        template_path = get_template_path("default")

        # Создаем структуру директорий
        dirs = [
            template_path,
            template_path / "partials",
            template_path / "styles",
            template_path / "scripts",
            template_path / "images"
        ]

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

        # Здесь можно создать базовые файлы шаблона
        # Это потребовало бы значительного количества кода
        # В реальном приложении эти файлы можно загрузить из ресурсов пакета

        return True
    except Exception as e:
        logger.error(f"Ошибка при создании шаблона по умолчанию: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Форматирует размер файла в читаемый вид.

    Parameters:
    -----------
    size_bytes : int
        Размер в байтах

    Returns:
    --------
    str
        Отформатированная строка с размером
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def to_json(data: Any) -> str:
    """
    Преобразует данные в JSON-строку.

    Parameters:
    -----------
    data : Any
        Данные для преобразования

    Returns:
    --------
    str
        JSON-строка
    """
    return json.dumps(data, ensure_ascii=False)


def create_empty_report(output_path: Path, title: str = "Пустой отчет") -> bool:
    """
    Создает пустой отчет для случаев, когда нет данных.

    Parameters:
    -----------
    output_path : Path
        Путь для сохранения отчета
    title : str
        Заголовок отчета

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    try:
        context = {
            "title": title,
            "timestamp": "",
            "tasks": [],
            "tasks_by_category": {},
            "has_dependencies": False,
            "dependency_data": "{}"
        }

        html = render_template("base.html", context)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return True
    except Exception as e:
        logger.error(f"Ошибка при создании пустого отчета: {e}")
        return False


def get_relative_artifact_path(artifact_path: Union[str, Path], html_report_dir: Path) -> str:
    """
    Вычисляет относительный путь к артефакту от директории отчета.

    Parameters:
    -----------
    artifact_path : str or Path
        Абсолютный путь к артефакту
    html_report_dir : Path
        Путь к директории отчета

    Returns:
    --------
    str
        Относительный путь к артефакту
    """
    artifact_path = Path(artifact_path)

    try:
        # Пытаемся вычислить относительный путь
        rel_path = os.path.relpath(artifact_path, html_report_dir)
        return rel_path.replace("\\", "/")  # Для совместимости с URL в HTML
    except ValueError:
        # В случае ошибки (например, разные диски в Windows)
        logger.warning(f"Не удалось вычислить относительный путь для {artifact_path}")
        return str(artifact_path)