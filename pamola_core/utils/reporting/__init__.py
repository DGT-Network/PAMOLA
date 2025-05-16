"""
Пакет для создания отчетов о выполнении задач.

Этот пакет предоставляет функциональность для создания HTML-отчетов
на основе отчетов о выполнении задач.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from pamola_core.utils.logging import configure_logging
logger = configure_logging()

# Импортируем основные функции и классы
from pamola_core.utils.reporting.config import (
    load_config, save_config, get_reporting_settings, update_reporting_settings,
    get_task_dependencies, update_task_dependencies, add_task_dependency,
    remove_task_dependency, get_templates_dir, get_reports_dir
)

from pamola_core.utils.reporting.template_engine import (
    render_template, copy_static_resources, include_external_resources,
    check_template_exists, initialize_template, list_available_templates
)

from pamola_core.utils.reporting.dependency_graph import (
    get_all_tasks, resolve_dependencies, check_circular_dependencies,
    topological_sort, enrich_tasks_with_dependencies, create_dependency_graph_data,
    create_dependency_graph_svg
)

from pamola_core.utils.reporting.html_formatter import (
    create_html_report, load_task_report, load_all_task_reports,
    prepare_task_reports, create_dashboard_data
)


def generate_report(
        reports_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        template_name: Optional[str] = None,
        auto_open: bool = True
) -> Tuple[Path, bool]:
    """
    Генерирует HTML-отчет на основе всех доступных отчетов задач.

    Parameters:
    -----------
    reports_dir : Path, optional
        Директория с отчетами задач (если не указана, используется директория по умолчанию)
    output_path : Path, optional
        Путь для сохранения отчета (если не указан, используется путь по умолчанию)
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)
    auto_open : bool
        Открывать ли отчет в браузере после создания

    Returns:
    --------
    Tuple[Path, bool]
        Путь к созданному отчету и флаг успешности
    """
    # Загружаем отчеты
    task_reports = load_all_task_reports(reports_dir)

    if not task_reports:
        # Если отчетов нет, создаем пустой отчет
        from pamola_core.utils.reporting.template_engine import create_empty_report

        if output_path is None:
            output_path = get_reports_dir() / "empty_report.html"

        create_empty_report(output_path, "Отчеты не найдены")

        if auto_open:
            import webbrowser
            webbrowser.open(f"file://{output_path}")

        return output_path, False

    # Обновляем настройки отчетности
    if auto_open is not None:
        update_reporting_settings(auto_open=auto_open)

    # Создаем отчет
    return create_html_report(task_reports, output_path, template_name)


def generate_report_for_tasks(
        task_ids: List[str],
        reports_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        template_name: Optional[str] = None,
        auto_open: bool = True
) -> Tuple[Path, bool]:
    """
    Генерирует HTML-отчет только для указанных задач.

    Parameters:
    -----------
    task_ids : List[str]
        Список идентификаторов задач
    reports_dir : Path, optional
        Директория с отчетами задач (если не указана, используется директория по умолчанию)
    output_path : Path, optional
        Путь для сохранения отчета (если не указан, используется путь по умолчанию)
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)
    auto_open : bool
        Открывать ли отчет в браузере после создания

    Returns:
    --------
    Tuple[Path, bool]
        Путь к созданному отчету и флаг успешности
    """
    # Загружаем все отчеты
    all_task_reports = load_all_task_reports(reports_dir)

    # Фильтруем только нужные задачи
    task_reports = {task_id: all_task_reports[task_id]
                    for task_id in task_ids if task_id in all_task_reports}

    if not task_reports:
        # Если отчетов нет, создаем пустой отчет
        from pamola_core.utils.reporting.template_engine import create_empty_report

        if output_path is None:
            output_path = get_reports_dir() / "empty_filtered_report.html"

        create_empty_report(output_path, "Отчеты для указанных задач не найдены")

        if auto_open:
            import webbrowser
            webbrowser.open(f"file://{output_path}")

        return output_path, False

    # Обновляем настройки отчетности
    if auto_open is not None:
        update_reporting_settings(auto_open=auto_open)

    # Создаем отчет
    return create_html_report(task_reports, output_path, template_name)


def install_templates() -> bool:
    """
    Устанавливает стандартные шаблоны в директорию шаблонов.

    Returns:
    --------
    bool
        True в случае успеха, False в случае ошибки
    """
    try:
        templates_dir = get_templates_dir()

        # Проверяем наличие директории шаблонов
        if not templates_dir.exists():
            templates_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория шаблонов: {templates_dir}")

        # Проверяем наличие шаблона по умолчанию
        default_template_dir = templates_dir / "default"
        if default_template_dir.exists() and any(default_template_dir.iterdir()):
            logger.info("Шаблон по умолчанию уже установлен")
            return True

        # Инициализируем шаблон по умолчанию
        success = initialize_template("default")
        if success:
            logger.info("Шаблон по умолчанию успешно установлен")
        else:
            logger.error("Не удалось установить шаблон по умолчанию")

        return success
    except Exception as e:
        logger.error(f"Ошибка при установке шаблонов: {e}")
        return False


__all__ = [
    # Функции для генерации отчетов
    'generate_report', 'generate_report_for_tasks', 'install_templates',

    # Функции для работы с конфигурацией
    'load_config', 'save_config', 'get_reporting_settings', 'update_reporting_settings',
    'get_task_dependencies', 'update_task_dependencies', 'add_task_dependency',
    'remove_task_dependency', 'get_templates_dir', 'get_reports_dir',

    # Функции для работы с шаблонами
    'render_template', 'copy_static_resources', 'include_external_resources',
    'check_template_exists', 'initialize_template', 'list_available_templates',

    # Функции для работы с зависимостями
    'get_all_tasks', 'resolve_dependencies', 'check_circular_dependencies',
    'topological_sort', 'enrich_tasks_with_dependencies', 'create_dependency_graph_data',
    'create_dependency_graph_svg',

    # Функции для форматирования HTML
    'create_html_report', 'load_task_report', 'load_all_task_reports',
    'prepare_task_reports', 'create_dashboard_data'
]