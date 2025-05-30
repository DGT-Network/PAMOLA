"""
Модуль для создания HTML-отчетов.

Функции для создания HTML-отчетов на основе отчетов задач.
"""
import json
import logging
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

from pamola_core.utils.reporting.config import (
    get_reports_dir, get_reporting_settings, categorize_tasks
)
from pamola_core.utils.reporting.dependency_graph import (
    enrich_tasks_with_dependencies, create_dependency_graph_data,
    check_missing_dependencies
)
from pamola_core.utils.reporting.template_engine import (
    render_template, copy_static_resources, include_external_resources,
    get_relative_artifact_path
)

# Настройка логирования
logger = logging.getLogger(__name__)


def prepare_task_reports(task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Подготавливает отчеты задач для отображения.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Подготовленный словарь отчетов
    """
    # Обогащаем отчеты зависимостями
    enriched_reports = enrich_tasks_with_dependencies(task_reports)

    # Проверяем отсутствующие зависимости
    missing_deps = check_missing_dependencies(enriched_reports)

    # Добавляем информацию о отсутствующих зависимостях
    for task_id, missing in missing_deps.items():
        if task_id in enriched_reports:
            enriched_reports[task_id]["missing_dependencies"] = missing

    return enriched_reports


def prepare_artifacts_for_report(task_reports: Dict[str, Dict[str, Any]],
                                 report_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Подготавливает артефакты для отчета, добавляя относительные пути.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}
    report_dir : Path
        Директория отчета

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Подготовленный словарь отчетов
    """
    for task_id, report in task_reports.items():
        if "artifacts" in report:
            for artifact in report["artifacts"]:
                # Добавляем относительный путь к артефакту
                artifact_path = artifact.get("path")
                if artifact_path:
                    artifact["relative_path"] = get_relative_artifact_path(artifact_path, report_dir)

                # Если это JSON и нужно отобразить предпросмотр
                if artifact.get("type") == "json" and os.path.exists(artifact.get("path", "")):
                    try:
                        with open(artifact["path"], 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Создаем HTML-предпросмотр для JSON
                        preview = create_json_preview(data)
                        artifact["preview"] = preview
                    except Exception as e:
                        logger.warning(f"Не удалось создать предпросмотр для {artifact['path']}: {e}")

    return task_reports


def create_json_preview(data: Any, max_depth: int = 2) -> str:
    """
    Создает HTML-предпросмотр для JSON-данных.

    Parameters:
    -----------
    data : Any
        JSON-данные
    max_depth : int
        Максимальная глубина вложенности для отображения

    Returns:
    --------
    str
        HTML-код предпросмотра
    """
    html = ['<div class="json-preview">']

    def _process_item(item, depth=0):
        indent = '  ' * depth

        if isinstance(item, dict):
            if depth >= max_depth:
                return f"{indent}{{...}}"

            result = []
            for key, value in item.items():
                if isinstance(value, (dict, list, tuple)) and depth == max_depth - 1:
                    result.append(f"{indent}<strong>{key}</strong>: {{...}}")
                else:
                    processed_value = _process_item(value, depth + 1)
                    result.append(f"{indent}<strong>{key}</strong>: {processed_value}")

            return "{\n" + ",\n".join(result) + f"\n{indent}}}"

        elif isinstance(item, (list, tuple)):
            if depth >= max_depth:
                return f"{indent}[...]"

            result = []
            for value in item[:5]:  # Ограничиваем количество элементов
                processed_value = _process_item(value, depth + 1)
                result.append(processed_value)

            if len(item) > 5:
                result.append(f"{indent}...")

            return "[\n" + ",\n".join([f"{indent}{val}" for val in result]) + f"\n{indent}]"

        elif isinstance(item, str):
            return f'"{item}"'

        elif item is None:
            return "null"

        else:
            return str(item)

    html.append(_process_item(data))
    html.append('</div>')

    return ''.join(html)


def create_dashboard_data(task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Создает данные для панели мониторинга.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, Any]
        Данные для панели мониторинга
    """
    # Статистика по статусам
    status_count = {
        "completed": 0,
        "failed": 0,
        "running": 0,
        "other": 0
    }

    # Общее количество артефактов
    total_artifacts = 0
    artifacts_by_type = {}

    # Общее время выполнения
    total_execution_time = 0

    # Общее количество операций
    total_operations = 0
    operations_by_status = {
        "success": 0,
        "error": 0,
        "warning": 0,
        "other": 0
    }

    # Категоризация задач
    tasks_by_category = categorize_tasks(list(task_reports.keys()))

    # Обрабатываем каждый отчет
    for task_id, report in task_reports.items():
        # Статус
        status = report.get("status", "other")
        if status in status_count:
            status_count[status] += 1
        else:
            status_count["other"] += 1

        # Артефакты
        artifacts = report.get("artifacts", [])
        total_artifacts += len(artifacts)

        for artifact in artifacts:
            artifact_type = artifact.get("type", "other")
            artifacts_by_type[artifact_type] = artifacts_by_type.get(artifact_type, 0) + 1

        # Время выполнения
        execution_time = report.get("execution_time_seconds", 0)
        if execution_time:
            total_execution_time += execution_time

        # Операции
        operations = report.get("operations", [])
        total_operations += len(operations)

        for operation in operations:
            operation_status = operation.get("status", "other")
            if operation_status in operations_by_status:
                operations_by_status[operation_status] += 1
            else:
                operations_by_status["other"] += 1

    return {
        "status_count": status_count,
        "total_tasks": len(task_reports),
        "total_artifacts": total_artifacts,
        "artifacts_by_type": artifacts_by_type,
        "total_execution_time": total_execution_time,
        "total_operations": total_operations,
        "operations_by_status": operations_by_status,
        "tasks_by_category": tasks_by_category
    }


def create_html_report(task_reports: Dict[str, Dict[str, Any]],
                       output_path: Optional[Path] = None,
                       template_name: Optional[str] = None) -> Tuple[Path, bool]:
    """
    Создает HTML-отчет на основе отчетов задач.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}
    output_path : Path, optional
        Путь для сохранения отчета (если не указан, используется путь по умолчанию)
    template_name : str, optional
        Имя шаблона (если не указано, используется шаблон по умолчанию)

    Returns:
    --------
    Tuple[Path, bool]
        Путь к созданному отчету и флаг успешности
    """
    # Если не указан путь, используем директорию по умолчанию
    if output_path is None:
        reports_dir = get_reports_dir()
        output_path = reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    # Создаем директорию для отчета, если она не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Получаем настройки отчетности
    settings = get_reporting_settings()

    # Если не указан шаблон, используем шаблон по умолчанию
    if template_name is None:
        template_name = settings.get("default_template", "default")

    try:
        # Подготавливаем отчеты
        prepared_reports = prepare_task_reports(task_reports)

        # Подготавливаем артефакты
        prepared_reports = prepare_artifacts_for_report(prepared_reports, output_path.parent)

        # Создаем данные для панели мониторинга
        dashboard_data = create_dashboard_data(prepared_reports)

        # Категоризируем задачи
        tasks_by_category = dashboard_data["tasks_by_category"]

        # Создаем данные для графа зависимостей
        dependency_data = create_dependency_graph_data(prepared_reports)

        # Проверяем наличие локальных ресурсов и пытаемся их загрузить, если нужно
        include_resources = settings.get("include_resources", False)
        resource_files_exist = False

        if include_resources:
            # Пытаемся скопировать статические ресурсы
            copy_static_resources(output_path.parent, template_name)

            # Определяем необходимые файлы ресурсов
            required_resources = [
                output_path.parent / "scripts" / "bootstrap.bundle.min.js",
                output_path.parent / "scripts" / "chart.min.js",
                output_path.parent / "scripts" / "d3.min.js",
                output_path.parent / "scripts" / "dagre-d3.min.js"
            ]

            # Проверяем наличие всех ресурсов
            missing_resources = [res for res in required_resources if not res.exists()]

            if missing_resources:
                logger.info(f"Некоторые локальные ресурсы отсутствуют. Загружаем их...")
                include_external_resources(output_path.parent)

                # Проверяем снова после загрузки
                missing_resources = [res for res in required_resources if not res.exists()]
                resource_files_exist = len(missing_resources) == 0

                if not resource_files_exist:
                    logger.warning(f"Невозможно загрузить все необходимые ресурсы. Будут использованы CDN.")
            else:
                resource_files_exist = True

        # Создаем контекст для шаблона
        context = {
            "title": "Отчет по проекту PAMOLA.CORE (Privacy-Preserving AI Data Processors) - Data Anonymization",
            "page_title": "Отчет по проекту",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tasks": list(prepared_reports.values()),
            "tasks_by_category": {category: [prepared_reports[task_id] for task_id in tasks]
                                  for category, tasks in tasks_by_category.items()},
            "has_dependencies": bool(dependency_data["edges"]),
            "dependency_data": json.dumps(dependency_data),
            "dashboard": dashboard_data,
            "project_info": {
                "name": "PAMOLA.CORE (Privacy-Preserving AI Data Processors) - Data Anonymization",
                "description": "Анализ и анонимизация данных резюме для обезличивания",
                "version": "1.0.0",
                "created": datetime.now().strftime("%Y-%m-%d"),
                "updated": datetime.now().strftime("%Y-%m-%d")
            },
            "include_resources": include_resources,
            "resource_files_exist": resource_files_exist  # Добавляем флаг наличия ресурсов в контекст
        }

        # Рендерим шаблон
        html = render_template("base.html", context, template_name)

        # Сохраняем HTML
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML-отчет создан: {output_path}")

        # Открываем отчет в браузере, если это указано в настройках
        if settings.get("auto_open", True):
            webbrowser.open(f"file://{output_path}")

        return output_path, True
    except Exception as e:
        logger.error(f"Ошибка при создании HTML-отчета: {e}")
        return output_path, False


def load_task_report(report_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Загружает отчет задачи из JSON-файла.

    Parameters:
    -----------
    report_path : str or Path
        Путь к файлу отчета

    Returns:
    --------
    Dict[str, Any]
        Загруженный отчет или пустой словарь в случае ошибки
    """
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report
    except Exception as e:
        logger.error(f"Ошибка при загрузке отчета {report_path}: {e}")
        return {}


def load_all_task_reports(reports_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Загружает все отчеты задач из директории.

    Parameters:
    -----------
    reports_dir : Path, optional
        Путь к директории с отчетами (если не указан, используется директория по умолчанию)

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}
    """
    if reports_dir is None:
        # Используем директорию из репозитория данных
        try:
            from pamola_core.utils.io import get_data_repository
            repo_dir = get_data_repository()
            reports_dir = repo_dir / "reports"
        except (ImportError, Exception):
            logger.warning("Не удалось получить директорию репозитория данных. Используем текущую директорию.")
            reports_dir = Path("data/reports")

    # Проверяем существование директории
    if not reports_dir.exists():
        logger.warning(f"Директория с отчетами не найдена: {reports_dir}")
        return {}

    # Загружаем все отчеты
    task_reports = {}

    for report_file in reports_dir.glob("*_report.json"):
        try:
            # Извлекаем идентификатор задачи из имени файла
            task_id = report_file.stem.split("_")[0]

            # Загружаем отчет
            report = load_task_report(report_file)

            if report:
                task_reports[task_id] = report
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {report_file}: {e}")

    return task_reports