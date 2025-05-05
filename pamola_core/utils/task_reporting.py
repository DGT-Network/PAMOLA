"""
Функции для создания и управления отчетами о выполнении задач.
"""

import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import psutil

from pamola_core.config.settings import get_data_repository
from pamola_core.utils.io import ensure_directory

# Конфигурация логгера
logger = logging.getLogger("hhr.utils.task_reporting")


def get_reports_directory() -> Path:
    """
    Получает директорию для отчетов о задачах.

    Returns:
    --------
    Path
        Путь к директории отчетов
    """
    repo_dir = get_data_repository()
    reports_dir = repo_dir / "reports"
    return ensure_directory(reports_dir)


def get_task_report_path(task_id: str) -> Path:
    """
    Получает путь к файлу отчета о задаче.

    Parameters:
    -----------
    task_id : str
        Идентификатор задачи

    Returns:
    --------
    Path
        Путь к файлу отчета
    """
    reports_dir = get_reports_directory()
    return reports_dir / f"{task_id}_report.json"


def create_task_report(task_id: str,
                       task_description: str,
                       task_type: str = "profiling",
                       script_path: Optional[str] = None,
                       dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Создает новый отчет о выполнении задачи.

    Parameters:
    -----------
    task_id : str
        Идентификатор задачи
    task_description : str
        Описание задачи
    task_type : str
        Тип задачи (profiling, cleaning, anonymization)
    script_path : str, optional
        Путь к скрипту задачи
    dependencies : List[str], optional
        Список идентификаторов задач, от которых зависит текущая задача

    Returns:
    --------
    Dict[str, Any]
        Структура отчета
    """
    # Получаем информацию о системе
    system_info = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "user": os.getlogin(),
        "machine": platform.node()
    }

    # Создаем базовую структуру отчета
    report = {
        "task_id": task_id,
        "task_description": task_description,
        "task_type": task_type,
        "script_path": script_path,
        "system_info": system_info,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "execution_time_seconds": None,
        "status": "running",
        "operations": [],
        "artifacts": [],
        "errors": [],
        "warnings": [],
        "memory_usage_mb": None,
        "dependencies": dependencies or []
    }

    logger.info(f"Created new task report for '{task_id}'")
    return report


def update_task_operation(report: Dict[str, Any],
                          operation: str,
                          status: str = "success",
                          details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Добавляет информацию об операции в отчет о задаче.

    Parameters:
    -----------
    report : Dict[str, Any]
        Существующий отчет
    operation : str
        Название операции
    status : str
        Статус операции (success/error/warning)
    details : Dict[str, Any], optional
        Дополнительные детали операции

    Returns:
    --------
    Dict[str, Any]
        Обновленный отчет
    """
    if details is None:
        details = {}

    operation_info = {
        "operation": operation,
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details
    }

    if "operations" not in report:
        report["operations"] = []

    report["operations"].append(operation_info)

    # Если операция завершилась с ошибкой, добавляем в список ошибок
    if status == "error" and "details" in details and "error" in details:
        if "errors" not in report:
            report["errors"] = []
        report["errors"].append({
            "operation": operation,
            "error": details["error"],
            "timestamp": operation_info["timestamp"]
        })

    # Если операция завершилась с предупреждением, добавляем в список предупреждений
    if status == "warning" and "details" in details and "warning" in details:
        if "warnings" not in report:
            report["warnings"] = []
        report["warnings"].append({
            "operation": operation,
            "warning": details["warning"],
            "timestamp": operation_info["timestamp"]
        })

    logger.debug(f"Added operation '{operation}' with status '{status}' to task report")
    return report


def add_task_artifact(report: Dict[str, Any],
                      artifact_type: str,
                      artifact_path: Union[str, Path],
                      description: str = "") -> Dict[str, Any]:
    """
    Добавляет информацию об артефакте в отчет о задаче.

    Parameters:
    -----------
    report : Dict[str, Any]
        Существующий отчет
    artifact_type : str
        Тип артефакта (json, csv, png, etc.)
    artifact_path : str or Path
        Путь к артефакту
    description : str
        Описание артефакта

    Returns:
    --------
    Dict[str, Any]
        Обновленный отчет
    """
    artifact_info = {
        "type": artifact_type,
        "path": str(artifact_path),
        "filename": Path(artifact_path).name,
        "size_bytes": Path(artifact_path).stat().st_size if Path(artifact_path).exists() else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description
    }

    if "artifacts" not in report:
        report["artifacts"] = []

    report["artifacts"].append(artifact_info)

    logger.debug(f"Added artifact '{Path(artifact_path).name}' to task report")
    return report


def add_task_dependency(report: Dict[str, Any], dependency_id: str) -> Dict[str, Any]:
    """
    Добавляет зависимость от другой задачи.

    Parameters:
    -----------
    report : Dict[str, Any]
        Существующий отчет
    dependency_id : str
        Идентификатор задачи, от которой зависит текущая

    Returns:
    --------
    Dict[str, Any]
        Обновленный отчет
    """
    if "dependencies" not in report:
        report["dependencies"] = []

    if dependency_id not in report["dependencies"]:
        report["dependencies"].append(dependency_id)
        logger.debug(f"Added dependency '{dependency_id}' to task report")

    return report


def remove_task_dependency(report: Dict[str, Any], dependency_id: str) -> Dict[str, Any]:
    """
    Удаляет зависимость от задачи.

    Parameters:
    -----------
    report : Dict[str, Any]
        Существующий отчет
    dependency_id : str
        Идентификатор задачи, от которой больше не зависит текущая

    Returns:
    --------
    Dict[str, Any]
        Обновленный отчет
    """
    if "dependencies" in report and dependency_id in report["dependencies"]:
        report["dependencies"].remove(dependency_id)
        logger.debug(f"Removed dependency '{dependency_id}' from task report")

    return report


def complete_task_report(report: Dict[str, Any], status: str = "completed") -> Dict[str, Any]:
    """
    Завершает отчет о задаче, добавляя информацию о времени выполнения и статусе.

    Parameters:
    -----------
    report : Dict[str, Any]
        Существующий отчет
    status : str
        Итоговый статус задачи (completed/failed/aborted)

    Returns:
    --------
    Dict[str, Any]
        Завершенный отчет
    """
    # Получаем текущее время
    end_time = datetime.now()

    # Рассчитываем время выполнения
    if "start_time" in report:
        start_time = datetime.strptime(report["start_time"], "%Y-%m-%d %H:%M:%S")
        execution_time = (end_time - start_time).total_seconds()
    else:
        execution_time = None

    # Обновляем отчет
    report["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    report["execution_time_seconds"] = execution_time
    report["status"] = status

    # Добавляем информацию об использовании памяти
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # В мегабайтах
    report["memory_usage_mb"] = round(memory_usage, 2)

    logger.info(f"Completed task report with status '{status}', execution time: {execution_time:.2f}s")
    return report


def convert_numpy_types(obj):
    """
    Рекурсивно преобразует типы данных NumPy в стандартные типы Python.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)


def save_task_report(report: Dict[str, Any]) -> Path:
    """
    Сохраняет отчет о выполнении задачи в JSON.

    Parameters:
    -----------
    report : Dict[str, Any]
        Отчет для сохранения

    Returns:
    --------
    Path
        Путь к сохраненному файлу
    """
    if "task_id" not in report:
        raise ValueError("Report must contain a task_id")

    task_id = report["task_id"]
    file_path = get_task_report_path(task_id)

    # Преобразуем numpy-типы в стандартные типы Python
    report = convert_numpy_types(report)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)  # type: ignore

    logger.info(f"Saved task report to {file_path}")
    return file_path


def load_task_report(task_id: str) -> Dict[str, Any]:
    """
    Загружает отчет о выполнении задачи из JSON.

    Parameters:
    -----------
    task_id : str
        Идентификатор задачи

    Returns:
    --------
    Dict[str, Any]
        Загруженный отчет
    """
    file_path = get_task_report_path(task_id)

    if not file_path.exists():
        logger.warning(f"Task report file not found: {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    logger.info(f"Loaded task report from {file_path}")
    return report


def generate_html_report_after_task(task_id: str) -> None:
    """
    Генерирует HTML-отчет после завершения задачи.

    Parameters:
    -----------
    task_id : str
        Идентификатор задачи
    """
    try:
        # Пытаемся импортировать модуль отчетности
        from pamola_core.utils.reporting import generate_report

        # Генерируем отчет
        reports_dir = get_reports_directory()
        html_dir = reports_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)

        output_path = html_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        # Генерируем отчет со всеми задачами
        generate_report(reports_dir=reports_dir, output_path=output_path, auto_open=False)

        logger.info(f"HTML report generated at {output_path}")
    except ImportError:
        logger.warning("Module hhr.utils.reporting not found. HTML report generation skipped.")
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")


class TaskReporter:
    """
    Класс для управления отчетами о задачах в контексте менеджера контекста.
    Позволяет автоматически создавать, обновлять и сохранять отчет.

    Example:
    --------
    ```
    with TaskReporter("identification", "Profiling identification table") as reporter:
        # Выполнение операций
        reporter.add_operation("Reading data")
        # ...
        reporter.add_artifact("json", "path/to/file.json", "Results of analysis")
    ```
    """

    def __init__(self, task_id: str, task_description: str, task_type: str = "profiling",
                 script_path: Optional[str] = None, dependencies: Optional[List[str]] = None,
                 auto_generate_html: bool = False):
        """
        Инициализирует репортер задачи.

        Parameters:
        -----------
        task_id : str
            Идентификатор задачи
        task_description : str
            Описание задачи
        task_type : str
            Тип задачи (profiling, cleaning, anonymization)
        script_path : str, optional
            Путь к скрипту задачи
        dependencies : List[str], optional
            Список идентификаторов задач, от которых зависит текущая задача
        auto_generate_html : bool
            Автоматически генерировать HTML-отчет после завершения задачи
        """
        self.task_id = task_id
        self.task_description = task_description
        self.task_type = task_type
        self.script_path = script_path
        self.dependencies = dependencies
        self.auto_generate_html = auto_generate_html
        self.report = None
        self.start_time = None

    def __enter__(self):
        """
        Создает отчет при входе в контекст.
        """
        self.start_time = time.time()
        self.report = create_task_report(
            self.task_id,
            self.task_description,
            self.task_type,
            self.script_path,
            self.dependencies
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Завершает и сохраняет отчет при выходе из контекста.
        """
        status = "completed"

        # Если произошло исключение, добавляем информацию о нем
        if exc_type is not None:
            status = "failed"
            if self.report:
                self.add_operation(
                    "Exception",
                    status="error",
                    details={"error": str(exc_val), "traceback": str(exc_tb)}
                )

        # Завершаем и сохраняем отчет
        if self.report:
            complete_task_report(self.report, status)
            save_task_report(self.report)

            # Генерируем HTML-отчет, если нужно
            if self.auto_generate_html:
                generate_html_report_after_task(self.task_id)

        return False  # Не подавляем исключение

    def add_operation(self, operation: str, status: str = "success", details: Optional[Dict[str, Any]] = None) -> None:
        """
        Добавляет информацию об операции в отчет.

        Parameters:
        -----------
        operation : str
            Название операции
        status : str
            Статус операции (success/error/warning)
        details : Dict[str, Any], optional
            Дополнительные детали операции
        """
        if self.report:
            update_task_operation(self.report, operation, status, details)

    def add_artifact(self, artifact_type: str, artifact_path: Union[str, Path], description: str = "") -> None:
        """
        Добавляет информацию об артефакте в отчет.

        Parameters:
        -----------
        artifact_type : str
            Тип артефакта (json, csv, png, etc.)
        artifact_path : str or Path
            Путь к артефакту
        description : str
            Описание артефакта
        """
        if self.report:
            add_task_artifact(self.report, artifact_type, artifact_path, description)

    def add_dependency(self, dependency_id: str) -> None:
        """
        Добавляет зависимость от другой задачи.

        Parameters:
        -----------
        dependency_id : str
            Идентификатор задачи, от которой зависит текущая
        """
        if self.report:
            add_task_dependency(self.report, dependency_id)

    def remove_dependency(self, dependency_id: str) -> None:
        """
        Удаляет зависимость от задачи.

        Parameters:
        -----------
        dependency_id : str
            Идентификатор задачи, от которой больше не зависит текущая
        """
        if self.report:
            remove_task_dependency(self.report, dependency_id)

    def save(self) -> Path:
        """
        Сохраняет отчет.

        Returns:
        --------
        Path
            Путь к сохраненному файлу
        """
        if self.report:
            return save_task_report(self.report)
        raise ValueError("Report not initialized")