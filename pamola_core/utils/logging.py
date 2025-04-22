"""
Настройка логирования для проекта HHR.
"""

import logging
import sys
from pathlib import Path


def configure_logging(log_file=None, level=logging.INFO, name="hhr"):
    """
    Настраивает логирование для проекта.

    Parameters:
    -----------
    log_file : str, optional
        Путь к файлу логов. Если None, логи будут выводиться только в консоль.
    level : int, optional
        Уровень логирования (по умолчанию INFO).
    name : str, optional
        Имя логгера (по умолчанию "hhr").

    Returns:
    --------
    logging.Logger
        Настроенный логгер
    """
    # Создаем форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Настраиваем корневой логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Очищаем существующие обработчики

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Обработчик для файла (если указан)
    if log_file:
        log_path = Path(log_file)
        # Создаем директорию для лога, если нужно
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_task_logging(task_dir, level=logging.INFO):
    """
    Настраивает логирование для конкретной задачи.

    Parameters:
    -----------
    task_dir : Path
        Директория задачи
    level : int, optional
        Уровень логирования (по умолчанию INFO).

    Returns:
    --------
    logging.Logger
        Настроенный логгер
    """
    # Создаем директорию логов (родительская директория/logs)
    logs_dir = task_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Имя файла лога - имя директории задачи
    task_name = task_dir.name
    log_file = logs_dir / f"{task_name}.log"

    return configure_logging(log_file, level, f"hhr.task.{task_name}")


def get_logger(name):
    """
    Получает логгер для указанного модуля.

    Parameters:
    -----------
    name : str
        Имя модуля/компонента для логирования

    Returns:
    --------
    logging.Logger
        Логгер для указанного модуля
    """
    return logging.getLogger(name)