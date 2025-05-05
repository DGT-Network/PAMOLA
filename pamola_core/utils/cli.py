"""
Утилиты для работы с аргументами командной строки в проекте анонимизации резюме HeadHunter.

Этот модуль содержит функции для стандартизации аргументов командной строки,
используемых в разных скриптах профилирования.
"""

import argparse


def add_profiling_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Добавляет стандартные аргументы для задач профилирования.

    Parameters:
    -----------
    parser : argparse.ArgumentParser
        Парсер аргументов
    """
    parser.add_argument("--input", type=str, default="data/raw/input.csv",
                        help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output directory")
    parser.add_argument("--log", type=str, default="docs/logs/profiling.log",
                        help="Path to log file")
    parser.add_argument("--encoding", type=str, default="utf-16",
                        help="File encoding (default: utf-16)")
    parser.add_argument("--delimiter", type=str, default=";",
                        help="CSV delimiter (default: ;)")
    parser.add_argument("--quotechar", type=str, default='"',
                        help="Quote character in CSV (default: \")")
    parser.add_argument("--escapechar", type=str, default=None,
                        help="Escape character in CSV (default: None)")
    parser.add_argument("--on_bad_lines", type=str, default="warn",
                        help="Action to take when encountering bad lines (default: warn)")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing artifacts")
    parser.add_argument("--generate-html", action="store_true",
                        help="Generate HTML report after profiling (default: False)")


def create_profiling_parser(description: str, default_input: str, default_log: str) -> argparse.ArgumentParser:
    """
    Создает парсер аргументов с настроенными параметрами для конкретной задачи профилирования.

    Parameters:
    -----------
    description : str
        Описание парсера
    default_input : str
        Путь к входному файлу по умолчанию
    default_log : str
        Путь к файлу логов по умолчанию

    Returns:
    --------
    argparse.ArgumentParser:
        Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(description=description)
    add_profiling_arguments(parser)

    # Переопределяем пути по умолчанию для конкретной задачи
    parser.set_defaults(input=default_input)
    parser.set_defaults(log=default_log)

    return parser