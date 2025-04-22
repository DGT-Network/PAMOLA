"""
Модуль для работы с зависимостями задач.

Функции для анализа, валидации и визуализации зависимостей между задачами.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from pamola_core.utils.reporting.config import get_task_dependencies

# Настройка логирования
logger = logging.getLogger(__name__)


def get_all_tasks(task_reports: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Получает список всех задач из отчетов.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    List[str]
        Список идентификаторов задач
    """
    return list(task_reports.keys())


def resolve_dependencies(task_dependencies: Dict[str, List[str]],
                         task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Разрешает зависимости между задачами, учитывая задачи из отчетов.

    Parameters:
    -----------
    task_dependencies : Dict[str, List[str]]
        Словарь {задача: [зависимости]}
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, List[str]]
        Обновленный словарь зависимостей
    """
    resolved_dependencies = {}
    all_tasks = set(task_reports.keys())

    for task_id, deps in task_dependencies.items():
        if task_id in all_tasks:
            # Оставляем только существующие зависимости
            valid_deps = [dep for dep in deps if dep in all_tasks]
            resolved_dependencies[task_id] = valid_deps

    # Добавляем задачи, которые есть в отчетах, но не указаны в зависимостях
    for task_id in all_tasks:
        if task_id not in resolved_dependencies:
            resolved_dependencies[task_id] = []

    return resolved_dependencies


def check_circular_dependencies(dependencies: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """
    Проверяет наличие циклических зависимостей.

    Parameters:
    -----------
    dependencies : Dict[str, List[str]]
        Словарь {задача: [зависимости]}

    Returns:
    --------
    List[Tuple[str, str]]
        Список пар задач, образующих циклические зависимости
    """
    cycles = []
    visited = set()
    path = []

    def dfs(node):
        if node in path:
            # Нашли цикл
            cycle_start = path.index(node)
            for i in range(cycle_start, len(path) - 1):
                cycles.append((path[i], path[i + 1]))
            cycles.append((path[-1], node))
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)

        for dep in dependencies.get(node, []):
            dfs(dep)

        path.pop()

    for node in dependencies:
        if node not in visited:
            dfs(node)

    return cycles


def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
    """
    Выполняет топологическую сортировку задач.

    Parameters:
    -----------
    dependencies : Dict[str, List[str]]
        Словарь {задача: [зависимости]}

    Returns:
    --------
    List[str]
        Список задач в порядке выполнения
    """
    # Создаем список с результатом
    result = []

    # Создаем копию зависимостей для модификации
    deps_copy = {task: list(deps) for task, deps in dependencies.items()}

    # Находим задачи без зависимостей
    no_deps = [task for task, deps in deps_copy.items() if not deps]

    # Пока есть задачи без зависимостей
    while no_deps:
        # Берем следующую задачу без зависимостей
        current = no_deps.pop(0)
        result.append(current)

        # Для всех задач, зависящих от текущей
        for task, deps in list(deps_copy.items()):
            if current in deps:
                # Удаляем текущую из зависимостей
                deps.remove(current)

                # Если больше нет зависимостей, добавляем в список без зависимостей
                if not deps:
                    no_deps.append(task)

    # Если остались неразрешенные зависимости, значит есть циклы
    if any(deps for deps in deps_copy.values()):
        logger.warning("Обнаружены циклические зависимости. Возвращаем частичный порядок.")

    return result


def get_dependent_tasks(task_id: str, dependencies: Dict[str, List[str]]) -> List[str]:
    """
    Получает список задач, зависящих от указанной.

    Parameters:
    -----------
    task_id : str
        Идентификатор задачи
    dependencies : Dict[str, List[str]]
        Словарь {задача: [зависимости]}

    Returns:
    --------
    List[str]
        Список идентификаторов зависимых задач
    """
    dependent_tasks = []

    for task, deps in dependencies.items():
        if task_id in deps:
            dependent_tasks.append(task)

    return dependent_tasks


def enrich_tasks_with_dependencies(task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Обогащает отчеты задач информацией о зависимостях.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Обновленный словарь отчетов
    """
    # Получаем зависимости из конфигурации
    config_dependencies = get_task_dependencies()

    # Разрешаем зависимости
    dependencies = resolve_dependencies(config_dependencies, task_reports)

    # Обновляем отчеты
    for task_id, report in task_reports.items():
        # Добавляем зависимости
        report["dependencies"] = dependencies.get(task_id, [])

        # Добавляем зависимые задачи
        report["dependents"] = get_dependent_tasks(task_id, dependencies)

    return task_reports


def create_dependency_graph_data(task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Создает данные для визуализации графа зависимостей.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, Any]
        Данные для визуализации графа
    """
    # Получаем зависимости из конфигурации
    config_dependencies = get_task_dependencies()

    # Разрешаем зависимости
    dependencies = resolve_dependencies(config_dependencies, task_reports)

    # Создаем узлы
    nodes = []
    for task_id, report in task_reports.items():
        nodes.append({
            "id": task_id,
            "name": report.get("task_description", task_id),
            "status": report.get("status", "unknown")
        })

    # Создаем ребра
    edges = []
    for task_id, deps in dependencies.items():
        for dep in deps:
            edges.append({
                "source": dep,
                "target": task_id,
                "label": ""
            })

    return {
        "nodes": nodes,
        "edges": edges
    }


def check_missing_dependencies(task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Проверяет наличие отсутствующих зависимостей.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, List[str]]
        Словарь {задача: [отсутствующие_зависимости]}
    """
    # Получаем зависимости из конфигурации
    config_dependencies = get_task_dependencies()

    # Получаем список всех задач
    all_tasks = set(task_reports.keys())

    # Проверяем наличие отсутствующих зависимостей
    missing_dependencies = {}

    for task_id, deps in config_dependencies.items():
        if task_id in all_tasks:
            missing = [dep for dep in deps if dep not in all_tasks]
            if missing:
                missing_dependencies[task_id] = missing

    return missing_dependencies


def suggest_dependencies(task_reports: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Предлагает возможные зависимости на основе времени выполнения задач.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}

    Returns:
    --------
    Dict[str, List[str]]
        Словарь {задача: [предлагаемые_зависимости]}
    """
    # Получаем существующие зависимости
    config_dependencies = get_task_dependencies()

    # Сортируем задачи по времени начала
    tasks_by_time = [(task_id, report.get("start_time", ""))
                     for task_id, report in task_reports.items()]
    tasks_by_time.sort(key=lambda x: x[1])

    # Создаем список задач в порядке выполнения
    task_sequence = [task_id for task_id, _ in tasks_by_time]

    # Предлагаем зависимости
    suggestions = {}

    for i in range(1, len(task_sequence)):
        current_task = task_sequence[i]
        prev_task = task_sequence[i - 1]

        # Если текущая задача уже имеет зависимости, пропускаем
        if current_task in config_dependencies and config_dependencies[current_task]:
            continue

        # Предлагаем предыдущую задачу как зависимость
        suggestions[current_task] = [prev_task]

    return suggestions


def create_dependency_graph_svg(task_reports: Dict[str, Dict[str, Any]],
                                output_path: Optional[Path] = None) -> Optional[str]:
    """
    Создает SVG-изображение графа зависимостей.

    Parameters:
    -----------
    task_reports : Dict[str, Dict[str, Any]]
        Словарь {id_задачи: отчет}
    output_path : Path, optional
        Путь для сохранения SVG (если не указан, SVG не сохраняется)

    Returns:
    --------
    str or None
        SVG-код или None в случае ошибки
    """
    try:
        # Пытаемся импортировать graphviz
        import graphviz
    except ImportError:
        logger.warning("Модуль graphviz не установлен. Создание SVG невозможно.")
        return None

    try:
        # Создаем граф
        dot = graphviz.Digraph("dependency_graph", format="svg")

        # Настраиваем стиль графа
        dot.attr("graph", rankdir="LR", nodesep="0.5", ranksep="1")
        dot.attr("node", shape="box", style="filled", fontname="Arial", fontsize="12")
        dot.attr("edge", fontname="Arial", fontsize="10")

        # Получаем данные для графа
        graph_data = create_dependency_graph_data(task_reports)

        # Добавляем узлы
        for node in graph_data["nodes"]:
            # Выбираем цвет в зависимости от статуса
            fillcolor = "#FFFFFF"  # По умолчанию белый
            if node["status"] == "completed":
                fillcolor = "#D1E7DD"  # Зеленый
            elif node["status"] == "error" or node["status"] == "failed":
                fillcolor = "#F8D7DA"  # Красный
            elif node["status"] == "running":
                fillcolor = "#FFF3CD"  # Желтый

            # Добавляем узел
            dot.node(node["id"], label=node["name"], fillcolor=fillcolor)

        # Добавляем ребра
        for edge in graph_data["edges"]:
            dot.edge(edge["source"], edge["target"], label=edge["label"])

        # Рендерим SVG
        svg_data = dot.pipe().decode("utf-8")

        # Сохраняем SVG, если указан путь
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg_data)
            logger.info(f"SVG-граф сохранен в {output_path}")

        return svg_data
    except Exception as e:
        logger.error(f"Ошибка при создании SVG-графа: {e}")
        return None