# План реализации модуля `progress_manager.py`

На основе проанализированной документации и предоставленных рекомендаций, предлагаю следующий обновленный план реализации `progress_manager.py`:

## 1. Общая структура модуля

```python
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Progress Manager
Description: Centralized progress tracking and logging coordination
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides centralized management of progress bars and logging,
ensuring clean display of execution progress without conflicts between
progress indicators and log messages.

Key features:
- Hierarchical progress bars with proper positioning
- Coordinated logging that doesn't break progress displays
- Integration with task reporting
- Support for memory and performance metrics
- Clean exit and resource management
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

import tqdm

from pamola_core.utils.tasks.task_reporting import TaskReporter


class ProgressTracker:
    """Progress tracker for individual operations with fixed positioning."""
    # Реализация класса...


class TaskProgressManager:
    """Centralized manager for task progress and logging coordination."""
    # Реализация класса...


class ProgressContext:
    """Context manager for operation execution with progress tracking."""
    # Реализация класса...
```

## 2. Класс `ProgressTracker`

```python
class ProgressTracker:
    """
    Progress tracker for individual operations with fixed positioning.
    
    This class wraps tqdm with additional functionality for hierarchical
    display, metrics collection, and proper positioning.
    """
    
    def __init__(
            self,
            total: int,
            description: str,
            unit: str = "items",
            position: int = 0,
            leave: bool = True,
            parent: Optional['ProgressTracker'] = None,
            color: Optional[str] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress (e.g., "items", "records")
            position: Fixed position on screen (0 = top)
            leave: Whether to leave the progress bar after completion
            parent: Parent progress tracker (for hierarchical display)
            color: Color of the progress bar (None for default)
        """
        self.total = total
        self.description = description
        self.unit = unit
        self.position = position
        self.leave = leave
        self.parent = parent
        self.color = color
        self.children: List['ProgressTracker'] = []
        
        # Start time and memory tracking
        self.start_time = time.time()
        self.start_memory = self._get_current_memory()
        self.peak_memory = self.start_memory
        
        # Custom metrics
        self.metrics: Dict[str, Any] = {}
        
        # Create progress bar with fixed positioning
        self.pbar = tqdm.tqdm(
            total=total,
            desc=description,
            unit=unit,
            position=position,
            leave=leave,
            file=sys.stdout,
            colour=color
        )
    
    def update(
            self, 
            steps: int = 1, 
            postfix: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Update progress by specified number of steps.
        
        Args:
            steps: Number of steps completed
            postfix: Dictionary of metrics to display after the progress bar
        """
        # Update progress bar
        self.pbar.update(steps)
        
        # Update memory metrics
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Update postfix with memory info if not provided
        if postfix is None:
            postfix = {}
        
        if 'mem' not in postfix:
            postfix['mem'] = f"{current_memory:.1f}MB"
        
        # Set postfix
        if postfix:
            self.pbar.set_postfix(**postfix)
    
    def set_description(self, description: str) -> None:
        """
        Update the description of the progress bar.
        
        Args:
            description: New description text
        """
        self.description = description
        self.pbar.set_description(description)
    
    def set_postfix(self, postfix: Dict[str, Any]) -> None:
        """
        Set the postfix metrics display.
        
        Args:
            postfix: Dictionary of metrics to display
        """
        self.pbar.set_postfix(**postfix)
    
    def close(self, failed: bool = False) -> None:
        """
        Close the progress bar and compute final metrics.
        
        Args:
            failed: Whether the operation failed
        """
        # Close all child progress bars first
        for child in self.children:
            child.close(failed=failed)
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        # Update metrics
        self.metrics.update({
            'execution_time': execution_time,
            'peak_memory_mb': self.peak_memory,
            'memory_delta_mb': self.peak_memory - self.start_memory,
            'items_per_second': self.pbar.n / execution_time if execution_time > 0 else 0
        })
        
        # Change color if failed
        if failed and hasattr(self.pbar, 'colour'):
            self.pbar.colour = 'red'
        
        # Close the progress bar
        self.pbar.close()
        
        # Update parent progress if exists
        if self.parent:
            self.parent.update(1)
    
    def _get_current_memory(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def __enter__(self) -> 'ProgressTracker':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close(failed=exc_type is not None)
```

## 3. Класс `TaskProgressManager`

```python
class TaskProgressManager:
    """
    Centralized manager for task progress and logging coordination.
    
    This class coordinates progress display and logging to ensure they
    don't interfere with each other, creating a clean user experience.
    """
    
    def __init__(
            self,
            task_id: str,
            task_type: str,
            logger: logging.Logger,
            reporter: Optional[TaskReporter] = None,
            total_operations: int = 0,
            quiet: bool = False
    ):
        """
        Initialize progress manager.
        
        Args:
            task_id: Task identifier
            task_type: Type of task
            logger: Logger for the task
            reporter: Task reporter for metrics
            total_operations: Total number of operations (if known)
            quiet: Whether to disable progress bars
        """
        self.task_id = task_id
        self.task_type = task_type
        self.logger = logger
        self.reporter = reporter
        self.total_operations = total_operations
        self.quiet = quiet
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Task state tracking
        self.operations_completed = 0
        self.active_operations: Dict[str, ProgressTracker] = {}
        self.start_time = time.time()
        
        # Main progress bar for overall task progress
        self.main_progress = None
        if not quiet and total_operations > 0:
            self.main_progress = ProgressTracker(
                total=total_operations,
                description=f"Task: {task_id} ({task_type})",
                unit="operations",
                position=0,
                leave=True
            )
    
    def start_operation(
            self,
            name: str,
            total: int,
            description: Optional[str] = None,
            unit: str = "items",
            leave: bool = False
    ) -> ProgressTracker:
        """
        Start tracking a new operation.
        
        Args:
            name: Operation name (unique identifier)
            total: Total number of steps in the operation
            description: Description of the operation (defaults to name)
            unit: Unit of progress
            leave: Whether to leave the progress bar after completion
        
        Returns:
            Progress tracker for the operation
        """
        with self.lock:
            # Skip if in quiet mode
            if self.quiet:
                return ProgressTracker(
                    total=total, 
                    description=description or name,
                    unit=unit,
                    position=0,
                    leave=False
                )
            
            # Create operation description if not provided
            if description is None:
                description = f"Operation: {name}"
            
            # Calculate position based on number of active operations
            position = len(self.active_operations) + 1  # +1 because main_progress is at position 0
            
            # Create progress tracker
            progress = ProgressTracker(
                total=total,
                description=description,
                unit=unit,
                position=position,
                leave=leave,
                parent=self.main_progress
            )
            
            # Register active operation
            self.active_operations[name] = progress
            
            # Log operation start
            self.log_info(f"Starting operation: {name}")
            
            return progress
    
    def update_operation(
            self,
            name: str,
            steps: int = 1,
            postfix: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress of an operation.
        
        Args:
            name: Operation name
            steps: Number of steps completed
            postfix: Additional metrics to display
        """
        with self.lock:
            # Skip if in quiet mode
            if self.quiet:
                return
            
            # Update operation progress
            if name in self.active_operations:
                self.active_operations[name].update(steps, postfix)
    
    def complete_operation(
            self,
            name: str,
            success: bool = True,
            metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark an operation as completed.
        
        Args:
            name: Operation name
            success: Whether the operation completed successfully
            metrics: Final metrics for the operation
        """
        with self.lock:
            # Update operation counter
            self.operations_completed += 1
            
            # Skip progress update if in quiet mode
            if self.quiet:
                # Log completion
                level = logging.INFO if success else logging.ERROR
                status = "completed successfully" if success else "failed"
                self.logger.log(level, f"Operation {name} {status}")
                
                # Report metrics if available
                if self.reporter and metrics:
                    status_name = "success" if success else "error"
                    self.reporter.add_operation(
                        name=f"Complete {name}",
                        status=status_name,
                        details=metrics or {}
                    )
                return
            
            # Close progress tracker for the operation
            if name in self.active_operations:
                progress = self.active_operations[name]
                
                # Update metrics if provided
                if metrics:
                    progress.metrics.update(metrics)
                
                # Close progress tracker
                progress.close(failed=not success)
                
                # Remove from active operations
                del self.active_operations[name]
                
                # Add to reporter if available
                if self.reporter:
                    status_name = "success" if success else "error"
                    self.reporter.add_operation(
                        name=f"Complete {name}",
                        status=status_name,
                        details=progress.metrics
                    )
            
            # Log completion
            level = logging.INFO if success else logging.ERROR
            status = "completed successfully" if success else "failed"
            self.log_message(level, f"Operation {name} {status}")
    
    def log_message(
            self,
            level: int,
            message: str,
            preserve_progress: bool = True
    ) -> None:
        """
        Log a message without breaking progress bars.
        
        Args:
            level: Logging level
            message: Message to log
            preserve_progress: Whether to preserve progress bars after logging
        """
        with self.lock:
            # First, log through the logger (this goes to file)
            self.logger.log(level, message)
            
            # Then, display on console without breaking progress bars
            if self.main_progress and preserve_progress:
                # Use tqdm.write to preserve progress bars
                level_name = logging.getLevelName(level)
                self.main_progress.pbar.write(f"[{level_name}] {message}")
            else:
                # Direct output to stderr if no progress bars
                print(f"[{logging.getLevelName(level)}] {message}", file=sys.stderr)
    
    def log_info(self, message: str) -> None:
        """Convenience method for logging info messages."""
        self.log_message(logging.INFO, message)
    
    def log_warning(self, message: str) -> None:
        """Convenience method for logging warning messages."""
        self.log_message(logging.WARNING, message)
    
    def log_error(self, message: str) -> None:
        """Convenience method for logging error messages."""
        self.log_message(logging.ERROR, message)
    
    def log_debug(self, message: str) -> None:
        """Convenience method for logging debug messages."""
        self.log_message(logging.DEBUG, message)
    
    def create_operation_context(
            self,
            name: str,
            total: int,
            description: Optional[str] = None,
            unit: str = "items",
            leave: bool = False
    ) -> 'ProgressContext':
        """
        Create a context manager for an operation.
        
        Args:
            name: Operation name
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress
            leave: Whether to leave the progress bar after completion
        
        Returns:
            Context manager for the operation
        """
        return ProgressContext(
            progress_manager=self,
            operation_name=name,
            total=total,
            description=description,
            unit=unit,
            leave=leave
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get overall metrics for the task.
        
        Returns:
            Dictionary of task metrics
        """
        execution_time = time.time() - self.start_time
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'operations_completed': self.operations_completed,
            'operations_total': self.total_operations,
            'execution_time': execution_time,
            'operations_per_second': self.operations_completed / execution_time if execution_time > 0 else 0
        }
    
    def close(self) -> None:
        """Close all progress bars and release resources."""
        with self.lock:
            # Close all active operations
            for name, progress in list(self.active_operations.items()):
                progress.close()
            
            # Clear active operations
            self.active_operations.clear()
            
            # Close main progress bar
            if self.main_progress:
                self.main_progress.close()
                self.main_progress = None
    
    def __enter__(self) -> 'TaskProgressManager':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Log any exception
        if exc_type:
            self.log_error(f"Task failed with exception: {exc_val}")
        
        # Close all progress bars
        self.close()
```

## 4. Класс `ProgressContext`

```python
class ProgressContext:
    """
    Context manager for operation execution with progress tracking.
    
    This class provides a convenient way to track progress of an operation
    using the context manager pattern (with statement).
    """
    
    def __init__(
            self,
            progress_manager: TaskProgressManager,
            operation_name: str,
            total: int,
            description: Optional[str] = None,
            unit: str = "items",
            leave: bool = False
    ):
        """
        Initialize progress context.
        
        Args:
            progress_manager: Task progress manager
            operation_name: Name of the operation
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress
            leave: Whether to leave the progress bar after completion
        """
        self.progress_manager = progress_manager
        self.operation_name = operation_name
        self.total = total
        self.description = description
        self.unit = unit
        self.leave = leave
        self.tracker = None
        self.metrics = {}
    
    def __enter__(self) -> ProgressTracker:
        """
        Start tracking operation progress.
        
        Returns:
            Progress tracker for the operation
        """
        self.tracker = self.progress_manager.start_operation(
            name=self.operation_name,
            total=self.total,
            description=self.description,
            unit=self.unit,
            leave=self.leave
        )
        return self.tracker
    
    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
    ) -> None:
        """
        Complete operation tracking.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        # Determine if operation was successful
        success = exc_type is None
        
        # Collect metrics from tracker
        if self.tracker:
            self.metrics.update(self.tracker.metrics)
        
        # Add exception info to metrics if failed
        if not success:
            self.metrics['error_type'] = exc_type.__name__
            self.metrics['error_message'] = str(exc_val)
        
        # Complete operation
        self.progress_manager.complete_operation(
            name=self.operation_name,
            success=success,
            metrics=self.metrics
        )
```

## 5. Функции-утилиты

```python
def create_task_progress_manager(
        task_id: str,
        task_type: str,
        logger: logging.Logger,
        reporter: Optional[TaskReporter] = None,
        total_operations: int = 0,
        quiet: Optional[bool] = None
) -> TaskProgressManager:
    """
    Create a task progress manager.
    
    This is a convenience function for creating a TaskProgressManager 
    with optional auto-detection of quiet mode.
    
    Args:
        task_id: Task identifier
        task_type: Type of task
        logger: Logger for the task
        reporter: Task reporter for metrics
        total_operations: Total number of operations (if known)
        quiet: Whether to disable progress bars (auto-detected if None)
    
    Returns:
        Task progress manager
    """
    # Auto-detect quiet mode if not specified
    if quiet is None:
        # Detect if running in non-interactive environment
        quiet = not sys.stdout.isatty()
    
    return TaskProgressManager(
        task_id=task_id,
        task_type=task_type,
        logger=logger,
        reporter=reporter,
        total_operations=total_operations,
        quiet=quiet
    )
```

## 6. Интеграция с существующими компонентами

### 6.1. Модификация `BaseTask`

Для интеграции `progress_manager.py` с `BaseTask`, будут необходимы следующие изменения:

```python
# В методе initialize()
self.progress_manager = create_task_progress_manager(
    task_id=self.task_id,
    task_type=self.task_type,
    logger=self.logger,
    reporter=self.reporter,
    total_operations=0  # Будет обновлено после configure_operations()
)

# После configure_operations(), обновляем total_operations
if self.operations:
    self.progress_manager.total_operations = len(self.operations)
    if self.progress_manager.main_progress:
        self.progress_manager.main_progress.total = len(self.operations)

# В методе execute()
for i, operation in enumerate(self.operations):
    operation_name = operation.name if hasattr(operation, 'name') else f"Operation {i + 1}"
    
    with self.progress_manager.create_operation_context(
        name=operation_name,
        total=100,  # Если есть возможность определить total из метаданных операции
        description=f"Executing: {operation_name}"
    ) as progress:
        # Выполнение операции с передачей progress в параметры
        operation_params["progress_tracker"] = progress
        result = operation.run(**operation_params)
        
        # Добавляем метрики из результата операции в progress
        if hasattr(result, 'metrics') and result.metrics:
            for key, value in result.metrics.items():
                progress.metrics[key] = value
```

### 6.2. Модификация `operation_executor.py`

Для интеграции с `operation_executor.py`, нужны следующие изменения:

```python
# В методе __init__()
self.progress_manager = kwargs.get('progress_manager', None)

# В методе execute_with_retry()
if self.progress_manager:
    context_name = f"{operation.__class__.__name__}_{id(operation)}"
    with self.progress_manager.create_operation_context(
        name=context_name,
        total=max_retries + 1,  # Общее число попыток 
        description=f"Executing with retry: {operation.__class__.__name__}",
        unit="attempts"
    ) as progress:
        # Внутри цикла retry
        for attempt in range(max_retries + 1):
            try:
                # Обновляем description с номером текущей попытки
                if attempt > 0:
                    progress.set_description(
                        f"Retry {attempt}/{max_retries}: {operation.__class__.__name__}"
                    )
                
                # Выполняем операцию
                result = self.execute_operation(
                    operation=operation,
                    params=params,
                    progress_tracker=params.get('progress_tracker')
                )
                
                # Обновляем прогресс и метрики
                progress.update(1, {
                    'status': result.status.name if hasattr(result.status, 'name') else str(result.status)
                })
                
                # В случае успеха, выходим из цикла
                if result.status != OperationStatus.ERROR:
                    progress.metrics.update({
                        'attempts': attempt + 1,
                        'success': True,
                        'execution_time': result.execution_time
                    })
                    return result
                
                # Операция завершилась с ошибкой, но без исключения - продолжаем retry
                continue
                
            except Exception as e:
                # Обновляем прогресс и метрики
                progress.update(1, {'status': 'error'})
                
                # Проверяем, можно ли делать retry
                if not self.is_retriable_error(e) or attempt >= max_retries:
                    # Не можем делать retry - завершаем с ошибкой
                    progress.metrics.update({
                        'attempts': attempt + 1,
                        'success': False,
                        'error_type': e.__class__.__name__,
                        'error_message': str(e)
                    })
                    raise
                
                # Ждем перед следующей попыткой
                wait_time = self._calculate_wait_time(attempt + 1, backoff_factor, initial_wait, max_wait, jitter)
                progress.set_postfix({'wait': f"{wait_time:.1f}s"})
                time.sleep(wait_time)
else:
    # Существующая логика без использования progress_manager
```

## 7. Рекомендации по тестированию

1. **Базовые тесты функциональности**:
   - Тест создания `TaskProgressManager` и его основных методов
   - Проверка корректного отображения прогресс-баров
   - Тест работы логгирования через `log_message`

2. **Тесты интеграции с существующими компонентами**:
   - Интеграция с `BaseTask`
   - Интеграция с `operation_executor.py`
   - Перенаправление логов и совместимость с системой логирования

3. **Тесты в различных режимах**:
   - Проверка работы в тихом режиме (`quiet=True`)
   - Тестирование при различном количестве операций
   - Тестирование при асинхронных операциях

4. **Тесты обработки ошибок**:
   - Проверка корректного отображения ошибок в прогресс-барах
   - Тестирование блокировок при многопоточной работе

## 8. Заключение

Предложенный подход к реализации `progress_manager.py` полностью соответствует требованиям, указанным в `progress_update.md`, и учитывает дополнительные ограничения:

1. **Разделение потоков вывода**: 
   - Логи направляются в `stderr` через `tqdm.write` и существующий `logger`
   - Прогресс-бары выводятся в `stdout` с фиксированными позициями

2. **Иерархическая структура прогресса**:
   - Основной прогресс-бар задачи на позиции 0
   - Прогресс-бары операций на позициях 1+ с возможностью скрытия после завершения

3. **Координация логирования**:
   - Метод `log_message` обеспечивает корректный вывод логов без нарушения прогресс-баров
   - Поддержка различных уровней логирования и сохранение в файл через `logger`

4. **Контекстные менеджеры**:
   - Возможность использования `with` для отслеживания операций
   - Автоматическое закрытие и обновление прогресс-баров при выходе из контекста

5. **Интеграция с существующими компонентами**:
   - Корректная работа с `TaskReporter` для сохранения метрик
   - Совместимость с существующим API модулей `BaseTask` и `operation_executor.py`

6. **Потокобезопасность**:
   - Использование `threading.Lock` для блокировки критических секций

Данная реализация позволит значительно улучшить пользовательский опыт при работе с задачами PAMOLA.CORE, обеспечивая четкое отображение прогресса и понятный вывод логов.