# План улучшения модулей pamola_core.utils.ops в PAMOLA.CORE

## 1. Краткосрочные улучшения (без изменения API)

### 1.1. Рефакторинг op_data_source.py

```python
# Рефакторинг метода get_dataframe для использования словаря loader'ов
def get_dataframe(self, name: str, load_if_path: bool = True) -> Optional[pd.DataFrame]:
    """
    Get a DataFrame by name.
    
    If the DataFrame is not in memory but a file path with the same name
    exists, it will be loaded (if load_if_path is True).
    
    Parameters:
    -----------
    name : str
        Name of the DataFrame
    load_if_path : bool
        Whether to load from file if DataFrame is not in memory
        
    Returns:
    --------
    pd.DataFrame or None
        The requested DataFrame, or None if not found
    """
    # First check if DataFrame is already in memory
    if name in self.dataframes:
        return self.dataframes[name]
        
    # If not, check if file path exists and should be loaded
    if load_if_path and name in self.file_paths:
        file_path = self.file_paths[name]
        if file_path.exists():
            # Use dictionary for file format handling
            loaders = {
                '.csv': lambda path: read_full_csv(path),
                '.parquet': lambda path: pd.read_parquet(path),
                '.xlsx': lambda path: pd.read_excel(path),
                '.xls': lambda path: pd.read_excel(path),
                '.json': lambda path: pd.read_json(path),
                '.txt': lambda path: pd.read_csv(path, sep='\t')
            }
            
            try:
                ext = file_path.suffix.lower()
                loader = loaders.get(ext)
                
                if loader:
                    df = loader(file_path)
                    # Store in memory for future use
                    self.dataframes[name] = df
                    return df
                else:
                    logger.warning(f"Unsupported file format: {ext}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                return None
                
    return None
```

### 1.2. Улучшение обработки ошибок в op_base.py

```python
# Улучшение метода _log_operation_end для более надежной обработки
def _log_operation_end(self, result: OperationResult):
    """Log the end of an operation with results."""
    if not hasattr(result, 'status'):
        logger.error(f"Invalid OperationResult object: missing status attribute")
        return
        
    self.logger.info(f"Operation {self.name} completed with status: {result.status.name}")
    
    # Безопасно получаем время выполнения
    execution_time = getattr(result, 'execution_time', None)
    if execution_time is not None:
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        
    # Безопасно получаем сообщение об ошибке
    if result.status == OperationStatus.ERROR:
        error_message = getattr(result, 'error_message', 'Unknown error')
        self.logger.error(f"Error: {error_message}")
```

### 1.3. Оптимизация кеширования в op_cache.py

```python
# Оптимизация _reduce_cache_size для более эффективной работы с файлами
def _reduce_cache_size(self, current_size: int) -> None:
    """
    Reduce cache size by removing oldest files.
    
    Parameters:
    -----------
    current_size : int
        Current cache size in bytes
    """
    # Получаем информацию о файлах за один проход
    cache_files = []
    
    for file in self.cache_dir.glob("**/*.json"):
        try:
            mod_time = os.path.getmtime(file)
            size = os.path.getsize(file)
            cache_files.append((file, mod_time, size))
        except Exception as e:
            logger.warning(f"Error checking cache file {file}: {e}")
    
    # Сортировка по времени изменения (сначала старые)
    cache_files.sort(key=lambda x: x[1])
    
    # Удаляем файлы до достижения целевого размера
    target_size = self.max_size_bytes * 0.8  # Целевой размер 80% от максимума
    removed_size = 0
    files_removed = 0
    
    for file, _, size in cache_files:
        if current_size - removed_size <= target_size:
            break
            
        try:
            os.remove(file)
            removed_size += size
            files_removed += 1
        except Exception as e:
            logger.warning(f"Error removing cache file {file}: {e}")
    
    if files_removed > 0:
        logger.info(f"Removed {files_removed} cache files to reduce cache size")
```

### 1.4. Улучшение документации в op_result.py

```python
def add_artifact(self,
                 artifact_type: str,
                 path: Union[str, Path],
                 description: str = "",
                 category: str = "output",
                 tags: Optional[List[str]] = None,
                 group: Optional[str] = None) -> OperationArtifact:
    """
    Add an artifact to the result.
    
    This method creates an OperationArtifact and adds it to the result.
    Artifacts represent files produced by the operation, such as output data,
    metrics, or visualizations.
    
    Parameters:
    -----------
    artifact_type : str
        Type of artifact (e.g., "json", "csv", "png"). This should match the file extension.
    path : Union[str, Path]
        Path to the artifact file. Can be a string or pathlib.Path object.
    description : str, optional
        Human-readable description of the artifact. Default is empty string.
    category : str, optional
        Category of the artifact for organization (e.g., "output", "metric", "visualization").
        Default is "output".
    tags : List[str], optional
        Tags for categorizing the artifact. Default is None.
    group : str, optional
        Name of the group to add this artifact to. If the group doesn't exist,
        it will be created. Default is None.
        
    Returns:
    --------
    OperationArtifact
        The created artifact object.
    
    Examples:
    ---------
    >>> result = OperationResult(status=OperationStatus.SUCCESS)
    >>> result.add_artifact("csv", "output/data.csv", "Anonymized data")
    >>> result.add_artifact("png", "visualizations/histogram.png", "Distribution visualization", 
    ...                    category="visualization", tags=["histogram", "distribution"])
    """
    # Existing implementation...
```

## 2. Среднесрочные улучшения (минимальные изменения API)

### 2.1. Добавить класс op_config.py для управления конфигурацией операций

```python
"""
Operation configuration utilities for the PAMOLA.CORE project.

This module provides classes for managing operation configuration,
including parameter validation, storage, and serialization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union

import jsonschema

# Type variable for configuration classes
T = TypeVar('T')

class OperationConfig(Generic[T]):
    """
    Base class for operation configuration.
    
    This class provides utilities for validating, storing, and serializing
    operation configuration parameters.
    """
    
    # JSON Schema for configuration validation
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {}
    }
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters
        """
        self._validate_params(kwargs)
        self._params = kwargs
        
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate parameters against the schema.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        try:
            jsonschema.validate(instance=params, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid configuration: {e.message}")
            
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.
        
        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the configuration file
        """
        path = Path(path) if isinstance(path, str) else path
        with open(path, 'w') as f:
            json.dump(self._params, f, indent=2)
            
    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        """
        Load configuration from a JSON file.
        
        Parameters:
        -----------
        path : Union[str, Path]
            Path to the configuration file
            
        Returns:
        --------
        OperationConfig
            Loaded configuration
        """
        path = Path(path) if isinstance(path, str) else path
        with open(path, 'r') as f:
            params = json.load(f)
        return cls(**params)
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.
        
        Parameters:
        -----------
        key : str
            Parameter name
        default : Any, optional
            Default value if parameter is not found
            
        Returns:
        --------
        Any
            Parameter value or default
        """
        return self._params.get(key, default)
        
    def __getitem__(self, key: str) -> Any:
        """Get a parameter by key."""
        return self._params[key]
        
    def __contains__(self, key: str) -> bool:
        """Check if a parameter exists."""
        return key in self._params
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        return self._params.copy()
```

### 2.2. Добавить поддержку контекстного менеджера для OperationCache

```python
# Добавить в op_cache.py
def __enter__(self):
    """
    Enter context manager protocol.
    
    This allows using the cache with 'with' statements:
    
    with OperationCache() as cache:
        result = cache.get_cache(key)
        
    Returns:
    --------
    OperationCache
        The cache instance
    """
    return self
    
def __exit__(self, exc_type, exc_val, exc_tb):
    """
    Exit context manager protocol.
    
    This ensures cache health is checked when the context is exited.
    
    Parameters:
    -----------
    exc_type : Type[BaseException] or None
        Exception type if an exception was raised, None otherwise
    exc_val : BaseException or None
        Exception instance if an exception was raised, None otherwise
    exc_tb : traceback or None
        Traceback if an exception was raised, None otherwise
        
    Returns:
    --------
    bool
        False to propagate exceptions
    """
    self._check_cache_health()
    return False  # Не подавляем исключения
```

### 2.3. Улучшить логирование в op_base.py с фильтрацией чувствительных параметров

```python
def _log_operation_start(self, **kwargs):
    """Log the start of an operation with parameters."""
    self.logger.info(f"Starting operation: {self.name}")
    
    # Список параметров, которые не следует логировать (чувствительные данные)
    sensitive_params = ['encryption_key', 'password', 'token', 'secret', 'api_key']
    
    # Логируем параметры, пропуская чувствительные
    for key, value in kwargs.items():
        if key in sensitive_params:
            self.logger.debug(f"Parameter {key}: [REDACTED]")
        else:
            # Обрезаем длинные значения для читаемости логов
            if isinstance(value, str) and len(value) > 100:
                self.logger.debug(f"Parameter {key}: {value[:100]}... [truncated]")
            else:
                self.logger.debug(f"Parameter {key}: {value}")
```

## 3. Долгосрочные улучшения (заметные изменения API)

### 3.1. TODO: Добавить поддержку асинхронных операций

```python
# TODO: Реализовать асинхронные версии методов execute и run в BaseOperation
# Примерный интерфейс:
"""
async def execute_async(self, 
                     data_source: DataSource,
                     task_dir: Path,
                     reporter: Any,
                     progress_tracker: Optional[ProgressTracker] = None,
                     **kwargs) -> OperationResult:
    \"""
    Execute the operation asynchronously.
    
    This is the async version of the execute method. Subclasses should override
    this method to provide async functionality if needed.
    \"""
    # Default implementation: запустить синхронный execute в executor
    
async def run_async(self,
                 data_source: DataSource,
                 task_dir: Path,
                 reporter: Any,
                 **kwargs) -> OperationResult:
    \"""
    Run the operation asynchronously with timing and error handling.
    
    This is the async version of the run method.
    \"""
    # Implementation similar to run() but using async/await
"""
```

### 3.2. TODO: Добавить поддержку распределенных вычислений с Dask

```python
# TODO: Добавить в DataSource поддержку Dask DataFrames
"""
def get_dask_dataframe(self, name: str) -> Optional['dask.dataframe.DataFrame']:
    \"""
    Get a Dask DataFrame by name, converting from pandas if necessary.
    
    Requires dask to be installed.
    \"""
    # Implementation
    
def add_dask_dataframe(self, name: str, df: 'dask.dataframe.DataFrame'):
    \"""
    Add a Dask DataFrame to the data source.
    
    Requires dask to be installed.
    \"""
    # Implementation
"""
```

### 3.3. TODO: Расширить функциональность OperationResult для поддержки потоковой обработки

```python
# TODO: Добавить поддержку потоковых результатов в OperationResult
"""
def create_stream_result(self) -> 'OperationStreamResult':
    \"""
    Create a stream result for incremental processing.
    
    Returns:
    --------
    OperationStreamResult
        A stream result object that can be updated incrementally.
    \"""
    # Implementation

class OperationStreamResult:
    \"""
    Represents a streaming result from an operation.
    
    This allows operations to report progress and partial results
    during long-running operations.
    \"""
    
    def update_progress(self, progress: float, message: str = None):
        \"""
        Update the progress of the operation.
        
        Parameters:
        -----------
        progress : float
            Progress value between 0 and 1
        message : str, optional
            Progress message
        \"""
        # Implementation
        
    def add_partial_artifact(self, artifact_type: str, path: Union[str, Path], description: str = ""):
        \"""
        Add a partial artifact to the result.
        
        Parameters:
        -----------
        artifact_type : str
            Type of artifact
        path : Union[str, Path]
            Path to the artifact
        description : str, optional
            Description of the artifact
        \"""
        # Implementation
"""
```

### 3.4. TODO: Создать систему плагинов для операций

```python
# TODO: Реализовать систему плагинов для динамической загрузки операций
"""
class OperationPlugin:
    \"""
    Base class for operation plugins.
    
    This allows extending the operation framework with new operations
    without modifying the pamola core code.
    \"""
    
    @classmethod
    def register_operations(cls) -> List[Type[BaseOperation]]:
        \"""
        Register operations provided by this plugin.
        
        Returns:
        --------
        List[Type[BaseOperation]]
            List of operation classes to register
        \"""
        # Implementation

def load_plugins(plugin_dir: Optional[Union[str, Path]] = None) -> int:
    \"""
    Load operation plugins from the specified directory.
    
    Parameters:
    -----------
    plugin_dir : Union[str, Path], optional
        Directory containing plugins. If None, uses default plugin directory.
        
    Returns:
    --------
    int
        Number of plugins loaded
    \"""
    # Implementation
"""
```

## Заключение

План улучшений модулей pamola_core.utils.ops в PAMOLA.CORE состоит из трех частей:

1. **Краткосрочные улучшения**: Рефакторинг существующего кода для улучшения производительности, читаемости и обработки ошибок без изменения API. Эти изменения могут быть внедрены безопасно с минимальным риском нарушения обратной совместимости.

2. **Среднесрочные улучшения**: Добавление новых возможностей с минимальными изменениями API, включая лучшее управление конфигурацией и поддержку контекстных менеджеров. Эти изменения расширяют функциональность, сохраняя обратную совместимость с существующим кодом.

3. **Долгосрочные улучшения**: Обозначены как TODO, включают более значительные изменения API для поддержки асинхронных операций, распределенных вычислений и потоковой обработки результатов. Эти улучшения требуют тщательного планирования и могут быть реализованы постепенно с сохранением обратной совместимости через соответствующие адаптеры или прослойки.

Предложенный план обеспечивает постепенное улучшение кодовой базы с поддержанием обратной совместимости и позволяет планировать более существенные изменения в будущем.