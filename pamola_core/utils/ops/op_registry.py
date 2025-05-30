"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Registry
Description: Registration and discovery system for operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for registering, discovering, and retrieving
operation classes throughout the PAMOLA Core framework. It allows operations to be
referenced by name and provides metadata about available operations.

Key features:
- Operation registration and discovery (REQ-OPS-007)
- Version management and compatibility checking
- Metadata extraction and categorization
- Dependency resolution
- Automatic operation instantiation

Implementation notes:
- Uses functional API for registry operations to maintain backward compatibility
- Global registry dictionaries for storing operation classes and metadata
- Support for semantic versioning and dependency checking

TODO:
- Consider class-based wrapper for functional API in future major versions
- Migrate inline version validation to pamola_core.utils.io.version_utils (future)
"""

import importlib
import inspect
import logging
import pkgutil
import re
from typing import Dict, List, Any, Type, Optional, Tuple

from packaging import version

# Configure logger
logger = logging.getLogger(__name__)

# Global registry for operation classes
_OPERATION_REGISTRY = {}
_OPERATION_METADATA = {}
_OPERATION_DEPENDENCIES = {}
_OPERATION_VERSIONS = {}


class OpsError(Exception):
    """Base class for all operation-related errors."""
    pass


class RegistryError(OpsError):
    """Errors related to operation registration and lookup."""
    pass


def _is_base_operation(operation_class) -> bool:
    """
    Duck-typing check: is the class a subclass of BaseOperation
    within pamola_core.utils.ops (needed to avoid direct import of BaseOperation).

    Parameters:
    -----------
    operation_class : Type
        Class to check

    Returns:
    --------
    bool
        True if the class inherits from BaseOperation, False otherwise
    """
    return any(
        base.__name__ == "BaseOperation" and base.__module__.startswith("pamola_core.utils.ops")
        for base in operation_class.__mro__
    )

def register_operation(operation_class,
                       override: bool = False,
                       dependencies: Optional[List[Dict[str, str]]] = None,
                       version: Optional[str] = None) -> bool:
    """
    Register an operation class in the registry.

    Satisfies REQ-OPS-007: Provides operation registration for discovery.

    Parameters:
    -----------
    operation_class : Type
        The operation class to register
    override : bool
        Whether to override an existing registration with the same name
    dependencies : List[Dict[str, str]], optional
        List of dependencies for the operation, each with 'name' and 'version' keys
    version : str, optional
        Version of the operation (defaults to the class's version attribute if present)

    Returns:
    --------
    bool
        True if registration was successful, False otherwise
    """
    # Check if class is a BaseOperation subclass
    if not _is_base_operation(operation_class):
        logger.error(f"Cannot register {operation_class.__name__}: not a subclass of BaseOperation")
        return False

    # Get operation name
    operation_name = operation_class.__name__

    # Check if already registered
    if operation_name in _OPERATION_REGISTRY and not override:
        # Changed from warning to debug level to reduce log noise when operation is registered multiple times
        # This allows silent re-registration of the same operation for different fields/datasets
        logger.debug(f"Operation {operation_name} already registered and override=False, skipping registration")
        return True  # Return True to indicate operation is available without error

    # Register the operation class
    _OPERATION_REGISTRY[operation_name] = operation_class

    # Extract and register metadata
    _OPERATION_METADATA[operation_name] = {
        'module': operation_class.__module__,
        'class': operation_class.__name__,
        'category': _determine_operation_category(operation_class),
        'parameters': _extract_init_parameters(operation_class)
    }

    # Register dependencies
    if dependencies:
        _OPERATION_DEPENDENCIES[operation_name] = dependencies
    else:
        _OPERATION_DEPENDENCIES[operation_name] = []

    # Register version
    if version:
        _OPERATION_VERSIONS[operation_name] = version
    elif hasattr(operation_class, 'version'):
        _OPERATION_VERSIONS[operation_name] = operation_class.version
    else:
        _OPERATION_VERSIONS[operation_name] = '1.0.0'

    logger.debug(f"Registered operation: {operation_name} (version: {_OPERATION_VERSIONS[operation_name]})")
    return True


def unregister_operation(operation_name: str) -> bool:
    """
    Unregister an operation from the registry.

    Parameters:
    -----------
    operation_name : str
        Name of the operation to unregister

    Returns:
    --------
    bool
        True if unregistered successfully, False if not found
    """
    if operation_name in _OPERATION_REGISTRY:
        del _OPERATION_REGISTRY[operation_name]
        if operation_name in _OPERATION_METADATA:
            del _OPERATION_METADATA[operation_name]
        if operation_name in _OPERATION_DEPENDENCIES:
            del _OPERATION_DEPENDENCIES[operation_name]
        if operation_name in _OPERATION_VERSIONS:
            del _OPERATION_VERSIONS[operation_name]
        logger.debug(f"Unregistered operation: {operation_name}")
        return True
    else:
        logger.warning(f"Cannot unregister: operation {operation_name} not found in registry")
        return False


def get_operation_class(operation_name: str) -> Optional[Type]:
    """
    Get an operation class by name.

    Parameters:
    -----------
    operation_name : str
        Name of the operation to retrieve

    Returns:
    --------
    Optional[Type]
        The requested operation class, or None if not found
    """
    return _OPERATION_REGISTRY.get(operation_name)


def get_operation_metadata(operation_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for an operation.

    Parameters:
    -----------
    operation_name : str
        Name of the operation to get metadata for

    Returns:
    --------
    Optional[Dict[str, Any]]
        Metadata dictionary, or None if operation not found
    """
    return _OPERATION_METADATA.get(operation_name)


def get_operation_version(operation_name: str) -> Optional[str]:
    """
    Get the version of an operation.

    Parameters:
    -----------
    operation_name : str
        Name of the operation

    Returns:
    --------
    Optional[str]
        Version string, or None if operation not found
    """
    return _OPERATION_VERSIONS.get(operation_name)


def get_operation_dependencies(operation_name: str) -> List[Dict[str, str]]:
    """
    Get the dependencies of an operation.

    Parameters:
    -----------
    operation_name : str
        Name of the operation

    Returns:
    --------
    List[Dict[str, str]]
        List of dependencies, each with 'name' and 'version' keys
    """
    return _OPERATION_DEPENDENCIES.get(operation_name, [])


def check_dependencies(operation_name: str) -> Tuple[bool, List[str]]:
    """
    Check if all dependencies of an operation are satisfied.

    Parameters:
    -----------
    operation_name : str
        Name of the operation

    Returns:
    --------
    Tuple[bool, List[str]]
        (all_satisfied, unsatisfied_dependencies)
    """
    dependencies = get_operation_dependencies(operation_name)
    if not dependencies:
        return True, []

    unsatisfied = []

    for dep in dependencies:
        dep_name = dep.get('name')
        dep_constraint = dep.get('version', '>=1.0.0')

        # Check if dependency exists
        if dep_name not in _OPERATION_REGISTRY:
            unsatisfied.append(f"{dep_name} not found")
            continue

        # Check version compatibility
        dep_version = get_operation_version(dep_name)
        if not check_version_compatibility(dep_version, dep_constraint):
            unsatisfied.append(f"{dep_name} version {dep_version} does not satisfy {dep_constraint}")

    return len(unsatisfied) == 0, unsatisfied


def check_version_compatibility(version_str: str, constraint: str) -> bool:
    """
    Check if a version string satisfies a version constraint.

    Parameters:
    -----------
    version_str : str
        Version to check
    constraint : str
        Version constraint (e.g., ">=1.0.0", "1.x.x")

    Returns:
    --------
    bool
        True if version satisfies constraint, False otherwise
    """
    try:
        # Handle wildcards in constraint
        if 'x' in constraint.lower() or '*' in constraint:
            return _check_wildcard_compatibility(version_str, constraint)

        # Handle comparison operators
        if any(op in constraint for op in ['>=', '<=', '>', '<', '==']):
            op, ver = _parse_version_constraint(constraint)
            return _compare_versions(version_str, op, ver)

        # Default to exact match
        return version_str == constraint
    except Exception as e:
        logger.error(f"Error checking version compatibility: {e}")
        return False


def list_operations(category: Optional[str] = None) -> List[str]:
    """
    List all registered operations, optionally filtered by category.

    Parameters:
    -----------
    category : str, optional
        Category to filter by

    Returns:
    --------
    List[str]
        List of operation names
    """
    if category is None:
        return list(_OPERATION_REGISTRY.keys())

    return [
        name for name, metadata in _OPERATION_METADATA.items()
        if metadata.get('category') == category
    ]


def list_categories() -> List[str]:
    """
    List all available operation categories.

    Returns:
    --------
    List[str]
        List of distinct operation categories
    """
    categories = set()
    for metadata in _OPERATION_METADATA.values():
        if 'category' in metadata:
            categories.add(metadata['category'])

    return sorted(list(categories))


def get_operations_by_category() -> Dict[str, List[str]]:
    """
    Get all operations organized by category.

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping categories to lists of operation names
    """
    result = {}

    for op_name, metadata in _OPERATION_METADATA.items():
        category = metadata.get('category', 'general')
        if category not in result:
            result[category] = []
        result[category].append(op_name)

    # Sort the operation lists
    for category in result:
        result[category].sort()

    return result


def lazily_load_operation(operation_name: str) -> bool:
    """
    Attempt to lazily load an operation by trying to import modules that might contain it.

    This function tries to locate and import a module that might contain the specified
    operation by checking common package paths.

    Parameters:
    -----------
    operation_name : str
        Name of the operation to load

    Returns:
    --------
    bool
        True if the operation was successfully loaded, False otherwise
    """
    # Common package paths where operations might be defined
    common_paths = [
        f"pamola_core.profiling.analyzers.{operation_name.lower().replace('operation', '')}",
        f"pamola_core.profiling.analyzers.{operation_name.lower()}",
        f"pamola_core.anonymization.{operation_name.lower().replace('operation', '')}",
        f"pamola_core.anonymization.{operation_name.lower()}",
        f"pamola_core.metrics.operations.{operation_name.lower().replace('operation', '')}",
        f"pamola_core.metrics.operations.{operation_name.lower()}",
        f"pamola_core.fake_data.operations.{operation_name.lower().replace('operation', '')}"
    ]

    # Try to guess the module based on class name
    # For example, "GroupAnalyzerOperation" might be in "group.py"
    operation_name_lower = operation_name.lower()
    if "analyzer" in operation_name_lower:
        module_name = operation_name_lower.replace("analyzer", "").replace("operation", "")
        if module_name:
            common_paths.append(f"pamola_core.profiling.analyzers.{module_name}")

    # Try importing modules in order
    for module_path in common_paths:
        try:
            logger.debug(f"Attempting to lazily load operation {operation_name} from {module_path}")
            importlib.import_module(module_path)

            # Check if operation is now registered
            if operation_name in _OPERATION_REGISTRY:
                logger.info(f"Successfully loaded operation {operation_name} from {module_path}")
                return True
        except ImportError:
            # Expected when module doesn't exist
            continue
        except Exception as e:
            # Unexpected error during import
            logger.warning(f"Error importing module {module_path}: {str(e)}")

    # If the operation couldn't be loaded from any of the paths
    logger.warning(f"Failed to lazily load operation {operation_name}")
    return False


def create_operation_instance(operation_name: str, **kwargs) -> Optional[Any]:
    """
    Create an instance of an operation by name.

    Parameters:
    -----------
    operation_name : str
        Name of the operation to create
    **kwargs : dict
        Parameters to pass to the operation constructor

    Returns:
    --------
    Optional[Any]
        The created operation instance, or None if operation not found
    """
    operation_class = get_operation_class(operation_name)

    # If operation not found, try lazy loading
    if operation_class is None:
        lazily_load_operation(operation_name)
        # Try again after lazy loading
        operation_class = get_operation_class(operation_name)

    if operation_class is None:
        logger.error(f"Operation {operation_name} not found in registry")
        return None

    # Check dependencies
    deps_satisfied, unsatisfied = check_dependencies(operation_name)
    if not deps_satisfied:
        logger.error(f"Cannot create {operation_name}: dependencies not satisfied: {', '.join(unsatisfied)}")
        return None

    try:
        instance = operation_class(**kwargs)
        return instance
    except Exception as e:
        logger.error(f"Error creating instance of {operation_name}: {e}")
        return None


def discover_operations(package_name: str = 'pamola_core') -> int:
    """
    Discover and register operations from a package.

    Recursively imports modules in the specified package and registers
    all found operation classes.

    Parameters:
    -----------
    package_name : str
        Name of the package to scan for operations

    Returns:
    --------
    int
        Number of operations registered
    """
    count = 0

    try:
        package = importlib.import_module(package_name)
        package_path = getattr(package, '__path__', None)

        if package_path:
            # Process subpackages recursively
            for _, name, is_pkg in pkgutil.iter_modules(package_path):
                full_name = f"{package_name}.{name}"

                if is_pkg:
                    # Recursively process subpackage
                    count += discover_operations(full_name)
                else:
                    # Import module and register operations
                    try:
                        module = importlib.import_module(full_name)

                        # Find all BaseOperation subclasses in the module
                        # using duck typing instead of direct import
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                    obj.__module__ == module.__name__):

                                # Check if it looks like a BaseOperation
                                if any(base.__name__ == 'BaseOperation' for base in obj.__mro__ if base != obj):
                                    # Extract version from class attribute
                                    version = getattr(obj, 'version', '1.0.0')

                                    # Extract dependencies from class attribute if present
                                    dependencies = getattr(obj, 'dependencies', None)

                                    if register_operation(obj, version=version, dependencies=dependencies):
                                        count += 1

                    except Exception as e:
                        logger.warning(f"Error importing module {full_name}: {e}")

    except Exception as e:
        logger.error(f"Error discovering operations in package {package_name}: {e}")

    return count


def initialize_registry() -> int:
    """
    Initialize the registry by discovering all operations in the PAMOLA Core package.

    Returns:
    --------
    int
        Number of operations registered
    """
    # Clear existing registry
    _OPERATION_REGISTRY.clear()
    _OPERATION_METADATA.clear()
    _OPERATION_DEPENDENCIES.clear()
    _OPERATION_VERSIONS.clear()

    # Discover operations
    return discover_operations('core')

#Decorators
def register(override: bool = False, version: str = None, dependencies: List[Dict[str, str]] = None):
    """
    Decorator to register an operation class.

    Parameters:
    -----------
    override : bool
        Whether to override existing registrations
    version : str, optional
        Version of the operation
    dependencies : List[Dict[str, str]], optional
        List of dependencies for the operation

    Returns:
    --------
    Callable
        Decorator function
    """

    def decorator(cls):
        register_operation(cls, override, dependencies, version)
        return cls

    return decorator

# Private helper functions
def _is_valid_semver(ver_str: str) -> bool:
    """
    Check if a string is a valid semantic version.

    Parameters:
    -----------
    ver_str : str
        Version string to check

    Returns:
    --------
    bool
        True if valid semantic version format, False otherwise
    """
    # Simple regex for semver (major.minor.patch with optional pre-release and build metadata)
    semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'

    # Also check if it has "x" as a placeholder (e.g., "1.x.x")
    wildcard_pattern = r'^(0|[1-9]\d*|x|X|\*)\.((0|[1-9]\d*|x|X|\*)\.(0|[1-9]\d*|x|X|\*))$'

    return bool(re.match(semver_pattern, ver_str)) or bool(re.match(wildcard_pattern, ver_str))


def _extract_init_parameters(operation_class: Type) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameter information from the operation class's __init__ method.

    Parameters:
    -----------
    operation_class : Type
        The operation class to extract parameters from

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary mapping parameter names to their metadata
    """
    try:
        # Get the __init__ method signature
        signature = inspect.signature(operation_class.__init__)
        parameters = {}

        # Skip 'self' parameter
        for name, param in signature.parameters.items():
            if name == 'self':
                continue

            # Extract parameter metadata
            parameters[name] = {
                'default': None if param.default is inspect.Parameter.empty else param.default,
                'has_default': param.default is inspect.Parameter.empty,
                'is_required': param.default is inspect.Parameter.empty,
                'annotation': str(param.annotation) if param.annotation is not inspect.Parameter.empty else None
            }

        return parameters
    except Exception as e:
        logger.error(f"Error extracting parameters from {operation_class.__name__}: {e}")
        return {}


def _determine_operation_category(operation_class: Type) -> str:
    """
    Determine the category of an operation based on its class hierarchy and module.

    Parameters:
    -----------
    operation_class : Type
        The operation class to categorize

    Returns:
    --------
    str
        The determined category
    """
    # Check module path first
    module = operation_class.__module__

    if 'profiling' in module:
        return 'profiling'
    elif 'anonymization' in module:
        return 'anonymization'
    elif 'security' in module:
        return 'security'
    elif 'quality' in module:
        return 'quality'
    elif 'visualization' in module:
        return 'visualization'

    # Check base classes
    for base in operation_class.__mro__:
        if base.__name__ == 'FieldOperation':
            return 'field'
        elif base.__name__ == 'DataFrameOperation':
            return 'dataframe'

    # Default category
    return 'general'


def _parse_version_constraint(constraint: str) -> Tuple[str, str]:
    """
    Parse a version constraint into operator and version.

    Parameters:
    -----------
    constraint : str
        Version constraint (e.g., ">=1.0.0")

    Returns:
    --------
    Tuple[str, str]
        (operator, version)
    """
    for op in ['>=', '<=', '==', '>', '<']:
        if constraint.startswith(op):
            return op, constraint[len(op):]

    # Default to equality
    return '==', constraint


def _compare_versions(version_str: str, op: str, constraint_ver: str) -> bool:
    """
    Compare two versions using the specified operator.

    Parameters:
    -----------
    version_str : str
        Version to check
    op : str
        Comparison operator
    constraint_ver : str
        Version to compare against

    Returns:
    --------
    bool
        Result of the comparison
    """
    ver1 = version.parse(version_str)
    ver2 = version.parse(constraint_ver)

    if op == '>=':
        return ver1 >= ver2
    elif op == '<=':
        return ver1 <= ver2
    elif op == '>':
        return ver1 > ver2
    elif op == '<':
        return ver1 < ver2
    elif op == '==':
        return ver1 == ver2

    return False


def _check_wildcard_compatibility(version_str: str, constraint: str) -> bool:
    """
    Check if a version satisfies a wildcard constraint.

    Parameters:
    -----------
    version_str : str
        Version to check
    constraint : str
        Wildcard constraint (e.g., "1.x.x")

    Returns:
    --------
    bool
        True if version satisfies constraint, False otherwise
    """
    # Replace x, X, or * with wildcards
    pattern = constraint.lower().replace('x', '\\d+').replace('*', '\\d+')
    # Replace dots with escaped dots
    pattern = pattern.replace('.', '\\.')
    # Create regex pattern
    regex = re.compile(f'^{pattern}$')

    return bool(regex.match(version_str))