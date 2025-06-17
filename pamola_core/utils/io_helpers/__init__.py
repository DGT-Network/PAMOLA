"""
Helper modules for the PAMOLA CORE I/O utilities.

This package contains helper modules that provide specialized functionality
for the main I/O module. These helpers handle integration with optional
dependencies like Dask, encryption/decryption, and support for various
file formats.
"""

# Import directory management utilities
from pamola_core.utils.io_helpers.directory_utils import (
    ensure_directory,
    get_timestamped_filename,
    get_file_stats,
    list_directory_contents,
    clear_directory,
    make_unique_path,
    create_secure_temp_directory,
    create_secure_temp_file,
    safe_remove_temp_file,
    normalize_path,
    is_path_in_directory,
    create_secure_path,
    is_path_writable,
    protect_path,
    secure_cleanup,
    get_temp_file_for_decryption,
    get_temp_file_for_encryption,
    ensure_parent_directory,
    get_unique_filename,
    create_secure_directory_structure
)

# Import CSV utilities
from pamola_core.utils.io_helpers.csv_utils import (
    estimate_csv_size,
    prepare_csv_reader_options,
    prepare_csv_writer_options,
    count_csv_lines,
    validate_csv_structure,
    monitor_csv_operation,
    report_memory_usage,
    filter_csv_columns,
    detect_csv_dialect,
    get_optimal_csv_chunk_size,
    read_csv_in_efficient_chunks,
    optimize_csv_datatypes,
    validate_csv_file
)

# Import JSON utilities
from pamola_core.utils.io_helpers.json_utils import (
    convert_numpy_types,
    validate_json_structure,
    merge_json_objects_in_memory,
    prettify_json,
    detect_array_or_object,
    extract_json_subset,
    prepare_json_writer_options
)

# Import format utilities
from pamola_core.utils.io_helpers.format_utils import (
    detect_format_from_extension,
    detect_format_from_content,
    is_format_supported,
    get_format_extension,
    check_dependencies,
    check_pyarrow_available,
    check_openpyxl_available,
    check_matplotlib_available,
    convert_dataframe_to_json,
    get_pandas_dtypes_info,
    get_dataframe_stats,
    is_encrypted_file,
    detect_encoding,
    get_file_metadata,
    validate_file_format
)

# Import crypto router functions
from pamola_core.utils.io_helpers.crypto_router import (
    detect_encryption_mode,
    register_provider,
    get_provider,
    get_all_providers,
    encrypt_file_router,
    decrypt_file_router,
    encrypt_data_router,
    decrypt_data_router
)

# Import Dask utilities
from pamola_core.utils.io_helpers.dask_utils import (
    is_dask_available,
    read_csv_in_chunks,
    read_full_csv,
    write_dataframe_to_csv,
    compute_dask_stats
)

# Import image utilities
from pamola_core.utils.io_helpers.image_utils import (
    save_plot,
    get_figure_format,
    prepare_figure_options,
    get_optimal_figure_size
)

# Import error utilities
from pamola_core.utils.io_helpers.error_utils import (
    create_error_info,
    handle_io_errors,
    extract_error_message,
    is_error_info,
    is_recoverable_error,
    combine_error_infos,
    raise_if_error
)

# Import memory utilities
from pamola_core.utils.io_helpers.memory_utils import (
    get_system_memory,
    get_process_memory_usage,
    estimate_dataframe_size,
    estimate_csv_size as estimate_csv_memory_size,
    get_optimal_chunk_size,
    estimate_file_memory,
    optimize_dataframe_memory,
    check_memory_critical,
    calculate_safe_chunk_count
)

# Import multi-file utilities
from pamola_core.utils.io_helpers.multi_file_utils import (
    detect_file_format,
    get_file_reader,
    validate_files_exist,
    get_common_columns,
    stack_files_vertically,
    process_files_in_batches,
    read_multi_csv,
    read_similar_files,
    memory_efficient_processor
)

# Import provider interface
from pamola_core.utils.io_helpers.provider_interface import CryptoProvider

# Define version
__version__ = '1.1.0'

# Define public API
__all__ = [
    # Directory utilities
    'ensure_directory',
    'get_timestamped_filename',
    'get_file_stats',
    'list_directory_contents',
    'clear_directory',
    'make_unique_path',
    'create_secure_temp_directory',
    'create_secure_temp_file',
    'safe_remove_temp_file',
    'normalize_path',
    'is_path_in_directory',
    'create_secure_path',
    'is_path_writable',
    'protect_path',
    'secure_cleanup',
    'get_temp_file_for_decryption',
    'get_temp_file_for_encryption',
    'ensure_parent_directory',
    'get_unique_filename',
    'create_secure_directory_structure',

    # CSV utilities
    'estimate_csv_size',
    'prepare_csv_reader_options',
    'prepare_csv_writer_options',
    'count_csv_lines',
    'validate_csv_structure',
    'monitor_csv_operation',
    'report_memory_usage',
    'filter_csv_columns',
    'detect_csv_dialect',
    'get_optimal_csv_chunk_size',
    'read_csv_in_efficient_chunks',
    'optimize_csv_datatypes',
    'validate_csv_file',

    # JSON utilities
    'convert_numpy_types',
    'validate_json_structure',
    'merge_json_objects_in_memory',
    'prettify_json',
    'detect_array_or_object',
    'extract_json_subset',
    'prepare_json_writer_options',

    # Format utilities
    'detect_format_from_extension',
    'detect_format_from_content',
    'is_format_supported',
    'get_format_extension',
    'check_dependencies',
    'check_pyarrow_available',
    'check_openpyxl_available',
    'check_matplotlib_available',
    'convert_dataframe_to_json',
    'get_pandas_dtypes_info',
    'get_dataframe_stats',
    'is_encrypted_file',
    'detect_encoding',
    'get_file_metadata',
    'validate_file_format',

    # Crypto router utilities
    'detect_encryption_mode',
    'register_provider',
    'get_provider',
    'get_all_providers',
    'encrypt_file_router',
    'decrypt_file_router',
    'encrypt_data_router',
    'decrypt_data_router',

    # Dask utilities
    'is_dask_available',
    'read_csv_in_chunks',
    'read_full_csv',
    'write_dataframe_to_csv',
    'compute_dask_stats',

    # Image utilities
    'save_plot',
    'get_figure_format',
    'prepare_figure_options',
    'get_optimal_figure_size',

    # Error utilities
    'create_error_info',
    'handle_io_errors',
    'extract_error_message',
    'is_error_info',
    'is_recoverable_error',
    'combine_error_infos',
    'raise_if_error',

    # Memory utilities
    'get_system_memory',
    'get_process_memory_usage',
    'estimate_dataframe_size',
    'estimate_csv_memory_size',
    'get_optimal_chunk_size',
    'estimate_file_memory',
    'optimize_dataframe_memory',
    'check_memory_critical',
    'calculate_safe_chunk_count',

    # Multi-file utilities
    'detect_file_format',
    'get_file_reader',
    'validate_files_exist',
    'get_common_columns',
    'stack_files_vertically',
    'process_files_in_batches',
    'read_multi_csv',
    'read_similar_files',
    'memory_efficient_processor',

    # Provider interface
    'CryptoProvider'
]

# Note: crypto_utils functions (encrypt_file, decrypt_file, etc.) are not directly imported
# to avoid circular dependencies. Users should import them explicitly when needed:
#
# from pamola_core.utils.io_helpers.crypto_utils import encrypt_file, decrypt_file