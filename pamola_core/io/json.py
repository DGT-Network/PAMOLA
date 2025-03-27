"""
    PAMOLA Core - IO Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all IO operations in the PAMOLA Core
    library. It defines the standard contracts that all format-specific handlers must implement.

    Key features:
    - Implementation of DataReader protocol for standardized JSON reading
    - Implementation of DataWriter protocol for standardized JSON writing

    This module serves as the foundation for all IO operations in PAMOLA Core, ensuring
    consistency across different data formats and storage methods.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:
        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from typing import Any
from .base import data_read, data_write

class DataJSON:
    def __init__(self): ...

    def read(self, path: str, **kwargs) -> Any:
        return pd.read_json(path_or_buf = path
                           , orient = kwargs.get('orient')
                           , typ = kwargs.get('typ') if 'typ' in kwargs else 'frame'
                           , dtype = kwargs.get('dtype')
                           , convert_axes = kwargs.get('convert_axes')
                           , convert_dates = kwargs.get('convert_dates') if 'convert_dates' in kwargs else True
                           , keep_default_dates = kwargs.get('keep_default_dates') if 'keep_default_dates' in kwargs else True
                           , precise_float = kwargs.get('precise_float') if 'precise_float' in kwargs else False
                           , date_unit = kwargs.get('date_unit')
                           , encoding = kwargs.get('encoding') if 'encoding' in kwargs else 'utf-8'
                           , encoding_errors = kwargs.get('encoding_errors') if 'encoding_errors' in kwargs else 'strict'
                           , lines = kwargs.get('lines') if 'lines' in kwargs else False
                           , chunksize = kwargs.get('chunksize')
                           , compression = kwargs.get('compression') if 'compression' in kwargs else 'infer'
                           , nrows = kwargs.get('nrows')
                           , storage_options = kwargs.get('storage_options')
                           , dtype_backend = kwargs.get('dtype_backend') if 'dtype_backend' in kwargs else 'numpy_nullable'
                           , engine = kwargs.get('engine') if 'engine' in kwargs else 'ujson'
                           )

    def write(self, data: Any, path: str, **kwargs) -> Any:
        return data.to_json(path_or_buf = path
                           , orient = kwargs.get('orient')
                           , date_format = kwargs.get('date_format')
                           , double_precision = kwargs.get('double_precision') if 'double_precision' in kwargs else 10
                           , force_ascii = kwargs.get('force_ascii') if 'force_ascii' in kwargs else True
                           , date_unit = kwargs.get('date_unit') if 'date_unit' in kwargs else 'ms'
                           , default_handler = kwargs.get('default_handler')
                           , lines = kwargs.get('lines') if 'lines' in kwargs else False
                           , compression = kwargs.get('compression') if 'compression' in kwargs else 'infer'
                           , index = kwargs.get('index')
                           , indent = kwargs.get('indent')
                           , storage_options = kwargs.get('storage_options')
                           , mode = kwargs.get('mode') if 'mode' in kwargs else 'w'
                           )

def read_json(path: str, **kwargs) -> Any:
    """
    Convert a JSON string to pandas object.

    Parameters
    ----------
    path : a valid JSON str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, and file.

    orient : str, optional
        Indication of expected JSON string format. Compatible JSON strings can be produced by
        to_json() with a corresponding orient value. The set of possible orients is:
        - ``'split'`` : dict like ``{{index -> [index], columns -> [columns], data -> [values]}}``
        - ``'records'`` : list like ``[{{column -> value}}, ... , {{column -> value}}]``
        - ``'index'`` : dict like ``{{index -> {{column -> value}}}}``
        - ``'columns'`` : dict like ``{{column -> {{index -> value}}}}``
        - ``'values'`` : just the values array
        - ``'table'`` : dict like ``{{'schema': {{schema}}, 'data': {{data}}}}``
        The allowed and default values depend on the value of the `typ` parameter.
        * when ``typ == 'series'``,
          - allowed orients are ``{{'split','records','index'}}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.
        * when ``typ == 'frame'``,
          - allowed orients are ``{{'split','records','index', 'columns','values', 'table'}}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``, ``'columns'``, and ``'records'``

    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.

    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those; if False, then don't
        infer dtypes at all, applies only to the data.
        For all ``orient`` values except ``'table'``, default is True.

    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.
        For all ``orient`` values except ``'table'``, default is True.

    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on keep_default_dates).
        If False, no dates will be converted. If a list of column names, then those columns will
        be converted and default datelike columns may also be converted (depending on
        keep_default_dates).

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the default datelike
        columns.

    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when decoding string to double
        values. Default (False) is to use fast but less precise builtin functionality.

    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour is to try and
        detect the correct precision, but if this is not desired then pass one of 's', 'ms',
        'us' or 'ns' to force parsing only seconds, milliseconds, microseconds or nanoseconds
        respectively.

    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    encoding_errors : str, optional, default "strict"
        How encoding errors are treated.

    lines : bool, default False
        Read the file as a json object per line.

    chunksize : int, optional
        Return JsonReader object for iteration.

    compressions : str or dict, default ‘infer’
        For on-the-fly compression of the output data. If ‘infer’ and ‘path_or_buf’ is path-like,
        then detect compression from the following extensions: ‘.gz’, ‘.bz2’, ‘.zip’, ‘.xz’, ‘.zst’,
        ‘.tar’, ‘.tar.gz’, ‘.tar.xz’ or ‘.tar.bz2’ (otherwise no compression). Set to None for no
        compression. Can also be a dict with key 'method' set to one of {'zip', 'gzip', 'bz2', 'zstd',
        'xz', 'tar'} and other key-value pairs are forwarded to zipfile.ZipFile, gzip.GzipFile,
        bz2.BZ2File, zstandard.ZstdCompressor, lzma.LZMAFile or tarfile.TarFile, respectively.

    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read. This can only be
        passed if `lines=True`. If this is None, all the rows will be returned.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`. Behaviour is as follows:
        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame` (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype` DataFrame.

    engine : {{"ujson", "pyarrow"}}, default "ujson"
        Parser engine to use. The ``"pyarrow"`` engine is only available when ``lines=True``.

    Returns
    -------
    Series, DataFrame, or pandas.api.typing.JsonReader
        A JsonReader is returned when ``chunksize`` is not ``0`` or ``None``. Otherwise, the type
        returned depends on the value of ``typ``.
    """
    return data_read(DataJSON(), path, **kwargs)

def write_json(df: pd.DataFrame, path: str, **kwargs) -> Any:
    """
    Convert the object to a JSON string.

    Note NaN's and None will be converted to null and datetime objects will be converted to
    UNIX timestamps.

    Parameters
    ----------
    df : DataFrame
        The input data to be processed.

    path : str, path object, file-like object, or None, default None
        String, path object (implementing os.PathLike[str]), or file-like object implementing
        a write() function. If None, the result is returned as a string.

    orient : str
        Indication of expected JSON string format.
        * Series:
            - default is 'index'
            - allowed values are: {{'split', 'records', 'index', 'table'}}.
        * DataFrame:
            - default is 'columns'
            - allowed values are: {{'split', 'records', 'index', 'columns', 'values', 'table'}}.
        * The format of the JSON string:
            - 'split' : dict like {{'index' -> [index], 'columns' -> [columns], 'data' -> [values]}}
            - 'records' : list like [{{column -> value}}, ... , {{column -> value}}]
            - 'index' : dict like {{index -> {{column -> value}}}}
            - 'columns' : dict like {{column -> {{index -> value}}}}
            - 'values' : just the values array
            - 'table' : dict like {{'schema': {{schema}}, 'data': {{data}}}}
            Describing the data, where data component is like ``orient='records'``.

    date_format : {{None, 'epoch', 'iso'}}
        Type of date conversion. 'epoch' = epoch milliseconds, 'iso' = ISO8601. The default
        depends on the `orient`. For ``orient='table'``, the default is 'iso'. For all other
        orients, the default is 'epoch'.

    double_precision : int, default 10
        The number of decimal places to use when encoding floating point values. The possible
        maximal value is 15. Passing double_precision greater than 15 will raise a ValueError.

    force_ascii : bool, default True
        Force encoded string to be ASCII.

    date_unit : str, default 'ms' (milliseconds)
        The time unit to encode to, governs timestamp and ISO8601 precision.  One of 's',
        'ms', 'us', 'ns' for second, millisecond, microsecond, and nanosecond respectively.

    default_handler : callable, default None
        Handler to call if object cannot otherwise be converted to a suitable format for JSON.
        Should receive a single argument which is the object to convert and return a
        serialisable object.

    lines : bool, default False
        If 'orient' is 'records' write out line-delimited json format. Will throw ValueError if
        incorrect 'orient' since others are not list-like.

    compressions : str or dict, default ‘infer’
        For on-the-fly compression of the output data. If ‘infer’ and ‘path_or_buf’ is path-like,
        then detect compression from the following extensions: ‘.gz’, ‘.bz2’, ‘.zip’, ‘.xz’, ‘.zst’,
        ‘.tar’, ‘.tar.gz’, ‘.tar.xz’ or ‘.tar.bz2’ (otherwise no compression). Set to None for no
        compression. Can also be a dict with key 'method' set to one of {'zip', 'gzip', 'bz2', 'zstd',
        'xz', 'tar'} and other key-value pairs are forwarded to zipfile.ZipFile, gzip.GzipFile,
        bz2.BZ2File, zstandard.ZstdCompressor, lzma.LZMAFile or tarfile.TarFile, respectively.

    index : bool or None, default None
        The index is only used when 'orient' is 'split', 'index', 'column', or 'table'. Of these,
        'index' and 'column' do not support `index=False`.

    indent : int, optional
       Length of whitespace used to indent each record.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    mode : str, default 'w' (writing)
        Specify the IO mode for output when supplying a path_or_buf. Accepted args are 'w' (writing)
        and 'a' (append) only. mode='a' is only supported when lines is True and orient is 'records'.

    Returns
    -------
    None or str
        If path_or_buf is None, returns the resulting json format as a string. Otherwise returns None.
    """
    return data_write(DataJSON(), df, path, **kwargs)
