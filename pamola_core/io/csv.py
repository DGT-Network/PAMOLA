"""
    PAMOLA Core - IO Package Base Module
    -----------------------------------
    This module provides base interfaces and protocols for all IO operations in the PAMOLA Core
    library. It defines the standard contracts that all format-specific handlers must implement.

    Key features:
    - Implementation of DataReader protocol for standardized CSV reading
    - Implementation of DataWriter protocol for standardized CSV writing

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

class DataCSV:
    def __init__(self): ...

    def read(self, path: str, **kwargs) -> Any:
        return pd.read_csv(filepath_or_buffer = path
                           , sep = kwargs.get('sep') if 'sep' in kwargs else ','
                           , delimiter = kwargs.get('delimiter')
                           , header = kwargs.get('header') if 'header' in kwargs else 'infer'
                           , names = kwargs.get('names')
                           , index_col = kwargs.get('index_col')
                           , usecols = kwargs.get('usecols')
                           , dtype = kwargs.get('dtype')
                           , engine = kwargs.get('engine')
                           , converters = kwargs.get('converters')
                           , true_values = kwargs.get('true_values')
                           , false_values = kwargs.get('false_values')
                           , skipinitialspace = kwargs.get('skipinitialspace') if 'skipinitialspace' in kwargs else False
                           , skiprows = kwargs.get('skiprows')
                           , skipfooter = kwargs.get('skipfooter') if 'skipfooter' in kwargs else 0
                           , nrows = kwargs.get('nrows')
                           , na_values = kwargs.get('na_values')
                           , keep_default_na = kwargs.get('keep_default_na') if 'keep_default_na' in kwargs else True
                           , na_filter = kwargs.get('na_filter') if 'na_filter' in kwargs else True
                           , skip_blank_lines = kwargs.get('skip_blank_lines') if 'skip_blank_lines' in kwargs else True
                           , parse_dates = kwargs.get('parse_dates') if 'parse_dates' in kwargs else False
                           , date_format = kwargs.get('date_format')
                           , dayfirst = kwargs.get('dayfirst') if 'dayfirst' in kwargs else False
                           , cache_dates = kwargs.get('cache_dates') if 'cache_dates' in kwargs else True
                           , iterator = kwargs.get('iterator') if 'iterator' in kwargs else False
                           , chunksize = kwargs.get('chunksize')
                           , compression = kwargs.get('compression') if 'compression' in kwargs else 'infer'
                           , thousands = kwargs.get('thousands')
                           , decimal = kwargs.get('decimal') if 'decimal' in kwargs else '.'
                           , lineterminator = kwargs.get('lineterminator')
                           , quotechar = kwargs.get('quotechar') if 'quotechar' in kwargs else '"'
                           , quoting = kwargs.get('quoting') if 'quoting' in kwargs else 0
                           , doublequote = kwargs.get('doublequote') if 'doublequote' in kwargs else True
                           , escapechar = kwargs.get('escapechar')
                           , comment = kwargs.get('comment')
                           , encoding = kwargs.get('encoding') if 'encoding' in kwargs else 'utf-8'
                           , encoding_errors = kwargs.get('encoding_errors') if 'encoding_errors' in kwargs else 'strict'
                           , dialect = kwargs.get('dialect')
                           , on_bad_lines = kwargs.get('on_bad_lines') if 'on_bad_lines' in kwargs else 'error'
                           , low_memory = kwargs.get('low_memory') if 'low_memory' in kwargs else True
                           , memory_map = kwargs.get('memory_map') if 'memory_map' in kwargs else False
                           , float_precision = kwargs.get('float_precision')
                           , storage_options = kwargs.get('storage_options')
                           , dtype_backend = kwargs.get('dtype_backend') if 'dtype_backend' in kwargs else 'numpy_nullable'
                           )

    def write(self, data: Any, path: str, **kwargs) -> Any:
        return data.to_csv(path_or_buf = path
                           , sep = kwargs.get('sep') if 'sep' in kwargs else ','
                           , na_rep = kwargs.get('na_rep') if 'na_rep' in kwargs else ''
                           , float_format = kwargs.get('float_format')
                           , columns = kwargs.get('columns')
                           , header = kwargs.get('header') if 'header' in kwargs else True
                           , index = kwargs.get('index') if 'index' in kwargs else True
                           , index_label = kwargs.get('index_label')
                           , mode = kwargs.get('mode') if 'mode' in kwargs else 'w'
                           , encoding = kwargs.get('encoding')
                           , compression = kwargs.get('compression') if 'compression' in kwargs else 'infer'
                           , quoting = kwargs.get('quoting')
                           , quotechar = kwargs.get('quotechar') if 'quotechar' in kwargs else '"'
                           , lineterminator = kwargs.get('lineterminator')
                           , chunksize = kwargs.get('chunksize')
                           , date_format = kwargs.get('date_format')
                           , doublequote = kwargs.get('doublequote') if 'doublequote' in kwargs else True
                           , escapechar = kwargs.get('escapechar')
                           , decimal = kwargs.get('decimal') if 'decimal' in kwargs else '.'
                           , errors = kwargs.get('errors') if 'errors' in kwargs else 'strict'
                           , storage_options = kwargs.get('storage_options')
                           )

def read_csv(path: str, **kwargs) -> Any:
    """
    Read a comma-separated values (csv) file into DataFrame.

    Also supports optionally iterating or breaking of the file into chunks.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. For file URLs, a host is  expected. A local file could be:
        file://localhost/path/to/table.csv.
        If you want to pass in a path object, pandas accepts any ``os.PathLike``.
        By file-like object, we refer to objects with a ``read()`` method, such as a file handle
        (e.g. via builtin ``open`` function) or ``StringIO``.

    sep : str, default ','
        Character or regex pattern to treat as the delimiter. If ``sep=None``, the C engine cannot
        automatically detect the separator, but the Python parsing engine can, meaning the latter
        will be used and automatically detect the separator from only the first valid row of the
        file by Python's builtin sniffer tool, ``csv.Sniffer``. In addition, separators longer
        than 1 character and different from ``'\s+'`` will be interpreted as regular expressions
        and will also force the use of the Python parsing engine. Note that regex delimiters are
        prone to ignoring quoted data. Regex example: ``'\r\t'``.

    delimiter : str, optional
        Alias for ``sep``.

    header : int, Sequence of int, 'infer' or None, default 'infer'
        Row number(s) containing column labels and marking the start of the data (zero-indexed).
        Default behavior is to infer the column names: if no ``names`` are passed the behavior
        is identical to ``header=0`` and column names are inferred from the first line of the
        file, if column names are passed explicitly to ``names`` then the behavior is identical
        to ``header=None``. Explicitly pass ``header=0`` to be able to replace existing names.
        The header can be a list of integers that specify row locations for a
        :class:`~pandas.MultiIndex` on the columns e.g. ``[0, 1, 3]``. Intervening rows that
        are not specified will be skipped (e.g. 2 in this example is skipped). Note that this
        parameter ignores commented lines and empty lines if ``skip_blank_lines=True``, so
        ``header=0`` denotes the first line of data rather than the first line of the file.

    names : Sequence of Hashable, optional
        Sequence of column labels to apply. If the file contains a header row, then you should
        explicitly pass ``header=0`` to override the column names. Duplicates in this list are
        not allowed.

    index_col : Hashable, Sequence of Hashable or False, optional
          Column(s) to use as row label(s), denoted either by column labels or column indices.
          If a sequence of labels or indices is given, :class:`~pandas.MultiIndex` will be
          formed for the row labels.
          Note: ``index_col=False`` can be used to force pandas to *not* use the first column
          as the index, e.g., when you have a malformed file with delimiters at the end of
          each line.

    usecols : Sequence of Hashable or Callable, optional
        Subset of columns to select, denoted either by column labels or column indices. If
        list-like, all elements must either be positional (i.e. integer indices into the
        document columns) or strings that correspond to column names provided either by the
        user in ``names`` or inferred from the document header row(s). If ``names`` are
        given, the document header row(s) are not taken into account. For example, a valid
        list-like ``usecols`` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
        Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``. To
        instantiate a :class:`~pandas.DataFrame` from ``data`` with element order preserved
        use ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]`` for columns in
        ``['foo', 'bar']`` order or ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
        for ``['bar', 'foo']`` order.
        If callable, the callable function will be evaluated against the column names, returning
        names where the callable function evaluates to ``True``. An example of a valid callable
        argument would be ``lambda x: x.upper() in ['AAA', 'BBB', 'DDD']``. Using this parameter
        results in much faster parsing time and lower memory usage.

    dtype : dtype or dict of {{Hashable : dtype}}, optional
        Data type(s) to apply to either the whole dataset or individual columns. E.g.,
        ``{{'a': np.float64, 'b': np.int32, 'c': 'Int64'}}`` Use ``str`` or ``object`` together
        with suitable ``na_values`` settings to preserve and not interpret ``dtype``. If
        ``converters`` are specified, they will be applied INSTEAD of ``dtype`` conversion.

    engine : {{'c', 'python', 'pyarrow'}}, optional
        Parser engine to use. The C and pyarrow engines are faster, while the python engine is
        currently more feature-complete. Multithreading is currently only supported by the pyarrow
        engine.

    converters : dict of {{Hashable : Callable}}, optional
        Functions for converting values in specified columns. Keys can either be column labels
        or column indices.

    true_values : list, optional
        Values to consider as ``True`` in addition to case-insensitive variants of 'True'.

    false_values : list, optional
        Values to consider as ``False`` in addition to case-insensitive variants of 'False'.

    skipinitialspace : bool, default False
        Skip spaces after delimiter.

    skiprows : int, list of int or Callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (``int``) at the start
        of the file.
        If callable, the callable function will be evaluated against the row indices, returning
        ``True`` if the row should be skipped and ``False`` otherwise. An example of a valid
        callable argument would be ``lambda x: x in [0, 2]``.

    skipfooter : int, default 0
        Number of lines at bottom of file to skip (Unsupported with ``engine='c'``).

    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large files.

    na_values : Hashable, Iterable of Hashable or dict of {{Hashable : Iterable}}, optional
        Additional strings to recognize as ``NA``/``NaN``. If ``dict`` passed, specific
        per-column ``NA`` values.

    keep_default_na : bool, default True
        Whether or not to include the default ``NaN`` values when parsing the data. Depending on
        whether ``na_values`` is passed in, the behavior is as follows:
        * If ``keep_default_na`` is ``True``, and ``na_values`` are specified, ``na_values`` is
          appended to the default ``NaN`` values used for parsing.
        * If ``keep_default_na`` is ``True``, and ``na_values`` are not specified, only the default
          ``NaN`` values are used for parsing.
        * If ``keep_default_na`` is ``False``, and ``na_values`` are specified, only the ``NaN``
          values specified ``na_values`` are used for parsing.
        * If ``keep_default_na`` is ``False``, and ``na_values`` are not specified, no strings
          will be parsed as ``NaN``.
        Note that if ``na_filter`` is passed in as ``False``, the ``keep_default_na`` and
        ``na_values`` parameters will be ignored.

    na_filter : bool, default True
        Detect missing value markers (empty strings and the value of ``na_values``). In data without
        any ``NA`` values, passing ``na_filter=False`` can improve the performance of reading a large
        file.

    skip_blank_lines : bool, default True
        If ``True``, skip over blank lines rather than interpreting as ``NaN`` values.

    parse_dates : bool, list of Hashable, list of lists or dict of {{Hashable : list}}, default False
        The behavior is as follows:
        * ``bool``. If ``True`` -> try parsing the index. Note: Automatically set to ``True`` if
          ``date_format`` or ``date_parser`` arguments have been passed.
        * ``list`` of ``int`` or names. e.g. If ``[1, 2, 3]`` -> try parsing columns 1, 2, 3 each as
          a separate date column.
        * ``list`` of ``list``. e.g.  If ``[[1, 3]]`` -> combine columns 1 and 3 and parse as a single
          date column. Values are joined with a space before parsing.
        * ``dict``, e.g. ``{{'foo' : [1, 3]}}`` -> parse columns 1, 3 as date and call result 'foo'.
          Values are joined with a space before parsing.
        If a column or index cannot be represented as an array of ``datetime``, say because of an
        unparsable value or a mixture of timezones, the column or index will be returned unaltered as
        an ``object`` data type. For non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime`
        after :func:`~pandas.read_csv`.
        Note: A fast-path exists for iso8601-formatted dates.

    date_format : str or dict of column -> format, optional
        Format to use for parsing dates when used in conjunction with ``parse_dates``. The strftime to
        parse time, e.g. :const:`"%d/%m/%Y"`. You can also pass:
        - "ISO8601", to parse any `ISO8601` time string (not necessarily in exactly the same format);
        - "mixed", to infer the format for each element individually. This is risky, and you should
          probably use it along with `dayfirst`.

    dayfirst : bool, default False
        DD/MM format dates, international and European format.

    cache_dates : bool, default True
        If ``True``, use a cache of unique, converted dates to apply the ``datetime`` conversion. May
        produce significant speed-up when parsing duplicate date strings, especially ones with timezone
        offsets.

    iterator : bool, default False
        Return ``TextFileReader`` object for iteration or getting chunks with ``get_chunk()``.

    chunksize : int, optional
        Number of lines to read from the file per chunk. Passing a value will cause the function to
        return a ``TextFileReader`` object for iteration.

    compression : str or dict, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and '%s' is path-like,  then detect
        compression from the following extensions: '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar',
        '.tar.gz', '.tar.xz' or '.tar.bz2' (otherwise no compression). If using 'zip' or 'tar', the
        ZIP file must contain only one data file to be read in. Set to ``None`` for no decompression.
        Can also be a dict with key ``'method'`` set to one of {``'zip'``, ``'gzip'``, ``'bz2'``,
        ``'zstd'``, ``'xz'``, ``'tar'``} and other key-value pairs are forwarded to ``zipfile.ZipFile``,
        ``gzip.GzipFile``, ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.

    thousands : str (length 1), optional
        Character acting as the thousands separator in numerical values.

    decimal : str (length 1), default '.'
        Character to recognize as decimal point (e.g., use ',' for European data).

    lineterminator : str (length 1), optional
        Character used to denote a line break. Only valid with C parser.

    quotechar : str (length 1), optional
        Character used to denote the start and end of a quoted item. Quoted items can include the
        ``delimiter`` and it will be ignored.

    quoting : {{0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC,
    3 or csv.QUOTE_NONE}}, default csv.QUOTE_MINIMAL
        Control field quoting behavior per ``csv.QUOTE_*`` constants. Default is ``csv.QUOTE_MINIMAL``
        (i.e., 0) which implies that only fields containing special characters are quoted (e.g.,
        characters defined in ``quotechar``, ``delimiter``, or ``lineterminator``.

    doublequote : bool, default True
       When ``quotechar`` is specified and ``quoting`` is not ``QUOTE_NONE``, indicate whether or not to
       interpret two consecutive ``quotechar`` elements INSIDE a field as a single ``quotechar`` element.

    escapechar : str (length 1), optional
        Character used to escape other characters.

    comment : str (length 1), optional
        Character indicating that the remainder of line should not be parsed. If found at the beginning of
        a line, the line will be ignored altogether. This parameter must be a single character. Like empty
        lines (as long as ``skip_blank_lines=True``), fully commented lines are ignored by the parameter
        ``header`` but not by ``skiprows``. For example, if ``comment='#'``, parsing
        ``#empty\\na,b,c\\n1,2,3`` with ``header=0`` will result in ``'a,b,c'`` being treated as the header.

    encoding : str, optional, default 'utf-8'
        Encoding to use for UTF when reading/writing (ex. ``'utf-8'``).

    encoding_errors : str, optional, default 'strict'
        How encoding errors are treated.

    dialect : str or csv.Dialect, optional
        If provided, this parameter will override values (default or not) for the following parameters:
        ``delimiter``, ``doublequote``, ``escapechar``, ``skipinitialspace``, ``quotechar``, and ``quoting``.
        If it is necessary to override values, a ``ParserWarning`` will be issued.

    on_bad_lines : {{'error', 'warn', 'skip'}} or Callable, default 'error'
        Specifies what to do upon encountering a bad line (a line with too many fields).
        Allowed values are :
        - ``'error'``, raise an Exception when a bad line is encountered.
        - ``'warn'``, raise a warning when a bad line is encountered and skip that line.
        - ``'skip'``, skip bad lines without raising or warning when they are encountered.

    low_memory : bool, default True
        Internally process the file in chunks, resulting in lower memory use while parsing, but possibly
        mixed type inference.  To ensure no mixed types either set ``False``, or specify the type with the
        ``dtype`` parameter. Note that the entire file is read into a single :class:`~pandas.DataFrame`
        regardless, use the ``chunksize`` or ``iterator`` parameter to return the data in chunks. (Only
        valid with C parser).

    memory_map : bool, default False
        If a filepath is provided for ``filepath_or_buffer``, map the file object directly onto memory
        and access the data directly from there. Using this option can improve performance because there
        is no longer any I/O overhead.

    float_precision : {{'high', 'legacy', 'round_trip'}}, optional
        Specifies which converter the C engine should use for floating-point values. The options are
        ``None`` or ``'high'`` for the ordinary converter, ``'legacy'`` for the original lower precision
        pandas converter, and ``'round_trip'`` for the round-trip converter.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`. Behaviour is as follows:
        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame` (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype` DataFrame.

    Returns
    -------
    DataFrame or TextFileReader
        A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
    """
    return data_read(DataCSV(), path, **kwargs)

def write_csv(df: pd.DataFrame, path: str, **kwargs) -> Any:
    """
    Write object to a comma-separated values (csv) file.

    Parameters
    ----------
    df : DataFrame
        The input data to be processed.

    path : str, path object, file-like object, or None, default None
        String, path object (implementing os.PathLike[str]), or file-like object implementing
        a write() function. If None, the result is returned as a string. If a non-binary file
        object is passed, it should be opened with `newline=''`, disabling universal newlines.
        If a binary file object is passed, `mode` might need to contain a `'b'`.

    sep : str, default ','
        String of length 1. Field delimiter for the output file.

    na_rep : str, default ''
        Missing data representation.

    float_format : str, Callable, default None
        Format string for floating point numbers. If a Callable is given, it takes precedence
        over other numeric formatting parameters, like decimal.

    columns : sequence, optional
        Columns to write.

    header : bool or list of str, default True
        Write out the column names. If a list of strings is given it is assumed to be aliases
        for the column names.

    index : bool, default True
        Write row names (index).

    index_label : str or sequence, or False, default None
        Column label for index column(s) if desired. If None is given, and `header` and `index`
        are True, then the index names are used. A sequence should be given if the object uses
        MultiIndex. If False do not print fields for index names. Use index_label=False for
        easier importing in R.

    mode : {{'w', 'x', 'a'}}, default 'w'
        Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control the file opening.
        Typical values include:
        - 'w', truncate the file first.
        - 'x', exclusive creation, failing if the file already exists.
        - 'a', append to the end of file if it exists.

    encoding : str, optional
        A string representing the encoding to use in the output file, defaults to 'utf-8'.
        `encoding` is not supported if `path_or_buf` is a non-binary file object.

    compressions : str or dict, default ‘infer’
        For on-the-fly compression of the output data. If ‘infer’ and ‘path_or_buf’ is path-like,
        then detect compression from the following extensions: ‘.gz’, ‘.bz2’, ‘.zip’, ‘.xz’, ‘.zst’,
        ‘.tar’, ‘.tar.gz’, ‘.tar.xz’ or ‘.tar.bz2’ (otherwise no compression). Set to None for no
        compression. Can also be a dict with key 'method' set to one of {'zip', 'gzip', 'bz2', 'zstd',
         'xz', 'tar'} and other key-value pairs are forwarded to zipfile.ZipFile, gzip.GzipFile,
         bz2.BZ2File, zstandard.ZstdCompressor, lzma.LZMAFile or tarfile.TarFile, respectively.

    quoting : optional constant from csv module
        Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format` then floats are converted to
        strings and thus csv.QUOTE_NONNUMERIC will treat them as non-numeric.

    quotechar : str, default '"'
        String of length 1. Character used to quote fields.

    lineterminator : str, optional
        The newline character or character sequence to use in the output file. Defaults to `os.linesep`,
        which depends on the OS in which this method is called ('\\n' for linux, '\\r\\n' for Windows, i.e.).

    chunksize : int or None
        Rows to write at a time.

    date_format : str, default None
        Format string for datetime objects.

    doublequote : bool, default True
        Control quoting of `quotechar` inside a field.

    escapechar : str, default None
        String of length 1. Character used to escape `sep` and `quotechar` when appropriate.

    decimal : str, default '.'
        Character recognized as decimal separator. E.g. use ',' for European data.

    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g. host, port, username,
        password, etc. For HTTP(S) URLs the key-value pairs are forwarded to urllib.request.Request
        as header options. For other URLs (e.g. starting with “s3://”, and “gcs://”) the key-value
        pairs are forwarded to fsspec.open.

    Returns
    -------
    None or str
        If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
    """
    return data_write(DataCSV(), df, path, **kwargs)
