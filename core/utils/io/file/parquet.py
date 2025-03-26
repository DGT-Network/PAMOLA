"""
    PAMOLA.CORE - I/O File Utilities
    --------------------------------------
    This file is part of the PAMOLA ecosystem, a comprehensive suite for
    privacy-enhancing technologies. PAMOLA.CORE serves as the open-source
    foundation for secure, structured, and scalable file operations.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:

        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Module: Apache Parquet (parquet)
    --------------------------


    NOTE: This module requires `pyarrow`.

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pyarrow as pa
import pyarrow.parquet as pq

def read(
    path_or_buf
):
    """
        Convert parquet format to pandas object

        Parameters
        ----------
        path_or_buf : a valid JSON str, path object or file-like object
            Input data.

        Returns
        -------
        DataFrame
    """
    table = pq.read_table(path_or_buf)
    df = table.to_pandas()
    return df

def write(
    df,
    path_or_buf
):
    """
        Convert the object to parquet format

        Parameters
        ----------
        df : DataFrame
            Input data.

        path_or_buf : str, path object, file-like object
            Input data.

        Returns
        -------
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path_or_buf)
