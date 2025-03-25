"""
    PAMOLA.CORE - Missing Values Handling Utilities
    --------------------------------------
    This file is part of the PAMOLA ecosystem, a comprehensive suite for
    privacy-enhancing technologies. PAMOLA.CORE serves as the open-source
    foundation for secure, structured, and scalable file operations.

    (C) 2024 Realm Inveo Inc. and DGT Network Inc.

    This software is licensed under the BSD 3-Clause License.
    For details, see the LICENSE file or visit:

        https://opensource.org/licenses/BSD-3-Clause
        https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

    Module: Removing
    --------------------------


    NOTE: This module requires `pandas`.

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

def rows(
    df
):
    """
        Removing rows with missing values

        Parameters
        ----------
        df : DataFrame
            Input data.

        Returns
        -------
        DataFrame or None
    """
    df_cleaned = df.dropna()
    return df_cleaned

def columns(
    df,
    columns
):
    """
        Drop columns

        Parameters
        ----------
        df : DataFrame
            Input data.

        columns : single label or list-like
            Input data.

        Returns
        -------
        DataFrame or None
    """
    df_cleaned = df.drop(columns=columns)
    return df_cleaned
