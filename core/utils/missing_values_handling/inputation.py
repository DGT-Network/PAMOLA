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

    Module: Inputation
    --------------------------


    NOTE: This module requires `pandas`.

    Author: Realm Inveo Inc. & DGT Network Inc.
"""

def default_value(
    df,
    column,
    value
):
    """
        Fill missing values with default value

        Parameters
        ----------
        df : DataFrame
            Input data.

        column : single label
            Input data.

        value : object
            Input data.

        Returns
        -------
        DataFrame or None
    """
    values = {f"{column}": value}
    df_inputation = df.fillna(value=values)
    return df_inputation

def mean(
    df,
    column,
    round_to=6
):
    """
        Fill missing values with the mean

        Parameters
        ----------
        df : DataFrame
            Input data.

        column : string
            Input data.

        round_to : int, default 6
            Input data.

        Returns
        -------
        DataFrame or None
    """
    values = {f"{column}": df[column].mean().round(round_to)}
    df_inputation = df.fillna(value=values)
    return df_inputation

def median(
    df,
    column
):
    """
        Fill missing values with the median

        Parameters
        ----------
        df : DataFrame
            Input data.

        column : single label
            Input data.

        Returns
        -------
        DataFrame or None
    """
    values = {f"{column}": df[column].median()}
    df_inputation = df.fillna(value=values)
    return df_inputation

def mode(
    df,
    column
):
    """
        Fill missing values with the mode

        Parameters
        ----------
        df : DataFrame
            Input data.

        column : single label
            Input data.

        Returns
        -------
        DataFrame or None
    """
    values = {f"{column}": df[column].mode().iloc[0]}
    df_inputation = df.fillna(value=values)
    return df_inputation

def forward_fill(
    df,
    column
):
    """
        Fill missing values with the forward fill

        Parameters
        ----------
        df : DataFrame
            Input data.

        column : single label
            Input data.

        Returns
        -------
        DataFrame or None
    """
    df_inputation = df.copy(deep=True)
    df_inputation[column] = df[column].ffill()
    return df_inputation

def backward_fill(
    df,
    column
):
    """
        Fill missing values with the backward fill

        Parameters
        ----------
        df : DataFrame
            Input data.

        column : single label
            Input data.

        Returns
        -------
        DataFrame or None
    """
    df_inputation = df.copy(deep=True)
    df_inputation[column] = df[column].bfill()
    return df_inputation
