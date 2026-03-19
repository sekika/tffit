"""
Data loading and preprocessing module for soil-to-wheat radiocesium transfer data.

This module provides functionalities to precisely extract and format subsets of data
from the Supplementary Excel file. It ensures that the spatiotemporal identifiers 
(Year, Site) and the measured physicochemical properties are correctly typed and 
standardized for subsequent mathematical modeling and cross-validation.
"""
import pandas as pd
import numpy as np

REQUIRED_COLS = ["Year", "Site"]

NUMERIC_COLS = [
    "TF", "Ex-K", "137Cs", "CEC", "Ex-Ca", "Ex-Mg", "BS", "Av-P",
    "Humus", "RIP", "Sand", "Silt", "Clay", "pH",
]


def load_data(file_path, sheet_name, header_row, data_start, data_end):
    """
    Load, slice, and standardize data from a specified Excel worksheet.

    This function reads a specific block of data from an Excel file using 1-based
    row indices (matching the Excel UI) to ensure reproducibility when extracting
    data from the published Supplementary dataset. It validates the presence of
    essential identifiers and forces strict numeric typing for physicochemical
    variables, coercing unparseable values to NaN.

    Parameters
    ----------
    file_path : str
        The file path to the target Excel workbook (e.g., 'supplementary.xlsx').
    sheet_name : str
        The exact name of the worksheet containing the data (e.g., 'Table S1').
    header_row : int
        The 1-based row number in Excel that contains the column headers.
    data_start : int
        The 1-based row number in Excel where the actual data records begin.
    data_end : int
        The 1-based row number in Excel where the actual data records end (inclusive).

    Returns
    -------
    df : pandas.DataFrame
        The cleaned, sliced, and correctly typed dataset ready for modeling.

    Raises
    ------
    KeyError
        If mandatory spatiotemporal identifier columns ('Year', 'Site') are missing
        from the extracted headers.
    """
    header_idx = header_row - 1

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header_idx
    )
    df.columns = df.columns.astype(str).str.strip()

    # Slice rows: interpret data_start/data_end as 1-based Excel row numbers
    start_idx = data_start - header_row - 1
    end_idx = data_end - header_row
    df = df.iloc[start_idx:end_idx].copy()

    # Basic presence check for id columns
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise KeyError(
                f"Required column '{c}' not found. Available: {list(df.columns)}")

    # Clean types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Site"] = df["Site"].astype(str).str.strip()

    # Convert numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
