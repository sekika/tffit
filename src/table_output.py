import sys
import pandas as pd
import numpy as np


def _format_value(value, digit=3):
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):.{digit}f}"
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    return str(value)


def format_table(df, table_format="text", digit=3):
    """
    Format a DataFrame as text, csv, or markdown.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df_disp = df.copy()

    for col in df_disp.columns:
        df_disp[col] = df_disp[col].map(lambda x: _format_value(x, digit=digit))

    if table_format == "csv":
        return df_disp.to_csv(index=False)

    if table_format == "markdown":
        headers = list(df_disp.columns)
        rows = df_disp.astype(str).values.tolist()

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    if table_format == "text":
        return df_disp.to_string(index=False)

    raise ValueError(f"Unknown table format: {table_format}")


def write_table(df, path=None, table_format="text", digit=3):
    """
    Write table to stdout or file.
    """
    s = format_table(df, table_format=table_format, digit=digit)

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
            if not s.endswith("\n"):
                f.write("\n")
    else:
        sys.stdout.write(s)
        if not s.endswith("\n"):
            sys.stdout.write("\n")
