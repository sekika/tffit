import io
import numpy as np
import pandas as pd
import pytest

from src.table_output import write_table


def make_table_df():
    return pd.DataFrame({
        "Model": ["k", "kr"],
        "RMSE": [1.23456, 2.34567],
        "Delta": [0.11111, -0.22222],
    })


def test_write_table_text_stdout(capsys):
    df = make_table_df()

    write_table(
        df,
        path=None,
        table_format="text",
        digit=2,
    )

    captured = capsys.readouterr()
    out = captured.out

    assert "Model" in out
    assert "RMSE" in out
    assert "k" in out
    assert "kr" in out

    # Check that rounding is applied.
    assert "1.23" in out
    assert "2.35" in out


def test_write_table_csv_stdout(capsys):
    df = make_table_df()

    write_table(
        df,
        path=None,
        table_format="csv",
        digit=3,
    )

    captured = capsys.readouterr()
    out = captured.out

    assert "Model,RMSE,Delta" in out
    assert "k" in out
    assert "kr" in out

    # CSV output should contain rounded numeric values.
    assert "1.235" in out
    assert "2.346" in out


def test_write_table_markdown_stdout(capsys):
    df = make_table_df()

    write_table(
        df,
        path=None,
        table_format="markdown",
        digit=2,
    )

    captured = capsys.readouterr()
    out = captured.out

    assert "|" in out
    assert "Model" in out
    assert "RMSE" in out
    assert "1.23" in out
    assert "2.35" in out


def test_write_table_csv_file(tmp_path):
    df = make_table_df()
    out_file = tmp_path / "table.csv"

    write_table(
        df,
        path=str(out_file),
        table_format="csv",
        digit=2,
    )

    assert out_file.exists()

    text = out_file.read_text()
    assert "Model,RMSE,Delta" in text
    assert "1.23" in text
    assert "2.35" in text

    loaded = pd.read_csv(out_file)
    assert list(loaded.columns) == ["Model", "RMSE", "Delta"]
    assert loaded.shape == (2, 3)


def test_write_table_markdown_file(tmp_path):
    df = make_table_df()
    out_file = tmp_path / "table.md"

    write_table(
        df,
        path=str(out_file),
        table_format="markdown",
        digit=2,
    )

    assert out_file.exists()

    text = out_file.read_text()
    assert "|" in text
    assert "Model" in text
    assert "RMSE" in text
    assert "1.23" in text


def test_write_table_text_file(tmp_path):
    df = make_table_df()
    out_file = tmp_path / "table.txt"

    write_table(
        df,
        path=str(out_file),
        table_format="text",
        digit=2,
    )

    assert out_file.exists()

    text = out_file.read_text()
    assert "Model" in text
    assert "RMSE" in text
    assert "1.23" in text


def test_write_table_invalid_format_raises():
    df = make_table_df()

    with pytest.raises(ValueError):
        write_table(
            df,
            path=None,
            table_format="json",
            digit=2,
        )
