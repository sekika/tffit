import numpy as np
import pandas as pd

from src.model_comparison import paired_comparison_table, cluster_loss_table


def make_model_comparison_df():
    """
    Synthetic dataset suitable for LOSO and LOYO model-comparison tests.

    The response is generated exactly from the K model:
        log10(TF) = -1.0 - 0.7 * log10(Ex-K)

    RIP is included so that KR can also be used as a comparison model.
    """
    n = 24
    K = np.geomspace(0.05, 1.5, n)

    y = -1.0 - 0.7 * np.log10(K)
    TF = 10 ** y

    rng = np.random.default_rng(123)

    return pd.DataFrame({
        "TF": TF,
        "Ex-K": K,
        "RIP": rng.uniform(100.0, 2000.0, n),
        "Site": ["A", "B", "C", "D"] * 6,
        "Year": np.repeat([2020, 2021, 2022], 8),
    })


def _find_column(df, candidates):
    normalized = {
        str(c).strip().lower().replace(" ", "_").replace("-", "_"): c
        for c in df.columns
    }

    for name in candidates:
        key = name.strip().lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]

    for c in df.columns:
        c_norm = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if any(name in c_norm for name in candidates):
            return c

    raise AssertionError(
        f"None of the candidate columns {candidates} found in {list(df.columns)}"
    )


def _assert_numeric_output_is_valid(table):
    """
    Check numeric output robustly.

    Some columns may legitimately contain NaN, for example p-values when
    paired_test='none' or statistics that are not defined for a given setting.
    Therefore, this helper allows NaN but rejects +/-inf and requires at least
    one finite numeric value.
    """
    numeric_cols = table.select_dtypes(include=[np.number])

    assert not numeric_cols.empty

    values = numeric_cols.to_numpy(dtype=float)

    # Infinite values should never appear.
    assert not np.isinf(values).any()

    # At least some numeric results should be finite.
    assert np.isfinite(values).any()


def test_paired_comparison_table_loso():
    df = make_model_comparison_df()

    table = paired_comparison_table(
        df=df,
        base_model_name="k",
        compare_model_names=["kr"],
        validation="loso",
        n_bootstrap=30,
        seed=123,
        ci_level=0.95,
        paired_test="none",
        site_col="Site",
        year_col="Year",
        exclude_years=None,
        fix_klim=False,
        klim_fixed=None,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    validation_col = _find_column(table, ["validation"])
    validations = {str(v).lower() for v in table[validation_col]}

    assert validations == {"loso"}

    _assert_numeric_output_is_valid(table)


def test_paired_comparison_table_both_validations():
    df = make_model_comparison_df()

    table = paired_comparison_table(
        df=df,
        base_model_name="k",
        compare_model_names=["kr"],
        validation="both",
        n_bootstrap=30,
        seed=123,
        ci_level=0.90,
        paired_test="none",
        site_col="Site",
        year_col="Year",
        exclude_years=None,
        fix_klim=False,
        klim_fixed=None,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    validation_col = _find_column(table, ["validation"])
    validations = {str(v).lower() for v in table[validation_col]}

    assert validations == {"loso", "loyo"}

    _assert_numeric_output_is_valid(table)


def test_cluster_loss_table_loso():
    df = make_model_comparison_df()

    table = cluster_loss_table(
        df=df,
        base_model_name="k",
        compare_model_names=["kr"],
        validation="loso",
        site_col="Site",
        year_col="Year",
        exclude_years=None,
        fix_klim=False,
        klim_fixed=None,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    validation_col = _find_column(table, ["validation"])
    validations = {str(v).lower() for v in table[validation_col]}

    assert validations == {"loso"}

    # There are four sites, so cluster-level output should contain at least
    # one row per site for the single comparison model.
    assert len(table) >= 4

    _assert_numeric_output_is_valid(table)


def test_cluster_loss_table_both_validations():
    df = make_model_comparison_df()

    table = cluster_loss_table(
        df=df,
        base_model_name="k",
        compare_model_names=["kr"],
        validation="both",
        site_col="Site",
        year_col="Year",
        exclude_years=None,
        fix_klim=False,
        klim_fixed=None,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    validation_col = _find_column(table, ["validation"])
    validations = {str(v).lower() for v in table[validation_col]}

    assert {"loso", "loyo"}.issubset(validations)

    # Expected minimum:
    #   LOSO: 4 sites
    #   LOYO: 3 years
    # for one comparison model.
    assert len(table) >= 7

    _assert_numeric_output_is_valid(table)


def test_paired_comparison_respects_exclude_years_for_loyo():
    df = make_model_comparison_df()

    table = paired_comparison_table(
        df=df,
        base_model_name="k",
        compare_model_names=["kr"],
        validation="loyo",
        n_bootstrap=20,
        seed=123,
        ci_level=0.95,
        paired_test="none",
        site_col="Site",
        year_col="Year",
        exclude_years=[2022],
        fix_klim=False,
        klim_fixed=None,
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    validation_col = _find_column(table, ["validation"])
    validations = {str(v).lower() for v in table[validation_col]}

    assert validations == {"loyo"}

    _assert_numeric_output_is_valid(table)
