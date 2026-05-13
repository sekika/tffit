import numpy as np
import pandas as pd

from src.bootstrap import coefficient_bootstrap


class SimpleBootstrapModel:
    """
    Minimal model for testing coefficient_bootstrap().

    It fits:
        log10(TF) = k1 + k2 * Ex-K

    This keeps the test independent of the scientific model implementations
    while still exercising the bootstrap code path.
    """

    def __init__(self):
        self.features = []
        self.formula_str = "log10(TF) = k1 + k2 * Ex-K"
        self.target_col = "log10_TF"
        self.params_meta = [
            {"key": "k1", "label": "k1", "description": "intercept"},
            {"key": "k2", "label": "k2", "description": "slope"},
        ]

    def fit(self, y, K, X=None, train_df=None):
        K = np.asarray(K, dtype=float)
        y = np.asarray(y, dtype=float)

        A = np.column_stack([np.ones_like(K), K])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)

        y_pred = A @ beta
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        return {
            "k1": float(beta[0]),
            "k2": float(beta[1]),
            "rmse_log10": rmse,
        }

    def predict(self, K, X, fit):
        K = np.asarray(K, dtype=float)
        return fit["k1"] + fit["k2"] * K


def make_bootstrap_df(n=20):
    K = np.linspace(0.1, 2.0, n)
    y = 0.5 + 0.8 * K
    TF = 10 ** y

    return pd.DataFrame({
        "TF": TF,
        "Ex-K": K,
        "Site": ["A", "B"] * (n // 2),
        "Year": [2020, 2021] * (n // 2),
    })


def _find_column(df, candidates):
    """
    Find a column using tolerant matching.

    This makes the test robust to small naming changes such as:
        Parameter vs parameter
        Bootstrap mean vs bootstrap_mean
    """
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


def test_coefficient_bootstrap_returns_summary_and_draws():
    df = make_bootstrap_df()
    model = SimpleBootstrapModel()

    summary_df, draws_df = coefficient_bootstrap(
        model=model,
        df=df,
        n_bootstrap=30,
        seed=123,
        ci_level=0.90,
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(draws_df, pd.DataFrame)

    assert not summary_df.empty
    assert not draws_df.empty

    param_col = _find_column(summary_df, ["parameter", "param"])
    params = set(summary_df[param_col].astype(str))

    assert {"k1", "k2"}.issubset(params)

    numeric_summary = summary_df.select_dtypes(include=[np.number])
    assert not numeric_summary.empty
    assert np.isfinite(numeric_summary.to_numpy()).all()


def test_coefficient_bootstrap_is_reproducible_with_same_seed():
    df = make_bootstrap_df()
    model1 = SimpleBootstrapModel()
    model2 = SimpleBootstrapModel()

    summary1, draws1 = coefficient_bootstrap(
        model=model1,
        df=df,
        n_bootstrap=20,
        seed=999,
        ci_level=0.95,
    )

    summary2, draws2 = coefficient_bootstrap(
        model=model2,
        df=df,
        n_bootstrap=20,
        seed=999,
        ci_level=0.95,
    )

    pd.testing.assert_frame_equal(summary1, summary2)
    pd.testing.assert_frame_equal(draws1, draws2)


def test_coefficient_bootstrap_full_data_estimate_is_reasonable():
    df = make_bootstrap_df()
    model = SimpleBootstrapModel()

    summary_df, _ = coefficient_bootstrap(
        model=model,
        df=df,
        n_bootstrap=10,
        seed=1,
        ci_level=0.95,
    )

    param_col = _find_column(summary_df, ["parameter", "param"])
    estimate_col = _find_column(summary_df, ["estimate", "full_estimate"])

    estimates = {
        str(row[param_col]): float(row[estimate_col])
        for _, row in summary_df.iterrows()
    }

    assert np.isclose(estimates["k1"], 0.5, atol=1e-10)
    assert np.isclose(estimates["k2"], 0.8, atol=1e-10)
