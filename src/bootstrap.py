import numpy as np
import pandas as pd

from .cross_validation import _build_yKX


def percentile_ci(values, ci_level=0.95):
    """
    Percentile bootstrap confidence interval.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan, np.nan

    alpha = 1.0 - float(ci_level)
    lo = 100.0 * alpha / 2.0
    hi = 100.0 * (1.0 - alpha / 2.0)

    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def coefficient_bootstrap(
    model,
    df,
    n_bootstrap=10000,
    seed=12345,
    ci_level=0.95,
    param_names=None,
):
    """
    Bootstrap confidence intervals for model coefficients.

    The model structure is fixed. Rows are resampled with replacement, and the
    model coefficients are re-estimated for each bootstrap sample.

    Parameters
    ----------
    model : object
        Model instance.
    df : pandas.DataFrame
        Full dataset.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        Random seed.
    ci_level : float
        Confidence interval level.
    param_names : list[str], optional
        Parameters to summarize. If None, parameters present in the full-data
        fit are used.

    Returns
    -------
    summary_df : pandas.DataFrame
        Summary table for coefficients.
    draws_df : pandas.DataFrame
        Bootstrap draws.
    """
    rng = np.random.default_rng(seed)

    y, K, X = _build_yKX(df, model)
    full_fit = model.fit(y, K, X, train_df=df)

    if param_names is None:
        candidate_names = ["k1", "k2", "k3", "k4", "k5", "klim"]
        param_names = [
            k for k in candidate_names
            if k in full_fit and full_fit[k] is not None
        ]

    draws = []
    n = len(df)

    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx].copy()

        try:
            y_b, K_b, X_b = _build_yKX(boot_df, model)
            fit_b = model.fit(y_b, K_b, X_b, train_df=boot_df)

            row = {"bootstrap": b + 1, "success": True}
            for p in param_names:
                row[p] = float(fit_b[p]) if p in fit_b and fit_b[p] is not None else np.nan
            draws.append(row)

        except Exception:
            row = {"bootstrap": b + 1, "success": False}
            for p in param_names:
                row[p] = np.nan
            draws.append(row)

    draws_df = pd.DataFrame(draws)

    rows = []
    for p in param_names:
        vals = draws_df.loc[draws_df["success"], p].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        ci_lo, ci_hi = percentile_ci(vals, ci_level=ci_level)

        estimate = float(full_fit[p]) if p in full_fit and full_fit[p] is not None else np.nan

        rows.append({
            "Parameter": p,
            "Estimate": estimate,
            "Bootstrap mean": float(np.mean(vals)) if vals.size else np.nan,
            "Bootstrap SD": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            f"{int(ci_level * 100)}% CI": f"[{ci_lo:.6g}, {ci_hi:.6g}]" if np.isfinite(ci_lo) else "",
            "Successful fits": int(vals.size),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df, draws_df


def paired_cluster_bootstrap(
    se_model,
    se_base,
    clusters,
    n_bootstrap=10000,
    seed=12345,
    ci_level=0.95,
):
    """
    Paired cluster bootstrap for differences in RMSE and MSE.

    Parameters
    ----------
    se_model : array-like
        Squared errors of the comparison model.
    se_base : array-like
        Squared errors of the base model.
    clusters : array-like
        Cluster labels. For LOSO, site labels. For LOYO, year labels.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        Random seed.
    ci_level : float
        Confidence interval level.

    Returns
    -------
    dict
        Bootstrap summaries for delta RMSE and delta MSE.
    """
    se_model = np.asarray(se_model, dtype=float)
    se_base = np.asarray(se_base, dtype=float)
    clusters = np.asarray(clusters)

    if not (len(se_model) == len(se_base) == len(clusters)):
        raise ValueError("se_model, se_base, and clusters must have the same length.")

    unique_clusters = np.array(pd.unique(clusters))
    rng = np.random.default_rng(seed)

    d_rmse = []
    d_mse = []

    for _ in range(int(n_bootstrap)):
        sampled_clusters = rng.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True
        )

        idx_parts = []
        for c in sampled_clusters:
            idx_parts.append(np.where(clusters == c)[0])

        idx = np.concatenate(idx_parts)

        mse_model_b = float(np.mean(se_model[idx]))
        mse_base_b = float(np.mean(se_base[idx]))

        rmse_model_b = float(np.sqrt(mse_model_b))
        rmse_base_b = float(np.sqrt(mse_base_b))

        d_rmse.append(rmse_model_b - rmse_base_b)
        d_mse.append(mse_model_b - mse_base_b)

    rmse_lo, rmse_hi = percentile_ci(d_rmse, ci_level=ci_level)
    mse_lo, mse_hi = percentile_ci(d_mse, ci_level=ci_level)

    return {
        "delta_rmse_draws": np.asarray(d_rmse, dtype=float),
        "delta_mse_draws": np.asarray(d_mse, dtype=float),
        "delta_rmse_ci": (rmse_lo, rmse_hi),
        "delta_mse_ci": (mse_lo, mse_hi),
    }
