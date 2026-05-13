import itertools
import math
import numpy as np
import pandas as pd

from .model.registry import get_model
from .cross_validation import cv_predictions
from .bootstrap import paired_cluster_bootstrap


def _make_model(model_name, fix_klim=False, klim_fixed=None):
    cls = get_model(model_name)

    try:
        return cls(fix_klim=fix_klim, klim_fixed=klim_fixed)
    except TypeError:
        return cls()


def _pretty_model_name(name):
    if name.lower() == "absalom":
        return "Absalom"
    return name.upper()


def _validation_list(validation):
    if validation == "both":
        return ["loso", "loyo"]
    return [validation]


def _validation_label(validation):
    return validation.upper()


def _rmse_from_se(se):
    se = np.asarray(se, dtype=float)
    return float(np.sqrt(np.mean(se)))


def _ci_string(ci_tuple):
    lo, hi = ci_tuple
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{lo:.6g}, {hi:.6g}]"


def cluster_level_pvalue(diffs, method="signflip", seed=12345):
    """
    Cluster-level paired p-value.

    diffs are cluster-level mean loss differences:
        e_model^2 - e_base^2

    Positive values indicate lower error for the base model.

    The default sign-flip test is two-sided.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]

    n = len(diffs)
    if n == 0 or method == "none":
        return np.nan

    obs = float(np.mean(diffs))

    if method == "signflip":
        # Exact sign-flip test for small n; otherwise Monte Carlo fallback.
        if n <= 20:
            vals = []
            for signs in itertools.product([-1.0, 1.0], repeat=n):
                vals.append(float(np.mean(np.asarray(signs) * diffs)))
            vals = np.asarray(vals, dtype=float)
            return float(np.mean(np.abs(vals) >= abs(obs)))
        else:
            rng = np.random.default_rng(seed)
            vals = []
            for _ in range(20000):
                signs = rng.choice([-1.0, 1.0], size=n, replace=True)
                vals.append(float(np.mean(signs * diffs)))
            vals = np.asarray(vals, dtype=float)
            return float(np.mean(np.abs(vals) >= abs(obs)))

    if method == "ttest":
        # Use scipy if available. If not, use a normal approximation.
        sd = float(np.std(diffs, ddof=1)) if n > 1 else np.nan
        if not np.isfinite(sd) or sd == 0:
            return np.nan

        t_stat = obs / (sd / math.sqrt(n))

        try:
            from scipy import stats
            return float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        except Exception:
            # Normal approximation fallback.
            z = abs(t_stat)
            p = math.erfc(z / math.sqrt(2.0))
            return float(p)

    raise ValueError(f"Unknown paired test method: {method}")


def _paired_prediction_frame(
    df,
    base_model_name,
    compare_model_name,
    validation,
    site_col="Site",
    year_col="Year",
    exclude_years=None,
    fix_klim=False,
    klim_fixed=None,
):
    base_model = _make_model(base_model_name, fix_klim=fix_klim, klim_fixed=klim_fixed)
    comp_model = _make_model(compare_model_name, fix_klim=fix_klim, klim_fixed=klim_fixed)

    base_pred = cv_predictions(
        model=base_model,
        df=df,
        cv=validation,
        model_name=base_model_name,
        site_col=site_col,
        year_col=year_col,
        exclude_years=exclude_years,
    )

    comp_pred = cv_predictions(
        model=comp_model,
        df=df,
        cv=validation,
        model_name=compare_model_name,
        site_col=site_col,
        year_col=year_col,
        exclude_years=exclude_years,
    )

    keep_base = [
        "row_id", "validation", "fold", "cluster",
        "y_obs", "y_pred", "residual", "squared_error"
    ]
    keep_comp = keep_base.copy()

    base_pred = base_pred[keep_base].rename(columns={
        "y_pred": "y_pred_base",
        "residual": "residual_base",
        "squared_error": "se_base",
    })

    comp_pred = comp_pred[keep_comp].rename(columns={
        "y_pred": "y_pred_model",
        "residual": "residual_model",
        "squared_error": "se_model",
    })

    merged = pd.merge(
        comp_pred,
        base_pred,
        on=["row_id", "validation", "fold", "cluster", "y_obs"],
        how="inner",
        validate="one_to_one",
    )

    return merged


def paired_comparison_table(
    df,
    base_model_name,
    compare_model_names,
    validation="both",
    n_bootstrap=10000,
    seed=12345,
    ci_level=0.95,
    paired_test="signflip",
    site_col="Site",
    year_col="Year",
    exclude_years=None,
    fix_klim=False,
    klim_fixed=None,
):
    """
    Create paired comparison table against a base model.

    Returns a DataFrame with:
        Validation, Comparison, RMSE model, RMSE base,
        delta RMSE, CI, delta MSE, CI, p-value
    """
    rows = []

    for val in _validation_list(validation):
        for comp_name in compare_model_names:
            paired = _paired_prediction_frame(
                df=df,
                base_model_name=base_model_name,
                compare_model_name=comp_name,
                validation=val,
                site_col=site_col,
                year_col=year_col,
                exclude_years=exclude_years,
                fix_klim=fix_klim,
                klim_fixed=klim_fixed,
            )

            se_model = paired["se_model"].to_numpy(dtype=float)
            se_base = paired["se_base"].to_numpy(dtype=float)
            clusters = paired["cluster"].to_numpy()

            rmse_model = _rmse_from_se(se_model)
            rmse_base = _rmse_from_se(se_base)

            mse_model = float(np.mean(se_model))
            mse_base = float(np.mean(se_base))

            delta_rmse = rmse_model - rmse_base
            delta_mse = mse_model - mse_base

            boot = paired_cluster_bootstrap(
                se_model=se_model,
                se_base=se_base,
                clusters=clusters,
                n_bootstrap=n_bootstrap,
                seed=seed,
                ci_level=ci_level,
            )

            cluster_diffs = (
                paired.assign(diff=paired["se_model"] - paired["se_base"])
                .groupby("cluster")["diff"]
                .mean()
                .to_numpy(dtype=float)
            )

            p_value = cluster_level_pvalue(cluster_diffs, method=paired_test, seed=seed)

            rows.append({
                "Validation": _validation_label(val),
                "Comparison": f"{_pretty_model_name(comp_name)} - {_pretty_model_name(base_model_name)}",
                "RMSE model": rmse_model,
                f"RMSE {_pretty_model_name(base_model_name)}": rmse_base,
                f"Delta RMSE vs {_pretty_model_name(base_model_name)}": delta_rmse,
                f"{int(ci_level * 100)}% CI Delta RMSE": _ci_string(boot["delta_rmse_ci"]),
                f"Delta MSE vs {_pretty_model_name(base_model_name)}": delta_mse,
                f"{int(ci_level * 100)}% CI Delta MSE": _ci_string(boot["delta_mse_ci"]),
                "p-value": p_value,
            })

    return pd.DataFrame(rows)


def cluster_loss_table(
    df,
    base_model_name,
    compare_model_names,
    validation="both",
    site_col="Site",
    year_col="Year",
    exclude_years=None,
    fix_klim=False,
    klim_fixed=None,
):
    """
    Cluster-level mean squared-error difference table.

    Difference is:
        e_model^2 - e_base^2

    Positive values indicate lower squared error for the base model.
    """
    all_tables = []

    for val in _validation_list(validation):
        cluster_col_name = "Site" if val == "loso" else "Year"

        first = True
        out = None

        for comp_name in compare_model_names:
            paired = _paired_prediction_frame(
                df=df,
                base_model_name=base_model_name,
                compare_model_name=comp_name,
                validation=val,
                site_col=site_col,
                year_col=year_col,
                exclude_years=exclude_years,
                fix_klim=fix_klim,
                klim_fixed=klim_fixed,
            )

            label = f"{_pretty_model_name(comp_name)} - {_pretty_model_name(base_model_name)}"

            tmp = (
                paired.assign(diff=paired["se_model"] - paired["se_base"])
                .groupby("cluster", sort=False)["diff"]
                .mean()
                .reset_index()
                .rename(columns={"cluster": cluster_col_name, "diff": label})
            )

            if first:
                out = tmp
                first = False
            else:
                out = pd.merge(out, tmp, on=cluster_col_name, how="outer")

        out.insert(0, "Validation", _validation_label(val))
        all_tables.append(out)

    return pd.concat(all_tables, ignore_index=True, sort=False)
