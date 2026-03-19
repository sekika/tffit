import numpy as np
from src.model.common import log10_strict


def _build_yKX(df, model):
    """
    Extract and prepare the target variable and feature arrays from the dataset.

    This function extracts the transfer factor (TF) and exchangeable potassium (Ex-K),
    applying a strict base-10 logarithmic transformation to TF. It also extracts any
    additional predictive features required by the specific mathematical model.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing soil physicochemical properties and TF.
    model : object
        The instantiated model object. Must have a `features` attribute (list of strings)
        specifying required column names.

    Returns
    -------
    y : numpy.ndarray
        The target variable, log10(TF), as a float array.
    K : numpy.ndarray
        The primary predictor, exchangeable potassium (Ex-K), as a float array.
    X : dict or None
        A dictionary mapping feature names to their corresponding numpy.ndarray of values.
        Returns None if the model requires no additional features.

    Raises
    ------
    KeyError
        If 'TF', 'Ex-K', or any model-specific required features are missing from `df`.
    """
    if "TF" not in df.columns:
        raise KeyError("Column 'TF' not found in DataFrame.")
    if "Ex-K" not in df.columns:
        raise KeyError("Column 'Ex-K' not found in DataFrame.")

    y = log10_strict(df["TF"].to_numpy(dtype=float), name="TF")
    K = df["Ex-K"].to_numpy(dtype=float)

    feats = getattr(model, "features", []) or []
    if len(feats) == 0:
        X = None
    else:
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required feature columns for model {type(model).__name__}: {missing}")
        X = {c: df[c].to_numpy(dtype=float) for c in feats}

    return y, K, X


def _rmse_micro(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true : array_like
        Ground truth (correct) target values.
    y_pred : array_like
        Estimated target values returned by the model.

    Returns
    -------
    float
        The calculated RMSE value.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r = y_true - y_pred
    return float(np.sqrt(np.mean(r * r)))


def loso(model, df, site_col="Site"):
    """
    Perform Leave-One-Site-Out (LOSO) cross-validation to evaluate spatial generalization.

    In each iteration, all observations from a single physical site are held out as the test set,
    and the model is trained on the remaining sites. This process assesses the model's capability
    to robustly extrapolate predictions to unobserved geographical locations.

    Parameters
    ----------
    model : object
        The mathematical model object to be evaluated. Must implement `fit` and `predict` methods.
    df : pandas.DataFrame
        The dataset containing all observations, including the site identifier column.
    site_col : str, optional
        The name of the column in `df` that contains the site identifiers. Default is "Site".

    Returns
    -------
    float
        The micro-averaged Root Mean Square Error (RMSE) computed by pooling the squared
        prediction errors over all held-out observations across all folds.

    Raises
    ------
    KeyError
        If the specified `site_col` is not found in the DataFrame.
    RuntimeError
        If a fold results in an empty training or testing set, or if no predictions are produced.
    """
    if site_col not in df.columns:
        raise KeyError(f"Site column '{site_col}' not found.")

    sites = df[site_col].dropna().unique().tolist()

    all_obs = []
    all_pred = []

    for site in sites:
        train_df = df[df[site_col] != site].copy()
        test_df = df[df[site_col] == site].copy()

        if len(test_df) == 0 or len(train_df) == 0:
            raise RuntimeError(f"Empty fold encountered for site={site!r}")

        y_train, K_train, X_train = _build_yKX(train_df, model)
        fit_result = model.fit(y_train, K_train, X_train, train_df=train_df)

        y_test, K_test, X_test = _build_yKX(test_df, model)
        y_pred = model.predict(K_test, X_test, fit_result)

        all_obs.append(y_test)
        all_pred.append(y_pred)

    if not all_obs:
        raise RuntimeError("LOSO produced no predictions.")

    y_all = np.concatenate(all_obs)
    yhat_all = np.concatenate(all_pred)
    return _rmse_micro(y_all, yhat_all)


def loyo(model, df, year_col="Year", exclude_years=None):
    """
    Perform Leave-One-Year-Out (LOYO) cross-validation to evaluate temporal generalization.

    In each iteration, all observations from a single sampling year are held out as the test set,
    and the model is trained on the remaining years. This process assesses the model's capability
    to robustly extrapolate predictions to unobserved years or future scenarios, accounting for
    long-term temporal dynamics.

    Parameters
    ----------
    model : object
        The mathematical model object to be evaluated. Must implement `fit` and `predict` methods.
    df : pandas.DataFrame
        The dataset containing all observations, including the year identifier column.
    year_col : str, optional
        The name of the column in `df` that contains the year identifiers. Default is "Year".
    exclude_years : list or set of int, optional
        Specific years to exclude from being evaluated as a hold-out test set.

    Returns
    -------
    overall_rmse : float
        Micro-averaged RMSE computed by pooling the squared prediction errors over all 
        validation points across all evaluated years.
    per_year_rmse : dict[int, float]
        RMSE for each left-out year, computed solely within that specific year's validation points.

    Raises
    ------
    KeyError
        If the specified `year_col` is not found in the DataFrame.
    RuntimeError
        If a fold results in an empty training or testing set, or if no predictions are produced.
    """
    if year_col not in df.columns:
        raise KeyError(f"Year column '{year_col}' not found.")

    years = df[year_col].dropna().unique().tolist()
    years = sorted([int(y) for y in years])

    if exclude_years is not None:
        exclude_set = set(int(y) for y in exclude_years)
        years = [y for y in years if y not in exclude_set]

    all_obs = []
    all_pred = []
    per_year_rmse = {}

    for year in years:
        train_df = df[df[year_col] != year].copy()
        test_df = df[df[year_col] == year].copy()

        if len(test_df) == 0 or len(train_df) == 0:
            raise RuntimeError(f"Empty fold encountered for year={year}")

        y_train, K_train, X_train = _build_yKX(train_df, model)
        fit_result = model.fit(y_train, K_train, X_train, train_df=train_df)

        y_test, K_test, X_test = _build_yKX(test_df, model)
        y_pred = model.predict(K_test, X_test, fit_result)

        # store for overall micro RMSE
        all_obs.append(y_test)
        all_pred.append(y_pred)

        # compute per-year RMSE (within-fold RMSE)
        per_year_rmse[int(year)] = _rmse_micro(y_test, y_pred)

    if not all_obs:
        raise RuntimeError("LOYO produced no predictions.")

    y_all = np.concatenate(all_obs)
    yhat_all = np.concatenate(all_pred)
    overall_rmse = _rmse_micro(y_all, yhat_all)

    return overall_rmse, per_year_rmse
