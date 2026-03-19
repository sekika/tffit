import pytest
import numpy as np
import pandas as pd
from src.cross_validation import _build_yKX, _rmse_micro, loso, loyo

# --- Helper Function Tests ---


def test_build_yKX_success(cv_test_df, mock_model):
    """Verify that features and targets are correctly extracted."""
    y, K, X = _build_yKX(cv_test_df, mock_model)

    # TF=10.0 -> log10(10.0) = 1.0
    assert np.allclose(y, 1.0)
    assert 'RIP' in X
    assert len(K) == len(cv_test_df)


def test_build_yKX_missing_column(cv_test_df, mock_model):
    df_broken = cv_test_df.drop(columns=['TF'])
    with pytest.raises(KeyError, match="Column 'TF' not found"):
        _build_yKX(df_broken, mock_model)


def test_rmse_micro():
    y_true = [1.0, 2.0]
    y_pred = [1.0, 4.0]  # error is [0, -2]
    # sqrt(mean(0^2 + 2^2)) = sqrt(2) = 1.414...
    assert np.isclose(_rmse_micro(y_true, y_pred), np.sqrt(2))

# --- LOSO Tests ---


def test_loso_execution(cv_test_df, mock_model):
    """
    Test Leave-One-Site-Out execution.
    With 3 sites (A, B, C), it should run 3 folds.
    """
    # mock_model predicts -1.0, true y is 1.0 -> error is 2.0 per point
    # Overall RMSE should be exactly 2.0
    total_rmse = loso(mock_model, cv_test_df, site_col="Site")
    assert np.isclose(total_rmse, 2.0)


def test_loso_invalid_site_col(cv_test_df, mock_model):
    with pytest.raises(KeyError, match="Site column 'Location' not found"):
        loso(mock_model, cv_test_df, site_col="Location")

# --- LOYO Tests ---


def test_loyo_execution(cv_test_df, mock_model):
    """
    Test Leave-One-Year-Out execution.
    With 2 years (2020, 2021), it should return overall RMSE and per-year dict.
    """
    overall, per_year = loyo(mock_model, cv_test_df, year_col="Year")

    assert np.isclose(overall, 2.0)
    assert len(per_year) == 2
    assert per_year[2020] == 2.0
    assert per_year[2021] == 2.0


def test_loyo_exclude_years(cv_test_df, mock_model):
    """Check if exclude_years correctly skips a fold."""
    # Exclude 2021, so only 2020 is evaluated as a test set
    overall, per_year = loyo(mock_model, cv_test_df,
                             year_col="Year", exclude_years=[2021])
    assert list(per_year.keys()) == [2020]


def test_loyo_empty_fold_error(cv_test_df, mock_model):
    """If a year has no data, it should raise RuntimeError."""
    # Create a situation where one year exists but has no samples (shouldn't happen with unique(),
    # but we can force it by passing a filtered DF)
    df_only_2020 = cv_test_df[cv_test_df['Year'] == 2020]
    # Here, train_df for year 2020 will be empty
    with pytest.raises(RuntimeError, match="Empty fold encountered"):
        loyo(mock_model, df_only_2020, year_col="Year")
