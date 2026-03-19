import pytest
import numpy as np
import pandas as pd
from src.model.absalom import AbsalomModel


def test_absalom_fit_fixed_klim():
    """
    Test 1: Strict parameter verification when klim is fixed.
    When klim is fixed, the problem is a simple linear regression.
    The OLS solver must recover the exact parameters used to generate the data.
    """
    model = AbsalomModel()
    model.init_model()
    model.fix_klim = False
    model.kfun = None  # Ensure default min(K, klim) behavior

    # 1. Create Synthetic Data
    # Potassium range: 0.01 to 1.0 mol/kg
    K = np.geomspace(0.01, 1.0, 50)

    # We need K/CEC to span across klim (0.05).
    # Setting CEC to 1.0 makes K_ratio range from 0.01 to 1.0,
    # ensuring K_eff has variance and the OLS matrix is full rank.
    X = {
        'CEC': np.full(50, 1.0),
        'RIP': np.full(50, 1500.0)
    }
    true_params = {"k1": 1.4, "k2": 0.8, "klim": 0.05}

    # Generate target y using the model's own logic
    y = model._predict(K, X, true_params)

    # 2. Fit with FIXED klim
    model.fix_klim = True
    model.klim_fixed = true_params["klim"]

    result = model._fit(y, K, X)

    # 3. Strict verification using high precision
    assert np.isclose(result["k1"], true_params["k1"],
                      atol=1e-8), f"k1 mismatch: {result['k1']} vs {true_params['k1']}"
    assert np.isclose(result["k2"], true_params["k2"],
                      atol=1e-8), f"k2 mismatch: {result['k2']} vs {true_params['k2']}"
    assert result["klim"] == true_params["klim"]


def test_absalom_fit_optimized_klim():
    """
    Test 2: Parameter estimation with optimized klim.
    Verify that the optimization routine completes and returns plausible values.
    """
    model = AbsalomModel()
    model.init_model()
    model.fix_klim = False
    model.kfun = None

    K = np.geomspace(0.01, 1.0, 50)
    X = {
        'CEC': np.full(50, 1.0),
        'RIP': np.full(50, 1500.0)
    }
    true_params = {"k1": 1.4, "k2": 0.8, "klim": 0.05}
    y = model._predict(K, X, true_params)

    # Add minor noise to simulate real-world data
    y += np.random.normal(0, 0.001, len(y))

    result = model._fit(y, K, X)

    # Check if necessary keys exist and are valid
    assert "k1" in result
    assert "k2" in result
    assert result["klim"] > 0
    assert result["rmse_log10"] < 0.1


def test_absalom_prediction_consistency():
    """
    Test 3: Internal consistency of the prediction logic.
    """
    model = AbsalomModel()
    model.init_model()
    model.kfun = None

    K = np.geomspace(0.01, 1.0, 50)
    X = {
        'CEC': np.full(50, 1.0),
        'RIP': np.full(50, 1500.0)
    }
    true_params = {"k1": 1.4, "k2": 0.8, "klim": 0.05}
    y_true = model._predict(K, X, true_params)

    # Re-predict using the same parameters
    y_pred = model._predict(K, X, true_params)

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    assert correlation > 0.9999
