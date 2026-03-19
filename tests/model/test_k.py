import pytest
import numpy as np
import pandas as pd
from src.model.k import KModel


def test_k_fit_estimation():
    """
    Test 1: Parameter estimation for the K model.
    K model equation: log10(TF) = -k1 - k2*log10(Kex)
    Since it uses simple OLS, it should recover parameters accurately.
    """
    model = KModel()
    model.init_model()

    # 1. Create Synthetic Data
    K = np.geomspace(0.01, 1.0, 50)
    X = {}  # K model does not use extended variables

    true_params = {"k1": 1.2, "k2": 0.9}

    # Generate target y using the model's own logic
    y = model._predict(K, X, fit=true_params)

    # Add minimal noise
    y += np.random.normal(0, 0.0001, len(y))

    # 2. Fit the model
    result = model._fit(y, K, X)

    # 3. Verification
    # Even with slight noise, OLS should recover the parameters closely
    assert "k1" in result
    assert "k2" in result
    assert np.isclose(result["k1"], true_params["k1"], atol=1e-3)
    assert np.isclose(result["k2"], true_params["k2"], atol=1e-3)
    assert result["rmse_log10"] >= 0


def test_k_prediction_consistency():
    """
    Test 2: Internal consistency of the prediction logic.
    """
    model = KModel()
    model.init_model()

    K = np.geomspace(0.01, 1.0, 50)
    X = {}
    true_params = {"k1": 1.5, "k2": 0.7}

    y_true = model._predict(K, X, fit=true_params)

    # Re-predict using the exact same parameters
    y_pred = model._predict(K, X, fit=true_params)

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    assert correlation > 0.9999


def test_k_insufficient_data():
    """
    Test 3: Error handling for insufficient data points.
    K model requires at least 2 data points to fit a line.
    """
    model = KModel()
    model.init_model()

    # Only 1 data point provided
    K = np.array([0.1])
    y = np.array([1.0])
    X = {}

    with pytest.raises(ValueError, match="Too few data points"):
        model._fit(y, K, X)
