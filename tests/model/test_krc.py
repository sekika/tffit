import pytest
import numpy as np
from src.model.krc import KRCModel


def test_krc_fit_estimation():
    """
    Test 1: Parameter estimation for the KRC model.
    KRC model equation: log10(TF) = -k1 - k2*log10(Kex) + k3*log10(RIP) + k4*log10(CEC)
    Uses OLS to accurately recover k1, k2, k3, and k4.
    """
    model = KRCModel()
    model.init_model()

    # 1. Create Synthetic Data
    K = np.geomspace(0.01, 1.0, 50)

    # Generate independent variables to avoid multicollinearity
    np.random.seed(42)
    X = {
        "RIP": np.random.uniform(100, 2000, 50),
        "CEC": np.random.uniform(5, 50, 50)  # Assuming a typical range for CEC
    }

    # Note that k3 and k4 have positive signs in the KRC equation
    true_params = {"k1": 1.5, "k2": 0.8, "k3": 0.4, "k4": 0.3}

    # Generate target y using the model's own logic
    y = model._predict(K, X, fit=true_params)

    # Add minimal noise
    y += np.random.normal(0, 0.0001, len(y))

    # 2. Fit the model
    result = model._fit(y, K, X)

    # 3. Verification
    # OLS should recover the parameters closely
    assert "k1" in result
    assert "k2" in result
    assert "k3" in result
    assert "k4" in result
    assert np.isclose(result["k1"], true_params["k1"], atol=1e-3)
    assert np.isclose(result["k2"], true_params["k2"], atol=1e-3)
    assert np.isclose(result["k3"], true_params["k3"], atol=1e-3)
    assert np.isclose(result["k4"], true_params["k4"], atol=1e-3)
    assert result["rmse_log10"] >= 0


def test_krc_prediction_consistency():
    """
    Test 2: Internal consistency of the prediction logic.
    """
    model = KRCModel()
    model.init_model()

    K = np.geomspace(0.01, 1.0, 50)
    np.random.seed(123)
    X = {
        "RIP": np.random.uniform(100, 2000, size=len(K)),
        "CEC": np.random.uniform(5, 50, size=len(K))
    }
    true_params = {"k1": 1.2, "k2": 0.9, "k3": 0.5, "k4": 0.2}

    y_true = model._predict(K, X, fit=true_params)

    # Re-predict using the exact same parameters
    y_pred = model._predict(K, X, fit=true_params)

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    assert correlation > 0.9999


def test_krc_insufficient_data():
    """
    Test 3: Error handling for insufficient data points.
    KRC model has 4 parameters (k1, k2, k3, k4), requiring at least 4 data points.
    """
    model = KRCModel()
    model.init_model()

    # Only 3 data points provided
    K = np.array([0.1, 0.2, 0.3])
    y = np.array([1.0, 2.0, 3.0])
    X = {
        "RIP": np.array([1000.0, 1500.0, 1200.0]),
        "CEC": np.array([10.0, 15.0, 20.0])
    }

    with pytest.raises(ValueError, match="Too few data points"):
        model._fit(y, K, X)
