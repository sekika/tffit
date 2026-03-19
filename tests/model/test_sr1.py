import pytest
import numpy as np
import pandas as pd
from src.model.sr1 import SR1Model


def test_sr1_fit_recovery():
    """
    Test 1: Parameter recovery for SR1 model.
    Equation: log10(TF) = -(RIP - log10(RIP)) - k1*(k2 - RIP)*K
    """
    model = SR1Model()
    model.init_model()

    # 1. Create Synthetic Data
    np.random.seed(42)
    n_samples = 100

    # K is Ex-K (raw potassium)
    K = np.random.uniform(0.01, 0.5, n_samples)

    # RIP values must be within the search bounds (0.5 - 5.0)
    RIP = np.random.uniform(1.0, 4.0, n_samples)
    X = {"RIP": RIP}

    # Define ground truth parameters within the bounds [10, 0.5] to [1000, 5]
    true_params = {
        "k1": 150.0,
        "k2": 2.5
    }

    y = model._predict(K, X, true_params)

    # 2. Fit
    result = model._fit(y, K, X)

    # 3. Verification
    # Use a slightly larger atol for numerical stability if needed,
    # but 1e-3 is generally safe for clean synthetic data.
    assert np.isclose(result["k1"], true_params["k1"], atol=1e-3)
    assert np.isclose(result["k2"], true_params["k2"], atol=1e-3)
    assert result["opt_success"] is True


def test_sr1_insufficient_data():
    """
    Test 2: Error handling for SR1.
    Validation: if len(y) < 3: raise ValueError
    """
    model = SR1Model()
    model.init_model()

    K = np.array([0.1, 0.2])
    y = np.array([1.0, 2.0])
    X = {"RIP": np.array([100.0, 100.0])}

    with pytest.raises(ValueError, match="Too few data points"):
        model._fit(y, K, X)


def test_sr1_predict_shape():
    """
    Test 3: Ensure prediction output shape matches input.
    """
    model = SR1Model()
    fit = {"k1": 100.0, "k2": 300.0}
    K = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    X = {"RIP": np.array([100, 150, 200, 250, 300])}

    y_pred = model._predict(K, X, fit)
    assert y_pred.shape == (5,)
