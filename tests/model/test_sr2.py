import pytest
import numpy as np
import pandas as pd
from src.model.sr2 import SR2Model


def test_sr2_fit_recovery():
    """
    Test 1: Parameter recovery for SR2 model.
    Equation: log10(TF) = -0.85*RIP - k1*max(k2 - RIP, 0)*K
    """
    model = SR2Model()
    model.init_model()

    # 1. Create Synthetic Data
    np.random.seed(42)
    n_samples = 100

    # K is Ex-K (raw potassium)
    K = np.random.uniform(0.01, 0.5, n_samples)

    # RIP values within the search bounds (0.5 - 5.0)
    # Ensure some values are below and some are above k2=3.0
    RIP = np.random.uniform(1.0, 5.0, n_samples)
    X = {"RIP": RIP}

    # Define ground truth parameters within the bounds [10, 0.5] to [1000, 5]
    true_params = {
        "k1": 100.0,
        "k2": 3.0
    }

    y = model._predict(K, X, true_params)

    # 2. Fit
    result = model._fit(y, K, X)

    # 3. Verification
    assert np.isclose(result["k1"], true_params["k1"], atol=1e-3)
    assert np.isclose(result["k2"], true_params["k2"], atol=1e-3)
    assert result["opt_success"] is True


def test_sr2_insufficient_data():
    """
    Test 2: Error handling for SR2.
    """
    model = SR2Model()
    model.init_model()

    K = np.array([0.1, 0.2])
    y = np.array([1.0, 2.0])
    X = {"RIP": np.array([1.0, 1.0])}

    with pytest.raises(ValueError, match="Too few data points"):
        model._fit(y, K, X)


def test_sr2_predict_threshold_effect():
    """
    Test 3: Verify the threshold (k2) logic in prediction.
    When RIP > k2, the second term should be zero.
    """
    model = SR2Model()
    # Parameters within bounds
    fit = {"k1": 100.0, "k2": 3.0}

    K = np.array([0.5])
    X = {"RIP": np.array([4.0])}

    y_pred = model._predict(K, X, fit)

    # Expected: -0.85 * 4.0 - 100 * 0 * 0.5 = -3.4
    assert np.isclose(y_pred[0], -0.85 * 4.0)
