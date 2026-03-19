import pytest
import numpy as np
import pandas as pd

# Import all models from the src directory
from src.model import (
    AbsalomModel,
    KModel,
    KRModel,
    KRCModel,
    KRPModel,
    KRCsModel,
    SR1Model,
    SR2Model
)


def generate_base_features(n=100, seed=42):
    """
    Generate a base set of features required for all model variants.
    Includes RIP, CEC, pH, and K_NH4 for the mechanistic Absalom(1999) model.
    """
    np.random.seed(seed)
    # Potassium (K) concentration typically follows a log-normal distribution in soil
    K = np.random.lognormal(mean=2, sigma=1, size=n)

    X = pd.DataFrame({
        'RIP': np.random.uniform(100, 3000, n),
        'CEC': np.random.uniform(5, 50, n),
        'pH': np.random.uniform(4.5, 7.5, n),
        '137Cs': np.random.uniform(10, 1000, n),
        # Required for original Absalom model
        'K_NH4': np.random.uniform(0.01, 0.5, n),
        # Organic matter for SR models
        'OM': np.random.uniform(1, 20, n)
    })
    return K, X


@pytest.fixture
def cv_test_df():
    """Generate a DataFrame suitable for LOSO and LOYO testing."""
    n = 12
    return pd.DataFrame({
        'TF': [10.0] * n,
        'Ex-K': [1.0] * n,
        'RIP': [100.0] * n,
        'Site': ['SiteA', 'SiteA', 'SiteB', 'SiteB', 'SiteC', 'SiteC'] * 2,
        'Year': [2020] * 6 + [2021] * 6
    })


@pytest.fixture
def mock_model():
    """A simple mock model that returns y = -1.0 regardless of input."""
    class MockModel:
        def __init__(self):
            self.features = ['RIP']

        def fit(self, y, K, X, train_df=None):
            return {'bias': -1.0}

        def predict(self, K, X, fit):
            return np.full(len(K), fit['bias'])
    return MockModel()
