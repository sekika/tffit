import numpy as np
from .registry import register_model
from .base import BaseModel


@register_model('kr')
class KRModel(BaseModel):
    """
    KR model implementation based on exchangeable potassium and RIP.
    Equation: log10(TF) = -k1 - k2 * log10(Kex) - k3 * log10(RIP)
    """

    def init_model(self):
        """
        Initialize the structural metadata for the KR model.
        """
        # The KR model uses RIP in addition to Kex
        self.features = ['RIP']

        # Formula string for reference
        self.formula_str = 'log10(TF) = -k1 - k2 * log10(Kex) - k3 * log10(RIP)'

        # Target column name
        self.target_col = 'log10_TF'

        # Parameters metadata
        self.params_meta = [
            {'key': 'k1', 'label': 'k1', 'desc': ''},
            {'key': 'k2', 'label': 'k2', 'desc': '(expected >0)'},
            {'key': 'k3', 'label': 'k3', 'desc': '(expected >0)'},
        ]

    def _fit(self, y, K, X=None, train_df=None):
        """
        Optimize the model parameters (k1, k2, k3) using Ordinary Least Squares.
        """
        # Convert inputs to numpy arrays
        y = np.asarray(y, float)
        K = np.asarray(K, float)
        RIP = np.asarray(X['RIP'], float)

        # Mask valid entries
        m = np.isfinite(y) & np.isfinite(K) & (
            K > 0) & np.isfinite(RIP) & (RIP > 0)
        y, K, RIP = y[m], K[m], RIP[m]

        if len(y) < 3:
            raise ValueError('Too few data points (>=3 required for KR model)')

        # Equation: y = -k2 * log10(K) - k3 * log10(RIP) - k1
        # Let x1 = log10(K) and x2 = log10(RIP).
        # We fit y = a*x1 + b*x2 + c, where a = -k2, b = -k3, and c = -k1.
        x1 = np.log10(K)
        x2 = np.log10(RIP)

        # Create design matrix [x1, x2, 1]
        Xmat = np.column_stack([x1, x2, np.ones_like(x1)])

        # Solve OLS
        coef, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
        a, b, c = coef

        k2 = -a
        k3 = -b
        k1 = -c

        # Calculate predicted values and residuals
        yhat = Xmat @ coef
        resid = y - yhat
        sst = float(np.sum((y - np.mean(y))**2))

        return {
            'k1': float(k1),
            'k2': float(k2),
            'k3': float(k3),
            'rmse_log10': float(np.sqrt(np.mean(resid**2))),
            'mae_log10': float(np.mean(np.abs(resid))),
            'r2_log10': float(1 - np.sum(resid**2) / sst if sst > 0 else np.nan),
            'n_used': int(len(y))
        }

    def _predict(self, K, X=None, fit=None):
        """
        Estimate log10(TF) values using the fitted parameters.
        """
        K = np.asarray(K, float)
        RIP = np.asarray(X['RIP'], float)

        # Predict log10(TF) = -k1 - k2*log10(Kex) - k3*log10(RIP)
        return -fit['k1'] - fit['k2'] * np.log10(K) - fit['k3'] * np.log10(RIP)
