import numpy as np
from .registry import register_model
from .base import BaseModel


@register_model('k')
class KModel(BaseModel):
    """
    K model implementation based on exchangeable potassium.
    Equation: log10(TF) = -k1 - k2 * log10(Kex)
    """

    def init_model(self):
        """
        Initialize the structural metadata for the K model.
        """
        # The K model only uses Kex (passed as K), so no extra features are needed from X
        self.features = []

        # Formula string for reference
        self.formula_str = 'log10(TF) = -k1 - k2 * log10(Kex)'

        # Target column name
        self.target_col = 'log10_TF'

        # Parameters metadata
        self.params_meta = [
            {'key': 'k1', 'label': 'k1', 'desc': ''},
            {'key': 'k2', 'label': 'k2', 'desc': '(expected >0)'},
        ]

    def _fit(self, y, K, X=None, train_df=None):
        """
        Optimize the model parameters (k1, k2) using Ordinary Least Squares.
        Note: The primary variable is passed as 'K' representing Kex.
        """
        # Convert inputs to numpy arrays
        y = np.asarray(y, float)
        K = np.asarray(K, float)

        # Mask valid entries
        m = np.isfinite(y) & np.isfinite(K) & (K > 0)
        y, K = y[m], K[m]

        if len(y) < 2:
            raise ValueError('Too few data points (>=2 required for K model)')

        # Equation: y = -k2 * log10(K) - k1
        # Let x_val = log10(K). We fit y = a * x_val + c, where a = -k2 and c = -k1
        x_val = np.log10(K)

        # Create design matrix [x_val, 1]
        Xmat = np.column_stack([x_val, np.ones_like(x_val)])

        # Solve OLS
        coef, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
        a, c = coef

        k2 = -a
        k1 = -c

        # Calculate predicted values and residuals
        yhat = Xmat @ coef
        resid = y - yhat
        sst = float(np.sum((y - np.mean(y))**2))

        return {
            'k1': float(k1),
            'k2': float(k2),
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

        # Predict log10(TF) = -k1 - k2 * log10(Kex)
        return -fit['k1'] - fit['k2'] * np.log10(K)
