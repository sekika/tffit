import numpy as np
from .registry import register_model
from .base import BaseModel


@register_model('krcs')
class KRCsModel(BaseModel):
    """
    KRCs model implementation based on exchangeable potassium, RIP, and soil 137Cs concentration.
    Equation: log10(TF) = -k1 - k2*log10(Kex) - k3*log10(RIP) + k4*log10(137Cs)
    """

    def init_model(self):
        """
        Initialize the structural metadata for the KRCs model.
        """
        # Use the correct raw column name from the dataset
        self.features = ['RIP', '137Cs']

        # Formula string for reference (Note the positive sign for k4)
        self.formula_str = 'log10(TF) = -k1 - k2*log10(Kex) - k3*log10(RIP) + k4*log10(137Cs)'

        # Target column name
        self.target_col = 'log10_TF'

        # Parameters metadata
        self.params_meta = [
            {'key': 'k1', 'label': 'k1', 'desc': ''},
            {'key': 'k2', 'label': 'k2', 'desc': '(expected >0)'},
            {'key': 'k3', 'label': 'k3', 'desc': '(expected >0)'},
            {'key': 'k4', 'label': 'k4', 'desc': '(expected >0)'},
        ]

    def _fit(self, y, K, X=None, train_df=None):
        """
        Optimize the model parameters (k1, k2, k3, k4) using Ordinary Least Squares.
        """
        # Convert inputs to numpy arrays
        y = np.asarray(y, float)
        K = np.asarray(K, float)
        RIP = np.asarray(X['RIP'], float)
        Cs = np.asarray(X['137Cs'], float)

        # Mask valid entries
        m = (np.isfinite(y) & np.isfinite(K) & (K > 0) &
             np.isfinite(RIP) & (RIP > 0) &
             np.isfinite(Cs) & (Cs > 0))
        y, K, RIP, Cs = y[m], K[m], RIP[m], Cs[m]

        if len(y) < 4:
            raise ValueError(
                'Too few data points (>=4 required for KRCs model)')

        # Equation: y = -k2 * log10(K) - k3 * log10(RIP) + k4 * log10(Cs) - k1
        # Let x1 = log10(K), x2 = log10(RIP), x3 = log10(Cs).
        # We fit y = a*x1 + b*x2 + c*x3 + d
        # where a = -k2, b = -k3, c = k4, and d = -k1.
        x1 = np.log10(K)
        x2 = np.log10(RIP)
        x3 = np.log10(Cs)

        # Create design matrix [x1, x2, x3, 1]
        Xmat = np.column_stack([x1, x2, x3, np.ones_like(x1)])

        # Solve OLS
        coef, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
        a, b, c, d = coef

        # Map OLS coefficients to model parameters based on the specific signs in the equation
        k2 = -a
        k3 = -b
        k4 = c   # Positive sign in the formula
        k1 = -d

        # Calculate predicted values and residuals
        yhat = Xmat @ coef
        resid = y - yhat
        sst = float(np.sum((y - np.mean(y))**2))

        return {
            'k1': float(k1),
            'k2': float(k2),
            'k3': float(k3),
            'k4': float(k4),
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
        Cs = np.asarray(X['137Cs'], float)

        # Predict log10(TF) = -k1 - k2*log10(Kex) - k3*log10(RIP) + k4*log10(137Cs)
        return (-fit['k1'] - fit['k2'] * np.log10(K)
                - fit['k3'] * np.log10(RIP) + fit['k4'] * np.log10(Cs))
