import numpy as np
from .registry import register_model
from .base import BaseModel
from .common import keff_log10


@register_model('absalom')
class AbsalomModel(BaseModel):
    """
    Mathematical implementation of the modified Absalom radiocesium transfer model.

    The model incorporates CEC and RIP, assuming an inverse log-linear relationship 
    between the transfer factor and the effective exchangeable soil potassium ratio.
    The governing equation is:

    log10(TF) = -k1 - k2 * log10(Keff) - log10(RIP)

    where Keff = min(Ex-K / CEC, klim). 
    """

    def init_model(self):
        """
        Initialize the structural metadata for the Absalom model.
        """
        # List of features used in the model
        self.features = ['CEC', 'RIP']

        # Formula string for reference
        self.formula_str = 'log10(TF) = -k1 - k2 * log10(min(Ex-K/CEC, klim)) - log10(RIP)'

        # Target column name
        self.target_col = 'log10_TF'

        # Parameters metadata
        self.params_meta = [
            {'key': 'k1', 'label': 'k1', 'desc': ''},
            {'key': 'k2', 'label': 'k2', 'desc': '(expected >0)'},
        ]

    def _fit(self, y, K, X=None, train_df=None):
        """
        Optimize the model parameters (k1, k2, klim) against the dataset.
        """
        # Convert inputs to numpy arrays
        y = np.asarray(y, float)
        K = np.asarray(K, float)
        CEC = np.asarray(X['CEC'], float)
        RIP = np.asarray(X['RIP'], float)

        # Mask valid entries
        m = np.isfinite(y) & np.isfinite(K) & (K > 0) & np.isfinite(
            CEC) & (CEC > 0) & np.isfinite(RIP) & (RIP > 0)
        y, K, CEC, RIP = y[m], K[m], CEC[m], RIP[m]

        if len(y) < 3:
            raise ValueError('Too few data points (>=3 required)')

        # Adjust target variable and K to linearize the equation
        # Original: log10(TF) = -k1 - k2*log10(Keff) - log10(RIP)
        # Modified: log10(TF) + log10(RIP) = -k2*log10(Keff) - k1
        y_adj = y + np.log10(RIP)
        K_ratio = K / CEC

        # Function to calculate SSE for a given klim
        def sse_for_klim(klim):
            zlog = keff_log10(K_ratio, klim, kfun=self.kfun)
            Xmat = np.column_stack([zlog, np.ones_like(zlog)])
            coef, *_ = np.linalg.lstsq(Xmat, y_adj, rcond=None)
            a, c = coef  # a = -k2, c = -k1
            yhat = Xmat @ coef
            sse = float(np.sum((y_adj - yhat)**2))
            return sse, a, c

        # Determine klim
        if self.fix_klim:
            klim_hat = float(self.klim_fixed)
            sse, a, c = sse_for_klim(klim_hat)
        else:
            Kpos = K_ratio[K_ratio > 0]
            low = np.log(max(np.min(Kpos) * 0.1, 1e-12))
            high = np.log(np.max(Kpos) * 10.0)
            from scipy import optimize
            res = optimize.minimize_scalar(
                lambda lg: sse_for_klim(np.exp(lg))[0],
                bounds=(low, high),
                method='bounded',
                options={'maxiter': 200}
            )
            klim_hat = float(np.exp(res.x))
            sse, a, c = sse_for_klim(klim_hat)

        k2 = -a
        k1 = -c
        zlog = keff_log10(K_ratio, klim_hat, kfun=self.kfun)

        # Calculate predicted log10(TF) values and residuals
        yhat = -(k2 * zlog + k1) - np.log10(RIP)
        resid = y - yhat
        sst = float(np.sum((y - np.mean(y))**2))

        return {
            'k1': float(k1),
            'k2': float(k2),
            'klim': float(klim_hat),
            'rmse_log10': float(np.sqrt(np.mean(resid**2))),
            'mae_log10': float(np.mean(np.abs(resid))),
            'r2_log10': float(1 - np.sum(resid**2) / sst if sst > 0 else np.nan),
            'n_used': int(len(y)),
            'klim_is_fixed': bool(self.fix_klim)
        }

    def _predict(self, K, X=None, fit=None):
        """
        Estimate log10(TF) values using the fitted parameters.
        """
        K = np.asarray(K, float)
        CEC = np.asarray(X['CEC'], float)
        RIP = np.asarray(X['RIP'], float)

        K_ratio = K / CEC
        zlog = keff_log10(K_ratio, fit['klim'], kfun=self.kfun)

        # Predict log10(TF) based on fitted parameters
        return -(fit['k2'] * zlog + fit['k1']) - np.log10(RIP)
