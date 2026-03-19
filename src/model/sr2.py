import numpy as np
from scipy.optimize import least_squares

from .registry import register_model
from .base import BaseModel


@register_model('sr2')
class SR2Model(BaseModel):
    """
    Symbolic Regression 2 (SR2) model for radiocesium soil-to-plant transfer.

    This model is another empirical formulation discovered via symbolic regression.
    A key feature of this model is the incorporation of a threshold or 'hinge' 
    function: max(k2 - RIP, 0). 

    This structure implies that the inhibitory effect of exchangeable potassium (K) 
    only occurs when the soil's Radiocesium Interception Potential (RIP) is below 
    a certain threshold (k2). If RIP exceeds k2, the second term drops to zero, 
    and the log10(TF) is dictated entirely by a fixed linear response to RIP (-0.85).

    The governing equation is:
    log10(TF) = -0.85*RIP - k1*max(k2 - RIP, 0)*K, where K = Ex-K
    """

    def init_model(self):
        """
        Initialize the structural metadata for the SR2 model.

        Registers 'RIP' as a mandatory feature alongside the default 'K' (Ex-K). 
        Defines the parameters (k1, k2), where k1 scales the interaction effect, 
        and k2 acts as the RIP threshold parameter.
        """
        self.features = ['RIP']
        self.formula_str = 'log10(TF) = -0.85*RIP - k1*max(k2 - RIP, 0)*K, where K = Ex-K'
        self.target_col = 'log10_TF'
        self.params_meta = [
            {'key': 'k1', 'label': 'k1', 'desc': ''},
            {'key': 'k2', 'label': 'k2', 'desc': ''},
        ]

    def _fit(self, y, K, X, train_df=None):
        """
        Optimize the model parameters (k1, k2) against the dataset.

        Uses non-linear least squares (`scipy.optimize.least_squares`) with 
        multiple initial guesses and robust loss functions to solve for k1 and k2. 
        The optimization handles the piecewise nature of the `maximum` function 
        without the need for the klim parameter.

        Parameters
        ----------
        y : array_like
            Target variable array, strictly representing log10(TF).
        K : array_like
            Predictor array representing raw exchangeable potassium (Ex-K).
        X : dict or pandas.DataFrame
            Must contain the 'RIP' column.
        train_df : pandas.DataFrame or None, optional
            Reference to the training data.

        Returns
        -------
        dict
            Optimized model parameters (k1, k2) and evaluation metrics.
        """
        y = np.asarray(y, float)
        Kex = np.asarray(K, float)            # Ex-K
        RIP = np.asarray(X['RIP'], float)

        # Mask valid entries
        m = np.isfinite(y) & np.isfinite(
            Kex) & np.isfinite(RIP) & (Kex >= 0) & (RIP > 0)
        y, Kex, RIP = y[m], Kex[m], RIP[m]

        n_used = len(y)
        if n_used < 3:
            raise ValueError("Too few data points (>=3 required)")

        term1 = -0.85 * RIP

        def resid_fn(params):
            k1, k2 = params
            # Equation: log(TF) = -0.85 * RIP - k1 * max(k2 - RIP, 0) * Kex
            yhat = term1 - k1 * np.maximum(k2 - RIP, 0.0) * Kex
            return y - yhat

        # Multi-start initialization (Using the same logic as sr1)
        init_candidates = []
        for k1_0 in (50, 120, 200):
            for k2_0 in (1.5, 2.5):
                init_candidates.append(np.array([k1_0, k2_0], dtype=float))

        # Optimization settings (Using the same bounds as sr1)
        from scipy.optimize import least_squares
        best_res = None
        lower = np.array([10, 0.5], dtype=float)
        upper = np.array([1000, 5], dtype=float)

        # Multi-start optimization to find the global minimum
        for p0 in init_candidates:
            try:
                res = least_squares(
                    resid_fn, p0,
                    bounds=(lower, upper),
                    method='trf',
                    loss='soft_l1',  # Robust to outliers
                    f_scale=1.0,
                    max_nfev=20000
                )
                if (best_res is None) or (res.cost < best_res.cost):
                    best_res = res
            except Exception:
                continue

        if best_res is None or not best_res.success:
            raise RuntimeError("Optimization failed to converge for SR2.")

        k1, k2 = (float(v) for v in best_res.x)

        # Compute final metrics
        final_resid = resid_fn(best_res.x)
        rmse = float(np.sqrt(np.mean(final_resid**2)))
        mae = float(np.mean(np.abs(final_resid)))
        sst = float(np.sum((y - np.mean(y))**2))
        r2 = float(1 - np.sum(final_resid**2) / sst) if sst > 0 else np.nan

        return {
            "k1": k1,
            "k2": k2,
            "rmse_log10": rmse,
            "mae_log10": mae,
            "r2_log10": r2,
            "n_used": int(n_used),
            "opt_success": bool(best_res.success),
        }

    def _predict(self, K, X, fit):
        """
        Estimate log10(TF) values using the fitted SR2 parameters.

        Parameters
        ----------
        K : array_like
            Input array of raw exchangeable potassium (Ex-K).
        X : dict or pandas.DataFrame
            Must contain the 'RIP' column.
        fit : dict
            Dictionary containing the optimized parameters ('k1', 'k2').

        Returns
        -------
        array_like
            The predicted log10(TF) values.
        """
        Kex = np.asarray(K, float)
        RIP = np.asarray(X['RIP'], float)

        term1 = -0.85 * RIP
        return term1 - fit["k1"] * np.maximum(fit["k2"] - RIP, 0.0) * Kex
