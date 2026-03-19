# src/model/base.py
"""
Base class module for radiocesium soil-to-plant transfer models.

This module defines the foundational abstract interface for all mathematical 
models used to predict the soil-to-wheat transfer factor (TF) of radiocesium. 
It establishes a unified workflow for parameter optimization (fitting) and 
extrapolation (prediction), ensuring seamless integration with the cross-validation 
and evaluation pipelines.
"""

import numpy as np


class BaseModel:
    """
    Abstract base class for soil-to-plant radiocesium transfer models.

    This class defines the standard interface for processing soil physicochemical 
    properties to estimate the base-10 logarithm of the transfer factor, log10(TF).
    All specific mathematical formulations (e.g., empirical Absalom-based models 
    or symbolic regression models) must inherit from this class and implement 
    the protected `_fit` and `_predict` methods.
    """

    def __init__(self, fix_klim=False, klim_fixed=None, kfun=None):
        """
        Initialize the baseline model parameters and configuration.

        Parameters
        ----------
        fix_klim : bool, optional
            If True, constrains the potassium limitation threshold (k_lim) 
            to a predefined constant rather than estimating it as a free 
            parameter during the optimization process. Default is False.
        klim_fixed : float or None, optional
            The specific empirical value to use for k_lim if `fix_klim` is True.
            This conceptually represents the saturation point of plant potassium 
            uptake (e.g., where effective potassium, K_eff, is capped).
        kfun : callable or None, optional
            An optional mathematical function applied during the computation of 
            effective exchangeable potassium (K_eff) or other custom K dynamics.
        """
        self.fix_klim = fix_klim
        self.klim_fixed = klim_fixed
        self.kfun = kfun
        self.features = []
        self.formula_str = ''
        self.target_col = ''
        self.params_meta = []

        # call init_model in derived class
        if hasattr(self, 'init_model'):
            self.init_model()

    def fit(self, y, K, X=None, train_df=None):
        """
        Fit the model parameters to the provided soil and crop dataset.

        This method acts as a public wrapper that delegates the actual numerical 
        optimization to the subclass-specific `_fit` implementation.

        Parameters
        ----------
        y : array_like
            The target variable array, strictly representing log10(TF).
        K : array_like
            The primary predictor array, typically exchangeable potassium (Ex-K) 
            in linear scale (mol/kg).
        X : dict or None, optional
            A dictionary of additional predictive features (e.g., RIP, pH, CEC) 
            required by the specific model formulation.
        train_df : pandas.DataFrame or None, optional
            The original training DataFrame, passed for reference or for models 
            that require complex conditional logic based on raw data.

        Returns
        -------
        dict
            A dictionary containing the optimized model parameters (e.g., 'k1', 
            'k2', 'klim') and any relevant convergence metrics.
        """
        return self._fit(y, K, X, train_df=train_df)

    def predict(self, K, X=None, fit=None):
        """
        Estimate log10(TF) values using the optimized model parameters.

        This method applies the derived mathematical formulation to new or 
        held-out soil data, delegating the calculation to the subclass-specific 
        `_predict` implementation.

        Parameters
        ----------
        K : array_like
            The primary predictor array, typically exchangeable potassium (Ex-K) 
            in linear scale (mol/kg).
        X : dict or None, optional
            A dictionary of additional predictive features (e.g., RIP, pH, CEC) 
            matching those used during the fitting phase.
        fit : dict
            The dictionary of optimized parameters returned by the `fit` method.

        Returns
        -------
        array_like
            An array of predicted log10(TF) values.

        Raises
        ------
        ValueError
            If the `fit` dictionary containing model parameters is not provided.
        """
        if fit is None:
            raise ValueError("fit dictionary must be provided for prediction.")
        return self._predict(K, X, fit)

    def _fit(self, y, K, X, train_df=None):
        """
        Model-specific parameter optimization routine.

        Must be strictly implemented in any inheriting subclass to define 
        how the specific mathematical equation is fitted to the data 
        (e.g., via scipy.optimize.curve_fit).
        """
        raise NotImplementedError("Subclasses must implement _fit method.")

    def _predict(self, K, X, fit):
        """
        Model-specific prediction routine.

        Must be strictly implemented in any inheriting subclass to define 
        how the optimized parameters and input features are mathematically 
        combined to compute log10(TF).
        """
        raise NotImplementedError("Subclasses must implement _predict method.")
