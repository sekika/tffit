"""
Common numerical utility module for radiocesium transfer models.

This module provides shared mathematical functions, focusing on robust and 
scientifically validated data transformations. It ensures that critical 
physicochemical parameters (such as transfer factors and potassium concentrations) 
strictly adhere to physically meaningful domains (e.g., strictly positive values) 
prior to logarithmic transformation, preventing silent numerical instability 
during optimization and cross-validation.
"""
import numpy as np


def log10_strict(x, name="value"):
    """
    Compute the base-10 logarithm with strict positivity enforcement.

    In the context of soil-to-plant transfer models, variables like TF, Ex-K, 
    and RIP represent physical quantities that must be strictly positive. 
    This function acts as a safeguard to prevent anomalous non-positive values 
    (e.g., from raw data errors or faulty interpolations) from producing invalid 
    numeric results (such as -inf or NaN) that would silently corrupt the fitting.

    Parameters
    ----------
    x : array_like
        The input numerical array (or scalar) to be log-transformed.
    name : str, optional
        A descriptive identifier for the variable being transformed. Used to 
        generate highly specific error messages. Default is "value".

    Returns
    -------
    numpy.ndarray
        The base-10 logarithm of the input array.

    Raises
    ------
    ValueError
        If any finite element in the input array is less than or equal to zero.
    """
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if np.any(m & (x <= 0.0)):
        bad = x[m & (x <= 0.0)]
        # show up to first few offending values
        preview = ", ".join([repr(float(v)) for v in bad[:5]])
        raise ValueError(
            f"log10 domain error in '{name}': found non-positive values (e.g., {preview})")
    return np.log10(x)


def keff_log10(K_ex, klim, kfun=None):
    """
    Compute the base-10 logarithm of the effective exchangeable potassium (K_eff).

    In semi-empirical radiocesium transfer models (e.g., Absalom et al.), plant 
    potassium uptake does not increase indefinitely with available soil potassium. 
    Instead, it saturates at a specific empirical threshold (k_lim). This function 
    calculates K_eff = min(K_ex, k_lim) and applies a strict log10 transformation, 
    ensuring that the physiological saturation concept is mathematically enforced.

    Parameters
    ----------
    K_ex : array_like
        The measured exchangeable potassium concentration in the soil (mol/kg).
    klim : float
        The empirical saturation threshold for potassium uptake (k_lim). 
        Must be a strictly positive finite number.
    kfun : callable, optional
        A custom mathematical function to dynamically compute K_eff, overriding 
        the default `min(K_ex, k_lim)` behavior. It must accept (K_ex, klim) as 
        arguments and return an array of strictly positive values.

    Returns
    -------
    numpy.ndarray
        The computed log10(K_eff) values ready for use in linear fitting equations.

    Raises
    ------
    ValueError
        If K_ex contains non-positive values, if klim is invalid (<= 0 or non-finite), 
        or if the custom `kfun` produces non-positive results prior to transformation.
    """
    K_ex = np.asarray(K_ex, float)

    # Strict check: K_ex must be > 0 where finite
    m = np.isfinite(K_ex)
    if np.any(m & (K_ex <= 0.0)):
        bad = K_ex[m & (K_ex <= 0.0)]
        preview = ", ".join([repr(float(v)) for v in bad[:5]])
        raise ValueError(f"K_ex must be > 0. Found (e.g., {preview})")

    if not np.isfinite(klim) or klim <= 0.0:
        raise ValueError(
            f"klim must be a positive finite number. Got {klim!r}")

    if kfun is not None:
        z = kfun(K_ex, klim)
        # strict check for output of kfun
        mz = np.isfinite(z)
        if np.any(mz & (z <= 0.0)):
            bad = z[mz & (z <= 0.0)]
            preview = ", ".join([repr(float(v)) for v in bad[:5]])
            raise ValueError(
                f"kfun output must be > 0 before log10. Found (e.g., {preview})")
        return log10_strict(z, name="Keff(kfun)")
    else:
        Keff = np.minimum(K_ex, klim)
        return log10_strict(Keff, name="Keff=min(K_ex,klim)")
