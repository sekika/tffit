"""
Model registry module for dynamic instantiation of radiocesium transfer models.

This module implements a centralized registry pattern to manage the mathematical models
evaluated in the study. It provides a modular architecture that allows researchers to 
seamlessly register, retrieve, and test various semi-empirical (e.g., Absalom, ARC) 
and data-driven (e.g., SR1, SR2) models without altering the core execution pipeline.
"""

_model_registry = {}


def register_model(name):
    """
    Decorator to register a mathematical model class under a specified string identifier.

    This mechanism facilitates an extensible design, allowing users to effortlessly 
    incorporate custom soil-to-plant transfer models into the evaluation framework.

    Parameters
    ----------
    name : str
        The unique string identifier under which the model will be registered 
        (e.g., 'absalom', 'sr1').

    Returns
    -------
    callable
        The decorator function that registers and returns the class.
    """
    def decorator(cls):
        _model_registry[name] = cls
        return cls
    return decorator


def get_model(name):
    """
    Retrieve a registered transfer model class by its string identifier.

    Parameters
    ----------
    name : str
        The string identifier of the desired mathematical model.

    Returns
    -------
    cls : type
        The registered model class ready for instantiation.

    Raises
    ------
    KeyError
        If the requested model name is not found within the registry.
    """
    if name not in _model_registry:
        raise KeyError(f"Model '{name}' is not registered.")
    return _model_registry[name]


def list_models():
    """
    List all available string identifiers for the registered transfer models.

    Returns
    -------
    list of str
        A list containing the names of all currently registered models.
    """
    return list(_model_registry.keys())
