# tests/model/test_registry.py
import pytest
# Import the public functions from the registry module
from src.model.registry import list_models, get_model
from src.model.base import BaseModel


def test_expected_models_are_registered():
    """
    Ensure all required models are correctly registered and accessible.
    """
    available_models = list_models()
    expected_models = ['absalom', 'k', 'kr',
                       'krc', 'krp', 'krcs', 'sr1', 'sr2']

    for model_name in expected_models:
        assert model_name in available_models, f"Model '{model_name}' should be present in the registry."


def test_model_instances_inherit_base_model():
    """
    Check if all registered model classes correctly inherit from the BaseModel interface.
    """
    for model_name in list_models():
        model_class = get_model(model_name)
        instance = model_class()
        assert isinstance(
            instance, BaseModel), f"Model '{model_name}' must inherit from BaseModel."


def test_models_have_required_metadata():
    """
    Verify that each model correctly initializes mandatory metadata attributes.

    These attributes are essential for downstream analysis, visualization, 
    and reporting in the associated publication.
    """
    for model_name in list_models():
        model_class = get_model(model_name)
        model = model_class()
        # Initialize to populate metadata
        model.init_model()

        # Mandatory features list for input validation
        assert hasattr(
            model, 'features'), f"Model '{model_name}' is missing the 'features' attribute."
        assert isinstance(
            model.features, list), f"'features' in '{model_name}' should be a list."

        # Mathematical formula string for documentation/display
        assert hasattr(
            model, 'formula_str'), f"Model '{model_name}' is missing 'formula_str'."

        # Target column specification (typically log10_TF)
        assert hasattr(
            model, 'target_col'), f"Model '{model_name}' is missing 'target_col'."

        # Metadata for parameters (keys, labels, and descriptions)
        assert hasattr(
            model, 'params_meta'), f"Model '{model_name}' is missing 'params_meta'."
        assert len(
            model.params_meta) > 0, f"Model '{model_name}' must define at least one parameter."


def test_ar_parameter_consistency():
    """
    Perform a targeted check on the KR model metadata to ensure 
    parameter keys match the expected mechanistic formulation.
    """
    ar_class = get_model('kr')
    ar_model = ar_class()
    ar_model.init_model()

    keys = [p['key'] for p in ar_model.params_meta]
    # The AR model (Absalom + RIP) typically expects k1, k2, and k3
    for expected_key in ['k1', 'k2', 'k3']:
        assert expected_key in keys, f"Parameter key '{expected_key}' missing in AR model metadata."
