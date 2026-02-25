"""
DEPRECATED: Registry system has been removed. Import classes directly.

This module is kept temporarily for backward compatibility with code
that imports RegistrationError. It will be fully removed once all
dependent code is updated.
"""


class RegistrationError(Exception):
    """Deprecated. Registration system has been removed."""

    pass


def validate_config_schema(attr_value, type_name, error_class=None):
    """Deprecated stub. No-op."""
    pass


def validate_default_energy_config(attr_value, type_name, error_class=None):
    """Deprecated stub. No-op."""
    pass


class Registry:
    """Deprecated. Registry system has been removed."""

    def __init__(self, **kwargs):
        self._registry = {}

    def set_error_class(self, error_class):
        pass

    def register(self, type_name):
        def decorator(cls):
            return cls

        return decorator

    def get(self, type_name):
        raise ValueError(f"Registry system removed. Import classes directly.")

    def list_types(self):
        return []

    def unregister(self, type_name):
        pass

    def clear(self):
        pass

    def discover_external(self):
        pass

    def __contains__(self, type_name):
        return False

    def __len__(self):
        return 0
