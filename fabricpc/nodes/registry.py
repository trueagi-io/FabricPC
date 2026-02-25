"""
DEPRECATED: Node registry has been removed. Import node classes directly.

This module is kept temporarily for backward compatibility.
"""


class NodeRegistrationError(Exception):
    """Deprecated. Node registration system has been removed."""

    pass


def register_node(node_type: str):
    """Deprecated no-op decorator."""

    def decorator(cls):
        return cls

    return decorator


def get_node_class(node_type: str):
    """Deprecated. Use node objects directly from GraphStructure."""
    raise ValueError(
        f"Registry removed. Import node classes directly instead of using get_node_class('{node_type}')."
    )


def list_node_types():
    """Deprecated."""
    return []


def unregister_node(node_type: str):
    """Deprecated no-op."""
    pass


def clear_registry():
    """Deprecated no-op."""
    pass


def validate_node_config(node_class, config):
    """Deprecated stub. Returns config as-is."""
    return config


def discover_external_nodes():
    """Deprecated no-op."""
    pass
