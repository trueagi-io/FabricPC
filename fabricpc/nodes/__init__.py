"""
Node types for JAX predictive coding networks.

This module provides:
- NodeBase: Abstract base class for all node types
- Built-in node implementations (LinearNode, TransformerBlockNode)
- Direct class imports for object-based graph construction
"""

from fabricpc.nodes.base import (
    SlotSpec,
    Slot,
    NodeBase,
    FlattenInputMixin,
    _register_node_class,
    _get_node_class_from_info,
)

# Import concrete node classes (also triggers _register_node_class calls)
from fabricpc.nodes.linear import LinearNode, LinearExplicitGrad
from fabricpc.nodes.transformer import TransformerBlockNode

# Convenience aliases matching the target API
Linear = LinearNode
TransformerBlock = TransformerBlockNode

__all__ = [
    # Base classes and mixins
    "SlotSpec",
    "Slot",
    "NodeBase",
    "FlattenInputMixin",
    # Built-in nodes (full names)
    "LinearNode",
    "LinearExplicitGrad",
    "TransformerBlockNode",
    # Convenience aliases
    "Linear",
    "TransformerBlock",
    # Internal dispatch helpers
    "_register_node_class",
    "_get_node_class_from_info",
]
