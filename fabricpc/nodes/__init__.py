"""
Node types for JAX predictive coding networks.

This module provides:
- NodeBase: Abstract base class for all node types
- Built-in node implementations (Linear, TransformerBlock)
- Direct class imports for object-based graph construction
"""

from fabricpc.nodes.base import (
    SlotSpec,
    Slot,
    NodeBase,
    FlattenInputMixin,
)

from fabricpc.nodes.linear import Linear, LinearExplicitGrad
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.transformer_v2 import EmbeddingNode

# Convenience aliases matching the target API
Linear = Linear
TransformerBlock = TransformerBlock

__all__ = [
    # Base classes and mixins
    "SlotSpec",
    "Slot",
    "NodeBase",
    "FlattenInputMixin",
    # Built-in nodes (full names)
    "Linear",
    "LinearExplicitGrad",
    "TransformerBlock",
    "IdentityNode",
    "EmbeddingNode",
]
