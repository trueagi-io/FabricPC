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
from fabricpc.nodes.transformer_v2 import (
    EmbeddingNode,
    MhaResidualNode,
    LnMlp1Node,
    Mlp2ResidualNode,
    VocabProjectionNode,
)
from fabricpc.nodes.storkey_hopfield import StorkeyHopfield
from fabricpc.nodes.skip_connection import SkipConnection
from fabricpc.nodes.linear_residual import LinearResidual
from fabricpc.nodes.convolutional import Conv1DNode, Conv2DNode, Conv3DNode

# Convenience aliases matching the target API
Linear = Linear
TransformerBlock = TransformerBlock
Identity = IdentityNode  # Standard alias for IdentityNode

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
    "Identity",
    "EmbeddingNode",
    "MhaResidualNode",
    "LnMlp1Node",
    "Mlp2ResidualNode",
    "VocabProjectionNode",
    "StorkeyHopfield",
    "SkipConnection",
    "LinearResidual",
    "Conv1DNode",
    "Conv2DNode",
    "Conv3DNode",
]
