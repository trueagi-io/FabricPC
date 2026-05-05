"""Declarative topology primitives for predictive coding graphs.

This module hosts the small, dependency-free types used to describe graph
topology before assembly:

- ``Edge`` / ``SlotRef`` — directed connection from a source node to a target
  node's slot.
- ``GraphNamespace`` / ``_get_current_namespace`` — thread-local hierarchical
  naming context for nodes.

These primitives intentionally have no internal ``fabricpc`` dependencies so
they can be imported by any layer (including ``nodes.base``) without forming
cycles with the heavier graph-assembly code.
"""

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotRef:
    """Reference to a specific slot on a node."""

    node: object  # NodeBase instance
    slot: str


class Edge:
    """
    Edge connecting a source node to a target node's slot.

    If target is a NodeBase, defaults to the "in" slot.
    If target is a SlotRef, uses the specified slot.
    """

    def __init__(self, source, target):
        self.source = source
        if isinstance(target, SlotRef):
            self.target_node = target.node
            self.target_slot = target.slot
        else:
            self.target_node = target
            self.target_slot = "in"


_namespace_stack = threading.local()


def _get_current_namespace():
    """Get the current namespace prefix, or empty string if none."""
    stack = getattr(_namespace_stack, "stack", [])
    if not stack:
        return ""
    return "/".join(stack)


class GraphNamespace:
    """
    Context manager for hierarchical node naming.

    Nodes created inside a GraphNamespace block have their names
    prefixed with the namespace path.

    Example:
        with GraphNamespace("block1"):
            layer = Linear(shape=(64,), name="hidden")
            # layer.name == "block1/hidden"

            with GraphNamespace("sub"):
                inner = Linear(shape=(32,), name="deep")
                # inner.name == "block1/sub/deep"
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if not hasattr(_namespace_stack, "stack"):
            _namespace_stack.stack = []
        _namespace_stack.stack.append(self.name)
        return self

    def __exit__(self, *args):
        _namespace_stack.stack.pop()
