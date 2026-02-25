"""Edge and slot reference types for graph construction."""

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
