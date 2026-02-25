"""Thread-local graph namespace for hierarchical node naming."""

import threading

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
