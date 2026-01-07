# import methods from submodules
from fabricpc.utils.helpers import (
    update_node_in_state,
)


# Lazy import for dashboarding to avoid requiring aim at import time
def __getattr__(name: str):
    if name == "dashboarding":
        from fabricpc.utils import dashboarding

        return dashboarding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "update_node_in_state",
    "dashboarding",
]
