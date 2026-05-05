"""Utility functions and data loading for FabricPC."""

import importlib

from fabricpc.utils.helpers import layernorm

# Submodules
from fabricpc.utils import data


# Lazy import for dashboarding to avoid requiring aim at import time
def __getattr__(name: str):
    if name == "dashboarding":
        return importlib.import_module("fabricpc.utils.dashboarding")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Helpers
    "layernorm",
    # Submodules
    "data",
    "dashboarding",
]
