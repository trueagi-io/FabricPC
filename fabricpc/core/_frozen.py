"""Shared immutability mixin for stateless config value objects.

``ActivationBase``, ``EnergyFunctional``, and ``InitializerBase`` place a single
default instance directly in node ``__init__`` signatures. A signature default
is evaluated once at import and shared by every defaulted call, so that instance
is only safe if it cannot be mutated. ``FrozenConfig`` is the single source of
that freeze for all three families.
"""

import types


class FrozenConfig:
    """Freezes an instance after construction.

    Once ``__init__`` has run, attributes cannot be set or deleted, and
    ``config`` is a read-only mapping whose keys cannot be added, removed, or
    reassigned. The freeze is shallow: a mutable value stored under a key is not
    itself deep-frozen, so construct only with immutable scalar config values.
    """

    def __init__(self, **config):
        # object.__setattr__ bypasses the freeze below to set these two fields
        # once; every later assignment goes through __setattr__ and is rejected.
        object.__setattr__(self, "config", types.MappingProxyType(config))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")
