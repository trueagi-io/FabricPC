"""Shared immutability mixin for stateless config value objects.

``ActivationBase``, ``EnergyFunctional``, and ``InitializerBase`` place a single
default instance directly in node ``__init__`` signatures. A signature default
is evaluated once at import and shared by every defaulted call, so that instance
is only safe if it cannot be mutated. ``FrozenConfig`` is the single source of
that freeze for all three families.
"""

import types

# Immutable scalar types accepted as config values. Tuples are accepted too,
# recursively. A future structured value (e.g. a per-channel alpha) is a tuple,
# not a list or an array.
_IMMUTABLE_SCALARS = (bool, int, float, str, bytes, type(None))


def _validate_immutable(value, key, owner):
    """Reject any config value that is not an immutable scalar or a tuple of them."""
    if isinstance(value, _IMMUTABLE_SCALARS):
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_immutable(item, key, owner)
        return
    raise TypeError(
        f"{owner} config value {key!r} must be an immutable scalar "
        f"(int, float, str, bool, bytes, None) or a tuple of those; "
        f"got {type(value).__name__}"
    )


class FrozenConfig:
    """Freezes an instance after construction.

    Once ``__init__`` has run, attributes cannot be set or deleted, and
    ``config`` is a read-only mapping whose keys cannot be added, removed, or
    reassigned. Config values are validated at construction: only immutable
    scalars (int, float, str, bool, bytes, None) and tuples of those are
    accepted, so the whole object is immutable, not just its top level.
    """

    def __init__(self, **config):
        for key, value in config.items():
            _validate_immutable(value, key, type(self).__name__)
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
