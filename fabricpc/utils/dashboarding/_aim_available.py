"""Lazy import helper for Aim dependency.

This module provides utilities for checking Aim availability and importing
it lazily. This allows the dashboarding package to be imported even when
Aim is not installed, with graceful degradation.
"""

from functools import wraps
from typing import Any, Callable, TypeVar

_aim_available: bool | None = None
_aim_module: Any = None

F = TypeVar("F", bound=Callable[..., Any])


def is_aim_available() -> bool:
    """Check if Aim is installed.

    Returns:
        True if Aim is available, False otherwise.
    """
    global _aim_available
    if _aim_available is None:
        try:
            import aim

            _aim_available = True
        except ImportError:
            _aim_available = False
    return _aim_available


def get_aim() -> Any:
    """Get the aim module, raising helpful error if not installed.

    Returns:
        The aim module.

    Raises:
        ImportError: If Aim is not installed.
    """
    global _aim_module
    if _aim_module is None:
        if not is_aim_available():
            raise ImportError(
                "Aim is not installed. Install with: pip install fabricpc[viz] "
                "or pip install aim"
            )
        import aim

        _aim_module = aim
    return _aim_module


def require_aim(func: F) -> F:
    """Decorator that checks Aim availability before calling function.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that raises ImportError if Aim is not available.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        get_aim()  # Raises if not available
        return func(*args, **kwargs)

    return wrapper  # type: ignore
