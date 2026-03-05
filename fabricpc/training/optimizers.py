"""
Optimizer utilities using Optax.
"""

from fabricpc.training.natural_gradients import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)

__all__ = [
    "scale_by_natural_gradient_diag",
    "scale_by_natural_gradient_layerwise",
]
