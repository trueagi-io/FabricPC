"""Fluid-field diagnostics and metrics for NHWC predictive coding tasks."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp


def _channel_map(channel_map: Dict[str, int] | None) -> Dict[str, int]:
    mapping = dict(channel_map or {"u": 0, "v": 1, "p": 2})
    required = {"u", "v", "p"}
    missing = required.difference(mapping)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"channel_map must define channels for {missing_str}")
    return mapping


def _validate_field(field: jnp.ndarray, channel_map: Dict[str, int]) -> None:
    if field.ndim != 4:
        raise ValueError(
            f"fluid metrics require rank-4 tensors (batch, H, W, C); got {field.shape}"
        )
    _, height, width, channels = field.shape
    if height < 3 or width < 3:
        raise ValueError(f"fluid metrics require H >= 3 and W >= 3; got {(height, width)}")
    if channels <= max(channel_map.values()):
        raise ValueError(
            "fluid metrics require channels for u, v, and p; "
            f"got C={channels} with channel_map={channel_map}"
        )


def periodic_central_diff_x(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Central difference along width for periodic NHWC scalar fields."""
    return (jnp.roll(field, -1, axis=2) - jnp.roll(field, 1, axis=2)) / (2.0 * dx)


def periodic_central_diff_y(field: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Central difference along height for periodic NHWC scalar fields."""
    return (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2.0 * dy)


def periodic_laplacian(field: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """2D periodic Laplacian for scalar fields stored as `(batch, H, W)`."""
    d2dx2 = (jnp.roll(field, -1, axis=2) - 2.0 * field + jnp.roll(field, 1, axis=2)) / (
        dx**2
    )
    d2dy2 = (jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1)) / (
        dy**2
    )
    return d2dx2 + d2dy2


def compute_navier_stokes_diagnostics(
    field: jnp.ndarray,
    viscosity: float,
    dx: float = 1.0,
    dy: float = 1.0,
    channel_map: Dict[str, int] | None = None,
) -> Dict[str, jnp.ndarray]:
    """Compute divergence and steady-state residual maps for `(u, v, p)` fields."""
    mapping = _channel_map(channel_map)
    _validate_field(field, mapping)

    u = field[..., mapping["u"]]
    v = field[..., mapping["v"]]
    p = field[..., mapping["p"]]

    dudx = periodic_central_diff_x(u, dx)
    dudy = periodic_central_diff_y(u, dy)
    dvdx = periodic_central_diff_x(v, dx)
    dvdy = periodic_central_diff_y(v, dy)
    dpdx = periodic_central_diff_x(p, dx)
    dpdy = periodic_central_diff_y(p, dy)

    lap_u = periodic_laplacian(u, dx, dy)
    lap_v = periodic_laplacian(v, dx, dy)

    residual_u = u * dudx + v * dudy + dpdx - viscosity * lap_u
    residual_v = u * dvdx + v * dvdy + dpdy - viscosity * lap_v
    divergence = dudx + dvdy

    return {
        "divergence": divergence,
        "residual_u": residual_u,
        "residual_v": residual_v,
        "momentum_residual": residual_u**2 + residual_v**2,
    }


def compute_fluid_metrics(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    viscosity: float,
    dx: float = 1.0,
    dy: float = 1.0,
    channel_map: Dict[str, int] | None = None,
    mask: jnp.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute reconstruction and physics metrics for fluid-field predictions.

    Returns mean-squared quantities over all batch and spatial dimensions.
    """
    mapping = _channel_map(channel_map)
    _validate_field(prediction, mapping)
    _validate_field(target, mapping)

    diff = prediction - target
    metrics = {
        "field_mse": float(jnp.mean(diff**2)),
        "velocity_mse": float(
            jnp.mean(
                (
                    prediction[..., [mapping["u"], mapping["v"]]]
                    - target[..., [mapping["u"], mapping["v"]]]
                )
                ** 2
            )
        ),
        "pressure_mse": float(
            jnp.mean((prediction[..., mapping["p"]] - target[..., mapping["p"]]) ** 2)
        ),
    }

    diagnostics = compute_navier_stokes_diagnostics(
        prediction,
        viscosity=viscosity,
        dx=dx,
        dy=dy,
        channel_map=mapping,
    )
    metrics["divergence_norm"] = float(jnp.mean(diagnostics["divergence"] ** 2))
    metrics["momentum_residual_norm"] = float(jnp.mean(diagnostics["momentum_residual"]))

    if mask is not None:
        if mask.shape != prediction.shape:
            raise ValueError(
                f"mask must match prediction shape; got mask={mask.shape}, prediction={prediction.shape}"
            )
        unobserved = 1.0 - mask
        denom = jnp.sum(unobserved)
        if float(denom) > 0.0:
            metrics["masked_region_mse"] = float(jnp.sum(diff**2 * unobserved) / denom)

    return metrics
