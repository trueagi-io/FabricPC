# Navier-Stokes Energy Function Plan

## Summary

Create `NavierStokesEnergy` for 2D incompressible, steady-state flow on a single
channels-last tensor and rely on the existing `graph_net.py` local-gradient path
without changing its core algorithm.

The v1 energy targets tensors shaped `(batch, H, W, 3)` with channel order
`(u, v, p)`, uses periodic finite differences, and combines:
- a Gaussian data-alignment term between `z_latent` and `z_mu`
- a Navier-Stokes momentum residual on `z_latent`
- a Navier-Stokes momentum residual on `z_mu`
- an incompressibility penalty on both `z_latent` and `z_mu`

## Key Changes

### Energy API

- Change `energy(z_latent, z_mu, config=None)` to
  `energy(z_latent, z_mu, config=None, context=None)`
- Change `grad_latent(z_latent, z_mu, config=None)` to
  `grad_latent(z_latent, z_mu, config=None, context=None)`
- Update `compute_energy`, `compute_energy_gradient`,
  `get_energy_and_gradient`, and `NodeBase.energy_functional()` to thread
  `context`
- In v1, `context` contains `{"node_info": node_info}` only

### Navier-Stokes Energy

Add `NavierStokesEnergy(EnergyFunctional)` in `fabricpc/core/energy.py` and
export it from `fabricpc/core/__init__.py`.

Use config keys with these defaults:
- `viscosity`: required
- `dx=1.0`
- `dy=1.0`
- `data_weight=1.0`
- `latent_ns_weight=1.0`
- `prediction_ns_weight=1.0`
- `momentum_weight=1.0`
- `divergence_weight=1.0`
- `channel_map={"u": 0, "v": 1, "p": 2}`

Implement periodic central differences with `jnp.roll`:
- `d/dx`, `d/dy`
- Laplacian as `d2/dx2 + d2/dy2`

For a field `x = (u, v, p)`, define:
- `R_u(x) = u * du/dx + v * du/dy + dp/dx - viscosity * laplacian(u)`
- `R_v(x) = u * dv/dx + v * dv/dy + dp/dy - viscosity * laplacian(v)`
- `D(x) = du/dx + dv/dy`

Per-sample energy:
- `E_data = 0.5 * sum((z_latent - z_mu)^2)`
- `E_ns(x) = momentum_weight * 0.5 * sum(R_u(x)^2 + R_v(x)^2) + divergence_weight * 0.5 * sum(D(x)^2)`
- `E_total = data_weight * E_data + latent_ns_weight * E_ns(z_latent) + prediction_ns_weight * E_ns(z_mu)`

### Interaction With `graph_net.py`

Do not change the `graph_net.py` learning algorithm.

### Validation And Guardrails

Add explicit validation in `NavierStokesEnergy`:
- require rank-4 tensors `(batch, H, W, C)`
- require channels for `u`, `v`, and `p`
- reject `H < 3` or `W < 3`
- document that v1 is 2D, incompressible, steady-state, periodic-boundary only

Do not include forcing, time derivatives, boundary masks, separate
velocity/pressure nodes, or 3D support in v1.

## Test Plan

Add unit tests in `tests/test_energy.py` for:
- zero or near-zero PDE energy on constant divergence-free fields with zero pressure gradient
- positive divergence penalty on a clearly divergent field
- positive momentum residual on a field with nonzero advection or pressure mismatch
- `data_weight=0` and PDE weights disabled behaving as expected
- shape validation and missing-channel failures

Add integration tests for:
- graph construction with `NavierStokesEnergy()` on a fluid-output node
- inference running without shape/runtime errors
- `compute_local_weight_gradients()` returning correctly shaped gradients
- compatibility with existing energies to confirm the optional `context` change is non-breaking
