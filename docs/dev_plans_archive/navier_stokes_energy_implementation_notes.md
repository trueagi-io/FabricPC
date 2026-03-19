# Navier-Stokes Energy Implementation Notes

## Overview

This document describes the changes made to FabricPC to introduce a Navier-Stokes-based energy functional, the improvements relative to the previous implementation, and how the Navier-Stokes terms map onto the predictive-coding concepts already used across the repository.

The implementation was added without changing the core graph-learning algorithm in `graph_net.py`. Instead, it extends the energy abstraction so that physics-informed residuals can participate in the same inference and local-learning pipeline already used by the rest of the library.

## What Changed

### 1. The energy interface was extended

The original `EnergyFunctional` abstraction only accepted:

- `z_latent`
- `z_mu`
- `config`

The new interface adds an optional `context` argument:

```python
energy(z_latent, z_mu, config=None, context=None)
grad_latent(z_latent, z_mu, config=None, context=None)
```

This is a forward-compatible change. It keeps all existing energy functions usable while allowing richer energies to access runtime metadata such as `node_info`.

The older built-in energies were also updated to accept this optional `context` argument. That was required because node-level energy evaluation now passes `context` uniformly from `NodeBase.energy_functional()`, and the convenience wrappers thread that same argument through the shared runtime path. This preserves polymorphic dispatch across all energy classes.

Importantly, those older energies did not change mathematically. Gaussian, Bernoulli, CrossEntropy, Laplacian, Huber, and KLDivergence still compute the same energies and latent gradients as before. Their changes were limited to method-signature compatibility and minor internal cleanup such as deduplicating repeated non-batch summation logic into a helper function.

This change was threaded through:

- `fabricpc/core/energy.py`
- `fabricpc/nodes/base.py`
- helper functions such as `compute_energy`, `compute_energy_gradient`, and `get_energy_and_gradient`

### 2. A new `NavierStokesEnergy` was added

A new energy class, `NavierStokesEnergy`, was added to `fabricpc/core/energy.py` and exported through `fabricpc/core/__init__.py`.

It targets 2D incompressible, steady-state fluid fields represented as a single NHWC tensor:

- shape: `(batch, H, W, C)`
- default channel interpretation: `u=0`, `v=1`, `p=2`

The implementation assumes:

- 2D only
- incompressible flow
- steady-state residuals
- periodic boundaries
- finite differences using `jnp.roll`

### 3. Node-level energy evaluation now passes runtime context

`NodeBase.energy_functional()` now passes:

```python
context = {"node_info": node_info}
```

into energy computation.

This is important because the original repo treated energy functions as purely tensor-level penalties. The new interface makes it possible to build future energy functions that depend on additional node metadata without forcing a redesign of the graph training path.

### 4. New tests were added

`tests/test_energy.py` now includes:

- zero-energy behavior for constant divergence-free fields
- positive divergence penalty checks
- positive momentum-residual checks
- validation failures for invalid field shapes/channels
- an integration test that runs inference and local weight gradients with `NavierStokesEnergy`

The full repository suite was also run successfully after the change:

- `87 passed, 1 warning`

## What Did *Not* Change

### `graph_net.py` did not change structurally

This is deliberate and important.

`fabricpc/graph/graph_net.py` already had the correct role:

- gather the final inferred node states
- ask each node to compute local parameter gradients via `forward_learning()`
- return those local gradients for optimizer application

Because energy enters the learning loop through node-local forward computation, the Navier-Stokes energy could be integrated without rewriting the graph-level optimization logic.

This means the extension is architectural, not invasive.

## Improvements Over the Previous Implementation

## A. Before: the repo only supported discrepancy-style energies

Previously, the built-in energies were all direct mismatch penalties between `z_latent` and `z_mu`, for example:

- Gaussian: squared error
- Bernoulli: binary cross-entropy
- CrossEntropy: categorical log loss
- Laplacian: L1 distance
- Huber: robust smooth L1
- KLDivergence: distribution matching

These are all forms of:

- pointwise discrepancy
- distribution discrepancy
- robust discrepancy

They do not encode spatial physics.

### Improvement

`NavierStokesEnergy` introduces a **structured field constraint** instead of only a direct prediction-target mismatch. It allows the model to be penalized not only for being different from a target field, but also for violating the governing PDE of incompressible flow.

## B. Before: the energy API could not naturally express PDE residuals

The original energy API was enough for static mismatch penalties, but PDE residuals need more structure:

- field interpretation
- channel semantics
- grid spacing
- derivative operators
- potential future access to node metadata

### Improvement

The optional `context` parameter is a minimal, compatible extension that keeps the original API usable while opening a path for richer physics-informed or geometry-aware energies.

## C. Before: latent inference and weight learning only saw generic energy penalties

Original predictive coding in this repo minimized energy based on error between inferred states and predicted states. That worked well for standard supervised or reconstruction-style problems, but it did not let the model treat internal states as fluid fields obeying a PDE.

### Improvement

The new energy contributes to both:

- **latent inference** through `grad_latent()`
- **weight learning** through local differentiation of node energy with respect to parameters

This matters because the PDE is now part of both:

- the inferred hidden/target state geometry
- the learned predictor geometry

## D. Before: no spatial derivative infrastructure existed in the energy layer

The original energy layer had no finite-difference operators or field validation.

### Improvement

The new implementation adds:

- periodic central difference in `x`
- periodic central difference in `y`
- Laplacian via second derivatives
- shape validation for `(batch, H, W, C)`
- semantic channel validation for `u`, `v`, and `p`

This is the first energy in the repo that treats tensors as spatial fields instead of generic arrays.

## Navier-Stokes Terms in This Implementation

The implementation uses the steady-state, incompressible 2D Navier-Stokes residual in a penalty form.

### Fluid channels

A single tensor `x` is interpreted as:

- `u`: horizontal velocity
- `v`: vertical velocity
- `p`: pressure

These are taken from channels of the NHWC tensor.

### Grid spacing

- `dx`: spacing in the horizontal direction
- `dy`: spacing in the vertical direction

These scale the finite-difference derivative operators.

### First derivatives

The implementation computes:

- `du/dx`
- `du/dy`
- `dv/dx`
- `dv/dy`
- `dp/dx`
- `dp/dy`

using periodic central differences.

### Laplacian

The Laplacian is computed as:

- `laplacian(u) = d2u/dx2 + d2u/dy2`
- `laplacian(v) = d2v/dx2 + d2v/dy2`

This is the diffusion term in the PDE.

### Viscosity

`viscosity` is the coefficient multiplying the Laplacian term.

In physical terms, it controls the smoothing or diffusive strength of the fluid. In this energy implementation, higher viscosity increases the contribution of diffusion to the residual.

### Momentum residuals

The code defines:

```text
R_u = u * du/dx + v * du/dy + dp/dx - viscosity * laplacian(u)
R_v = u * dv/dx + v * dv/dy + dp/dy - viscosity * laplacian(v)
```

Interpretation:

- `u * du/dx + v * du/dy` is advection of the horizontal velocity
- `u * dv/dx + v * dv/dy` is advection of the vertical velocity
- `dp/dx`, `dp/dy` are pressure-gradient terms
- `viscosity * laplacian(...)` is diffusion

These terms are not used to simulate time evolution directly. They are used as **residual penalties**: if the field violates the momentum equations, energy increases.

### Divergence term

The implementation also computes:

```text
D = du/dx + dv/dy
```

This is the incompressibility condition.

If `D != 0`, the field is not divergence-free, so the energy increases.

## Composite Energy Used in the Code

The implemented energy is:

```text
E_total = E_data
        + latent_ns_weight * E_ns(z_latent)
        + prediction_ns_weight * E_ns(z_mu)
```

with:

```text
E_data = 0.5 * data_weight * sum((z_latent - z_mu)^2)
```

and:

```text
E_ns(x) = 0.5 * momentum_weight * sum(R_u(x)^2 + R_v(x)^2)
        + 0.5 * divergence_weight * sum(D(x)^2)
```

This is a hybrid energy with two roles:

- a **predictive coding alignment term** between latent and predicted state
- a **physics residual term** enforcing Navier-Stokes consistency

## How the Navier-Stokes Terms Map to Predictive Coding Concepts

## 1. `z_latent` in the original repo

In the original FabricPC design, `z_latent` is the inferred state of a node.

It is updated during inference by descending the latent gradient of energy.

### In the new implementation

`z_latent` can now be interpreted as an inferred fluid field:

- inferred velocity field
- inferred pressure field
- inferred physically plausible state

Adding Navier-Stokes energy means inference is no longer only trying to make `z_latent` close to `z_mu`; it is also trying to make `z_latent` satisfy fluid constraints.

## 2. `z_mu` in the original repo

In the original repo, `z_mu` is the node’s prediction from its incoming inputs and parameters.

### In the new implementation

`z_mu` is now the predicted fluid field.

The PDE penalty on `z_mu` means parameter learning is no longer driven only by target mismatch. The predicted field itself is judged by whether it looks like a valid incompressible Navier-Stokes field.

## 3. Original energy concept

Originally, energy measured error or mismatch.

Examples:

- squared difference
- cross-entropy
- KL divergence

### In the new implementation

Energy becomes a combination of:

- mismatch energy
- physics-consistency energy

This is a conceptual shift from “make prediction match latent” to:

- make prediction match latent
- make both states physically valid

## 4. Original `latent_grad`

Originally, `latent_grad` came from differentiating error-based energies with respect to `z_latent`.

### In the new implementation

`grad_latent()` for `NavierStokesEnergy` is computed with `jax.grad` over the full latent-side energy. That is an important change in style:

- old energies mostly had simple closed-form gradients
- Navier-Stokes residuals are more complex and spatially coupled
- automatic differentiation is used instead of manually deriving every latent-gradient term

This makes the implementation safer and easier to extend, especially for future PDE energies.

## Why `graph_net.py` Still Works

`graph_net.py` computes local parameter gradients by calling each node’s `forward_learning()` after inference converges.

That means the graph layer only cares that:

- the node can compute a forward energy
- JAX can differentiate that energy with respect to node parameters

Because `NavierStokesEnergy` is embedded inside the node-local energy computation, it automatically participates in:

- local predictive-coding weight updates
- optimizer updates after local gradients are collected

So the graph-level training model did not need to become “fluid-specific.” The energy model became richer while the graph algorithm stayed generic.

## Practical Interpretation of the New Behavior

Before this change, a node using Gaussian energy said:

- “Make `z_latent` close to `z_mu`.”

After this change, a node using Navier-Stokes energy says:

- “Make `z_latent` close to `z_mu`.”
- “Make the inferred field satisfy incompressible Navier-Stokes structure.”
- “Make the predicted field satisfy incompressible Navier-Stokes structure.”

That is the key improvement.

## Limitations of the Current Navier-Stokes Implementation

This implementation is intentionally narrow:

- 2D only
- steady-state only
- no explicit forcing term
- periodic boundaries only
- no separate velocity/pressure graph nodes
- no time derivative term
- no boundary masks or obstacle geometry
- no 3D support

So this is not yet a general CFD solver. It is a physics-informed energy penalty for fluid-like fields inside the existing predictive-coding framework.

## Test Coverage Performed

The implementation was validated with:

- targeted Navier-Stokes energy tests
- the full `tests/test_energy.py` file
- broader repository regression tests
- the full repository test suite

Final repository result:

```text
87 passed, 1 warning
```

The warning was a JAX deprecation warning and did not affect correctness.

## Summary of the Main Architectural Gain

The most important architectural improvement is not just the addition of one new energy class. It is that FabricPC now supports a new category of energy:

- not only label or reconstruction mismatch
- but also field-structured, physics-based residual penalties

That makes the repository meaningfully more expressive for scientific machine learning, physics-informed predictive coding, and future PDE-constrained latent inference.
