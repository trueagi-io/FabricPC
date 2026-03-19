# Navier-Stokes Energy Math Companion

## Purpose

This note explains the mathematics of the current `NavierStokesEnergy` implementation in FabricPC and ties the equations directly to the repository’s predictive-coding concepts.

It is meant to complement, not replace, the longer technical narrative in `navier_stokes_energy_implementation_notes.md`.

For deeper prose explanation and broader architectural context, see:
- `navier_stokes_energy_implementation_notes.md`

## Notation

We use the following symbols, matching the implementation:

- `z_latent`: the inferred node state after or during predictive-coding inference
- `z_mu`: the node’s predicted state produced by its forward computation
- `u`: horizontal velocity component
- `v`: vertical velocity component
- `p`: pressure
- `dx`: grid spacing in the horizontal direction
- `dy`: grid spacing in the vertical direction
- `nu`: viscosity coefficient, called `viscosity` in code
- `div`: divergence of the velocity field
- `Delta`: Laplacian operator

Tensor convention in the implementation:

- field shape: `(batch, H, W, C)`
- default channel map: `u=0`, `v=1`, `p=2`

So each field tensor is interpreted as a 2D fluid state in channels-last format.

## Original Repository Energy Model

Before this change, the repository’s built-in energy functions were discrepancy-based. Their role was to compare `z_latent` and `z_mu` directly.

Examples include:

- Gaussian:
  \[
  E(z_{latent}, z_{mu}) = \frac{\alpha}{2} \| z_{latent} - z_{mu} \|^2
  \]
- Cross-entropy
- KL divergence
- L1 / Huber variants

Conceptually, these energies say:

- penalize mismatch between inferred state and predicted state
- optionally use a different mismatch geometry or probabilistic interpretation

They do **not** say:

- penalize violation of a spatial PDE
- interpret the tensor as a velocity-pressure field
- enforce incompressibility or momentum balance

That is the core conceptual shift introduced by `NavierStokesEnergy`.

## New Composite Energy

The implemented energy combines two kinds of penalties:

1. predictive-coding alignment between `z_latent` and `z_mu`
2. physics residual penalties on fluid fields

### Data-alignment term

The data or alignment term is:

\[
E_{data} = \frac{1}{2} \cdot data\_weight \cdot \sum (z_{latent} - z_{mu})^2
\]

This is still the familiar squared discrepancy term from the original style of energy function.

### Differential operators

The implementation uses periodic central differences:

\[
\frac{\partial f}{\partial x} \approx \frac{f(x+dx) - f(x-dx)}{2dx}
\]

\[
\frac{\partial f}{\partial y} \approx \frac{f(y+dy) - f(y-dy)}{2dy}
\]

and the Laplacian:

\[
\Delta f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
\]

In code, these are built with `jnp.roll`, which imposes periodic boundaries.

## Navier-Stokes Residual Terms

For a field \(x = (u, v, p)\), the implementation defines the steady-state incompressible momentum residuals:

\[
R_u(x) = u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \Delta u
\]

\[
R_v(x) = u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \Delta v
\]

and the incompressibility residual:

\[
D(x) = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
\]

where:

- the advection terms are the nonlinear transport terms
- the pressure-gradient terms couple pressure to motion
- the viscosity term smooths the velocity field
- the divergence term enforces incompressibility

## Physics Energy

The code converts those residuals into a scalar per-sample energy:

\[
E_{ns}(x) = \frac{1}{2} \cdot momentum\_weight \cdot \sum \left(R_u(x)^2 + R_v(x)^2\right)
+ \frac{1}{2} \cdot divergence\_weight \cdot \sum D(x)^2
\]

This is a penalty formulation, not a direct PDE solver.

Interpretation:

- if the field perfectly satisfies the chosen steady-state incompressible equations, the residual energy approaches zero
- if the field violates them, the residual energy grows

## Full Implemented Energy

The total implemented energy is:

\[
E_{total}(z_{latent}, z_{mu}) = E_{data}
+ latent\_ns\_weight \cdot E_{ns}(z_{latent})
+ prediction\_ns\_weight \cdot E_{ns}(z_{mu})
\]

Expanding that:

\[
E_{total} = \frac{1}{2} \cdot data\_weight \cdot \sum (z_{latent} - z_{mu})^2
+ latent\_ns\_weight \cdot E_{ns}(z_{latent})
+ prediction\_ns\_weight \cdot E_{ns}(z_{mu})
\]

This is the key implementation idea: both the inferred state and the predicted state are judged by fluid-physics structure.

## How This Maps To The Code

### `fabricpc/core/energy.py`

This file now does three important things:

1. extends the base `EnergyFunctional` API to accept `context`
2. defines spatial derivative helpers and field validation
3. defines `NavierStokesEnergy.energy()` and `NavierStokesEnergy.grad_latent()`

The legacy energies remain discrepancy-based and mathematically unchanged. Their signatures were widened only so the runtime can call every energy through the same `context`-aware contract. That is why the API change appears across all energy classes even though only `NavierStokesEnergy` introduces new PDE residual terms.

The fluid residual pieces in code correspond directly to:

- `_periodic_central_diff_x` -> \(\partial / \partial x\)
- `_periodic_central_diff_y` -> \(\partial / \partial y\)
- `_periodic_laplacian` -> \(\Delta\)
- `_navier_stokes_residual_energy` -> \(E_{ns}(x)\)

### `fabricpc/nodes/base.py`

`NodeBase.energy_functional()` now passes:

```python
context = {"node_info": node_info}
```

and then evaluates:

- `energy_cls.energy(...)`
- `energy_cls.grad_latent(...)`

That is where the new energy enters the node-level predictive-coding loop.

### `graph_net.py`

`graph_net.py` remains structurally unchanged.

This is mathematically consistent because the graph layer already assumes:

- local node energy exists
- local weight gradients can be obtained by differentiating node energy with respect to parameters

Since `NavierStokesEnergy` modifies the node-local scalar energy, it automatically participates in the same local gradient machinery.

## How This Changes Inference And Learning

## 1. Latent-side effect

The latent update in predictive coding depends on:

\[
\frac{\partial E}{\partial z_{latent}}
\]

For `NavierStokesEnergy`, the implementation computes `grad_latent()` by automatic differentiation over the total latent-side energy.

That means latent inference is now shaped by:

- the standard mismatch term `E_data`
- the latent fluid residual term `E_ns(z_latent)`

So inference is no longer only “make `z_latent` match `z_mu`.” It is also “make `z_latent` physically plausible as a fluid field.”

## 2. Prediction-side effect on learning

The term:

\[
prediction\_ns\_weight \cdot E_{ns}(z_{mu})
\]

has no direct latent-gradient effect with respect to `z_latent`, because it depends on `z_mu`.

But it **does** affect weight learning because `z_mu` depends on node parameters.

So during node-local differentiation with respect to parameters, the model learns not only to reduce data mismatch, but also to reduce physics inconsistency in the prediction itself.

## 3. Why `graph_net.py` did not need structural changes

Let node parameters be \(\theta_i\). The graph-level local-learning flow already computes node-local parameter gradients of the form:

\[
\frac{\partial E_i}{\partial \theta_i}
\]

As long as the node energy is differentiable and expressed through the forward pass, the graph-level algorithm does not need to know whether the energy came from:

- squared error
- cross-entropy
- KL divergence
- PDE residual penalties

This is why the extension is mathematically local and architecturally clean.

## Contrast With Original Repository Concepts

### Mismatch penalties vs PDE residual penalties

Original repo:

- energy measures whether prediction and latent state disagree

New implementation:

- energy measures disagreement **and** fluid-law violation

### Generic tensors vs spatial fields

Original repo:

- tensors are mostly generic activations or distributions

New implementation:

- tensors are explicitly interpreted as structured 2D fields with velocity and pressure channels

### Closed-form simple gradients vs autodiff over coupled fields

Original repo:

- many energies have simple analytic latent gradients

New implementation:

- the fluid residual introduces nonlinear and spatially coupled terms
- the latent gradient is computed with `jax.grad`

This is an important mathematical difference because PDE residuals couple neighboring points through derivatives and Laplacians.

## Assumptions And Limitations

This mathematical description matches the actual current implementation only under these assumptions:

- 2D only
- incompressible flow only
- steady-state residuals only
- periodic boundaries only
- no forcing term
- no time derivative term
- no 3D extension
- no explicit obstacle or boundary masks

So the current implementation is best understood as a **physics-informed residual energy** for predictive coding, not a general-purpose Navier-Stokes simulator.

## Further Detail

For broader implementation and repository-level explanation, see:

- `navier_stokes_energy_implementation_notes.md`

For the shortest, non-technical explanation for mixed audiences, see:

- `navier_stokes_energy_executive_brief.md`
