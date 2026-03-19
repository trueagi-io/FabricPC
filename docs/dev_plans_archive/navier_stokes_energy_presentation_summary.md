# Navier-Stokes Energy Presentation Summary

## One-Line Summary

We extended FabricPC so a node can be trained not only to match a target state, but also to satisfy a simplified incompressible Navier-Stokes physics penalty.

## What Changed

- A new `NavierStokesEnergy` class was added to the energy system.
- The energy API was widened slightly so energy functions can receive optional runtime context.
- Older energy classes were updated only to match that expanded interface; their formulas did not change.
- The core predictive-coding graph and local-learning flow were left intact.
- The new energy adds fluid-physics residual penalties on top of the existing mismatch penalty.

## Why It Matters

Before this change, the repo mostly judged predictions by asking: “How far is the prediction from the inferred or target state?”

After this change, the repo can ask two questions at once:

- Is the prediction close to the target or inferred state?
- Does the prediction behave like a physically plausible fluid field?

That makes the system more useful for physics-informed learning, scientific modeling, and experiments where structure matters as much as accuracy.

## What Stayed The Same

- The graph-building API did not change in a fundamental way.
- Predictive-coding inference still updates latent states by minimizing node energy.
- Local parameter learning still happens through the existing node-level gradient path.
- `graph_net.py` did not need an algorithmic rewrite.

This is important: the improvement came from a richer energy definition, not from replacing the training architecture.

## Core Idea In Plain Language

The original repo treated tensors mostly as values to compare.

The new energy can treat a tensor as a small fluid field with:

- `u`: horizontal velocity
- `v`: vertical velocity
- `p`: pressure

It then penalizes fields that:

- disagree with the inferred state
- break incompressibility
- violate the simplified steady-state momentum balance

## The New Energy At A Glance

The total energy now has three parts:

- data alignment between `z_latent` and `z_mu`
- Navier-Stokes residual penalty on `z_latent`
- Navier-Stokes residual penalty on `z_mu`

Conceptually:

`total energy = mismatch penalty + latent physics penalty + prediction physics penalty`

This means:

- inference is pushed toward physically plausible latent states
- learning is pushed toward predictions that are themselves more physically plausible

## Key Fluid Terms For Non-Specialists

- Velocity: how the fluid moves.
- Pressure: what pushes the fluid from place to place.
- Viscosity: how strongly the fluid resists sharp motion changes.
- Divergence: whether fluid appears to be unrealistically appearing or disappearing.
- Incompressibility: the condition that divergence should stay near zero.
- Residual: a score for how badly a field violates the governing equation.

## Technical View In One Slide

The implementation assumes a 2D channels-last tensor of shape `(batch, H, W, 3)`.

Default channel layout:

- `u = 0`
- `v = 1`
- `p = 2`

The residual terms are based on:

- advection
- pressure gradients
- viscous diffusion
- divergence penalty

Spatial derivatives are approximated with periodic central differences using `jnp.roll`.

## Why `graph_net.py` Did Not Change

`graph_net.py` already knows how to:

- gather node-local inputs
- compute node-local parameter gradients
- apply the existing learning path

Because the new feature is expressed as a differentiable node-local energy, it automatically fits the existing design.

So the architecture gained new capability without taking on new training-system complexity.

## Improvements Over The Previous Implementation

- Adds physics-aware structure rather than only pointwise mismatch.
- Makes the repo more relevant for scientific and simulation-adjacent use cases.
- Preserves the existing predictive-coding training flow.
- Creates a cleaner extension point for future structured energy functions.

## Current Scope And Limits

This first version is intentionally narrow.

Supported:

- 2D fields only
- incompressible flow only
- steady-state residuals only
- periodic boundaries only

Not yet supported:

- 3D flow
- forcing terms
- time-dependent Navier-Stokes
- boundary masks or obstacle geometry
- full CFD solver behavior

## Validation Status

Validation completed in the repo-local test environment.

- Full repository test suite passed.
- Result: `87 passed, 1 warning`
- The warning was a JAX deprecation warning and did not affect correctness.

## Suggested Talk Track

Use this change as an example of a clean architectural extension:

1. FabricPC originally optimized mismatch-based predictive-coding energies.
2. We added a new energy that embeds fluid-physics structure.
3. We did this without changing the core graph-learning algorithm.
4. The result is a more expressive system for physics-informed experiments.

## Recommended Reading Order

- For non-technical audiences: `navier_stokes_energy_executive_brief.md`
- For equation-level detail: `navier_stokes_energy_math_companion.md`
- For the full implementation story: `navier_stokes_energy_implementation_notes.md`
