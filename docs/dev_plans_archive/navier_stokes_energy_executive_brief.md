# Navier-Stokes Energy Executive Brief

## Summary

We added a new way for FabricPC to judge whether a predicted state is not only close to a target, but also behaves like a physically valid fluid field. In practical terms, the software can now penalize solutions that break simple incompressible fluid rules, rather than only penalizing ordinary prediction error. This makes the system more useful for scientific and physics-informed machine learning work. Just as importantly, this was done without changing the main graph-learning engine of the repository.

For deeper technical detail, see:
- `navier_stokes_energy_implementation_notes.md`
- `navier_stokes_energy_math_companion.md`

## Why This Matters

Before this change, FabricPC mainly knew how to compare one tensor to another and ask, “How different are these?” That is useful for classification, reconstruction, and many standard machine learning tasks. But it is not enough when the thing being predicted is supposed to obey physical laws.

With this update, FabricPC can now say two things at once:

- “The prediction should match the inferred or target state.”
- “The prediction should also look like a valid fluid field.”

That is a meaningful step forward because many scientific problems are not just about matching data. They are also about respecting structure, constraints, and governing rules.

## What Changed In The Software

A new energy function called `NavierStokesEnergy` was added.

In FabricPC, an energy function is the rule that says what counts as “bad” or “costly” during inference and learning. The new energy function adds a fluid-physics penalty on top of the existing prediction-matching penalty.

The software now supports:

- checking whether a field is close to a prediction target
- checking whether a field violates incompressible fluid behavior
- using both of those checks together during predictive-coding inference and local learning

The implementation also slightly widened the internal energy API so richer energy functions can use runtime context when needed. Some existing energy functions were lightly adjusted so they all fit that same expanded interface. This was a compatibility and extensibility change, not a mathematical redesign, so the older energy types still behave the same from a modeling perspective.

## What Stayed The Same

The core graph training logic did **not** change.

This is important for explaining the change to collaborators: we did not rebuild the main predictive-coding engine. The same graph structure, local learning path, and node-level update flow remain in place.

That means:

- existing graph construction still works the same way
- local parameter learning still flows through the same machinery
- the new capability comes from a richer energy definition, not from replacing the overall training system

In short, the repository became more expressive without becoming structurally different.

## How This Improves The Previous Implementation

Previously, the repository supported several useful energy types such as squared error, cross-entropy, L1-style penalties, and KL divergence. Those are all good for measuring mismatch, but they treat tensors mostly as generic values.

The new implementation improves on that in three ways:

### 1. It introduces physics-aware structure

The model can now treat a tensor as a fluid field rather than just an array of numbers.

### 2. It improves scientific usefulness

This makes the repository more suitable for physics-informed experiments, especially where the output is expected to follow known governing rules.

### 3. It improves extensibility

The updated energy interface makes it easier to add future energy functions that depend on more than simple point-by-point differences.

## Key Fluid Concepts In Simple Terms

### Velocity

Velocity describes how the fluid is moving.

In this implementation, there are two velocity components:

- `u`: horizontal motion
- `v`: vertical motion

### Pressure

Pressure is the part of the field that pushes fluid from one place to another.

In this implementation, pressure is stored as `p`.

### Viscosity

Viscosity measures how much the fluid resists sharp changes in motion.

A higher viscosity means the fluid tends to smooth out more strongly.

### Divergence / Incompressibility

Divergence tells us whether fluid appears to be unrealistically piling up or disappearing at a point.

For incompressible flow, we want divergence to stay near zero. In plain language, that means the fluid should not magically appear or vanish.

### Residual / Physics Penalty

A residual is a measure of how badly a field violates the governing fluid equation.

In this implementation, the software computes a penalty when the velocity and pressure field do not behave the way a simple incompressible Navier-Stokes system would expect.

So the system is not only asking “Is this close?” It is also asking “Does this make physical sense?”

## Limitations

This is a deliberately narrow first version.

It supports:

- 2D fields only
- incompressible flow only
- steady-state residuals only
- periodic boundary assumptions only

It does **not** yet support:

- 3D flow
- forcing terms
- time-dependent Navier-Stokes evolution
- general boundary-condition masks
- a full CFD solver workflow

So this should be understood as a first physics-informed energy model, not a full fluid simulation framework.

## Where To Read More

For a deeper explanation of the implementation and design decisions:

- `navier_stokes_energy_implementation_notes.md`

For the equation-level, implementation-focused mathematical view:

- `navier_stokes_energy_math_companion.md`
