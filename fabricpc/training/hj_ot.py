import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

class HJOTState(NamedTuple):
    """State for the HJ-OT optimizer, tracks parameter momentum (velocity field)."""
    momentum: optax.Updates

def scale_by_hj_ot(
    viscosity: float = 0.9, 
    transport_cost: float = 1e-3, 
    dt: float = 1.0
) -> optax.GradientTransformation:
    """
    Scales updates by simulating Hamilton-Jacobi Optimal Transport.
    
    This interprets the weights as a fluid distribution flowing to minimize the 
    free energy, subject to a Wasserstein transport cost and kinematic viscosity.
    The gradient field \nabla F acts as the force, and the parameter momentum
    acts as the velocity field governed by a viscous Burgers-style equation.
    
    Args:
        viscosity: Decay factor analogous to classical momentum. 
                  (1.0 means no drag, 0.0 means immediate stop).
        transport_cost: Coefficient for the non-linear convective transport penalty.
        dt: Integration time step for the transport.
    """
    def init_fn(params):
        return HJOTState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))
        
    def update_fn(updates, state, params=None):
        def _update_momentum(g, m):
            # g is the raw gradient (force field from Navier Stokes prediction energy)
            # m is the current momentum (velocity field of the parameters)
            
            # 1. Viscous drag (classical classical momentum retention is 1 - drag)
            # We formulate it such that high viscosity retains momentum.
            drag = (1.0 - viscosity) * m 
            
            # 2. Convective transport cost (non-linear dissipation).
            # This represents the transport penalty in Wasserstein space.
            # Faster moving parameters face a quadratically increasing cost.
            convective = transport_cost * m * jnp.abs(m)
            
            # 3. Compute optimal velocity step
            new_m = m + dt * (g - drag - convective)
            
            return new_m
            
        new_momentum = jax.tree_util.tree_map(_update_momentum, updates, state.momentum)
        
        # The update returned to optax is the new momentum (velocity).
        # It will be scaled by the learning rate.
        return new_momentum, HJOTState(momentum=new_momentum)
        
    return optax.GradientTransformation(init_fn, update_fn)

def hj_ot_optimizer(
    learning_rate: float,
    viscosity: float = 0.9,
    transport_cost: float = 1e-4,
    dt: float = 1.0
) -> optax.GradientTransformation:
    """
    Creates an optimizer based on Hamilton-Jacobi Optimal Transport.
    
    Args:
        learning_rate: Step size scaling the final transport velocity.
        viscosity: Parameter for momentum retention (0 to 1).
        transport_cost: Coefficient for the non-linear transport penalty.
        dt: Internal integration timestep.
    """
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        scale_by_hj_ot(viscosity=viscosity, transport_cost=transport_cost, dt=dt),
        optax.scale_by_learning_rate(learning_rate)
    )
