import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

class HJOTState(NamedTuple):
    """State for the HJ-OT optimizer, tracks parameter momentum (velocity field) and step count."""
    momentum: optax.Updates
    count: jnp.ndarray

def scale_by_hj_ot(
    viscosity: float = 0.9, 
    transport_cost: float = 1e-3, 
    dt: float = 1.0,
    mass: float = 1.0,
    viscosity_decay: float = 1.0,  # Multiplier per step (1.0 = no decay)
    viscosity_min: float = 0.1     # Minimum floor for viscosity
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
        mass: Inertia coefficient. Higher mass makes parameters harder to accelerate.
        viscosity_decay: Exponential decay rate for viscosity per step.
        viscosity_min: The lowest allowed value for viscosity during decay.
    """
    def init_fn(params):
        return HJOTState(
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            count=jnp.zeros([], jnp.int32)
        )
        
    def update_fn(updates, state, params=None):
        # Calculate current dynamic viscosity
        # v_t = max(v_min, v_0 * (decay^t))
        current_visc = jnp.maximum(
            viscosity_min, 
            viscosity * jnp.power(viscosity_decay, state.count.astype(jnp.float32))
        )

        def _update_momentum(g, m):
            # 1. Viscous drag using dynamic viscosity
            drag = (1.0 - current_visc) * m 
            
            # 2. Convective transport cost (non-linear dissipation).
            convective = transport_cost * m * jnp.abs(m)
            
            # 3. Compute optimal velocity step with Inertia (mass)
            new_m = m + (dt / mass) * (g - drag - convective)
            
            return new_m
            
        new_momentum = jax.tree_util.tree_map(_update_momentum, updates, state.momentum)
        
        return new_momentum, HJOTState(momentum=new_momentum, count=state.count + 1)
        
    return optax.GradientTransformation(init_fn, update_fn)

def hj_ot_optimizer(
    learning_rate: float,
    viscosity: float = 0.9,
    transport_cost: float = 1e-4,
    dt: float = 1.0,
    mass: float = 1.0,
    viscosity_decay: float = 1.0,
    viscosity_min: float = 0.1
) -> optax.GradientTransformation:
    """
    Creates an optimizer based on Hamilton-Jacobi Optimal Transport.
    
    Args:
        learning_rate: Step size scaling the final transport velocity.
        viscosity: Parameter for momentum retention (0 to 1).
        transport_cost: Coefficient for the non-linear transport penalty.
        dt: Internal integration timestep.
        mass: Inertia coefficient (default: 1.0).
        viscosity_decay: Rate of viscosity decay (default: 1.0).
        viscosity_min: Minimum viscosity floor.
    """
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        scale_by_hj_ot(
            viscosity=viscosity, 
            transport_cost=transport_cost, 
            dt=dt, 
            mass=mass,
            viscosity_decay=viscosity_decay,
            viscosity_min=viscosity_min
        ),
        optax.scale_by_learning_rate(learning_rate)
    )
