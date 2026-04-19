"""
Elastic Weight Consolidation (EWC) for Continual Learning.

Implements EWC regularization to prevent catastrophic forgetting by:
1. Computing Fisher Information after each task to estimate weight importance
2. Storing optimal parameters for each task
3. Adding a quadratic penalty for deviating from important weights

References:
- Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
- Schwarz et al. (2018) "Progress & Compress: A scalable framework for continual learning"
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphParams


@dataclass
class EWCState:
    """State for EWC regularization."""

    # Running Fisher information (diagonal approximation)
    fisher: Optional[GraphParams] = None

    # Optimal parameters from previous tasks
    optimal_params: Optional[GraphParams] = None

    # Number of tasks consolidated
    num_tasks: int = 0

    # Statistics for monitoring
    fisher_max: float = 0.0
    fisher_mean: float = 0.0


class EWCManager:
    """
    Manages EWC (Elastic Weight Consolidation) for continual learning.

    Computes Fisher Information using gradient samples and applies
    quadratic penalty to prevent forgetting important weights.
    """

    def __init__(
        self,
        lambda_ewc: float = 5000.0,
        online: bool = True,
        gamma: float = 0.95,
        normalize_fisher: bool = True,
    ):
        """
        Initialize EWC manager.

        Args:
            lambda_ewc: Regularization strength (higher = less forgetting, more rigidity)
            online: Use online EWC (running Fisher) vs offline (per-task Fisher)
            gamma: Decay factor for online Fisher (0-1, higher keeps older info longer)
            normalize_fisher: Normalize Fisher by max value for numerical stability
        """
        self.lambda_ewc = lambda_ewc
        self.online = online
        self.gamma = gamma
        self.normalize_fisher = normalize_fisher
        self.state = EWCState()

    def compute_fisher(
        self,
        params: GraphParams,
        gradient_fn,
        data_loader,
        num_samples: int = 200,
        rng_key: jax.Array = None,
    ) -> GraphParams:
        """
        Compute diagonal Fisher Information using empirical gradient samples.

        Fisher Information F_i = E[(d log p(y|x,w) / d w_i)^2]

        For neural networks, we approximate this using the squared gradients
        of the loss on the training data.

        Args:
            params: Current model parameters
            gradient_fn: Function that computes gradients given (params, batch, rng_key)
            data_loader: Data loader for sampling
            num_samples: Number of samples to use for Fisher estimation
            rng_key: JAX random key

        Returns:
            Fisher information (same structure as params)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Initialize Fisher accumulator with zeros
        fisher = jax.tree_util.tree_map(jnp.zeros_like, params)

        sample_count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            # Convert batch
            if isinstance(batch_data, (list, tuple)):
                x, y = batch_data
                batch = {
                    "x": jnp.array(x),
                    "y": jnp.array(y) if y.ndim > 1 else jax.nn.one_hot(y, 10),
                }
            else:
                batch = batch_data

            batch_size = batch["x"].shape[0]

            # Split rng key
            rng_key, subkey = jax.random.split(rng_key)

            # Compute gradients for this batch
            grads, _, _ = gradient_fn(params, batch, subkey)

            # Accumulate squared gradients (Fisher diagonal)
            fisher = jax.tree_util.tree_map(
                lambda f, g: f + jnp.square(g) * batch_size,
                fisher,
                grads,
            )

            sample_count += batch_size

        # Average over samples
        if sample_count > 0:
            fisher = jax.tree_util.tree_map(
                lambda f: f / sample_count,
                fisher,
            )

        # Optionally normalize by max value
        if self.normalize_fisher:
            fisher_max = jax.tree_util.tree_reduce(
                lambda acc, f: max(acc, float(jnp.max(f))),
                fisher,
                initializer=1e-10,
            )
            if fisher_max > 1e-10:
                fisher = jax.tree_util.tree_map(
                    lambda f: f / fisher_max,
                    fisher,
                )

        return fisher

    def consolidate_task(
        self,
        params: GraphParams,
        gradient_fn,
        data_loader,
        num_samples: int = 200,
        rng_key: jax.Array = None,
    ) -> None:
        """
        Consolidate knowledge after training on a task.

        Computes Fisher Information and stores optimal parameters.
        For online EWC, blends new Fisher with running estimate.

        Args:
            params: Optimized parameters for current task
            gradient_fn: Function that computes gradients
            data_loader: Data loader for current task
            num_samples: Number of samples for Fisher estimation
            rng_key: JAX random key
        """
        # Compute Fisher for current task
        new_fisher = self.compute_fisher(
            params, gradient_fn, data_loader, num_samples, rng_key
        )

        if self.online and self.state.fisher is not None:
            # Online EWC: blend with running Fisher
            self.state.fisher = jax.tree_util.tree_map(
                lambda old_f, new_f: self.gamma * old_f + new_f,
                self.state.fisher,
                new_fisher,
            )
            # Also blend optimal params (weighted toward more recent)
            self.state.optimal_params = jax.tree_util.tree_map(
                lambda old_p, new_p: self.gamma * old_p + (1 - self.gamma) * new_p,
                self.state.optimal_params,
                params,
            )
        else:
            # First task or offline mode: just store
            self.state.fisher = new_fisher
            self.state.optimal_params = jax.tree_util.tree_map(
                lambda p: jnp.copy(p), params
            )

        self.state.num_tasks += 1

        # Update statistics
        self.state.fisher_max = float(
            jax.tree_util.tree_reduce(
                lambda acc, f: max(acc, float(jnp.max(f))),
                self.state.fisher,
                initializer=0.0,
            )
        )
        self.state.fisher_mean = float(
            jax.tree_util.tree_reduce(
                lambda acc, f: acc + float(jnp.mean(f)),
                self.state.fisher,
                initializer=0.0,
            )
        ) / max(1, len(jax.tree_util.tree_leaves(self.state.fisher)))

    def compute_penalty(self, params: GraphParams) -> jnp.ndarray:
        """
        Compute EWC penalty for current parameters.

        Penalty = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

        Args:
            params: Current model parameters

        Returns:
            Scalar EWC penalty
        """
        if self.state.fisher is None or self.state.optimal_params is None:
            return jnp.array(0.0)

        # Compute weighted squared difference
        penalty = jax.tree_util.tree_map(
            lambda f, p, p_star: f * jnp.square(p - p_star),
            self.state.fisher,
            params,
            self.state.optimal_params,
        )

        # Sum over all parameters
        total_penalty = jax.tree_util.tree_reduce(
            lambda acc, p: acc + jnp.sum(p),
            penalty,
            initializer=jnp.array(0.0),
        )

        return (self.lambda_ewc / 2.0) * total_penalty

    def compute_penalty_gradient(self, params: GraphParams) -> GraphParams:
        """
        Compute gradient of EWC penalty with respect to parameters.

        d(penalty)/d(theta_i) = lambda * F_i * (theta_i - theta*_i)

        Args:
            params: Current model parameters

        Returns:
            Gradient of EWC penalty (same structure as params)
        """
        if self.state.fisher is None or self.state.optimal_params is None:
            return jax.tree_util.tree_map(jnp.zeros_like, params)

        # Compute gradient: lambda * F * (theta - theta*)
        grad = jax.tree_util.tree_map(
            lambda f, p, p_star: self.lambda_ewc * f * (p - p_star),
            self.state.fisher,
            params,
            self.state.optimal_params,
        )

        return grad

    def save_state(self) -> Dict[str, Any]:
        """Save EWC state for checkpointing."""
        import numpy as np

        state_dict = {
            "num_tasks": self.state.num_tasks,
            "fisher_max": self.state.fisher_max,
            "fisher_mean": self.state.fisher_mean,
        }

        if self.state.fisher is not None:
            state_dict["fisher"] = jax.tree_util.tree_map(np.array, self.state.fisher)

        if self.state.optimal_params is not None:
            state_dict["optimal_params"] = jax.tree_util.tree_map(
                np.array, self.state.optimal_params
            )

        return state_dict

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load EWC state from checkpoint."""
        self.state.num_tasks = state_dict.get("num_tasks", 0)
        self.state.fisher_max = state_dict.get("fisher_max", 0.0)
        self.state.fisher_mean = state_dict.get("fisher_mean", 0.0)

        if "fisher" in state_dict:
            self.state.fisher = jax.tree_util.tree_map(jnp.array, state_dict["fisher"])

        if "optimal_params" in state_dict:
            self.state.optimal_params = jax.tree_util.tree_map(
                jnp.array, state_dict["optimal_params"]
            )


def create_ewc_train_step(
    base_gradient_fn,
    ewc_manager: EWCManager,
):
    """
    Create a training step function that includes EWC regularization.

    Args:
        base_gradient_fn: Base gradient computation function
        ewc_manager: EWC manager for computing penalty gradients

    Returns:
        Modified gradient function that includes EWC penalty
    """

    def ewc_gradient_fn(params, batch, rng_key):
        # Compute base gradients
        grads, energy, final_state = base_gradient_fn(params, batch, rng_key)

        # Add EWC penalty gradient
        ewc_grad = ewc_manager.compute_penalty_gradient(params)

        # Combine gradients
        combined_grads = jax.tree_util.tree_map(
            lambda g, e: g + e,
            grads,
            ewc_grad,
        )

        # Add EWC penalty to energy for monitoring
        ewc_penalty = ewc_manager.compute_penalty(params)
        total_energy = energy + float(ewc_penalty)

        return combined_grads, total_energy, final_state

    return ewc_gradient_fn


def ewc_gradient_transform(ewc_manager: EWCManager):
    """
    Create an optax gradient transformation that adds EWC penalty gradients.

    This can be chained with other optimizers to add EWC regularization
    during training. The penalty gradients are computed based on the
    difference from optimal parameters weighted by Fisher information.

    Args:
        ewc_manager: EWC manager that tracks Fisher information and optimal params

    Returns:
        optax.GradientTransformation that adds EWC penalty gradients

    Example:
        >>> ewc_manager = EWCManager(lambda_ewc=5000.0)
        >>> optimizer = optax.chain(
        ...     ewc_gradient_transform(ewc_manager),
        ...     optax.adam(1e-3),
        ... )
    """
    import optax

    def init_fn(params):
        # No state needed - EWC manager maintains its own state
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            # If params not provided, can't compute EWC penalty
            return updates, state

        # Get EWC penalty gradient
        ewc_grad = ewc_manager.compute_penalty_gradient(params)

        # Add EWC gradient to updates (which are the base gradients)
        new_updates = jax.tree_util.tree_map(
            lambda u, e: u + e,
            updates,
            ewc_grad,
        )

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
