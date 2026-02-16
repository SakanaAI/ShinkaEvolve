# Disable progress bar for cleaner output logs
import os

os.environ["TQDM_DISABLE"] = "1"

# Fixed parameters
n, m, p = 2, 4, 5

# EVOLVE-BLOCK-START
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
import tqdm


# --- Straight-Through Estimator for Rounding ---
@jax.custom_vjp
def round_to_half_ste(x):
    """Forward pass: snaps values to the nearest half-integer."""
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    """Standard forward pass and identity for backward pass."""
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    """Backward pass: Identity function, passes gradient straight through."""
    return (g,)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)
# --- End of STE definition ---


# --- Loss Functions ---
def weighted_l2_loss(reconstructed: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    error = reconstructed - target
    weights = jnp.where(target != 0, 100.0, 1.0)
    return jnp.mean(weights * (error**2))


def l2_loss_real(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)


# --- Hyperparameters ---
@dataclass
class Hyperparameters:
    rank: int = 55
    # Phase 1: Continuous Search
    num_restarts: int = 10
    phase1_steps: int = 80000
    phase1_lr: float = 0.01
    init_scale: float = 0.1
    l1_strength: float = 1e-6
    clamp_range: float = 4.0
    # Phase 2: Discrete Fine-tuning
    phase2_steps: int = 20000
    phase2_lr: float = 1e-4  # A much smaller learning rate for fine-tuning


# --- Optimizer Classes ---
class ContinuousOptimizer:
    """Finds a high-quality approximate continuous solution."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.adam(hypers.phase1_lr)

    def _get_constrained_decomposition(self, latent_decomposition: tuple) -> tuple:
        """Applies a scaled tanh to map latent parameters to the desired range."""
        return jax.tree_util.tree_map(
            lambda x: self.hypers.clamp_range * jnp.tanh(x), latent_decomposition
        )

    def _loss_fn(self, latent_decomposition: tuple) -> jnp.ndarray:
        constrained = self._get_constrained_decomposition(latent_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained)
        recon_loss = weighted_l2_loss(reconstructed, self.target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + self.hypers.l1_strength * l1_penalty


class DiscreteOptimizer:
    """Refines a continuous solution into an exact discrete one using an STE."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.adam(hypers.phase2_lr)

    def _loss_fn(self, continuous_decomposition: tuple) -> jnp.ndarray:
        # Snap the continuous parameters to the discrete grid
        discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_decomposition)
        # Compute the loss using only these exact half-integer values
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition)
        return l2_loss_real(reconstructed, self.target_tensor)


# --- JIT-compatible Train Step ---
def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def get_matrix_multiplication_tensor(n, m, p):
    T = jnp.zeros((n * m, m * p, n * p))
    for i, j, k in np.ndindex(n, m, p):
        T = T.at[i * m + j, j * p + k, k * n + i].set(1)
    return T


def run():
    hypers = Hyperparameters()
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)

    # --- PHASE 1: CONTINUOUS EXPLORATION ---
    print(f"\n{'='*20} PHASE 1: Continuous Exploration {'='*20}")
    best_loss_phase1 = float("inf")
    best_latent_decomp = None

    continuous_optimizer = ContinuousOptimizer(target_tensor, hypers)

    # JIT the train_step for the continuous phase
    jit_train_step_continuous = jax.jit(train_step, static_argnums=(2, 3))

    for i in range(hypers.num_restarts):
        print(f"\n--- Restart {i+1}/{hypers.num_restarts} ---")
        main_key, restart_key = jax.random.split(main_key)
        init_fn = jax.nn.initializers.normal(stddev=hypers.init_scale)
        latent_decomp = (
            init_fn(restart_key, (n * m, hypers.rank)),
            init_fn(restart_key, (m * p, hypers.rank)),
            init_fn(restart_key, (n * p, hypers.rank)),
        )
        opt_state = continuous_optimizer.opt.init(latent_decomp)

        for _ in tqdm.tqdm(range(hypers.phase1_steps), desc="Continuous Search"):
            latent_decomp, opt_state, loss = jit_train_step_continuous(
                latent_decomp,
                opt_state,
                continuous_optimizer.opt,
                continuous_optimizer._loss_fn,
            )

        final_loss = l2_loss_real(
            target_tensor,
            jnp.einsum(
                "ir,jr,kr->ijk",
                *continuous_optimizer._get_constrained_decomposition(latent_decomp),
            ),
        )
        print(f"End of Trial | Final continuous loss: {final_loss:.8f}")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp

    # --- PHASE 2: DISCRETE FINE-TUNING ---
    print(f"\n{'='*20} PHASE 2: Discrete Fine-tuning (STE) {'='*20}")
    print(f"Starting with best continuous solution (loss: {best_loss_phase1:.8f})")

    continuous_params = continuous_optimizer._get_constrained_decomposition(best_latent_decomp)

    discrete_optimizer = DiscreteOptimizer(target_tensor, hypers)
    opt_state = discrete_optimizer.opt.init(continuous_params)

    # JIT the train_step for the discrete phase
    jit_train_step_discrete = jax.jit(train_step, static_argnums=(2, 3))

    for step in tqdm.tqdm(range(hypers.phase2_steps), desc="Discrete Fine-tuning"):
        continuous_params, opt_state, loss = jit_train_step_discrete(
            continuous_params, opt_state, discrete_optimizer.opt, discrete_optimizer._loss_fn
        )
        if (step + 1) % 2000 == 0:
            print(f"Step {step+1} | Discrete Loss: {loss:.8f}")
        if loss < 1e-7:
            print("\nFound a perfect solution!")
            break

    final_discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_params)
    final_loss = l2_loss_real(
        target_tensor, jnp.einsum("ir,jr,kr->ijk", *final_discrete_decomposition)
    )
    print(f"Search complete. Final discrete loss: {final_loss:.8f}")

    final_decomposition_np = jax.tree_util.tree_map(np.array, final_discrete_decomposition)
    return final_decomposition_np, n, m, p, float(final_loss), hypers.rank


# EVOLVE-BLOCK-END
