"""
Optimized version of quantum_run_mc using jitted functions from shadow_mcs_jitted.py

This module provides JIT-compiled alternatives to the slow parts of the notebook.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Dict, Tuple, Any
import sys
import os
import tensorcircuit as tc

# Import the jitted functions
sys.path.append(os.path.abspath("modules"))

# Import jitted trajectory sampling from shadow_mcs_jitted
import shadow_mcs_jitted as shadow_module
from shadow_mcs_jitted import (
    get_kraus_operators,
    generate_psi_F_vector,
    sample_trajectories,
    group_trajectories_jax,
    trajectory_from_key_jit
)

# Type alias for JAX arrays
Array = jax.Array

# =============================================================================
# OPTIMIZED CATALOG GENERATION (REPLACES cs.generate_efficient_noisy_samples)
# =============================================================================

@partial(jax.jit, static_argnames=("nq", "num_samples"))
def generate_catalog_jitted(
    key: jax.Array,
    fs: jax.Array,
    nq: int,
    Ks: jax.Array,
    num_samples: int
) -> Dict[str, jax.Array]:
    """
    JIT-compiled catalog generation using trajectory sampling.
    
    Args:
        key: PRNG key
        fs: F vector (2^nq,)
        nq: number of qubits
        Ks: Kraus operators from get_kraus_operators
        num_samples: number of Monte Carlo samples
    
    Returns:
        Dict with:
            - states_sorted: (S, 2^nq) sorted unique states
            - multiplicities: (S,) counts for each state
            - num_errors: (S,) error counts
            - is_group_start: (S,) boolean mask for group starts
    """
    # Sample trajectories
    psi0 = generate_psi_F_vector(fs, nq)
    keys = jax.random.split(key, num_samples)
    one = lambda k: trajectory_from_key_jit(k, psi0, nq, Ks)
    states, errs, codes = jax.vmap(one)(keys)
    
    # Group by trajectory code
    grouped = group_trajectories_jax(states, errs, codes)
    
    return {
        "states_sorted": grouped["states_sorted"],
        "counts_at_starts": grouped["counts_at_starts"],
        "num_errors": grouped["errs_sorted"],
        "is_group_start": grouped["is_group_start"],
        "num_groups": grouped["num_groups"]
    }


# =============================================================================
# IMPORTANCE SAMPLING WEIGHTS (VECTORIZED)
# =============================================================================

@jax.jit
def calculate_binomial_weights_jitted(
    num_errors: jax.Array,
    multiplicities: jax.Array,
    nq: int,
    amps: jax.Array,
    p_high: float,
    total_samples: int
) -> jax.Array:
    """
    Vectorized importance sampling for binomial channels (dephasing, bit_flip).
    
    Args:
        num_errors: (S,) error counts per state
        multiplicities: (S,) observed counts per state
        nq: number of qubits
        amps: (N_amps,) target amplitudes
        p_high: high probability used in sampling
        total_samples: total MC samples for normalization
    
    Returns:
        prob_weights: (S, N_amps) normalized probability weights
    """
    observed_probs = multiplicities / jnp.maximum(total_samples, 1)
    
    # Vectorized reweighting: (N_amps, S)
    p_low = amps[:, jnp.newaxis]
    k = num_errors[jnp.newaxis, :]
    
    # Avoid division by zero
    p_high_safe = jnp.maximum(p_high, 1e-9)
    
    factors = (p_low / p_high_safe)**k * ((1 - p_low) / (1 - p_high_safe))**(nq - k)
    
    # prob_weights: (S, N_amps)
    prob_weights = observed_probs[:, jnp.newaxis] * factors.T
    
    # Normalize columns
    col_sums = prob_weights.sum(axis=0, keepdims=True)
    prob_weights = prob_weights / jnp.where(col_sums > 1e-9, col_sums, 1.0)
    
    return prob_weights


@jax.jit
def calculate_ad_weights_jitted(
    num_errors: jax.Array,
    num_ones_initial: jax.Array,
    multiplicities: jax.Array,
    amps: jax.Array,
    gamma_high: float,
    total_samples: int
) -> jax.Array:
    """
    Vectorized importance sampling for amplitude damping (state-dependent).
    
    Args:
        num_errors: (S,) number of decay events (k)
        num_ones_initial: (S,) number of susceptible qubits (m)
        multiplicities: (S,) observed counts
        amps: (N_amps,) target gammas
        gamma_high: high gamma used in sampling
        total_samples: total MC samples
    
    Returns:
        prob_weights: (S, N_amps)
    """
    observed_probs = multiplicities / jnp.maximum(total_samples, 1)
    
    k = num_errors[:, jnp.newaxis]  # (S, 1)
    m = num_ones_initial[:, jnp.newaxis]  # (S, 1)
    gamma_low = amps[jnp.newaxis, :]  # (1, N_amps)
    
    # Handle m=0 case: if no qubits can decay, weight is just observed prob
    gamma_high_safe = jnp.maximum(gamma_high, 1e-9)
    
    prob_at_high = (gamma_high_safe**k) * ((1 - gamma_high_safe)**(m - k))
    prob_at_low = (gamma_low**k) * ((1 - gamma_low)**(m - k))
    
    # Reweighting factor
    factor = jnp.where(
        prob_at_high > 1e-12,
        prob_at_low / prob_at_high,
        0.0
    )
    
    # Handle m=0: set weight to observed_prob for k=0, else 0
    factor = jnp.where(
        m == 0,
        jnp.where(k == 0, 1.0, 0.0),
        factor
    )
    
    prob_weights = observed_probs[:, jnp.newaxis] * factor
    
    # Normalize
    col_sums = prob_weights.sum(axis=0, keepdims=True)
    prob_weights = prob_weights / jnp.where(col_sums > 1e-9, col_sums, 1.0)
    
    return prob_weights


# =============================================================================
# BATCHED F-VECTOR PROCESSING
# =============================================================================

@partial(jax.jit, static_argnames=("nq", "num_samples"))
def process_f_batch_catalogs(
    key: jax.Array,
    f_batch: jax.Array,
    nq: int,
    Ks: jax.Array,
    num_samples: int
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Process a batch of F vectors to generate catalogs in parallel.
    
    Args:
        key: PRNG key
        f_batch: (B, 2^nq) batch of F vectors
        nq: number of qubits
        Ks: Kraus operators (from get_kraus_operators)
        num_samples: MC samples per F vector
    
    Returns:
        Tuple of:
            - all_states: (B, S, 2^nq)
            - all_counts: (B, S)
            - all_errors: (B, S)
            - all_masks: (B, S) - True at group starts
    """
    batch_size = f_batch.shape[0]
    
    # Split keys for each F vector
    keys = jax.random.split(key, batch_size)
    
    def process_one(k, f):
        catalog = generate_catalog_jitted(k, f, nq, Ks, num_samples)
        return (
            catalog["states_sorted"],
            catalog["counts_at_starts"],
            catalog["num_errors"],
            catalog["is_group_start"]
        )
    
    # VMAP over the batch
    return jax.vmap(process_one)(keys, f_batch)


# =============================================================================
# OPTIMIZED OUTCOME PROCESSING AND ACCURACY CALCULATION
# =============================================================================

def compute_alpha_permutations(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the qubit permutation that maps the canonical decoder layout to the
    provided alpha configuration, together with its inverse.

    The canonical template groups all control qubits (where alpha[i] = 1 for
    i < nq-1) at the beginning of the register and keeps the last qubit fixed.
    For a general alpha we therefore need to:
      1. Gather the indices of control qubits (alpha[:-1] == 1) in ascending order.
      2. Append the indices of the remaining data qubits (alpha[:-1] == 0).
      3. Keep the last qubit (the decoder output) fixed at position nq-1.

    The resulting permutation gives the mapping from canonical qubit indices to
    the indices used by the specific alpha. The inverse permutation allows us to
    map measurement results back to the canonical ordering used by the circuit
    template.

    Args:
        alpha: Array-like of shape (nq,) with entries 0/1.

    Returns:
        canonical_to_actual: np.ndarray of shape (nq,), mapping canonical index -> actual index.
        actual_to_canonical: np.ndarray of shape (nq,), mapping actual index -> canonical index.
    """
    alpha_np = np.asarray(alpha, dtype=np.int32)
    nq = alpha_np.shape[0]
    if nq == 0:
        raise ValueError("Alpha must have at least one qubit.")

    # Separate data-region (all but last qubit) from decoder output (last qubit).
    alpha_head = alpha_np[:-1]
    control_indices = np.flatnonzero(alpha_head)
    passive_indices = np.flatnonzero(alpha_head == 0)

    canonical_to_actual = np.concatenate(
        [control_indices, passive_indices, np.array([nq - 1], dtype=np.int32)]
    ).astype(np.int32)

    if canonical_to_actual.shape[0] != nq:
        raise ValueError("Permutation construction failed; wrong number of indices.")

    actual_to_canonical = np.empty_like(canonical_to_actual)
    actual_to_canonical[canonical_to_actual] = np.arange(nq, dtype=np.int32)

    return canonical_to_actual, actual_to_canonical


def permute_state_vector_batch(states: np.ndarray, canonical_to_actual_perm: np.ndarray) -> np.ndarray:
    """
    Reorder a batch of state vectors so they match the canonical decoder layout.

    Args:
        states: np.ndarray of shape (batch, 2**nq) or (2**nq,), complex dtype expected.
        canonical_to_actual_perm: np.ndarray of shape (nq,) giving the mapping
            from canonical qubit index to the index used by the specific alpha.

    Returns:
        np.ndarray with the same shape as `states`, where the qubit axes have been
        permuted so that the canonical decoder template acts on the correct qubits.
    """
    if canonical_to_actual_perm is None:
        return states

    states_np = np.asarray(states)
    if states_np.ndim == 1:
        states_np = states_np[np.newaxis, :]
        squeeze_back = True
    elif states_np.ndim == 2:
        squeeze_back = False
    else:
        raise ValueError("State vectors must have shape (2**nq,) or (batch, 2**nq).")

    nq = canonical_to_actual_perm.shape[0]
    target_dim = 1 << nq
    if states_np.shape[-1] != target_dim:
        raise ValueError(
            f"State dimension mismatch: expected {target_dim}, got {states_np.shape[-1]}"
        )

    reshaped = states_np.reshape((-1,) + (2,) * nq)
    axes = (0,) + tuple(int(i) + 1 for i in canonical_to_actual_perm)
    permuted = np.transpose(reshaped, axes=axes)
    result = permuted.reshape(states_np.shape)

    if squeeze_back:
        result = result[0]
    return result



@partial(jax.jit, static_argnames=("nq", "n_measured"))
def process_results_to_accuracy(
    results: jax.Array,
    measurement_indices: jax.Array,
    f_vec: jax.Array,
    alpha: jax.Array,
    nq: int,
    n_measured: int
) -> float:
    """
    JIT-compiled version matching ns.get_accuracy logic.
    
    The original logic:
    - Takes last bit as 'b'
    - Takes rest of bits, adds "0", converts to 'y'
    - Checks if F[y XOR alpha] XOR F[y] == b
    - Sums probabilities where this is true
    
    Args:
        results: (shots, nq) measurement outcomes as 0/1
        measurement_indices: (n_measured,) which qubits to measure
        f_vec: (2^nq,) F vector
        alpha: (nq,) alpha bitstring
        nq: total number of qubits
        n_measured: number of measured qubits
    
    Returns:
        accuracy: float scalar
    """
    # Extract measured qubits: (shots, n_measured)
    measured = results[:, measurement_indices]
    
    # Convert bitstrings to integers: (shots,)
    powers = 2 ** jnp.arange(n_measured - 1, -1, -1, dtype=jnp.int32)
    outcome_ints = (measured @ powers).astype(jnp.int32)
    
    # Histogram: (2^n_measured,)
    n_outcomes = 2 ** n_measured
    histogram = jnp.bincount(outcome_ints, length=n_outcomes)
    
    # For each possible outcome, compute accuracy contribution
    # Following the exact logic from compute_noisy_accuracy:
    # 1. Convert outcome to bit array (n_measured bits)
    # 2. Set last bit to 0 to get y_bits
    # 3. XOR y_bits with alpha bits (in full nq space)
    # 4. Convert back to integers for F lookup
    
    outcome_values = jnp.arange(n_outcomes, dtype=jnp.int32)
    
    # Convert outcomes to bit arrays: (n_outcomes, n_measured)
    # MSB first: bit at position i corresponds to 2^(n_measured-1-i)
    # Position 0 is MSB, position -1 (last) is LSB
    outcome_bits = ((outcome_values[:, None] >> jnp.arange(n_measured - 1, -1, -1, dtype=jnp.int32)) & 1).astype(jnp.int32)
    
    # Extract b (last bit of outcome) - this is the LSB (position -1 in the bit array)
    b_vals = outcome_bits[:, -1].astype(jnp.int32)  # Last bit (LSB)
    
    # Create y_bits: same as outcome_bits but with last bit set to 0
    y_bits = outcome_bits.at[:, -1].set(0)
    
    # Get alpha bits for measured qubits: (n_measured,)
    alpha_measured = alpha[measurement_indices].astype(jnp.int32)
    
    # For F lookups, we need to work in full nq-bit space
    # Map y_bits to full space and XOR with alpha in full space
    full_powers = 2 ** jnp.arange(nq - 1, -1, -1, dtype=jnp.int32)
    
    # Map y_bits to full space: (n_outcomes, n_measured) -> (n_outcomes, nq)
    # Place bits at measurement_indices positions
    y_bits_full = jnp.zeros((n_outcomes, nq), dtype=jnp.int32)
    # Use scatter: y_bits_full[i, measurement_indices[j]] = y_bits[i, j]
    y_bits_full = y_bits_full.at[jnp.arange(n_outcomes)[:, None], measurement_indices[None, :]].set(y_bits)
    y_vals_full = (y_bits_full @ full_powers).astype(jnp.int32)
    
    # Get alpha bits in full nq-bit space: (nq,)
    alpha_bits_full = alpha.astype(jnp.int32)  # (nq,)
    
    # XOR y_bits_full with alpha_bits_full: (n_outcomes, nq) XOR (1, nq) -> (n_outcomes, nq)
    # This XORs each row of y_bits_full with alpha_bits_full
    y_xor_alpha_bits_full = jnp.bitwise_xor(y_bits_full, alpha_bits_full[None, :])
    y_xor_alpha_full = (y_xor_alpha_bits_full @ full_powers).astype(jnp.int32)
    
    # Ensure indices are within bounds
    y_vals_safe = y_vals_full % f_vec.shape[0]
    y_xor_alpha_safe = y_xor_alpha_full % f_vec.shape[0]
    
    f_y = f_vec[y_vals_safe]
    f_y_xor_alpha = f_vec[y_xor_alpha_safe]
    
    predicted_b = jnp.bitwise_xor(f_y, f_y_xor_alpha)
    correct = (predicted_b == b_vals).astype(jnp.float64)
    
    # Sum probabilities of correct outcomes
    total_shots = jnp.sum(histogram)
    accuracy = jnp.sum(histogram.astype(jnp.float64) * correct) / jnp.maximum(total_shots, 1.0)
    
    # Note: Debug printing cannot be done inside JIT-compiled function
    # Debug info is printed in the calling function instead
    
    return accuracy


@partial(jax.jit, static_argnames=("nq",))
def process_batch_results_to_accuracies(
    results_batch: jax.Array,
    measurement_indices: jax.Array,
    f_vec: jax.Array,
    alpha: jax.Array,
    nq: int,
) -> jax.Array:
    """
    Vectorized accuracy computation for a batch of state results.
    
    Args:
        results_batch: (n_states, shots, nq) measurement outcomes
        measurement_indices: (n_measured,) which qubits to measure
        f_vec: (2^nq,) F vector
        alpha: (nq,) alpha bitstring
        nq: number of qubits
    
    Returns:
        accuracies: (n_states,) accuracy for each state
    """
    n_measured = measurement_indices.shape[0]
    
    def compute_one(results_for_state):
        return process_results_to_accuracy(
            results_for_state, measurement_indices, f_vec, alpha, nq, n_measured
        )
    
    return jax.vmap(compute_one)(results_batch)


# =============================================================================
# HELPER: COMPUTE NUM_ONES_INITIAL FOR AD WEIGHTING
# =============================================================================

@partial(jax.jit, static_argnames=("nq",))
def count_ones_in_states(states: jax.Array, nq: int) -> jax.Array:
    """
    Count number of |1⟩s in each state vector by checking amplitudes.
    
    For a state in the computational basis, this counts the Hamming weight
    of the basis states with non-zero amplitude.
    
    Args:
        states: (S, 2^nq) complex state vectors
        nq: number of qubits (static, must be known at compile time)
    
    Returns:
        num_ones: (S,) integer counts
    """
    dim = 1 << nq
    # Find which basis state has max amplitude (assume near-computational basis)
    max_idx = jnp.argmax(jnp.abs(states), axis=1)
    
    # Count bits in the index
    def count_bits(x):
        return jnp.sum(((x >> jnp.arange(nq)) & 1).astype(jnp.int32))
    
    return jax.vmap(count_bits)(max_idx)


# =============================================================================
# OPTIMIZED ACCURACY CALCULATION (USING JAX)
# =============================================================================

@partial(jax.jit, static_argnames=("nq",))
def compute_accuracy_from_counts_jitted(
    f_vec: jax.Array,
    alpha: jax.Array,
    outcome_counts: jax.Array,
    outcome_indices: jax.Array,
    total_shots: int,
    nq: int
) -> float:
    """
    JIT-compiled accuracy calculation.
    
    Args:
        f_vec: (2^nq,) F vector
        alpha: (nq,) alpha bitstring
        outcome_counts: (K,) counts for each unique outcome
        outcome_indices: (K,) integer representation of outcomes
        total_shots: total number of shots
        nq: number of qubits
    
    Returns:
        accuracy: float
    """
    # Convert alpha to integer
    alpha_int = jnp.sum(alpha * (2 ** jnp.arange(nq - 1, -1, -1)))
    
    # For each outcome, compute alpha XOR outcome
    xor_results = jnp.bitwise_xor(alpha_int, outcome_indices)
    
    # Look up F values
    f_vals = f_vec[xor_results]
    
    # Compute (-1)^F values
    signs = (-1.0) ** f_vals
    
    # Weighted sum
    numerator = jnp.sum(outcome_counts * signs)
    accuracy = numerator / jnp.maximum(total_shots, 1)
    
    return jnp.abs(accuracy)


# =============================================================================
# WRAPPER: Handle "relaxation" as alias for "thermal"
# =============================================================================

def get_kraus_operators_with_aliases(channel_config: Dict[str, Any]) -> Array:
    """
    Wrapper around shadow_mcs_jitted.get_kraus_operators that handles aliases.
    
    Supports:
    - 'relaxation' as alias for 'thermal' (amplitude damping)
    - All other types passed through unchanged
    """
    config = channel_config.copy()
    
    # Map "relaxation" to "thermal"
    if config.get('type') == 'relaxation':
        config['type'] = 'thermal'
        # Default to no thermal excitation if not specified
        if 'thermal_p_exc' not in config:
            config['thermal_p_exc'] = 0.0
    
    return get_kraus_operators(config)


# =============================================================================
# SIMULATION HELPER FUNCTIONS
# =============================================================================

def draw_alpha(nq):
    """
    Draws a binary vector 'alpha' of length nq where the Hamming distance
    is chosen uniformly and the last bit is always 1.
    """
    # Choose a Hamming distance 'd' with equal probability from 1 to nq.
    d = np.random.randint(0, nq)
    
    # Start with a vector of zeros and set the last entry to 1.
    alpha = np.zeros(nq, dtype=int)
    alpha[-1] = 1
    alpha[:d-1] = 1
    
    return alpha


def pick_minmax_alpha(nq):
    """Pick either minimal or maximal alpha configuration."""
    a = np.random.randint(2, size=1)[0]
    if a == 0:
        return np.concatenate((np.zeros(nq-1), np.ones(1))).astype(np.int32)
    elif a == 1:
        return np.ones(nq, dtype=np.int32)


@partial(jax.jit)
def run_simulation_for_one_state_jax(state_vector, key, transpiled_template, noise_conf, shots, with_noise):
    """Simulates all shots for a SINGLE state vector with proper key management."""
    if not with_noise:
        # For the noiseless case
        @partial(jax.jit)
        def _run_single_noiseless_shot(k):
            c = transpiled_template.copy()
            c.replace_inputs(state_vector)
            return c.sample(allow_state=True, random_generator=k)[0]
        
        shot_keys = jax.random.split(key, shots)
        return jax.vmap(_run_single_noiseless_shot)(shot_keys)

    # Noisy simulation
    num_status_params = len(transpiled_template.to_qir())
    statuses = jax.random.uniform(key, shape=(shots, num_status_params))
    
    @partial(jax.jit)
    def scan_body(carry, status_for_shot):
        carry_key, state_vec = carry
        key_i, subkey = jax.random.split(carry_key)
        
        c2 = transpiled_template.copy()
        c2 = tc.noisemodel.circuit_with_noise(c2, noise_conf, status=status_for_shot)
        c2.replace_inputs(state_vec)
        sample = c2.sample(allow_state=True, random_generator=subkey)[0]
        
        return (key_i, state_vec), sample

    initial_carry = (key, state_vector)
    _, all_samples = jax.lax.scan(scan_body, initial_carry, statuses)
    return all_samples


def run_batched_simulation_jax(transpiled_template, batch_states, key, noise_conf, shots, with_noise):
    """Orchestrates the simulation for a BATCH of states using vmap."""
    state_keys = jax.random.split(key, batch_states.shape[0])
    vmap_states = jax.vmap(run_simulation_for_one_state_jax, in_axes=(0, 0, None, None, None, None))
    return vmap_states(batch_states, state_keys, transpiled_template, noise_conf, shots, with_noise)


# =============================================================================
# BATCHING HELPER FUNCTIONS
# =============================================================================

def get_max_batch_size(nq, mcs_for_catalog, shots_per_state, target_memory=2e9):
    """
    Returns maximum batch size based on available memory and qubit count.
    
    Args:
        nq: Number of qubits
        mcs_for_catalog: Monte Carlo samples per catalog
        shots_per_state: Shots per state
        target_memory: Target memory limit in bytes (default 2GB for safety)
    
    Returns:
        max_batch: Maximum batch size
    """
    state_memory_per_function = mcs_for_catalog * (2**nq) * 8  # bytes
    
    # JAX vmap creates arrays of shape (batch_size, ...) for all intermediate operations
    # During JIT compilation, XLA needs to allocate memory for the ENTIRE computation graph
    # Be VERY conservative: vmap + JIT + trajectory sampling = lots of copies
    overhead_factor = 5.0  # Increased from 2.5 to be more conservative
    total_memory_per_function = state_memory_per_function * overhead_factor
    
    max_batch = max(1, int(target_memory / total_memory_per_function))
    
    # Very conservative hard limits based on state dimension
    if nq >= 14:
        max_batch = 1
    elif nq >= 13:
        max_batch = min(max_batch, 2)
    elif nq >= 12:
        max_batch = min(max_batch, 4)
    elif nq >= 10:
        max_batch = min(max_batch, 8)
    elif nq >= 8:
        max_batch = min(max_batch, 10)
    
    return max_batch


def get_decoder_batch_size(nq, n_catalog_states, shots_per_state, target_memory=500e6):
    """
    Determines how many catalog states to simulate through the decoder at once.
    
    Args:
        nq: Number of qubits
        n_catalog_states: Number of catalog states
        shots_per_state: Number of shots per state
        target_memory: Target memory in bytes (default 500MB for safety)
    
    Returns:
        batch_size: Number of states to process at once
    """
    # Memory for results array (int8)
    results_memory_per_state = shots_per_state * nq * 1  # bytes
    
    # Memory for intermediate state vectors (complex128 = 16 bytes)
    state_vector_memory = 2**nq * 16  # bytes per state
    
    # JAX typically needs 2-3 copies during computation (overhead factor)
    overhead_factor = 3.0
    total_memory_per_state = results_memory_per_state + (state_vector_memory * overhead_factor)
    
    max_states_per_batch = int(target_memory / total_memory_per_state)
    
    # Conservative hard limits - memory grows exponentially with nq
    if nq >= 16:
        batch_limit = 1  # Only 1 state at a time for nq >= 16!
    elif nq == 15:
        batch_limit = 2  # Very small batch for nq=15
    elif nq == 14:
        batch_limit = 5
    elif nq == 13:
        batch_limit = 10
    elif nq == 12:
        batch_limit = 20
    elif nq <= 11:
        batch_limit = 100
    elif nq <= 9:
        batch_limit = 500
    else:
        batch_limit = 1000
    
    return min(max_states_per_batch, batch_limit, n_catalog_states)


def split_into_sub_batches(data, max_batch_size):
    """Splits grouped data into sub-batches of maximum size."""
    n_total = len(data['labels'])
    sub_batches = []
    
    for i in range(0, n_total, max_batch_size):
        end_idx = min(i + max_batch_size, n_total)
        sub_batch = {
            'labels': data['labels'][i:end_idx],
            'F_vectors': data['F_vectors'][i:end_idx],
            'alphas': data['alphas'][i:end_idx]
        }
        sub_batches.append(sub_batch)
    
    return sub_batches


print("✅ Optimized quantum_run_mc functions loaded successfully!")
print("   - generate_catalog_jitted: JIT-compiled catalog generation")
print("   - calculate_binomial_weights_jitted: Vectorized binomial weights")
print("   - calculate_ad_weights_jitted: Vectorized AD weights")
print("   - process_f_batch_catalogs: Batched F-vector processing")
print("   - get_kraus_operators_with_aliases: Handles 'relaxation' alias")
print("   - Helper functions: batching, simulation, and configuration")

