"""
Wrapper module that provides MC optimization routines for quantum simulations.

This module imports core functionality from shadow_mcs_jitted and provides
convenience functions for trajectory processing and accuracy computation.
"""

import sys
import os
from pathlib import Path

# Import from shadows_simulation directory
# __file__ is in code/quantum_simulation/modules/
# Go up 2 levels: modules/ -> quantum_simulation/ -> code/
# shadows_simulation lives as a sibling of quantum_simulation inside code/
quantum_sim_dir = Path(__file__).parent.parent  # code/quantum_simulation/
project_root = quantum_sim_dir.parent           # code/
shadows_sim_path = project_root / "shadows_simulation"  # code/shadows_simulation/

if str(shadows_sim_path) not in sys.path:
    sys.path.insert(0, str(shadows_sim_path))

try:
    from shadow_mcs_jitted import (
        get_raus_operators,
        generate_psi_F_vector,
        trajectory_from_key_jit,
        sample_trajectories,
        sample_trajectories_batched,
        group_trajectories_jax,
        generate_efficient_noisy_samples_arrays,
        generate_efficient_noisy_samples_named,
        expand_grouped_to_megabatch_from_arrays,
        expand_grouped_to_megabatch_named,
        mcs_shadows_all_jit,
        mcs_shadows_streaming_jit,
        mcs_shadows_streaming_2stage_jit,
        generate_efficient_noisy_samples,
        generate_all_nps_checkpoints,
        generate_all_nps_checkpoints_batched,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import from shadow_mcs_jitted: {e}\n"
        f"Make sure shadows_simulation/shadow_mcs_jitted.py exists."
    ) from e

import jax
import jax.numpy as jnp
import numpy as np


def get_kraus_operators_with_aliases(channel_config):
    """Wrapper around get_kraus_operators with channel name aliasing.

    Maps common channel names to shadow_mcs_jitted's supported types:
      - 'relaxation' -> 'thermal'

    Args:
        channel_config: dict with 'type' and 'strength' keys

    Returns:
        Kraus operators as JAX array
    """
    _aliases = {'relaxation': 'thermal'}
    cfg = dict(channel_config)
    cfg['type'] = _aliases.get(cfg.get('type', ''), cfg.get('type', ''))
    return get_kraus_operators(cfg)


def process_results_to_accuracy(
    results: jnp.ndarray,
    measurement_indices: jnp.ndarray,
    f_vec: jnp.ndarray,
    alpha: jnp.ndarray,
    nq: int,
    n_measured: int
) -> float:
    """
    JIT-compatible version: pure JAX operations, works inside vmap/jit.
    
    Computes accuracy matching ns.get_accuracy logic:
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
    
    outcome_values = jnp.arange(n_outcomes, dtype=jnp.int32)
    
    # Convert outcomes to bit arrays: (n_outcomes, n_measured)
    outcome_bits = ((outcome_values[:, None] >> jnp.arange(n_measured - 1, -1, -1, dtype=jnp.int32)) & 1).astype(jnp.int32)
    
    # Extract b (last bit of outcome)
    b_vals = outcome_bits[:, -1].astype(jnp.int32)
    
    # Create y_bits: outcome_bits with last bit set to 0
    y_bits = outcome_bits.at[:, -1].set(0)
    
    # Map y_bits to full space
    full_powers = 2 ** jnp.arange(nq - 1, -1, -1, dtype=jnp.int32)
    
    y_bits_full = jnp.zeros((n_outcomes, nq), dtype=jnp.int32)
    y_bits_full = y_bits_full.at[jnp.arange(n_outcomes)[:, None], measurement_indices[None, :]].set(y_bits)
    y_vals_full = (y_bits_full @ full_powers).astype(jnp.int32)
    
    # Get alpha bits in full nq-bit space
    alpha_bits_full = alpha.astype(jnp.int32)
    
    # XOR y_bits_full with alpha_bits_full
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
    
    return accuracy


__all__ = [
    'get_kraus_operators',
    'get_kraus_operators_with_aliases',
    'generate_psi_F_vector',
    'trajectory_from_key_jit',
    'sample_trajectories',
    'sample_trajectories_batched',
    'group_trajectories_jax',
    'generate_efficient_noisy_samples_arrays',
    'generate_efficient_noisy_samples_named',
    'expand_grouped_to_megabatch_from_arrays',
    'expand_grouped_to_megabatch_named',
    'mcs_shadows_all_jit',
    'mcs_shadows_streaming_jit',
    'mcs_shadows_streaming_2stage_jit',
    'generate_efficient_noisy_samples',
    'generate_all_nps_checkpoints',
    'generate_all_nps_checkpoints_batched',
    'process_results_to_accuracy',
]
