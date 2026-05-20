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
        get_kraus_operators,
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
    measurement_results,
    qubit_indices,
    f_values,
    alpha_pattern,
    nq,
    num_qubits,
):
    """Compute shadow tomography accuracy from measurement results.
    
    Args:
        measurement_results: (shots, nq) array of measurement outcomes
        qubit_indices: indices of measured qubits
        f_values: (2**nq,) F function values
        alpha_pattern: (nq,) binary array indicating probed qubits
        nq: number of qubits
        num_qubits: total number of qubits (may differ from nq in mapped scenarios)
        
    Returns:
        Scalar accuracy value
    """
    shots = measurement_results.shape[0]

    # Convert to numpy first — handles both numpy and JAX arrays
    import jax
    bitstrings = np.asarray(jax.device_get(measurement_results)).astype(np.int32)

    # Compute parity-based accuracy: for each measurement outcome,
    # compute (-1)^(f · b) where b is the measurement bitstring
    f_array = np.asarray(f_values, dtype=np.int32)
    alpha_array = np.asarray(alpha_pattern, dtype=np.int32)
    
    # Compute f dot product with bitstring for each shot
    # Only include qubits in the alpha pattern
    alpha_indices = np.where(alpha_array > 0)[0]
    
    if len(alpha_indices) == 0:
        # If no qubits in alpha pattern, accuracy is 1
        return 1.0
    
    # Extract f values and bitstrings for alpha qubits
    f_alpha = f_array[alpha_indices]
    bitstrings_alpha = bitstrings[:, alpha_indices]
    
    # Compute parity: f · b mod 2
    parities = np.sum(f_alpha[np.newaxis, :] * bitstrings_alpha, axis=1) % 2
    
    # Compute (-1)^parity
    signs = (-1.0) ** parities
    
    # Accuracy is mean of signs
    accuracy = np.mean(signs)
    
    return float(accuracy)


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
