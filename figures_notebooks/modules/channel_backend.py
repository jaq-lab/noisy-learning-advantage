
import numpy as np
import numba
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
import tensorcircuit as tc
from tensorcircuit import backend

# ======================================================================
# SECTION 1: NOISE CHANNEL DEFINITIONS
# ======================================================================

def get_kraus_operators(channel_config: Dict) -> List[np.ndarray]:
    """
    Returns the single-qubit Kraus operators for a given channel as NumPy arrays.

    Args:
        channel_config (Dict): A dictionary specifying the channel type and strength.
            Supported types: 'dephasing', 'bit_flip', 'thermal'.

    Returns:
        List[np.ndarray]: A list of Kraus operators.
    """
    channel_type = channel_config.get('type')
    p = channel_config.get('strength', 0.0)

    if channel_type == 'dephasing':
        # Models loss of quantum information without loss of energy.
        # K0: No error, K1: Z-error
        K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=np.complex128)
        K1 = np.array([[0, 0], [0, np.sqrt(p)]], dtype=np.complex128)
        return [K0, K1] 

    elif channel_type == 'bit_flip':
        # Models a bit-flip error (X-error).
        # K0: No error, K1: X-error
        K0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]], dtype=np.complex128)
        K1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]], dtype=np.complex128)
        return [K0, K1]

    elif channel_type == 'relaxation':
        # Models thermal relaxation (Generalized Amplitude Damping).
        # 'strength' (gamma) is the decay probability |1> -> |0|.
        gamma = p 
        # 'p_exc' is the equilibrium probability of being in the |1> state.
        p_exc = channel_config.get('p_exc', 0.0)
        
        # Define the four Kraus operators directly
        K0 = np.sqrt(1 - p_exc) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
        K1 = np.sqrt(1 - p_exc) * np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
        K2 = np.sqrt(p_exc) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]], dtype=np.complex128)
        K3 = np.sqrt(p_exc) * np.array([[0, 0], [np.sqrt(gamma), 0]], dtype=np.complex128)
        
        return [K0, K1, K2, K3]
        
    else:
        raise ValueError(f"Channel type '{channel_type}' is not supported.")

# ======================================================================
# SECTION 2: CORE STATE GENERATION
# ======================================================================

@numba.jit(nopython=True)
def _generate_psi_F_vector_numba(F_list_vals, N_val, dim_val):
    """Helper Numba function to generate the state vector psi_F."""
    psi_F = np.empty(dim_val, dtype=np.complex128)
    for k_idx in range(dim_val):
        psi_F[k_idx] = (-1)**F_list_vals[k_idx]
    norm_psi_F = np.linalg.norm(psi_F)
    return psi_F / norm_psi_F if norm_psi_F > 1e-9 else psi_F

def generate_psi_F_vector(F_list_vals, N_val):
    """Generates the normalized state vector psi_F = sum_k (-1)^F_k |k> / sqrt(2^N)."""
    dim_val = 2**N_val
    return _generate_psi_F_vector_numba(F_list_vals, N_val, dim_val)

# ======================================================================
# SECTION 3: SIMULATION ENGINES
# ======================================================================

# --- Helpers for Quantum Trajectory Mode ---

@numba.njit
def _apply_local_op(psi, op, nq, shape, axes_to_front, axes_to_back):
    """Applies a local 2x2 operator using pre-calculated permutation tuples."""
    tensor = psi.reshape(shape)
    tensor_moved = tensor.transpose(axes_to_front)
    tensor_matrix = tensor_moved.copy().reshape(2, 2**(nq - 1))
    new_tensor_matrix = op @ tensor_matrix
    new_tensor_moved = new_tensor_matrix.reshape(shape)
    final_tensor = new_tensor_moved.transpose(axes_to_back)
    return final_tensor.flatten()

def _run_single_trajectory(psi_initial, nq, kraus_ops_np):
    """Simulates one full quantum trajectory."""
    psi_current = np.copy(psi_initial)
    error_occurred = False
    shape_tuple = (2,) * nq
    for i in range(nq):
        axes_list = list(range(nq))
        axes_list.pop(i); axes_list.insert(0, i)
        axes_to_front = tuple(axes_list)
        axes_to_back = tuple(np.argsort(axes_to_front))
        outcomes = [_apply_local_op(psi_current, k_op, nq, shape_tuple, axes_to_front, axes_to_back) for k_op in kraus_ops_np]
        probs = np.array([np.real(np.vdot(o, o)) for o in outcomes])
        probs_sum = np.sum(probs)
        if probs_sum > 1e-9: probs /= probs_sum 
        else: probs = np.ones(len(kraus_ops_np)) / len(kraus_ops_np)
        chosen_idx = np.random.choice(len(kraus_ops_np), p=probs)
        if chosen_idx != 0: error_occurred = True # Assumes K0 is always the "no error" operator
        norm_factor = np.sqrt(np.real(np.vdot(outcomes[chosen_idx], outcomes[chosen_idx])))
        if norm_factor > 1e-9: psi_current = outcomes[chosen_idx] / norm_factor
        else: psi_current = outcomes[chosen_idx]
    return psi_current, error_occurred

# --- The Main Dual-Mode Simulation Function ---

def generate_efficient_noisy_samples(
    fs: np.ndarray, nq: int, channel_config: Dict,
    num_samples: int, group_noisy_states: bool = True,
    return_density_matrix: bool = False,
    return_prob: bool = True
) -> Union[Dict, np.ndarray]:
    """
    Generates noisy states from an initial state vector using one of two methods.
    """
    # Get Kraus operators in NumPy format first, as both modes can use them.
    kraus_ops_np = get_kraus_operators(channel_config)
    psi_initial = generate_psi_F_vector(fs, nq)
    # --- Path 1: Density Matrix Evolution ---
    if return_density_matrix:

        rho_initial_np = np.outer(psi_initial, np.conjugate(psi_initial))
        rho_initial_tc = tc.array_to_tensor(rho_initial_np)
        
        # Convert NumPy Kraus ops to TensorCircuit Tensors
        kraus_ops_tc = [tc.array_to_tensor(k) for k in kraus_ops_np]

        dmc = tc.DMCircuit(nq, dminputs=rho_initial_tc)
        
        #for i in tqdm(range(nq), desc="Applying noisy channels"):
        for i in range(nq):
            dmc.general_kraus(kraus_ops_tc, i)
        
        final_rho_tensor = dmc.state()
        final_rho_numpy = backend.numpy(final_rho_tensor)
        return final_rho_numpy

    # --- Path 2: Quantum Trajectory Simulation ---
    else:
        # Theoretical probability of no error occurring on any qubit
        prob_no_error = np.real(np.vdot(kraus_ops_np[0].flatten(), kraus_ops_np[0].flatten())) ** nq/2**nq

        n_ideal = int(round(num_samples * prob_no_error))
        n_noisy = num_samples - n_ideal

        raw_noisy_realizations = []
        if n_noisy > 0:
            for _ in tqdm(range(n_noisy), desc="Sampling Noisy Realizations"):
                while True:
                    final_state, had_error = _run_single_trajectory(psi_initial, nq, kraus_ops_np)
                    if had_error:
                        raw_noisy_realizations.append(final_state)
                        break
        
        final_noisy_output: list

        # check the same state vector is not counted multiple times
        if group_noisy_states and n_noisy > 0:
            unique_noisy_states_map = {}
            for state_vec in raw_noisy_realizations:
                state_key = state_vec.tobytes()
                if state_key in unique_noisy_states_map:
                    if return_prob:
                        unique_noisy_states_map[state_key]['prob'] += 1 / num_samples
                    unique_noisy_states_map[state_key]['multiplicity'] += 1
                else:
                    if return_prob:
                        unique_noisy_states_map[state_key] = {"state_vector": state_vec, "prob": 1 / num_samples, "multiplicity": 1}
                    else:
                        unique_noisy_states_map[state_key] = {"state_vector": state_vec, "multiplicity": 1}
            final_noisy_output = list(unique_noisy_states_map.values())
        else:
            final_noisy_output = [{"state_vector": vec, "multiplicity": 1, "prob": 1 / num_samples} for vec in raw_noisy_realizations]

        return {
                "ideal_state": {"state_vector": psi_initial, "multiplicity": n_ideal, "prob": n_ideal / num_samples},
                "noisy_realizations": final_noisy_output
            }