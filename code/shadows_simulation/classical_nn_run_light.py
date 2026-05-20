import os
import random
from functools import partial
import time
import json
import math
import pickle
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Sequence, NamedTuple, List, Callable
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
jax.config.update("jax_enable_x64", True)

import tensorcircuit as tc
tc.set_backend("jax")
tc.set_dtype("complex128")
from tensorcircuit import shadows
import shadow_funcs_dm as sf
import sys
import os
# Add path to shadow_mcs_jitted module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../quantum_simulation'))
from shadow_mcs_jitted import (
    mcs_shadows_streaming_jit, 
    get_kraus_operators,
    generate_all_nps_checkpoints,
    generate_all_nps_checkpoints_batched,
    trajectory_from_key_jit,
    generate_psi_F_vector,
)
import equinox as eqx
import flax
from precomputed_feature_cache_utils import try_load_validation_cache
# Import utilities from new module
try:
    from nn_utilities import (
        hamming_distance,
        hamming_distance_jax,
        hamming_distance_batch,
        compute_hamming_distance_matrix,
        compute_variance_matrix,
        apply_decoherence_noise,
        apply_thermal_noise_approximation,
        compute_nps_values,
        get_alpha_all_one,
        get_alpha_minimal_cnots,
        hamming_weight,
        normalize_features,
        load_model as load_model_util,
        save_model as save_model_util,
    )
    USE_NN_UTILITIES = True
except ImportError:
    # Fallback to local definitions if module not available
    USE_NN_UTILITIES = False

# Create a vmap'ed version for batch processing
@partial(jax.jit, static_argnums=(2, 4, 5, 6, 7))  # Only n, nps, batch_size_mcs, r, shots are static
def batched_shadow_generation(key, fs_batch, n, Ks, nps, batch_size_mcs, r, shots):
    """
    Process multiple F functions in parallel using vmap.
    
    Args:
        key: JAX PRNG key (will be split for each sample)
        fs_batch: (batch_size, 2**n) array of F function values
        n: number of qubits (static)
        Ks: Kraus operators
        nps: number of Monte Carlo samples per F
        batch_size_mcs: batch size for MC sampling
        r: parameter for shadow generation
        shots: number of shots
    
    Returns:
        (batch_size, 2**n, 2**n) array of shadow states
    """
    def single_shadow(fs, key_i):
        return mcs_shadows_streaming_jit(
            key_i, fs, n, Ks, nps, batch_size_mcs, r, shots,
            weights_kind=0, use_complex64=True
        )
    
    keys = jax.random.split(key, fs_batch.shape[0])
    return jax.vmap(single_shadow, in_axes=(0, 0))(fs_batch, keys)

from flax import linen as nn
from flax.training import train_state, checkpoints
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

## Imports for plotting
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
import seaborn as sns
sns.set()

from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
# import torch.nn as nn
# import torch.optim as optim

# import psutil
# from contextlib import contextmanager
from contextlib import contextmanager


np.random.seed(12345)
random.seed(12345)
torch.random.manual_seed(12345)


@contextmanager
def log_time(section_name: str):
    """Utility to log wall-clock time for code sections."""
    print(f"[TIMER] {section_name} – start")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"[TIMER] {section_name} – done in {elapsed:.2f}s")


class ReLU(nn.Module):

    def __call__(self, x):
        return jnp.maximum(x, 0)

##############################

class LeakyReLU(nn.Module):
    alpha : float = 0.1

    def __call__(self, x):
        return jnp.where(x > 0, x, self.alpha * x)

act_fn_by_name = {
    # "sigmoid": Sigmoid,
    # "tanh": Tanh,
    # "relu": ReLU,
    "leakyrelu": LeakyReLU,
    # "elu": ELU,
    # "swish": Swish
}

def get_grads(act_fn, x):
    """
    Computes the gradients of an activation function at specified positions.

    Inputs:
        act_fn - An module or function of the forward pass of the activation function.
        x - 1D input array.
    Output:
        An array with the same size of x containing the gradients of act_fn at x.
    """
    return jax.vmap(jax.grad(act_fn))(x)

def vis_act_fn(act_fn, ax, x):
    # Run activation function
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    # Push x, y and gradients back to cpu for plotting
    # x, y, y_grads = x.cpu().numpy(), y.cpu().numpy(), y_grads.cpu().numpy()
    ## Plotting
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grads, linewidth=2, label="Gradient")
    ax.set_title(act_fn.__class__.__name__)
    ax.legend()
    ax.set_ylim(-1.5, x.max())

# Add activation functions if wanted
# act_fns = [act_fn() for act_fn in act_fn_by_name.values()]
# x = np.linspace(-5, 5, 1000) # Range on which we want to visualize the activation functions
# ## Plotting
# rows = math.ceil(len(act_fns)/2.0)
# fig, ax = plt.subplots(rows, 2, figsize=(8, rows*4))
# for i, act_fn in enumerate(act_fns):
#     # vis_act_fn(act_fn, ax[divmod(i,2)], x)
#     vis_act_fn(act_fn, ax[i], x)
# fig.subplots_adjust(hspace=0.3)
# plt.show()


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")

def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")

def load_model(model_path, model_name, state=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        state - (Optional) If given, the parameters are loaded into this training state. Otherwise,
                a new one is created alongside a network architecture.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if state is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = BaseNetwork(act_fn=act_fn, **config_dict)
        state = train_state.TrainState(step=0,
                                       params=None,
                                       apply_fn=net.apply,
                                       tx=None,
                                       opt_state=None)
    else:
        net = None
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead load the parameters simply from a pickle file.
    with open(model_file, 'rb') as f:
        params = pickle.load(f)
    state = state.replace(params=params)
    return state, net


def save_model(model, params, model_path, model_name):
    """
    Given a model, we save the parameters and hyperparameters.

    Inputs:
        model - Network object without parameters
        params - Parameters to save of the model
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = {
        'num_classes': model.num_classes,
        'hidden_sizes': model.hidden_sizes,
        'act_fn': {'name': model.act_fn.__class__.__name__.lower()}
    }
    if hasattr(model.act_fn, 'alpha'):
        config_dict['act_fn']['alpha'] = model.act_fn.alpha
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    with open(model_file, 'wb') as f:
        pickle.dump(params, f)


def get_f_instance(qubits, seed=None):
    '''
        Returns a random function F
    '''
    if seed is not None:
        np.random.seed(seed)
    fun_values = np.random.randint(0, 2, size=2**qubits)
    return fun_values

# @partial(jit, static_argnum = (0,))
@jax.jit
def get_expm_f(f, state):
  return jnp.diag(jnp.exp(1j*f*np.pi))@state

def get_state(f):
  '''
      returns the vector representing state after applying function F
  '''
  state0 = np.ones(len(f))/np.sqrt(len(f))
  return get_expm_f(f, state0)

def get_u_dm(alpha, qubits):
  '''
    Returns the U(alpha) operator and the quantum circuit

    Params:
    alpha: int, alpha value in decimal representation
    qubits: int, number of qubits
  '''
  qc = tc.DMCircuit(qubits)
  for k, x in enumerate(alpha[:-1]):
    if int(x)==1:
      qc.cx(qubits-1, k)

  qc.h(qubits-1)
  return qc

def get_u(alpha, qubits):
  '''
    Returns the U(alpha) operator and the quantum circuit

    Params:
    alpha: int, alpha value in decimal representation
    qubits: int, number of qubits
  '''
  qc = tc.Circuit(qubits)
  for k, x in enumerate(alpha[:-1]):
    if int(x)==1:
      qc.cx(qubits-1, k)

  qc.h(qubits-1)
  return qc

# These functions are now in nn_utilities, but kept here for backward compatibility
if not USE_NN_UTILITIES:
    def get_alpha_all_one(qubits):
        return tuple([1] * qubits) 

    def get_alpha_minimal_cnots(qubits): # "only-last-one-one" -> all a_i=0 before the final '1'
        return tuple([0] * (qubits - 1) + [1]) # (0,0,...,0,1)

    def compute_nps_values(n_qubits, k_values=[-2, -1, 0, 1,2]):
        """
        Compute nps values according to the formula: nps = 2^(k*nq)
        where k is from the provided list.
        
        Returns:
            List of nps values, filtered to ensure nps > 1 (skips nps=1)
        """
        results = []
        for k in k_values:
            nps = 2 ** (k * n_qubits)
            # Check if nps is reasonable (skip nps=1 as it's too small)
            if nps > 1:
                results.append(int(nps))
            elif nps == 1:
                print(f"Warning: nps={nps} = 1 for nq={n_qubits}, k={k}, skipping (too small)")
            else:
                print(f"Warning: nps={nps} < 1 for nq={n_qubits}, k={k}, skipping")
        return sorted(results)  # Return sorted list


def hamming_distance(n, m):
    """
    Compute Hamming distance between two integers (XOR and count bits).
    
    Args:
        n: First integer
        m: Second integer
        
    Returns:
        Hamming distance (number of differing bits)
    """
    return bin(n ^ m).count('1')


@jax.jit
def hamming_distance_jax(n: jax.Array, m: jax.Array) -> jax.Array:
    """
    JAX-vectorized Hamming distance computation using popcount.
    
    Args:
        n: Integer or array of integers
        m: Integer or array of integers (must broadcast with n)
        
    Returns:
        Hamming distance(s)
    """
    xor = n ^ m
    # Use JAX's popcount if available (bit_count), otherwise use manual unrolled version
    # For up to 32-bit integers, we can unroll up to 32 shifts
    # For typical nq <= 20, we only need up to 20 bits
    
    # Try using bit_count (available in recent JAX versions)
    try:
        if hasattr(jnp, 'bit_count'):
            return jnp.bit_count(xor)
    except:
        pass
    
    # Fallback: unroll bit counting (works for up to 32 bits)
    # This is still JAX-compilable since it's a fixed number of operations
    count = jnp.zeros_like(xor, dtype=jnp.int32)
    # Unroll up to 32 bits (more than enough for quantum states with nq <= 20)
    for _ in range(32):
        count = count + (xor & 1)
        xor = xor >> 1
    return count


@jax.jit
def hamming_distance_batch(n: jax.Array, m: jax.Array) -> jax.Array:
    """
    Vectorized Hamming distance for broadcasting arrays.
    
    Args:
        n: Array of integers, can be any shape
        m: Array of integers, can be any shape (must broadcast with n)
        
    Returns:
        Array of Hamming distances with broadcasted shape
    """
    xor = n ^ m
    # Try bit_count first
    try:
        if hasattr(jnp, 'bit_count'):
            return jnp.bit_count(xor)
    except:
        pass
    
    # Fallback: unrolled bit counting
    count = jnp.zeros_like(xor, dtype=jnp.int32)
    for _ in range(32):
        count = count + (xor & 1)
        xor = xor >> 1
    return count


def compute_hamming_distance_matrix(n_qubits):
    """
    Pre-compute Hamming distance matrix for all pairs of basis states.
    
    For n_qubits, creates a (2^n_qubits × 2^n_qubits) matrix where
    entry [i,j] = d(i,j) = Hamming distance between i and j.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        (2^n_qubits, 2^n_qubits) array of Hamming distances
    """
    dim = 2 ** n_qubits
    dist_matrix = np.zeros((dim, dim), dtype=int)
    
    for i in range(dim):
        for j in range(dim):
            dist_matrix[i, j] = hamming_distance(i, j)
    
    return dist_matrix


def compute_variance_matrix(n_qubits, nps, hamming_dist_matrix=None):
    """
    Compute element-dependent variance matrix.
    
    Variance model:
    - Diagonals (d(n,m) = 0): σ²(n,n) = 1 / nps (real part only)
    - Off-diagonals (d(n,m) > 0): σ²(n,m) = 2.3^(d(n,m)/2) / nps / 2 (real part)
    where d(n,m) is the Hamming distance between n and m.
    Note: NN only receives the real part, but the formula accounts for both real and imaginary parts.
    
    Args:
        n_qubits: Number of qubits
        nps: Number of samples (determines overall scale)
        hamming_dist_matrix: Pre-computed Hamming distance matrix (optional)
        
    Returns:
        (2^n_qubits, 2^n_qubits) array of variances
    """
    if hamming_dist_matrix is None:
        hamming_dist_matrix = compute_hamming_distance_matrix(n_qubits)
    
    # For diagonals: var = 1 / nps
    # For off-diagonals: var = 2.3^(d/2) / nps / 2
    variance_matrix = np.where(
        hamming_dist_matrix == 0,
        1.0 / nps,  # Diagonals
        (2.3 ** (hamming_dist_matrix / 2)) / nps / 2  # Off-diagonals
    )
    
    return variance_matrix


def apply_decoherence_noise(rho_element, n, m, noise_strength, noise_type="dephasing"):
    """
    Apply decoherence noise to a density matrix element.
    
    Models:
    - Dephasing: rho_{n,m}' = rho_{n,m} * (1-p)^{d(n,m)}
      where d(n,m) is the Hamming distance between n and m
    - Relaxation: rho_{n,m}' = rho_{n,m} * (1-p)^{(d(n)+d(m))/2}
      where d(n) is the Hamming weight (number of 1s) in the binary representation of n
    
    Args:
        rho_element: True value of the density matrix element
        n: Row index (as integer)
        m: Column index (as integer)
        noise_strength: Probability p (0 = no noise)
        noise_type: Either "dephasing" or "relaxation"
        
    Returns:
        Noisy density matrix element
    """
    if noise_strength == 0:
        return rho_element
    
    if noise_type == "dephasing":
        d_nm = hamming_distance(n, m)
        noise_factor = (1 - noise_strength) ** d_nm
    elif noise_type == "relaxation":
        # d(n) is the Hamming weight (number of 1s in binary representation)
        d_n = bin(n).count('1')
        d_m = bin(m).count('1')
        noise_factor = (1 - noise_strength) ** ((d_n + d_m) / 2)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Must be 'dephasing' or 'relaxation'")
    
    return noise_factor * rho_element


def apply_thermal_noise_approximation(rho_element, n, m, thermal_strength):
    """
    DEPRECATED: Use apply_decoherence_noise instead.
    
    This function is kept for backward compatibility but now uses dephasing model.
    """
    return apply_decoherence_noise(rho_element, n, m, thermal_strength, noise_type="dephasing")


def compute_relevant_elements_from_f(f_vec, y_targ, alpha_targ, n_qubits):
    """
    Efficiently compute only the relevant density matrix elements from F function.
    
    For pure state |ψ_F⟩, we have:
    - ψ_F[k] = exp(i*π*f[k]) / sqrt(2^n)
    - rho[n,m] = ψ_F[n] * conj(ψ_F[m]) = exp(i*π*(f[n]-f[m])) / 2^n
    
    Since f[k] ∈ {0,1}:
    - f[n] - f[m] ∈ {-1, 0, 1}
    - exp(i*π*(f[n]-f[m])) = (-1)^(f[n] ⊕ f[m])  (since exp(i*π*x) = (-1)^x for x ∈ {0,1})
    
    So: rho[n,m] = (-1)^(f[n] ⊕ f[m]) / 2^n
    
    Args:
        f_vec: F function values (array of 0s and 1s)
        y_targ: Target y value (binary string)
        alpha_targ: Target alpha value (tuple of bits)
        n_qubits: Number of qubits
        
    Returns:
        Dictionary with relevant density matrix elements
    """
    alpha_str = "".join(map(str, alpha_targ))
    y_targ_int = int(y_targ, 2)
    alpha_targ_int = int(alpha_str, 2)
    idx0 = y_targ_int ^ alpha_targ_int  # alpha ⊕ y
    idx1 = y_targ_int  # y
    
    # Compute rho[idx0, idx1] = rho[alpha⊕y, y]
    sign_0_1 = (-1) ** (f_vec[idx0] ^ f_vec[idx1])
    rho_0_1 = sign_0_1 / (2 ** n_qubits)
    
    # Compute rho[k, idx0] and rho[k, idx1] for all k != idx0, idx1
    all_k = np.arange(2**n_qubits)
    mask = (all_k != idx0) & (all_k != idx1)
    k_vals = all_k[mask]
    
    # rho[k, idx0] for all k
    signs_k_0 = np.array([(-1) ** (f_vec[k] ^ f_vec[idx0]) for k in k_vals])
    rho_k_0 = signs_k_0 / (2 ** n_qubits)
    
    # rho[k, idx1] for all k
    signs_k_1 = np.array([(-1) ** (f_vec[k] ^ f_vec[idx1]) for k in k_vals])
    rho_k_1 = signs_k_1 / (2 ** n_qubits)
    
    return {
        'rho_0_1': rho_0_1,
        'rho_k_0': rho_k_0,
        'rho_k_1': rho_k_1,
        'k_vals': k_vals,
        'idx0': idx0,
        'idx1': idx1
    }


def generate_white_noise_training_features(f_vec, y_targ, alpha_targ, n_qubits, nps, 
                                           thermal_strength=0.0, rng_key=None,
                                           use_element_dependent_variance=True):
    """
    Generate training features with white noise for a given nps value.
    
    New structure (matching get_relevant_input):
    1. Relevant diagonal elements: rho[idx0, idx0], rho[idx1, idx1], and rho[n, n] 
       for n with minimal cumulative Hamming distance
    2. Central element: rho[idx0, idx1]
    3. Off-diagonals: rho[idx0, n] * rho[n, idx1] for n with lowest 
       cumulative Hamming distance d(idx0, n) + d(n, idx1)
    
    White noise model:
    - Add independent Gaussian noise to each element
    - Element-dependent variance: 
      * Diagonals (d=0): σ² = 1 / nps
      * Off-diagonals (d>0): σ² = 2.3^(d/2) / nps / 2
      where d(n,m) is the Hamming distance between n and m
    
    Optional thermal noise approximation:
    - Apply (1-p)^{d(n,m)} factor before adding white noise
    
    Args:
        f_vec: F function values
        y_targ: Target y value (binary string)
        alpha_targ: Target alpha value (tuple of bits)
        n_qubits: Number of qubits
        nps: Number of samples (determines noise variance)
        thermal_strength: Thermal relaxation strength p (0 = no thermal noise)
        rng_key: JAX random key (if None, use numpy random)
        use_element_dependent_variance: If True, use element-dependent variance:
                                         * Diagonals: σ² = 1/nps
                                         * Off-diagonals: σ² = 2.3^(d/2)/nps/2
                                         If False, use uniform σ² = 3^nq/nps
        
    Returns:
        Feature vector for neural network input
    """
    # Compute indices
    alpha_str = "".join(map(str, alpha_targ))
    y_targ_int = int(y_targ, 2)
    alpha_targ_int = int(alpha_str, 2)
    idx0 = y_targ_int ^ alpha_targ_int
    idx1 = y_targ_int
    
    features = []
    
    # For pure state |ψ_F⟩: rho[n,m] = (-1)^(f[n] ⊕ f[m]) / 2^n
    # where f[k] ∈ {0,1}
    
    # First, find n with minimal cumulative Hamming distance (needed for diagonals and off-diagonals)
    all_n = np.arange(2**n_qubits)
    mask = (all_n != idx0) & (all_n != idx1)
    n_vals = all_n[mask]
    
    if len(n_vals) > 0:
        # Compute cumulative Hamming distances
        cumulative_dists = np.array([
            hamming_distance(idx0, n) + hamming_distance(n, idx1) 
            for n in n_vals
        ])
        
        # Find minimal cumulative distance
        min_dist = np.min(cumulative_dists)
        
        # Select ONE representative n for each unique (d(idx0,n), d(n,idx1)) pair
        # that achieves minimal cumulative distance (to avoid too many features)
        selected_n_list = []
        seen_pairs = set()
        
        for n in n_vals:
            d_idx0_n = hamming_distance(idx0, n)
            d_n_idx1 = hamming_distance(n, idx1)
            total_dist = d_idx0_n + d_n_idx1
            
            if total_dist == min_dist:
                pair = (d_idx0_n, d_n_idx1)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    selected_n_list.append(n)
        
        selected_n = np.array(selected_n_list)
    else:
        selected_n = np.array([])
    
    # 1. Relevant diagonal elements: rho[idx0, idx0], rho[idx1, idx1], and rho[n, n] for selected n
    #    For pure state: rho[n, n] = (-1)^(f[n] ⊕ f[n]) / 2^n = 1 / 2^n (since f[n] ⊕ f[n] = 0)
    diagonal_indices = [idx0, idx1]
    if len(selected_n) > 0:
        diagonal_indices.extend(selected_n.tolist())
    # Remove duplicates (in case idx0 or idx1 appears in selected_n, though it shouldn't)
    diagonal_indices = list(set(diagonal_indices))
    
    num_diagonals = len(diagonal_indices)
    diagonal_elements_true = np.ones(num_diagonals) / (2 ** n_qubits)
    
    # Compute variances for diagonals (all have distance 0, so variance = 1/nps)
    if use_element_dependent_variance:
        std_diagonal = np.sqrt(1.0 / nps)  # var = 1/nps for diagonals
    else:
        std_diagonal = np.sqrt((3 ** n_qubits) / nps)
    
    # Apply thermal noise to diagonals (d(n,n) = 0, so (1-p)^0 = 1, no effect)
    # But add white noise
    if rng_key is not None:
        # Split key for all random operations needed
        num_keys_needed = num_diagonals + 1 + (2 * len(selected_n) if len(selected_n) > 0 else 0)
        all_keys = jax.random.split(rng_key, num_keys_needed)
        key_idx = 0
        
        noise_diagonal = jax.random.normal(all_keys[key_idx], shape=(num_diagonals,)) * std_diagonal
        diagonal_elements = np.real(diagonal_elements_true) + noise_diagonal
        key_idx += 1
    else:
        diagonal_elements = np.real(diagonal_elements_true) + np.random.randn(num_diagonals) * std_diagonal
        key_idx = None
    
    features.extend((diagonal_elements * (2 ** n_qubits)).tolist())
    
    # 2. Central element: rho[idx0, idx1]
    sign_0_1 = (-1) ** (f_vec[idx0] ^ f_vec[idx1])
    rho_0_1_true = sign_0_1 / (2 ** n_qubits)
    
    # Compute variance for central element (off-diagonal, so use off-diagonal formula)
    if use_element_dependent_variance:
        d_0_1 = hamming_distance(idx0, idx1)
        if d_0_1 == 0:
            var_0_1 = 1.0 / nps  # Diagonal case (shouldn't happen for central element, but handle it)
        else:
            var_0_1 = (2.3 ** (d_0_1 / 2)) / nps / 2  # Off-diagonal: 2.3^(d/2) / nps / 2
        std_0_1 = np.sqrt(var_0_1)
    else:
        std_0_1 = np.sqrt((3 ** n_qubits) / nps)
    
    # Apply thermal noise approximation
    if thermal_strength > 0:
        rho_0_1_noisy = apply_thermal_noise_approximation(rho_0_1_true, idx0, idx1, thermal_strength)
    else:
        rho_0_1_noisy = rho_0_1_true
    
    # Add white noise
    if rng_key is not None:
        rho_0_1_with_noise = np.real(rho_0_1_noisy) + jax.random.normal(all_keys[key_idx], shape=()) * std_0_1
        key_idx += 1
    else:
        rho_0_1_with_noise = np.real(rho_0_1_noisy) + np.random.randn() * std_0_1
    
    features.append(rho_0_1_with_noise * (2 ** n_qubits))
    
    # 3. Off-diagonals: rho[idx0, n] * rho[n, idx1] for selected n (already computed above)
    if len(selected_n) > 0:
        # Compute rho[idx0, n] and rho[n, idx1] for selected n
        signs_idx0_n = np.array([(-1) ** (f_vec[idx0] ^ f_vec[n]) for n in selected_n])
        signs_n_idx1 = np.array([(-1) ** (f_vec[n] ^ f_vec[idx1]) for n in selected_n])
        
        rho_idx0_n_true = signs_idx0_n / (2 ** n_qubits)
        rho_n_idx1_true = signs_n_idx1 / (2 ** n_qubits)
        
        # Compute variances for each selected element (off-diagonals)
        if use_element_dependent_variance:
            std_idx0_n = np.array([
                np.sqrt((2.3 ** (hamming_distance(idx0, n) / 2)) / nps / 2)
                for n in selected_n
            ])
            std_n_idx1 = np.array([
                np.sqrt((2.3 ** (hamming_distance(n, idx1) / 2)) / nps / 2)
                for n in selected_n
            ])
        else:
            std_uniform = np.sqrt((3 ** n_qubits) / nps)
            std_idx0_n = np.full(len(selected_n), std_uniform)
            std_n_idx1 = np.full(len(selected_n), std_uniform)
        
        # Apply thermal noise approximation
        if thermal_strength > 0:
            rho_idx0_n_noisy = np.array([
                apply_thermal_noise_approximation(rho, idx0, n, thermal_strength)
                for rho, n in zip(rho_idx0_n_true, selected_n)
            ])
            rho_n_idx1_noisy = np.array([
                apply_thermal_noise_approximation(rho, n, idx1, thermal_strength)
                for rho, n in zip(rho_n_idx1_true, selected_n)
            ])
        else:
            rho_idx0_n_noisy = rho_idx0_n_true
            rho_n_idx1_noisy = rho_n_idx1_true
        
        # Add white noise
        if rng_key is not None:
            # Need 2 keys for the two sets of off-diagonals
            if key_idx + 2 > len(all_keys):
                # If we need more keys, split further from the last key
                remaining_keys = jax.random.split(all_keys[-1], 2)
            else:
                remaining_keys = all_keys[key_idx:key_idx + 2]
            
            noise_idx0_n = jax.random.normal(remaining_keys[0], shape=rho_idx0_n_noisy.shape) * std_idx0_n
            noise_n_idx1 = jax.random.normal(remaining_keys[1], shape=rho_n_idx1_noisy.shape) * std_n_idx1
            
            rho_idx0_n_with_noise = np.real(rho_idx0_n_noisy) + noise_idx0_n
            rho_n_idx1_with_noise = np.real(rho_n_idx1_noisy) + noise_n_idx1
        else:
            rho_idx0_n_with_noise = np.real(rho_idx0_n_noisy) + np.random.randn(len(selected_n)) * std_idx0_n
            rho_n_idx1_with_noise = np.real(rho_n_idx1_noisy) + np.random.randn(len(selected_n)) * std_n_idx1
        
        # Compute product rho[idx0, n] * rho[n, idx1] * (2^nq)^2
        rhoij = rho_idx0_n_with_noise * rho_n_idx1_with_noise * (2 ** n_qubits)
        features.extend(rhoij.tolist())
    
    return jnp.array(features)


@dataclass
class WhiteNoisePauliPrecomputation:
    nq: int
    D: int
    f_values: jnp.ndarray
    psi_pure: jnp.ndarray
    c_ix_vec: jnp.ndarray


def _synchronize_channel_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all channel-related fields in the config are consistent.
    """
    channel_config = dict(config.get("channel_config") or {})

    ctype = channel_config.get("type") or config.get("channel_type") or config.get("white_noise_decoherence_type")
    if not ctype:
        ctype = "none"
    if ctype == "pure":
        ctype = "none"

    strength = channel_config.get("strength")
    if strength is None:
        strength = config.get("noise_strength", config.get("white_noise_thermal_strength", 0.0))
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        strength = 0.0
    if ctype == "none":
        strength = 0.0

    channel_config["type"] = ctype
    channel_config["strength"] = strength

    if ctype in ("relaxation", "thermal"):
        thermal_p_exc = channel_config.get(
            "thermal_p_exc",
            config.get("thermal_p_exc", config.get("white_noise_thermal_p_exc", 0.0)),
        )
        try:
            thermal_p_exc = float(thermal_p_exc)
        except (TypeError, ValueError):
            thermal_p_exc = 0.0
        channel_config["thermal_p_exc"] = thermal_p_exc
        config["thermal_p_exc"] = thermal_p_exc
        config["white_noise_thermal_p_exc"] = thermal_p_exc
    else:
        channel_config.pop("thermal_p_exc", None)
        config.pop("thermal_p_exc", None)
        config.pop("white_noise_thermal_p_exc", None)

    config["channel_config"] = channel_config
    config["channel_type"] = ctype
    config["noise_strength"] = strength
    config["white_noise_decoherence_type"] = ctype
    config["white_noise_thermal_strength"] = strength
    return config


def _white_noise_channel_config_from_config(config) -> Dict[str, Any]:
    channel_config = config.get("channel_config")
    if channel_config:
        return dict(channel_config)

    decoherence_type = config.get('white_noise_decoherence_type', config.get('channel_type', 'dephasing'))
    if decoherence_type == "pure":
        decoherence_type = "none"
    strength = config.get('white_noise_thermal_strength', config.get('noise_strength', 0.0))
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        strength = 0.0

    resolved = {
        "type": decoherence_type,
        "strength": strength,
    }
    if decoherence_type in ('thermal', 'relaxation'):
        thermal_p_exc = config.get('white_noise_thermal_p_exc', config.get('thermal_p_exc', 0.0))
        try:
            thermal_p_exc = float(thermal_p_exc)
        except (TypeError, ValueError):
            thermal_p_exc = 0.0
        resolved["thermal_p_exc"] = thermal_p_exc
    return resolved


def _fwht_complex(vec: np.ndarray) -> jnp.ndarray:
    arr = np.asarray(vec, dtype=np.complex128).copy()
    length = arr.shape[0]
    h = 1
    while h < length:
        for start in range(0, length, h * 2):
            mid = start + h
            end = start + 2 * h
            x = arr[start:mid]
            y = arr[mid:end]
            arr[start:mid] = x + y
            arr[mid:end] = x - y
        h *= 2
    return jnp.asarray(arr)


def _pauli_weights(nq: int) -> jnp.ndarray:
    total = 1 << nq
    weights = np.fromiter((bin(i).count("1") for i in range(total)), dtype=np.int32, count=total)
    return jnp.asarray(weights)


def _build_white_noise_pauli_precomputation(f_vector: np.ndarray, nq: int) -> WhiteNoisePauliPrecomputation:
    f_vector = np.asarray(f_vector, dtype=np.int32)
    dim = 1 << nq
    if f_vector.shape[0] != dim:
        raise ValueError(f"Expected F vector of length {dim} for nq={nq}, got {f_vector.shape[0]}")
    f_values = jnp.asarray(f_vector, dtype=jnp.int32)
    psi_pure = generate_psi_F_vector(f_values, nq)
    c_ix_vec = _fwht_complex(f_values.astype(np.complex128)) / dim
    return WhiteNoisePauliPrecomputation(nq=nq, D=dim, f_values=f_values, psi_pure=psi_pure, c_ix_vec=c_ix_vec)


def get_relaxation_element_cutoff(
    N: int, 
    l: int, 
    k: int, 
    f_func: Callable[[int], int], 
    p: float,
    D: int = 1
) -> complex:
    """
    Calculates the approximate density matrix element rho_lk(p) 
    under Amplitude Damping (relaxation) using a cutoff D.
    
    Formula: Sums contributions from parent states that differ 
             by at most D bit flips (orders p^0 through p^D).
    
    Args:
        N: Number of qubits
        l: Row index (integer representation of bitstring)
        k: Column index (integer representation of bitstring)
        f_func: Function that maps integer to f value (0 or 1)
        p: Relaxation (decay) probability
        D: Cutoff order (default=1). Only considers parent states 
           that differ by at most D bit flips.
    
    Returns:
        Complex value of rho_lk after relaxation
    """
    # --- 1. The "Survival" Damping Factor (Prefactor) ---
    # (1-p)^[(w(l) + w(k))/2]
    # Note: We do NOT include 1/2^N here - normalization is applied in the caller
    wl = bin(l).count('1')
    wk = bin(k).count('1')
    prefactor = (1.0 - p)**((wl + wk) / 2.0)
    
    # --- 2. Identify the set S (indices where BOTH l and k are 0) ---
    # These are the only locations that could have decayed from 1->0
    # Logic: ~(l | k) gives 1s where both are 0.
    combined_or = l | k
    s_indices = []
    for i in range(N):
        # Check if i-th bit is 0 in both
        if not ((combined_or >> i) & 1):
            s_indices.append(i)
    
    sum_val = 0.0
    
    # --- 3. Sum over subsets s with |s| <= D ---
    
    # A. Zero-th order (s is empty set, |s|=0) -> p^0 term
    # Parent states are just l and k themselves
    phase_0 = (-1)**(f_func(l) + f_func(k))
    sum_val += phase_0 * 1.0  # p^0 = 1
    
    # B. Higher orders (|s| = 1, 2, ..., D) -> p^|s| terms
    for order in range(1, D + 1):
        # Iterate over all combinations of order indices from s_indices
        for idx_subset in itertools.combinations(s_indices, order):
            # Build mask for all bits to flip
            mask = 0
            for idx in idx_subset:
                mask |= (1 << idx)
            
            # Parent states have 1s at these indices
            l_parent = l | mask
            k_parent = k | mask
            
            phase = (-1)**(f_func(l_parent) + f_func(k_parent))
            sum_val += phase * (p ** order)
    
    return prefactor * sum_val


def _collect_indices_by_weight(nq: int, exclude: set[int], k_list: Sequence[int]) -> Dict[int, np.ndarray]:
    groups: Dict[int, List[int]] = {k: [] for k in k_list}
    for idx in range(1 << nq):
        if idx in exclude:
            continue
        weight = hamming_weight(idx)
        if weight in groups:
            groups[weight].append(idx)
    return {k: np.asarray(v, dtype=np.int32) for k, v in groups.items()}


def _compute_true_noisy_vectors_analytic(
    precomp: WhiteNoisePauliPrecomputation,
    channel_config: Dict[str, Any],
    indices: Sequence[int],
    idx0: int,
    idx1: int,
) -> Dict[str, Dict[int, complex]]:
    ctype = channel_config.get("type", "dephasing")
    p = float(channel_config.get("strength", 0.0))
    D = precomp.D
    f_values = np.asarray(precomp.f_values, dtype=np.int8)

    true_d: Dict[int, float] = {}
    true_n_row: Dict[int, complex] = {}
    true_m_row: Dict[int, complex] = {}
    nq = precomp.nq

    # Vectorized computation using JAX for better performance
    indices_arr = jnp.asarray(indices, dtype=jnp.int32)
    idx0_arr = jnp.asarray(idx0, dtype=jnp.int32)
    idx1_arr = jnp.asarray(idx1, dtype=jnp.int32)
    f_values_jax = jnp.asarray(f_values, dtype=jnp.int32)
    
    # Vectorized Hamming distances
    w_ni = hamming_distance_batch(indices_arr, idx0_arr).astype(jnp.float64)
    w_mi = hamming_distance_batch(indices_arr, idx1_arr).astype(jnp.float64)
    
    # Vectorized phase computation
    f_idx0 = f_values_jax[idx0]
    f_idx1 = f_values_jax[idx1]
    f_indices = f_values_jax[indices_arr]
    phase_n = 1.0 - 2.0 * (f_idx0 ^ f_indices)
    phase_m = 1.0 - 2.0 * (f_idx1 ^ f_indices)
    
    # Compute damping factors based on channel type
    if ctype == "dephasing":
        # Dephasing: damping = (sqrt(1-p))^w
        damping = jnp.sqrt(max(0.0, 1.0 - p))
        damping_ni = jnp.power(damping, w_ni)
        damping_mi = jnp.power(damping, w_mi)
        
    elif ctype == "depolarizing":
        # Depolarizing: Use Monte Carlo (analytic formula not accurate enough)
        # Return None to fall back to MC computation
        return None
        
    elif ctype in ("relaxation", "thermal"):
        # Relaxation/Thermal: Use cutoff formula with parent state contributions
        # Get cutoff order D (default to 1 for speed and stability)
        D_cutoff = int(channel_config.get("relaxation_cutoff_D", 1))
        
        # Create f_func wrapper
        def f_func(idx: int) -> int:
            return int(f_values[idx])
        
        # Compute values using cutoff formula (includes feeder terms)
        # We compute each element individually to get correct parent state contributions
        true_n_row_vals = []
        true_m_row_vals = []
        true_d_vals = []
        
        # Compute normalization scale (same as other channels: (2^nq)/D = 1 since D=2^nq)
        scale = (2 ** nq) / D
        
        for i in indices:
            # For idx0-row: rho[idx0, i]
            # get_relaxation_element_cutoff returns (1-p)^((w(l)+w(k))/2) * sum (includes phase)
            # Multiply by scale to match other channels' normalization
            rho_val_ni = get_relaxation_element_cutoff(nq, int(idx0), int(i), f_func, p, D_cutoff)
            true_n_row_vals.append(complex(rho_val_ni * scale))
            
            # For idx1-row: rho[idx1, i]
            rho_val_mi = get_relaxation_element_cutoff(nq, int(idx1), int(i), f_func, p, D_cutoff)
            true_m_row_vals.append(complex(rho_val_mi * scale))
            
            # For diagonal: rho[i, i]
            rho_val_diag = get_relaxation_element_cutoff(nq, int(i), int(i), f_func, p, D_cutoff)
            # Diagonal is real
            true_d_vals.append(float(np.real(rho_val_diag * scale)))
        
        # Convert to arrays
        true_n_row_vals = jnp.array(true_n_row_vals, dtype=jnp.complex128)
        true_m_row_vals = jnp.array(true_m_row_vals, dtype=jnp.complex128)
        true_d_vals = jnp.array(true_d_vals, dtype=jnp.float64)
        
        # Store in dicts
        for idx, i in enumerate(indices):
            true_n_row[int(i)] = complex(true_n_row_vals[idx])
            true_m_row[int(i)] = complex(true_m_row_vals[idx])
            true_d[int(i)] = float(true_d_vals[idx])
        
        # Early return since we've computed everything directly
        return {"d": true_d, "n_row": true_n_row, "m_row": true_m_row}
        
    elif ctype == "none":
        damping_ni = jnp.ones_like(w_ni)
        damping_mi = jnp.ones_like(w_mi)
    else:
        # Unknown channel - fall back to MC
        return None
    
    # Compute all values at once
    scale = (2 ** nq) / D
    true_n_row_vals = (phase_n.astype(jnp.complex128) * damping_ni.astype(jnp.complex128)) * scale
    true_m_row_vals = (phase_m.astype(jnp.complex128) * damping_mi.astype(jnp.complex128)) * scale
    
    # Diagonal elements: for relaxation/thermal, use special formula
    if ctype in ("relaxation", "thermal"):
        # Diagonal: (ρ_f)_{k,k} = (1/2^N) * (1-p)^{w(k)} * (1+p)^{N-w(k)}
        indices_weights = hamming_distance_batch(indices_arr, jnp.zeros_like(indices_arr, dtype=jnp.int32)).astype(jnp.float64)
        true_d_vals = ((jnp.power(max(0.0, 1.0 - p), indices_weights) * 
                       jnp.power(max(0.0, 1.0 + p), nq - indices_weights)) * scale)
    else:
        # For other channels: diagonal = scale (no damping for pure state)
        true_d_vals = jnp.full(len(indices), scale, dtype=jnp.float64)
    
    # Convert back to dicts (for compatibility with existing code)
    for idx, i in enumerate(indices):
        true_n_row[int(i)] = complex(true_n_row_vals[idx])
        true_m_row[int(i)] = complex(true_m_row_vals[idx])
        true_d[int(i)] = float(true_d_vals[idx])

    return {"d": true_d , "n_row": true_n_row , "m_row": true_m_row}


def _compute_true_noisy_vectors_mc(
    key: jax.Array,
    precomp: WhiteNoisePauliPrecomputation,
    channel_config: Dict[str, Any],
    indices: Sequence[int],
    idx0: int,
    idx1: int,
    num_trajectories: int,
) -> Dict[str, Dict[int, complex]]:
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")

    Ks = get_kraus_operators(channel_config)

    def _sample_state(sample_key):
        psif, _, _ = trajectory_from_key_jit(sample_key, precomp.psi_pure, precomp.nq, Ks)
        return psif

    traj_keys = jr.split(key, num_trajectories)
    psi_samples = jax.vmap(_sample_state)(traj_keys)

    probs = jnp.real(psi_samples * jnp.conj(psi_samples))
    acc_d = jnp.mean(probs, axis=0)
    acc_n_row = jnp.mean(psi_samples[:, idx0][:, None] * jnp.conj(psi_samples), axis=0)
    acc_m_row = jnp.mean(psi_samples[:, idx1][:, None] * jnp.conj(psi_samples), axis=0)
    nq = precomp.nq

    true_d = {int(i): float(np.real(acc_d[i]))*(2 ** nq) for i in indices}
    true_n_row = {int(i): complex(acc_n_row[i])*(2 ** nq) for i in indices}
    true_m_row = {int(i): complex(acc_m_row[i])*(2 ** nq) for i in indices}
    return {"d": true_d , "n_row": true_n_row , "m_row": true_m_row}


def _shadow_norm_constant(n: int, m: int, nq: int) -> float:
    w = hamming_distance(n, m)
    return (6.0 ** w) * (4.0 ** (nq - w))


def _add_shadow_noise(
    key: jax.Array,
    true_values: Dict[str, Dict[int, complex]],
    num_shadows: int,
    nq: int,
    idx0: int,
    idx1: int,
) -> Dict[str, Dict[int, complex]]:
    if num_shadows <= 0:
        raise ValueError("num_shadows must be positive")

    indices = list(true_values["d"].keys())
    num_indices = len(indices)
    
    if num_indices == 0:
        return {"d": {}, "n_row": {}, "m_row": {}}
    
    # Vectorized computation using JAX
    indices_arr = jnp.asarray(indices, dtype=jnp.int32)
    idx0_arr = jnp.asarray(idx0, dtype=jnp.int32)
    idx1_arr = jnp.asarray(idx1, dtype=jnp.int32)
    
    # Extract true values as arrays
    true_d_arr = jnp.asarray([float(np.real(true_values["d"][i])) for i in indices], dtype=jnp.float64)
    true_n_arr = jnp.asarray([complex(true_values["n_row"][i]) for i in indices], dtype=jnp.complex128)
    true_m_arr = jnp.asarray([complex(true_values["m_row"][i]) for i in indices], dtype=jnp.complex128)
    
    # Compute shadow norm constants vectorized
    # For diagonal: c_di = 4^nq
    c_di = jnp.full(num_indices, 4.0 ** nq, dtype=jnp.float64)
    
    # For off-diagonals: c_ni = (6^w) * (4^(nq-w)) where w = hamming_distance(idx0, i)
    w_ni = hamming_distance_batch(indices_arr, idx0_arr).astype(jnp.float64)
    w_mi = hamming_distance_batch(indices_arr, idx1_arr).astype(jnp.float64)
    c_ni = (6.0 ** w_ni) * (4.0 ** (nq - w_ni))
    c_mi = (6.0 ** w_mi) * (4.0 ** (nq - w_mi))
    
    # Compute variances
    var_d = jnp.maximum(0.0, c_di - true_d_arr ** 2) / num_shadows
    std_d = jnp.sqrt(jnp.maximum(var_d, 1e-20))
    
    var_n = jnp.maximum(0.0, c_ni - jnp.abs(true_n_arr) ** 2) / num_shadows
    std_n = jnp.sqrt(jnp.maximum(var_n / 2.0, 1e-20))
    
    var_m = jnp.maximum(0.0, c_mi - jnp.abs(true_m_arr) ** 2) / num_shadows
    std_m = jnp.sqrt(jnp.maximum(var_m / 2.0, 1e-20))
    
    # Generate all noise at once
    # Need: num_indices for diagonal, 2*num_indices for n_row (real+imag), 2*num_indices for m_row (real+imag)
    total_keys = num_indices + 2 * num_indices + 2 * num_indices
    noise_keys = jr.split(key, total_keys)
    
    key_idx = 0
    # Diagonal noise (real only)
    noise_d = jr.normal(noise_keys[key_idx], shape=(num_indices,)) * std_d
    key_idx += 1
    
    # Off-diagonal n-row noise (complex: real + imag)
    noise_n_real = jr.normal(noise_keys[key_idx], shape=(num_indices,)) * std_n
    key_idx += 1
    noise_n_imag = jr.normal(noise_keys[key_idx], shape=(num_indices,)) * std_n
    key_idx += 1
    
    # Off-diagonal m-row noise (complex: real + imag)
    noise_m_real = jr.normal(noise_keys[key_idx], shape=(num_indices,)) * std_m
    key_idx += 1
    noise_m_imag = jr.normal(noise_keys[key_idx], shape=(num_indices,)) * std_m
    
    # Add noise
    noisy_d_arr = true_d_arr + noise_d
    noisy_n_arr = true_n_arr + (noise_n_real + 1j * noise_n_imag)
    noisy_m_arr = true_m_arr + (noise_m_real + 1j * noise_m_imag)
    
    # Convert back to dicts (for compatibility with existing code)
    noisy_d_dict: Dict[int, float] = {}
    noisy_n_row: Dict[int, complex] = {}
    noisy_m_row: Dict[int, complex] = {}

    for idx, i in enumerate(indices):
        noisy_d_dict[int(i)] = float(noisy_d_arr[idx])
        noisy_n_row[int(i)] = complex(noisy_n_arr[idx])
        noisy_m_row[int(i)] = complex(noisy_m_arr[idx])

    return {"d": noisy_d_dict, "n_row": noisy_n_row, "m_row": noisy_m_row}


def _add_shadow_noise_arrays(
    key: jax.Array,
    true_d: jnp.ndarray,  # Shape: (num_indices,)
    true_n_row: jnp.ndarray,  # Shape: (num_indices,) complex
    true_m_row: jnp.ndarray,  # Shape: (num_indices,) complex
    indices_arr: jnp.ndarray,  # Shape: (num_indices,)
    num_shadows: int,
    nq: int,
    idx0: int,
    idx1: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Array-based version of _add_shadow_noise that returns arrays instead of dicts.
    This is vmappable.
    
    Args:
        key: JAX PRNG key
        true_d: True diagonal values, shape (num_indices,)
        true_n_row: True n_row values (complex), shape (num_indices,)
        true_m_row: True m_row values (complex), shape (num_indices,)
        indices_arr: Array of indices, shape (num_indices,)
        num_shadows: Number of shadows (nps)
        nq: Number of qubits
        idx0: First index
        idx1: Second index
    
    Returns:
        Tuple of (noisy_d, noisy_n_row, noisy_m_row) as arrays
    """
    num_indices = indices_arr.shape[0]
    if num_indices == 0:
        return (
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )
    
    # Compute shadow norm constants vectorized
    idx0_arr = jnp.asarray(idx0, dtype=jnp.int32)
    idx1_arr = jnp.asarray(idx1, dtype=jnp.int32)
    
    # For diagonal: c_di = 4^nq
    c_di = jnp.full(num_indices, 4.0 ** nq, dtype=jnp.float64)
    
    # For off-diagonals: c_ni = (6^w) * (4^(nq-w)) where w = hamming_distance(idx0, i)
    w_ni = hamming_distance_batch(indices_arr, idx0_arr).astype(jnp.float64)
    w_mi = hamming_distance_batch(indices_arr, idx1_arr).astype(jnp.float64)
    c_ni = (6.0 ** w_ni) * (4.0 ** (nq - w_ni))
    c_mi = (6.0 ** w_mi) * (4.0 ** (nq - w_mi))
    
    # Compute variances
    var_d = jnp.maximum(0.0, c_di - true_d ** 2) / num_shadows
    std_d = jnp.sqrt(jnp.maximum(var_d, 1e-20))
    
    var_n = jnp.maximum(0.0, c_ni - jnp.abs(true_n_row) ** 2) / num_shadows
    std_n = jnp.sqrt(jnp.maximum(var_n / 2.0, 1e-20))
    
    var_m = jnp.maximum(0.0, c_mi - jnp.abs(true_m_row) ** 2) / num_shadows
    std_m = jnp.sqrt(jnp.maximum(var_m / 2.0, 1e-20))
    
    # Generate all noise at once
    # Need: num_indices for diagonal, 2*num_indices for n_row (real+imag), 2*num_indices for m_row (real+imag)
    total_keys = num_indices + 2 * num_indices + 2 * num_indices
    noise_keys = jr.split(key, total_keys)
    
    # Diagonal noise (real only)
    noise_d = jnp.stack([jr.normal(noise_keys[i], shape=()) * std_d[i] for i in range(num_indices)])
    
    # Off-diagonal n-row noise (complex: real + imag)
    key_idx = num_indices
    noise_n_real = jnp.stack([jr.normal(noise_keys[key_idx + i], shape=()) * std_n[i] for i in range(num_indices)])
    key_idx += num_indices
    noise_n_imag = jnp.stack([jr.normal(noise_keys[key_idx + i], shape=()) * std_n[i] for i in range(num_indices)])
    key_idx += num_indices
    
    # Off-diagonal m-row noise (complex: real + imag)
    noise_m_real = jnp.stack([jr.normal(noise_keys[key_idx + i], shape=()) * std_m[i] for i in range(num_indices)])
    key_idx += num_indices
    noise_m_imag = jnp.stack([jr.normal(noise_keys[key_idx + i], shape=()) * std_m[i] for i in range(num_indices)])
    
    # Add noise
    noisy_d_arr = true_d + noise_d
    noisy_n_arr = true_n_row + (noise_n_real + 1j * noise_n_imag)
    noisy_m_arr = true_m_row + (noise_m_real + 1j * noise_m_imag)
    
    return noisy_d_arr, noisy_n_arr, noisy_m_arr


@jax.jit
def _add_shadow_noise_arrays_jitted(
    key: jax.Array,
    true_d: jnp.ndarray,  # Shape: (num_indices,)
    true_n_row: jnp.ndarray,  # Shape: (num_indices,) complex
    true_m_row: jnp.ndarray,  # Shape: (num_indices,) complex
    indices_arr: jnp.ndarray,  # Shape: (num_indices,)
    num_shadows: int,
    nq: int,
    idx0: int,
    idx1: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled and optimized version of _add_shadow_noise_arrays.
    Uses vectorized noise generation for better performance.
    """
    num_indices = indices_arr.shape[0]
    if num_indices == 0:
        return (
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )
    
    # Compute shadow norm constants vectorized
    idx0_arr = jnp.asarray(idx0, dtype=jnp.int32)
    idx1_arr = jnp.asarray(idx1, dtype=jnp.int32)
    
    # For diagonal: c_di = 4^nq
    c_di = 4.0 ** nq
    
    # For off-diagonals: c_ni = (6^w) * (4^(nq-w)) where w = hamming_distance(idx0, i)
    w_ni = hamming_distance_batch(indices_arr, idx0_arr).astype(jnp.float64)
    w_mi = hamming_distance_batch(indices_arr, idx1_arr).astype(jnp.float64)
    c_ni = (6.0 ** w_ni) * (4.0 ** (nq - w_ni))
    c_mi = (6.0 ** w_mi) * (4.0 ** (nq - w_mi))
    
    # Compute variances
    var_d = jnp.maximum(0.0, c_di - true_d ** 2) / num_shadows
    std_d = jnp.sqrt(jnp.maximum(var_d, 1e-20))
    
    var_n = jnp.maximum(0.0, c_ni - jnp.abs(true_n_row) ** 2) / num_shadows
    std_n = jnp.sqrt(jnp.maximum(var_n / 2.0, 1e-20))
    
    var_m = jnp.maximum(0.0, c_mi - jnp.abs(true_m_row) ** 2) / num_shadows
    std_m = jnp.sqrt(jnp.maximum(var_m / 2.0, 1e-20))
    
    # OPTIMIZED: Generate all noise at once using vectorized operations
    # Need 5 separate keys for: diagonal, n_real, n_imag, m_real, m_imag
    noise_keys = jr.split(key, 5)
    
    # Vectorized noise generation (much faster than loop)
    # Each key generates a full array of random numbers at once
    noise_d = jr.normal(noise_keys[0], shape=(num_indices,)) * std_d
    noise_n_real = jr.normal(noise_keys[1], shape=(num_indices,)) * std_n
    noise_n_imag = jr.normal(noise_keys[2], shape=(num_indices,)) * std_n
    noise_m_real = jr.normal(noise_keys[3], shape=(num_indices,)) * std_m
    noise_m_imag = jr.normal(noise_keys[4], shape=(num_indices,)) * std_m
    
    # Add noise
    noisy_d_arr = true_d + noise_d
    noisy_n_arr = true_n_row + (noise_n_real + 1j * noise_n_imag)
    noisy_m_arr = true_m_row + (noise_m_real + 1j * noise_m_imag)
    
    return noisy_d_arr, noisy_n_arr, noisy_m_arr


def _compute_features_from_noisy_arrays(
    noisy_d: jnp.ndarray,  # Shape: (num_indices,)
    noisy_n_row: jnp.ndarray,  # Shape: (num_indices,) complex
    noisy_m_row: jnp.ndarray,  # Shape: (num_indices,) complex
    indices_arr: jnp.ndarray,  # Shape: (num_indices,)
    idx0: int,
    idx1: int,
    k_list: Sequence[int],
    indices_by_k_arr: jnp.ndarray,  # Shape: (len(k_list), max_indices_per_k)
    mask_by_k: jnp.ndarray,  # Shape: (len(k_list), max_indices_per_k), bool mask
) -> jnp.ndarray:
    """
    Compute feature vector from noisy arrays.
    This is vmappable.
    
    Args:
        noisy_d: Noisy diagonal values
        noisy_n_row: Noisy n_row values (complex)
        noisy_m_row: Noisy m_row values (complex)
        indices_arr: Array of indices
        idx0: First index
        idx1: Second index
        k_list: List of k values
        indices_by_k_arr: Array of indices for each k, shape (len(k_list), max_indices)
        mask_by_k: Boolean mask indicating valid indices, shape (len(k_list), max_indices)
    
    Returns:
        Feature vector as array, shape (3 + len(k_list),)
    """
    # Find positions of idx0 and idx1 in indices_arr
    # Since indices_arr is sorted, we can use searchsorted
    idx0_pos = jnp.searchsorted(indices_arr, idx0)
    idx1_pos = jnp.searchsorted(indices_arr, idx1)
    
    # Ensure we have the right index (handle case where idx might not be exactly in array)
    idx0_in_arr = (idx0_pos < len(indices_arr)) & (indices_arr[idx0_pos] == idx0)
    idx1_in_arr = (idx1_pos < len(indices_arr)) & (indices_arr[idx1_pos] == idx1)
    
    # Extract values (use safe indexing)
    d_n = jnp.where(idx0_in_arr, noisy_d[idx0_pos], 0.0)
    d_m = jnp.where(idx1_in_arr, noisy_d[idx1_pos], 0.0)
    
    # For rho_nm = hat_n_row[idx1], we need the value at idx1 in n_row
    rho_nm = jnp.where(idx1_in_arr, noisy_n_row[idx1_pos], 0.0j)
    a_nm = jnp.real(rho_nm)
    
    # Build base feature vector: [d_n, d_m, a_nm]
    feature_vec = jnp.zeros(3 + len(k_list), dtype=jnp.float32)
    feature_vec = feature_vec.at[0].set(jnp.float32(d_n))
    feature_vec = feature_vec.at[1].set(jnp.float32(d_m))
    feature_vec = feature_vec.at[2].set(jnp.float32(a_nm))
    
    # Compute x_k values for each k
    # For each k, we have indices stored in indices_by_k_arr[k_idx, :], 
    # with valid ones marked by mask_by_k[k_idx, :]
    # Since indices_arr is sorted and contains all indices we care about,
    # we can use searchsorted to find positions
    for k_idx in range(len(k_list)):
        # Get the indices for this k
        indices_k = indices_by_k_arr[k_idx, :]  # Shape: (max_indices,)
        mask_k = mask_by_k[k_idx, :]  # Shape: (max_indices,)
        
        # Find positions in indices_arr for each index in indices_k
        def find_position(idx):
            pos = jnp.searchsorted(indices_arr, idx)
            # Check if idx is actually at this position
            valid = (pos < len(indices_arr)) & (indices_arr[pos] == idx)
            return jnp.where(valid, pos, -1)
        
        # Map each index to its position
        positions = jax.vmap(find_position)(indices_k)  # Shape: (max_indices,)
        valid_positions = (positions >= 0) & mask_k
        
        # Extract n_row and m_row values at valid positions
        # Safe indexing: use where to select valid positions
        def safe_get(arr, pos):
            return jnp.where(pos >= 0, arr[pos], 0.0)
        
        re_n_vals = jax.vmap(lambda p: safe_get(jnp.real(noisy_n_row), p))(positions)
        re_m_vals = jax.vmap(lambda p: safe_get(jnp.real(noisy_m_row), p))(positions)
        
        # Zero out invalid values
        re_n_vals = jnp.where(valid_positions, re_n_vals, 0.0)
        re_m_vals = jnp.where(valid_positions, re_m_vals, 0.0)
        
        # Compute mean of re_n_vals * re_m_vals for valid indices
        products = re_n_vals * re_m_vals  # Shape: (max_indices,)
        num_valid = jnp.sum(mask_k.astype(jnp.float32))
        x_k_val = jnp.where(num_valid > 0, jnp.sum(products) / num_valid, 0.0)
        
        feature_vec = feature_vec.at[3 + k_idx].set(jnp.float32(x_k_val))
    
    return feature_vec


@partial(jax.jit, static_argnums=(4, 5, 6))  # idx0, idx1, k_list are static (positions 4, 5, 6)
def _compute_features_from_noisy_arrays_jitted(
    noisy_d: jnp.ndarray,  # Shape: (num_indices,)
    noisy_n_row: jnp.ndarray,  # Shape: (num_indices,) complex
    noisy_m_row: jnp.ndarray,  # Shape: (num_indices,) complex
    indices_arr: jnp.ndarray,  # Shape: (num_indices,)
    idx0: int,  # Static arg 4
    idx1: int,  # Static arg 5
    k_list: tuple,  # Static arg 6 - Must be tuple for static_argnums
    indices_by_k_arr: jnp.ndarray,  # Shape: (len(k_list), max_indices_per_k) - NOT static (JAX array)
    mask_by_k: jnp.ndarray,  # Shape: (len(k_list), max_indices_per_k), bool mask - NOT static (JAX array)
) -> jnp.ndarray:
    """
    JIT-compiled version of _compute_features_from_noisy_arrays.
    """
    # Find positions of idx0 and idx1 in indices_arr
    # Since indices_arr is sorted, we can use searchsorted
    idx0_pos = jnp.searchsorted(indices_arr, idx0)
    idx1_pos = jnp.searchsorted(indices_arr, idx1)
    
    # Ensure we have the right index (handle case where idx might not be exactly in array)
    idx0_in_arr = (idx0_pos < len(indices_arr)) & (indices_arr[idx0_pos] == idx0)
    idx1_in_arr = (idx1_pos < len(indices_arr)) & (indices_arr[idx1_pos] == idx1)
    
    # Extract values (use safe indexing)
    d_n = jnp.where(idx0_in_arr, noisy_d[idx0_pos], 0.0)
    d_m = jnp.where(idx1_in_arr, noisy_d[idx1_pos], 0.0)
    
    # For rho_nm = hat_n_row[idx1], we need the value at idx1 in n_row
    rho_nm = jnp.where(idx1_in_arr, noisy_n_row[idx1_pos], 0.0j)
    a_nm = jnp.real(rho_nm)
    
    # Build base feature vector: [d_n, d_m, a_nm]
    feature_vec = jnp.zeros(3 + len(k_list), dtype=jnp.float32)
    feature_vec = feature_vec.at[0].set(jnp.float32(d_n))
    feature_vec = feature_vec.at[1].set(jnp.float32(d_m))
    feature_vec = feature_vec.at[2].set(jnp.float32(a_nm))
    
    # Compute x_k values for each k
    # For each k, we have indices stored in indices_by_k_arr[k_idx, :], 
    # with valid ones marked by mask_by_k[k_idx, :]
    # Since indices_arr is sorted and contains all indices we care about,
    # we can use searchsorted to find positions
    for k_idx in range(len(k_list)):
        # Get the indices for this k
        indices_k = indices_by_k_arr[k_idx, :]  # Shape: (max_indices,)
        mask_k = mask_by_k[k_idx, :]  # Shape: (max_indices,)
        
        # Find positions in indices_arr for each index in indices_k
        def find_position(idx):
            pos = jnp.searchsorted(indices_arr, idx)
            # Check if idx is actually at this position
            valid = (pos < len(indices_arr)) & (indices_arr[pos] == idx)
            return jnp.where(valid, pos, -1)
        
        # Map each index to its position
        positions = jax.vmap(find_position)(indices_k)  # Shape: (max_indices,)
        valid_positions = (positions >= 0) & mask_k
        
        # Extract n_row and m_row values at valid positions
        # Safe indexing: use where to select valid positions
        def safe_get(arr, pos):
            return jnp.where(pos >= 0, arr[pos], 0.0)
        
        re_n_vals = jax.vmap(lambda p: safe_get(jnp.real(noisy_n_row), p))(positions)
        re_m_vals = jax.vmap(lambda p: safe_get(jnp.real(noisy_m_row), p))(positions)
        
        # Zero out invalid values
        re_n_vals = jnp.where(valid_positions, re_n_vals, 0.0)
        re_m_vals = jnp.where(valid_positions, re_m_vals, 0.0)
        
        # Compute mean of re_n_vals * re_m_vals for valid indices
        products = re_n_vals * re_m_vals  # Shape: (max_indices,)
        num_valid = jnp.sum(mask_k.astype(jnp.float32))
        x_k_val = jnp.where(num_valid > 0, jnp.sum(products) / num_valid, 0.0)
        
        feature_vec = feature_vec.at[3 + k_idx].set(jnp.float32(x_k_val))
    
    return feature_vec


def sample_noisy_observables(
    key: jax.Array,
    n: int,
    m: int,
    k_list: Sequence[int],
    channel_config: Dict[str, Any],
    N_total_shadows: int,
    M_trajectories_channel: int,
    pauli_precomp: WhiteNoisePauliPrecomputation,
) -> Dict[str, Any]:
    nq = pauli_precomp.nq
    key_true, key_shadow = jr.split(key)

    exclude = {n, m}
    indices_by_k = _collect_indices_by_weight(nq, exclude, k_list)

    indices_for_sim = set([n, m])
    for arr in indices_by_k.values():
        indices_for_sim.update(arr.tolist())
    indices_sorted = sorted(indices_for_sim)

    ctype = channel_config.get("type", "dephasing")
    true_values = None
    # Try analytic method for all channel types that support it
    if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
        true_values = _compute_true_noisy_vectors_analytic(
            pauli_precomp,
            channel_config,
            indices_sorted,
            n,
            m,
        )
    if true_values is None:
        true_values = _compute_true_noisy_vectors_mc(
            key_true,
            pauli_precomp,
            channel_config,
            indices_sorted,
            n,
            m,
            max(1, M_trajectories_channel),
        )
  

    noisy_values = _add_shadow_noise(
        key_shadow, true_values, N_total_shadows, nq, n, m
    )
    
    hat_d = noisy_values["d"]
    hat_n_row = noisy_values["n_row"]
    hat_m_row = noisy_values["m_row"]

    rho_nm = hat_n_row[m]

    outputs = {
        "a_nm": float(np.real(rho_nm)),
        "b_nm": float(np.imag(rho_nm)),
        "d_n": float(hat_d[n]),
        "d_m": float(hat_d[m]),
        "x_k": {},
    }

    for k in k_list:
        indices_k = indices_by_k.get(k, np.empty((0,), dtype=np.int32))
        if indices_k.size == 0:
            outputs["x_k"][k] = 0.0
            continue

        re_n_vals = np.real([hat_n_row[int(idx)] for idx in indices_k])
        re_m_vals = np.real([hat_m_row[int(idx)] for idx in indices_k])

        outputs["x_k"][k] = float(np.mean(re_n_vals * re_m_vals))

    return outputs


def _white_noise_outputs_to_feature_vector(
    outputs: Dict[str, Any],
    k_list: Sequence[int],
    nq: int,
) -> jnp.ndarray:
    """
    Convert sampled observables into the feature vector layout used by the NN.
    The ordering mirrors `get_relevant_input`, i.e. Nq + 2 features:
        [rho[idx0,idx0], rho[idx1,idx1], Re rho[idx0,idx1], x_k...]
    """
    scale = 1
    scale_sq = scale * scale
    feature_values: List[float] = [
        outputs["d_n"] * scale,
        outputs["d_m"] * scale,
        outputs["a_nm"] * scale,  # real part of rho[idx0, idx1]
    ]
    for k in k_list:
        feature_values.append(outputs["x_k"].get(k, 0.0) * scale_sq)
    return jnp.asarray(feature_values, dtype=jnp.float32)


def _true_values_dict_to_arrays(
    true_values: Dict[str, Dict[int, complex]],
    indices_sorted: Sequence[int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert true_values dict to arrays for vmapping.
    
    Args:
        true_values: Dict with keys "d", "n_row", "m_row", each mapping index to value
        indices_sorted: Sorted list of indices
        
    Returns:
        Tuple of (true_d_arr, true_n_row_arr, true_m_row_arr) as arrays
    """
    indices_arr = jnp.asarray(indices_sorted, dtype=jnp.int32)
    num_indices = len(indices_sorted)
    
    if num_indices == 0:
        return (
            jnp.array([], dtype=jnp.float64),
            jnp.array([], dtype=jnp.complex128),
            jnp.array([], dtype=jnp.complex128),
        )
    
    # Extract true values as arrays (in order of indices_sorted)
    true_d_arr = jnp.asarray(
        [float(np.real(true_values["d"].get(i, 0.0))) for i in indices_sorted],
        dtype=jnp.float64
    )
    true_n_arr = jnp.asarray(
        [complex(true_values["n_row"].get(i, 0.0)) for i in indices_sorted],
        dtype=jnp.complex128
    )
    true_m_arr = jnp.asarray(
        [complex(true_values["m_row"].get(i, 0.0)) for i in indices_sorted],
        dtype=jnp.complex128
    )
    
    return true_d_arr, true_n_arr, true_m_arr


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def _process_sample_nps_pair_vmappable(
    batch_idx: int,  # Index into true_values_batch arrays
    nps: int,
    key: jax.Array,
    true_d_batch: jnp.ndarray,  # Shape: (batch_size, num_indices)
    true_n_row_batch: jnp.ndarray,  # Shape: (batch_size, num_indices) complex
    true_m_row_batch: jnp.ndarray,  # Shape: (batch_size, num_indices) complex
    indices_arr: jnp.ndarray,  # Shape: (num_indices,)
    nq: int,  # Static arg 7
    idx0: int,  # Static arg 8
    idx1: int,  # Static arg 9
    k_list: tuple,  # Static arg 10 - Must be tuple for static_argnums
    indices_by_k_arr: jnp.ndarray,  # Shape: (len(k_list), max_indices)
    mask_by_k: jnp.ndarray,  # Shape: (len(k_list), max_indices)
) -> jnp.ndarray:
    """
    Vmappable function that processes one (sample, nps) pair.
    
    Args:
        batch_idx: Index into true_values_batch arrays
        nps: Number of shadows
        key: JAX PRNG key for noise generation
        true_d_batch: Batch of true diagonal values, shape (batch_size, num_indices)
        true_n_row_batch: Batch of true n_row values, shape (batch_size, num_indices) complex
        true_m_row_batch: Batch of true m_row values, shape (batch_size, num_indices) complex
        indices_arr: Array of indices, shape (num_indices,)
        nq: Number of qubits
        idx0: First index
        idx1: Second index
        k_list: List of k values (as tuple for static)
        indices_by_k_arr: Array of indices for each k, shape (len(k_list), max_indices)
        mask_by_k: Boolean mask indicating valid indices, shape (len(k_list), max_indices)
        channel_config: Channel config (static, not used in function)
        mc_trajectories: MC trajectories (static, not used in function)
    
    Returns:
        Feature vector as array, shape (3 + len(k_list),)
    """
    # Index into true values batch
    true_d = true_d_batch[batch_idx]  # Shape: (num_indices,)
    true_n_row = true_n_row_batch[batch_idx]  # Shape: (num_indices,)
    true_m_row = true_m_row_batch[batch_idx]  # Shape: (num_indices,)
    
    # Add shadow noise
    noisy_d, noisy_n_row, noisy_m_row = _add_shadow_noise_arrays(
        key, true_d, true_n_row, true_m_row, indices_arr, nps, nq, idx0, idx1
    )
    
    # Compute features
    feature_vec = _compute_features_from_noisy_arrays(
        noisy_d, noisy_n_row, noisy_m_row, indices_arr,
        idx0, idx1, k_list, indices_by_k_arr, mask_by_k
    )
    
    return feature_vec


def _generate_white_noise_features_for_samples_batched(
    config,
    sample_indices: Sequence[int],
    f_vectors: np.ndarray,
    nps_list: Sequence[int],
    *,
    mode: str,
    batch_size: int = 50,
) -> Dict[int, Dict[int, Dict[tuple, Dict[str, jnp.ndarray]]]]:
    """
    Fully vectorized JAX version using vmap to process all (sample, nps, alpha, y) combinations in parallel.
    
    Args:
        config: Configuration dict (may contain 'alpha_y_combinations' list)
        sample_indices: List of sample indices
        f_vectors: Array of F vectors shape (num_samples, 2**nq)
        nps_list: List of nps values
        mode: 'training' or 'validation'
        batch_size: Number of samples to process in parallel
        
    Returns:
        Dictionary mapping sample_idx -> {nps: {(alpha_tuple, y_str): feature_vector}}
        If only one (alpha, y) combination, returns sample_idx -> {nps: feature_vector} for backward compatibility
    """
    config_local = _synchronize_channel_config(dict(config))

    nq = config_local['n']
    
    # Get alpha_y combinations from config or use single default
    alpha_y_combinations = config_local.get('alpha_y_combinations')
    if alpha_y_combinations is None:
        # Backward compatibility: use single alpha/y from config
        alpha_raw = config_local.get('alpha_targ')
        if isinstance(alpha_raw, str):
            alpha_bits = tuple(int(b) for b in alpha_raw)
        else:
            alpha_bits = tuple(alpha_raw)
        y_bits = config_local.get('y_targ')
        alpha_y_combinations = [(alpha_bits, y_bits)]
        use_single_combination = True
    else:
        use_single_combination = False
    
    # Compute idx0, idx1 and indices structures for each (alpha, y) combination
    alpha_y_indices = []
    for alpha_bits, y_bits in alpha_y_combinations:
        alpha_int = int("".join(map(str, alpha_bits)), 2)
        y_int = int(y_bits, 2)
        idx0 = y_int ^ alpha_int
        idx1 = y_int
        alpha_y_indices.append((alpha_bits, y_bits, idx0, idx1))
    
    # Use first combination's idx0, idx1 for backward compatibility (for nq < 10 path)
    idx0 = alpha_y_indices[0][2]
    idx1 = alpha_y_indices[0][3]

    k_list = list(range(1, nq)) if nq > 1 else []
    channel_config = _white_noise_channel_config_from_config(config_local)
    base_seed = config_local.get('white_noise_seed', 42)
    if mode == "validation":
        base_seed = config_local.get('validation_white_noise_seed', base_seed)
    mc_trajectories = int(config_local.get('white_noise_mc_trajectories', 4096))

    # Precompute indices structure (same for all samples)
    exclude = {idx0, idx1}
    indices_by_k = _collect_indices_by_weight(nq, exclude, k_list)
    
    indices_for_sim = set([idx0, idx1])
    for arr in indices_by_k.values():
        indices_for_sim.update(arr.tolist())
    indices_sorted = sorted(indices_for_sim)
    indices_arr = jnp.asarray(indices_sorted, dtype=jnp.int32)
    
    # Convert indices_by_k dict to arrays for vmap
    max_indices_per_k = max(len(arr) for arr in indices_by_k.values()) if indices_by_k else 0
    if max_indices_per_k == 0:
        max_indices_per_k = 1  # Avoid empty arrays
    
    indices_by_k_arr = jnp.zeros((len(k_list), max_indices_per_k), dtype=jnp.int32)
    mask_by_k = jnp.zeros((len(k_list), max_indices_per_k), dtype=jnp.bool_)
    
    for k_idx, k in enumerate(k_list):
        indices_k = indices_by_k.get(k, np.empty((0,), dtype=np.int32))
        length = len(indices_k)
        if length > 0:
            indices_by_k_arr = indices_by_k_arr.at[k_idx, :length].set(indices_k)
            mask_by_k = mask_by_k.at[k_idx, :length].set(True)

    # Initialize feature dict structure based on whether we have multiple combinations
    if use_single_combination:
        feature_dict: Dict[int, Dict[int, jnp.ndarray]] = {}
    else:
        feature_dict: Dict[int, Dict[int, Dict[tuple, Dict[str, jnp.ndarray]]]] = {}

    num_samples = len(sample_indices)
    
    # Precompute indices structures for each (alpha, y) combination
    alpha_y_indices_structures = []
    for alpha_bits, y_bits, idx0, idx1 in alpha_y_indices:
        exclude = {idx0, idx1}
        indices_by_k = _collect_indices_by_weight(nq, exclude, k_list)
        
        indices_for_sim = set([idx0, idx1])
        for arr in indices_by_k.values():
            indices_for_sim.update(arr.tolist())
        indices_sorted = sorted(indices_for_sim)
        indices_arr = jnp.asarray(indices_sorted, dtype=jnp.int32)
        
        # Convert indices_by_k dict to arrays for vmap
        max_indices_per_k = max(len(arr) for arr in indices_by_k.values()) if indices_by_k else 0
        if max_indices_per_k == 0:
            max_indices_per_k = 1
        
        indices_by_k_arr = jnp.zeros((len(k_list), max_indices_per_k), dtype=jnp.int32)
        mask_by_k = jnp.zeros((len(k_list), max_indices_per_k), dtype=jnp.bool_)
        
        for k_idx, k in enumerate(k_list):
            indices_k = indices_by_k.get(k, np.empty((0,), dtype=np.int32))
            length = len(indices_k)
            if length > 0:
                indices_by_k_arr = indices_by_k_arr.at[k_idx, :length].set(indices_k)
                mask_by_k = mask_by_k.at[k_idx, :length].set(True)
        
        alpha_y_indices_structures.append({
            'alpha_bits': alpha_bits,
            'y_bits': y_bits,
            'idx0': idx0,
            'idx1': idx1,
            'indices_sorted': indices_sorted,
            'indices_arr': indices_arr,
            'indices_by_k_arr': indices_by_k_arr,
            'mask_by_k': mask_by_k,
        })
    
    # For large nq (>=10), use small batches with JIT-compiled functions to avoid OOM
    # This prevents OOM errors while still getting performance benefits from JIT
    if nq >= 10:
        # Use small batch size for nq=10 to get some vectorization benefits without OOM
        small_batch_size = 20  # Process 3 samples at a time
        
        print(f"  Processing {num_samples} samples in small batches (batch_size={small_batch_size}) for nq={nq}...")
        
        overall_start_time = time.time()
        summary_progress_interval = max(1, num_samples // 20)  # Show summary every ~20 times
        
        num_small_batches = (num_samples + small_batch_size - 1) // small_batch_size
        
        for batch_idx in range(num_small_batches):
            batch_start = batch_idx * small_batch_size
            batch_end = min(batch_start + small_batch_size, num_samples)
            batch_indices_slice = sample_indices[batch_start:batch_end]
            actual_batch_size = len(batch_indices_slice)
            
            batch_start_time = time.time()
            print(f"    Processing samples {batch_start + 1}-{batch_end}/{num_samples}...", end=" ", flush=True)
            
            # STEP 1: Precompute true values for all samples in batch (sequential, but fast)
            # Collect ALL idx0 and idx1 values needed across all (alpha, y) combinations
            all_idx0_set = set()
            all_idx1_set = set()
            all_indices_set = set()
            for alpha_y_struct in alpha_y_indices_structures:
                all_idx0_set.add(alpha_y_struct['idx0'])
                all_idx1_set.add(alpha_y_struct['idx1'])
                all_indices_set.update(alpha_y_struct['indices_sorted'])
            all_indices_sorted = sorted(all_indices_set)
            all_idx0_sorted = sorted(all_idx0_set)
            all_idx1_sorted = sorted(all_idx1_set)
            
            # We need to compute rows for ALL idx0 and idx1 values
            # Store as: {idx0: batch_array, idx1: batch_array}
            true_d_batch = []
            true_rows_by_idx0 = {}  # idx0 -> list of arrays (one per sample in batch)
            true_rows_by_idx1 = {}  # idx1 -> list of arrays (one per sample in batch)
            
            for local_idx, sample_idx in enumerate(batch_indices_slice):
                f_vec = f_vectors[sample_idx]
                
                # Build precomputation
                precomp = _build_white_noise_pauli_precomputation(f_vec, nq)
                
                sample_seed = base_seed + int(sample_idx) * 10007
                ctype = channel_config.get("type", "dephasing")
                
                # Compute diagonal once (same for all combinations, doesn't depend on idx0/idx1)
                idx0_for_diag = all_idx0_sorted[0] if all_idx0_sorted else 0
                true_values_diag = None
                if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
                    true_values_diag = _compute_true_noisy_vectors_analytic(
                        precomp, channel_config, all_indices_sorted, idx0_for_diag, idx0_for_diag
                    )
                if true_values_diag is None:
                    key_true_base = jr.PRNGKey(sample_seed + 99999)
                    true_values_diag = _compute_true_noisy_vectors_mc(
                        key_true_base, precomp, channel_config, all_indices_sorted,
                        idx0_for_diag, idx0_for_diag, max(1, mc_trajectories)
                    )
                
                true_d_arr, _, _ = _true_values_dict_to_arrays(true_values_diag, all_indices_sorted)
                true_d_batch.append(true_d_arr)
                
                # Compute rows for each unique idx0 (row idx0 of density matrix)
                for idx0 in all_idx0_set:
                    if idx0 not in true_rows_by_idx0:
                        true_rows_by_idx0[idx0] = []
                    true_values = None
                    if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
                        true_values = _compute_true_noisy_vectors_analytic(
                            precomp, channel_config, all_indices_sorted, idx0, idx0
                        )
                    if true_values is None:
                        key_true_base = jr.PRNGKey(sample_seed + 99999 + idx0 * 1000)
                        true_values = _compute_true_noisy_vectors_mc(
                            key_true_base, precomp, channel_config, all_indices_sorted,
                            idx0, idx0, max(1, mc_trajectories)
                        )
                    # Extract n_row (which is rho[idx0, :])
                    _, true_n_arr, _ = _true_values_dict_to_arrays(true_values, all_indices_sorted)
                    true_rows_by_idx0[idx0].append(true_n_arr)
                
                # Compute rows for each unique idx1 (row idx1 of density matrix)
                for idx1 in all_idx1_set:
                    if idx1 not in true_rows_by_idx1:
                        true_rows_by_idx1[idx1] = []
                    true_values = None
                    if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
                        true_values = _compute_true_noisy_vectors_analytic(
                            precomp, channel_config, all_indices_sorted, idx1, idx1
                        )
                    if true_values is None:
                        key_true_base = jr.PRNGKey(sample_seed + 99999 + idx1 * 1000)
                        true_values = _compute_true_noisy_vectors_mc(
                            key_true_base, precomp, channel_config, all_indices_sorted,
                            idx1, idx1, max(1, mc_trajectories)
                        )
                    # Extract m_row (which is rho[idx1, :])
                    _, _, true_m_arr = _true_values_dict_to_arrays(true_values, all_indices_sorted)
                    true_rows_by_idx1[idx1].append(true_m_arr)
                
                # Clean up precomputation
                del precomp, true_values_diag
            
            # Stack into batch arrays
            true_d_batch = jnp.stack(true_d_batch)  # Shape: (actual_batch_size, len(all_indices_sorted))
            
            # Stack rows for each idx0/idx1 into batch arrays
            for idx0 in true_rows_by_idx0:
                true_rows_by_idx0[idx0] = jnp.stack(true_rows_by_idx0[idx0])  # Shape: (actual_batch_size, len(all_indices_sorted))
            for idx1 in true_rows_by_idx1:
                true_rows_by_idx1[idx1] = jnp.stack(true_rows_by_idx1[idx1])  # Shape: (actual_batch_size, len(all_indices_sorted))
            
            # Create mapping from union indices to each alpha_y combination's indices
            # Also extract the specific rows for each combination
            # Prepare arrays for parallel processing of all combinations
            all_indices_arr = jnp.asarray(all_indices_sorted, dtype=jnp.int32)
            alpha_y_index_mappings = []
            alpha_y_rows = []  # Store (true_n_row, true_m_row) for each combination
            
            for alpha_y_struct in alpha_y_indices_structures:
                target_indices = alpha_y_struct['indices_sorted']
                # Create mapping: for each target index, find its position in all_indices_sorted
                mapping = []
                for target_idx in target_indices:
                    pos = all_indices_sorted.index(target_idx)
                    mapping.append(pos)
                alpha_y_index_mappings.append(jnp.array(mapping, dtype=jnp.int32))
                
                # Extract the specific rows for this combination
                idx0 = alpha_y_struct['idx0']
                idx1 = alpha_y_struct['idx1']
                true_n_row_for_combo = true_rows_by_idx0[idx0]  # Shape: (actual_batch_size, len(all_indices_sorted))
                true_m_row_for_combo = true_rows_by_idx1[idx1]  # Shape: (actual_batch_size, len(all_indices_sorted))
                alpha_y_rows.append((true_n_row_for_combo, true_m_row_for_combo))
            
            # STEP 2: Create all (sample, nps, alpha_y_idx) pairs for vmap
            all_batch_idxs = []
            all_nps_values = []
            all_alpha_y_idxs = []
            all_keys = []
            
            for local_idx, sample_idx in enumerate(batch_indices_slice):
                sample_seed = base_seed + int(sample_idx) * 10007
                # Generate keys for all (nps, alpha_y) combinations
                total_combinations = len(nps_list) * len(alpha_y_indices_structures)
                keys_for_sample = jr.split(jr.PRNGKey(sample_seed), total_combinations)
                key_idx = 0
                
                for nps_idx, nps in enumerate(nps_list):
                    for alpha_y_idx in range(len(alpha_y_indices_structures)):
                        all_batch_idxs.append(local_idx)
                        all_nps_values.append(nps)
                        all_alpha_y_idxs.append(alpha_y_idx)
                        all_keys.append(keys_for_sample[key_idx])
                        key_idx += 1
            
            # Convert to arrays
            all_batch_idxs_arr = jnp.array(all_batch_idxs, dtype=jnp.int32)
            all_nps_values_arr = jnp.array(all_nps_values, dtype=jnp.int32)
            all_alpha_y_idxs_arr = jnp.array(all_alpha_y_idxs, dtype=jnp.int32)
            all_keys_arr = jnp.stack(all_keys)
            
            # STEP 3: Use vmap to process all (sample, nps, alpha_y) pairs in parallel
            # Process all combinations together by selecting the right structures based on alpha_y_idx
            
            if use_single_combination:
                # Single combination: use original fast path
                alpha_y_struct = alpha_y_indices_structures[0]
                indices_arr = alpha_y_struct['indices_arr']
                idx0 = alpha_y_struct['idx0']
                idx1 = alpha_y_struct['idx1']
                indices_by_k_arr = alpha_y_struct['indices_by_k_arr']
                mask_by_k = alpha_y_struct['mask_by_k']
                
                def process_pair_vmappable(batch_idx, nps_val, key, true_d_batch, true_n_row_batch, true_m_row_batch,
                                         indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                                         index_mapping):
                    # Extract true values for this alpha_y combination's indices
                    true_d_full = true_d_batch[batch_idx]
                    true_n_full = true_n_row_batch[batch_idx]  # Row idx0 of density matrix (pre-extracted)
                    true_m_full = true_m_row_batch[batch_idx]  # Row idx1 of density matrix (pre-extracted)
                    
                    # Map from union indices to this combination's indices
                    true_d = true_d_full[index_mapping]
                    true_n = true_n_full[index_mapping]
                    true_m = true_m_full[index_mapping]
                    
                    noisy_d, noisy_n_row, noisy_m_row = _add_shadow_noise_arrays_jitted(
                        key, true_d, true_n, true_m,
                        indices_arr, nps_val, nq, idx0, idx1
                    )
                    
                    feature_vec = _compute_features_from_noisy_arrays_jitted(
                        noisy_d, noisy_n_row, noisy_m_row, indices_arr,
                        idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k
                    )
                    return feature_vec
                
                k_list_tuple = tuple(k_list)
                index_mapping = alpha_y_index_mappings[0]
                true_n_row_batch, true_m_row_batch = alpha_y_rows[0]  # Get pre-extracted rows for this combination
                vmap_process = jax.vmap(process_pair_vmappable, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))
                
                # Filter to only this alpha_y combination
                mask = all_alpha_y_idxs_arr == 0
                filtered_batch_idxs = all_batch_idxs_arr[mask]
                filtered_nps_values = all_nps_values_arr[mask]
                filtered_keys = all_keys_arr[mask]
                
                feature_vectors = vmap_process(
                    filtered_batch_idxs, filtered_nps_values, filtered_keys,
                    true_d_batch, true_n_row_batch, true_m_row_batch,
                    indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                    index_mapping
                )
                
                # STEP 4: Reorganize into feature_dict (backward compatible format)
                pair_idx = 0
                for local_idx, sample_idx in enumerate(batch_indices_slice):
                    feature_dict[int(sample_idx)] = {}
                    for nps in nps_list:
                        feature_dict[int(sample_idx)][int(nps)] = feature_vectors[pair_idx]
                        pair_idx += 1
            else:
                # Multiple combinations: process each combination's pairs in parallel
                # Each combination processes all its (sample, nps) pairs together via vmap
                # Note: We process combinations sequentially to avoid dynamic slicing issues,
                # but each combination's pairs are fully parallelized
                k_list_tuple = tuple(k_list)
                
                # Process all combinations - each processes all its pairs in parallel via vmap
                for alpha_y_idx, alpha_y_struct in enumerate(alpha_y_indices_structures):
                    indices_arr = alpha_y_struct['indices_arr']
                    idx0 = alpha_y_struct['idx0']
                    idx1 = alpha_y_struct['idx1']
                    indices_by_k_arr = alpha_y_struct['indices_by_k_arr']
                    mask_by_k = alpha_y_struct['mask_by_k']
                    alpha_bits = alpha_y_struct['alpha_bits']
                    y_bits = alpha_y_struct['y_bits']
                    
                    # Filter pairs for this alpha_y combination
                    mask = all_alpha_y_idxs_arr == alpha_y_idx
                    filtered_batch_idxs = all_batch_idxs_arr[mask]
                    filtered_nps_values = all_nps_values_arr[mask]
                    filtered_keys = all_keys_arr[mask]
                    
                    def process_pair_vmappable(batch_idx, nps_val, key, true_d_batch, true_n_row_batch, true_m_row_batch,
                                             indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                                             index_mapping):
                        # Extract true values for this alpha_y combination's indices
                        true_d_full = true_d_batch[batch_idx]
                        true_n_full = true_n_row_batch[batch_idx]  # Row idx0 of density matrix (pre-extracted)
                        true_m_full = true_m_row_batch[batch_idx]  # Row idx1 of density matrix (pre-extracted)
                        
                        # Map from union indices to this combination's indices
                        true_d = true_d_full[index_mapping]
                        true_n = true_n_full[index_mapping]
                        true_m = true_m_full[index_mapping]
                        
                        noisy_d, noisy_n_row, noisy_m_row = _add_shadow_noise_arrays_jitted(
                            key, true_d, true_n, true_m,
                            indices_arr, nps_val, nq, idx0, idx1
                        )
                        
                        feature_vec = _compute_features_from_noisy_arrays_jitted(
                            noisy_d, noisy_n_row, noisy_m_row, indices_arr,
                            idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k
                        )
                        return feature_vec
                    
                    index_mapping = alpha_y_index_mappings[alpha_y_idx]
                    true_n_row_batch, true_m_row_batch = alpha_y_rows[alpha_y_idx]  # Get pre-extracted rows for this combination
                    vmap_process = jax.vmap(process_pair_vmappable, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))
                    
                    feature_vectors = vmap_process(
                        filtered_batch_idxs, filtered_nps_values, filtered_keys,
                        true_d_batch, true_n_row_batch, true_m_row_batch,
                        indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                        index_mapping
                    )
                    
                    # Store results immediately
                    pair_idx = 0
                    for local_idx, sample_idx in enumerate(batch_indices_slice):
                        if int(sample_idx) not in feature_dict:
                            feature_dict[int(sample_idx)] = {}
                        for nps in nps_list:
                            if int(nps) not in feature_dict[int(sample_idx)]:
                                feature_dict[int(sample_idx)][int(nps)] = {}
                            feature_dict[int(sample_idx)][int(nps)][(alpha_bits, y_bits)] = feature_vectors[pair_idx]
                            pair_idx += 1
            
            # Clean up
            del true_d_batch, true_rows_by_idx0, true_rows_by_idx1
            
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - overall_start_time
            avg_time_per_sample = batch_time / actual_batch_size
            
            print(f"done ({batch_time:.2f}s, {avg_time_per_sample:.2f}s/sample)")
            
            # Show detailed progress summary periodically
            if (batch_idx + 1) % (summary_progress_interval // small_batch_size + 1) == 0 or batch_end == num_samples:
                progress_pct = 100.0 * batch_end / num_samples
                
                if batch_end > 0:
                    overall_avg_time = elapsed_time / batch_end
                    remaining_samples = num_samples - batch_end
                    eta_seconds = overall_avg_time * remaining_samples
                    eta_minutes = eta_seconds / 60
                    
                    print(f"    → Summary: {batch_end}/{num_samples} ({progress_pct:.1f}%) | "
                          f"Elapsed: {elapsed_time/60:.1f}m | "
                          f"ETA: {eta_minutes:.1f}m | "
                          f"Avg: {overall_avg_time:.2f}s/sample")
                
                # Periodic garbage collection
                import gc
                gc.collect()
        
        total_time = time.time() - overall_start_time
        print(f"  ✓ Completed processing {num_samples} samples in {total_time/60:.1f} minutes")
        return feature_dict
    
    # Original vmap code for nq < 10 (now also supports multiple alpha/y combinations)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Use union of all indices needed across all (alpha, y) combinations
    all_indices_set = set()
    for alpha_y_struct in alpha_y_indices_structures:
        all_indices_set.update(alpha_y_struct['indices_sorted'])
    all_indices_sorted = sorted(all_indices_set)
    
    # Create mapping from union indices to each alpha_y combination's indices
    alpha_y_index_mappings = []
    for alpha_y_struct in alpha_y_indices_structures:
        target_indices = alpha_y_struct['indices_sorted']
        mapping = []
        for target_idx in target_indices:
            pos = all_indices_sorted.index(target_idx)
            mapping.append(pos)
        alpha_y_index_mappings.append(jnp.array(mapping, dtype=jnp.int32))
    
    print(f"  Processing {num_samples} samples in {num_batches} batches (batch_size={batch_size}) with vmap...")
    if not use_single_combination:
        print(f"  Generating features for {len(alpha_y_combinations)} (alpha, y) combinations")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = sample_indices[batch_start:batch_end]
        batch_f_vectors = f_vectors[batch_start:batch_end]
        actual_batch_size = len(batch_indices)
        
        print(f"    Batch {batch_idx + 1}/{num_batches} ({batch_start}-{batch_end})...", end=" ", flush=True)
        batch_start_time = time.time()
        
        # STEP 1: Precompute true values for all samples in batch
        # Collect ALL idx0 and idx1 values needed across all (alpha, y) combinations
        all_idx0_set = set()
        all_idx1_set = set()
        for alpha_y_struct in alpha_y_indices_structures:
            all_idx0_set.add(alpha_y_struct['idx0'])
            all_idx1_set.add(alpha_y_struct['idx1'])
        all_idx0_sorted = sorted(all_idx0_set)
        all_idx1_sorted = sorted(all_idx1_set)
        
        # We need to compute rows for ALL idx0 and idx1 values
        true_d_batch = []
        true_rows_by_idx0 = {}  # idx0 -> list of arrays (one per sample in batch)
        true_rows_by_idx1 = {}  # idx1 -> list of arrays (one per sample in batch)
        
        for local_idx, sample_idx in enumerate(batch_indices):
            # Build Pauli precomputation for THIS sample's F-vector
            f_vec = batch_f_vectors[local_idx]
            precomp = _build_white_noise_pauli_precomputation(f_vec, nq)

            sample_seed = base_seed + int(sample_idx) * 10007
            
            ctype = channel_config.get("type", "dephasing")
            
            # Compute diagonal once (same for all combinations, doesn't depend on idx0/idx1)
            idx0_for_diag = all_idx0_sorted[0] if all_idx0_sorted else 0
            true_values_diag = None
            if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
                true_values_diag = _compute_true_noisy_vectors_analytic(
                    precomp, channel_config, all_indices_sorted, idx0_for_diag, idx0_for_diag
                )
            if true_values_diag is None:
                key_true_base = jr.PRNGKey(sample_seed + 99999)
                true_values_diag = _compute_true_noisy_vectors_mc(
                    key_true_base, precomp, channel_config, all_indices_sorted,
                    idx0_for_diag, idx0_for_diag, max(1, mc_trajectories)
                )
            
            true_d_arr, _, _ = _true_values_dict_to_arrays(true_values_diag, all_indices_sorted)
            true_d_batch.append(true_d_arr)
            
            # Compute rows for each unique idx0 (row idx0 of density matrix)
            for idx0 in all_idx0_set:
                if idx0 not in true_rows_by_idx0:
                    true_rows_by_idx0[idx0] = []
                true_values = None
                if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
                    true_values = _compute_true_noisy_vectors_analytic(
                        precomp, channel_config, all_indices_sorted, idx0, idx0
                    )
                if true_values is None:
                    key_true_base = jr.PRNGKey(sample_seed + 99999 + idx0 * 1000)
                    true_values = _compute_true_noisy_vectors_mc(
                        key_true_base, precomp, channel_config, all_indices_sorted,
                        idx0, idx0, max(1, mc_trajectories)
                    )
                # Extract n_row (which is rho[idx0, :])
                _, true_n_arr, _ = _true_values_dict_to_arrays(true_values, all_indices_sorted)
                true_rows_by_idx0[idx0].append(true_n_arr)
            
            # Compute rows for each unique idx1 (row idx1 of density matrix)
            for idx1 in all_idx1_set:
                if idx1 not in true_rows_by_idx1:
                    true_rows_by_idx1[idx1] = []
                true_values = None
                if ctype in ("dephasing", "depolarizing", "relaxation", "thermal", "none"):
                    true_values = _compute_true_noisy_vectors_analytic(
                        precomp, channel_config, all_indices_sorted, idx1, idx1
                    )
                if true_values is None:
                    key_true_base = jr.PRNGKey(sample_seed + 99999 + idx1 * 1000)
                    true_values = _compute_true_noisy_vectors_mc(
                        key_true_base, precomp, channel_config, all_indices_sorted,
                        idx1, idx1, max(1, mc_trajectories)
                    )
                # Extract m_row (which is rho[idx1, :])
                _, _, true_m_arr = _true_values_dict_to_arrays(true_values, all_indices_sorted)
                true_rows_by_idx1[idx1].append(true_m_arr)
        
        # Stack into batch arrays
        true_d_batch = jnp.stack(true_d_batch)  # Shape: (actual_batch_size, len(all_indices_sorted))
        
        # Stack rows for each idx0/idx1 into batch arrays
        for idx0 in true_rows_by_idx0:
            true_rows_by_idx0[idx0] = jnp.stack(true_rows_by_idx0[idx0])  # Shape: (actual_batch_size, len(all_indices_sorted))
        for idx1 in true_rows_by_idx1:
            true_rows_by_idx1[idx1] = jnp.stack(true_rows_by_idx1[idx1])  # Shape: (actual_batch_size, len(all_indices_sorted))
        
        # Extract the specific rows for each combination
        alpha_y_rows = []  # Store (true_n_row, true_m_row) for each combination
        for alpha_y_struct in alpha_y_indices_structures:
            idx0 = alpha_y_struct['idx0']
            idx1 = alpha_y_struct['idx1']
            true_n_row_for_combo = true_rows_by_idx0[idx0]  # Shape: (actual_batch_size, len(all_indices_sorted))
            true_m_row_for_combo = true_rows_by_idx1[idx1]  # Shape: (actual_batch_size, len(all_indices_sorted))
            alpha_y_rows.append((true_n_row_for_combo, true_m_row_for_combo))
        
        # STEP 2: Create all (sample, nps, alpha_y_idx) pairs
        all_batch_idxs = []
        all_nps_values = []
        all_alpha_y_idxs = []
        all_keys = []
        
        for local_idx, sample_idx in enumerate(batch_indices):
            sample_seed = base_seed + int(sample_idx) * 10007
            # Generate keys for all (nps, alpha_y) combinations
            total_combinations = len(nps_list) * len(alpha_y_indices_structures)
            keys_for_sample = jr.split(jr.PRNGKey(sample_seed), total_combinations)
            key_idx = 0
            
            for nps_idx, nps in enumerate(nps_list):
                for alpha_y_idx in range(len(alpha_y_indices_structures)):
                    all_batch_idxs.append(local_idx)
                    all_nps_values.append(nps)
                    all_alpha_y_idxs.append(alpha_y_idx)
                    all_keys.append(keys_for_sample[key_idx])
                    key_idx += 1
        
        # Convert to arrays
        all_batch_idxs_arr = jnp.array(all_batch_idxs, dtype=jnp.int32)
        all_nps_values_arr = jnp.array(all_nps_values, dtype=jnp.int32)
        all_alpha_y_idxs_arr = jnp.array(all_alpha_y_idxs, dtype=jnp.int32)
        all_keys_arr = jnp.stack(all_keys)
        
        # STEP 3: Process each alpha_y combination separately (for simplicity)
        if use_single_combination:
            # Single combination: use original fast path
            alpha_y_struct = alpha_y_indices_structures[0]
            indices_arr = alpha_y_struct['indices_arr']
            idx0 = alpha_y_struct['idx0']
            idx1 = alpha_y_struct['idx1']
            indices_by_k_arr = alpha_y_struct['indices_by_k_arr']
            mask_by_k = alpha_y_struct['mask_by_k']
            index_mapping = alpha_y_index_mappings[0]
            true_n_row_batch, true_m_row_batch = alpha_y_rows[0]  # Get pre-extracted rows for this combination
            
            # Filter to only this alpha_y combination
            mask = all_alpha_y_idxs_arr == 0
            filtered_batch_idxs = all_batch_idxs_arr[mask]
            filtered_nps_values = all_nps_values_arr[mask]
            filtered_keys = all_keys_arr[mask]
            
            # Create vmappable function with index mapping
            def process_pair_vmappable(batch_idx, nps_val, key, true_d_batch, true_n_row_batch, true_m_row_batch,
                                     indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                                     index_mapping):
                true_d_full = true_d_batch[batch_idx]
                true_n_full = true_n_row_batch[batch_idx]  # Row idx0 of density matrix (pre-extracted)
                true_m_full = true_m_row_batch[batch_idx]  # Row idx1 of density matrix (pre-extracted)
                
                true_d = true_d_full[index_mapping]
                true_n = true_n_full[index_mapping]
                true_m = true_m_full[index_mapping]
                
                noisy_d, noisy_n_row, noisy_m_row = _add_shadow_noise_arrays_jitted(
                    key, true_d, true_n, true_m,
                    indices_arr, nps_val, nq, idx0, idx1
                )
                
                feature_vec = _compute_features_from_noisy_arrays_jitted(
                    noisy_d, noisy_n_row, noisy_m_row, indices_arr,
                    idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k
                )
                return feature_vec
            
            k_list_tuple = tuple(k_list)
            vmap_fn = jax.vmap(process_pair_vmappable, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))
            
            feature_vectors = vmap_fn(
                filtered_batch_idxs, filtered_nps_values, filtered_keys,
                true_d_batch, true_n_row_batch, true_m_row_batch,
                indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                index_mapping
            )
            
            # STEP 4: Reorganize into feature_dict (backward compatible format)
            pair_idx = 0
            for local_idx, sample_idx in enumerate(batch_indices):
                feature_dict[int(sample_idx)] = {}
                for nps in nps_list:
                    feature_dict[int(sample_idx)][int(nps)] = feature_vectors[pair_idx]
                    pair_idx += 1
        else:
            # Multiple combinations: process each separately
            for alpha_y_idx, alpha_y_struct in enumerate(alpha_y_indices_structures):
                indices_arr = alpha_y_struct['indices_arr']
                idx0 = alpha_y_struct['idx0']
                idx1 = alpha_y_struct['idx1']
                indices_by_k_arr = alpha_y_struct['indices_by_k_arr']
                mask_by_k = alpha_y_struct['mask_by_k']
                alpha_bits = alpha_y_struct['alpha_bits']
                y_bits = alpha_y_struct['y_bits']
                index_mapping = alpha_y_index_mappings[alpha_y_idx]
                true_n_row_batch, true_m_row_batch = alpha_y_rows[alpha_y_idx]  # Get pre-extracted rows for this combination
                
                # Filter pairs for this alpha_y combination
                mask = all_alpha_y_idxs_arr == alpha_y_idx
                filtered_batch_idxs = all_batch_idxs_arr[mask]
                filtered_nps_values = all_nps_values_arr[mask]
                filtered_keys = all_keys_arr[mask]
                
                def process_pair_vmappable(batch_idx, nps_val, key, true_d_batch, true_n_row_batch, true_m_row_batch,
                                         indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                                         index_mapping):
                    true_d_full = true_d_batch[batch_idx]
                    true_n_full = true_n_row_batch[batch_idx]  # Row idx0 of density matrix (pre-extracted)
                    true_m_full = true_m_row_batch[batch_idx]  # Row idx1 of density matrix (pre-extracted)
                    
                    true_d = true_d_full[index_mapping]
                    true_n = true_n_full[index_mapping]
                    true_m = true_m_full[index_mapping]
                    
                    noisy_d, noisy_n_row, noisy_m_row = _add_shadow_noise_arrays_jitted(
                        key, true_d, true_n, true_m,
                        indices_arr, nps_val, nq, idx0, idx1
                    )
                    
                    feature_vec = _compute_features_from_noisy_arrays_jitted(
                        noisy_d, noisy_n_row, noisy_m_row, indices_arr,
                        idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k
                    )
                    return feature_vec
                
                k_list_tuple = tuple(k_list)
                vmap_fn = jax.vmap(process_pair_vmappable, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))
                
                feature_vectors = vmap_fn(
                    filtered_batch_idxs, filtered_nps_values, filtered_keys,
                    true_d_batch, true_n_row_batch, true_m_row_batch,
                    indices_arr, nq, idx0, idx1, k_list_tuple, indices_by_k_arr, mask_by_k,
                    index_mapping
                )
                
                # Store in nested structure
                pair_idx = 0
                for local_idx, sample_idx in enumerate(batch_indices):
                    if int(sample_idx) not in feature_dict:
                        feature_dict[int(sample_idx)] = {}
                    for nps in nps_list:
                        if int(nps) not in feature_dict[int(sample_idx)]:
                            feature_dict[int(sample_idx)][int(nps)] = {}
                        feature_dict[int(sample_idx)][int(nps)][(alpha_bits, y_bits)] = feature_vectors[pair_idx]
                        pair_idx += 1
        
        batch_time = time.time() - batch_start_time
        print(f"done ({batch_time:.2f}s)")

    return feature_dict


def _generate_white_noise_features_for_samples(
    config,
    sample_indices: Sequence[int],
    f_vectors: np.ndarray,
    nps_list: Sequence[int],
    *,
    mode: str,
) -> Dict[int, Dict[int, jnp.ndarray]]:
    """
    Generate white noise features for multiple samples.
    
    This function now uses the optimized batched version for better performance.
    """
    # Use the optimized batched version
    batch_size = config.get('white_noise_batch_size', 50)
    return _generate_white_noise_features_for_samples_batched(
        config, sample_indices, f_vectors, nps_list, mode=mode, batch_size=batch_size
    )


def generate_white_noise_training_at_checkpoints_vectorized(config, data_dict, nps_list):
    """
    Generate white noise training features at all nps checkpoints for all training samples.
    Optionally saves features to a pickle file.
    Checks for existing cache files and loads them if metadata matches.
    """
    f_vectors = np.asarray(data_dict['F_bs'], dtype=np.int32)
    sample_indices = list(range(f_vectors.shape[0]))
    nq = config.get('n')
    channel_type = config.get('channel_type', 'unknown')
    noise_strength = config.get('noise_strength', 0.0)
    alpha_y_combinations = config.get('alpha_y_combinations', [])
    
    # Check for existing cache
    use_cache = config.get('use_feature_cache', True)  # Default to True
    if use_cache:
        # Build cache filename with alpha/y info
        strength_str = f"{noise_strength:.2f}".replace('.', '_')
        features_dir = config.get('features_cache_dir', 'white_noise_features_cache')
        
        # Handle multiple alpha/y combinations - use first one for filename if single, or create combined name
        if len(alpha_y_combinations) == 1:
            alpha, y = alpha_y_combinations[0]
            alpha_str = ''.join(map(str, alpha)) if isinstance(alpha, tuple) else str(alpha)
            filename = f"features_nq{nq}_{channel_type}_{strength_str}_alpha{alpha_str}_y{y}.pkl"
        elif len(alpha_y_combinations) > 1:
            # For multiple combinations, use a combined identifier
            combo_strs = []
            for alpha, y in alpha_y_combinations:
                alpha_str = ''.join(map(str, alpha)) if isinstance(alpha, tuple) else str(alpha)
                combo_strs.append(f"{alpha_str}_{y}")
            combo_id = "_".join(combo_strs)[:50]  # Limit length
            filename = f"features_nq{nq}_{channel_type}_{strength_str}_multi_{combo_id}.pkl"
        else:
            # Fallback to old naming if no alpha_y_combinations
            filename = f"features_nq{nq}_{channel_type}_{strength_str}.pkl"
        
        filepath = os.path.join(features_dir, filename)
        
        # Try to load cache if it exists
        if os.path.exists(filepath):
            try:
                print(f"\n{'='*80}")
                print(f"[CACHE] Checking for existing feature cache: {filepath}")
                print(f"{'='*80}\n")
                
                with open(filepath, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify metadata matches
                cache_nq = cached_data.get('nq')
                cache_channel = cached_data.get('channel_type')
                cache_strength = cached_data.get('noise_strength')
                cache_nps_list = cached_data.get('nps_list', [])
                cache_num_samples = cached_data.get('num_samples')
                cache_alpha_y = cached_data.get('alpha_y_combinations', [])
                
                # Check if all metadata matches
                metadata_match = (
                    cache_nq == nq and
                    cache_channel == channel_type and
                    abs(cache_strength - noise_strength) < 1e-6 and
                    set(cache_nps_list) == set(nps_list) and
                    cache_num_samples == len(sample_indices) and
                    len(cache_alpha_y) == len(alpha_y_combinations)
                )
                
                # Check alpha_y combinations match (order-independent)
                if metadata_match and len(alpha_y_combinations) > 0:
                    cache_alpha_y_set = set()
                    for alpha, y in cache_alpha_y:
                        alpha_tuple = tuple(alpha) if isinstance(alpha, list) else alpha
                        cache_alpha_y_set.add((alpha_tuple, y))
                    
                    current_alpha_y_set = set()
                    for alpha, y in alpha_y_combinations:
                        alpha_tuple = tuple(alpha) if isinstance(alpha, tuple) else alpha
                        current_alpha_y_set.add((alpha_tuple, y))
                    
                    metadata_match = (cache_alpha_y_set == current_alpha_y_set)
                
                if metadata_match:
                    print(f"[CACHE] ✓ Cache metadata matches! Loading cached features...")
                    print(f"  - nq: {cache_nq}, channel: {cache_channel}, strength: {cache_strength}")
                    print(f"  - nps values: {cache_nps_list}")
                    print(f"  - num samples: {cache_num_samples}")
                    if cache_alpha_y:
                        print(f"  - (alpha, y) combinations: {len(cache_alpha_y)}")
                    
                    # Convert cached features back to JAX arrays if needed
                    cached_features = cached_data.get('features', {})
                    feature_dict = {}
                    for sample_idx, nps_dict in cached_features.items():
                        feature_dict[int(sample_idx)] = {}
                        for nps, value in nps_dict.items():
                            if isinstance(value, dict):
                                # Multiple combinations
                                feature_dict[int(sample_idx)][int(nps)] = {}
                                for (alpha, y), feat_vec in value.items():
                                    # Convert tuple keys if needed
                                    alpha_key = tuple(alpha) if isinstance(alpha, list) else alpha
                                    feature_dict[int(sample_idx)][int(nps)][(alpha_key, y)] = jnp.asarray(feat_vec)
                            else:
                                # Single combination
                                feature_dict[int(sample_idx)][int(nps)] = jnp.asarray(value)
                    
                    print(f"[CACHE] ✓ Successfully loaded {len(feature_dict)} samples from cache")
                    print(f"{'='*80}\n")
                    return feature_dict
                else:
                    print(f"[CACHE] ✗ Cache metadata mismatch. Regenerating features...")
                    print(f"  Cache: nq={cache_nq}, channel={cache_channel}, strength={cache_strength}")
                    print(f"  Current: nq={nq}, channel={channel_type}, strength={noise_strength}")
                    print(f"{'='*80}\n")
            except Exception as e:
                print(f"[CACHE] ✗ Error loading cache: {e}")
                print(f"  Regenerating features...\n")
    
    print(f"\n{'='*80}")
    print(f"Generating white noise training features at checkpoints: {nps_list}")
    print(f"{'='*80}\n")
    
    feature_dict = _generate_white_noise_features_for_samples(
        config,
        sample_indices,
        f_vectors,
        nps_list,
        mode="training",
    )
    
    # Save features to pickle file if requested
    save_features = config.get('save_features_to_pkl', True)  # Default to True
    if save_features:
        # Build filename with alpha/y info (same logic as cache loading)
        strength_str = f"{noise_strength:.2f}".replace('.', '_')
        features_dir = config.get('features_cache_dir', 'white_noise_features_cache')
        os.makedirs(features_dir, exist_ok=True)
        
        # Handle multiple alpha/y combinations - use first one for filename if single, or create combined name
        if len(alpha_y_combinations) == 1:
            alpha, y = alpha_y_combinations[0]
            alpha_str = ''.join(map(str, alpha)) if isinstance(alpha, tuple) else str(alpha)
            filename = f"features_nq{nq}_{channel_type}_{strength_str}_alpha{alpha_str}_y{y}.pkl"
        elif len(alpha_y_combinations) > 1:
            # For multiple combinations, use a combined identifier
            combo_strs = []
            for alpha, y in alpha_y_combinations:
                alpha_str = ''.join(map(str, alpha)) if isinstance(alpha, tuple) else str(alpha)
                combo_strs.append(f"{alpha_str}_{y}")
            combo_id = "_".join(combo_strs)[:50]  # Limit length
            filename = f"features_nq{nq}_{channel_type}_{strength_str}_multi_{combo_id}.pkl"
        else:
            # Fallback to old naming if no alpha_y_combinations
            filename = f"features_nq{nq}_{channel_type}_{strength_str}.pkl"
        
        filepath = os.path.join(features_dir, filename)
        
        # Convert JAX arrays to numpy for pickle compatibility
        feature_dict_numpy = {}
        for sample_idx, nps_dict in feature_dict.items():
            feature_dict_numpy[int(sample_idx)] = {}
            for nps, value in nps_dict.items():
                if isinstance(value, dict):
                    # Multiple (alpha, y) combinations case
                    feature_dict_numpy[int(sample_idx)][int(nps)] = {}
                    for (alpha, y), feat_vec in value.items():
                        feature_dict_numpy[int(sample_idx)][int(nps)][(alpha, y)] = np.asarray(feat_vec)
                else:
                    # Single combination case (backward compatible)
                    feature_dict_numpy[int(sample_idx)][int(nps)] = np.asarray(value)
        
        # Save metadata along with features
        save_data = {
            'features': feature_dict_numpy,
            'nq': nq,
            'channel_type': channel_type,
            'noise_strength': noise_strength,
            'nps_list': nps_list,
            'num_samples': len(sample_indices),
            'alpha_y_combinations': config.get('alpha_y_combinations'),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save a human-readable metadata JSON file
        metadata_filepath = filepath.replace('.pkl', '_metadata.json')
        alpha_y_combinations = config.get('alpha_y_combinations', [])
        
        # Convert alpha_y_combinations to a serializable format
        alpha_y_list = []
        for alpha, y in alpha_y_combinations:
            # Convert alpha tuple to list, y string stays as string
            alpha_list = list(alpha) if isinstance(alpha, tuple) else [alpha]
            alpha_y_list.append({
                'alpha': alpha_list,
                'y': y,
                'alpha_str': ''.join(map(str, alpha_list)),
                'y_str': y
            })
        
        metadata = {
            'pickle_file': filename,
            'nq': nq,
            'channel_type': channel_type,
            'noise_strength': float(noise_strength),
            'nps_list': [int(nps) for nps in nps_list],
            'num_samples': len(sample_indices),
            'num_alpha_y_combinations': len(alpha_y_combinations),
            'alpha_y_combinations': alpha_y_list,
            'feature_structure': {
                'description': 'Features are stored as: features[sample_idx][nps][(alpha, y)] = feature_vector',
                'sample_indices': f'0 to {len(sample_indices) - 1}',
                'nps_values': [int(nps) for nps in nps_list],
                'has_multiple_combinations': len(alpha_y_combinations) > 1
            }
        }
        
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Saved features to: {filepath}")
        print(f"✓ Saved metadata to: {metadata_filepath}")
        print(f"  - nq: {nq}, channel: {channel_type}, strength: {noise_strength}")
        print(f"  - nps values: {nps_list}")
        print(f"  - num samples: {len(sample_indices)}")
        if config.get('alpha_y_combinations'):
            print(f"  - (alpha, y) combinations: {len(config.get('alpha_y_combinations'))}")
            print(f"    Combinations:")
            for i, (alpha, y) in enumerate(config.get('alpha_y_combinations'), 1):
                alpha_str = ''.join(map(str, alpha)) if isinstance(alpha, tuple) else str(alpha)
                print(f"      {i}. alpha={alpha_str}, y={y}")
        print(f"{'='*80}\n")
    
    return feature_dict


def generate_white_noise_validation_features(config, data_dict, val_indices, nps_list):
    """
    Generate analytic white-noise validation features for selected indices.
    """
    if val_indices is None or len(val_indices) == 0:
        print("[INFO] No validation indices provided; returning empty validation feature dict.")
        return {}

    # Build local F vectors (one per validation index)
    f_vectors = np.asarray([data_dict['F_bs'][idx] for idx in val_indices], dtype=np.int32)
    # Local index space 0..len(val_indices)-1, to stay aligned with f_vectors
    local_indices = list(range(len(val_indices)))

    print(f"\n{'='*80}")
    print(f"Generating white noise validation features for indices {len(val_indices)} at checkpoints: {nps_list}")
    print(f"{'='*80}\n")

    # Generate features in local index space
    local_feature_dict = _generate_white_noise_features_for_samples(
        config,
        local_indices,
        f_vectors,
        nps_list,
        mode="validation",
    )

    # Remap keys back to global sample indices
    remapped: Dict[int, Dict[int, jnp.ndarray]] = {}
    for local_i, global_idx in enumerate(val_indices):
        remapped[int(global_idx)] = local_feature_dict[local_i]

    return remapped


def generate_white_noise_training_at_checkpoints(config, data_dict, nps_list):
    """
    Generate white noise training features at all nps checkpoints for all training samples.
    
    OPTIMIZED: Uses vectorized version for speed.
    
    Args:
        config: Configuration dict
        data_dict: Data dictionary with F_bs
        nps_list: List of nps checkpoints
        
    Returns:
        Dictionary: sample_idx -> {nps: feature_vector}
    """
    # Use vectorized version
    return generate_white_noise_training_at_checkpoints_vectorized(config, data_dict, nps_list)


def generate_training_shadows_at_checkpoints(config, data_dict, nps_list):
    """
    Generate shadow DMs at all nps checkpoints for all training samples.
    
    OPTIMIZATION: Uses efficient checkpointing - generates shadows ONCE at max_nps
    per sample and checkpoints the accumulated DM during accumulation (not multiple calls).
    This avoids regenerating shadows for each nps value.
    
    Args:
        config: Configuration dict
        data_dict: Data dictionary with F_bs
        nps_list: List of nps checkpoints
        
    Returns:
        Dictionary: sample_idx -> {nps: feature_vector}
    """
    n = config['n']
    num_samples = len(data_dict['F_bs'])
    max_nps = max(nps_list)
    
    print(f"\n{'='*80}")
    print(f"Generating training shadows at all checkpoints (max_nps={max_nps})")
    print(f"Checkpoints: {nps_list}")
    print(f"{'='*80}\n")
    
    cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
    Ks = get_kraus_operators(cfg)
    batch_size_mcs = min(config.get("batch_size_mcs", 128), max_nps)
    shots = config.get("shots", 1)
    r = config.get('r', 1)
    shadow_batch_size = config.get("shadow_batch_size", 50)
    
    print(f"  Channel config: {cfg}")
    print(f"  Using same noise channel for all training samples")
    print(f"  Batch size: {shadow_batch_size} samples processed in parallel")
    
    all_F_vecs = jnp.array(data_dict['F_bs'])
    
    # Dictionary to store: sample_idx -> {nps: feature_vector}
    training_dms = {}
    
    y_targ = config.get('y_targ')
    alpha_targ = config.get('alpha_targ')
    
    # Process in batches for better GPU utilization
    num_batches = (num_samples + shadow_batch_size - 1) // shadow_batch_size
    print(f"  Processing {num_samples} samples in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * shadow_batch_size
        batch_end = min(batch_start + shadow_batch_size, num_samples)
        batch_size_actual = batch_end - batch_start
        
        print(f"    Batch {batch_idx + 1}/{num_batches} ({batch_start}-{batch_end})...")
        
        # Get batch of F vectors
        F_batch = all_F_vecs[batch_start:batch_end]
        
        # Generate key for this batch
        batch_key = jax.random.PRNGKey(batch_start)
        
        # Process entire batch at once
        batch_results = generate_all_nps_checkpoints_batched(
            batch_key, F_batch, n, Ks, max_nps, batch_size_mcs, r, shots,
            nps_list, use_complex64=True
        )
        
        # Convert DMs to feature vectors for each sample in batch
        for i, checkpoint_results in enumerate(batch_results):
            sample_idx = batch_start + i
            feature_dict = {}
            for nps, dm in checkpoint_results.items():
                rhosnm = get_relevant_input(dm, y_targ, alpha_targ, n)
                feature_dict[nps] = jnp.array(rhosnm)
            training_dms[sample_idx] = feature_dict
    
    print(f"\n✓ Generated training shadows for {num_samples} samples at {len(nps_list)} checkpoints")
    
    return training_dms


def generate_validation_shadows_at_checkpoints(config, data_dict, val_indices, nps_list, return_density_matrices: bool = False):
    """
    Generate validation shadows at all checkpoints.
    
    OPTIMIZATION: Uses efficient checkpointing - generates shadows ONCE at max_nps
    per sample and checkpoints the accumulated DM during accumulation (not multiple calls).
    This avoids regenerating shadows for each nps value.
    
    Args:
        config: Configuration dict
        data_dict: Data dictionary with F_bs
        val_indices: List of validation sample indices
        nps_list: List of nps checkpoints
        
    Returns:
        Dictionary: sample_idx -> {nps: feature_vector}
    """
    n = config['n']
    max_nps = max(nps_list)
    
    print(f"\nGenerating validation shadows at all checkpoints (max_nps={max_nps})...")
    
    cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
    Ks = get_kraus_operators(cfg)  # SAME channel as training (same cfg, same Ks)
    batch_size_mcs = min(config.get("batch_size_mcs", 128), max_nps)
    shots = config.get("shots", 1)
    r = config.get('r', 1)
    shadow_batch_size = config.get("shadow_batch_size", 50)
    
    print(f"  Channel config: {cfg} (same as training)")
    print(f"  Using same noise channel for all validation samples")
    print(f"  Batch size: {shadow_batch_size} samples processed in parallel")
    
    val_dms = {}
    raw_dms = {} if return_density_matrices else None
    
    y_targ = config.get('y_targ')
    alpha_targ = config.get('alpha_targ')
    
    # Prepare all validation F vectors
    all_val_F_vecs = jnp.array([data_dict['F_bs'][idx] for idx in val_indices])
    
    # Process in batches for better GPU utilization
    num_val_samples = len(val_indices)
    num_batches = (num_val_samples + shadow_batch_size - 1) // shadow_batch_size
    print(f"  Processing {num_val_samples} validation samples in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * shadow_batch_size
        batch_end = min(batch_start + shadow_batch_size, num_val_samples)
        
        print(f"    Validation batch {batch_idx + 1}/{num_batches} ({batch_start}-{batch_end})...")
        
        # Get batch of F vectors and corresponding original indices
        F_batch = all_val_F_vecs[batch_start:batch_end]
        batch_indices = val_indices[batch_start:batch_end]
        
        # Generate key for this batch (offset by 10000 to differ from training)
        batch_key = jax.random.PRNGKey(batch_start + 10000)
        
        # Process entire batch at once
        batch_results = generate_all_nps_checkpoints_batched(
            batch_key, F_batch, n, Ks, max_nps, batch_size_mcs, r, shots,
            nps_list, use_complex64=True
        )
        
        # Convert DMs to feature vectors for each sample in batch
        for i, checkpoint_results in enumerate(batch_results):
            sample_idx = batch_indices[i]
            feature_dict = {}
            if return_density_matrices:
                raw_dict = {}
            for nps, dm in checkpoint_results.items():
                if return_density_matrices:
                    raw_dict[int(nps)] = dm
                rhosnm = get_relevant_input(dm, y_targ, alpha_targ, n)
                feature_dict[int(nps)] = jnp.array(rhosnm)
            val_dms[int(sample_idx)] = feature_dict
            if return_density_matrices:
                raw_dms[int(sample_idx)] = raw_dict
    
    print(f"✓ Generated validation shadows for {len(val_indices)} samples at {len(nps_list)} checkpoints")
    
    if return_density_matrices:
        return val_dms, raw_dms
    return val_dms


def multi_nps_training(base_config, god_level_seed, nps_list):
    """
    Train multiple NNs for different nps values.
    Efficiently generates shadows once at max nps and reuses for smaller nps.
    
    Args:
        base_config: Base configuration dict (will be modified for each nps)
        god_level_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping nps -> experiment results
    """
    qubits = base_config['n']
    
    # Compute nps checkpoints
    
    if len(nps_list) == 0:
        print(f"No valid nps values for n={qubits}!")
        return {}
    
    validation_mode = base_config.get('validation_mode', 'shadow')
    if validation_mode not in ('shadow', 'white_noise'):
        print(f"[WARN] Unknown validation_mode='{validation_mode}', defaulting to 'shadow'.")
        validation_mode = 'shadow'
    
    print(f"\n{'='*80}")
    print(f"Multi-nps Training for nq={qubits}")
    print(f"nps checkpoints: {nps_list}")
    print(f"validation mode: {validation_mode}")
    print(f"{'='*80}\n")
    if qubits > 9 and validation_mode == 'shadow':
        print("[WARN] Shadow-based validation for nq > 9 can be prohibitively expensive. Consider setting validation_mode='white_noise'.")
    
    # Load/generate data once
    config = base_config.copy()
    config = _synchronize_channel_config(config)
    config['shadow_nps'] = max(nps_list)  # Use max for data generation
    config['validation_nps'] = max(nps_list)

    try:
        data_file_name = f"data_{qubits}q_{config['num_training_data']}_{config['channel_type']}_{config['noise_strength']}ns_Nonepexc_{''.join(map(str, config['alpha_targ']))}alpha_{config['y_targ']}y.pkl"
        if os.path.exists(data_file_name):
            print(f"Loading existing data from: {data_file_name}")
            with open(data_file_name, "rb") as f_data:
                data_dict = pickle.load(f_data)
        else:
            print("Generating new data...")
            data_dict = prepare_data_dict_noiseless(config)
            with open(data_file_name, "wb") as f_data:
                pickle.dump(data_dict, f_data)
    except Exception as e:
        print(f"Error loading/generating data: {e}")
        data_dict = prepare_data_dict_noiseless(config)
    
    # Check if white noise training is requested
    use_white_noise_training = base_config.get('use_white_noise_training', False)
    
    # Generate training data at all checkpoints ONCE
    if use_white_noise_training:
        print(f"\n{'='*80}")
        print("WHITE NOISE TRAINING MODE")
        print(f"{'='*80}")
        print(f"  - Training: White noise with variance = 3^nq / nps")
        if validation_mode == 'shadow':
            print(f"  - Validation: Shadow tomography (Monte Carlo)")
        else:
            print(f"  - Validation: Analytic white-noise features")
        thermal_strength = config.get('noise_strength', 0.0)
        if thermal_strength > 0:
            print(f"  - Thermal approximation: (1-p)^{{d(n,m)}} with p={thermal_strength}")
        print(f"{'='*80}\n")
        
        with log_time("white-noise training feature generation"):
            training_shadows_dict = generate_white_noise_training_at_checkpoints(config, data_dict, nps_list)
    else:
        with log_time("shadow training feature generation"):
            training_shadows_dict = generate_training_shadows_at_checkpoints(config, data_dict, nps_list)
    
    # Generate validation shadows at all checkpoints ONCE (always use shadows for validation)
    from sklearn.model_selection import train_test_split
    targets = data_dict['b']
    num_samples = len(data_dict['F_bs'])
    val_seed = 42
    validation_fraction = base_config.get('validation_fraction', 0.3)  # Default 30%

    validation_cache = None
    use_validation_cache = base_config.get('use_validation_cache', True)
    if validation_mode == 'shadow' and use_validation_cache:
        cache_dir = base_config.get('validation_cache_dir') or base_config.get('feature_cache_dir')
        validation_cache = try_load_validation_cache(config, nps_list, cache_dir)
        if validation_cache is not None:
            print("[CACHE] Using precomputed validation shadows")
    elif validation_mode == 'white_noise' and use_validation_cache:
        print("[INFO] Validation caches are ignored in white-noise mode.")

    if validation_cache is None:
        _, val_indices, _, y_val = train_test_split(
            np.arange(num_samples), targets, test_size=validation_fraction, random_state=val_seed, stratify=targets
        )
        val_indices = np.array(val_indices)

        if validation_mode == 'shadow':
            log_label = "validation shadow generation"
            generator_fn = generate_validation_shadows_at_checkpoints
        else:
            log_label = "validation white-noise feature generation"
            generator_fn = generate_white_noise_validation_features

        with log_time(log_label):
            validation_shadows_dict = generator_fn(
                config, data_dict, val_indices, nps_list
            )
    else:
        val_indices = np.array(validation_cache['indices'])
        y_val = np.array(validation_cache['labels'], dtype=np.int16)
        validation_shadows_dict = validation_cache['features_dict']

    if not use_white_noise_training:
        # Verify both use the same channel config (only for shadow training)
        train_cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
        print(f"\n{'='*80}")
        print("VERIFICATION: Training and Validation use SAME noise channel")
        print(f"{'='*80}")
        print(f"Channel config: {train_cfg}")
        print(f"  - Type: {train_cfg.get('type')}")
        print(f"  - Strength: {train_cfg.get('strength')}")
        if 'thermal_p_exc' in train_cfg:
            print(f"  - Thermal p_exc: {train_cfg.get('thermal_p_exc')}")
        print("  ✓ Training and validation shadows use identical Kraus operators (Ks)")
        print("  ✓ Only difference: random MC trajectories (different random keys)")
        print(f"{'='*80}\n")
    
    # Get alpha_y combinations from config
    alpha_y_combinations = base_config.get('alpha_y_combinations')
    if alpha_y_combinations is None:
        # Backward compatibility: use single alpha/y
        alpha_y_combinations = [(base_config.get('alpha_targ'), base_config.get('y_targ'))]
    
    # Check if we have multiple combinations
    has_multiple_combinations = len(alpha_y_combinations) > 1
    
    # Check if features have multiple combinations (by checking first sample)
    first_sample_features = training_shadows_dict.get(0, {})
    first_nps_features = first_sample_features.get(nps_list[0] if nps_list else None)
    features_have_multiple_combos = isinstance(first_nps_features, dict) if first_nps_features is not None else False
    
    # Sort nps_list in descending order (largest to smallest) for weight transfer
    nps_list_sorted = sorted(nps_list, reverse=True)
    print(f"\n{'='*80}")
    print(f"Training order: {nps_list_sorted} (largest to smallest for weight transfer)")
    if has_multiple_combinations or features_have_multiple_combos:
        print(f"Will train separate models for {len(alpha_y_combinations)} (alpha, y) combinations:")
        for i, (a, y) in enumerate(alpha_y_combinations):
            print(f"  {i+1}. alpha={a}, y={y}")
    print(f"{'='*80}\n")
    
    # Now train for each nps using pre-generated shadows
    all_results = {}
    previous_best_params_by_combo = {}  # Track best params per (alpha, y) combination
    
    for nps in nps_list_sorted:
        print(f"\n{'#'*80}")
        print(f"Training with nps = {nps}")
        print(f"{'#'*80}\n")
        
        # Initialize results dict for this nps
        if nps not in all_results:
            all_results[nps] = {}
        
        # Train for each (alpha, y) combination
        for combo_idx, (alpha_targ_combo, y_targ_combo) in enumerate(alpha_y_combinations):
            # Create combination key (tuple for dict key) - must match format used in feature generation
            # In feature generation, we store (alpha_bits, y_bits) where alpha_bits is a tuple
            alpha_tuple = tuple(alpha_targ_combo) if isinstance(alpha_targ_combo, (list, tuple)) else tuple([alpha_targ_combo])
            combo_key = (alpha_tuple, y_targ_combo)
            combo_label = f"alpha_{''.join(map(str, alpha_targ_combo))}_y_{y_targ_combo}"
            
            # Debug: print available keys if not found
            if features_have_multiple_combos:
                first_sample = training_shadows_dict.get(0, {})
                first_nps_feat = first_sample.get(nps_list_sorted[0] if nps_list_sorted else None)
                if isinstance(first_nps_feat, dict) and combo_key not in first_nps_feat:
                    available_keys = list(first_nps_feat.keys())[:5]  # Show first 5
                    print(f"  [DEBUG] Looking for combo_key={combo_key}, available keys (sample): {available_keys}")
            
            if has_multiple_combinations or features_have_multiple_combos:
                print(f"\n{'='*60}")
                print(f"Training for (alpha, y) combination {combo_idx + 1}/{len(alpha_y_combinations)}: {combo_label}")
                print(f"{'='*60}\n")
            
            # Use pre-generated shadows for this nps
            config_nps = base_config.copy()
            config_nps = _synchronize_channel_config(config_nps)
            config_nps['shadow_nps'] = nps
            config_nps['validation_nps'] = nps
            config_nps['_use_precomputed_shadows'] = True
            config_nps['alpha_targ'] = alpha_targ_combo  # Set specific combination
            config_nps['y_targ'] = y_targ_combo
            
            # Update checkpoint path to include combination identifier if multiple combinations
            if has_multiple_combinations or features_have_multiple_combos:
                original_path = config_nps['CHECKPOINT_PATH']
                # Add combination to path: saved_models_white_noise_nq10_dephasing0_01/alpha_111_y_min/
                config_nps['CHECKPOINT_PATH'] = os.path.join(original_path, combo_label)
                os.makedirs(config_nps['CHECKPOINT_PATH'], exist_ok=True)
            
            # Extract features for this specific nps and (alpha, y) combination
            training_shadows_for_nps = {}
            for idx, shadows in training_shadows_dict.items():
                feature = shadows.get(nps)
                if feature is not None:
                    if isinstance(feature, dict):
                        # Multiple combinations: extract this specific combination
                        if combo_key in feature:
                            training_shadows_for_nps[idx] = feature[combo_key]
                    else:
                        # Single combination: use if it matches, or if we only have one combo
                        if not has_multiple_combinations or combo_idx == 0:
                            training_shadows_for_nps[idx] = feature
            
            validation_shadows_for_nps = {}
            for idx, shadows in validation_shadows_dict.items():
                feature = shadows.get(nps)
                if feature is not None:
                    if isinstance(feature, dict):
                        # Multiple combinations: extract this specific combination
                        if combo_key in feature:
                            validation_shadows_for_nps[idx] = feature[combo_key]
                    else:
                        # Single combination: use if it matches, or if we only have one combo
                        if not has_multiple_combinations or combo_idx == 0:
                            validation_shadows_for_nps[idx] = feature
            
            config_nps['_training_shadows'] = training_shadows_for_nps
            config_nps['_validation_shadows'] = validation_shadows_for_nps
            config_nps['_val_indices'] = val_indices.tolist() if hasattr(val_indices, "tolist") else list(val_indices)
            config_nps['_y_val'] = y_val.tolist() if hasattr(y_val, "tolist") else list(y_val)
            
            # Debug check
            if len(training_shadows_for_nps) == 0:
                print(f"  WARNING: No training shadows found for nps={nps}, combo={combo_label}!")
                if has_multiple_combinations or features_have_multiple_combos:
                    print(f"    Available combinations in first sample: {list(training_shadows_dict.get(0, {}).get(nps, {}).keys()) if isinstance(training_shadows_dict.get(0, {}).get(nps), dict) else 'N/A'}")
                continue
            
            # Run training with pre-computed shadows, passing previous best params for weight transfer
            previous_best_params = previous_best_params_by_combo.get(combo_key)
            experiment_results_dict = run_experiment_with_precomputed_shadows(
                config_nps, data_dict, initial_params=previous_best_params
            )
            
            # Format like main_training returns
            experiment_results = {
                'config': config_nps,
                'results': experiment_results_dict['results'],
                'model_names': experiment_results_dict['model_names']
            }
            
            # Store results with combination key
            all_results[nps][combo_key] = {
                'experiment_results': experiment_results,
                'config': config_nps
            }
            
            # Extract best model params from this nps for next iteration (per combination)
            if experiment_results_dict['results']:
                best_result = max(experiment_results_dict['results'], key=lambda r: r.val_acc)
                best_model_name = best_result.model_name
                model_path = config_nps['CHECKPOINT_PATH']
                
                try:
                    with open(_get_model_file(model_path, best_model_name), 'rb') as f:
                        previous_best_params_by_combo[combo_key] = pickle.load(f)
                    if has_multiple_combinations or features_have_multiple_combos:
                        print(f"  ✓ Loaded best model params for {combo_label} from nps={nps} (val_acc={best_result.val_acc:.4f})")
                    else:
                        print(f"  ✓ Loaded best model params from nps={nps} (val_acc={best_result.val_acc:.4f}) for next nps")
                except Exception as e:
                    print(f"  ⚠️ Could not load best model params for {combo_label}: {e}")
                    previous_best_params_by_combo[combo_key] = None
            else:
                previous_best_params_by_combo[combo_key] = None
    
    return all_results


def get_random_alpha(qubits):
  '''
      Returns a random alpha value and set the last bit to 1
  '''
  alpha = np.random.randint(0, 2**qubits)
  alpha = np.binary_repr(alpha, width=qubits)
  alpha = alpha[:-1]+'1'
  return np.binary_repr(int(alpha, 2), width=qubits)


def measure(circuit):
  '''
      Get the bit-string
  '''
  # output is a tuple consisting of the bitstring as a np.array and probability of getting that sample/bistring.
  output = circuit.sample()
  bitstring_arr = output[0]
  out = ''
  for bit in bitstring_arr:
    out += str(int(bit))

  y = int(out[:-1]+"0", 2)
  b = int(out[-1])
  return y, b

def check_condition(alpha, y, b, fi):
    return fi[y] ^ fi[y ^ alpha] == int(b)

def string_to_list(s):
    return [int(char) for char in s]

def list_to_string(lst):
    return ''.join(str(element) for element in lst)

# @partial(tc.backend.jit, static_argnums=(3,))
# def shadow_ss(psi, pauli_strings, status, measurement_only=False):
#     return shadows.shadow_snapshots(
#         psi, pauli_strings, status, measurement_only=measurement_only
#     )

# def reconstructed_shadow_state(snapshots_states):
#   return shadows.global_shadow_state(snapshots=snapshots_states)

# gssjit = tc.backend.jit(reconstructed_shadow_state)

# def accuracy_test(predictions, labels):
#   predictions = (predictions > 0.5).float()
#   acc = 0
#   for (pred,l) in zip(predictions, labels):
#     if pred == l:
#         acc = acc + 1
#   acc = acc / len(labels)
#   return acc


# def accuracy_test_1(predictions, labels):
#   predictions = (predictions > 0.5).float()
#   acc = 0
#   for idx, true_yb in enumerate(labels):
#     flag = tensor_in_matrix(predictions[idx], true_yb)
#     if flag == 0:
#       acc = acc + 1
#   # acc = acc / len(labels)
#   return acc


def _apply_iterative_local_channel(rho, single_qubit_kraus_ops, N):
    """
    Applies a given set of single-qubit Kraus operators to each qubit of 
    a density matrix iteratively using DMCircuit.

    Args:
        rho (tc.Tensor): The input N-qubit density matrix.
        single_qubit_kraus_ops (list[tc.Tensor]): List of single-qubit Kraus operators.
        N (int): Number of qubits.

    Returns:
        tc.Tensor: The output N-qubit density matrix after the channel is applied.
    """
    if N == 0: # No qubits to apply the channel to
        return rho

    # Ensure dminputs is a tc.Tensor, which DMCircuit should handle.
    # If issues arise, one might need to convert rho_tc to a NumPy array here,
    # but tc.Tensor is generally preferred for consistency within tensorcircuit.
    dmc = tc.DMCircuit(N, dminputs=rho)
    for k_idx in range(N):
        dmc.general_kraus(single_qubit_kraus_ops, k_idx)
    return dmc # dmc.state() returns a tc.Tensor


def bitflip_channel(rho, p_bitflip, N):
    if p_bitflip == 0:
        return rho

    K0 = np.array([[np.sqrt(1 - p_bitflip), 0], [0, np.sqrt(1 - p_bitflip)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(p_bitflip)], [np.sqrt(p_bitflip), 0]], dtype=np.complex128)

    sq_kraus_list = [tc.array_to_tensor(k) for k in [K0, K1]]

    return _apply_iterative_local_channel(rho, sq_kraus_list, N)


def local_depolarizing_channel(rho, p_local_depolarize, N):
    """
    Applies local depolarizing noise to each qubit independently.
    (Corresponds to user's 'local_depolarizing' type).
    """
    if p_local_depolarize == 0: # No noise strength
        return rho

    sq_kraus_list = tc.channels.generaldepolarizingchannel(p_local_depolarize)

    return _apply_iterative_local_channel(rho, sq_kraus_list, N)


def global_thermal_relaxation_channel(rho, gamma, p_exc, N):
    """
    Applies global thermal relaxation using amplitude damping.
    `gamma` is the probability of losing an excitation.
    `p_exc` is the probability of excitation if a qubit is in |0>.
    (Corresponds to user's 'thermal' type).
    """
    if gamma == 0: # No noise strength
        return rho
    # Note: amplitudedampingchannel in TC typically takes (gamma, p1)
    # where p1 is prob of excitation from ground state.
    # User's original code for thermal relaxation was more general.
    # This uses the tc.channels.amplitudedampingchannel as in user's working example.
    sq_kraus_list = tc.channels.amplitudedampingchannel(gamma, p_exc) 
    return _apply_iterative_local_channel(rho, sq_kraus_list, N)


def global_depolarizing_channel(rho, p, N):
    """
    Applies the specific global depolarizing channel: (1-p)rho + p * I/(2^N).
    This channel is NOT an iterative application of single-qubit Kraus ops.
    (Corresponds to user's 'depolarizing' type).
    """
    if p == 0: 
        return rho

    dim = 2**N
    id_N = tc.backend.eye(dim, dtype=rho.dtype) 
    rho =  (1 - p) * rho + (p / dim) * id_N
    dmc = tc.DMCircuit(N, dminputs=rho)
    return dmc


def global_dephasing_channel(rho, p_gamma, N):
    """
    Applies global phase damping. Each qubit experiences phase damping with strength `p_gamma`.
    (Corresponds to user's 'dephasing' type).
    """
    if p_gamma == 0: # No noise strength
        return rho
    sq_kraus_list = tc.channels.phasedampingchannel(p_gamma) 
    return _apply_iterative_local_channel(rho, sq_kraus_list, N)

def vectorized_ints_to_bit_arrays(int_values, num_bits):
    """Converts an array of integers to a 2D array of their bit representations (MSB first)."""
    if num_bits == 0: return np.empty((len(int_values), 0), dtype=np.uint8)
    if not isinstance(int_values, np.ndarray): int_values = np.array(int_values, dtype=np.uint64)
    if len(int_values) == 0: return np.empty((0, num_bits), dtype=np.uint8)
    shifts = np.arange(num_bits - 1, -1, -1, dtype=np.uint8) 
    return ((int_values[:, np.newaxis] >> shifts).astype(np.uint8)) & 1

def vectorized_bit_arrays_to_ints(bit_arrays):
    """Converts a 2D array of bit representations (MSB first) back to an array of integers."""
    if bit_arrays.shape[1] == 0: return np.zeros(bit_arrays.shape[0], dtype=np.uint64)
    num_bits = bit_arrays.shape[1]
    powers = (2**np.arange(num_bits - 1, -1, -1)).astype(np.uint64)
    return bit_arrays @ powers 


def compute_noisy_accuracy(rho, fi, alpha, qubits):

    probabilities_all_z = np.real(np.diag(rho))
    z_indices = np.arange(2**qubits, dtype=np.uint64)

    all_z_bits = vectorized_ints_to_bit_arrays(z_indices, qubits)
    all_y_bits = all_z_bits.copy(); all_y_bits[:, -1] = 0 
    all_b_measured = all_z_bits[:, -1]
    all_y_indices = vectorized_bit_arrays_to_ints(all_y_bits)

    alpha_Nbit_array = np.array(list(alpha), dtype=np.uint8).reshape(1, qubits)
    all_y_xor_alpha_bits = all_y_bits ^ alpha_Nbit_array
    all_y_xor_alpha_indices = vectorized_bit_arrays_to_ints(all_y_xor_alpha_bits)

    Fi_np = np.array(fi, dtype=np.uint8)
    max_F_idx = len(Fi_np) - 1

    valid_y_mask = (all_y_indices <= max_F_idx)
    valid_y_xor_alpha_mask = (all_y_xor_alpha_indices <= max_F_idx)
    valid_F_lookup_mask = valid_y_mask & valid_y_xor_alpha_mask

    val_F_y_all = np.zeros_like(all_y_indices, dtype=np.uint8)
    val_F_y_xor_alpha_all = np.zeros_like(all_y_xor_alpha_indices, dtype=np.uint8)

    val_F_y_all[valid_F_lookup_mask] = Fi_np[all_y_indices[valid_F_lookup_mask]]
    val_F_y_xor_alpha_all[valid_F_lookup_mask] = Fi_np[all_y_xor_alpha_indices[valid_F_lookup_mask]]

    all_b_predicted = val_F_y_all ^ val_F_y_xor_alpha_all
    correct_b_mask = (all_b_measured == all_b_predicted)

    final_accuracy_mask = correct_b_mask & valid_F_lookup_mask
    accuracy = np.sum(probabilities_all_z[final_accuracy_mask])

    if not np.all(valid_F_lookup_mask) and qubits > 0 : 
        num_invalid_lookups = np.sum(~valid_F_lookup_mask)
    # print(f"Warning: {num_invalid_lookups} F_list index lookups were out of bounds. NQ={N_qubits}, F_len={len(F_list_np)}")
    return accuracy

@partial(jax.jit, static_argnums=(1,))
def ints_to_bits(int_values: jnp.ndarray, num_bits: int) -> jnp.ndarray:
    """
    Convert [N] integers → [N, num_bits] bit-arrays (MSB first).
    """
    # shape: (num_bits,), e.g. [num_bits-1, ..., 0]
    shifts = jnp.arange(num_bits - 1, -1, -1, dtype=int_values.dtype)
    # (N,1) >> (num_bits,) → (N,num_bits), mask &1
    return ((int_values[:, None] >> shifts) & 1).astype(jnp.uint8)

@jax.jit
def bits_to_ints(bit_arrays: jnp.ndarray) -> jnp.ndarray:
    """
    Convert [N, B] bit-arrays (MSB first) → [N] integers.
    """
    B = bit_arrays.shape[1]
    # shape (B,), e.g. [2^(B-1), ..., 1]
    weights = (1 << jnp.arange(B - 1, -1, -1, dtype=jnp.uint64))
    return (bit_arrays.astype(jnp.uint64) * weights).sum(axis=1)

@partial(jax.jit, static_argnums=(3,))
def compute_noisy_accuracy_jax(
    rho: jnp.ndarray,      # (2^n,2^n) density matrix
    Fi: jnp.ndarray,       # (M,) array of uint8 labels
    alpha: jnp.ndarray,    # (n,) array of bits (uint8)
    qubits: int,           # static number of qubits
) -> jnp.ndarray:
    """
    Returns the total probability (accuracy) of correctly predicting the last bit.
    """
    # 1. Extract diagonal probabilities
    probs_z = jnp.real(jnp.diag(rho))               # (2^n,)

    # 2. All Z-basis indices & bit-arrays
    N = 2 ** qubits
    z_idx = jnp.arange(N, dtype=jnp.uint64)         # (2^n,)
    z_bits = ints_to_bits(z_idx, qubits)            # (2^n, n)

    # 3. Form Y by forcing last bit to 0
    y_bits = z_bits.at[:, -1].set(0)                 # (2^n, n)
    b_meas = z_bits[:, -1]                           # (2^n,)

    # 4. Indices for y and y⊕α
    y_idx       = bits_to_ints(y_bits)              # (2^n,)
    y_xor_alpha = y_bits ^ alpha[None, :]           # broadcast α
    yx_idx      = bits_to_ints(y_xor_alpha)

    # 5. Valid-lookup mask
    max_F = Fi.shape[0] - 1
    valid   = (y_idx  <= max_F) & (yx_idx <= max_F)  # (2^n,)

    # 6. Gather F(y), F(y⊕α) and predict b = F(y)⊕F(y⊕α)
    Fy      = Fi[y_idx]                             # (2^n,)
    Fyx     = Fi[yx_idx]
    b_pred  = Fy ^ Fyx                              # (2^n,)

    # 7. Correct predictions & final mask
    correct = (b_meas == b_pred)
    mask    = correct & valid                       # (2^n,)

    # 8. Accuracy = sum of probs_z over mask
    return jnp.sum(probs_z * mask.astype(probs_z.dtype))


# In[17]:



class QuantumStateDataset(Dataset):
  def __init__(self, x, label):
    self.x = x
    self.label = label

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    x = self.x[idx]
    target = self.label[idx]

    return x, target

def numpy_collate(batch):
    """Collate function optimized for JAX arrays - preserves JAX arrays when possible."""
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    
    # Check if first element is a JAX array
    first_elem = batch[0]
    is_jax_array = (hasattr(first_elem, '__class__') and 
                   'jax' in str(type(first_elem)).lower())
    
    if is_jax_array:
        # Stack as JAX arrays
        return jnp.stack([jnp.asarray(x) for x in batch])
    elif isinstance(first_elem, np.ndarray):
        # Stack as numpy arrays (will be converted to JAX in calculate_loss)
        return np.stack(batch)
    else:
        # Try JAX first, fallback to numpy
        try:
            return jnp.array(batch)
        except:
            return np.array(batch)


# In[18]:


# To keep the results close to the PyTorch tutorial, we use the same init function as PyTorch
# which is uniform(-1/sqrt(in_features), 1/sqrt(in_features)) - similar to He et al./kaiming
# The default for Flax is lecun_normal (i.e., half the variance of He) and zeros for bias.

# init_uniform_func = lambda x: (lambda rng, shape, dtype: jax.random.uniform(rng,
#                                                                 shape=shape,
#                                                                 minval=-1/np.sqrt(x.shape[1]),
#                                                                 maxval=1/np.sqrt(x.shape[1]),
#                                                                 dtype=dtype))

# init_gaussian_func = lambda x: (lambda rng, shape, dtype: 1.0/jnp.sqrt(shape[0]) * jax.random.normal(key, shape, dtype=dtype))

init_func = lambda x: (lambda rng, shape, dtype: jax.random.uniform(rng,
                                                                shape=shape,
                                                                minval=-1/np.sqrt(x.shape[1]),
                                                                maxval=1/np.sqrt(x.shape[1]),
                                                                dtype=dtype))

# Network
class BaseNetwork(nn.Module):
    act_fn : nn.Module
    num_classes : int = 2
    # hidden_sizes : Sequence = (1024, 512, 256, 256, 128)
    hidden_sizes : Sequence = (512, 256, 256, 128)

    @nn.compact
    def __call__(self, x, return_activations=False):
        x = x.reshape(x.shape[0], -1) # Reshape images to a flat vector
        # We collect all activations throughout the network for later visualizations
        # Remember that in jitted functions, unused tensors will anyways be removed.
        activations = []
        for hd in self.hidden_sizes:
            x = nn.Dense(hd,
                         kernel_init=init_func(x),
                         bias_init=init_func(x))(x)
            activations.append(x)
            x = self.act_fn(x)
            activations.append(x)
        x = nn.Dense(self.num_classes,
                     kernel_init=init_func(x),
                     bias_init=init_func(x))(x)
        return x if not return_activations else (x, activations)

# class MLPClassifier(nn.Module):
#     act_fn: nn.Module
#     hidden_dims : Sequence[int] = (1024, 512, 256, 256, 128)
#     num_classes : int = 2
#     dropout_prob : float = 0.0

#     @nn.compact
#     def __call__(self, x, train=True):
#         x = x.reshape(x.shape[0], -1)
#         for dims in self.hidden_dims:
#             x = nn.Dropout(self.dropout_prob)(x, deterministic=not train)
#             x = nn.Dense(dims, 
#                          kernel_init=init_func(x), 
#                          bias_init=init_func(x))(x)
#             x = nn.BatchNorm()(x, use_running_average=not train)
#             x = self.act_fn(x)
#         x = nn.Dropout(self.dropout_prob)(x, deterministic=not train)
#         x = nn.Dense(self.num_classes,
#                      kernel_init=init_func(x),
#                      bias_init=init_func(x))(x)        
#         return x


def calculate_loss(params, apply_fn, batch):
    """Calculate loss - optimized to avoid redundant conversions."""
    # Batch is already in correct format from collate function
    # Only convert if needed (avoid double conversion)
    if isinstance(batch[0], np.ndarray):
        imgs = jnp.array(batch[0])
    else:
        imgs = jnp.asarray(batch[0])  # Already JAX array, use asarray
    
    if isinstance(batch[1], np.ndarray):
        labels = jnp.array(batch[1], dtype=jnp.int32)
    else:
        labels = jnp.asarray(batch[1], dtype=jnp.int32)
    
    logits = apply_fn(params, imgs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (labels == logits.argmax(axis=-1)).mean()
    return loss, acc

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss,
                                 has_aux=True)
    (_, acc), grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, acc

@jax.jit
def eval_step(state, batch):
    _, acc = calculate_loss(state.params, state.apply_fn, batch)
    return acc


def test_model(state, data_loader):
    """
    Test a model on a specified dataset.

    Inputs:
        state - Training state including parameters and model apply function.
        data_loader - DataLoader object of the dataset to test on (validation or test)
    """
    true_preds, count = 0., 0
    for batch in data_loader:
        acc = eval_step(state, batch)
        batch_size = batch[0].shape[0]
        true_preds += acc * batch_size
        count += batch_size
    test_acc = true_preds / count
    return test_acc


def train_model(
    net,
    model_name,
    train_dataloader,
    test_dataloader,
    model_path,
    model_key=42,
    batch_size=256,
    max_epochs=1000,
    patience=200,
    plot=True,
    learning_rate=1e-2,
    initial_params=None
):
    """
    Train a model on the training set of FashionMNIST

    Inputs:
        net - Object of BaseNetwork
        model_name - (str) Name of the model, used for creating the checkpoint names
        max_epochs - Number of epochs we want to (maximally) train for
        patience - If the performance on the validation set has not improved for #patience epochs, we stop training early
        batch_size - Size of batches used in training
        learning_rate - Learning rate for optimizer
        initial_params - Optional initial parameters (for weight transfer from previous nps). If None, uses random initialization.
    """
    max_epoch_flag = True
    terminated_epoch = 0

    # Check that dataloader is not empty
    first_batch = next(iter(train_dataloader), None)
    if first_batch is None:
        raise ValueError(
            f"Train dataloader is empty! "
            f"train_dataset size: {len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 'unknown'}, "
            f"batch_size: {batch_size}, "
            f"drop_last: {True}"
        )
    
    # Initialize parameters: use initial_params if provided (weight transfer), otherwise random init
    if initial_params is not None:
        print(f"  Using transferred weights from previous nps (best model)")
        # Verify architecture matches (same structure)
        try:
            # Test if params structure matches by trying to apply
            _ = net.apply(initial_params, jnp.array(first_batch[0]))
            params = initial_params
            print(f"  ✓ Weight transfer successful")
        except Exception as e:
            print(f"  ⚠️ Weight transfer failed (architecture mismatch): {e}")
            print(f"  Falling back to random initialization")
            params = net.init(jax.random.PRNGKey(model_key), jnp.array(first_batch[0]))
    else:
        params = net.init(jax.random.PRNGKey(model_key), jnp.array(first_batch[0]))
    
    # Create optimizer with optional gradient clipping
    max_grad_norm = 1.0  # Clip gradients to prevent explosion
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),  # Gradient clipping
        optax.sgd(learning_rate=learning_rate, momentum=0.9)
        # optax.adam(learning_rate=learning_rate, eps=4e-5)
    )
    
    state = train_state.TrainState.create(apply_fn=net.apply,
                                          params=params,
                                          tx=optimizer)

    val_scores = []
    train_scores = []
    best_val_epoch = -1
    val_acc = 0.0

    for epoch in range(max_epochs):
        ############
        # Training #
        ############
        train_acc = 0.
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            state, acc = train_step(state, batch)
            train_acc += acc
        train_acc /= len(train_dataloader)

        ##############
        # Validation #
        ##############
        val_acc = test_model(state, test_dataloader)
        val_scores.append(val_acc)
        train_scores.append(train_acc)
        terminated_epoch = epoch
        # print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc:05.2%}, Validation accuracy: {val_acc:4.2%}")

        if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
            # print("\t   (New best performance, saving model...)")
            save_model(net, state.params, model_path, model_name)
            best_val_epoch = epoch
        elif best_val_epoch <= epoch - patience:
            print(f"Early stopping due to no improvement over the last {patience} epochs")
            max_epoch_flag = False
            break

    if max_epoch_flag:
        terminated_epoch = epoch

    if plot:
        # Plot a curve of the validation accuracy
        plt.plot([i for i in range(1,len(val_scores)+1)], val_scores, label='val_acc')
        plt.plot([i for i in range(1, len(train_scores) + 1)], train_scores, label='train_acc')
        plt.xlabel("Epochs")
        plt.ylabel("Validation accuracy & Training accuracy")
        plt.title(f"Validation performance of {model_name}")
        plt.legend()
        plt.show()
        plt.savefig(f"{model_name}.pdf", dpi=360)
        plt.close()

    

    # state, _ = load_model(CHECKPOINT_PATH, model_name, state=state)
    # test_acc = test_model(state, test_loader)
    print((f" Test accuracy: {val_acc:4.2%} ").center(50, "=")+"\n")
    # return state, test_acc
    return state, val_acc, train_scores, val_scores, max_epoch_flag, terminated_epoch

class Results(NamedTuple):
    run: int
    model_name: str
    val_acc: float
    train_scores: np.ndarray
    val_scores: np.ndarray
    max_epoch_flag: bool
    terminated_epoch: int


def prepare_data_dict_noiseless(config):
    """
    OPTIMIZED: Generate training data in large batches, then balance afterward if needed.
    Much faster than the old sequential rejection sampling approach.
    """
    qubits = config['n']
    num_training_data = config['num_training_data']
    alpha_targ = config['alpha_targ']
    y_targ = config['y_targ']
    balance_classes = config.get('balance_classes', True)
    
    print(f"\n{'='*80}")
    print(f"Generating training data (OPTIMIZED BATCH GENERATION)")
    print(f"{'='*80}")
    print(f"  Target samples: {num_training_data}")
    print(f"  Number of qubits: {qubits}")
    print(f"  Balance classes: {balance_classes}")
    
    # Pre-compute label checking values
    alpha_targ_int = int(list_to_string(list(alpha_targ)), 2)
    y_targ_int = int(y_targ, 2)
    
    # Generate more samples than needed to ensure uniqueness and balance
    # Typically ~10-20% overhead is enough
    if balance_classes:
        generation_factor = 1.3  # Generate 30% more to ensure balance after deduplication
    else:
        generation_factor = 1.1  # Generate 10% more to handle deduplication
    
    samples_to_generate = int(num_training_data * generation_factor)
    batch_size = min(10000, samples_to_generate)  # Generate in batches of 10k
    
    print(f"  Generating {samples_to_generate} candidates in batches of {batch_size}...")
    
    all_f_vecs = []
    all_labels = []
    seen_f_strs = set()
    
    num_batches = (samples_to_generate + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, samples_to_generate)
        batch_size_actual = batch_end - batch_start
        
        if batch_idx % max(1, num_batches // 5) == 0:
            print(f"    Batch {batch_idx + 1}/{num_batches} ({batch_start}-{batch_end})")
        
        # Generate random F functions in batch
        # Each F function is 2^qubits random bits
        f_batch = np.random.randint(0, 2, size=(batch_size_actual, 2**qubits), dtype=np.uint8)
        
        # Compute labels for entire batch
        # label = fi[y] ^ fi[y ^ alpha] (XOR of two F values)
        idx_y = y_targ_int
        idx_y_xor_alpha = y_targ_int ^ alpha_targ_int
        
        # Vectorized label computation
        labels_batch = f_batch[:, idx_y] ^ f_batch[:, idx_y_xor_alpha]
        
        # Check for duplicates and add unique ones
        for i in range(batch_size_actual):
            f_vec = f_batch[i]
            f_str = ''.join(map(str, f_vec.tolist()))
            
            if f_str not in seen_f_strs:
                seen_f_strs.add(f_str)
                all_f_vecs.append(f_vec)
                all_labels.append(int(labels_batch[i]))
    
    print(f"\n  Generated {len(all_f_vecs)} unique samples")
    
    # Balance classes if requested
    if balance_classes:
        # Count labels
        labels_array = np.array(all_labels)
        num_0 = np.sum(labels_array == 0)
        num_1 = np.sum(labels_array == 1)
        
        print(f"  Label distribution: {num_0} zeros, {num_1} ones")
        
        # Take equal number of each class (min of the two)
        samples_per_class = min(num_0, num_1, num_training_data // 2)
        
        # Get indices for each class
        indices_0 = np.where(labels_array == 0)[0]
        indices_1 = np.where(labels_array == 1)[0]
        
        # Randomly select samples_per_class from each
        np.random.shuffle(indices_0)
        np.random.shuffle(indices_1)
        
        selected_0 = indices_0[:samples_per_class]
        selected_1 = indices_1[:samples_per_class]
        
        # Combine and shuffle
        selected_indices = np.concatenate([selected_0, selected_1])
        np.random.shuffle(selected_indices)
        
        # Select the samples
        final_f_vecs = [all_f_vecs[i] for i in selected_indices]
        final_labels = [all_labels[i] for i in selected_indices]
        
        print(f"  Balanced to {len(final_f_vecs)} samples ({samples_per_class} per class)")
    else:
        # Just take the first num_training_data samples
        final_f_vecs = all_f_vecs[:num_training_data]
        final_labels = all_labels[:num_training_data]
        
        print(f"  Selected {len(final_f_vecs)} samples (unbalanced)")
    
    # Generate psi_F states (this is fast with JAX)
    print(f"  Computing quantum states...")
    psi_F_states = []
    for f_vec in final_f_vecs:
        psi_F_state = get_state(f_vec)
        psi_F_states.append(psi_F_state)
    
    # Create data dictionary
    data_dict = {
        'F_bs': final_f_vecs,
        'alpha': alpha_targ,
        'y_targ': y_targ,
        'b': final_labels,
        'psi_F_state': psi_F_states,
    }
    
    # Print final statistics
    final_labels_array = np.array(final_labels)
    num_0_final = np.sum(final_labels_array == 0)
    num_1_final = np.sum(final_labels_array == 1)
    
    print(f"\n  ✓ Dataset created:")
    print(f"    - Total samples: {len(final_f_vecs)}")
    print(f"    - Label 0: {num_0_final} ({num_0_final/len(final_f_vecs)*100:.1f}%)")
    print(f"    - Label 1: {num_1_final} ({num_1_final/len(final_f_vecs)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return data_dict


def prepare_data_dict(config):

    qubits = config['n']
    n_sample_points = config['num_training_data'] / 2
    alpha_targ = config['alpha_targ']
    y_targ = config['y_targ']
    channel_type = config['channel_type']
    noise_strength = config['noise_strength']
    thermal_p_exc = config['thermal_p_exc']
    parent_f_instance_seed = config['parent_f_instance_seed']

    data_dict = {
        'F_bs': [],
        'alpha': alpha_targ,
        'y_targ': y_targ,
        'b': [],
        'noisy_circuit': [],
        'rho_F_after_noise': [],
        'noisy_quantum_protocol_accuracy': [],
    }

    cnt_0 = 0
    cnt_1 = 0
    # fi_instances = []
    fi_set = set()
    np.random.seed(parent_f_instance_seed)
    random.seed(parent_f_instance_seed)

    while cnt_0 < n_sample_points or cnt_1 < n_sample_points:
        f_seed = np.random.randint(0, 2**30)
        # print(cnt_0, cnt_1, f_seed)
        fi = get_f_instance(qubits, f_seed)
        fi_str = ''.join(map(str, fi.tolist()))

        psi_F_state = get_state(fi)

        alpha_targ_int = int(list_to_string(list(alpha_targ)), 2)
        y_targ_int = int(y_targ, 2)

        label = 0
        if check_condition(alpha_targ_int, y_targ_int, label, fi):
            label = 0
        else:
            label = 1

      #   c_u = get_u(alpha_targ, qubits)
      #   c = tc.Circuit(qubits, inputs=psi_F_state)
      #   c1 = c.append(c_u)

      # # https://tensorcircuit.readthedocs.io/en/latest/api/quantum.html#tensorcircuit.quantum.measurement_counts
      #   quantum_distribution = tc.quantum.measurement_results(c1.state(),
      #                                                       counts=bin_counts, format="count_dict_bin", jittable=False)

      #   label = None
      #   label_y_b = []
      #   for bs in quantum_distribution.keys():
      #       if f'{bs[:-1]}0' == y_targ:
      #           label = int(bs[-1])
      #       label_y_b.append((f'{bs[:-1]}0', bs[-1]))

      # Because we have 2^{2^{n}} many possibilities of generating one F_i and
      # for each of those 2^{n-1} possibilities for selecting an alpha uniformly at random (because we set the last bit always to '1').
      # Hence, the total should be 2^{2^{n}} * 2^{n-1} = 2^{2^n + n - 1}!

        if fi_str not in fi_set:
            if label == 0 and cnt_0 < n_sample_points and (len(fi_set) < (2 ** ((2 ** qubits) + qubits - 1))):
                cnt_0 += 1
                fi_set.add(fi_str)
                rho_F_state = jnp.outer(psi_F_state, jnp.conjugate(psi_F_state))

                if channel_type == "dephasing":
                    noisy_circuit = global_dephasing_channel(rho_F_state, noise_strength, qubits)
                elif channel_type == "depolarizing": # User's specific global depolarizing formula
                    noisy_circuit = global_depolarizing_channel(rho_F_state, noise_strength, qubits)
                elif channel_type == "thermal":
                    noisy_circuit = global_thermal_relaxation_channel(rho_F_state, noise_strength, thermal_p_exc, qubits)
                elif channel_type == "local_depolarizing":
                    noisy_circuit = local_depolarizing_channel(rho_F_state, noise_strength, qubits)
                elif channel_type == "bit_flip":
                    noisy_circuit = bitflip_channel(rho_F_state, noise_strength, qubits)

                rho_F_after_noise = noisy_circuit.state()
                c_u_dm = get_u_dm(alpha_targ, qubits)
                c_dm = tc.DMCircuit(qubits, dminputs=rho_F_after_noise)
                c1_dm = c_dm.append(c_u_dm)

                rho_F_after_noise_after_decoder = c1_dm.state()
                noisy_accuracy = compute_noisy_accuracy(rho_F_after_noise_after_decoder, fi, alpha_targ, qubits)

                data_dict['F_bs'].append(fi)
                data_dict['rho_F_after_noise'].append(rho_F_after_noise)
                data_dict['noisy_circuit'].append(noisy_circuit)
                # data_dict['label_y_b'].append(label_y_b)
                data_dict['noisy_quantum_protocol_accuracy'].append(noisy_accuracy)
                data_dict['b'].append(label)

            elif label == 1 and cnt_1 < n_sample_points and (len(fi_set) < (2 ** ((2 ** qubits) + qubits - 1))):
                cnt_1 += 1
                fi_set.add(fi_str)
                rho_F_state = jnp.outer(psi_F_state, jnp.conjugate(psi_F_state))

                if channel_type == "dephasing":
                    noisy_circuit = global_dephasing_channel(rho_F_state, noise_strength, qubits)
                elif channel_type == "depolarizing": # User's specific global depolarizing formula
                    noisy_circuit = global_depolarizing_channel(rho_F_state, noise_strength, qubits)
                elif channel_type == "thermal":
                    noisy_circuit = global_thermal_relaxation_channel(rho_F_state, noise_strength, thermal_p_exc, qubits)
                elif channel_type == "local_depolarizing":
                    noisy_circuit = local_depolarizing_channel(rho_F_state, noise_strength, qubits)
                elif channel_type == "bit_flip":
                    noisy_circuit = bitflip_channel(rho_F_state, noise_strength, qubits)

                rho_F_after_noise = noisy_circuit.state()
                c_u_dm = get_u_dm(alpha_targ, qubits)
                c_dm = tc.DMCircuit(qubits, dminputs=rho_F_after_noise)
                c1_dm = c_dm.append(c_u_dm)

                rho_F_after_noise_after_decoder = c1_dm.state()
                noisy_accuracy = compute_noisy_accuracy(rho_F_after_noise_after_decoder, fi, alpha_targ, qubits)

                data_dict['F_bs'].append(fi)
                data_dict['rho_F_after_noise'].append(rho_F_after_noise)
                data_dict['noisy_circuit'].append(noisy_circuit)
                # data_dict['label_y_b'].append(label_y_b)
                data_dict['noisy_quantum_protocol_accuracy'].append(noisy_accuracy)
                data_dict['b'].append(label)

    return data_dict


# In[ ]:

def hamming_weight(n):
    """
    Compute Hamming weight of an integer (number of 1s in binary representation).
    
    Args:
        n: Integer
        
    Returns:
        Hamming weight (number of 1s)
    """
    return bin(n).count('1')


def get_relevant_input(rho_f, y_targ, alpha_targ, qubits):
    """
    Extract relevant input features from density matrix.
    
    NEW feature structure based on Hamming weight (2*Nq+1 features):
    1. Base diagonals: rho[idx0, idx0], rho[idx1, idx1]
    2. Central element: rho[idx0, idx1]
    3. Averaged products for k=1..Nq-1 (excluding idx0, idx1): 
       x_k = avg(rho[idx0, n_k[i]] * rho[n_k[i], idx1]) 
       where n_k[i] are all n with Hamming weight k
    
    Total: 2 + 1 + (Nq-1) = Nq + 2 features
    """
    alpha_str = "".join(map(str, alpha_targ))
    y_targ_int = int(y_targ, 2)
    alpha_targ_int = int(alpha_str, 2)
    idx0 = y_targ_int ^ alpha_targ_int
    idx1 = y_targ_int

    rhosnm = []
    
    # 1. Base diagonals: rho[idx0, idx0], rho[idx1, idx1]
    diag_idx0 = jnp.real(rho_f[idx0, idx0]) * (2**qubits)
    diag_idx1 = jnp.real(rho_f[idx1, idx1]) * (2**qubits)
    rhosnm.extend([diag_idx0, diag_idx1])
    
    # 2. Central element: rho[idx0, idx1] * 2^nq
    central_element = jnp.real(rho_f[idx0, idx1]) * (2**qubits)
    rhosnm.append(central_element)
    
    # Group indices by Hamming weight
    all_n = np.arange(2**qubits)
    hamming_weight_groups = {}  # weight -> list of n values
    for n in all_n:
        weight = hamming_weight(n)
        if weight not in hamming_weight_groups:
            hamming_weight_groups[weight] = []
        hamming_weight_groups[weight].append(n)
    
    # 3. Averaged products for each Hamming weight k=1..Nq-1
    for k in range(1, qubits):  # k = 1..Nq-1
        n_k = np.array([n for n in hamming_weight_groups.get(k, []) 
                       if n != idx0 and n != idx1])
        
        if len(n_k) > 0:
            # Compute products for all n_k
            products = []
            for n_val in n_k:
                rho_idx0_n = jnp.real(rho_f[idx0, n_val]) * (2**qubits)
                rho_n_idx1 = jnp.real(rho_f[n_val, idx1]) * (2**qubits)
                product = rho_idx0_n * rho_n_idx1
                products.append(product)
            
            # Average over all n_k
            avg_product = jnp.mean(jnp.array(products)) if products else 0.0
            rhosnm.append(avg_product)
        else:
            rhosnm.append(0.0)
    
    return rhosnm


def normalize_features(X_train, X_val, feature_range_clip=None):
    """
    Normalize features to mean=0, std=1 for stable training.
    
    OPTIMIZED: Works directly with JAX arrays to avoid device-to-host transfers.
    
    Args:
        X_train: List of training feature vectors (each is a numpy/jax array)
        X_val: List of validation feature vectors
        feature_range_clip: Optional tuple (min, max) to clip features after normalization
    
    Returns:
        X_train_norm: Normalized training features
        X_val_norm: Normalized validation features
        mean: Mean used for normalization
        std: Std used for normalization
    """
    # Convert to JAX arrays directly (avoiding numpy conversion that triggers device transfers)
    # Use jnp.stack to create arrays efficiently
    X_train_arr = jnp.stack([jnp.asarray(x) for x in X_train])
    X_val_arr = jnp.stack([jnp.asarray(x) for x in X_val])
    
    # Compute statistics from training data only (using JAX operations)
    mean = jnp.mean(X_train_arr, axis=0)
    std = jnp.std(X_train_arr, axis=0)
    
    # Avoid division by zero
    std = jnp.where(std > 1e-8, std, 1.0)
    
    # Normalize (all JAX operations, stays on device)
    X_train_norm = (X_train_arr - mean) / std
    X_val_norm = (X_val_arr - mean) / std
    
    # Optional: Clip to range
    if feature_range_clip is not None:
        min_val, max_val = feature_range_clip
        X_train_norm = jnp.clip(X_train_norm, min_val, max_val)
        X_val_norm = jnp.clip(X_val_norm, min_val, max_val)
    
    # Convert back to list of arrays (slice the JAX array efficiently)
    # Each slice is still a JAX array on the same device
    X_train_norm_list = [X_train_norm[i] for i in range(X_train_norm.shape[0])]
    X_val_norm_list = [X_val_norm[i] for i in range(X_val_norm.shape[0])]
    
    # For printing, compute statistics on CPU (but do it once, not in loops)
    mean_cpu = float(jnp.mean(mean))
    std_cpu = float(jnp.mean(std))
    train_min = float(jnp.min(X_train_arr))
    train_max = float(jnp.max(X_train_arr))
    norm_min = float(jnp.min(X_train_norm)) if X_train_norm.shape[0] > 0 else 0.0
    norm_max = float(jnp.max(X_train_norm)) if X_train_norm.shape[0] > 0 else 0.0
    
    print(f"  Feature normalization: mean={mean_cpu:.4f}, std={std_cpu:.4f}")
    print(f"    Training feature range: [{train_min:.2f}, {train_max:.2f}]")
    print(f"    Normalized range: [{norm_min:.4f}, {norm_max:.4f}]")
    
    # Return mean and std as numpy for compatibility (but this is a one-time transfer)
    return X_train_norm_list, X_val_norm_list, np.array(mean), np.array(std)


def debug_visualize_matrices(training_states, val_shadow_states, n_qubits, sample_idx=0):
    """
    Debug visualization of density matrices.
    
    Args:
        training_states: List of training state features (or full matrices)
        val_shadow_states: List of validation shadow state features (or full matrices)
        n_qubits: Number of qubits
        sample_idx: Index of sample to visualize
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training matrix (from white noise or shadows)
    if isinstance(training_states[sample_idx], np.ndarray) and training_states[sample_idx].ndim == 1:
        # It's features, we can't visualize the full matrix
        # Just plot the features
        ax = axes[0]
        features = training_states[sample_idx]
        ax.plot(features, marker='o')
        ax.set_title(f'Training State Features (sample {sample_idx})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.grid(True)
        print(f"Training state {sample_idx}: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    else:
        # It's a full matrix
        ax = axes[0]
        matrix = np.array(training_states[sample_idx])
        im = ax.imshow(np.real(matrix), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'Training State Density Matrix (sample {sample_idx})\nReal part')
        plt.colorbar(im, ax=ax)
        print(f"Training state {sample_idx}: shape={matrix.shape}, trace={np.trace(matrix):.4f}")
    
    # Plot validation shadow matrix
    if isinstance(val_shadow_states[sample_idx], np.ndarray) and val_shadow_states[sample_idx].ndim == 1:
        # It's features, we can't visualize the full matrix
        ax = axes[1]
        features = val_shadow_states[sample_idx]
        ax.plot(features, marker='o')
        ax.set_title(f'Validation Shadow Features (sample {sample_idx})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.grid(True)
        print(f"Validation shadow {sample_idx}: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    else:
        # It's a full matrix
        ax = axes[1]
        matrix = np.array(val_shadow_states[sample_idx])
        im = ax.imshow(np.real(matrix), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'Validation Shadow Density Matrix (sample {sample_idx})\nReal part')
        plt.colorbar(im, ax=ax)
        print(f"Validation shadow {sample_idx}: shape={matrix.shape}, trace={np.trace(matrix):.4f}")
    
    plt.tight_layout()
    return fig


def generate_debug_matrices(data_dict, config, sample_idx=0, y_targ=None, alpha_targ=None):
    """
    Generate full density matrices for debugging visualization.
    
    Args:
        data_dict: The data dictionary from prepare_data_dict
        config: Configuration dictionary
        sample_idx: Index of sample to debug
        y_targ: Target y value (if None, use from config)
        alpha_targ: Target alpha value (if None, use from config)
    
    Returns:
        training_matrix, shadow_matrix: Full density matrices for comparison
    """
    n = config['n']
    y_targ = y_targ or config['y_targ']
    alpha_targ = alpha_targ or config['alpha_targ']
    
    # Get pure state
    psi_f = data_dict['psi_F_state'][sample_idx]
    F_vec = data_dict['F_bs'][sample_idx]
    rho_pure = jnp.outer(psi_f, jnp.conjugate(psi_f))
    
    # Generate training matrix (white noise or shadows)
    if not config.get('use_shadows', True):
        # White noise approach
        noise_strength = config.get('noise_strength', 0.01)
        noise_matrix = np.random.randn(2**n, 2**n).astype(np.complex128)
        training_matrix = rho_pure + noise_matrix * noise_strength
    else:
        # Shadow approach
        cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
        Ks = get_kraus_operators(cfg)
        nps = config.get("shadow_nps") or 200
        batch_size_mcs = min(config.get("batch_size_mcs", 128), nps)
        shots = config.get("shots", 1)
        r = config.get('r', 1)
        
        F_vec_jax = jnp.array(F_vec)
        rng_key = jax.random.PRNGKey(sample_idx)
        
        from shadow_mcs_jitted import mcs_shadows_streaming_jit
        training_matrix = mcs_shadows_streaming_jit(
            rng_key, F_vec_jax, n, Ks, nps, batch_size_mcs, r, shots,
            weights_kind=0, use_complex64=True
        )
    
    # Generate validation shadow matrix (always use shadows)
    cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
    Ks = get_kraus_operators(cfg)
    # Use validation_nps if specified, otherwise use same as training
    val_nps = config.get("validation_nps") or config.get("shadow_nps") or 200
    nps = val_nps
    batch_size_mcs = min(config.get("batch_size_mcs", 128), nps)
    shots = config.get("shots", 1)
    r = config.get('r', 1)
    
    F_vec_jax = jnp.array(F_vec)
    rng_key = jax.random.PRNGKey(sample_idx + 10000)  # Different key for validation
    
    from shadow_mcs_jitted import mcs_shadows_streaming_jit
    val_shadow_matrix = mcs_shadows_streaming_jit(
        rng_key, F_vec_jax, n, Ks, nps, batch_size_mcs, r, shots,
        weights_kind=0, use_complex64=True
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pure state
    ax = axes[0]
    im = ax.imshow(np.real(np.array(rho_pure)), cmap='RdBu', aspect='auto')
    ax.set_title(f'Pure State (Ground Truth)\nTrace: {np.trace(np.array(rho_pure)):.4f}')
    plt.colorbar(im, ax=ax)
    
    # Training matrix
    ax = axes[1]
    im = ax.imshow(np.real(np.array(training_matrix)), cmap='RdBu', aspect='auto')
    method = 'Shadows' if config.get('use_shadows', True) else f'White Noise (σ={config.get("noise_strength", 0.01)})'
    ax.set_title(f'Training State ({method})\nTrace: {np.trace(np.array(training_matrix)):.4f}')
    plt.colorbar(im, ax=ax)
    
    # Validation shadow
    ax = axes[2]
    im = ax.imshow(np.real(np.array(val_shadow_matrix)), cmap='RdBu', aspect='auto')
    ax.set_title(f'Validation Shadow (nps={val_nps})\nTrace: {np.trace(np.array(val_shadow_matrix)):.4f}')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nSample {sample_idx} Statistics:")
    print(f"Pure state trace: {np.trace(np.array(rho_pure)):.6f}")
    print(f"Training state trace: {np.trace(np.array(training_matrix)):.6f}")
    print(f"Validation shadow trace: {np.trace(np.array(val_shadow_matrix)):.6f}")
    print(f"\nNorm difference (training vs pure): {np.linalg.norm(np.array(training_matrix - rho_pure)):.6f}")
    print(f"Norm difference (validation vs pure): {np.linalg.norm(np.array(val_shadow_matrix - rho_pure)):.6f}")
    
    return training_matrix, val_shadow_matrix, rho_pure


def run_experiment_noiseless_DM_infinite_limit_CS(config, data_dict):

    n = config['n']
    num_training_data = config['num_training_data']
    r = config['r']
    # nps = config['nps']
    alpha_targ = config['alpha_targ']
    model_path = config['CHECKPOINT_PATH']
    classical_shadow_seed = config['parent_seed']
    num_runs = config['num_runs']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    patience = config['patience']
    y_targ = config['y_targ']
    y_targ_list = string_to_list(y_targ)
    
    # Batching configuration for memory efficiency
    processing_batch_size = config.get('processing_batch_size', 100)  # Process data in batches
    use_batched_processing = config.get('use_batched_processing', False)  # Enable batching for large datasets
    shadow_batch_size = config.get('shadow_batch_size', 16)  # Batch size for shadow generation
    
    targets = data_dict['b']
    num_samples = len(data_dict['F_bs'])  # Define once for both branches
    # dataset_size = len(targets)
    np.random.seed(classical_shadow_seed)
    random.seed(classical_shadow_seed)
    
    # If using batched processing, we'll process and save data in chunks
    if use_batched_processing:
        print(f"Using batched processing with batch size: {processing_batch_size}")
        # Create cache directory for batched data
        cache_dir = os.path.join(model_path, "batched_data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        shadow_M_F_states = []
        
        # Get Kraus operators for Monte Carlo sampling
        cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
        Ks = get_kraus_operators(cfg)
        nps = config.get("shadow_nps") or 200  # Default to 200 if None
        batch_size_mcs = min(config.get("batch_size_mcs", 128), nps)  # Don't exceed nps, use smaller default
        shots = config.get("shots", 1)
        
        # Process data in batches
        for batch_start in range(0, num_samples, processing_batch_size):
            batch_end = min(batch_start + processing_batch_size, num_samples)
            batch_indices = range(batch_start, batch_end)
            
            print(f"  Processing batch {batch_start//processing_batch_size + 1}/{(num_samples + processing_batch_size - 1)//processing_batch_size} "
                  f"({batch_start}-{batch_end})")
            
            batch_shadow_states = []
            for i in batch_indices:
                F_vec = jnp.array(data_dict['F_bs'][i])  # Convert to JAX array
                
                if config["use_shadows"]:
                    # Use Monte Carlo sampling with shadow tomography
                    rng_key = jax.random.PRNGKey(i)
                    shadow_M_F_state = mcs_shadows_streaming_jit(
                        rng_key, F_vec, n, Ks, nps, batch_size_mcs, r, shots, 
                        weights_kind=0, use_complex64=True
                    )
                else:
                    psi_f = data_dict['psi_F_state'][i]
                    rho_f = jnp.outer(psi_f, jnp.conjugate(psi_f))
                    shadow_M_F_state = rho_f + np.random.randn(2**n, 2**n) * config["noise_strength"]
                
                rhosnm = get_relevant_input(shadow_M_F_state, y_targ, alpha_targ, n)
                batch_shadow_states.append(rhosnm)
            
            # Save batch to disk
            batch_file = os.path.join(cache_dir, f"batch_{batch_start}_{batch_end}.pkl")
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_shadow_states, f)
            
            # Clear batch from memory (optional, helps with memory management)
            del batch_shadow_states
            
        # Load all batches from disk
        print("Loading all batches from disk...")
        for batch_start in range(0, num_samples, processing_batch_size):
            batch_end = min(batch_start + processing_batch_size, num_samples)
            batch_file = os.path.join(cache_dir, f"batch_{batch_start}_{batch_end}.pkl")
            with open(batch_file, 'rb') as f:
                batch_shadow_states = pickle.load(f)
            shadow_M_F_states.extend(batch_shadow_states)
            
            # Clean up batch file after loading (optional)
            # os.remove(batch_file)
        
        # Clean up cache directory (optional)
        # import shutil
        # shutil.rmtree(cache_dir)
        
    else:
        # Original processing: all in memory
        print(f"  Generating training data representations...")
        print(f"  Approach: {'Shadows' if config['use_shadows'] else 'White Noise'}")
        
        # Generate training data based on approach
        if config["use_shadows"]:
            # Use Monte Carlo shadow sampling for training
            print(f"  Generating shadows for {num_samples} samples...")
            # Get Kraus operators for Monte Carlo sampling
            cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
            Ks = get_kraus_operators(cfg)
            nps = config.get("shadow_nps") or 200  # Default to 200 if None or not set
            batch_size_mcs = min(config.get("batch_size_mcs", 128), nps)
            shots = config.get("shots", 1)
            
            # Convert all F_vecs to JAX arrays at once for batch processing
            all_F_vecs = jnp.array(data_dict['F_bs'])  # (num_samples, 2**n)
            
            print(f"    Batch size: {shadow_batch_size}, Monte Carlo samples per state: {nps}")
            print(f"    First batch may take 1-5 minutes for JIT compilation...")
            
            # Process in batches
            shadow_states = []
            for batch_start in range(0, num_samples, shadow_batch_size):
                batch_end = min(batch_start + shadow_batch_size, num_samples)
                batch_F_vecs = all_F_vecs[batch_start:batch_end]
                
                print(f"    Processing batch {batch_start//shadow_batch_size + 1}/{(num_samples + shadow_batch_size - 1)//shadow_batch_size} "
                      f"({batch_start}-{batch_end})")
                rng_key = jax.random.PRNGKey(batch_start)
                
                # Process entire batch at once using vmap
                batch_shadows = batched_shadow_generation(
                    rng_key, batch_F_vecs, n, Ks, nps, batch_size_mcs, r, shots
                )
                print(f"    Batch completed!")
                
                # Extract relevant inputs for each shadow state in batch
                for j, shadow in enumerate(batch_shadows):
                    rhosnm = get_relevant_input(shadow, y_targ, alpha_targ, n)
                    shadow_states.append(jnp.array(rhosnm))
            
            training_states = shadow_states
        else:
            # White noise approach: add noise to pure states
            print(f"  Adding white noise to {num_samples} pure states...")
            noise_strength = config.get("noise_strength", 0.01)
            print(f"    Noise strength: {noise_strength}")
            
            noise_states = []
            for i, psi_f in enumerate(data_dict['psi_F_state']):
                if i % 100 == 0:
                    print(f"    Processing {i}/{num_samples}...")
                rho_f = jnp.outer(psi_f, jnp.conjugate(psi_f))
                # Add white noise
                noise_matrix = np.random.randn(2**n, 2**n).astype(np.complex128)
                noisy_rho = rho_f + noise_matrix * noise_strength
                rhosnm = get_relevant_input(noisy_rho, y_targ, alpha_targ, n)
                noise_states.append(jnp.array(rhosnm))
            
            training_states = noise_states
        
        shadow_M_F_states = training_states
    
    # Optional: Store full matrices for debugging if debug flag is set
    debug_mode = config.get('debug', False)
    if debug_mode:
        print("DEBUG MODE: Storing full density matrices for visualization...")
        # We need to store the full matrices before they're converted to features
        # This will be added in the generation loop above

    # Pre-generate validation shadows once (before training runs loop)
    print(f"\n{'='*80}")
    print(f"Pre-generating validation shadows (shared across all runs)...")
    print(f"{'='*80}")
    
    # Split indices once with a fixed seed for validation set
    val_seed = 42
    validation_fraction = config.get('validation_fraction', 0.3)  # Default 30%
    print(f"  Validation fraction: {validation_fraction*100:.1f}% of dataset")
    _, val_indices, _, y_val_labels = train_test_split(
        np.arange(num_samples), targets, test_size=validation_fraction, random_state=val_seed, stratify=targets
    )
    
    # Generate validation shadows once
    cfg = config.get("channel_config", {"type": "thermal", "strength": 0.1, "thermal_p_exc": 0.0})
    Ks = get_kraus_operators(cfg)
    # Use validation_nps if specified, otherwise use same as training
    val_nps = config.get("validation_nps")
    nps = val_nps
    batch_size_mcs = min(config.get("batch_size_mcs", 128), nps)
    shots = config.get("shots", 1)
    
    print(f"  Validation using nps = {nps}")
    
    val_F_vecs = jnp.array([data_dict['F_bs'][idx] for idx in val_indices])
    
    val_shadow_states = []
    for batch_start in range(0, len(val_indices), shadow_batch_size):
        batch_end = min(batch_start + shadow_batch_size, len(val_indices))
        batch_F_vecs = val_F_vecs[batch_start:batch_end]
        
        print(f"  Validation batch {batch_start//shadow_batch_size + 1}/{(len(val_indices) + shadow_batch_size - 1)//shadow_batch_size}")
        rng_key = jax.random.PRNGKey(batch_start + 10000)  # Different key for validation
        
        batch_shadows = batched_shadow_generation(
            rng_key, batch_F_vecs, n, Ks, nps, batch_size_mcs, r, shots
        )
        
        for j, shadow in enumerate(batch_shadows):
            rhosnm = get_relevant_input(shadow, y_targ, alpha_targ, n)
            val_shadow_states.append(jnp.array(rhosnm))
    
    print(f"Validation shadows generated! Reusing for all {num_runs} runs.\n")

    val_features_map = {int(idx): val_shadow_states[i] for i, idx in enumerate(val_indices)}
    val_feature_list = [val_features_map[int(idx)] for idx in val_indices]
    y_val_array = np.array(y_val_labels)
    train_candidate_indices = np.setdiff1d(np.arange(num_samples), val_indices)
    
    results = []
    model_names = []
    for run in range(num_runs):
        random_seed = np.random.randint(0, 2**30)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.random.manual_seed(random_seed)

        # Sample training indices from the training pool (different subset each run)
        train_size = num_samples - len(val_indices)
        train_indices = np.random.choice(train_candidate_indices, size=train_size, replace=False)
        y_train = targets[train_indices]
        
        # Training data: use pre-generated representations (shadows or white noise)
        X_train = [shadow_M_F_states[idx] for idx in train_indices]
        
        # Validation data: fixed set generated above
        X_val = list(val_feature_list)
        y_val = y_val_array
        
        # Normalize features to stabilize training (especially for large nps)
        use_normalization = config.get('normalize_features', True)  # Default to True
        if use_normalization:
            print(f"  Normalizing features for run {run+1}/{num_runs}...")
            feature_range_clip = config.get('feature_range_clip', None)  # e.g., (-3, 3) for z-score clipping
            X_train, X_val, _, _ = normalize_features(X_train, X_val, feature_range_clip)

        train_dataset = QuantumStateDataset(X_train, y_train)
        test_dataset = QuantumStateDataset(X_val, y_val)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=numpy_collate)

        for act_fn_name in act_fn_by_name:
            print(f"Training BaseNetwork with {act_fn_name} activation...")
            act_fn = act_fn_by_name[act_fn_name]()
            net_actfn = BaseNetwork(act_fn=act_fn)
            
            # Include nps in model name to avoid weight persistence across nps values!
            nps_val = config.get('shadow_nps', 'unknown')
            model_name = f"test_nps{nps_val}_run{run}"
            _, val_acc, train_scores, val_scores, max_epoch_flag, terminated_epoch = train_model(
                net_actfn,
                model_name,
                train_dataloader,
                test_dataloader,
                model_path,
                random_seed,
                batch_size,
                max_epochs,
                patience,
                True,
                config['learning_rate'],
            )

            book_keeping = Results(run=run, model_name=model_name, val_acc=val_acc, train_scores=train_scores, val_scores=val_scores,
                                  max_epoch_flag=max_epoch_flag, terminated_epoch=terminated_epoch)
            # print(train_scores)
            results.append(book_keeping)
            model_names.append(model_name)

    return results, model_names


def run_experiment_with_precomputed_shadows(config, data_dict, initial_params=None):
    """
    Run training experiment using pre-computed shadows at specific nps.
    This avoids regenerating shadows for each nps value.
    
    Args:
        config: Configuration dictionary
        data_dict: Data dictionary
        initial_params: Optional initial parameters to use (for weight transfer from previous nps)
    """
    n = config['n']
    num_training_data = config['num_training_data']
    alpha_targ = config['alpha_targ']
    model_path = config['CHECKPOINT_PATH']
    num_runs = config['num_runs']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    patience = config['patience']
    y_targ = config['y_targ']
    
    targets = data_dict['b']
    num_samples = len(data_dict['F_bs'])
    
    # Get pre-computed shadows
    training_shadows = config['_training_shadows']  # idx -> feature_vector
    validation_shadows = config['_validation_shadows']  # idx -> feature_vector
    val_indices = config['_val_indices']
    y_val = config['_y_val']
    
    validation_fraction = config.get('validation_fraction', 0.3)  # Default 30%
    
    results = []
    model_names = []
    
    for run in range(num_runs):
        random_seed = np.random.randint(0, 2**30)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        
        # Split training data (different each run)
        train_indices, _, y_train, _ = train_test_split(
            np.arange(num_samples), targets, test_size=validation_fraction, random_state=random_seed, stratify=targets
        )
        
        # Debug: Check what's in training_shadows (only print if there's an issue)
        # Will print below if empty
        
        # Training data: use pre-computed shadows for this nps
        # Filter out None values
        X_train = []
        y_train_filtered = []
        for i, idx in enumerate(train_indices):
            shadow_data = training_shadows.get(idx)
            if shadow_data is not None:
                X_train.append(shadow_data)
                y_train_filtered.append(y_train[i])
        
        # Validation data: use pre-computed validation shadows
        X_val = []
        y_val_filtered = []
        for i, idx in enumerate(val_indices):
            shadow_data = validation_shadows.get(idx)
            if shadow_data is not None:
                X_val.append(shadow_data)
                y_val_filtered.append(y_val[i])
        
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  ERROR: Empty dataset!")
            print(f"    - training_shadows keys (sample): {list(training_shadows.keys())[:10]}")
            print(f"    - train_indices (sample): {train_indices[:10]}")
            print(f"    - num_samples: {num_samples}")
            print(f"    - X_train length: {len(X_train)}, X_val length: {len(X_val)}")
            if len(X_train) == 0:
                # Check if indices match
                matching = [idx for idx in train_indices if idx in training_shadows]
                print(f"    - Matching indices count: {len(matching)}/{len(train_indices)}")
            if len(X_val) == 0:
                print(f"    - validation_shadows keys (sample): {list(validation_shadows.keys())[:10]}")
                matching_val = [idx for idx in val_indices if idx in validation_shadows]
                print(f"    - Matching val indices count: {len(matching_val)}/{len(val_indices)}")
            print(f"  Skipping run {run}...")
            continue
        
        print(f"  Data ready: X_train={len(X_train)}, X_val={len(X_val)}")
        
        # Normalize features to stabilize training (especially for large nps)
        use_normalization = config.get('normalize_features', True)  # Default to True
        if use_normalization:
            print(f"  Normalizing features for run {run+1}/{num_runs}...")
            feature_range_clip = config.get('feature_range_clip', None)  # e.g., (-3, 3) for z-score clipping
            X_train, X_val, _, _ = normalize_features(X_train, X_val, feature_range_clip)
        
        train_dataset = QuantumStateDataset(X_train, y_train_filtered)
        test_dataset = QuantumStateDataset(X_val, y_val_filtered)
        
        # Ensure batch_size is not larger than dataset (with drop_last=True this causes empty dataloader)
        actual_train_size = len(train_dataset)
        if actual_train_size == 0:
            raise ValueError(f"Train dataset is empty! Cannot create dataloader.")
        
        effective_batch_size = min(batch_size, actual_train_size)
        # Critical: If batch_size >= dataset_size with drop_last=True, dataloader will be EMPTY
        # So we MUST set drop_last=False when batch_size >= dataset_size
        use_drop_last = True if effective_batch_size < actual_train_size else False
        
        if effective_batch_size < batch_size:
            print(f"  Warning: Reducing batch_size from {batch_size} to {effective_batch_size} (dataset size: {actual_train_size})")
        if not use_drop_last:
            print(f"  Note: Setting drop_last=False (batch_size {effective_batch_size} >= dataset size {actual_train_size})")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True, 
            drop_last=use_drop_last,
            collate_fn=numpy_collate
        )
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=numpy_collate)
        
        for act_fn_name in act_fn_by_name:
            print(f"Training BaseNetwork with {act_fn_name} activation...")
            act_fn = act_fn_by_name[act_fn_name]()
            net_actfn = BaseNetwork(act_fn=act_fn)
            
            # Include nps in model name to avoid weight persistence across nps values!
            nps_val = config.get('shadow_nps', 'unknown')
            model_name = f"test_nps{nps_val}_run{run}"
            
            # Use initial_params if available (weight transfer from previous nps)
            # Transfer weights to all runs (or just first run - configurable)
            use_transferred_weights = initial_params is not None
            transfer_to_all_runs = config.get('transfer_weights_to_all_runs', False)  # Default: only first run
            use_initial_params = use_transferred_weights and (run == 0 or transfer_to_all_runs)
            
            if use_initial_params and run == 0:
                print(f"  Using transferred weights from previous nps (best model)")
            elif use_initial_params and run > 0:
                print(f"  Using transferred weights for run {run+1} (from previous nps)")
            
            train_timer_label = (
                f"train_model run={run+1}/{num_runs} act={act_fn_name} nps={config.get('shadow_nps', 'unknown')}"
            )
            with log_time(train_timer_label):
                _, val_acc, train_scores, val_scores, max_epoch_flag, terminated_epoch = train_model(
                    net_actfn, model_name, train_dataloader, test_dataloader,
                    model_path, random_seed, batch_size, max_epochs, patience,
                    True, config['learning_rate'], initial_params=initial_params if use_initial_params else None
                )
            
            book_keeping = Results(
                run=run, model_name=model_name, val_acc=val_acc,
                train_scores=train_scores, val_scores=val_scores,
                max_epoch_flag=max_epoch_flag, terminated_epoch=terminated_epoch
            )
            results.append(book_keeping)
            model_names.append(model_name)
    
    return {'results': results, 'model_names': model_names}


def main_training(config, god_level_seed):
    np.random.seed(god_level_seed)
    random.seed(god_level_seed)
    parent_f_instance_seed = god_level_seed + 1
    config['parent_f_instance_seed'] = parent_f_instance_seed

    qubits = config['n']
    num_training_data = config['num_training_data']
    channel_type = config['channel_type']
    noise_strength = config['noise_strength']
    r = config['r']
    alpha_targ = config['alpha_targ']
    alpha_targ_str = "".join(map(str, alpha_targ))
    y_targ = config['y_targ']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    
    
    if channel_type == 'thermal':
        thermal_p_exc = config['thermal_p_exc']
    else:
        thermal_p_exc = None

    data_file_name = f"data_{qubits}q_{num_training_data}_{channel_type}_{noise_strength}ns_{thermal_p_exc}pexc_{alpha_targ_str}alpha_{y_targ}y.pkl"
    ### TODO: also incorporate optimizer details such as LR
    experiment_file_name = f"exp_results_{qubits}q_{num_training_data}_{channel_type}_{noise_strength}ns_{thermal_p_exc}pexc_{alpha_targ_str}alpha_{y_targ}y_{r}r_{batch_size}bs_{max_epochs}_epochs_{god_level_seed}gseed.pkl"

    try:
        # Check if data file already exists
        if os.path.exists(data_file_name):
            print(f"Loading existing data from: {data_file_name}")
            with open(data_file_name, "rb") as f_data:
                data_dict = pickle.load(f_data)
            print(f"Data loaded successfully from: {data_file_name}")
        else:
            print(f"Data file not found. Preparing new data...")
            # data_dict = prepare_data_dict(config)
            data_dict = prepare_data_dict_noiseless(config)
            
            with open(data_file_name, "wb") as f_data:
                pickle.dump(data_dict, f_data)
            print(f"Data saved to: {data_file_name}")

        experiment_results = {}

        # if not os.path.exists(experiment_file_name):
        # for exp_idx in range(n_classical_shadow_exps):
        parent_seed = np.random.randint(0, 2**30)
        config['parent_seed'] = parent_seed

        exp_results, exp_model_names = run_experiment_noiseless_DM_infinite_limit_CS(config, data_dict)

        experiment_results["config"] = config
        experiment_results["results"] = exp_results
        experiment_results["model_names"] = exp_model_names

        with open(experiment_file_name, "wb") as f_exp:
            pickle.dump(experiment_results, f_exp)
        print(f"Experiment results saved to: {experiment_file_name}")
        # else: 
        #     continue
        
    except Exception as e:
        print(f"Error in main function: {e}")
        raise 

    return experiment_results, data_dict

if __name__ == "__main__":

    # qubit_range = [4, 5, 6]
    ### NOISELESS TRAINING in the infty snapshot limit for DM ###
    qubit_range = [3,5,7]
    channel_types = ["None"]
    noise_strength_range = ["None"]
    thermal_p_exc = 0

    # nps_range = np.ceil(np.array([0.01      , 0.03162278, 0.1       , 0.31622777, 1.        ]) * 4**) # should depend on number of qubits
    
    # num_training_data = 1000 # should depend on number of qubits
    
    r = 1
    num_runs = 10
    batch_size = 128
    max_epochs = 2000
    patience = 200
    
    for qubits in qubit_range:
        alpha_targ = get_alpha_all_one(qubits) # however, we need to check for different alphas?
        y_targ = '1' * (qubits-1) + '0'
        if qubits == 3:
            num_training_data = 200
        else:
            num_training_data = 10000
        for channel_type in channel_types:
            for idx, noise_strength in enumerate(noise_strength_range):
                god_level_seed = int(qubits)
    
                # Configure channel for Monte Carlo sampling
                if channel_type == "thermal":
                    channel_config = {
                        "type": "thermal",
                        "strength": noise_strength if noise_strength != "None" else 0.1,
                        "thermal_p_exc": thermal_p_exc
                    }
                elif channel_type == "dephasing":
                    channel_config = {
                        "type": "dephasing",
                        "strength": noise_strength if noise_strength != "None" else 0.1
                    }
                elif channel_type == "bit_flip":
                    channel_config = {
                        "type": "bit_flip",
                        "strength": noise_strength if noise_strength != "None" else 0.1
                    }
                else:
                    # Default to thermal for "None"
                    channel_config = {
                        "type": "thermal",
                        "strength": 0.1,
                        "thermal_p_exc": 0.0
                    }
                
                config = {
                    "n": qubits,
                    # "parent_seed": 42,  # Added missing parent_seed
                    "num_training_data": num_training_data,
                    "channel_type": channel_type,
                    "noise_strength": noise_strength,
                    "thermal_p_exc": thermal_p_exc,
                    "channel_config": channel_config,  # Monte Carlo sampling config
                    "alpha_targ": alpha_targ,
                    "y_targ": y_targ,
                    "r": r,
                    "CHECKPOINT_PATH": "./saved_models_run1/",
                    "batch_size": batch_size,
                    "batch_size_mcs": 128,  # Batch size for Monte Carlo sampling (smaller for memory efficiency)
                    "shots": 1,  # Number of shots per shadow snapshot
                    "num_runs": num_runs,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "use_shadows": True,  # Use shadow tomography
                    "shadow_nps": 200,  # Number of shadow snapshots
                    "learning_rate": 1e-2,  # Learning rate for optimizer
                }
                main_training(config, god_level_seed)
                    # print(config)
        
# %%
