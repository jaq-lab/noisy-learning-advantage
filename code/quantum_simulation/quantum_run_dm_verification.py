#!/usr/bin/env python3
"""
Density Matrix (DM) simulation for verification against Monte Carlo results

This script runs full density matrix simulations to verify that the Monte Carlo
trajectory method produces the same results. Uses tc.DMCircuit for exact DM evolution.

Parameters:
- nq: 6, 8 (smaller systems for DM simulation)
- Channels: relaxation, depolarizing
- Devices: A, B, C
- Many f states: 20-50 for averaging
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.noisemodel import NoiseConf
from collections import defaultdict
from tqdm import tqdm
from functools import partial
import sys
import os
from datetime import datetime
from pathlib import Path
import json
import gc
import argparse
import hashlib

# Add modules to path — use __file__-relative path so the script works from any CWD
_modules_dir = str(Path(__file__).parent / "modules")
if _modules_dir not in sys.path:
    sys.path.insert(0, _modules_dir)

# Helper function to print to stderr (appears in .err file)
def eprint(*args, **kwargs):
    """Print to stderr with timestamp - goes to .err file for real-time monitoring"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, file=sys.stderr, **kwargs)
    sys.stderr.flush()  # Force flush to ensure immediate output

# Import from our other project files
import quantum_device_sim as qd
import channel_sampler as cs
import noisy_sim as ns
import quantum_run_mc_optimized as qrmc_opt
import device_config as dc

# Import and reload for fresh start
import importlib
importlib.reload(cs)
importlib.reload(ns)
importlib.reload(qd)
importlib.reload(qrmc_opt)

# Set JAX backend
K = tc.set_backend("jax")

# Configure JAX for better memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

# ===================================================================
# PARAMETERS
# ===================================================================

parser = argparse.ArgumentParser(description='Run density matrix simulation for verification')
parser.add_argument('--device', type=str, required=True, choices=['I', 'R', 'T', 'S','S2'],
                    help='Device to run (I, R, T, or S)')
parser.add_argument('--channel', type=str, required=True, 
                    choices=['relaxation', 'depolarizing', 'dephasing'],
                    help='Channel type to run')
parser.add_argument('--nq', type=int, required=True,
                    help='Number of qubits (2-12)')
parser.add_argument('--alpha_pattern', type=str, required=True,
                    choices=['nq/4', 'nq/2', '3/4nq', 'nq'],
                    help='Alpha pattern: nq/4, nq/2, 3/4nq, or nq')
parser.add_argument('--amp', type=float, required=True,
                    help='Preparation noise amplitude (continuous value)')
parser.add_argument('--nf', type=int, default=30,
                    help='Number of f states to process (default: 30)')
parser.add_argument('--f_seed', type=int, default=42,
                    help='Random seed for F matrix generation (default: 42)')
parser.add_argument('--shots', type=int, default=1000,
                    help='Number of shots per f state (default: 1000)')

args = parser.parse_args()

TARGET_DEVICE = args.device
TARGET_CHANNEL = args.channel
N_QUBITS = args.nq
ALPHA_PATTERN = args.alpha_pattern
AMP = args.amp
NUM_F_STATES = args.nf
# Use the provided seed directly (deterministic & matches MC/debug expectations)
F_SEED = args.f_seed
SHOTS_PER_F = args.shots

# Validate nq range
if N_QUBITS < 2 or N_QUBITS > 12:
    raise ValueError(f"nq must be between 4 and 12, got {N_QUBITS}")

nqs_to_run = [N_QUBITS]
channels_to_run = [TARGET_CHANNEL]

OUTPUT_BASE_DIR = Path("dm_verification_results")

# T1/T2 sweep parameters
T2_RATIO = dc.T2_RATIO

# Import device configurations from common config module
DEVICE_CONFIGS = dc.DEVICE_CONFIGS

print(f"Running DM simulation for Device: {TARGET_DEVICE}, Channel: {TARGET_CHANNEL}, nq: {N_QUBITS}")
print(f"Alpha pattern: {ALPHA_PATTERN}, Amp: {AMP}")
print(f"Number of f states: {NUM_F_STATES}, Shots per f: {SHOTS_PER_F}")

# ===================================================================
# DEVICE & CONNECTIVITY DEFINITIONS
# ===================================================================

# Import connectivity and noise functions from common config module
get_all_to_all_connectivity = dc.get_all_to_all_connectivity
get_square_lattice_connectivity = dc.get_square_lattice_connectivity
get_connectivity_for_device = dc.get_connectivity_for_device
create_base_noise_conf = dc.create_base_noise_conf
add_thermal_relaxation_to_noise_conf = dc.add_thermal_relaxation_to_noise_conf

QISKIT_BASIS_GATES = dc.QISKIT_BASIS_GATES
readout_error = dc.READOUT_ERROR

# Import noise configuration objects
noise_confI = dc.noise_confI
noise_confR = dc.noise_confR
noise_confT = dc.noise_confT
noise_confS = dc.noise_confS


def _identify_delay_gates_and_compute_channels(qir, delay_info):
    """
    Identify delay gates in QIR and pre-compute thermal relaxation channels and probabilities.
    
    This function matches the logic from quantum_run_cluster_single_without_IS.py to ensure
    consistent delay gate identification between MC and DM approaches.
    
    Args:
        qir: Quantum IR from transpiled template
        delay_info: Dict with device config (delay_gate_configs, Tidle, idle_error, f1q, f2q)
    
    Returns:
        delay_gate_map: Dict mapping gate_idx -> (gate_name, qubit_idx, delay_amount, p, kraus_ops)
        delay_gate_configs_map: Dict mapping gate_name -> (delay_amount, p, kraus_ops)
        delay_periods: List of (start_idx, end_idx, qubit_idx, total_delay, p, kraus_ops)
    """
    if delay_info is None:
        return {}, {}, []
    
    # Handle case where delay_info is a list (legacy format) - return empty maps
    if isinstance(delay_info, list):
        return {}, {}, []
    
    # delay_info should be a dict with device configuration
    if not isinstance(delay_info, dict):
        return {}, {}, []
    
    delay_gate_configs = delay_info.get('delay_gate_configs', [])
    Tidle = delay_info.get('Tidle', delay_info.get('T1', 1e8))
    idle_error = delay_info.get('idle_error', 'T2')
    
    # Pre-compute thermal relaxation channels and probabilities for each delay gate type
    thermal_noise_scale = delay_info.get('thermal_noise_scale', 1.0)
    
    delay_gate_configs_map = {}
    for gate_name, delay_amount in delay_gate_configs:
        if idle_error == "T2":
            p_base = 1 - np.exp(-2*delay_amount / Tidle)
            p = min(1.0, p_base * thermal_noise_scale)
        elif idle_error == "T1":
            p_base = 1 - np.exp(-delay_amount / Tidle)
            p = min(1.0, p_base * thermal_noise_scale)
        else:
            raise ValueError(f"Unknown idle error: {idle_error}")
        
        # Manually construct Kraus operators as numpy arrays from channel parameters
        if idle_error == "T2":
            # Phase damping channel (T2 dephasing)
            K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=np.complex128)
            K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(p)]], dtype=np.complex128)
            kraus_ops = [K0, K1]
        elif idle_error == "T1":
            # Amplitude damping channel (T1 relaxation)
            K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=np.complex128)
            K1 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=np.complex128)
            kraus_ops = [K0, K1]
        else:
            raise ValueError(f"Unknown idle error: {idle_error}")
        
        delay_gate_configs_map[gate_name.lower()] = (delay_amount, p, kraus_ops)
    
    # OPTIMIZATION: Group consecutive delay gates on the same qubit into delay periods
    delay_periods = []  # List of (start_idx, end_idx, qubit_idx, total_delay, p, kraus_ops)
    delay_gate_map = {}  # Map gate_idx -> (gate_name, qubit_idx, delay_amount, p, kraus_ops)
    
    i = 0
    while i < len(qir):
        gate_info = qir[i]
        gate_name = gate_info.get('name', '').lower()
        
        if gate_name in delay_gate_configs_map:
            qubits = gate_info.get('index', [])
            if len(qubits) > 0:
                qubit_idx = qubits[0]
                start_idx = i
                total_delay = 0.0
                
                # Collect consecutive delay gates on the same qubit
                delay_gates_in_period = []
                while i < len(qir):
                    gate_info_curr = qir[i]
                    gate_name_curr = gate_info_curr.get('name', '').lower()
                    qubits_curr = gate_info_curr.get('index', [])
                    
                    if (gate_name_curr in delay_gate_configs_map and 
                        len(qubits_curr) > 0 and 
                        qubits_curr[0] == qubit_idx):
                        delay_amount, p_single, kraus_ops_single = delay_gate_configs_map[gate_name_curr]
                        total_delay += delay_amount
                        delay_gates_in_period.append((i, gate_name_curr, delay_amount))
                        i += 1
                    else:
                        break
                
                # Compute combined p and Kraus operators for the total delay
                end_idx = i - 1
                if idle_error == "T2":
                    p_combined = 1 - np.exp(-2 * total_delay / Tidle)
                elif idle_error == "T1":
                    p_combined = 1 - np.exp(-total_delay / Tidle)
                else:
                    raise ValueError(f"Unknown idle error: {idle_error}")
                
                # Apply scaling if provided
                p_combined = min(1.0, p_combined * thermal_noise_scale)
                
                # Construct Kraus operators for combined delay
                if idle_error == "T2":
                    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p_combined)]], dtype=np.complex128)
                    K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(p_combined)]], dtype=np.complex128)
                    kraus_ops_combined = [K0, K1]
                elif idle_error == "T1":
                    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p_combined)]], dtype=np.complex128)
                    K1 = np.array([[0.0, np.sqrt(p_combined)], [0.0, 0.0]], dtype=np.complex128)
                    kraus_ops_combined = [K0, K1]
                
                delay_periods.append((start_idx, end_idx, qubit_idx, total_delay, p_combined, kraus_ops_combined))
                
                # Also add individual gates to delay_gate_map for per-gate application
                # Use the combined p and kraus_ops for the last gate in the period
                for gate_idx, gate_name_period, delay_amount_period in delay_gates_in_period:
                    if gate_idx == end_idx:
                        # Last gate in period gets the combined noise
                        delay_gate_map[gate_idx] = (gate_name_period, qubit_idx, delay_amount_period, p_combined, kraus_ops_combined)
                    else:
                        # Other gates in period: mark them but don't apply noise (handled by last gate)
                        delay_gate_map[gate_idx] = (gate_name_period, qubit_idx, delay_amount_period, None, None)
            else:
                i += 1
        else:
            i += 1
    
    return delay_gate_map, delay_gate_configs_map, delay_periods

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def pick_alpha_for_pattern(alpha_pattern, nq):
    """
    Generate alpha pattern based on target |alpha| value.
    Pattern: first |alpha|-1 qubits are 1, then zeros, then last qubit is 1.
    So for |alpha|=3, nq=5: [1,1,0,0,1]
    
    Args:
        alpha_pattern: One of 'nq/4', 'nq/2', '3/4nq', 'nq'
        nq: Number of qubits
    
    Returns:
        alpha: numpy array of shape (nq,) with 0s and 1s
    """
    alpha = np.zeros(nq, dtype=np.int32)
    
    if alpha_pattern == 'nq/4':
        target = nq // 4
    elif alpha_pattern == 'nq/2':
        target = nq // 2
    elif alpha_pattern == '3/4nq':
        target = int(3 * nq / 4)
    elif alpha_pattern == 'nq':
        target = nq
    else:
        raise ValueError(f"Unknown alpha pattern: {alpha_pattern}")
    
    # Pattern: first (target-1) qubits are 1, then zeros, then last qubit is 1
    if target > 0:
        alpha[:target-1] = 1  # First (target-1) qubits
        alpha[-1] = 1  # Last qubit
    
    return alpha

def run_dm_simulation_for_one_state(
    initial_state_vector, transpiled_template, noise_conf, shots, mapping, 
    f1q, f2q, Tidle, delay_gate_configs, idle_error="T2", prep_channel_kraus=None, nq=None, f_state_idx=None, channel_type=None
):
    """
    Run density matrix simulation for a single initial state.
    
    Args:
        initial_state_vector: Initial state vector (2^nq,)
        transpiled_template: Transpiled circuit template (tc.Circuit)
        noise_conf: NoiseConf object (for reference, we reconstruct channels)
        shots: Number of measurement shots
        mapping: Qubit mapping array
        f1q: Single-qubit gate fidelity
        f2q: Two-qubit gate fidelity
        Tidle: Idle time constant in nanoseconds (T1 for amplitude damping, T2 for phase damping)
        delay_gate_configs: List of (gate_name, delay_ns) tuples
        idle_error: Type of idle error ("T1" for amplitude damping, "T2" for phase damping)
    
    Returns:
        samples: (shots, nq) array of measurement results
    """
    # Get number of qubits from the circuit
    if hasattr(transpiled_template, 'nqubits'):
        nq = transpiled_template.nqubits
    elif hasattr(transpiled_template, 'n'):
        nq = transpiled_template.n
    else:
        # Try to infer from the circuit structure
        nq = len(transpiled_template.qubits) if hasattr(transpiled_template, 'qubits') else transpiled_template.circuit_param['nqubits']
    
    # Convert initial state to density matrix
    psi0 = jnp.array(initial_state_vector, dtype=jnp.complex128)
    rho0 = jnp.outer(psi0, jnp.conj(psi0))
    
    # Apply preparation channel if provided (dephasing, depolarizing, or relaxation)
    if prep_channel_kraus is not None:
        # Convert JAX array to list if needed
        if hasattr(prep_channel_kraus, 'shape') and len(prep_channel_kraus.shape) == 3:
            # It's a stacked array, convert to list of individual operators
            Ks_list = [prep_channel_kraus[i] for i in range(prep_channel_kraus.shape[0])]
        elif isinstance(prep_channel_kraus, list):
            Ks_list = prep_channel_kraus
        else:
            # Single operator, wrap in list
            Ks_list = [prep_channel_kraus]
        
        # Apply preparation channel to each qubit
        rho0_dm = tc.DMCircuit(nq, dminputs=rho0)
        for i in range(nq):
            rho0_dm.general_kraus(Ks_list, i)
        # Get the noisy initial density matrix
        rho0 = rho0_dm.state()
    
    # Create DMCircuit with (possibly noisy) initial state
    dmc = tc.DMCircuit(nq, dminputs=rho0)
    
    # Reconstruct noise channels (same as used in NoiseConf)
    # These return lists of Kraus operators
    # TensorCircuit's isotropicdepolarizingchannel parameter is the depolarizing parameter p,
    # not the error probability. The relationship is:

    error1 = tc.channels.isotropicdepolarizingchannel(3/2*(1 - f1q), 1)
    error2 = tc.channels.isotropicdepolarizingchannel(5/4*(1 - f2q), 2)
    # Get QIR and identify delay gates (same logic as MC approach)
    qir = transpiled_template.to_qir()
    
    # Identify delay gates using the same function as MC approach
    delay_info_dict = {
        'delay_gate_configs': delay_gate_configs,
        'Tidle': Tidle,
        'idle_error': idle_error,
        'f1q': f1q,
        'f2q': f2q,
        'thermal_noise_scale': 1.0  # No scaling for DM (deterministic)
    }
    delay_gate_map, delay_gate_configs_map, delay_periods = _identify_delay_gates_and_compute_channels(qir, delay_info_dict)
    
    # DEBUG: Count gate types for comparison with MC
    if f_state_idx is not None and f_state_idx < 3:
        gate_counts = {'h': 0, 'cx': 0, 'rz': 0, 'ry': 0, 'rx': 0, 'delay': 0, 'other': 0}
        for gate_info in qir:
            gate_name = gate_info.get('name', '').lower()
            eprint("!!!!!!",gate_name)
            if gate_name in gate_counts:
                gate_counts[gate_name] += 1
            else:
                gate_counts['other'] += 1
        # Count delay gates from delay_gate_map
        delay_count = len([idx for idx in delay_gate_map if delay_gate_map[idx][3] is not None])  # Count gates with non-None p
        eprint(f"[DEBUG] DM Circuit: {len(qir)} total gates - H:{gate_counts['h']}, CX:{gate_counts['cx']}, RZ:{gate_counts['rz']}, RY:{gate_counts['ry']}, RX:{gate_counts['rx']}, Delay:{delay_count}, Other:{gate_counts['other']}")
        eprint(f"[DEBUG] DM Noise: Applying channels deterministically (full channel evolution)")
        eprint(f"[DEBUG] DM Delay gates: Found {delay_count} delay gates in {len(delay_periods)} delay period(s) (thermal relaxation applied deterministically)")
        
        # Print full circuit structure for comparison with MC
        eprint(f"[DEBUG] ========================================")
        eprint(f"[DEBUG] DM CIRCUIT STRUCTURE (Full QIR):")
        eprint(f"[DEBUG] ========================================")
        for gate_idx, gate_info in enumerate(qir):
            gate_name = gate_info.get('name', '').lower()
            qubits = gate_info.get('index', [])
            params = gate_info.get('parameters', {})
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()]) if params else ""
            qubit_str = ', '.join(map(str, qubits)) if qubits else ""
            eprint(f"[DEBUG]   Gate {gate_idx:2d}: {gate_name.upper():4s} on qubits [{qubit_str}] {param_str}")
        eprint(f"[DEBUG] ========================================")
    
    # Apply each gate and its associated noise
    for gate_idx, gate_info in enumerate(qir):
        gate_name = gate_info.get('name', '').lower()
        qubits = gate_info.get('index', [])
        params = gate_info.get('parameters', {})
        
        # Check if this is a delay gate (from delay_gate_map)
        if gate_idx in delay_gate_map:
            gate_name_delay, qubit_idx, delay_amount, p_delay, kraus_ops_delay = delay_gate_map[gate_idx]
            
            # Match the notebook behavior:
            # - only the last gate in a delay "period" gets the combined thermal Kraus
            # - earlier gates are omitted (RZ/RY/RX are typically R*(0) placeholders, so this is identity)
            if p_delay is not None and kraus_ops_delay is not None:
                # Apply the gate first (if it's RZ/RY/RX with theta=0)
                if gate_name == 'rz':
                    theta = params.get('theta', 0.0)
                    dmc.rz(qubits[0], theta=theta)
                elif gate_name == 'ry':
                    theta = params.get('theta', 0.0)
                    dmc.ry(qubits[0], theta=theta)
                elif gate_name == 'rx':
                    theta = params.get('theta', 0.0)
                    dmc.rx(qubits[0], theta=theta)
                
                # Apply thermal relaxation noise (combined for the delay period)
                # Convert numpy arrays to TensorCircuit format
                kraus_ops_tc = [tc.array_to_tensor(k) for k in kraus_ops_delay]
                dmc.general_kraus(kraus_ops_tc, qubits[0])
            continue  # Skip normal noise application for delay gates
        
        # Apply gate (non-delay gates)
        if gate_name == 'h' and len(qubits) > 0:
            dmc.h(qubits[0])
            # Apply single-qubit depolarizing noise
            dmc.general_kraus(error1, qubits[0])
        elif (gate_name == 'cx' or gate_name == 'cnot') and len(qubits) >= 2:
            dmc.cx(qubits[0], qubits[1])
            # Apply two-qubit depolarizing noise
            # For 2-qubit channels in TensorCircuit, isotropicdepolarizingchannel(..., 2) returns 
            # a KrausList with 16 2-qubit Kraus operators (4x4 matrices)
            # The correct syntax is: dmc.general_kraus(error2, qubits[0], qubits[1])
            # This matches MC's application of error2 (two-qubit noise with f2q fidelity)
            dmc.general_kraus(error2, qubits[0], qubits[1])
        elif gate_name == 'rz' and len(qubits) > 0:
            # Regular RZ gate (not a delay gate)
            theta = params.get('theta', 0.0)
            dmc.rz(qubits[0], theta=theta)
            # No thermal relaxation for non-delay RZ gates
        elif gate_name == 'ry' and len(qubits) > 0:
            # Regular RY gate (not a delay gate)
            theta = params.get('theta', 0.0)
            dmc.ry(qubits[0], theta=theta)
            # No thermal relaxation for non-delay RY gates
        elif gate_name == 'rx' and len(qubits) > 0:
            # Regular RX gate (not a delay gate)
            theta = params.get('theta', 0.0)
            dmc.rx(qubits[0], theta=theta)
            # No thermal relaxation for non-delay RX gates
    
    # Sample from final density matrix (match notebook PRNG logic)
    base_key = jax.random.PRNGKey(42)
    if f_state_idx is None:
        key = base_key
    else:
        key = jax.random.fold_in(base_key, int(f_state_idx) * 20000)
    keys = jax.random.split(key, shots)
    
    def sample_one(key):
        # Sample from the density matrix
        result = dmc.sample(allow_state=True, random_generator=key)
        if isinstance(result, (list, tuple)):
            return result[0]
        return result
    
    samples = jax.vmap(sample_one)(keys)
    
    # Apply mapping
    if mapping is not None:
        samples = samples[:, mapping]
    
    return samples

# ===================================================================
# MAIN LOOP
# ===================================================================

def main():
    try:
        jax.clear_backends()
        print("✅ Cleared JAX compilation cache")
    except:
        pass
    
    print("="*80)
    print(f"Density Matrix Simulation - Verification Mode")
    print(f"Device: {TARGET_DEVICE}, Channel: {TARGET_CHANNEL}, nq: {N_QUBITS}")
    print(f"Number of f states: {NUM_F_STATES}, Shots per f: {SHOTS_PER_F}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use format: nq_amp_device_channel_alpha (alpha pattern included to avoid conflicts)
    # Replace '/' in alpha_pattern with '_' for filesystem compatibility
    alpha_safe = ALPHA_PATTERN.replace('/', '_')
    run_output_root = OUTPUT_BASE_DIR / f"nq{N_QUBITS}_amp{AMP}_{TARGET_DEVICE}_{TARGET_CHANNEL}_{alpha_safe}"
    run_output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_output_root}")
    
    acc_results = np.zeros((1, 1, len(nqs_to_run), len(channels_to_run), NUM_F_STATES))
    alphas_results = np.zeros((1, 1, len(nqs_to_run), len(channels_to_run), NUM_F_STATES))
    
    for n, nq in enumerate(nqs_to_run):
        print(f"\n{'='*80}\n🎯 Processing nq={nq}\n{'='*80}")
        
        dev_config = DEVICE_CONFIGS[TARGET_DEVICE]
        tidle_val = dev_config['Tidle']
        # Use Tidle as T1 for device properties (for backward compatibility)
        t1_val = tidle_val
        t2_val = int(t1_val * T2_RATIO)
        noise_conf_name = dev_config['noise_conf']
        delay_gate_configs = dev_config['delay_gate_configs']
        idle_error = dev_config['idle_error']

        # IMPORTANT:
        # Do NOT key a dict by noise_conf_name to retrieve (f1q,f2q), because multiple devices
        # can share the same noise_conf (e.g. 'S' and 'S2' both use 'noise_confS').
        # Always take fidelities from the selected TARGET_DEVICE config.
        f1q = dev_config['f1q']
        f2q = dev_config['f2q']

        # Map noise_conf_name -> NoiseConf object (no fidelities here)
        noise_conf_objects = {
            'noise_confI': noise_confI,
            'noise_confR': noise_confR,
            'noise_confT': noise_confT,
            'noise_confS': noise_confS,
        }
        if noise_conf_name not in noise_conf_objects:
            raise ValueError(f"Unknown noise_conf '{noise_conf_name}' for device '{TARGET_DEVICE}'")
        conf = noise_conf_objects[noise_conf_name]

        eprint(
            f"[DEBUG] Effective device config: device={TARGET_DEVICE} "
            f"noise_conf={noise_conf_name} f1q={f1q} f2q={f2q} "
            f"Tidle={tidle_val} idle_error={idle_error} delay_gate_configs={delay_gate_configs}"
        )
        
        conf_with_delays = add_thermal_relaxation_to_noise_conf(conf, tidle_val, delay_gate_configs, idle_error)
        
        # Get connectivity for device from common config
        connectivity_for_device = get_connectivity_for_device(TARGET_DEVICE, nq)
        
        dev = qd.QuantumDeviceSimulator(nq, QISKIT_BASIS_GATES, connectivity_for_device, 
                                      conf_with_delays, readout_error, T1=t1_val, T2=t2_val, 
                                      delay_gate_configs=delay_gate_configs, Tidle=tidle_val)
        
        # Generate alpha pattern based on pattern type
        alpha_pattern = pick_alpha_for_pattern(ALPHA_PATTERN, nq)
        alpha_sum = int(np.sum(alpha_pattern))
        
        device_info = {
            'device': dev, 
            'global_idx': 0,
            't1': t1_val, 
            't2': t2_val,
            'amp': AMP,
            'alpha_pattern': alpha_pattern,
            'alpha_sum': alpha_sum,
            'delay_gate_configs': delay_gate_configs
        }
        
        print(f"\n--- Device {TARGET_DEVICE}: T1={device_info['t1']:.0f}, amp={device_info['amp']}, alpha_pattern={ALPHA_PATTERN}, alpha_sum={device_info['alpha_sum']} ---")
        templates, map_data, _, _, _, template_cnots, delay_info = qd.precompute_transpiled_templates(nq, device_info['device'])
        
        # Print gate counts for the target alpha_sum
        if device_info['alpha_sum'] in template_cnots:
            gate_info = template_cnots[device_info['alpha_sum']]
            print(f"\n--- Transpilation Gate Counts (alpha_sum={device_info['alpha_sum']}) ---")
            print(f"  Two-qubit gates (CNOT):")
            print(f"    Ideal: {gate_info['ideal_cnots']}")
            print(f"    Transpiled: {gate_info['transpiled_cnots']}")
            print(f"    Overhead: {gate_info['cnot_overhead']:.2f}x")
            print(f"  Single-qubit gates:")
            print(f"    Ideal: {gate_info['ideal_single_qubit']}")
            print(f"    Transpiled: {gate_info['transpiled_single_qubit']}")
            if gate_info['ideal_single_qubit'] > 0:
                single_qubit_overhead = gate_info['transpiled_single_qubit'] / gate_info['ideal_single_qubit']
                print(f"    Overhead: {single_qubit_overhead:.2f}x")
            print(f"  H gates:")
            print(f"    Ideal: {gate_info['ideal_h']}")
            print(f"    Transpiled: {gate_info['transpiled_h']}")
        
        mappings_jax = {k: jnp.array(v if v is not None else np.arange(nq), dtype=jnp.int32) 
                       for k, v in map_data.items()}
        
        target_alpha_sum = device_info['alpha_sum']
        alpha_pattern = device_info['alpha_pattern']
        device_amp = device_info['amp']
        
        # Use F_SEED directly for fair comparison with MC simulations
        # This ensures both MC and DM use the same F matrix when F_SEED matches
        # NOTE: For comparison purposes, we use F_SEED directly instead of hash-based seed
        # If you need unique seeds per job, use the hash-based approach below (commented out)
        # import hashlib
        # job_seed_str = f"{TARGET_DEVICE}_{TARGET_CHANNEL}_{nq}_{ALPHA_PATTERN}_{AMP}_{F_SEED}"
        # job_seed = int(hashlib.md5(job_seed_str.encode()).hexdigest()[:8], 16) % (2**31)
        # eprint(f"[DEBUG] Job seed: {job_seed} (from: {job_seed_str})")
        
        # Generate F matrix with F_SEED directly (for comparison with MC)
        np.random.seed(F_SEED)
        F_matrix = np.random.randint(2, size=(NUM_F_STATES, 2**nq))
        print(f"Generated F matrix: shape {F_matrix.shape} (seed: {F_SEED})")
        
        # Process each channel
        for nch, CHANNEL_TYPE in enumerate(channels_to_run):
            print(f"\n{'='*80}")
            print(f"[Progress] Device {TARGET_DEVICE}, Channel: {CHANNEL_TYPE} ({nch+1}/{len(channels_to_run)})")
            print(f"{'='*80}")
            
            # Get Kraus operators for the preparation channel
            Ks = qrmc_opt.get_kraus_operators_with_aliases({'type': CHANNEL_TYPE, 'strength': device_amp})
            
            # DEBUG: Print channel configuration to stderr (appears in .err file)
            eprint(f"[DEBUG] ========================================")
            eprint(f"[DEBUG] CHANNEL CONFIGURATION")
            eprint(f"[DEBUG] ========================================")
            eprint(f"[DEBUG] Channel type: {CHANNEL_TYPE}")
            eprint(f"[DEBUG] Device amp: {device_amp}")
            eprint(f"[DEBUG] Ks shape: {Ks.shape}")
            eprint(f"[DEBUG] Number of Kraus operators: {Ks.shape[0]}")
            eprint(f"[DEBUG] First Kraus operator K0:\n{Ks[0]}")
            if Ks.shape[0] > 1:
                eprint(f"[DEBUG] Second Kraus operator K1:\n{Ks[1]}")
            if Ks.shape[0] > 2:
                eprint(f"[DEBUG] Third Kraus operator K2:\n{Ks[2]}")
            if Ks.shape[0] > 3:
                eprint(f"[DEBUG] Fourth Kraus operator K3:\n{Ks[3]}")
            # Print a hash of Ks to verify they're different across channels
            Ks_np = np.array(Ks)  # Convert JAX array to numpy for hashing
            Ks_str = str(Ks_np)
            Ks_hash = hashlib.md5(Ks_str.encode()).hexdigest()
            eprint(f"[DEBUG] Ks hash (full): {Ks_hash}")
            eprint(f"[DEBUG] ========================================")
            
            # Get transpiled template
            template = templates[target_alpha_sum]
            mapping = mappings_jax[target_alpha_sum]
            noise_model = device_info['device'].properties['noise_model']
            
            # Process each f state
            for label in tqdm(range(NUM_F_STATES), desc=f"Processing f states"):
                f_vec = jnp.array(F_matrix[label], dtype=jnp.int32)
                
                # Print first f function (f_state=0)
                if label == 0:
                    f_vec_np = np.array(f_vec)
                    eprint(f"[DEBUG] ========================================")
                    eprint(f"[DEBUG] First f function (f_state=0):")
                    eprint(f"[DEBUG] f(x) values for all {2**nq} states:")
                    # Print in a readable format - show first 32 states, then summary
                    for i in range(min(32, 2**nq)):
                        state_bin = format(i, f'0{nq}b')
                        eprint(f"[DEBUG]   f({state_bin}) = {int(f_vec_np[i])}")
                    if 2**nq > 32:
                        eprint(f"[DEBUG]   ... (showing first 32 of {2**nq} states)")
                    eprint(f"[DEBUG] Sum of f function: {np.sum(f_vec_np)} out of {2**nq} states")
                    eprint(f"[DEBUG] ========================================")
                
                # Generate initial state from F vector
                from shadows_simulation.shadow_mcs_jitted import generate_psi_F_vector
                psi0 = generate_psi_F_vector(f_vec, nq)
                
                # DEBUG: Print initial state vector for first f state to verify f vector indexing
                if label == 0:
                    psi0_np = np.array(psi0)
                    eprint(f"[DEBUG] ========================================")
                    eprint(f"[DEBUG] DM Initial State Vector (from f function):")
                    eprint(f"[DEBUG] ========================================")
                    for i in range(min(8, len(psi0_np))):
                        state_bin = format(i, f'0{nq}b')
                        eprint(f"[DEBUG]   psi[{i}] (|{state_bin}⟩) = {psi0_np[i]:.6f}")
                    if len(psi0_np) > 8:
                        eprint(f"[DEBUG]   ... (showing first 8 of {len(psi0_np)})")
                    eprint(f"[DEBUG] ========================================")
                
                # Run DM simulation with preparation channel
                try:
                    samples = run_dm_simulation_for_one_state(
                        psi0, template, noise_model, SHOTS_PER_F, mapping,
                        f1q, f2q, tidle_val, delay_gate_configs, idle_error,
                        prep_channel_kraus=Ks, nq=nq, f_state_idx=label, channel_type=CHANNEL_TYPE
                    )
                    
                    # DEBUG: Print measurement results and histogram for first few f states
                    # This helps compare uniformity with MC approach
                    if label < 3:  # Print for first 3 f states
                        samples_np = np.array(samples)
                        num_shots_to_print = min(20, len(samples_np))
                        eprint(f"[DEBUG] === DM Measurement Results (first {num_shots_to_print} shots) ===")
                        eprint(f"[DEBUG] Device: {TARGET_DEVICE}, Channel: {CHANNEL_TYPE}, nq={nq}, f_state={label}")
                        eprint(f"[DEBUG] Tidle={tidle_val:.2f}, T1={t1_val:.2f}, T2={t2_val:.2f}")
                        eprint(f"[DEBUG] Samples shape: {samples_np.shape}")
                        eprint(f"[DEBUG] First {num_shots_to_print} shots:")
                        for i in range(num_shots_to_print):
                            shot_str = ''.join(map(str, samples_np[i].astype(int)))
                            eprint(f"[DEBUG]   Shot {i:3d}: {shot_str}")
                        
                        # Print histogram of outcomes
                        powers = 2 ** np.arange(nq - 1, -1, -1, dtype=np.int64)
                        outcome_ints = (samples_np @ powers).astype(np.int64)
                        unique_outcomes, counts = np.unique(outcome_ints, return_counts=True)
                        eprint(f"[DEBUG] Outcome histogram (showing top 10 most frequent):")
                        sorted_indices = np.argsort(counts)[::-1]
                        for idx in sorted_indices[:10]:
                            outcome_bin = format(unique_outcomes[idx], f'0{nq}b')
                            eprint(f"[DEBUG]   {outcome_bin}: {counts[idx]} times ({100*counts[idx]/len(samples_np):.2f}%)")
                        
                        eprint(f"[DEBUG] Total unique outcomes: {len(unique_outcomes)} out of {2**nq} possible")
                        eprint(f"[DEBUG] Most frequent outcome: {format(unique_outcomes[sorted_indices[0]], f'0{nq}b')} ({100*counts[sorted_indices[0]]/len(samples_np):.2f}%)")
                        eprint(f"[DEBUG] Uniformity check: Expected {len(samples_np)/2**nq:.2f} per outcome if uniform")
                        eprint(f"[DEBUG] ========================================")
                    
                    # Compute accuracy
                    accuracy = qrmc_opt.process_results_to_accuracy(
                        samples, jnp.arange(nq, dtype=jnp.int32),
                        f_vec, jnp.array(alpha_pattern, dtype=jnp.int32), nq, nq
                    )
                    
                    # DEBUG: Print accuracy calculation details for first f state (same as MC)
                    if label == 0:
                        samples_np = np.array(samples)
                        f_vec_np = np.array(f_vec)
                        alpha_np = np.array(alpha_pattern)
                        n_measured = nq
                        n_outcomes = 2 ** n_measured
                        
                        # Convert samples to integers for histogram
                        powers = 2 ** np.arange(n_measured - 1, -1, -1, dtype=np.int64)
                        outcome_ints = (samples_np @ powers).astype(np.int64)
                        unique_outcomes, counts = np.unique(outcome_ints, return_counts=True)
                        
                        # Compute accuracy details using same logic as process_results_to_accuracy
                        outcome_values = np.arange(n_outcomes, dtype=np.int32)
                        outcome_bits = ((outcome_values[:, None] >> np.arange(n_measured - 1, -1, -1, dtype=np.int32)) & 1).astype(np.int32)
                        b_vals = outcome_bits[:, -1].astype(np.int32)
                        y_bits = outcome_bits.copy()
                        y_bits[:, -1] = 0
                        
                        # Map to full space
                        full_powers = 2 ** np.arange(nq - 1, -1, -1, dtype=np.int32)
                        y_bits_full = np.zeros((n_outcomes, nq), dtype=np.int32)
                        measurement_indices = np.arange(nq, dtype=np.int32)
                        y_bits_full[np.arange(n_outcomes)[:, None], measurement_indices[None, :]] = y_bits
                        y_vals_full = (y_bits_full @ full_powers).astype(np.int32)
                        
                        alpha_bits_full = alpha_np.astype(np.int32)
                        y_xor_alpha_bits_full = np.bitwise_xor(y_bits_full, alpha_bits_full[None, :])
                        y_xor_alpha_full = (y_xor_alpha_bits_full @ full_powers).astype(np.int32)
                        
                        y_vals_safe = y_vals_full % f_vec_np.shape[0]
                        y_xor_alpha_safe = y_xor_alpha_full % f_vec_np.shape[0]
                        
                        f_y = f_vec_np[y_vals_safe]
                        f_y_xor_alpha = f_vec_np[y_xor_alpha_safe]
                        predicted_b = np.bitwise_xor(f_y, f_y_xor_alpha)
                        
                        alpha_int = int(''.join(map(str, alpha_np.astype(int))), 2)
                        
                        eprint(f"[DEBUG] === Accuracy Calculation Details (DM) ===")
                        eprint(f"[DEBUG] nq={nq}, n_measured={n_measured}, n_outcomes={n_outcomes}")
                        eprint(f"[DEBUG] f_vec shape: {f_vec_np.shape}, alpha: {alpha_np}")
                        eprint(f"[DEBUG] alpha_np={alpha_np}, alpha_int={alpha_int} (binary: {format(alpha_int, f'0{nq}b')})")
                        eprint(f"[DEBUG] Sample y_vals and F lookups (first 8 outcomes):")
                        for i in range(min(8, len(outcome_values))):
                            outcome_int = outcome_values[i]
                            outcome_bin = format(outcome_int, f'0{n_measured}b')
                            y_val = y_vals_full[i]
                            y_bin = format(y_val, f'0{nq}b')
                            y_safe = y_vals_safe[i]
                            y_xor_alpha = y_xor_alpha_full[i]
                            y_xor_alpha_safe_val = y_xor_alpha_safe[i]
                            y_xor_alpha_bin = format(y_xor_alpha, f'0{nq}b')
                            f_y_val = int(f_y[i])
                            f_y_xor_alpha_val = int(f_y_xor_alpha[i])
                            predicted_b_val = int(predicted_b[i])
                            b_val = int(b_vals[i])
                            is_correct = (predicted_b_val == b_val)
                            eprint(f"[DEBUG]   outcome[{i}]={outcome_int:2d} ({outcome_bin}) -> y={y_val:2d} ({y_bin}), "
                                  f"y^alpha={y_xor_alpha:2d} ({y_xor_alpha_bin}), "
                                  f"f[{y_safe}]={f_y_val}, f[{y_xor_alpha_safe_val}]={f_y_xor_alpha_val}, "
                                  f"predicted_b={predicted_b_val}, actual_b={b_val}, correct={is_correct}")
                        eprint(f"[DEBUG] ========================================")
                    
                    # DEBUG: Print accuracy for this f state to stderr (appears in .err file)
                    if label % 10 == 0 or label < 3:  # Print first 3 and every 10th
                        eprint(f"[DEBUG] Channel: {CHANNEL_TYPE} | f_state: {label} | Accuracy: {accuracy:.8f}")
                    
                    # Also print summary every 25 states
                    if (label + 1) % 25 == 0:
                        avg_acc_so_far = np.mean(acc_results[0, 0, n, nch, :label+1])
                        eprint(f"[DEBUG] Channel: {CHANNEL_TYPE} | Processed {label+1}/{NUM_F_STATES} f states | Avg accuracy so far: {avg_acc_so_far:.8f}")
                    
                    # Store result
                    acc_results[0, 0, n, nch, label] = accuracy
                    alphas_results[0, 0, n, nch, label] = int(''.join(map(str, alpha_pattern)), 2)
                    
                except Exception as e:
                    print(f"❌ ERROR: Failed to process f state {label}")
                    print(f"   Error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        # Save results
        acc_filename = f"acc_dm_nq{nq}.npy"
        alpha_filename = f"alphas_dm_nq{nq}.npy"
        np.save(run_output_root / acc_filename, acc_results[:, :, n, :, :])
        np.save(run_output_root / alpha_filename, alphas_results[:, :, n, :, :])
        
        # Save metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "device": TARGET_DEVICE,
            "channel": TARGET_CHANNEL,
            "nq": int(nq),
            "alpha_pattern": ALPHA_PATTERN,
            "amp": float(AMP),
            "num_f_states": int(NUM_F_STATES),
            "shots_per_f": int(SHOTS_PER_F),
            "f_seed": int(F_SEED),
            "simulation_method": "density_matrix",
            "device_config": {
                "T1": float(device_info['t1']),
                "T2": float(device_info['t2']),
                "amp": float(device_info['amp']),
                "alpha_pattern": ALPHA_PATTERN,
                "alpha_sum": int(device_info['alpha_sum']),
                "noise_conf": noise_conf_name
            }
        }
        with (run_output_root / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n✅ DM SIMULATION COMPLETE!")
    print(f"Results saved to: {run_output_root}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR: Simulation failed with exception:")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}")
        sys.exit(1)

