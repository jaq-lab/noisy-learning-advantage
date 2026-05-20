"""
Common device configuration module for quantum simulations.

This module provides:
- Device configurations (DEVICE_CONFIGS)
- Connectivity functions
- Noise configuration utilities
- Shared constants

Used by both DM verification and MC simulation scripts.
"""

import numpy as np
import tensorcircuit as tc
from tensorcircuit.noisemodel import NoiseConf

# ===================================================================
# DEVICE CONFIGURATIONS
# ===================================================================

DEVICE_CONFIGS = {
    'I': {
        'Tidle': 1e6,
        'noise_conf': 'noise_confI',
        'amp': 0.05,
        'delay_gate_configs': [
            ("rz", 5000),   # RZ(0) gate = 10000ns delay
            ("ry", 2000),   # RY(0) gate = 1500ns delay
            ("rx", 1e3),    # RX(0) gate = 500ns delay
        ],
        'idle_error': "T2",
        'connectivity': "all_to_all",
        'f1q': 0.9999,
        'f2q': 0.99
    },
    'R': {
        'Tidle': 999999999,
        'noise_conf': 'noise_confR',
        'amp': 0.05,
        'delay_gate_configs': [
            ("rz", 500),   # RZ(0) gate = 500ns delay
            ("ry", 200),    # RY(0) gate = 200ns delay
            ("rx", 100),    # RX(0) gate = 100ns delay
        ],
        'idle_error': "T2",
        'connectivity': "square_lattice",
        'f1q': 0.99999,
        'f2q': 0.99
    },
    'T': {
        'Tidle': 200e3,
        'noise_conf': 'noise_confT',
        'amp': 0.05,
        'delay_gate_configs': [
            ("rz", 5000),   # RZ(0) gate = 2000ns delay
            ("ry", 1500),   # RY(0) gate = 500ns delay
            ("rx", 200),   # RX(0) gate = 100ns delay
        ],
        'idle_error': "T1",
        'connectivity': "square_lattice",
        'f1q': 0.9999,
        'f2q': 0.999
    },
    'S': {
        'Tidle': 20e3,
        'noise_conf': 'noise_confS',
        'amp': 0.05,    
        'delay_gate_configs': [
            ("rz", 1000),   # RZ(0) gate = 2000ns delay
            ("ry", 250),    # RY(0) gate = 500ns delay
            ("rx", 50),    # RX(0) gate = 100ns delay
        ],
        'idle_error': "T2",
        'connectivity': "square_lattice",        
        'f1q': 0.999,
        'f2q': 0.99
    },
        'S2': {
        'Tidle': 20e3,
        'noise_conf': 'noise_confS',
        'amp': 0.05,    
        'delay_gate_configs': [
            ("rz", 1000),   # RZ(0) gate = 2000ns delay
            ("ry", 250),    # RY(0) gate = 500ns delay
            ("rx", 50),    # RX(0) gate = 100ns delay
        ],
        'idle_error': "T2",
        'connectivity': "square_lattice",        
        'f1q': 0.999,
        'f2q': 0.99
    }
}

# ===================================================================
# SHARED CONSTANTS
# ===================================================================

QISKIT_BASIS_GATES = ["h", "cx"]
READOUT_ERROR = 0.00
T2_RATIO = 1.5  # T2 = T1 * T2_RATIO

# ===================================================================
# CONNECTIVITY FUNCTIONS
# ===================================================================

def get_all_to_all_connectivity(nq):
    """Generate all-to-all connectivity for nq qubits."""
    return [(i, j) for i in range(nq) for j in range(i + 1, nq)]

def get_square_lattice_connectivity(nq):
    """Generate square lattice connectivity for nq qubits."""
    edges = []
    if nq <= 0: 
        return []
    width = round(np.sqrt(nq))
    for i in range(nq):
        if (i + 1 < nq) and ((i + 1) % width != 0):
            edges.append([i, i + 1])
        if (i + width < nq):
            edges.append([i, i + width])
    return edges

def get_connectivity_for_device(device_name, nq):
    """
    Get connectivity edges for a device based on its configuration.
    
    Args:
        device_name: Device name (e.g., 'I', 'R', 'T', 'S')
        nq: Number of qubits
        
    Returns:
        List of connectivity edges
    """
    if device_name not in DEVICE_CONFIGS:
        raise ValueError(f"Unknown device: {device_name}")
    
    connectivity_type = DEVICE_CONFIGS[device_name]['connectivity']
    
    if connectivity_type == "all_to_all":
        return get_all_to_all_connectivity(nq)
    elif connectivity_type == "square_lattice":
        return get_square_lattice_connectivity(nq)
    else:
        raise ValueError(f"Unknown connectivity type: {connectivity_type}")

# ===================================================================
# NOISE CONFIGURATION FUNCTIONS
# ===================================================================

def create_base_noise_conf(f1q, f2q):
    """
    Create a base noise configuration with gate errors (no thermal relaxation yet).
    
    Args:
        f1q: Single-qubit gate fidelity
        f2q: Two-qubit gate fidelity
        
    Returns:
        List [f1q, f2q] representing base noise configuration parameters
    """
    return [f1q, f2q]

def add_thermal_relaxation_to_noise_conf(noise_conf_param, Tidle, delay_gate_configs, idle_error="T2"):
    """
    Add thermal relaxation noise to delay-representing gates in the noise configuration.
    Different gate types represent different delay amounts, and each gets thermal relaxation
    with its corresponding delay duration (in nanoseconds, converted to seconds).
    
    Args:
        noise_conf_param: Base noise configuration parameters [f1q, f2q]
        Tidle: Idle time constant in nanoseconds (T1 for amplitude damping, T2 for phase damping)
        delay_gate_configs: List of (gate_name, delay_ns) tuples, device-specific
        idle_error: Type of idle error ("T1" for amplitude damping, "T2" for phase damping)
        
    Returns:
        NoiseConf object with gate errors and thermal relaxation
    """
    noise_conf = NoiseConf()
    f1q = noise_conf_param[0]
    f2q = noise_conf_param[1]
    
    # TensorCircuit's isotropicdepolarizingchannel parameter is the depolarizing parameter p,
    # not the error probability. The relationship is:
    # For 1-qubit: F = 1 - p*(3/4), so p = (4/3)*(1-F) to get fidelity F
    # For 2-qubit: F = 1 - p*(15/16), so p = (16/15)*(1-F) to get fidelity F
    error1 = tc.channels.isotropicdepolarizingchannel(3/2*(1 - f1q), 1)
    error2 = tc.channels.isotropicdepolarizingchannel(5/4*(1 - f2q), 2)
   
    # Add thermal relaxation to each gate type used for delays (device-specific)
    for gate_name, delay_amount in delay_gate_configs:
        if idle_error == "T2":
            p = 1 - np.exp(-2*delay_amount / Tidle)
            thermal_channel = tc.channels.phasedampingchannel(p)
        elif idle_error == "T1":
            p = 1 - np.exp(-delay_amount / Tidle)
            thermal_channel = tc.channels.amplitudedampingchannel(p, 0.0)
        else:
            raise ValueError(f"Unknown idle error: {idle_error}")
        noise_conf.add_noise(gate_name, thermal_channel)    
    
    # Add gate errors
    noise_conf.add_noise("cx", error2)
    noise_conf.add_noise("h", error1)
    
    return noise_conf

# ===================================================================
# NOISE CONFIGURATION OBJECTS
# ===================================================================

# Create base noise configurations for each device
noise_confI = create_base_noise_conf(DEVICE_CONFIGS['I']['f1q'], DEVICE_CONFIGS['I']['f2q'])
noise_confR = create_base_noise_conf(DEVICE_CONFIGS['R']['f1q'], DEVICE_CONFIGS['R']['f2q'])
noise_confT = create_base_noise_conf(DEVICE_CONFIGS['T']['f1q'], DEVICE_CONFIGS['T']['f2q'])
noise_confS = create_base_noise_conf(DEVICE_CONFIGS['S']['f1q'], DEVICE_CONFIGS['S']['f2q'])

# Dictionary mapping noise conf names to their parameters
NOISE_CONFS = {
    'noise_confI': noise_confI,
    'noise_confR': noise_confR,
    'noise_confT': noise_confT,
    'noise_confS': noise_confS,
}

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_device_config(device_name):
    """
    Get device configuration for a given device name.
    
    Args:
        device_name: Device name (e.g., 'I', 'R', 'T', 'S')
        
    Returns:
        Dictionary with device configuration
    """
    if device_name not in DEVICE_CONFIGS:
        raise ValueError(f"Unknown device: {device_name}. Available devices: {list(DEVICE_CONFIGS.keys())}")
    return DEVICE_CONFIGS[device_name]

def get_noise_conf_for_device(device_name):
    """
    Get base noise configuration parameters for a device.
    
    Args:
        device_name: Device name (e.g., 'I', 'R', 'T', 'S')
        
    Returns:
        List [f1q, f2q] representing base noise configuration parameters
    """
    config = get_device_config(device_name)
    noise_conf_name = config['noise_conf']
    return NOISE_CONFS[noise_conf_name]

