import numpy as np
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

import numpy as np
import json
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator
# Import the FakeBackend for local simulation
from qiskit.quantum_info import DensityMatrix
from functools import reduce # Needed for a clean way to apply Kronecker product repeatedly
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Layout
## LOG IN TO IBM
from qiskit_ibm_runtime import QiskitRuntimeService


from qiskit_aer.noise.errors import thermal_relaxation_error, depolarizing_error

from qiskit_aer.noise import NoiseModel

def get_non_ancilla_qubits(circuit: QuantumCircuit):
    used_qubits = set()
    for instr, qargs, _ in circuit.data:
        for q in qargs:
            used_qubits.add(q)
    return sorted(circuit.qubits.index(q) for q in used_qubits)


def manually_restrict_noise_model(full_noise_model: NoiseModel, active_qubits: list[int]) -> NoiseModel:
    """
    Creates a new NoiseModel containing only the errors from the full_noise_model
    that apply to the specified active_qubits.

    This is a manual implementation of the '.restrict_to_qubits()' method.

    Args:
        full_noise_model: The original NoiseModel created from a full backend.
        active_qubits: A list of qubit indices to keep.

    Returns:
        A new, lightweight NoiseModel.
    """
    
    new_noise_model = NoiseModel()
    active_qubits_set = set(active_qubits)

    # --- Manually copy the Quantum Errors (Gate errors, T1/T2) ---
    # We access the internal list of errors. This is not standard practice,
    # but necessary to solve this problem without the built-in method.
    
    # In legacy versions, errors were stored in a dict. In modern versions, it is a list of tuples.
    # We will check for both structures for maximum compatibility.
    if isinstance(full_noise_model._local_quantum_errors, dict):
        # Legacy Structure: {'gate_name': [(qubits, error_obj), ...]}
        for gate, error_data in full_noise_model._local_quantum_errors.items():
            for qubits, error_channel in error_data:
                # Check if all qubits for this error are in our active set
                if active_qubits_set.issuperset(qubits):
                    new_noise_model.add_quantum_error(error_channel, gate, qubits)
    else:
        # Modern Structure: [(gate_name, qubits, error_obj), ...]
        for error_tuple in full_noise_model._local_quantum_errors:
            gate, qubits, error_channel = error_tuple
            if active_qubits_set.issuperset(qubits):
                 new_noise_model.add_quantum_error(error_channel, gate, qubits)

    # Note: A similar loop can be done for `_local_readout_errors` if needed,
    # but the quantum errors are the cause of the SuperOperator memory issue.

    return new_noise_model

def extract_subcircuit(original_circuit, qubits):
    sub_circ = QuantumCircuit(len(qubits))
    for instr, qargs, cargs in original_circuit.data:
        # Check if all qubits involved in this gate are within target_qubits
        qubit_indices = [original_circuit.qubits.index(q) for q in qargs]
        if all(q in qubits for q in qubit_indices):
            # Map global qubit index to local one in subcircuit
            local_qargs = [sub_circ.qubits[qubits.index(q)] for q in qubit_indices]
            sub_circ.append(instr, local_qargs, cargs)
    return sub_circ

def alphas_to_superops(alphas: np.ndarray, nq: int) -> np.ndarray:
    superops = []
    circuits = []
    backend = service.backend('ibm_brisbane')
    noise_model = NoiseModel.from_backend(backend)
    noisy_simulator = AerSimulator(noise_model=noise_model)

    for k,alpha in enumerate(alphas):
        meas_circuit = QuantumCircuit(nq)
        meas_circuit = deploy_decoder(meas_circuit, alpha, nq)

        transpiled_circuit0 = transpile(meas_circuit, backend=backend,     optimization_level=3)

        non_ancilla = list(transpiled_circuit0.layout.final_virtual_layout().get_physical_bits().keys())


        transpiled_circuit = extract_subcircuit(transpiled_circuit0, non_ancilla)

        transpiled_circuit.save_superop()

        job = noisy_simulator.run(transpiled_circuit)
        result = job.result()
        superops.append(result.data(0)['superop'])
        circuits.append(transpiled_circuit)

    return superops, circuits


def deploy_decoder(qc, x, qubits):
    """
    Deploy the decoder circuit.

    Args:
        qc (QuantumCircuit): Quantum circuit.
        x (np.ndarray): Array of CNOTs.
        qubits (int): Number of qubits.

    Returns:
        QuantumCircuit: Quantum circuit with the deployed decoder.
    """
    for en, cnot in enumerate(x[:-1]):
        if cnot:
            qc.cx(qubits - 1, en)
    qc.h(qubits - 1)
    # The final measurement is now handled in the main function
    return qc


def build_confusion_matrix(per_qubit_matrices: list[np.ndarray]) -> np.ndarray:
    """
    Constructs the full system confusion matrix from a list of per-qubit matrices.

    Args:
        per_qubit_matrices: A list of 2x2 confusion matrices, ordered from
                            qubit 0 to qubit n-1. E.g., [C0, C1, C2, ...].

    Returns:
        The (2**n x 2**n) confusion matrix for the entire system.
    """
    # Reverse the list to match Qiskit's little-endian convention for the Kronecker product
    # C_total = C_{n-1} otimes ... otimes C_1 otimes C_0
    reversed_matrices = per_qubit_matrices[::-1]
    
    # Apply the Kronecker product iteratively
    # reduce(lambda x, y: np.kron(x, y), my_list) is equivalent to
    # np.kron(np.kron(my_list[0], my_list[1]), my_list[2])...
    full_matrix = reduce(np.kron, reversed_matrices)
    
    return full_matrix

# --- Step 1: Simulate Loading Your F-vector ---
def load_f_vector(nq: int):
    """
    This function simulates loading your F-vector from a file.
    """
    # Generate a sample F-vector (pretend this was loaded)
    dim = 2**nq
    # Using a memory-efficient integer type since f_vector is just 0s and 1s
    f_vector = np.array([i % 2 for i in range(dim)], dtype=np.int8) 
    
    return f_vector

import itertools

def generate_state_labels_itertools(n_qubits: int) -> list[str]:
    """
    Generates state labels for n qubits using itertools.product.
    This is a more concise alternative.
    """
    # itertools.product('01', repeat=n) generates all tuples of '0's and '1's
    # of length n, in lexicographical order.
    # e.g., ('0','0','0'), ('0','0','1'), ...
    bit_tuples = itertools.product('01', repeat=n_qubits)
    
    # We then join each tuple of characters into a single string
    return [''.join(bits) for bits in bit_tuples]


# --- Step 2: Run a BATCH of Noisy Simulations using a FakeBackend ---
def run_qiskit_noisy_simulation_batch(rho_list: list, nq: int, superops: np.ndarray, alphas: np.ndarray,cal_matrix_from_scratch: np.ndarray, shots: int = 1024):
    """
    Constructs a batch of Qiskit circuits for different initial states and
    runs them all at once for maximum efficiency.
    """


    # 1. Build a list of circuits, one for each initial state (f_vector)
    alpha_used = []
    probs = []
    state_labels = generate_state_labels_itertools(nq)

    for k,rho in enumerate(rho_list):
        alpha_ind = np.random.choice(len(alphas), 1)[0]  #
        superop = superops[alpha_ind]  # 
        alpha = alphas[alpha_ind]  #
        rho_f = DensityMatrix(rho)
        rho_final_noisy = rho_f.evolve(superop, qargs=list(range(nq)))
        ideal_probs = rho_final_noisy.probabilities_dict()
        ideal_probs_vec = np.array([ideal_probs.get(label, 0) for label in state_labels])
        noisy_probs_vec = cal_matrix_from_scratch @ ideal_probs_vec
        # Store the results
        alpha_used.append(alpha)
        probs.append(noisy_probs_vec)


    return alpha_used, probs, state_labels



def get_accuracy(f, x_int, results, nr_shots, states):
    accuracy = 0
    for k,state in enumerate(states):
        b = int(state[-1])
        y0 = (state[:-1] + "0")
        y = int(y0, 2)
       

        if check_outcome(f,x_int, y, b):
            accuracy += results[k] / nr_shots
    return accuracy


def check_outcome(f,x, y, b):
    x_int = int("".join(str(int(bit)) for bit in x), 2)
    return int(f[y ^ x_int]) ^ int(f[y]) == b