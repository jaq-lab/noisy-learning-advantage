import numpy as np
from tqdm import tqdm
import warnings
import sys
import os
from contextlib import contextmanager
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
import tensorcircuit as tc
from qiskit.circuit import QuantumCircuit
import channel_sampler as cs
import noisy_sim as ns
import jax.numpy as jnp
from qiskit.transpiler.instruction_durations import InstructionDurations

# Suppress Qiskit transpiler warnings about basis translation
warnings.filterwarnings('ignore', message='.*Unable to translate the operations.*')
warnings.filterwarnings('ignore', category=UserWarning, module='qiskit')

@contextmanager
def suppress_qiskit_warnings():
    """Context manager to suppress Qiskit transpiler warnings."""
    import logging
    # Suppress Qiskit logger warnings
    qiskit_logger = logging.getLogger('qiskit')
    old_level = qiskit_logger.level
    qiskit_logger.setLevel(logging.ERROR)
    
    # Also suppress warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Unable to translate the operations.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='qiskit')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='qiskit')
        try:
            yield
        finally:
            qiskit_logger.setLevel(old_level)

def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count the number of two-qubit gates (CNOTs) in a circuit."""
    count = 0
    for instruction in circuit.data:
        if len(instruction.qubits) == 2:
            count += 1
    return count

def count_single_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count the number of single-qubit gates (excluding measurements and delays) in a circuit."""
    count = 0
    for instruction in circuit.data:
        if len(instruction.qubits) == 1:
            gate_name = instruction.operation.name
            # Exclude measurement and delay gates
            if gate_name not in ['measure', 'delay']:
                count += 1
    return count

def get_qubits_with_only_measurement(ideal_circuit: QuantumCircuit) -> set:
    """
    Identify qubits that only have measurement operations in the ideal circuit.
    Returns a set of qubit indices.
    """
    qubits_with_gates = set()
    qubits_with_measurement = set()
    
    for instruction in ideal_circuit.data:
        gate_name = instruction.operation.name
        qubits = [ideal_circuit.find_bit(q).index for q in instruction.qubits]
        
        if gate_name == 'measure':
            qubits_with_measurement.update(qubits)
        else:
            qubits_with_gates.update(qubits)
    
    # Qubits that have measurement but no other gates
    return qubits_with_measurement - qubits_with_gates

def remove_delays_from_measurement_only_qubits(
    transpiled_circuit: QuantumCircuit, 
    ideal_circuit: QuantumCircuit,
    bit_mapping: list = None
) -> QuantumCircuit:
    """
    Remove delay gates from qubits that only have measurement in the ideal circuit.
    This is useful because Qiskit's scheduler adds unnecessary delays to synchronize
    qubits, but qubits that only have measurement don't need to wait for other qubits.
    
    Args:
        transpiled_circuit: The transpiled circuit (may have delay gates)
        ideal_circuit: The ideal circuit (to identify measurement-only qubits)
        bit_mapping: Optional mapping from logical to physical qubits (if available)
    
    Returns:
        A new circuit with delay gates removed from measurement-only qubits
    """
    # Get qubits that only have measurement in ideal circuit (logical qubit indices)
    measurement_only_logical_qubits = get_qubits_with_only_measurement(ideal_circuit)
    
    if not measurement_only_logical_qubits:
        # No measurement-only qubits, return circuit as-is
        return transpiled_circuit
    
    # Map logical qubits to physical qubits if bit_mapping is provided
    # bit_mapping[i] gives the logical qubit index for physical qubit i
    if bit_mapping is not None:
        # Create reverse mapping: logical -> physical
        logical_to_physical = {logical: physical for physical, logical in enumerate(bit_mapping)}
        measurement_only_physical_qubits = {
            logical_to_physical.get(logical) 
            for logical in measurement_only_logical_qubits 
            if logical in logical_to_physical
        }
    else:
        # If no mapping provided, try direct index matching (may not work if qubits are remapped)
        # Alternative: check if qubit only has delays and measurement in transpiled circuit
        measurement_only_physical_qubits = set()
        
        # For each qubit in transpiled circuit, check if it only has delays and measurement
        qubit_gates = {}
        for instruction in transpiled_circuit.data:
            gate_name = instruction.operation.name
            qubits = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
            
            for qubit_idx in qubits:
                if qubit_idx not in qubit_gates:
                    qubit_gates[qubit_idx] = set()
                if gate_name not in ['delay', 'measure']:
                    qubit_gates[qubit_idx].add('has_other_gates')
                elif gate_name == 'measure':
                    qubit_gates[qubit_idx].add('has_measurement')
        
        # Qubits that have measurement but no other gates (only delays and measurement)
        measurement_only_physical_qubits = {
            q for q, gates in qubit_gates.items() 
            if 'has_measurement' in gates and 'has_other_gates' not in gates
        }
    
    if not measurement_only_physical_qubits:
        return transpiled_circuit
    
    # Create a new circuit with the same structure
    new_circuit = QuantumCircuit(*transpiled_circuit.qregs, *transpiled_circuit.cregs)
    
    # Copy all instructions except delay gates on measurement-only qubits
    for instruction in transpiled_circuit.data:
        gate_name = instruction.operation.name
        qubits = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
        
        # Skip delay gates on qubits that only have measurement
        if gate_name == 'delay' and len(qubits) == 1 and qubits[0] in measurement_only_physical_qubits:
            continue
        
        # Copy all other instructions
        new_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
    
    return new_circuit

def remove_last_delays_before_measurements(transpiled_circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Remove the last delay gates at the end of the circuit for each qubit.
    This removes unnecessary delays that occur at the very end of the circuit execution,
    regardless of whether there are explicit measurement gates. These final delays don't
    affect the measurement outcome since they occur after all gates have been applied.
    
    For qubits that only have delays (no other gates), all delays are removed.
    
    Args:
        transpiled_circuit: The transpiled circuit (may have delay gates at the end)
    
    Returns:
        A new circuit with the last delay gates removed for each qubit
    """
    # Create a new circuit with the same structure
    new_circuit = QuantumCircuit(*transpiled_circuit.qregs, *transpiled_circuit.cregs)
    
    # Track the last non-delay instruction index for each qubit
    last_non_delay_idx = {}
    
    # First pass: find the last non-delay instruction for each qubit
    for i, instruction in enumerate(transpiled_circuit.data):
        gate_name = instruction.operation.name
        qubits = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
        
        # Update last non-delay index for each qubit involved
        for qubit_idx in qubits:
            if gate_name != 'delay':
                last_non_delay_idx[qubit_idx] = i
    
    # Second pass: identify delay gates that come after the last non-delay gate for each qubit
    # OR all delay gates for qubits that only have delays
    instructions_to_skip = set()
    
    for i, instruction in enumerate(transpiled_circuit.data):
        gate_name = instruction.operation.name
        qubits = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
        
        # If this is a delay gate on a single qubit
        if gate_name == 'delay' and len(qubits) == 1:
            qubit_idx = qubits[0]
            
            # Case 1: Qubit has non-delay gates - remove delays after the last non-delay gate
            if qubit_idx in last_non_delay_idx:
                if i > last_non_delay_idx[qubit_idx]:
                    instructions_to_skip.add(i)
            # Case 2: Qubit only has delays - remove ALL delays for this qubit
            else:
                instructions_to_skip.add(i)
    
    # Third pass: build the new circuit, skipping marked delay gates
    for i, instruction in enumerate(transpiled_circuit.data):
        if i not in instructions_to_skip:
            new_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
    
    return new_circuit

class QuantumDeviceSimulator:
    def __init__(self, nq, basis_gates, connectivity, noise_model=None, readout_error=None, transpiler_seed=42, num_transpilation_trials=200, T1=None, T2=None, delay_gate_configs=None, Tidle=None):
        self.nq = nq
        # delay_gate_configs: List of (gate_name, delay_ns) tuples, e.g., [("rz", 4000), ("ry", 1500), ("rx", 500)]
        # Default to old values if not provided for backward compatibility
        if delay_gate_configs is None:
            delay_gate_configs = [("rz", 10000), ("ry", 5000), ("rx", 2000)]
        self.properties = {"basis_gates": basis_gates, "connectivity": connectivity, "noise_model": noise_model, "readout_error": readout_error, "T1": T1, "T2": T2, "delay_gate_configs": delay_gate_configs, "Tidle": Tidle}
        self.transpiler_seed = transpiler_seed  # Seed for deterministic transpilation
        self.num_transpilation_trials = num_transpilation_trials  # Number of trials to find best transpilation

    def _qiskit_to_tc(self, qiskit_circuit, state=None, Device=None):
        """Converts a Qiskit circuit to a TensorCircuit circuit."""
        tc_c = tc.Circuit(qiskit_circuit.num_qubits, inputs=state)
        if qiskit_circuit.global_phase != 0: tc_c.global_phase(qiskit_circuit.global_phase)
        name_map = {
        "h": "H", 
        "rz": "RZ", 
        "cx": "CNOT", 
        "x": "X", 
        "id": "RZ",  # Map id to RZ with theta=0 (identity gate equivalent)
        "sx": "SX", 
        "ry": "RY",
         }
        
        param_map = {
        "RZ": ["theta"],
        "RX": ["theta"],
        "RY": ["theta"],
        "U": ["theta", "phi", "lmbda"],
        "RXX": ["theta"],
         }

        for instruction in qiskit_circuit.data:
            gate = instruction.operation
            qubits = [qiskit_circuit.qubits.index(q) for q in instruction.qubits]
            qiskit_name = gate.name
            
            # Handle delay gates specially - convert to multiple RZ(0) gates
            # Each RZ(0) gate represents BASE_DELAY_UNIT of delay time (equivalent to identity).
            # Thermal relaxation noise is added to RZ gates via noise configuration.
            # We use RZ(0) instead of identity gates because TensorCircuit's Circuit doesn't have .i() method.
            if qiskit_name == "delay":
                qubit_idx = qubits[0]
                
                # Get delay time to determine how many identity gates to create
                delay_time = None
                if hasattr(gate, 'duration') and gate.duration is not None:
                    delay_time = gate.duration
                elif hasattr(gate, 'params') and len(gate.params) > 0:
                    delay_time = gate.params[0]
                
                if delay_time is not None:
                    if hasattr(delay_time, 'eval'):
                        try: delay_time = float(delay_time.eval())
                        except: delay_time = 0.0
                    elif hasattr(delay_time, '__float__'):
                        try: delay_time = float(delay_time)
                        except: delay_time = 0.0
                
                # Get Tidle from device properties (preferred) or fall back to T1
                Tidle = self.properties.get("Tidle")
                if Tidle is None:
                    Tidle = self.properties.get("T1")
                
                # Skip gate if t/Tidle < 10e-4 (where t is delay_time)
                # This avoids adding delay gates when the delay is negligible compared to Tidle
                if Tidle is not None and delay_time is not None and delay_time > 0:
                    # Tidle is typically in nanoseconds, delay_time should also be in nanoseconds
                    ratio = delay_time / Tidle
                    if ratio < 10e-4:  # 0.0001
                        # Skip this gate - delay is too short relative to Tidle
                        continue
                
                # Convert delay to combination of identity-equivalent gates with different delay values
                # OPTIMIZATION: Use different gate types with different delay amounts to minimize gate count
                # Each gate type must be unique so noise configuration can distinguish them
                # Gate delay amounts are now device-specific and come from device properties
                
                # Get delay gate configs from device properties (device-specific)
                delay_gate_configs = self.properties.get("delay_gate_configs", [("rz", 10000), ("ry", 5000), ("rx", 2000)])
                
                # OPTIMIZATION: Pre-compute gate methods to avoid lambda closure overhead
                gate_method_map = {
                    "rz": tc_c.rz,
                    "ry": tc_c.ry,
                    "rx": tc_c.rx,
                }
                
                # Build DELAY_GATES list from device-specific config (sorted by delay, largest first)
                DELAY_GATES = []
                for gate_name, delay_ns in sorted(delay_gate_configs, key=lambda x: x[1], reverse=True):
                    if gate_name in gate_method_map:
                        DELAY_GATES.append((delay_ns, gate_method_map[gate_name]))
                
                if delay_time and delay_time > 0:
                    remaining_delay = float(delay_time)
                    # Use greedy approach: use largest gate first, then smaller ones
                    for gate_delay_ns, gate_method in DELAY_GATES:
                        if remaining_delay <= 0:
                            break
                        num_gates = int(remaining_delay / gate_delay_ns)
                        if num_gates > 0:
                            # Direct method calls - faster than lambda functions
                            for _ in range(num_gates):
                                gate_method(qubit_idx, theta=0.0)
                            remaining_delay -= num_gates * gate_delay_ns
                    
                    # If there's any remaining delay, use smallest gate (last in sorted list)
                    if remaining_delay > 0 and len(DELAY_GATES) > 0:
                        smallest_gate_method = DELAY_GATES[-1][1]  # Last gate is smallest
                        smallest_gate_method(qubit_idx, theta=0.0)
                else:
                    # Fallback: single smallest gate if delay_time is unknown
                    if len(DELAY_GATES) > 0:
                        smallest_gate_method = DELAY_GATES[-1][1]  # Last gate is smallest
                        smallest_gate_method(qubit_idx, theta=0.0)
                continue 
            
            tc_gate_name = name_map.get(qiskit_name, qiskit_name.upper())

            if not hasattr(tc_c, tc_gate_name):
                continue

            tc_gate_method = getattr(tc_c, tc_gate_name)

            # Special handling for identity gates: convert to RZ(0)
            if qiskit_name == "id":
                qubit_idx = qubits[0]
                tc_c.rz(qubit_idx, theta=0.0)
            elif tc_gate_name in param_map:
                params_dict = dict(zip(param_map[tc_gate_name], gate.params))
                tc_gate_method(*qubits, **params_dict)
            else:
                tc_gate_method(*qubits)
                
        return tc_c
       
    def extract_subcircuit(self, original_circuit: QuantumCircuit, qubits_to_keep: list[int], 
                          logical_order: dict = None) -> QuantumCircuit:
        """Simplified: Remove ancilla qubits, keep only virtual qubits in logical order."""
        if logical_order is not None:
            sorted_physical = [logical_order[log_idx] for log_idx in sorted(logical_order.keys())]
            ordered_qubits = [p for p in sorted_physical if p in qubits_to_keep]
        else:
            ordered_qubits = qubits_to_keep
            
        phys_to_position = {phys: log_idx for log_idx, phys in enumerate(ordered_qubits)}
        sub_circ = QuantumCircuit(len(ordered_qubits))
        original_q_to_idx_map = {q: i for i, q in enumerate(original_circuit.qubits)}
        
        for instr in original_circuit.data:
            phys_indices = [original_q_to_idx_map[q] for q in instr.qubits]
            if all(idx in qubits_to_keep for idx in phys_indices):
                logical_positions = [phys_to_position[idx] for idx in phys_indices]
                logical_qubits = [sub_circ.qubits[i] for i in logical_positions]
                sub_circ.append(instr.operation, logical_qubits, instr.clbits)
        return sub_circ
        
    def get_bit_mapping(self,original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit) -> dict:
        if getattr(transpiled_circuit, 'layout', None) is None:
            return {i: i for i in range(original_circuit.num_qubits)}
        final_layout = transpiled_circuit.layout.final_layout
        repr_to_logical_idx = {repr(q): i for i, q in enumerate(original_circuit.qubits)}
        phys_to_final_qubit = final_layout.get_physical_bits()
        bit_mapping = {}
        for physical_idx, final_qubit_obj in phys_to_final_qubit.items():
            final_qubit_repr = repr(final_qubit_obj)
            if final_qubit_repr in repr_to_logical_idx:
                original_idx = repr_to_logical_idx[final_qubit_repr]
                bit_mapping[physical_idx] = original_idx
        return bit_mapping

    def remap_circuit(self, circuit: QuantumCircuit, new_order: list[int]) -> QuantumCircuit:
        num_qubits = circuit.num_qubits
        if sorted(new_order) != list(range(num_qubits)):
            raise ValueError("The 'new_order' list must be a permutation of qubit indices.")
        new_circuit = QuantumCircuit(num_qubits, circuit.num_clbits, name=f"{circuit.name}_remapped")
        qubit_map = {circuit.qubits[old_idx]: new_circuit.qubits[new_idx]
                    for new_idx, old_idx in enumerate(new_order)}
        clbit_map = {circuit.clbits[i]: new_circuit.clbits[i] 
                    for i in range(circuit.num_clbits)}
        for instruction in circuit.data:
            op = instruction.operation
            new_qargs = [qubit_map[q] for q in instruction.qubits]
            new_cargs = [clbit_map[c] for c in instruction.clbits]
            new_circuit.append(op, new_qargs, new_cargs)
        return new_circuit

    def count_two_qubit_gates(self, circuit: QuantumCircuit) -> int:
        return count_two_qubit_gates(circuit)

    def transpile(self, ideal_circuit: tc.Circuit) -> QuantumCircuit:
        qiskit_ideal_c = ideal_circuit.to_qiskit()
        if not qiskit_ideal_c.data: return qiskit_ideal_c, None, qiskit_ideal_c
        coupling_map_obj = CouplingMap(self.properties["connectivity"])
        trial_results = []
        
        for trial in range(self.num_transpilation_trials):
            try:
                trial_seed = self.transpiler_seed + trial * 1
                with suppress_qiskit_warnings():
                    transpiled_qiskit_c = transpile(
                        qiskit_ideal_c, basis_gates=self.properties["basis_gates"],
                        coupling_map=coupling_map_obj, 
                        scheduling_method='asap',
                        instruction_durations=InstructionDurations(
                            [("h", None, 50), ("x", None, 50), ("cx", None, 100)], dt=1e-9
                        ),
                        optimization_level=3, 
                        layout_method='sabre', routing_method='sabre', approximation_degree=1,
                        seed_transpiler=trial_seed
                    )
                
                if getattr(transpiled_qiskit_c, 'layout', None) is None:
                    # Apply delay removal functions
                    transpiled_qiskit_c = remove_delays_from_measurement_only_qubits(
                        transpiled_qiskit_c, qiskit_ideal_c, None
                    )
                    transpiled_qiskit_c = remove_last_delays_before_measurements(transpiled_qiskit_c)
                    
                    cnot_count = self.count_two_qubit_gates(transpiled_qiskit_c)
                    trial_results.append({
                        'circuit': transpiled_qiskit_c,
                        'cnot_count': cnot_count,
                        'bit_mapping': None,
                        'full_circuit': transpiled_qiskit_c,
                    })
                    continue

                init_layout_phys = list(transpiled_qiskit_c.layout.initial_layout.get_physical_bits().keys())
                final_layout_phys = list(transpiled_qiskit_c.layout.final_virtual_layout().get_physical_bits().keys())
                init_virtual = [p for p in init_layout_phys if p in final_layout_phys]
                logical_order = {i: p for i, p in enumerate(init_virtual)}
                reordered_circuit = self.extract_subcircuit(transpiled_qiskit_c, init_virtual, logical_order=logical_order)
                bit_mapping = [init_virtual.index(final_phys) for final_phys in final_layout_phys if final_phys in init_virtual]
                
                # Apply delay removal functions to the reordered circuit
                reordered_circuit = remove_delays_from_measurement_only_qubits(
                    reordered_circuit, qiskit_ideal_c, bit_mapping
                )
                reordered_circuit = remove_last_delays_before_measurements(reordered_circuit)
                
                cnot_count = self.count_two_qubit_gates(reordered_circuit)
                
                trial_results.append({
                    'circuit': reordered_circuit,
                    'cnot_count': cnot_count,
                    'bit_mapping': bit_mapping,
                    'full_circuit': transpiled_qiskit_c,
                })
            except Exception as e:
                continue
        
        if len(trial_results) == 0:
            print(f"Warning: All {self.num_transpilation_trials} transpilation trials failed")
            return qiskit_ideal_c, None, qiskit_ideal_c
        
        best_result = min(trial_results, key=lambda x: x['cnot_count'])
        return best_result['circuit'], best_result['bit_mapping'], best_result['full_circuit']

def precompute_transpiled_templates(nq: int, device: QuantumDeviceSimulator) -> dict:
    print(f"\nPre-computing transpiled circuit templates for nq={nq}...")
    templates = {}
    measurement_map = {}
    full_circs = {}
    ideal_circs = {}
    trimmed_circs = {}
    template_cnots = {}
    delay_info = {}
    dummy_input_state = jnp.zeros(2**nq, dtype=jnp.complex64).at[0].set(1.0)
    
    for d in tqdm(range(1, nq + 1), desc="Transpiling templates"):
        canonical_alpha = np.zeros(nq, dtype=int)
        if d > 1: canonical_alpha[:d-1] = 1
        canonical_alpha[-1] = 1
        dummy_qc = ns.deploy_decoder(tc.Circuit(nq, inputs=dummy_input_state), canonical_alpha, nq)
        
        ideal_qiskit = dummy_qc.to_qiskit()
        ideal_cnots = count_two_qubit_gates(ideal_qiskit)
        ideal_single_qubit = count_single_qubit_gates(ideal_qiskit)
        ideal_h = sum(1 for instr in ideal_qiskit.data if instr.operation.name == 'h')
        
        transpiled_qiskit, bit_mapping, full_circ = device.transpile(dummy_qc)
        
        transpiled_cnots = count_two_qubit_gates(transpiled_qiskit)
        transpiled_single_qubit = count_single_qubit_gates(transpiled_qiskit)
        transpiled_h = sum(1 for instr in transpiled_qiskit.data if instr.operation.name == 'h')
        
        cnot_overhead = transpiled_cnots / ideal_cnots if ideal_cnots > 0 else float('inf')
        
        delay_positions = []
        for gate_idx, instruction in enumerate(transpiled_qiskit.data):
            gate = instruction.operation
            if gate.name == "delay":
                qubits = [transpiled_qiskit.qubits.index(q) for q in instruction.qubits]
                qubit_idx = qubits[0]
                
                delay_time = None
                if hasattr(gate, 'duration') and gate.duration is not None:
                    delay_time = gate.duration
                elif hasattr(gate, 'params') and len(gate.params) > 0:
                    delay_time = gate.params[0]
                
                if delay_time is not None:
                    if hasattr(delay_time, 'eval'):
                        try: delay_time = float(delay_time.eval())
                        except: continue
                    elif hasattr(delay_time, '__float__'):
                        try: delay_time = float(delay_time)
                        except: continue
                    
                    
                    
                    delay_positions.append((gate_idx, qubit_idx, delay_time))
        
        delay_info[d] = delay_positions
        full_circs[d] = full_circ
        ideal_circs[d] = ideal_qiskit
        trimmed_circs[d] = transpiled_qiskit
        measurement_map[d] = bit_mapping
        
        templates[d] = device._qiskit_to_tc(transpiled_qiskit, state=dummy_input_state)
        
        # Map Qiskit gate indices to TensorCircuit QIR positions
        tc_qir = templates[d].to_qir()
        tc_delay_info = []
        qiskit_idx = 0
        for tc_idx, tc_gate_entry in enumerate(tc_qir):
            if qiskit_idx >= len(transpiled_qiskit.data): break
            qiskit_gate = transpiled_qiskit.data[qiskit_idx].operation
            
            if qiskit_gate.name == "delay":
                for qiskit_delay_idx, qubit_idx, delay_time in delay_positions:
                    if qiskit_delay_idx == qiskit_idx:
                        gate_name = tc_gate_entry.get('name', '') if isinstance(tc_gate_entry, dict) else (tc_gate_entry.name if hasattr(tc_gate_entry, 'name') else str(tc_gate_entry))
                        if gate_name and ('i' in gate_name.lower() or 'id' in gate_name.lower()):
                            tc_delay_info.append((tc_idx, qubit_idx, delay_time))
                        break
            qiskit_idx += 1
        
        delay_info[d] = tc_delay_info
        template_cnots[d] = {
            "ideal_cnots": int(ideal_cnots),
            "transpiled_cnots": int(transpiled_cnots),
            "ideal_single_qubit": int(ideal_single_qubit),
            "transpiled_single_qubit": int(transpiled_single_qubit),
            "ideal_h": int(ideal_h),
            "transpiled_h": int(transpiled_h),
            "cnot_overhead": float(cnot_overhead)
        }

    print("✅ All templates pre-computed and cached.")
    return templates, measurement_map, full_circs, ideal_circs, trimmed_circs, template_cnots, delay_info