#!/usr/bin/env python3
"""
Quantum Monte Carlo simulation for shadow tomography — single device/channel, no importance sampling.

This script runs one device and one channel. It samples trajectories at the target probability,
simulates them in chunks with grouping of unique states, and writes the same outputs as
quantum_run_cluster_single_without_IS.py. Use the same CLI and the same external modules.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import argparse
import importlib
from datetime import datetime
from pathlib import Path
from functools import partial

# Prevent JAX from preallocating large GPU memory chunks
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

# Local modules (same as original)
sys.path.append(os.path.abspath("modules"))
import quantum_device_sim as qd
import channel_sampler as cs
import noisy_sim as ns
import quantum_run_mc_optimized as qrmc_opt
import device_config as dc

for mod in (cs, ns, qd, qrmc_opt):
    importlib.reload(mod)

# -----------------------------------------------------------------------------
# Constants and configuration (from original)
# -----------------------------------------------------------------------------

OUTPUT_BASE_DIR = Path("cluster_results")
T2_RATIO = 1.5
NUM_MC_SAMPLES_BASE = 2
SHOTS_PER_STATE_BASE = 1000

DEVICE_CONFIGS = dc.DEVICE_CONFIGS
QISKIT_BASIS_GATES = dc.QISKIT_BASIS_GATES
readout_error = dc.READOUT_ERROR
add_thermal_relaxation_to_noise_conf = dc.add_thermal_relaxation_to_noise_conf
get_connectivity_for_device = dc.get_connectivity_for_device

# Noise configs: name -> (conf, f1q, f2q)
NOISE_CONFIGS = {
    "noise_confI": (dc.noise_confI, dc.DEVICE_CONFIGS["I"]["f1q"], dc.DEVICE_CONFIGS["I"]["f2q"]),
    "noise_confR": (dc.noise_confR, dc.DEVICE_CONFIGS["R"]["f1q"], dc.DEVICE_CONFIGS["R"]["f2q"]),
    "noise_confT": (dc.noise_confT, dc.DEVICE_CONFIGS["T"]["f1q"], dc.DEVICE_CONFIGS["T"]["f2q"]),
    "noise_confS": (dc.noise_confS, dc.DEVICE_CONFIGS["S"]["f1q"], dc.DEVICE_CONFIGS["S"]["f2q"]),
}


def eprint(*args, **kwargs) -> None:
    """Print to stderr with timestamp (shows in .err for SLURM)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


# -----------------------------------------------------------------------------
# Adaptive parameters by nq (same logic as original)
# -----------------------------------------------------------------------------


def get_memory_fraction(nq: int) -> str:
    if nq >= 20:
        return "0.65"
    if nq >= 18:
        return "0.70"
    return "0.75"


def get_adaptive_run_params(nq: int) -> tuple[int, int, int]:
    """Returns (num_mc_samples, shots_per_state, trajectory_chunk_size)."""
    if nq >= 25:
        return (NUM_MC_SAMPLES_BASE, 250, 10)
    if nq >= 24:
        return (NUM_MC_SAMPLES_BASE, 250, 10)
    if nq >= 23:
        return (NUM_MC_SAMPLES_BASE, 250, 10)
    if nq >= 22:
        return (NUM_MC_SAMPLES_BASE, 500, 10)
    if nq >= 21:
        return (NUM_MC_SAMPLES_BASE, 500, 10)
    if nq >= 20:
        return (NUM_MC_SAMPLES_BASE, 1000, 145)
    if nq >= 19:
        return (NUM_MC_SAMPLES_BASE, 1000, 250)
    if nq >= 18:
        return (NUM_MC_SAMPLES_BASE, 1000, 500)
    if nq == 16:
        return (NUM_MC_SAMPLES_BASE, 1000, 1000)
    return (NUM_MC_SAMPLES_BASE, 1000, 1000)


def get_unique_batch_size(nq: int, num_unique: int) -> int:
    """Batch size for simulating unique states within a chunk."""
    if nq >= 20:
        return 20
    if nq >= 19:
        return 3
    if nq >= 18:
        return 50
    if nq >= 16:
        return 250
    return num_unique


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Run quantum simulation for single device/channel (no importance sampling)"
    )
    p.add_argument("--device", type=str, required=True, choices=["I", "R", "T", "S"])
    p.add_argument(
        "--channel",
        type=str,
        required=True,
        choices=["relaxation", "dephasing", "depolarizing"],
    )
    p.add_argument("--nq", type=int, default=15)
    p.add_argument("--f_start", type=int, default=0)
    p.add_argument("--nfk", type=int, default=1)
    p.add_argument("--total_nf", type=int, default=5)
    p.add_argument("--f_seed", type=int, default=42)
    p.add_argument("--job_index", type=int, default=None)
    p.add_argument("--job_timestamp", type=str, default=None)
    p.add_argument(
        "--alpha_pattern",
        type=str,
        default=None,
        choices=["nq/4", "nq/2", "3/4nq", "nq"],
    )
    p.add_argument("--amp", type=float, default=None)
    args = p.parse_args()

    if args.f_start < 0:
        raise ValueError(f"f_start must be >= 0, got {args.f_start}")
    if args.nfk < 1:
        raise ValueError(f"nfk must be >= 1, got {args.nfk}")
    if args.f_start + args.nfk > args.total_nf:
        raise ValueError(
            f"f_start ({args.f_start}) + nfk ({args.nfk}) exceeds total_nf ({args.total_nf})"
        )
    if args.job_index is None:
        args.job_index = args.f_start // args.nfk
    return args


# -----------------------------------------------------------------------------
# Alpha patterns
# -----------------------------------------------------------------------------


def pick_alpha_for_device(device_name: str, nq: int) -> np.ndarray:
    """Legacy device-based alpha (A,B,C,D). I,R,T,S must use --alpha_pattern."""
    if device_name in ("A", "D"):
        return np.ones(nq, dtype=np.int32)
    if device_name == "B":
        alpha = np.zeros(nq, dtype=np.int32)
        alpha[: nq // 2 - 1] = 1
        alpha[-1] = 1
        return alpha
    if device_name == "C":
        alpha = np.zeros(nq, dtype=np.int32)
        alpha[:4] = 1
        alpha[-1] = 1
        return alpha
    if device_name in ("I", "R", "T", "S"):
        raise ValueError(
            f"Device {device_name} requires --alpha_pattern. Use pick_alpha_for_pattern instead."
        )
    raise ValueError(f"Unknown device: {device_name}")


def pick_alpha_for_pattern(alpha_pattern: str, nq: int) -> np.ndarray:
    """Pattern: first (|alpha|-1) qubits 1, then zeros, then last qubit 1."""
    alpha = np.zeros(nq, dtype=np.int32)
    if alpha_pattern == "nq/4":
        target = nq // 4
    elif alpha_pattern == "nq/2":
        target = nq // 2
    elif alpha_pattern == "3/4nq":
        target = int(3 * nq / 4)
    elif alpha_pattern == "nq":
        target = nq
    else:
        raise ValueError(f"Unknown alpha pattern: {alpha_pattern}")
    if target > 0:
        alpha[: target - 1] = 1
        alpha[-1] = 1
    return alpha


# -----------------------------------------------------------------------------
# JAX/GPU setup (same behavior as original)
# -----------------------------------------------------------------------------


def setup_jax_and_devices() -> None:
    tc.set_backend("jax")
    print("=" * 80)
    print("Device Detection:")
    print("=" * 80)

    try:
        all_devices = jax.devices()
    except RuntimeError as e:
        if os.environ.get("SLURM_JOB_ID"):
            print(f"ERROR: GPU backend failed in SLURM job: {e}")
            raise
        print(f"GPU backend failed (expected on login nodes): {e}")
        all_devices = [jax.devices("cpu")[0]]

    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = []
    cpu_devices = jax.devices("cpu")

    print(f"All JAX devices: {all_devices}")
    print(f"GPU devices: {gpu_devices}")
    print(f"CPU devices: {cpu_devices}")

    if gpu_devices:
        print(f"GPU FOUND: {len(gpu_devices)} device(s)")
        try:
            test = jnp.sum(jax.device_put(jnp.array([1.0, 2.0, 3.0]), gpu_devices[0]) ** 2)
            test.block_until_ready()
            print("GPU computation test passed.")
            default = jax.devices()[0]
            if "gpu" in str(default).lower() or "cuda" in str(default).lower():
                print("GPU is default device.")
            else:
                jax.config.update("jax_platform_name", "gpu")
        except Exception as e:
            print(f"GPU test failed: {e}")
    else:
        print("No GPU; using CPU.")

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"


# -----------------------------------------------------------------------------
# Single-state simulation (same as original: noiseless path + noisy scan)
# -----------------------------------------------------------------------------


def run_simulation_for_one_state_jax(
    state_vector,
    key,
    transpiled_template,
    noise_conf,
    shots: int,
    with_noise: bool,
    delay_info=None,
    T1=None,
    T2=None,
):
    if not with_noise:
        @partial(jax.jit)
        def _shot(k):
            c = transpiled_template.copy()
            c.replace_inputs(state_vector)
            return c.sample(allow_state=True, random_generator=k)[0]

        keys = jax.random.split(key, shots)
        return jax.vmap(_shot)(keys)

    qir = transpiled_template.to_qir()
    num_status_params = len(qir)
    nq = int(np.log2(state_vector.shape[0]))

    def scan_body(carry, _):
        carry_key, state_vec = carry
        carry_key, meas_key, noise_status_key = jax.random.split(carry_key, 3)
        status_for_shot = (
            jax.random.uniform(
                noise_status_key,
                shape=(100 * num_status_params,),
                minval=0.0,
                maxval=1.0,
            )
            if num_status_params > 0
            else jnp.zeros((0,), dtype=jnp.float32)
        )
        c2 = tc.Circuit(nq, inputs=state_vec)
        for gate_info in qir:
            gate_name = gate_info.get("name", "").lower()
            qubits = gate_info.get("index", [])
            params = gate_info.get("parameters", {})
            if hasattr(c2, gate_name):
                gate_method = getattr(c2, gate_name)
                if params:
                    if "theta" in params:
                        gate_method(qubits[0], theta=params["theta"])
                    else:
                        gate_method(*qubits, **params)
                else:
                    gate_method(*qubits)
        c2 = tc.noisemodel.circuit_with_noise(c2, noise_conf, status=status_for_shot)
        c2.replace_inputs(state_vec)
        sample_result = c2.sample(allow_state=True, random_generator=meas_key)
        sample = (
            sample_result[0]
            if isinstance(sample_result, (list, tuple))
            else sample_result
        )
        return (carry_key, state_vec), jnp.asarray(sample)

    state_dim = state_vector.shape[0]
    estimated_nq = int(np.log2(state_dim)) if state_dim > 0 else 15
    chunk_size = 1000

    if shots <= chunk_size:
        _, all_samples = jax.lax.scan(scan_body, (key, state_vector), None, length=shots)
        return all_samples

    num_chunks = (shots + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(key, num_chunks + 1)[:-1]
    chunks = []
    for idx in range(num_chunks):
        start = idx * chunk_size
        end = min(start + chunk_size, shots)
        length = end - start
        k = chunk_keys[idx]
        _, seg = jax.lax.scan(scan_body, (k, state_vector), None, length=length)
        chunks.append(seg)
    return jnp.concatenate(chunks, axis=0)


def run_batched_simulation_jax(
    transpiled_template,
    batch_states,
    key,
    noise_conf,
    shots,
    with_noise,
    delay_info=None,
    T1=None,
    T2=None,
):
    batch_size = batch_states.shape[0]
    keys = jax.random.split(key, batch_size)
    vmap_fn = jax.vmap(
        run_simulation_for_one_state_jax,
        in_axes=(0, 0, None, None, None, None, None, None, None),
    )
    return vmap_fn(
        batch_states, keys, transpiled_template, noise_conf, shots, with_noise,
        delay_info, T1, T2,
    )


# -----------------------------------------------------------------------------
# Trajectory sampling and simulation (no importance sampling)
# -----------------------------------------------------------------------------


def sample_and_simulate_trajectories_batch(
    key,
    f_vec,
    nq,
    Ks,
    transpiled_template,
    noise_conf,
    num_samples,
    shots_per_state,
    alpha,
    mapping,
    delay_info,
    T1,
    T2,
):
    """Sample a batch of trajectories and simulate to get mean accuracy."""
    from shadow_mcs_jitted import generate_psi_F_vector, trajectory_from_key_jit

    psi0 = generate_psi_F_vector(f_vec, nq)
    keys = jax.random.split(key, num_samples)

    def one(k):
        return trajectory_from_key_jit(k, psi0, nq, Ks)

    states_batch, _, _ = jax.vmap(one)(keys)
    sim_key = jax.random.fold_in(key, int(T1) % 10000)
    sim_keys = jax.random.split(sim_key, num_samples)

    def sim_one(state, k):
        results = run_simulation_for_one_state_jax(
            state, k, transpiled_template, noise_conf,
            shots_per_state, True, delay_info, T1, T2,
        )
        results_mapped = results[:, mapping]
        return qrmc_opt.process_results_to_accuracy(
            results_mapped,
            jnp.arange(nq, dtype=jnp.int32),
            f_vec,
            alpha,
            nq,
            nq,
        )

    accs = jax.vmap(sim_one)(states_batch, sim_keys)
    return jnp.mean(accs)


def sample_and_simulate_trajectories_grouped(
    key,
    f_vec,
    nq,
    Ks,
    transpiled_template,
    noise_conf,
    total_samples: int,
    chunk_size: int,
    shots_per_state: int,
    alpha,
    mapping,
    delay_info,
    T1,
    T2,
) -> float:
    """Chunked trajectory sampling, group by unique state, simulate unique only; same logic as original."""
    from shadow_mcs_jitted import (
        generate_psi_F_vector,
        trajectory_from_key_jit,
        group_trajectories_jax,
    )

    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    eprint(
        f"[PROGRESS] Starting trajectory simulation: {total_samples} trajectories "
        f"in {num_chunks} chunk(s) of {chunk_size} samples each"
    )
    label_start = time.time()
    chunk_times = []
    psi0 = generate_psi_F_vector(f_vec, nq)
    weighted_sum = 0.0
    current_key = key

    for chunk_idx in range(num_chunks):
        chunk_start = time.time()
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_samples)
        chunk_samples = end - start

        if chunk_times:
            avg_chunk = sum(chunk_times) / len(chunk_times)
            est_rem = avg_chunk * (num_chunks - chunk_idx)
            eprint(
                f"[PROGRESS] Chunk {chunk_idx+1}/{num_chunks} | "
                f"Elapsed: {(time.time()-label_start)/60:.1f}min | "
                f"Est. remaining: {est_rem/60:.1f}min | Avg chunk: {avg_chunk:.1f}s"
            )
        else:
            eprint(
                f"[PROGRESS] Starting chunk {chunk_idx+1}/{num_chunks} | "
                f"Elapsed: {(time.time()-label_start)/60:.1f}min"
            )

        if chunk_idx > 0:
            gc.collect()
            time.sleep(0.5 if nq >= 20 else 0.2)

        chunk_key_base = jax.random.fold_in(current_key, chunk_idx)
        chunk_keys = jax.random.split(chunk_key_base, chunk_samples)
        current_key = jax.random.fold_in(current_key, (chunk_idx + 1) * 1000)

        def one(k):
            return trajectory_from_key_jit(k, psi0, nq, Ks)

        states_chunk, errs_chunk, codes_chunk = jax.vmap(one)(chunk_keys)

        if nq >= 20 and chunk_size > 150:
            states_chunk_np = np.array(states_chunk)
            errs_chunk_np = np.array(errs_chunk)
            codes_chunk_np = np.array(codes_chunk)
            del states_chunk, errs_chunk, codes_chunk
            jax.clear_backends()
            gc.collect()
            states_chunk = jnp.array(states_chunk_np)
            errs_chunk = jnp.array(errs_chunk_np)
            codes_chunk = jnp.array(codes_chunk_np)
            del states_chunk_np, errs_chunk_np, codes_chunk_np

        grouped = group_trajectories_jax(states_chunk, errs_chunk, codes_chunk)
        is_group_start_np = np.array(grouped["is_group_start"])
        states_sorted_np = np.array(grouped["states_sorted"])
        counts_at_starts_np = np.array(grouped["counts_at_starts"])
        unique_states = states_sorted_np[is_group_start_np]
        multiplicities = counts_at_starts_np[is_group_start_np]
        num_unique = len(unique_states)

        eprint(
            f"[PROGRESS] Chunk {chunk_idx+1}/{num_chunks}: Found {num_unique} unique states "
            f"(out of {chunk_samples} samples) in {time.time()-chunk_start:.1f}s"
        )
        del states_chunk, errs_chunk, codes_chunk, grouped

        if num_unique > 0:
            unique_batch_size = get_unique_batch_size(nq, num_unique)
            sim_key = jax.random.fold_in(key, chunk_idx * 10000 + int(T1) % 10000)
            chunk_accuracies = []
            chunk_multiplicities = []

            def simulate_one(state, sim_k):
                results = run_simulation_for_one_state_jax(
                    state, sim_k, transpiled_template, noise_conf,
                    shots_per_state, True, delay_info, T1, T2,
                )
                results_mapped = results[:, mapping]
                return qrmc_opt.process_results_to_accuracy(
                    results_mapped,
                    jnp.arange(nq, dtype=jnp.int32),
                    f_vec,
                    alpha,
                    nq,
                    nq,
                )

            for batch_start in range(0, num_unique, unique_batch_size):
                batch_end = min(batch_start + unique_batch_size, num_unique)
                batch_states = unique_states[batch_start:batch_end]
                batch_multiplicities = multiplicities[batch_start:batch_end]
                batch_size = batch_end - batch_start
                batch_keys = jax.random.split(sim_key, batch_size + 1)
                sim_key = batch_keys[0]
                batch_sim_keys = batch_keys[1:]

                if nq >= 20 and unique_batch_size == 1:
                    batch_accuracies_list = []
                    for i in range(batch_size):
                        state_jax = jnp.array(batch_states[i])
                        acc = simulate_one(state_jax, batch_sim_keys[i])
                        batch_accuracies_list.append(float(acc))
                        del state_jax, acc
                        if i % 5 == 0:
                            gc.collect()
                    batch_accuracies_np = np.array(batch_accuracies_list)
                else:
                    batch_accuracies = jax.vmap(simulate_one)(
                        jnp.array(batch_states),
                        batch_sim_keys,
                    )
                    batch_accuracies_np = np.array(batch_accuracies)

                chunk_accuracies.append(batch_accuracies_np)
                chunk_multiplicities.append(batch_multiplicities)
                if nq >= 20 and unique_batch_size == 1:
                    del batch_states
                else:
                    del batch_states, batch_accuracies
                if nq >= 18:
                    gc.collect()
                    if nq >= 20:
                        time.sleep(0.3)

            unique_accuracies_np = np.concatenate(chunk_accuracies)
            multiplicities_np = np.concatenate(chunk_multiplicities)
            chunk_weighted_sum = np.sum(unique_accuracies_np * multiplicities_np)
            weighted_sum += chunk_weighted_sum
            del unique_states, unique_accuracies_np, multiplicities_np
            del chunk_accuracies, chunk_multiplicities

        if nq >= 16:
            if chunk_idx % 3 == 0:
                t0 = time.time()
                eprint(f"[PROGRESS] Clearing JIT cache (chunk {chunk_idx+1}, every 3rd chunk)...")
                jax.clear_backends()
                eprint(f"[PROGRESS] JIT cache cleared in {(time.time()-t0)/60:.1f}min")
            gc.collect()
            if nq >= 20:
                time.sleep(0.5)

        chunk_times.append(time.time() - chunk_start)
        progress_pct = (chunk_idx + 1) / num_chunks * 100
        eprint(
            f"[PROGRESS] Chunk {chunk_idx+1}/{num_chunks} completed in {chunk_times[-1]:.1f}s | "
            f"Total: {(time.time()-label_start)/60:.1f}min | Progress: {progress_pct:.1f}%"
        )

    average_accuracy = weighted_sum / total_samples
    eprint(
        f"[PROGRESS] All {num_chunks} chunks completed! "
        f"Total trajectory simulation time: {(time.time()-label_start)/60:.1f}min | "
        f"Average accuracy: {average_accuracy:.6f}"
    )
    return float(average_accuracy)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device_name = args.device
    channel_type = args.channel
    nq = args.nq
    f_start = args.f_start
    nfk = args.nfk
    total_nf = args.total_nf
    f_seed = args.f_seed
    job_index = args.job_index
    job_timestamp = args.job_timestamp
    alpha_pattern_opt = args.alpha_pattern
    amp_override = args.amp

    nqs_to_run = [nq]
    channels_to_run = [channel_type]

    print(
        f"Processing f states: {f_start} to {f_start + nfk - 1} (total: {total_nf})"
    )
    print(f"Job index: k={job_index}")
    print("Running WITHOUT importance sampling — sampling directly at target probability")

    setup_jax_and_devices()
    try:
        jax.clear_backends()
        eprint("Cleared JAX compilation cache")
    except Exception:
        pass

    eprint("=" * 80)
    eprint(
        "Quantum Monte Carlo Simulation - Single Device/Channel (NO IMPORTANCE SAMPLING)"
    )
    eprint(f"Device: {device_name}, Channel: {channel_type}, nq: {nq}")
    eprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    eprint(f"Job Index: {job_index}, f_start: {f_start}, nfk: {nfk}, total_nf: {total_nf}")
    eprint("=" * 80)

    script_start = time.time()
    main_key = jax.random.PRNGKey(42)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = job_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    job_output_dir = OUTPUT_BASE_DIR / run_timestamp
    job_output_dir.mkdir(parents=True, exist_ok=True)

    acc_optimized = np.zeros((1, 1, len(nqs_to_run), len(channels_to_run), nfk))
    alphas_optimized = np.zeros((1, 1, len(nqs_to_run), len(channels_to_run), nfk))
    metadata_records = []

    for n, nq_val in enumerate(nqs_to_run):
        nq_start = time.time()
        eprint(f"\n{'='*80}")
        eprint(f"Processing nq={nq_val} ({n+1}/{len(nqs_to_run)})")
        eprint(f"{'='*80}")

        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = get_memory_fraction(nq_val)
        if nq_val >= 20:
            print(f"Memory fraction set to 0.65 for nq={nq_val}")

        dev_config = DEVICE_CONFIGS[device_name]
        tidle = dev_config["Tidle"]
        t1_val = tidle
        t2_val = int(tidle * T2_RATIO)
        noise_conf_name = dev_config["noise_conf"]
        conf, f1q, f2q = NOISE_CONFIGS[noise_conf_name]
        delay_gate_configs = dev_config["delay_gate_configs"]
        conf_with_delays = add_thermal_relaxation_to_noise_conf(
            conf, tidle, delay_gate_configs, dev_config["idle_error"]
        )
        print(dev_config["idle_error"])
        connectivity = get_connectivity_for_device(device_name, nq_val)
        dev = qd.QuantumDeviceSimulator(
            nq_val,
            QISKIT_BASIS_GATES,
            connectivity,
            conf_with_delays,
            readout_error,
            T1=t1_val,
            T2=t2_val,
            delay_gate_configs=delay_gate_configs,
            Tidle=tidle,
        )

        if alpha_pattern_opt is not None:
            alpha_pattern_array = pick_alpha_for_pattern(alpha_pattern_opt, nq_val)
            alpha_pattern_label = alpha_pattern_opt
        else:
            if device_name in ("I", "R", "T", "S"):
                raise ValueError(
                    f"Device {device_name} requires --alpha_pattern. "
                    "Choose from: nq/4, nq/2, 3/4nq, nq"
                )
            alpha_pattern_array = pick_alpha_for_device(device_name, nq_val)
            alpha_pattern_label = "device_default"

        device_amp = amp_override if amp_override is not None else dev_config["amp"]
        alpha_safe = alpha_pattern_label.replace("/", "_")
        amp_str = f"_amp{amp_override}" if amp_override is not None else ""
        run_output_root = (
            job_output_dir
            / f"{device_name}_{channel_type}_nq{nq_val}_k{job_index}_{alpha_safe}{amp_str}"
        )
        run_output_root.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {run_output_root}")

        def write_metadata(records: list, actual_amp: float | None = None) -> None:
            amp = actual_amp if actual_amp is not None else DEVICE_CONFIGS[device_name]["amp"]
            device_amps = [amp]
            device_t1s = [DEVICE_CONFIGS[device_name]["Tidle"]]
            metadata = {
                "generated_at": datetime.now().isoformat(),
                "run_timestamp": run_timestamp,
                "job_timestamp": run_timestamp,
                "device": device_name,
                "channel": channel_type,
                "available_nqs": list(nqs_to_run),
                "simulation_parameters": {
                    "total_nf": total_nf,
                    "f_start": f_start,
                    "nfk": nfk,
                    "f_seed": f_seed,
                    "job_index": job_index,
                    "NUM_MC_SAMPLES_BASE": NUM_MC_SAMPLES_BASE,
                    "SHOTS_PER_STATE_BASE": SHOTS_PER_STATE_BASE,
                    "note": "Parameters are adaptive based on nq (reduced for nq>=19)",
                    "method": "no_importance_sampling",
                    "device_configs": {
                        device_name: {
                            "T1": float(DEVICE_CONFIGS[device_name]["Tidle"]),
                            "noise_conf": DEVICE_CONFIGS[device_name]["noise_conf"],
                            "amp": float(amp),
                            "amp_source": (
                                "override"
                                if (actual_amp is not None and actual_amp != DEVICE_CONFIGS[device_name]["amp"])
                                else "device_config"
                            ),
                        }
                    },
                },
                "amplitudes": device_amps,
                "channels": list(channels_to_run),
                "T1_values": [int(t) for t in device_t1s],
                "T2_ratio": float(T2_RATIO),
                "T2_values": [int(t * T2_RATIO) for t in device_t1s],
                "noise_configs": {
                    "noise_confI": {"f1q": DEVICE_CONFIGS["I"]["f1q"], "f2q": DEVICE_CONFIGS["I"]["f2q"]},
                    "noise_confR": {"f1q": DEVICE_CONFIGS["R"]["f1q"], "f2q": DEVICE_CONFIGS["R"]["f2q"]},
                    "noise_confT": {"f1q": DEVICE_CONFIGS["T"]["f1q"], "f2q": DEVICE_CONFIGS["T"]["f2q"]},
                    "noise_confS": {"f1q": DEVICE_CONFIGS["S"]["f1q"], "f2q": DEVICE_CONFIGS["S"]["f2q"]},
                },
                "basis_gates": QISKIT_BASIS_GATES,
                "readout_error": readout_error,
                "connectivity": DEVICE_CONFIGS[device_name]["connectivity"],
                "records": records,
            }
            with (run_output_root / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        write_metadata([], actual_amp=device_amp)

        dev_info = {
            "device": dev,
            "global_idx": 0,
            "t1": t1_val,
            "t2": t2_val,
            "amp": device_amp,
            "alpha_pattern": alpha_pattern_array,
            "alpha_sum": int(np.sum(alpha_pattern_array)),
            "connectivity": dev_config["connectivity"],
        }

        np.random.seed(f_seed)
        F_matrix_full = np.random.randint(2, size=(total_nf, 2**nq_val))
        F_matrix = F_matrix_full[f_start : f_start + nfk]
        print(
            f"Generated F matrix: shape {F_matrix.shape} "
            f"(indices {f_start} to {f_start + nfk - 1} of {total_nf})"
        )

        amp_source = "override" if amp_override is not None else "device_config"
        print(
            f"\n--- Device {device_name}: T1={dev_info['t1']:.0f}, amp={dev_info['amp']} "
            f"(from {amp_source}), alpha_pattern={alpha_pattern_label}, "
            f"alpha_sum={dev_info['alpha_sum']} ---"
        )
        eprint(f"[DEBUG] Device {device_name}: Using amplitude {dev_info['amp']} (source: {amp_source})")

        templates, map_data, _, _, _, template_cnots, delay_info = qd.precompute_transpiled_templates(
            nq_val, dev_info["device"]
        )
        if dev_info["alpha_sum"] in template_cnots:
            gi = template_cnots[dev_info["alpha_sum"]]
            print(f"\n--- Transpilation Gate Counts (alpha_sum={dev_info['alpha_sum']}) ---")
            print(f"  Two-qubit gates (CNOT): Ideal: {gi['ideal_cnots']}, Transpiled: {gi['transpiled_cnots']}, Overhead: {gi['cnot_overhead']:.2f}x")
            print(f"  Single-qubit: Ideal: {gi.get('ideal_single_qubit', 'N/A')}, Transpiled: {gi.get('transpiled_single_qubit', 'N/A')}")
            if gi.get("ideal_single_qubit", 0) > 0:
                print(f"    Overhead: {gi['transpiled_single_qubit']/gi['ideal_single_qubit']:.2f}x")
            print(f"  H gates: Ideal: {gi['ideal_h']}, Transpiled: {gi['transpiled_h']}")

        mappings_jax = {
            k: jnp.array(v if v is not None else np.arange(nq_val), dtype=jnp.int32)
            for k, v in map_data.items()
        }
        connectivity_label = dev_config["connectivity"]
        connectivity_edges = connectivity
        device_list = [
            {
                "index": 0,
                "name": device_name,
                "nq": nq_val,
                "T1": int(dev_info["t1"]),
                "T2": int(dev_info["t2"]),
                "f1q": float(f1q),
                "f2q": float(f2q),
                "amp": float(dev_info["amp"]),
                "alpha_sum": int(dev_info["alpha_sum"]),
                "basis_gates": QISKIT_BASIS_GATES,
                "connectivity_label": connectivity_label,
                "connectivity_edges": connectivity_edges,
                "noise_model": "NoiseConf",
                "readout_error": readout_error,
            }
        ]
        template_cnots_serializable = {}
        if template_cnots:
            for hw, info in template_cnots.items():
                template_cnots_serializable[str(hw)] = {
                    "ideal_cnots": int(info.get("ideal_cnots", 0)),
                    "transpiled_cnots": int(info.get("transpiled_cnots", 0)),
                    "ideal_single_qubit": int(info.get("ideal_single_qubit", 0)),
                    "transpiled_single_qubit": int(info.get("transpiled_single_qubit", 0)),
                    "ideal_h": int(info.get("ideal_h", 0)),
                    "transpiled_h": int(info.get("transpiled_h", 0)),
                    "cnot_overhead": float(info.get("cnot_overhead", float("inf"))),
                }
        nq_record = {
            "nq": nq_val,
            "nq_index": n,
            "job_index": job_index,
            "f_start": f_start,
            "f_end": f_start + nfk - 1,
            "nfk": nfk,
            "total_nf": total_nf,
            "acc_file": f"acc_optimized_nq{nq_val}_f{f_start}to{f_start+nfk-1}.npy",
            "alpha_file": f"alphas_optimized_nq{nq_val}_f{f_start}to{f_start+nfk-1}.npy",
            "acc_slice_shape": list(acc_optimized[:, :, n, :, :].shape),
            "alpha_slice_shape": list(alphas_optimized[:, :, n, :, :].shape),
            "devices": device_list,
            "template_cnots": template_cnots_serializable,
        }

        target_alpha_sum = dev_info["alpha_sum"]
        alpha_pattern = dev_info["alpha_pattern"]
        device_amp = dev_info["amp"]

        for nch, CHANNEL_TYPE in enumerate(channels_to_run):
            channel_start = time.time()
            eprint(f"\n{'='*80}")
            eprint(f"[PROGRESS] Device {device_name}, Channel: {CHANNEL_TYPE} ({nch+1}/{len(channels_to_run)})")
            eprint(f"{'='*80}")

            Ks = qrmc_opt.get_kraus_operators_with_aliases(
                {"type": CHANNEL_TYPE, "strength": device_amp}
            )

            for label in range(nfk):
                label_start = time.time()
                eprint(f"\n[PROGRESS] Processing label {label} ({label+1}/{nfk}) for nq={nq_val}, channel={CHANNEL_TYPE}")

                if nq_val >= 20:
                    gc.collect()
                    time.sleep(1.5 if label > 0 else 1.0)
                elif nq_val >= 16:
                    gc.collect()
                    time.sleep(2.0 if (nq_val == 16 and label > 0) else 1.5)

                f_vec = jnp.array(F_matrix[label], dtype=jnp.int32)
                num_mc, shots_per_state, trajectory_chunk_size = get_adaptive_run_params(nq_val)
                eprint(
                    f"[PROGRESS] Label {label}: Parameters - {num_mc} MC samples, "
                    f"chunk_size={trajectory_chunk_size}, shots={shots_per_state}"
                )

                try:
                    main_key, base_trajectory_key = jax.random.split(main_key)
                    f_vec_np = np.array(f_vec)
                    global_f_idx = f_start + label
                    f_vec_hash = int(
                        np.sum(f_vec_np) * 1000
                        + global_f_idx * 100
                        + np.sum(f_vec_np[: min(16, len(f_vec_np))]) % 1000
                    )
                    trajectory_key = jax.random.fold_in(
                        base_trajectory_key,
                        nch * 10000 + global_f_idx * 100 + f_vec_hash,
                    )
                    eprint(f"[DEBUG] About to simulate with channel={CHANNEL_TYPE}, Ks shape={Ks.shape}, label={label}")
                    Ks_np = np.array(Ks)
                    eprint(f"[DEBUG] Ks hash verification: {hashlib.md5(str(Ks_np).encode()).hexdigest()}")
                    eprint(f"[DEBUG] === Starting simulation ===")
                    eprint(f"[DEBUG] Channel type: {CHANNEL_TYPE}, Ks shape: {Ks.shape}, Device amp: {device_amp}")
                    eprint(f"[DEBUG] nq: {nq_val}, label: {label}, f_idx: {f_start + label}")

                    final_acc = sample_and_simulate_trajectories_grouped(
                        trajectory_key,
                        f_vec,
                        nq_val,
                        Ks,
                        templates[target_alpha_sum],
                        dev_info["device"].properties["noise_model"],
                        num_mc,
                        trajectory_chunk_size,
                        shots_per_state,
                        jnp.array(alpha_pattern, dtype=jnp.int32),
                        mappings_jax[target_alpha_sum],
                        delay_info.get(target_alpha_sum),
                        dev_info["t1"],
                        dev_info["t2"],
                    )
                except Exception as e:
                    eprint(f"ERROR: Failed to process label {label}: {e}")
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    raise

                acc_optimized[0, 0, n, nch, label] = final_acc
                alphas_optimized[0, 0, n, nch, label] = int(
                    "".join(map(str, alpha_pattern)), 2
                )
                label_time = time.time() - label_start
                eprint(
                    f"[PROGRESS] Label {label} (f_idx={f_start + label}) completed in {label_time/60:.1f}min | "
                    f"Accuracy: {final_acc:.6f}"
                )
                if nq_val >= 20:
                    gc.collect()
                    time.sleep(1.0)
                elif nq_val >= 16:
                    gc.collect()
                    time.sleep(1.5 if nq_val == 16 else 1.0)

        acc_filename = f"acc_optimized_nq{nq_val}_f{f_start}to{f_start+nfk-1}.npy"
        alpha_filename = f"alphas_optimized_nq{nq_val}_f{f_start}to{f_start+nfk-1}.npy"
        np.save(run_output_root / acc_filename, acc_optimized[:, :, n, :, :])
        np.save(run_output_root / alpha_filename, alphas_optimized[:, :, n, :, :])
        f_indices_info = {
            "device": device_name,
            "channel": channel_type,
            "nq": int(nq_val),
            "job_index": int(job_index),
            "f_start": int(f_start),
            "f_end": int(f_start + nfk - 1),
            "nfk": int(nfk),
            "total_nf": int(total_nf),
            "f_seed": int(f_seed),
            "acc_file": acc_filename,
            "alpha_file": alpha_filename,
            "method": "no_importance_sampling",
        }
        with (run_output_root / f"f_indices_f{f_start}to{f_start+nfk-1}.json").open("w") as f:
            json.dump(f_indices_info, f, indent=2)
        metadata_records.append(nq_record)
        write_metadata(metadata_records, actual_amp=device_amp)
        np.save(
            run_output_root / f"acc_optimized_final_f{f_start}to{f_start+nfk-1}.npy",
            acc_optimized,
        )
        np.save(
            run_output_root / f"alphas_optimized_final_f{f_start}to{f_start+nfk-1}.npy",
            alphas_optimized,
        )

        eprint(
            f"\n[PROGRESS] Channel {channel_type} completed for nq={nq_val} in "
            f"{(time.time()-channel_start)/60:.1f}min"
        )
        eprint(
            f"[PROGRESS] nq={nq_val} processing completed in {(time.time()-nq_start)/60:.1f}min. "
            f"Results saved to: {run_output_root}"
        )

    total_time = time.time() - script_start
    eprint(f"\n{'='*80}")
    eprint(f"SIMULATION COMPLETE! Processed f states {f_start} to {f_start + nfk - 1}")
    eprint(f"Total execution time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    eprint(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    eprint(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        eprint(f"\n{'='*80}")
        eprint("ERROR: Simulation failed with exception:")
        eprint(f"{'='*80}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        eprint(f"{'='*80}")
        sys.exit(1)
