# Quantum Shadow Tomography Simulations

This directory contains quantum simulation pipelines for running shadow tomography with noisy quantum devices. Two approaches are provided:

1. **Monte Carlo (MC) Method** (`mc_quantum_sim.py`) - Fast trajectory-based simulation
2. **Density Matrix (DM) Method** (`dm_quantum_sim.ipynb` / `quantum_run_dm_verification.py`) - Exact verification on small systems

## Overview

### Monte Carlo (MC) Method
The simulation framework in `mc_quantum_sim.py`:
- **Samples quantum trajectories** at target probability from a parametrized quantum channel
- **Simulates noisy quantum circuits** using TensorCircuit with realistic noise models
- **Computes shadow tomography accuracy** using mean-field protocols
- **Supports multiple devices** (I, R, T, S) with different noise characteristics
- **Supports multiple noise channels** (relaxation, dephasing, depolarizing)
- **Adapts parameters** based on system size (nq) for memory efficiency
- **Runs on GPU/CPU** with automatic device detection
- **Fast and scalable** for large systems (nq ≥ 15)

### Density Matrix (DM) Method
The exact simulation framework in `dm_quantum_sim.ipynb` / `quantum_run_dm_verification.py`:
- **Full density matrix evolution** for exact quantum dynamics
- **Verification tool** for validating Monte Carlo results on small systems
- **Best for nq ≤ 8** due to exponential memory scaling (2^nq × 2^nq)
- **Higher accuracy** but slower than MC for larger systems
- **Ideal for understanding** convergence and noise effects

## File Structure

```
quantum_simulation/
├── mc_quantum_sim.py                              # Main MC trajectory script
├── dm_quantum_sim.ipynb                           # DM verification notebook (interactive)
├── quantum_run_dm_verification.py                 # DM verification script
├── README.md                                      # This file
├── modules/
│   ├── quantum_device_sim.py                      # Device simulator and transpilation
│   ├── channel_sampler.py                         # Channel noise sampling
│   ├── noisy_sim.py                               # Noisy circuit simulation
│   ├── quantum_run_mc_optimized.py                # MC trajectory processing
│   ├── device_config.py                           # Device configurations
│   ├── shadow_mcs_jitted.py                       # JAX-jitted shadow computation
│   └── ...
└── cluster_results/                               # Output directory (created at runtime)
```

## Quick Start

### Method 1: Monte Carlo (MC) - Fast & Scalable

Navigate to the quantum_simulation directory and run:

```bash
cd quantum_simulation
python mc_quantum_sim.py \
    --device S \
    --channel dephasing \
    --nq 15 \
    --nfk 1 \
    --total_nf 5 \
    --alpha_pattern nq/2
```

#### Test Run (Small System - nq=5)

To test the MC method quickly on a small system:

```bash
python mc_quantum_sim.py \
    --device S \
    --channel relaxation \
    --nq 5 \
    --nfk 1 \
    --total_nf 1 \
    --alpha_pattern nq/2
```

**Expected Runtime**: ~30 seconds to 2 minutes (on CPU/GPU)
**Output**: Results saved to `cluster_results/<timestamp>/S_relaxation_nq5_k0_nq_2/`

### Method 2: Density Matrix (DM) - Exact & Verified

For verification on small systems (nq ≤ 8), use the density matrix approach:

#### Option A: Interactive Jupyter Notebook

```bash
jupyter notebook dm_quantum_sim.ipynb
```

Then run cells sequentially to:
1. Define parameters
2. Setup device and noise models
3. Simulate with density matrix
4. Compare with MC results

#### Option B: Run as Python Script

```bash
python quantum_run_dm_verification.py \
    --device S \
    --channel relaxation \
    --nq 6 \
    --num_f_states 10
```

**Expected Runtime**: ~5-15 minutes for nq=6 (slower but exact)
**Output**: DM simulation results and comparison metrics

## Command-Line Arguments

### Required Arguments

| Argument | Type | Choices | Description |
|----------|------|---------|-------------|
| `--device` | str | `I`, `R`, `T`, `S` | Quantum device type |
| `--channel` | str | `relaxation`, `dephasing`, `depolarizing` | Noise channel type |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--nq` | int | 15 | Number of qubits |
| `--f_start` | int | 0 | Starting f state index |
| `--nfk` | int | 1 | Number of f states to process in this job |
| `--total_nf` | int | 5 | Total number of f states in the full run |
| `--f_seed` | int | 42 | Random seed for F matrix generation |
| `--job_index` | int | auto | Job index (auto-calculated from f_start/nfk) |
| `--job_timestamp` | str | auto | Output directory timestamp (auto-generated) |
| `--alpha_pattern` | str | device-based | Qubit pattern: `nq/4`, `nq/2`, `3/4nq`, `nq` |
| `--amp` | float | device-based | Amplitude override for noise strength |

## MC vs DM Comparison

| Aspect | Monte Carlo (`mc_quantum_sim.py`) | Density Matrix (`dm_quantum_sim.ipynb`) |
|--------|-----------------------------------|----------------------------------------|
| **Method** | Trajectory sampling | Full DM evolution |
| **Speed** | Fast | Slow (exponential in nq) |
| **Memory** | O(2^nq) per trajectory | O(4^nq) for full DM |
| **Accuracy** | Approx (converges with samples) | Exact |
| **Best for** | nq ≥ 10 | nq ≤ 8 (verification) |
| **Parallelizable** | Yes (many jobs) | Limited (density matrix size) |
| **Output** | Aggregated accuracy files | Detailed trajectory analysis |
| **Use Case** | Production runs | Validation & debugging |

### When to Use Each Method

- **Use MC** when: You need results for nq ≥ 10, want fast execution, or need to process many f states
- **Use DM** when: You want exact results for small systems, need to verify MC convergence, or debugging noise models

### Examples (MC)

#### Minimal Example
```bash
python mc_quantum_sim.py \
    --device S \
    --channel relaxation
```

#### With Custom Parameters
```bash
python mc_quantum_sim.py \
    --device T \
    --channel dephasing \
    --nq 18 \
    --nfk 5 \
    --total_nf 20 \
    --alpha_pattern nq/2 \
    --f_seed 12345
```

#### Distributed Processing (f state splitting)
Process f states 0-4 in job 1:
```bash
python mc_quantum_sim.py \
    --device I \
    --channel depolarizing \
    --nq 16 \
    --f_start 0 \
    --nfk 5 \
    --total_nf 20 \
    --job_index 0
```

Process f states 5-9 in job 2:
```bash
python mc_quantum_sim.py \
    --device I \
    --channel depolarizing \
    --nq 16 \
    --f_start 5 \
    --nfk 5 \
    --total_nf 20 \
    --job_index 1
```

## Device Types

| Device | Characteristics | Default Amp | Typical Use |
|--------|-----------------|-------------|------------|
| **I** | Ideal baseline | 0.01 | Reference |
| **R** | Relaxation-dominated | 0.01 | T1 focus |
| **T** | Thermal effects | 0.01 | Temperature dependence |
| **S** | Symmetric noise | 0.01 | Balanced studies |

## Alpha Patterns

The `--alpha_pattern` argument controls which qubits are included in the shadow tomography:

- **`nq/4`**: First 1/4 of qubits + last qubit = `|α| ≈ nq/4`
- **`nq/2`**: First 1/2 of qubits + last qubit = `|α| ≈ nq/2`
- **`3/4nq`**: First 3/4 of qubits + last qubit = `|α| ≈ 3*nq/4`
- **`nq`**: All qubits = `|α| = nq`

If not specified, device defaults are used (see `device_config.py`).

## Density Matrix (DM) Usage

### Running the DM Notebook Interactively

```bash
jupyter notebook dm_quantum_sim.ipynb
```

The notebook contains:
1. **Setup cells**: Device config, noise model initialization
2. **Small system test**: Run DM simulation for nq=4-6
3. **Comparison**: Plot MC vs DM accuracy for same parameters
4. **Analysis**: Convergence study and noise effect visualization

### Running DM as a Script

For batch processing without Jupyter:

```bash
python quantum_run_dm_verification.py \
    --device S \
    --channel relaxation \
    --nq 6 \
    --num_f_states 20
```

**Parameters**:
- `--device`: I, R, T, S
- `--channel`: relaxation, dephasing, depolarizing
- `--nq`: ≤ 8 recommended (2^nq × 2^nq matrix)
- `--num_f_states`: Number of random F states to average over

## Output

Results are saved to `cluster_results/<timestamp>/` with structure:

```
cluster_results/20250520_143022/
├── S_dephasing_nq18_k0_nq_2/
│   ├── metadata.json                          # Experiment metadata
│   ├── f_indices_f0to4.json                   # F state information
│   ├── acc_optimized_nq18_f0to4.npy           # Accuracy results
│   ├── alphas_optimized_nq18_f0to4.npy        # Alpha patterns used
│   ├── acc_optimized_final_f0to4.npy          # Final accuracy array
│   └── alphas_optimized_final_f0to4.npy       # Final alpha array
```

### Output Files

- **`metadata.json`**: Complete simulation configuration, device specs, noise parameters
- **`acc_optimized_*.npy`**: Accuracy array, shape `(1, 1, num_nq, num_channels, nfk)`
- **`alphas_optimized_*.npy`**: Binary representation of alpha patterns used
- **`f_indices_*.json`**: F state range and file references for this job

## Performance Considerations

### Memory Usage by Number of Qubits

| nq | Trajectory Chunk | Shots/State | Memory Fraction | Typical RAM |
|----|-----------------|-------------|-----------------|------------|
| 5 | 1000 | 1000 | 0.75 | ~4 GB |
| 15-16 | 1000 | 1000 | 0.75 | ~16 GB |
| 18 | 500 | 1000 | 0.70 | ~32 GB |
| 19 | 250 | 1000 | 0.70 | ~48 GB |
| 20+ | 10-145 | 250-500 | 0.65 | ~64 GB+ |

### Runtime Estimates

- **nq=5**: ~30 sec - 2 min per label
- **nq=15**: ~5-10 min per label
- **nq=18**: ~30-60 min per label
- **nq=19+**: ~2+ hours per label

## Running on HPC with SLURM

To run on an HPC cluster with SLURM job scheduler, create a simple job script:

### Example SLURM Job Script

Create a file `submit_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=quantum-sim
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /path/to/quantum_simulation

python quantum_run_cluster_single_without_IS_readable.py \
    --device S \
    --channel dephasing \
    --nq 18 \
    --nfk 5 \
    --total_nf 20 \
    --alpha_pattern nq/2
```

Submit with:
```bash
sbatch submit_job.sh
```

### Parallel Processing (Distributed f States)

For processing many f states in parallel, submit multiple jobs with different f ranges:

```bash
# Job 1: f_start=0-4
sbatch -J "quantum-sim-j0" submit_job.sh --f_start 0 --nfk 5 --total_nf 100

# Job 2: f_start=5-9
sbatch -J "quantum-sim-j1" submit_job.sh --f_start 5 --nfk 5 --total_nf 100

# Job 3: f_start=10-14
sbatch -J "quantum-sim-j2" submit_job.sh --f_start 10 --nfk 5 --total_nf 100
```

## Dependencies

### Python Packages
- `jax`, `jaxlib` (GPU acceleration)
- `tensorcircuit` (quantum simulation)
- `numpy`
- `qiskit` (transpilation)
- `scipy` (scientific computing)

### System Requirements
- Python 3.8+
- For GPU: CUDA 11.8+ compatible GPU
- For CPU-only: 16+ GB RAM (higher for nq > 18)

### Installation

```bash
# Create environment
conda create -n quantum-sim python=3.10

# Activate
conda activate quantum-sim

# Install dependencies (adjust for your HPC cluster)
pip install jax[cuda11] tensorcircuit qiskit numpy scipy
```

## Debugging and Monitoring

### Monitor a Running Job

```bash
# On SLURM cluster
squeue -u $USER  # See all your jobs
tail -f slurm-JOBID.out  # View stdout in real-time
tail -f slurm-JOBID.err  # View stderr with debug output
```

### Check Output

```bash
# List all results
ls -lah cluster_results/

# View metadata from a run
cat cluster_results/20250520_143022/S_dephasing_nq18_k0_nq_2/metadata.json | jq .

# Load results in Python
import numpy as np
acc = np.load('cluster_results/20250520_143022/S_dephasing_nq18_k0_nq_2/acc_optimized_nq18_f0to4.npy')
print(f"Accuracy shape: {acc.shape}")
```

### Common Issues

1. **GPU Memory Error**: Reduce `--nfk` or use smaller `--nq`
2. **JAX Compilation Hangs**: Check GPU drivers with `nvidia-smi`
3. **Module Import Error**: Ensure `modules/` subdirectory exists and has required files
4. **Timeout on SLURM**: Increase `--time` in shell scripts for larger systems

## Advanced Usage

### Parallel Processing of Large F Matrix

For a large number of f states (e.g., 100 total), split across multiple job submissions:

```bash
# Job 1: f_start=0, nfk=25
python quantum_run_cluster_single_without_IS_readable.py \
    --device S --channel dephasing --nq 18 \
    --f_start 0 --nfk 25 --total_nf 100

# Job 2: f_start=25, nfk=25
python quantum_run_cluster_single_without_IS_readable.py \
    --device S --channel dephasing --nq 18 \
    --f_start 25 --nfk 25 --total_nf 100

# Job 3: f_start=50, nfk=25
python quantum_run_cluster_single_without_IS_readable.py \
    --device S --channel dephasing --nq 18 \
    --f_start 50 --nfk 25 --total_nf 100

# Job 4: f_start=75, nfk=25
python quantum_run_cluster_single_without_IS_readable.py \
    --device S --channel dephasing --nq 18 \
    --f_start 75 --nfk 25 --total_nf 100
```

Each job outputs separate files that can be combined in post-processing.

### Custom Noise Amplitude

Override device defaults:

```bash
python quantum_run_cluster_single_without_IS_readable.py \
    --device S \
    --channel dephasing \
    --nq 18 \
    --amp 0.05 \
    --alpha_pattern nq/2
```

## References

- Device configurations: `modules/device_config.py`
- Trajectory sampling: `modules/shadow_mcs_jitted.py`
- Circuit simulation: `modules/quantum_device_sim.py`
- Main MC routines: `modules/quantum_run_mc_optimized.py`

---

**Last Updated**: May 2026
