# Noisy Learning Advantage

> ⚠️ **Work in Progress** — This repository is under active development. Structure, APIs, and results may change without notice.

This repository contains simulations and analysis investigating the learning advantage of quantum protocols under realistic noise conditions.

---

## Repository Structure

```
noisy-learning-advantage-3/
│
├── data/                        # All raw and processed data
│   ├── fig4_clean_m_vm_grids_nq*.json   # Pre-computed VM grids for figures
│   ├── old/                     # Archived older data
│   ├── curve_fitting/           # Curve fitting scripts and outputs
│   ├── ml_data/                 # Machine learning data (with/without HPO)
│   └── paper_data_2/            # Hypergraph, ML and shadow surrogate data
│
├── code/                        # All simulation and analysis code
│   ├── quantum_simulation/      # Main MC and DM quantum trajectory simulations
│   │   ├── mc_quantum_sim.py        # Monte Carlo trajectory simulation (CLI)
│   │   ├── quantum_run_dm_verification.py  # Density matrix verification (CLI)
│   │   ├── dm_quantum_sim.ipynb     # DM simulation notebook
│   │   ├── quantum_run_mc_optimized.py     # Top-level MC wrapper
│   │   └── modules/             # Helper modules (device config, noise, etc.)
│   └── shadows_simulation/      # JAX-jitted classical shadow utilities
│       ├── shadow_mcs_jitted.py     # Core trajectory sampling + grouping (JAX)
│       └── shadow_funcs_dm.py       # Density-matrix shadow computation
│
└── manuscript/                  # Manuscript figures and notebooks
    ├── figures_notebooks/       # Jupyter notebooks that generate all figures
    └── figures_manuscript/      # Final rendered figures for the paper
```

---

## Quick Start

### Monte Carlo Quantum Simulation

Run from **any directory**:
```bash
python3 code/quantum_simulation/mc_quantum_sim.py \
    --device S --channel relaxation \
    --nq 8 --nfk 10 --total_nf 10 \
    --alpha_pattern nq/2
```

Key arguments:

| Argument | Options | Description |
|---|---|---|
| `--device` | `I, R, T, S` | Device type |
| `--channel` | `relaxation, dephasing, depolarizing` | Noise channel |
| `--nq` | integer | Number of qubits |
| `--alpha_pattern` | `nq/4, nq/2, 3/4nq, nq` | Shadow probing pattern |

### Density Matrix Verification

```bash
python3 code/quantum_simulation/quantum_run_dm_verification.py \
    --device S --channel relaxation --nq 6
```

---

## Key Dependencies

- **JAX** — GPU-accelerated trajectory sampling and JIT compilation
- **TensorCircuit** — Quantum circuit simulation and shadow tomography
- **NumPy / SciPy** — Data processing and curve fitting

---

## Notes

- `shadows_simulation/` must remain a sibling of `quantum_simulation/` inside `code/` — the import paths are relative to this layout.
- Outputs are written to `cluster_results/` inside the working directory by default.
- For cluster runs, see the shell scripts in `code/quantum_simulation/`.
