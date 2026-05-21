# quantum_simulation

Monte Carlo (MC) and density-matrix (DM) quantum trajectory simulations for the *Noisy Learning Advantage* project.

---

## Directory layout

```
quantum_simulation/
├── quantum_run_cluster_single_without_IS_readable.py  # Main MC runner (cluster / CLI)
├── quantum_run_mc_optimized.py                        # Top-level MC wrapper (used by notebooks)
├── quantum_run_dm_verification.py                     # DM verification runner (CLI)
├── shadow_mcs_jitted.py                               # JAX-jitted shadow MCS (imports shadow_funcs_dm)
├── shadow_funcs_dm.py                                 # Density-matrix shadow functions
├── run_dm_verification_jobs.ipynb                     # Notebook to launch / inspect DM jobs
├── cluster_results/                                   # Output from cluster runs (git-ignored)
├── dm_verification_logs/                              # Stderr/stdout from DM jobs (git-ignored)
└── modules/
    ├── quantum_device_sim.py      # Core qubit + circuit simulation
    ├── quantum_run_mc_optimized.py # Inner MC loop (imported by top-level wrapper)
    ├── device_config.py           # Device A / S / R / T parameter sets
    ├── device_config2.py          # Extended device configurations
    ├── channel_backend.py         # Kraus operator builders
    ├── channel_sampler.py         # Noise-channel sampling utilities
    └── noisy_sim.py               # Noisy circuit helpers
```

> **Note:** `shadow_mcs_jitted.py` falls back to `../shadows_simulation/shadow_funcs_dm.py`
> if the local copy is absent. Keep `shadows_simulation/` as a sibling of this folder inside `code/`.

---

## Running the MC simulation

From the **repo root** (or any directory):

```bash
python3 code/quantum_simulation/quantum_run_cluster_single_without_IS_readable.py \
    --device S --channel relaxation \
    --nq 8 --nfk 10 --total_nf 10 \
    --alpha_pattern nq/2
```

### Key CLI arguments

| Argument | Options | Description |
|---|---|---|
| `--device` | `I, R, T, S` | Device noise profile |
| `--channel` | `relaxation, dephasing, depolarizing` | Noise channel |
| `--nq` | integer | Number of qubits |
| `--alpha_pattern` | `nq/4, nq/2, 3/4nq, nq` | Shadow measurement weight pattern |
| `--nfk` | integer | Number of f-values per batch |
| `--total_nf` | integer | Total number of f-values |

---

## Running DM verification

```bash
python3 code/quantum_simulation/quantum_run_dm_verification.py \
    --device S --channel relaxation --nq 6
```

---

## Outputs

Results land in `cluster_results/<TIMESTAMP>/<run_label>/`:

| File | Content |
|---|---|
| `acc_optimized_final_f0to<N>.npy` | Final accuracy array |
| `alphas_optimized_final_f0to<N>.npy` | Optimal alpha weights |
| `metadata.json` | Full run configuration |
| `f_indices_f0to<N>.json` | f-index slice info |

---

## TODO

- [ ] **Refactor file names and locations** — consolidate `shadow_funcs_dm.py` (currently duplicated between `quantum_simulation/` and `shadows_simulation/`); unify into a single canonical location.
- [ ] **Upload data to Zenodo** — `cluster_results/` and `dm_verification_logs/` to be deposited as a Zenodo dataset (DOI TBD).
- [ ] **Add hypergraph code** — integrate hypergraph shadow-protocol implementation currently living outside this repo.
