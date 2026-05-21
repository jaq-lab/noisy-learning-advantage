"""
Test functions for white noise model verification.
These will be used in the verify_white_noise_model.ipynb notebook.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add paths for shadow generation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../quantum_simulation'))
from shadow_mcs_jitted import mcs_shadows_streaming_jit, generate_psi_F_vector, get_kraus_operators

from classical_nn_run_light import (
    apply_decoherence_noise,
    hamming_distance,
    get_relevant_input,
    generate_white_noise_training_at_checkpoints_vectorized,
    compute_variance_matrix
)


def test_noise_models():
    """Test relaxation and dephasing noise models."""
    
    # Test parameters
    n_qubits = 4
    p = 0.1  # Noise strength
    
    # Generate test density matrix elements
    test_cases = [
        (0, 0, "Diagonal, |0⟩ state"),
        (1, 1, "Diagonal, |1⟩ state"),
        (0, 1, "Off-diagonal, distance 1"),
        (0, 3, "Off-diagonal, distance 2"),
        (0, 7, "Off-diagonal, distance 3"),
        (3, 7, "Off-diagonal, distance 1"),
        (0, 15, "Off-diagonal, distance 4 (max)"),
    ]
    
    # Initial value (for pure state, all off-diagonals have same magnitude)
    rho_0 = 1.0 / (2 ** n_qubits)
    
    print("=" * 80)
    print("NOISE MODEL VERIFICATION")
    print("=" * 80)
    print(f"Noise strength p = {p}")
    print(f"n_qubits = {n_qubits}")
    print()
    
    results = []
    
    for n, m, description in test_cases:
        rho_original = rho_0
        
        # Compute Hamming distance
        d_nm = hamming_distance(n, m)
        
        # Compute Hamming weights
        d_n = bin(n).count('1')
        d_m = bin(m).count('1')
        
        # Apply dephasing
        rho_dephasing = apply_decoherence_noise(rho_original, n, m, p, noise_type="dephasing")
        dephasing_factor = (1 - p) ** d_nm
        
        # Apply relaxation
        rho_relaxation = apply_decoherence_noise(rho_original, n, m, p, noise_type="relaxation")
        relaxation_factor = (1 - p) ** ((d_n + d_m) / 2)
        
        results.append({
            'n': n, 'm': m, 'description': description,
            'd_nm': d_nm, 'd_n': d_n, 'd_m': d_m,
            'original': rho_original,
            'dephasing': rho_dephasing, 'dephasing_factor': dephasing_factor,
            'relaxation': rho_relaxation, 'relaxation_factor': relaxation_factor
        })
        
        print(f"{description}:")
        print(f"  States: |{n:0{n_qubits}b}⟩, |{m:0{n_qubits}b}⟩")
        print(f"  Hamming distance d(n,m) = {d_nm}")
        print(f"  Hamming weights: d(n) = {d_n}, d(m) = {d_m}")
        print(f"  Original: {rho_original:.6f}")
        print(f"  Dephasing: {rho_dephasing:.6f} (factor: {dephasing_factor:.6f})")
        print(f"  Relaxation: {rho_relaxation:.6f} (factor: {relaxation_factor:.6f})")
        print()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Dephasing vs Hamming distance
    ax1 = axes[0]
    d_nm_vals = [r['d_nm'] for r in results]
    dephasing_vals = [r['dephasing'] for r in results]
    ax1.scatter(d_nm_vals, dephasing_vals, s=100, alpha=0.7, label='Dephasing')
    # Theoretical curve
    d_theory = np.linspace(0, max(d_nm_vals), 100)
    rho_theory = rho_0 * (1 - p) ** d_theory
    ax1.plot(d_theory, rho_theory, 'r--', alpha=0.5, label=f'Theoretical: (1-{p})^d')
    ax1.set_xlabel('Hamming Distance d(n,m)')
    ax1.set_ylabel('Noisy Element Value')
    ax1.set_title('Dephasing Model: (1-p)^(d(n,m))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relaxation vs average Hamming weight
    ax2 = axes[1]
    avg_weight_vals = [(r['d_n'] + r['d_m']) / 2 for r in results]
    relaxation_vals = [r['relaxation'] for r in results]
    ax2.scatter(avg_weight_vals, relaxation_vals, s=100, alpha=0.7, label='Relaxation', color='orange')
    # Theoretical curve
    w_theory = np.linspace(0, max(avg_weight_vals), 100)
    rho_theory_relax = rho_0 * (1 - p) ** w_theory
    ax2.plot(w_theory, rho_theory_relax, 'r--', alpha=0.5, label=f'Theoretical: (1-{p})^(d(n)+d(m))/2')
    ax2.set_xlabel('Average Hamming Weight (d(n)+d(m))/2')
    ax2.set_ylabel('Noisy Element Value')
    ax2.set_title('Relaxation Model: (1-p)^((d(n)+d(m))/2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_model_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Noise model verification complete!")
    print(f"✓ Saved figure to: noise_model_verification.png")
    
    return results


def test_white_noise_without_decoherence():
    """Test white noise model without decoherence.
    
    Generates:
    1. Variance plots by Hamming distance for nq=3 (d=0,1,2,3) and nq=4 (d=0,1,2,3,4)
    2. Density matrix visualization for 4-qubit case (2 rows: shadows and white noise,
       5 columns for k=1,1.5,2,2.5,3.0 corresponding to nps=2^(2k))
    """
    
    print("=" * 80)
    print("SHADOW DIFFUSION MODEL VERIFICATION (White Noise without Decoherence)")
    print("=" * 80)
    
    # Part 1: Variance vs nps, colored by Hamming distance for nq=3 and nq=4
    variance_results = {}
    
    # nps values to test
    nps_list = [10, 100, 200, 500, 1000,2000]
    
    for nq_idx, n_qubits in enumerate([4,5, 6]):
        
        num_samples = 10  # Reduced for efficiency
        num_repetitions = 10  # Reduced for efficiency
        
        # Generate F vectors
        np.random.seed(42)
        f_vecs = [np.random.randint(0, 2, size=2**n_qubits) for _ in range(num_samples)]
        
        # Channel config (no noise - use thermal with strength=0)
        channel_config = {"type": "thermal", "strength": 0.0, "thermal_p_exc": 0.0}
        Ks = get_kraus_operators(channel_config)
        
        max_d = n_qubits
        
        # Store variance for each (nps, distance) combination
        # Structure: shadow_vars[nps][d] = list of variances
        shadow_vars_by_nps_d = {nps: {d: [] for d in range(max_d + 1)} for nps in nps_list}
        white_vars_by_nps_d = {nps: {d: [] for d in range(max_d + 1)} for nps in nps_list}
        
        print(f"\nProcessing nq={n_qubits}...")
        print(f"Testing nps values: {nps_list}")
        
        for f_idx, f_vec in enumerate(f_vecs):
            # True density matrix
            psi_f = generate_psi_F_vector(jnp.array(f_vec), n_qubits)
            rho_true = np.array(jnp.outer(psi_f, jnp.conj(psi_f)))
            
            for nps in nps_list:
                # Generate shadow and white noise density matrices for this nps
                shadow_dms = []
                white_noise_dms = []
                
                for rep in range(num_repetitions):
                    # Shadow DM
                    rng_key = jax.random.PRNGKey(f_idx * len(nps_list) * num_repetitions + 
                                                nps_list.index(nps) * num_repetitions + rep)
                    batch_size_mcs = min(128, nps)
                    shadow_dm = np.array(mcs_shadows_streaming_jit(
                        rng_key, jnp.array(f_vec), n_qubits, Ks, nps, 
                        batch_size_mcs, r=1, shots=1, weights_kind=0, use_complex64=True
                    ))
                    shadow_dms.append(shadow_dm)
                    
                    # White noise DM
                    variance_matrix = compute_variance_matrix(n_qubits, nps)
                    rng_key_white = jax.random.PRNGKey(f_idx * len(nps_list) * num_repetitions + 
                                                       nps_list.index(nps) * num_repetitions + rep + 10000)
                    np.random.seed(f_idx * len(nps_list) * num_repetitions + 
                                 nps_list.index(nps) * num_repetitions + rep + 10000)
                    noise_re = np.random.normal(0, np.sqrt(variance_matrix))
                    noise_im = np.random.normal(0, np.sqrt(variance_matrix))
                    white_noise_dm = rho_true + noise_re
                    white_noise_dms.append(white_noise_dm)
                
                # Compute variance for each element by Hamming distance
                for i in range(2**n_qubits):
                    for j in range(2**n_qubits):
                        d_ij = hamming_distance(i, j)
                        
                        # Shadow variance
                        shadow_vals = [np.real(dm[i, j]) for dm in shadow_dms]
                        shadow_var = np.var(shadow_vals)
                        shadow_vars_by_nps_d[nps][d_ij].append(shadow_var)
                        
                        # White noise variance
                        white_vals = [np.real(dm[i, j]) for dm in white_noise_dms]
                        white_var = np.var(white_vals)
                        white_vars_by_nps_d[nps][d_ij].append(white_var)
            
            if (f_idx + 1) % 10 == 0:
                print(f"  Processed {f_idx + 1}/{num_samples} F vectors...")
        
        # Compute mean variances for each (nps, d) combination
        shadow_mean_vars = {nps: [np.mean(shadow_vars_by_nps_d[nps][d]) 
                                   for d in range(max_d + 1)] 
                            for nps in nps_list}
        white_mean_vars = {nps: [np.mean(white_vars_by_nps_d[nps][d]) 
                                 for d in range(max_d + 1)] 
                          for nps in nps_list}
        
        # Compute variance * nps for each individual element, then aggregate by distance
        # This allows us to check if all elements with same distance have same variance
        shadow_var_times_nps_mean = {}
        shadow_var_times_nps_std = {}
        shadow_var_times_nps_std_err = {}
        white_var_times_nps_mean = {}
        white_var_times_nps_std = {}
        white_var_times_nps_std_err = {}
        
        for nps in nps_list:
            shadow_var_times_nps_mean[nps] = []
            shadow_var_times_nps_std[nps] = []
            shadow_var_times_nps_std_err[nps] = []
            white_var_times_nps_mean[nps] = []
            white_var_times_nps_std[nps] = []
            white_var_times_nps_std_err[nps] = []
            
            for d in range(max_d + 1):
                # Get all variance * nps values for this distance
                shadow_vars_d = np.array(shadow_vars_by_nps_d[nps][d]) * nps
                white_vars_d = np.array(white_vars_by_nps_d[nps][d]) * nps
                
                # Compute statistics
                shadow_var_times_nps_mean[nps].append(np.mean(shadow_vars_d))
                shadow_var_times_nps_std[nps].append(np.std(shadow_vars_d))
                shadow_var_times_nps_std_err[nps].append(np.std(shadow_vars_d) / np.sqrt(len(shadow_vars_d)))
                
                white_var_times_nps_mean[nps].append(np.mean(white_vars_d))
                white_var_times_nps_std[nps].append(np.std(white_vars_d))
                white_var_times_nps_std_err[nps].append(np.std(white_vars_d) / np.sqrt(len(white_vars_d)))
        
        # Define colors for each Hamming distance (for notebook plotting)
        colors = plt.cm.viridis(np.linspace(0, 1, max_d + 1))
        
        # Store results (plotting moved to notebook)
        variance_results[n_qubits] = {
            'shadow': shadow_mean_vars,
            'white': white_mean_vars,
            'shadow_var_times_nps_mean': shadow_var_times_nps_mean,
            'shadow_var_times_nps_std': shadow_var_times_nps_std,
            'shadow_var_times_nps_std_err': shadow_var_times_nps_std_err,
            'white_var_times_nps_mean': white_var_times_nps_mean,
            'white_var_times_nps_std': white_var_times_nps_std,
            'white_var_times_nps_std_err': white_var_times_nps_std_err,
            'nps_list': nps_list,
            'colors': colors,
            'max_d': max_d
        }
    
    # Part 2: Density matrix visualization for nq=4
    print("\nGenerating density matrix visualization for nq=4...")
    n_qubits = 4
    
    # Channel config (no noise - use thermal with strength=0)
    channel_config = {"type": "thermal", "strength": 0.0, "thermal_p_exc": 0.0}
    Ks = get_kraus_operators(channel_config)
    
    # nps values corresponding to k = 1, 1.5, 2, 2.5, 3.0 (nps = 2^(k*nq))
    k_vals = [1.2, 1.5, 2, 2.5, 3]
    nps_list = [int(2**(k * n_qubits)) for k in k_vals]
    
    # Use a single F vector for visualization
    np.random.seed(42)
    f_vec = np.random.randint(0, 2, size=2**n_qubits)
    
    # True density matrix
    psi_f = generate_psi_F_vector(jnp.array(f_vec), n_qubits)
    rho_true = np.array(jnp.outer(psi_f, jnp.conj(psi_f)))
    
    # Generate shadow and white noise DMs for each nps
    shadow_dms_list = []
    white_noise_dms_list = []
    
    for nps_idx, nps in enumerate(nps_list):
        print(f"  Processing k={k_vals[nps_idx]:.1f}, nps={nps}...")
        
        # Shadow DM
        rng_key = jax.random.PRNGKey(42 + nps_idx)
        batch_size_mcs = min(128, nps)
        shadow_dm = np.array(mcs_shadows_streaming_jit(
            rng_key, jnp.array(f_vec), n_qubits, Ks, nps,
            batch_size_mcs, r=1, shots=1, weights_kind=0, use_complex64=True
        ))
        shadow_dms_list.append(shadow_dm)
        
        # White noise DM
        variance_matrix = compute_variance_matrix(n_qubits, nps)
        rng_key_white = jax.random.PRNGKey(42 + nps_idx + 10000)
        np.random.seed(42 + nps_idx + 10000)  # For reproducibility
        noise_re = np.random.normal(0, np.sqrt(variance_matrix))
        noise_im = np.random.normal(0, np.sqrt(variance_matrix))
        white_noise_dm = rho_true + noise_re + 1j * noise_im
        white_noise_dms_list.append(white_noise_dm)
    
    print("\n✓ White noise model verification complete!")
    return {
        'variance_data': variance_results,
        'density_matrices': {
            'shadow': shadow_dms_list,
            'white_noise': white_noise_dms_list,
            'true': rho_true,
            'k_vals': k_vals,
            'nps_list': nps_list
        }
    }


def test_combined_model():
    """Test white noise + decoherence combination."""
    
    n_qubits = 4
    num_samples = 100
    nps = 1000
    noise_strengths = [0.0, 0.05, 0.1, 0.2]
    noise_types = ['dephasing', 'relaxation']
    
    # Generate simple F vectors
    np.random.seed(42)
    f_vecs = [np.random.randint(0, 2, size=2**n_qubits) for _ in range(num_samples)]
    
    alpha_targ = tuple([1] * n_qubits)
    y_targ = '0' * n_qubits
    
    print("=" * 80)
    print("COMBINED MODEL VERIFICATION (White Noise + Decoherence)")
    print("=" * 80)
    print(f"n_qubits = {n_qubits}")
    print(f"nps = {nps}")
    print()
    
    results = {}
    
    for noise_type in noise_types:
        results[noise_type] = {}
        
        for p in noise_strengths:
            config = {
                'n': n_qubits,
                'alpha_targ': alpha_targ,
                'y_targ': y_targ,
                'white_noise_thermal_strength': p,
                'white_noise_decoherence_type': noise_type,
                'use_element_dependent_variance': True,
                'white_noise_seed': 42
            }
            
            data_dict = {'F_bs': f_vecs}
            
            training_features = generate_white_noise_training_at_checkpoints_vectorized(
                config, data_dict, [nps]
            )
            
            # Compute mean features
            features_array = np.array([training_features[i][nps] for i in range(num_samples)])
            mean_features = np.mean(features_array, axis=0)
            
            # Compute true features (with decoherence, no white noise)
            true_features_list = []
            for f_vec in f_vecs:
                # Construct density matrix
                rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
                for i in range(2**n_qubits):
                    for j in range(2**n_qubits):
                        sign = (-1) ** (f_vec[i] ^ f_vec[j])
                        rho_ij = sign / (2 ** n_qubits)
                        # Apply decoherence
                        if p > 0:
                            rho_ij = apply_decoherence_noise(rho_ij, i, j, p, noise_type=noise_type)
                        rho[i, j] = rho_ij
                
                features_true = get_relevant_input(jnp.array(rho), y_targ, alpha_targ, n_qubits)
                true_features_list.append(features_true)
            
            true_features_array = np.array(true_features_list)
            true_mean = np.mean(true_features_array, axis=0)
            
            results[noise_type][p] = {
                'mean_features': mean_features,
                'true_mean': true_mean,
                'error': np.mean(np.abs(mean_features - true_mean))
            }
            
            print(f"{noise_type.capitalize()}, p = {p:.2f}: Mean error = {results[noise_type][p]['error']:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx]
        
        ps = noise_strengths
        errors = [results[noise_type][p]['error'] for p in ps]
        
        ax.plot(ps, errors, 'o-', linewidth=2, markersize=8, label=f'{noise_type.capitalize()}')
        ax.set_xlabel('Noise Strength p')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'Combined Model Error ({noise_type.capitalize()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_model_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Combined model verification complete!")
    print(f"✓ Saved figure to: combined_model_verification.png")
    
    return results


def test_variance_constancy_within_weight_groups():
    """
    Test if variance is constant within Hamming weight groups.
    
    For products rho_{idx0, nk} * rho_{nk, idx1}, check if the variance across samples
    is the same for all nk values with the same Hamming weight, or if it varies significantly.
    
    Focus: Does variance vary across nk within the same weight group?
    """
    
    n_qubits = 6
    num_samples = 100
    k_vals = [1.5, 2.0]
    nps_list = [int(2**(k * n_qubits)) for k in k_vals]  # nps = [512, 4096] for nq=6
    
    # Channel config (no noise)
    channel_config = {"type": "thermal", "strength": 0.0, "thermal_p_exc": 0.0}
    Ks = get_kraus_operators(channel_config)
    batch_size_mcs = 128
    
    # Target indices (same as in test_averaging_justification)
    y_targ = "000000"  # |0⟩ state
    alpha_str = "000000"  # Identity
    y_targ_int = int(y_targ, 2)
    alpha_targ_int = int(alpha_str, 2)
    idx0 = y_targ_int ^ alpha_targ_int
    idx1 = y_targ_int
    
    print("=" * 80)
    print("VARIANCE CONSTANCY TEST (Within Hamming Weight Groups)")
    print("=" * 80)
    print(f"n_qubits = {n_qubits}")
    print(f"nps values: {nps_list} (k = {k_vals})")
    print(f"num_samples = {num_samples} (number of F vectors)")
    print(f"idx0 = {idx0} ({idx0:0{n_qubits}b}), Hamming weight = {hamming_weight(idx0)}")
    print(f"idx1 = {idx1} ({idx1:0{n_qubits}b}), Hamming weight = {hamming_weight(idx1)}")
    print()
    print("Question: Does variance of products rho(idx0,nk) × rho(nk,idx1) vary")
    print("          significantly across nk values within the same Hamming weight group?")
    print()
    
    # Generate pure state
    psi_0 = np.zeros(2**n_qubits, dtype=complex)
    psi_0[0] = 1.0
    rho_0 = np.outer(psi_0, psi_0.conj())
    
    # Generate F vectors (pure states)
    main_key = jax.random.PRNGKey(42)
    keys = jax.random.split(main_key, num_samples)
    fs = jax.vmap(lambda k: jax.random.uniform(k, shape=(2**n_qubits,)))(
        keys
    )
    fs = fs / jnp.linalg.norm(fs, axis=1, keepdims=True)
    
    # Group nk values by Hamming weight
    all_n = np.arange(2**n_qubits)
    mask = (all_n != idx0) & (all_n != idx1)
    n_vals = all_n[mask]
    grouped_by_weight = {}
    for nk in n_vals:
        w = hamming_weight(nk)
        if w not in grouped_by_weight:
            grouped_by_weight[w] = []
        grouped_by_weight[w].append(nk)
    
    # Generate shadow DMs for each nps
    shadow_dms_by_nps = {}
    for nps in nps_list:
        print(f"Generating shadow DMs for nps={nps}...")
        shadow_dms = []
        
        for i, f in enumerate(fs):
            if (i + 1) % 20 == 0:
                print(f"  Processing F vector {i+1}/{num_samples}")
            
            # Generate shadow measurement and reconstruct DM
            key = keys[i]
            shadow_dm = np.array(mcs_shadows_streaming_jit(
                key, jnp.array(f), n_qubits, Ks, nps,
                batch_size_mcs, r=1, shots=1, weights_kind=0, use_complex64=True
            ))
            shadow_dms.append(shadow_dm)
        
        shadow_dms_by_nps[nps] = shadow_dms
        print(f"  ✓ Completed nps={nps}\n")
    
    print("Computing variances...\n")
    print("Analysis: For each product V_k[i] = rho_{idx0, nk[i]} * rho_{nk[i], idx1}")
    print("  - Compute Var[V_k[i]] = variance of product across samples for each nk[i]")
    print("  - Compute Var[Var[V_k]] = variance of variances across nk values")
    print("  - Compute Var[Var[V_k]] / Mean[Var[V_k]] = CV of variances")
    print()
    
    # For each nps and each weight group, compute variance of products for each nk
    results_by_nps = {}
    
    for nps in nps_list:
        shadow_dms = shadow_dms_by_nps[nps]
        k_val = k_vals[nps_list.index(nps)]
        
        print(f"For nps={nps} (k={k_val:.1f}):")
        print("-" * 80)
        
        results_by_weight = {}
        
        for weight in sorted(grouped_by_weight.keys()):
            n_weight_list = grouped_by_weight[weight]
            
            # Compute products V_k[i] = rho_{idx0, nk[i]} * rho_{nk[i], idx1} for each shadow sample
            products_all_samples = []
            
            for shadow_dm in shadow_dms:
                products_sample = []
                for nk in n_weight_list:
                    # Product: V_k[i] = rho_{idx0, nk[i]} * rho_{nk[i], idx1}
                    rho_idx0_nk = shadow_dm[idx0, nk]
                    rho_nk_idx1 = shadow_dm[nk, idx1]
                    product = np.real(rho_idx0_nk) * np.real(rho_nk_idx1) * ((2 ** n_qubits) ** 2)
                    products_sample.append(product)
                products_all_samples.append(products_sample)
            
            products_all_samples = np.array(products_all_samples)  # Shape: (num_samples, len(n_weight_list))
            
            # For each nk[i], compute Var[V_k[i]] = variance across samples
            variances_per_nk = np.var(products_all_samples, axis=0)  # Var[V_k[i]] for each i
            means_per_nk = np.mean(products_all_samples, axis=0)    # Mean[V_k[i]] for each i
            
            # Statistics on variances
            mean_var_Vk = np.mean(variances_per_nk)      # Mean[Var[V_k]]
            var_var_Vk = np.var(variances_per_nk)        # Var[Var[V_k]]
            cv_var_Vk = var_var_Vk / (mean_var_Vk + 1e-10)  # Var[Var[V_k]] / Mean[Var[V_k]]
            
            results_by_weight[weight] = {
                'variances_per_nk': variances_per_nk,  # Var[V_k[i]] for each i
                'means_per_nk': means_per_nk,          # Mean[V_k[i]] for each i
                'mean_var_Vk': mean_var_Vk,            # Mean[Var[V_k]]
                'var_var_Vk': var_var_Vk,              # Var[Var[V_k]]
                'cv_var_Vk': cv_var_Vk,                # Var[Var[V_k]] / Mean[Var[V_k]]
                'n_weight_list': n_weight_list
            }
            
            print(f"  Hamming weight w = {weight} (n={len(n_weight_list)} nk values):")
            print(f"    Mean[Var[V_k]] = {mean_var_Vk:.6f} (average variance of products across nk)")
            print(f"    Var[Var[V_k]] = {var_var_Vk:.6f} (variance of variances)")
            print(f"    Var[Var[V_k]] / Mean[Var[V_k]] = {cv_var_Vk:.6f}")
            print(f"    → Lower ratio means variance is more constant across nk values")
            
            # Test significance
            if mean_var_Vk > 0:
                if cv_var_Vk < 0.1:
                    print(f"    → Variance is CONSTANT (ratio < 0.1: variance varies by <10%)")
                elif cv_var_Vk < 0.3:
                    print(f"    → Variance is RELATIVELY CONSTANT (ratio < 0.3: variance varies by <30%)")
                else:
                    print(f"    → Variance VARIES SIGNIFICANTLY (ratio >= 0.3: variance varies by >=30%)")
            
            # Show min/max variances for reference
            min_var = np.min(variances_per_nk)
            max_var = np.max(variances_per_nk)
            print(f"    Min Var[V_k[i]] = {min_var:.6f}, Max Var[V_k[i]] = {max_var:.6f}")
            print()
        
        results_by_nps[nps] = results_by_weight
    
    # Visualization: Plot variance distribution for each weight group
    num_weights = len(grouped_by_weight)
    cols = min(3, num_weights)
    rows = (num_weights + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if num_weights == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color scheme for different nps values
    nps_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(nps_list)))
    
    for idx, weight in enumerate(sorted(grouped_by_weight.keys())):
        ax = axes[idx]
        
        # Plot variance distributions for each nps, overlaid
        for nps_idx, nps in enumerate(nps_list):
            if weight in results_by_nps[nps]:
                data = results_by_nps[nps][weight]
                variances = data['variances_per_nk']  # Var[V_k[i]] for each i
                
                # Histogram of variances
                ax.hist(variances, bins=max(15, len(variances)//3), alpha=0.6, 
                       edgecolor='black', linewidth=0.5,
                       label=f'nps={nps} (k={k_vals[nps_idx]:.1f}), Var[Var]/Mean={data["cv_var_Vk"]:.4f}',
                       color=nps_colors[nps_idx], density=True)
                
                # Vertical line for mean variance
                ax.axvline(data['mean_var_Vk'], color=nps_colors[nps_idx], 
                          linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Variance of Products (across samples)')
        ax.set_ylabel('Density')
        ax.set_title(f'Hamming Weight w={weight}\n(Variance distribution across nk values)')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_weights, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Variance Distribution Across nk Values Within Each Weight Group\n' +
                 '(Narrower distributions → variance is more constant)', 
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('variance_constancy_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Variance constancy test complete!")
    print(f"✓ Saved figure to: variance_constancy_test.png")
    
    return results_by_nps


def test_averaging_justification():
    """
    Test if averaging within Hamming weight of nk makes sense.
    
    Tests if products rho_{idx0, nk} * rho_{nk, idx1} obtained from RECONSTRUCTED
    shadow density matrices can be averaged within groups of nk that have the same Hamming weight.
    """
    
    n_qubits = 6
    num_samples = 100  # Reduced for shadow generation
    # Test with different nps values: nps = 2^(k * nq) for k = 1.5 and 2.0, nq = 6
    k_vals = [1.5, 2.0]
    nps_list = [int(2**(k * n_qubits)) for k in k_vals]  # nps = [512, 4096] for nq=6
    
    # Channel config (no noise)
    channel_config = {"type": "thermal", "strength": 0.0, "thermal_p_exc": 0.0}
    Ks = get_kraus_operators(channel_config)
    batch_size_mcs = 128
    
    # Generate random F vectors
    np.random.seed(42)
    f_vecs = [np.random.randint(0, 2, size=2**n_qubits) for _ in range(num_samples)]
    
    alpha_targ = tuple([1] * n_qubits)
    y_targ = '0' * (n_qubits - 1) + '0'
    
    # Compute indices
    alpha_str = "".join(map(str, alpha_targ))
    y_targ_int = int(y_targ, 2)
    alpha_targ_int = int(alpha_str, 2)
    idx0 = y_targ_int ^ alpha_targ_int
    idx1 = y_targ_int
    
    print("=" * 80)
    print("AVERAGING JUSTIFICATION TEST (By Hamming Weight)")
    print("=" * 80)
    print(f"n_qubits = {n_qubits}")
    print(f"nps values: {nps_list} (k = {k_vals})")
    print(f"num_samples = {num_samples} (number of F vectors)")
    print(f"idx0 = {idx0} ({idx0:0{n_qubits}b}), Hamming weight = {hamming_weight(idx0)}")
    print(f"idx1 = {idx1} ({idx1:0{n_qubits}b}), Hamming weight = {hamming_weight(idx1)}")
    print()
    
    # Find all n values (exclude idx0 and idx1)
    all_n = np.arange(2**n_qubits)
    mask = (all_n != idx0) & (all_n != idx1)
    n_vals = all_n[mask]
    
    # Group n by Hamming weight
    grouped_by_weight = {}
    for n in n_vals:
        w_n = hamming_weight(n)
        if w_n not in grouped_by_weight:
            grouped_by_weight[w_n] = []
        grouped_by_weight[w_n].append(n)
    
    print(f"Found {len(grouped_by_weight)} Hamming weight groups")
    for w in sorted(grouped_by_weight.keys()):
        print(f"  Weight {w}: {len(grouped_by_weight[w])} values")
    print()
    
    print("Generating shadows and reconstructing density matrices for different nps...")
    print(f"  nps values: {nps_list} (k = {k_vals})")
    print()
    
    # Generate shadow DMs for each f_vec and each nps
    shadow_dms_by_nps = {}
    for nps in nps_list:
        print(f"Generating shadows for nps={nps}...")
        shadow_dms = []
        for f_idx, f_vec in enumerate(f_vecs):
            if (f_idx + 1) % 20 == 0:
                print(f"  Processing {f_idx + 1}/{num_samples}...")
            rng_key = jax.random.PRNGKey(42 + f_idx + nps * 1000)
            shadow_dm = np.array(mcs_shadows_streaming_jit(
                rng_key, jnp.array(f_vec), n_qubits, Ks, nps,
                batch_size_mcs, r=1, shots=1, weights_kind=0, use_complex64=True
            ))
            shadow_dms.append(shadow_dm)
        shadow_dms_by_nps[nps] = shadow_dms
        print(f"  ✓ Completed nps={nps}\n")
    
    print("Extracting products from reconstructed DMs...\n")
    
    # Compute products rho_{idx0, nk} * rho_{nk, idx1} from reconstructed shadow DMs for each nps
    product_data_by_nps = {}
    for nps in nps_list:
        shadow_dms = shadow_dms_by_nps[nps]
        product_data = {}
        
        for weight in sorted(grouped_by_weight.keys()):
            print(f"Processing weight {weight}...")
            n_weight_list = grouped_by_weight[weight]
            
            # Skip if empty
            if len(n_weight_list) == 0:
                print(f"WARNING: Weight {weight} has empty n_weight_list, skipping...")
                continue
            
            products_all_samples = []
            
            for shadow_dm in shadow_dms:
                products_sample = []
                for nk in n_weight_list:
                    # Extract rho_{idx0, nk} from reconstructed shadow DM
                    rho_idx0_nk = shadow_dm[idx0, nk]
                    
                    # Extract rho_{nk, idx1} from reconstructed shadow DM
                    rho_nk_idx1 = shadow_dm[nk, idx1]
                    
                    # Product (scaled by (2^n)^2 for normalization)
                    product = np.real(rho_idx0_nk) * np.real(rho_nk_idx1) * ((2 ** n_qubits) ** 2)
                    products_sample.append(product)
                
                # Ensure products_sample has the correct length
                if len(products_sample) != len(n_weight_list):
                    print(f"WARNING: products_sample length ({len(products_sample)}) != n_weight_list length ({len(n_weight_list)})")
                    print(f"  This might cause shape mismatch later")
                
                products_all_samples.append(products_sample)
            
      
            
            products_all_samples = np.array(products_all_samples)  # Shape: (num_samples, len(n_weight_list))
            

            std_total_nk = np.std(products_all_samples)
            stds_nk =np.std(products_all_samples, axis=0)
            std_of_std_nk = np.std(np.std(products_all_samples, axis=0))
            mean_of_std_nk = np.mean(np.std(products_all_samples, axis=0))
            std_of_mean_nk = np.std(np.mean(products_all_samples, axis=0))
            
          
            product_data[weight] = {
                'n_weight_list': n_weight_list,
                'std_nk': std_total_nk,
                'std_of_std_nk': std_of_std_nk,
                'mean_of_std_nk': mean_of_std_nk,
                'std_of_mean_nk': std_of_mean_nk,
                'stds': stds_nk
            }
        
        product_data_by_nps[nps] = product_data
        
        # Print statistics for this nps
        print(f"Statistics for nps={nps} (k={k_vals[nps_list.index(nps)]:.1f}):")
        for weight in sorted(product_data.keys()):
            data_w = product_data[weight]
            print(f"  Hamming weight w = {weight}:")
            print(f"    Number of nk values: {len(data_w['n_weight_list'])}")
            print(f"    Std of total products: {data_w['std_nk']:.6f}")
            print(f"    Std of std in nk values: {data_w['std_of_std_nk']:.6f}")
            print(f"    Mean of std in nk values: {data_w['mean_of_std_nk']:.6f}")
            print(f"    Std of mean in nk values: {data_w['std_of_mean_nk']:.6f}")
        print()
    
    # Visualization: Overlay histograms for different nps values
    # Interpretation: The histogram shows the distribution of mean product values across different nk 
    # values with the same Hamming weight. For averaging to be justified, this distribution should 
    # be narrow (low variance). As nps increases, shadow reconstruction becomes more accurate, so:
    # - Distribution becomes narrower (less variance across nk values)
    # - Mean becomes more accurate (closer to true value)
    
    num_weights = len(grouped_by_weight)
    cols = min(3, num_weights)
    rows = (num_weights + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if num_weights == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color scheme for different nps values
    nps_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(nps_list)))
    
    for idx, weight in enumerate(sorted(grouped_by_weight.keys())):
        ax = axes[idx]
        n_weight_list = grouped_by_weight[weight]
        
        # Plot histograms for each nps, overlaid
        for nps_idx, nps in enumerate(nps_list):
            if weight in product_data_by_nps[nps]:
                data_w = product_data_by_nps[nps][weight]
                data_to_plot = data_w['stds']
                
                # Use density=True for better comparison across different bin counts
                ax.hist(data_to_plot, bins=max(15, len(data_to_plot)//3), alpha=0.6, 
                       edgecolor='black', linewidth=0.5,
                       label=f'nps={nps} (k={k_vals[nps_idx]:.1f}), std={data_w["std_over_all"]:.4f}',
                       color=nps_colors[nps_idx], density=True)

        
        ax.set_xlabel('std of products in nk values')
        ax.set_ylabel('Density')
        ax.set_title(f'Hamming Weight w={weight}\n(n={len(n_weight_list)} nk values)')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    


def hamming_weight(n, n_qubits=None):
    """Compute Hamming weight (number of 1s in binary representation)."""
    return bin(n).count('1')


def test_dm_elements_scaling():
    """
    Verify white noise + decoherence by checking specific DM elements.
    
    Tests:
    - Diagonal elements: rho[m,m] with small/large Hamming weight
    - Off-diagonal elements: rho[m,n] with small/large Hamming distance and weight
    
    Verifies:
    - Mean scales correctly with decoherence strength p
    - Variance scales correctly with nps and p
    """
    n_qubits = 4
    nps_list = [100, 500, 1000, 2500]
    noise_strengths = [0.0, 0.05, 0.1, 0.15, 0.2]
    noise_types = ['dephasing', 'relaxation']
    num_samples = 500  # Number of samples to average over
    
    # Select test elements based on criteria
    def select_test_elements(n_qubits):
        """Select specific DM elements to test."""
        dim = 2 ** n_qubits
        elements = []
        
        # Diagonals: rho[m,m]
        # Small Hamming weight: |0000⟩, |0001⟩ (weight 0, 1)
        # Large Hamming weight: |1111⟩, |1110⟩ (weight n, n-1)
        small_weight_indices = [0, 1]  # |0000⟩, |0001⟩
        large_weight_indices = [dim-1, dim-2]  # |1111⟩, |1110⟩
        
        for m in small_weight_indices + large_weight_indices:
            elements.append({
                'type': 'diagonal',
                'i': m,
                'j': m,
                'hamming_weight': hamming_weight(m),
                'hamming_distance': 0,
                'description': f'ρ[{m},{m}] (|{m:0{n_qubits}b}⟩, w={hamming_weight(m)})'
            })
        
        # Off-diagonals: rho[m,n]
        # Small distance, small weight: |0000⟩, |0001⟩ (d=1, w=0,1)
        # Small distance, large weight: |1111⟩, |1110⟩ (d=1, w=4,3)
        # Large distance, small weight: |0000⟩, |1111⟩ (d=4, w=0,4)
        # Large distance, large weight: |0001⟩, |1110⟩ (d=3, w=1,3)
        
        off_diag_pairs = [
            (0, 1, 'small d, small w'),
            (dim-1, dim-2, 'small d, large w'),
            (0, dim-1, 'large d, small w'),
            (1, dim-2, 'large d, large w'),
        ]
        
        for m, n, desc in off_diag_pairs:
            d_nm = hamming_distance(m, n)
            w_m = hamming_weight(m)
            w_n = hamming_weight(n)
            elements.append({
                'type': 'off_diagonal',
                'i': m,
                'j': n,
                'hamming_distance': d_nm,
                'hamming_weight_m': w_m,
                'hamming_weight_n': w_n,
                'description': f'ρ[{m},{n}] (|{m:0{n_qubits}b}⟩,|{n:0{n_qubits}b}⟩, d={d_nm}, w_m={w_m}, w_n={w_n}, {desc})'
            })
        
        return elements
    
    test_elements = select_test_elements(n_qubits)
    
    print("=" * 80)
    print("DM ELEMENTS SCALING VERIFICATION")
    print("=" * 80)
    print(f"n_qubits = {n_qubits}")
    print(f"nps values: {nps_list}")
    print(f"Noise strengths: {noise_strengths}")
    print(f"Number of samples per test: {num_samples}")
    print(f"\nSelected test elements ({len(test_elements)}):")
    for elem in test_elements:
        print(f"  - {elem['description']}")
    print()
    
    # Generate a single F vector for testing
    np.random.seed(42)
    f_vec = np.random.randint(0, 2, size=2**n_qubits)
    
    # True density matrix (pure state)
    psi_f = generate_psi_F_vector(jnp.array(f_vec), n_qubits)
    rho_true = np.array(jnp.outer(psi_f, jnp.conj(psi_f)))
    
    results = {}
    
    for noise_type in noise_types:
        results[noise_type] = {}
        
        for p in noise_strengths:
            results[noise_type][p] = {}
            
            for nps in nps_list:
                print(f"Processing {noise_type}, p={p:.2f}, nps={nps}...")
                
                # Generate white noise + decoherence DMs
                element_values = {elem['description']: [] for elem in test_elements}
                
                for sample_idx in range(num_samples):
                    # Compute variance matrix
                    variance_matrix = compute_variance_matrix(n_qubits, nps)
                    
                    # Generate white noise
                    rng_seed = 42 + sample_idx * 10000 + int(p * 1000) + nps
                    np.random.seed(rng_seed)
                    noise_re = np.random.normal(0, np.sqrt(variance_matrix))
                    noise_im = np.random.normal(0, np.sqrt(variance_matrix))
                    
                    # Apply decoherence to true DM first
                    rho_noisy = np.zeros_like(rho_true)
                    for i in range(2**n_qubits):
                        for j in range(2**n_qubits):
                            rho_noisy[i, j] = apply_decoherence_noise(
                                rho_true[i, j], i, j, p, noise_type=noise_type
                            )
                    
                    # Add white noise
                    rho_with_noise = rho_noisy + noise_re + 1j * noise_im
                    
                    # Extract test elements
                    for elem in test_elements:
                        val = np.real(rho_with_noise[elem['i'], elem['j']])
                        element_values[elem['description']].append(val)
                
                # Compute statistics
                for elem in test_elements:
                    desc = elem['description']
                    values = np.array(element_values[desc])
                    mean_val = np.mean(values)
                    var_val = np.var(values)
                    
                    # Theoretical mean (with decoherence, no white noise)
                    true_val = rho_true[elem['i'], elem['j']]
                    if p > 0:
                        theoretical_mean = apply_decoherence_noise(
                            true_val, elem['i'], elem['j'], p, noise_type=noise_type
                        )
                    else:
                        theoretical_mean = true_val
                    theoretical_mean = np.real(theoretical_mean)
                    
                    # Theoretical variance
                    if elem['type'] == 'diagonal':
                        theoretical_var = 1.0 / nps
                    else:
                        d_nm = elem['hamming_distance']
                        theoretical_var = (2.3 ** (d_nm / 2)) / nps / 2
                    
                    results[noise_type][p][nps] = results[noise_type][p].get(nps, {})
                    results[noise_type][p][nps][desc] = {
                        'mean': mean_val,
                        'variance': var_val,
                        'theoretical_mean': theoretical_mean,
                        'theoretical_var': theoretical_var,
                        'mean_error': abs(mean_val - theoretical_mean),
                        'var_error': abs(var_val - theoretical_var),
                        'element': elem
                    }
    
    print("\n✓ DM elements scaling verification complete!")
    return results


if __name__ == "__main__":
    # Quick test
    test_noise_models()

