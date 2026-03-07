# Sensitivity Report
- device: I, readout_error: 0%
- amps: ['0.01', '0.05', '0.1']
- etas: [0.01, 0.02, 0.05]
- deltas: [3, 10, 15, 30]
- curve-bootstrap fit default: wls
- curve-bootstrap surfaces available: True
- Anchor policy: `last_meaningful`, `obs_max`, and `obs_max + delta`.
- Primary extrapolation anchor is `delta=3` (aligned with forward-CV max horizon=3). Larger deltas are stress tests.

## Variant Summary
| variant | n_total | ok | untrusted_extrap | random_baseline | gated_extrap_rate | rb_rate | CI hw med/p90 | HG<ML flip vs baseline | CI-hw Δ vs baseline med/p90/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | nan | nan / nan / nan |
| interval=cv | 486 | 322 | 28 | 128 | 0.0712 | 0.263 | 0.403 / 1.54 | 0 | 0.0877 / 0.597 / 1.69 |
| interval=conformal_hybrid | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| cv_thr=5-95 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.485 / 1.98 | 0 | 0.0364 / 0.471 / 1.48 |
| cv_thr=20-80 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.387 / 1.65 | 0 | 0.0542 / 0.653 / 2.93 |
| monotone=on | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| monotone=off | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 8.98e-05 |
| random_baseline=off | 486 | 337 | 0 | 0 | 0 | 0 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| random_eps0=0.01 | 486 | 337 | 0 | 114 | 0 | 0.235 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| random_eps0=0.02 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| extrap_min_h=2 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| extrap_min_h=3 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| extrap_min_h=4 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| extrap_rmse_ratio=2.0 | 486 | 292 | 45 | 128 | 0.115 | 0.263 | 0.474 / 1.91 | 0 | 0 / 0 / 0 |
| extrap_rmse_ratio=3.0 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| conformal_holdout=1 | 486 | 341 | 0 | 128 | 0 | 0.263 | 0.522 / 1.57 | 0 | 0.0565 / 0.421 / 4.93 |
| conformal_holdout=2 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| conformal_holdout=3 | 486 | 307 | 15 | 128 | 0.0382 | 0.263 | 0.4 / 1.81 | 0 | 0.074 / 0.375 / 3.34 |
| conformal_min_abs_errors=20 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0 |
| conformal_min_abs_errors=50 | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.473 / 1.9 | 0 | 0 / 0 / 0.144 |
| curve_fit=wls | 486 | 337 | 0 | 128 | 0 | 0.263 | 0.242 / 1.68 | 0 | 0.21 / 1.49 / 3.9 |
| curve_fit=ols | 486 | 337 | 0 | 128 | 0 | 0.263 | 1.08 / 2.27 | 0 | 0.769 / 2.05 / 3.64 |

## Two-Seed Baseline Robustness
- Seeds compared: 12345 vs 12346
- Common finite points: 337
- |Δlog2(nps)| median/p90/max: 0.00957 / 0.0495 / 0.724
- |ΔCI_hw(log2)| median/p90/max: 0.0046 / 0.0283 / 0.287
- HG<ML flip rate vs primary baseline: 0 (0/10)

## Bootstrap Sampling Variants
| variant | n_total | ok | HG<ML flip vs baseline | |Δlog2(nps)| median/p90/max | |ΔCI_hw(log2)| med/p90/max |
|---|---:|---:|---:|---:|---:|
| seed_plus_1 | 486 | 337 | 0 | 0.00957 / 0.0495 / 0.724 | 0.0046 / 0.0283 / 0.287 |
| hg_eig_mode_per_k | 486 | 337 | 0 | 0.00392 / 0.0581 / 0.525 | 0.00162 / 0.0155 / 0.0904 |
| hg_eig_mode_cluster | 486 | 337 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| nboot=400 | 486 | 337 | 0 | 0.0187 / 0.119 / 0.823 | 0.00429 / 0.0402 / 0.135 |
| nboot=4000 | 486 | 337 | 0 | 0.00552 / 0.0407 / 0.277 | 0.00301 / 0.0135 / 0.112 |
