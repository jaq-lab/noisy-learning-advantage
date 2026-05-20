#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as e:  # pragma: no cover
    plt = None
    MPL_IMPORT_ERROR = e

SCRIPT_DIR = Path(__file__).resolve().parent

# Requested layout
CHANNEL_ORDER = ["dephasing", "depolarizing", "relaxation"]
PLOT_AMPLITUDES = ["0.01", "0.05", "0.1"]
CHANNEL_TITLES = {
    "dephasing": "Dephasing",
    "depolarizing": "Depolarizing",
    "relaxation": "Relaxation",
}

# Requested raw ranges (no extrapolation)
HG_NQ_CANDIDATES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]  # user added 4..8, exclude 14 absent; 13 will be excluded from plot
ML_NQ_MIN = 5
ML_NQ_MAX = 12
HG_EXCLUDE_NQ_PLOT = {4, 13}  # user asked to start plotting from 5 and exclude 13

METHOD_LABELS = {
    "hypergraph": "Hypergraph",
    "ml": "ML (LR + features)",
}
METHOD_LINESTYLES = {
    "hypergraph": "-",
    "ml": "--",
}
METHOD_MARKERS = {
    "hypergraph": "o",
    "ml": "s",
}
METHOD_LINEWIDTHS = {
    "hypergraph": 2.0,
    "ml": 1.9,
}

# Hypergraph constants (from analyze_6 + extended names)
N_DRAWS = 2
QUBIT_NAME = {
    4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
    9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 15: 'fifteen'
}
K_OLD = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0], dtype=float)
K_OLD_NO13 = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0], dtype=float)
K_CPU1 = np.array([1.3, 1.4, 1.5, 1.6], dtype=float)
K_CPU2 = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=float)
PATCH_TOKEN_K = {
    '1p0': np.array([1.0], dtype=float),
    '1p0_1p1_1p2': np.array([1.0, 1.1, 1.2], dtype=float),
    '1p3': np.array([1.3], dtype=float),
    '1p7_1p8': np.array([1.7, 1.8], dtype=float),
    '1p9_2p0_2p1': np.array([1.9, 2.0, 2.1], dtype=float),
}
PATCHES_15 = {
    ('dephasing', 0.01): ['1p0_1p1_1p2'],
    ('dephasing', 0.05): ['1p0_1p1_1p2'],
    ('dephasing', 0.1): ['1p0_1p1_1p2'],
    ('depolarizing', 0.01): ['1p0'],
    ('depolarizing', 0.05): [],
    ('depolarizing', 0.1): [],
    ('relaxation', 0.01): ['1p0_1p1_1p2'],
    ('relaxation', 0.05): [],
    ('relaxation', 0.1): ['1p7_1p8', '1p9_2p0_2p1'],
}
L_BASELINE = 0.5

FILE_CASE_RE = re.compile(r"_(dephasing|depolarizing|relaxation)_([0-9.]+)_(f_benchmarks|f_targs|algo_bench)\.npy$")


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _find_hypergraph_merged(start: Optional[str] = None) -> Path:
    p = Path(start).resolve() if start else Path.cwd().resolve()
    for d in [p, *p.parents]:
        for c in [d / 'shadow-qml-analyze' / 'paper_data_2' / 'hypergraph_merged', d / 'paper_data_2' / 'hypergraph_merged', d / 'hypergraph_merged']:
            if (c / '4').exists() or (c / '9').exists():
                return c
    fallback = Path('/home/ypatel/data1/shadow-qml-analyze/paper_data_2/hypergraph_merged')
    if fallback.exists():
        return fallback
    raise FileNotFoundError('Cannot locate hypergraph_merged directory')


def _alpha_int(nq: int) -> int:
    bits = [0] * int(nq)
    for i in range(int(nq) // 2, int(nq)):
        bits[i] = 1
    a = 0
    for b in bits:
        a = (a << 1) | b
    return a


def compute_alpha_acc(f_bench: np.ndarray, f_targs: np.ndarray, nq: int) -> np.ndarray:
    ai = _alpha_int(int(nq))
    y = np.arange(2 ** int(nq))
    joint = y ^ ai
    pred = f_bench[0]
    od = f_targs[joint, :] ^ f_targs[y, :]
    pd = pred[joint, :, :, :] ^ pred[y, :, :, :]
    od_bc = od[:, np.newaxis, :, np.newaxis]
    return (pd == od_bc).astype(np.float32).mean(axis=0)  # (n_k, n_hg, n_draw)


def _infer_hg_k_grid(nq: int, noise: str, n_k: int) -> Optional[np.ndarray]:
    nq = int(nq)
    n_k = int(n_k)
    if n_k == len(K_OLD):
        return K_OLD.copy()
    if nq in (11, 12, 13) and noise in ('dephasing', 'relaxation') and n_k == len(K_OLD_NO13):
        return K_OLD_NO13.copy()
    if nq == 15 and noise in ('dephasing', 'relaxation') and n_k == len(K_CPU1):
        return K_CPU1.copy()
    if nq == 15 and noise == 'depolarizing' and n_k == len(K_CPU2):
        return K_CPU2.copy()
    return None


def _match_case_file(path: Path, noise: str, gamma: float, suffix_kind: str) -> bool:
    m = FILE_CASE_RE.search(path.name)
    if not m:
        return False
    noise_s, gamma_s, kind_s = m.groups()
    if kind_s != suffix_kind or noise_s != noise:
        return False
    try:
        return abs(float(gamma_s) - float(gamma)) < 1e-12
    except Exception:
        return False


def _scan_case_pair_files(case_dir: Path, noise: str, gamma: float) -> List[Tuple[Path, Path]]:
    pairs = []
    for fb in sorted(case_dir.glob('*_f_benchmarks.npy')):
        if not _match_case_file(fb, noise, gamma, 'f_benchmarks'):
            continue
        ft = fb.with_name(fb.name.replace('_f_benchmarks.npy', '_f_targs.npy'))
        if ft.exists() and _match_case_file(ft, noise, gamma, 'f_targs'):
            pairs.append((fb, ft))
    return pairs


def _scan_case_algo_files(case_dir: Path, noise: str, gamma: float) -> List[Path]:
    out = []
    for ab in sorted(case_dir.glob('*_algo_bench.npy')):
        if _match_case_file(ab, noise, gamma, 'algo_bench'):
            out.append(ab)
    return out


def _add_k_trials(bucket: Dict[float, List[np.ndarray]], k_vals: np.ndarray, rows_2d: np.ndarray):
    k_arr = np.asarray(k_vals, dtype=float)
    rr = np.asarray(rows_2d, dtype=float)
    for i, kv in enumerate(k_arr):
        kr = round(float(kv), 4)
        bucket.setdefault(kr, []).append(np.asarray(rr[i], dtype=float).ravel())


def load_hypergraph_acc_trials(hm_dir: Path, n_q: int, noise: str, gamma: float) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load/merge raw Hypergraph alpha-accuracy trajectories for one case.

    Supports both old (single base file + patches) and new (multiple batch files) layouts.
    Returns acc_trials (n_k_total, n_trials_total_varlen_padded), k_grid sorted, and metadata.
    """
    nq = int(n_q)
    gamma = float(gamma)
    base_dir = hm_dir / str(nq)
    if not base_dir.exists():
        raise FileNotFoundError(f'Hypergraph n_q directory missing: {base_dir}')

    k_bucket: Dict[float, List[np.ndarray]] = {}
    sources: List[str] = []
    anomalies: List[str] = []

    # 1) Use all available f_benchmarks/f_targs pairs in base dir (new 4..8 data may have multiple batches)
    pairs = _scan_case_pair_files(base_dir, noise, gamma)
    for fb_path, ft_path in pairs:
        fb = np.array(np.load(str(fb_path), mmap_mode='r'))
        ft = np.array(np.load(str(ft_path), mmap_mode='r'))
        if fb.ndim < 5 or ft.ndim < 2:
            anomalies.append(f'invalid pair shape: {fb_path.name} {fb.shape} / {ft.shape}')
            continue
        n_hg = int(ft.shape[1])
        n_k = int(fb.shape[2]) if fb.shape[0] >= 1 else 0
        k_vals = _infer_hg_k_grid(nq, noise, n_k)
        if k_vals is None or len(k_vals) != n_k:
            anomalies.append(f'cannot infer k-grid for pair {fb_path.name} (n_k={n_k})')
            continue
        acc = compute_alpha_acc(fb, ft, nq)  # (n_k, n_hg, n_draw)
        _add_k_trials(k_bucket, k_vals, acc.reshape(n_k, n_hg * acc.shape[2]))
        sources.append(f'base_pair:{fb_path.name}:n_hg={n_hg}')
        if n_hg != 48:
            anomalies.append(f'{fb_path.name} n_hg={n_hg} (expected 48 for many runs)')

    # 2) If no pair data in base dir, fallback to algo_bench[2] base files (raw simulation output, but less direct)
    if not pairs:
        algo_files = _scan_case_algo_files(base_dir, noise, gamma)
        for ab_path in algo_files:
            ab = np.load(str(ab_path))
            if ab.ndim < 4 or ab.shape[0] < 3:
                anomalies.append(f'invalid algo_bench shape {ab_path.name}: {ab.shape}')
                continue
            n_k = int(ab.shape[1])
            k_vals = _infer_hg_k_grid(nq, noise, n_k)
            if k_vals is None or len(k_vals) != n_k:
                anomalies.append(f'cannot infer k-grid for algo {ab_path.name} (n_k={n_k})')
                continue
            _add_k_trials(k_bucket, k_vals, ab[2].reshape(n_k, -1))
            sources.append(f'base_algo:{ab_path.name}')

    # 3) Legacy/patch directories (nq=11-13 1p3, nq=15 token patches)
    # 3a) 1p3 patch for missing k=1.3 in some nq=11,12,13 dephasing/relaxation cases
    if nq in (11, 12, 13) and noise in ('dephasing', 'relaxation'):
        p3_dir = hm_dir / '1p3' / str(nq)
        if p3_dir.exists():
            p3_pairs = _scan_case_pair_files(p3_dir, noise, gamma)
            if p3_pairs:
                for fb_path, ft_path in p3_pairs:
                    fb = np.array(np.load(str(fb_path), mmap_mode='r'))
                    ft = np.array(np.load(str(ft_path), mmap_mode='r'))
                    n_hg = int(ft.shape[1])
                    acc = compute_alpha_acc(fb, ft, nq)
                    n_k = int(acc.shape[0])
                    if n_k != 1:
                        anomalies.append(f'1p3 pair unexpected n_k={n_k} in {fb_path.name}')
                    k_vals = np.array([1.3], dtype=float)[:n_k]
                    _add_k_trials(k_bucket, k_vals, acc.reshape(n_k, n_hg * acc.shape[2]))
                    sources.append(f'patch1p3_pair:{fb_path.name}:n_hg={n_hg}')
            else:
                p3_algo = _scan_case_algo_files(p3_dir, noise, gamma)
                for ab_path in p3_algo:
                    ab = np.load(str(ab_path))
                    if ab.ndim >= 4 and ab.shape[0] >= 3:
                        n_k = int(ab.shape[1])
                        k_vals = np.array([1.3], dtype=float)[:n_k]
                        _add_k_trials(k_bucket, k_vals, ab[2].reshape(n_k, -1))
                        sources.append(f'patch1p3_algo:{ab_path.name}')

    # 3b) nq=15 token patches via algo_bench[2] (legacy layout)
    if nq == 15:
        for tok in PATCHES_15.get((noise, gamma), []):
            patch_dir = hm_dir / tok / str(nq)
            if not patch_dir.exists():
                continue
            # Prefer algo_bench here because that is what exists historically
            algo_files = _scan_case_algo_files(patch_dir, noise, gamma)
            if not algo_files:
                # Try pair files if present
                for fb_path, ft_path in _scan_case_pair_files(patch_dir, noise, gamma):
                    fb = np.array(np.load(str(fb_path), mmap_mode='r'))
                    ft = np.array(np.load(str(ft_path), mmap_mode='r'))
                    acc = compute_alpha_acc(fb, ft, nq)
                    n_k = int(acc.shape[0])
                    k_vals = PATCH_TOKEN_K.get(tok, np.array([], dtype=float))[:n_k]
                    if len(k_vals) != n_k:
                        anomalies.append(f'{tok} pair k-count mismatch {fb_path.name}: n_k={n_k}')
                        continue
                    _add_k_trials(k_bucket, k_vals, acc.reshape(n_k, -1))
                    sources.append(f'{tok}_pair:{fb_path.name}')
                continue
            for ab_path in algo_files:
                ab = np.load(str(ab_path))
                if ab.ndim < 4 or ab.shape[0] < 3:
                    anomalies.append(f'{tok} invalid algo_bench shape {ab_path.name}: {ab.shape}')
                    continue
                n_k = int(ab.shape[1])
                exp_k = PATCH_TOKEN_K.get(tok, np.array([], dtype=float))
                if n_k > len(exp_k):
                    anomalies.append(f'{tok} algo k-count mismatch file={n_k} expected={len(exp_k)}')
                k_vals = exp_k[:n_k]
                if len(k_vals) != n_k:
                    continue
                _add_k_trials(k_bucket, k_vals, ab[2].reshape(n_k, -1))
                sources.append(f'{tok}_algo:{ab_path.name}')

    if not k_bucket:
        raise FileNotFoundError(f'No raw Hypergraph data found for case {(nq, noise, gamma)} in {base_dir}')

    k_sorted = np.array(sorted(k_bucket.keys()), dtype=float)
    row_list = [np.concatenate(k_bucket[float(round(k, 4))]) for k in k_sorted]
    max_t = max(int(r.shape[0]) for r in row_list)
    acc_out = np.full((len(k_sorted), max_t), np.nan, dtype=float)
    for i, r in enumerate(row_list):
        acc_out[i, : r.shape[0]] = r

    meta = {
        'n_q': nq,
        'channel': noise,
        'p': gamma,
        'n_k': int(k_sorted.size),
        'n_trials_max': int(acc_out.shape[1]),
        'n_sources': int(len(sources)),
        'sources': sources,
        'anomalies': anomalies,
    }
    return acc_out, k_sorted, meta


def _aggregate_curve(acc_trials: np.ndarray, aggregate: str = 'mean') -> np.ndarray:
    x = np.asarray(acc_trials, dtype=float)
    if aggregate == 'median':
        return np.nanmedian(x, axis=1)
    return np.nanmean(x, axis=1)


def _curve_sem_from_trials(acc_trials: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-k uncertainty from raw trials.

    Returns
    -------
    sem_curve : (n_k,) one-sigma standard error of the mean at each observed k
    n_eff     : (n_k,) number of finite trials used
    std_curve : (n_k,) sample std across trials (ddof=1 when possible)
    """
    x = np.asarray(acc_trials, dtype=float)
    if x.ndim != 2:
        x = np.atleast_2d(x)
    n_eff = np.sum(np.isfinite(x), axis=1).astype(int)
    std_curve = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        xi = x[i]
        xi = xi[np.isfinite(xi)]
        if xi.size == 0:
            continue
        if xi.size == 1:
            std_curve[i] = 0.0
        else:
            std_curve[i] = float(np.std(xi, ddof=1))
    sem_curve = np.full_like(std_curve, np.nan, dtype=float)
    good = n_eff > 0
    sem_curve[good] = std_curve[good] / np.sqrt(n_eff[good])
    return sem_curve, n_eff, std_curve


def _monotone_curve(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    out = y.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    if not np.all(finite):
        idx = np.arange(out.size, dtype=float)
        out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return np.maximum.accumulate(out)


def _select_discrete_first_hit_k(
    k_grid: np.ndarray,
    y_curve: np.ndarray,
    threshold: float,
    monotone: bool = False,
) -> Tuple[str, float, float]:
    """Return smallest observed k with acc >= threshold, no interpolation.

    Returns (status, k_selected, acc_at_selected_k).
    """
    k = np.asarray(k_grid, dtype=float)
    y = np.asarray(y_curve, dtype=float)
    valid = np.isfinite(k) & np.isfinite(y)
    k = k[valid]
    y = y[valid]
    if k.size == 0:
        return 'invalid', np.nan, np.nan
    order = np.argsort(k)
    k = k[order]
    y = y[order]
    if monotone:
        y = np.maximum.accumulate(y)
    X = float(threshold)
    hit = np.where(y >= X)[0]
    if hit.size == 0:
        return 'no_hit', np.nan, np.nan
    j = int(hit[0])
    return ('ok' if j > 0 else 'ok_at_min_k'), float(k[j]), float(y[j])


def _select_nps_from_acc_curve_discrete(
    k_grid: np.ndarray,
    acc_trials: np.ndarray,
    n_q: int,
    target_accuracy: float,
    accuracy_eta: float = 0.0,
    aggregate: str = 'mean',
    monotone: bool = False,
) -> dict:
    acc_raw = float(target_accuracy)
    acc_target = acc_raw - float(accuracy_eta)
    curve = _aggregate_curve(acc_trials, aggregate=aggregate)
    sem_curve, n_eff_k, std_curve = _curve_sem_from_trials(acc_trials)
    curve_used = _monotone_curve(curve) if monotone else np.asarray(curve, dtype=float)
    status, kx, acc_at_k = _select_discrete_first_hit_k(k_grid, curve_used, acc_target, monotone=False)
    # One-sigma threshold-level uncertainty from raw trial variability:
    # use discrete first-hit on curve +/- SEM (no interpolation, no fitting).
    sem_used = np.asarray(sem_curve, dtype=float)
    if monotone:
        curve_plus = _monotone_curve(np.asarray(curve, dtype=float) + np.nan_to_num(sem_used, nan=0.0))
        curve_minus = _monotone_curve(np.asarray(curve, dtype=float) - np.nan_to_num(sem_used, nan=0.0))
    else:
        curve_plus = np.asarray(curve, dtype=float) + np.nan_to_num(sem_used, nan=0.0)
        curve_minus = np.asarray(curve, dtype=float) - np.nan_to_num(sem_used, nan=0.0)
    status_plus, kx_lo_req, _ = _select_discrete_first_hit_k(k_grid, curve_plus, acc_target, monotone=False)
    status_minus, kx_hi_req, _ = _select_discrete_first_hit_k(k_grid, curve_minus, acc_target, monotone=False)
    if _is_finite(kx):
        nps_val = float(2.0 ** (float(n_q) * float(kx)))
    else:
        nps_val = np.nan
    nps_lo = float(2.0 ** (float(n_q) * float(kx_lo_req))) if _is_finite(kx_lo_req) else np.nan
    nps_hi = float(2.0 ** (float(n_q) * float(kx_hi_req))) if _is_finite(kx_hi_req) else np.nan
    # If lower-curve no longer reaches target, keep upper error undefined (NaN).
    if _is_finite(nps_val) and _is_finite(nps_lo):
        err_lo = max(0.0, float(nps_val - nps_lo))
    else:
        err_lo = np.nan
    if _is_finite(nps_val) and _is_finite(nps_hi):
        err_hi = max(0.0, float(nps_hi - nps_val))
    else:
        err_hi = np.nan
    finite_curve = np.asarray(curve_used, dtype=float)[np.isfinite(curve_used)]
    selected_sem = np.nan
    selected_std = np.nan
    selected_n_eff = np.nan
    if _is_finite(kx):
        k_arr = np.asarray(k_grid, dtype=float)
        idx = np.where(np.isfinite(k_arr) & (np.abs(k_arr - float(kx)) < 1e-12))[0]
        if idx.size > 0:
            ii = int(idx[0])
            if ii < len(sem_curve):
                selected_sem = float(sem_curve[ii]) if _is_finite(sem_curve[ii]) else np.nan
            if ii < len(std_curve):
                selected_std = float(std_curve[ii]) if _is_finite(std_curve[ii]) else np.nan
            if ii < len(n_eff_k):
                selected_n_eff = int(n_eff_k[ii])
    return {
        'status': status,
        'accuracy_raw': acc_raw,
        'accuracy_target_eta': acc_target,
        'accuracy_used': float(acc_target) if _is_finite(acc_target) else np.nan,
        'accuracy_clipped_low': False,
        'accuracy_clipped_high': False,
        'k_x': float(kx) if _is_finite(kx) else np.nan,
        'acc_at_selected_k': float(acc_at_k) if _is_finite(acc_at_k) else np.nan,
        'acc_gap_selected_minus_target': float(acc_at_k - acc_target) if (_is_finite(acc_at_k) and _is_finite(acc_target)) else np.nan,
        'acc_sigma_sem_at_selected_k': selected_sem,
        'acc_sigma_std_at_selected_k': selected_std,
        'n_trials_eff_at_selected_k': selected_n_eff,
        'k_x_1sigma_lo': float(kx_lo_req) if _is_finite(kx_lo_req) else np.nan,
        'k_x_1sigma_hi': float(kx_hi_req) if _is_finite(kx_hi_req) else np.nan,
        'k_x_1sigma_lo_status': status_plus,
        'k_x_1sigma_hi_status': status_minus,
        'nps_pred_boundary': nps_val,
        'nps_pred': nps_val,
        'nps_1sigma_lo': nps_lo,
        'nps_1sigma_hi': nps_hi,
        'nps_err_lo_1sigma': err_lo,
        'nps_err_hi_1sigma': err_hi,
        'curve_acc_min': float(np.nanmin(finite_curve)) if finite_curve.size else np.nan,
        'curve_acc_max': float(np.nanmax(finite_curve)) if finite_curve.size else np.nan,
    }


def _build_shadow_raw_curves(shadow_mod, results_q5_10: Optional[str], results_q11_12: Optional[str]) -> List[dict]:
    r5, r11 = shadow_mod.find_shadow_results_json(results_q5_10, results_q11_12)
    return shadow_mod.load_shadowqmlml_curves([r5, r11])


def _accuracy_palette(accuracies: List[float]):
    if plt is None:
        return {a: None for a in accuracies}
    cmap = plt.get_cmap('viridis')
    n = max(1, len(accuracies))
    return {a: cmap(i / (n - 1) if n > 1 else 0.5) for i, a in enumerate(accuracies)}


def _plot_grid_multiacc(rows: List[dict], accuracies: List[float], out_path: Path, title: str, show_errorbars: bool = True):
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_IMPORT_ERROR}")

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 12), sharex=True, sharey=True)
    axes = np.asarray(axes)
    colors = _accuracy_palette(accuracies)

    grouped: Dict[Tuple[str, str, str, float], List[dict]] = {}
    for r in rows:
        grouped.setdefault((r['channel'], r['amplitude'], r['method_key'], float(r['accuracy_raw'])), []).append(r)
    for k in list(grouped):
        grouped[k] = sorted(grouped[k], key=lambda rr: int(rr['n_q']))

    all_x = [float(r['n_q']) for r in rows if r.get('plotted', True) and _is_finite(r.get('n_q'))]
    all_y = [float(r['nps_pred']) for r in rows if r.get('plotted', True) and _is_finite(r.get('nps_pred')) and float(r['nps_pred']) > 0]

    for i, ch in enumerate(CHANNEL_ORDER):
        for j, amp in enumerate(PLOT_AMPLITUDES):
            ax = axes[i, j]
            ax.set_title(f"{CHANNEL_TITLES.get(ch, ch)} | p={amp}")
            for acc in accuracies:
                for mk in ['hypergraph', 'ml']:
                    rs = [r for r in grouped.get((ch, amp, mk, float(acc)), []) if r.get('plotted', True)]
                    if not rs:
                        continue
                    x = np.array([int(r['n_q']) for r in rs], dtype=float)
                    y = np.array([float(r['nps_pred']) if _is_finite(r.get('nps_pred')) else np.nan for r in rs], dtype=float)
                    fin = np.isfinite(x) & np.isfinite(y) & (y > 0)
                    if not fin.any():
                        continue
                    ax.plot(
                        x[fin], y[fin],
                        color=colors[float(acc)],
                        linestyle=METHOD_LINESTYLES[mk],
                        linewidth=METHOD_LINEWIDTHS[mk],
                        marker=METHOD_MARKERS[mk],
                        markersize=3.8,
                        alpha=0.92,
                    )
                    if show_errorbars:
                        err_lo = np.array([float(r.get('nps_err_lo_1sigma')) if _is_finite(r.get('nps_err_lo_1sigma')) else np.nan for r in rs], dtype=float)
                        err_hi = np.array([float(r.get('nps_err_hi_1sigma')) if _is_finite(r.get('nps_err_hi_1sigma')) else np.nan for r in rs], dtype=float)
                        ef = fin.copy()
                        if ef.any():
                            x_e = x[ef]
                            y_e = y[ef]
                            elo = err_lo[ef]
                            ehi = err_hi[ef]
                            # Matplotlib handles 0-length bars; replace NaN with 0 to suppress missing sides.
                            elo = np.where(np.isfinite(elo) & (elo >= 0), elo, 0.0)
                            ehi = np.where(np.isfinite(ehi) & (ehi >= 0), ehi, 0.0)
                            ax.errorbar(
                                x_e, y_e,
                                yerr=np.vstack([elo, ehi]),
                                fmt='none',
                                ecolor=colors[float(acc)],
                                elinewidth=1.35 if mk == 'hypergraph' else 1.15,
                                capsize=3.0,
                                capthick=1.1 if mk == 'hypergraph' else 0.95,
                                alpha=0.45 if mk == 'hypergraph' else 0.38,
                                zorder=1,
                            )
            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            if i == 2:
                ax.set_xlabel('Qubits (n_q)')
            if j == 0:
                ax.set_ylabel('Least observed n_ps with acc >= target (log scale)')

    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        for ax in axes.flat:
            ax.set_xlim(xmin - 0.2, xmax + 0.2)
    if all_y:
        ymin = max(min(all_y) / 1.6, 1e0)
        ymax = max(all_y) * 1.6
        for ax in axes.flat:
            ax.set_ylim(ymin, ymax)

    # legends: accuracy colors and method styles
    acc_handles = [Line2D([0], [0], color=colors[a], linestyle='-', linewidth=2.2) for a in accuracies]
    acc_labels = [f"acc={a:.2f}" for a in accuracies]
    method_handles = [
        Line2D([0], [0], color='black', linestyle=METHOD_LINESTYLES['hypergraph'], marker=METHOD_MARKERS['hypergraph'], linewidth=2.0, markersize=4),
        Line2D([0], [0], color='black', linestyle=METHOD_LINESTYLES['ml'], marker=METHOD_MARKERS['ml'], linewidth=1.9, markersize=4),
    ]
    method_labels = [METHOD_LABELS['hypergraph'], METHOD_LABELS['ml']]

    leg1 = fig.legend(acc_handles, acc_labels, loc='center left', bbox_to_anchor=(0.86, 0.53), frameon=False, title='Target Accuracy', title_fontsize=10, fontsize=9)
    fig.add_artist(leg1)
    fig.legend(method_handles, method_labels, loc='center left', bbox_to_anchor=(0.86, 0.86), frameon=False, title='Method', title_fontsize=10, fontsize=9)

    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=[0, 0, 0.84, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _write_csv(path: Path, rows: List[dict]):
    cols = [
        'method_key', 'method_label', 'source_type', 'channel', 'amplitude', 'n_q', 'plotted', 'n_trials',
        'accuracy_raw', 'accuracy_target_eta', 'accuracy_eta', 'accuracy_used',
        'status', 'acc_at_selected_k', 'acc_gap_selected_minus_target',
        'acc_sigma_sem_at_selected_k', 'acc_sigma_std_at_selected_k', 'n_trials_eff_at_selected_k',
        'curve_acc_min', 'curve_acc_max', 'k_x',
        'k_x_1sigma_lo', 'k_x_1sigma_hi', 'k_x_1sigma_lo_status', 'k_x_1sigma_hi_status',
        'nps_pred_boundary', 'log2_nps_pred_boundary', 'nps_pred', 'log2_nps_pred'
        , 'nps_1sigma_lo', 'nps_1sigma_hi', 'nps_err_lo_1sigma', 'nps_err_hi_1sigma'
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, '')
                if isinstance(v, bool):
                    s = 'true' if v else 'false'
                elif v is None:
                    s = ''
                else:
                    s = str(v)
                if any(ch in s for ch in [',', '"', '\n']):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(','.join(vals) + '\n')


def _parse_accuracies(args_accuracies: Optional[List[float]]) -> List[float]:
    if args_accuracies:
        vals = [float(x) for x in args_accuracies]
    else:
        vals = list(np.arange(0.55, 0.95, 0.05))
    vals = sorted(set(round(float(v), 10) for v in vals))
    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(f'Accuracy values must be in (0,1], got {v}')
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Raw-data-only discrete first-hit n_ps plot for Hypergraph and ML. No inversion interpolation, no fitting, no extrapolation.'
    )
    parser.add_argument('--accuracies', type=float, nargs='*', default=None, help='Target accuracies. Default: np.arange(0.55, 0.95, 0.05)')
    parser.add_argument('--eta', type=float, default=0.0, help='Global accuracy slack eta enforcing Acc_C >= Acc_target - eta (implemented as target -> target-eta)')
    parser.add_argument('--aggregate', choices=['mean', 'median'], default='mean', help='Aggregate raw trials per k before discrete threshold selection')
    parser.add_argument('--monotone', action='store_true', help='Apply monotone cummax to aggregated acc(k) before discrete first-hit (default: off)')
    parser.add_argument('--no-errorbars', action='store_true', help='Disable 1-sigma error bars from raw trial variability')
    parser.add_argument('--hypergraph-merged-dir', type=str, default=None)
    parser.add_argument('--results-q5-10', type=str, default=None)
    parser.add_argument('--results-q11-12', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--nq-min-plot', type=int, default=5, help='Minimum n_q to show in plots (default 5)')
    parser.add_argument('--nq-max-plot', type=int, default=15, help='Maximum n_q to show in plots (default 15 for observed HG range)')
    args = parser.parse_args()

    if plt is None:
        print('ERROR: matplotlib unavailable:', MPL_IMPORT_ERROR)
        return 2
    if float(args.eta) < 0:
        raise ValueError(f'--eta must be >= 0, got {args.eta}')

    accuracies = _parse_accuracies(args.accuracies)
    shadow_mod = _load_module('shadow_raw_mod_multiacc', SCRIPT_DIR / 'plot_shadowqmlml_nps_from_quantum_accuracy_curves.py')
    hm_dir = Path(args.hypergraph_merged_dir) if args.hypergraph_merged_dir else _find_hypergraph_merged()
    out_dir = Path(args.output_dir) if args.output_dir else (SCRIPT_DIR / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    diag = {
        'accuracies': [float(a) for a in accuracies],
        'accuracy_eta': float(args.eta),
        'constraint': 'Acc_C >= Acc_target - eta (discrete first-hit on observed k grid)',
        'aggregate': args.aggregate,
        'monotone': bool(args.monotone),
        'show_errorbars': bool(not args.no_errorbars),
        'note': 'No interpolation inversion, no Step-4 fit, no bootstrap-threshold fit, no n_q extrapolation.',
        'nq_plot_range': [int(args.nq_min_plot), int(args.nq_max_plot)],
        'hypergraph_merged_dir': str(hm_dir),
        'hypergraph': {'cases_total': 0, 'cases_loaded': 0, 'cases_missing': 0, 'status_counts': {}, 'by_case': [], 'anomalies': []},
        'ml': {'cases_total': 0, 'cases_loaded': 0, 'status_counts': {}, 'by_case': []},
    }

    # Hypergraph raw curves (include new nq=4..8, but plot from nq>=5 and exclude 13)
    for ch in CHANNEL_ORDER:
        for p in [0.01, 0.05, 0.1]:
            for nq in HG_NQ_CANDIDATES:
                diag['hypergraph']['cases_total'] += 1
                try:
                    acc_trials, k_grid, meta = load_hypergraph_acc_trials(hm_dir, nq, ch, p)
                    for acc in accuracies:
                        inv = _select_nps_from_acc_curve_discrete(
                            k_grid=k_grid,
                            acc_trials=acc_trials,
                            n_q=nq,
                            target_accuracy=float(acc),
                            accuracy_eta=float(args.eta),
                            aggregate=args.aggregate,
                            monotone=bool(args.monotone),
                        )
                        plotted = (int(args.nq_min_plot) <= int(nq) <= int(args.nq_max_plot) and int(nq) not in HG_EXCLUDE_NQ_PLOT)
                        row = {
                            'method_key': 'hypergraph',
                            'method_label': METHOD_LABELS['hypergraph'],
                            'source_type': 'raw_hypergraph_merged',
                            'channel': ch,
                            'amplitude': f'{p:g}',
                            'n_q': int(nq),
                            'plotted': bool(plotted),
                            'n_trials': int(np.sum(np.isfinite(acc_trials[0])) if acc_trials.shape[0] > 0 else 0),
                            'accuracy_eta': float(args.eta),
                        }
                        row.update(inv)
                        row['log2_nps_pred_boundary'] = float(np.log2(row['nps_pred_boundary'])) if _is_finite(row.get('nps_pred_boundary')) and row['nps_pred_boundary'] > 0 else np.nan
                        row['log2_nps_pred'] = float(np.log2(row['nps_pred'])) if _is_finite(row.get('nps_pred')) and row['nps_pred'] > 0 else np.nan
                        rows.append(row)
                        st = str(inv.get('status', ''))
                        diag['hypergraph']['status_counts'][st] = int(diag['hypergraph']['status_counts'].get(st, 0)) + 1
                    diag['hypergraph']['cases_loaded'] += 1
                    diag['hypergraph']['by_case'].append({'channel': ch, 'p': p, 'n_q': int(nq), 'n_k': int(len(k_grid)), 'n_trials_max': int(acc_trials.shape[1])})
                    if meta.get('anomalies'):
                        diag['hypergraph']['anomalies'].append({'case': {'channel': ch, 'p': p, 'n_q': int(nq)}, 'anomalies': list(meta['anomalies'])})
                except Exception as e:
                    diag['hypergraph']['cases_missing'] += 1
                    diag['hypergraph']['by_case'].append({'channel': ch, 'p': p, 'n_q': int(nq), 'error': str(e)})

    # ML raw curves from scenario1_train_results
    shadow_curves = _build_shadow_raw_curves(shadow_mod, args.results_q5_10, args.results_q11_12)
    for curve in shadow_curves:
        nq = int(curve['n_q'])
        ch = str(curve['channel'])
        p = float(curve['p'])
        if ch not in CHANNEL_ORDER:
            continue
        if nq < ML_NQ_MIN or nq > ML_NQ_MAX:
            continue
        if min(abs(p - x) for x in [0.01, 0.05, 0.1]) > 1e-8:
            continue
        diag['ml']['cases_total'] += 1
        for acc in accuracies:
            inv = _select_nps_from_acc_curve_discrete(
                k_grid=np.asarray(curve['k_grid'], dtype=float),
                acc_trials=np.asarray(curve['acc_trials'], dtype=float),
                n_q=nq,
                target_accuracy=float(acc),
                accuracy_eta=float(args.eta),
                aggregate=args.aggregate,
                monotone=bool(args.monotone),
            )
            plotted = (int(args.nq_min_plot) <= int(nq) <= int(args.nq_max_plot))
            row = {
                'method_key': 'ml',
                'method_label': METHOD_LABELS['ml'],
                'source_type': 'raw_shadowqml_results',
                'channel': ch,
                'amplitude': f'{p:g}',
                'n_q': int(nq),
                'plotted': bool(plotted),
                'n_trials': int(curve.get('n_trials', np.asarray(curve['acc_trials']).shape[1])),
                'accuracy_eta': float(args.eta),
            }
            row.update(inv)
            row['log2_nps_pred_boundary'] = float(np.log2(row['nps_pred_boundary'])) if _is_finite(row.get('nps_pred_boundary')) and row['nps_pred_boundary'] > 0 else np.nan
            row['log2_nps_pred'] = float(np.log2(row['nps_pred'])) if _is_finite(row.get('nps_pred')) and row['nps_pred'] > 0 else np.nan
            rows.append(row)
            st = str(inv.get('status', ''))
            diag['ml']['status_counts'][st] = int(diag['ml']['status_counts'].get(st, 0)) + 1
        diag['ml']['cases_loaded'] += 1
        diag['ml']['by_case'].append({'channel': ch, 'p': p, 'n_q': int(nq), 'n_k': int(len(curve['k_grid'])), 'n_trials': int(curve.get('n_trials', np.asarray(curve['acc_trials']).shape[1]))})

    if not rows:
        raise RuntimeError('No rows produced from raw data')

    # Plot only flagged rows
    plot_rows = [r for r in rows if r.get('plotted', False)]
    if not plot_rows:
        raise RuntimeError('No plotted rows after nq/channel/p filters')

    accs_tag = f"accs_{accuracies[0]:.2f}_{accuracies[-1]:.2f}_step{(accuracies[1]-accuracies[0]) if len(accuracies)>1 else 0:.2f}".replace('.', 'p')
    eta_tag = (f"{float(args.eta):.3f}".rstrip('0').rstrip('.').replace('.', 'p') if float(args.eta) != 0 else '0')
    mono_tag = 'monotone' if args.monotone else 'rawcurve'
    base = f'hypergraph_vs_ml_required_nps_multiacc_from_raw_discrete_{accs_tag}_eta_{eta_tag}_{args.aggregate}_{mono_tag}_nq{args.nq_min_plot}_{args.nq_max_plot}'

    png = out_dir / f'{base}.png'
    pdf = out_dir / f'{base}.pdf'
    csv = out_dir / f'{base}.csv'
    diag_json = out_dir / f'{base}_diagnostics.json'

    title = (
        f'Least observed n_ps for acc >= target (raw simulation results only) | eta={float(args.eta):.3f}, '
        f'aggregate={args.aggregate}, monotone={bool(args.monotone)}\\n'
        f'Rows=channel type, Cols=noise strength p | Hypergraph n_q observed (plot excludes 4,13) and ML n_q=5..12'
    )
    _plot_grid_multiacc(plot_rows, accuracies, png, title, show_errorbars=not args.no_errorbars)
    _plot_grid_multiacc(plot_rows, accuracies, pdf, title, show_errorbars=not args.no_errorbars)
    _write_csv(csv, rows)
    diag['paths'] = {'png': str(png), 'pdf': str(pdf), 'csv': str(csv)}
    diag['plot_rows'] = int(len(plot_rows))
    diag['all_rows'] = int(len(rows))
    diag_json.write_text(json.dumps(diag, indent=2))

    print('Raw discrete multi-accuracy plot generation complete.')
    print('No inversion interpolation, no fitting, no extrapolation.')
    print(f'Accuracies: {accuracies}')
    print(f'eta={float(args.eta):.6f}, aggregate={args.aggregate}, monotone={bool(args.monotone)}')
    print('Output PNG:', png)
    print('Output PDF:', pdf)
    print('Output CSV:', csv)
    print('Diagnostics:', diag_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
