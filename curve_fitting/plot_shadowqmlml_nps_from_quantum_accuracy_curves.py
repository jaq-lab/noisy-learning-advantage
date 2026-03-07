#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    plt = None
    MPL_IMPORT_ERROR = e

try:
    from scipy.interpolate import PchipInterpolator
    HAS_PCHIP = True
except Exception:
    PchipInterpolator = None
    HAS_PCHIP = False

try:
    from scipy.interpolate import interp1d
except Exception:
    interp1d = None

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_ISOTONIC = True
except Exception:
    IsotonicRegression = None
    HAS_ISOTONIC = False

# ----------------------------
# Configuration / constants
# ----------------------------

L_BASELINE = 0.5
MIN_SIGMA_K = 1e-4
CI_TO_SIGMA_Z = 1.6448536269514722  # ~90% central interval to sigma
U_GRID_SIZE = 120
U_GRID_MARGIN = 5e-3
U_MAX = 0.999
SOFT_SLOPE_MIN = 1e-6
STEP4_MODEL = 'inv_n'  # analyze_8 default backbone for dense-threshold comparisons
THRESHOLDS_DENSE = np.arange(0.51, 0.98, 0.02)
BOOTSTRAP_REPS = 400
BOOTSTRAP_SEED = 12345
BOOTSTRAP_CI_LEVEL = 0.90

PLOT_AMPLITUDES = ['0.01', '0.05', '0.1']
PLOT_READOUTS = ['0%', '1%']
PLOT_NQ_MIN = 5
PLOT_NQ_MAX = 30

CHANNEL_ORDER = ['dephasing', 'relaxation', 'depolarizing']
PREDICTOR_CHANNEL_MAP = {
    'dephasing': 'dephasing',
    'depolarizing': 'depolarizing',
    'relaxation': 'thermal',
}
DEVICE_ORDER = ['I', 'S', 'T']
CHANNEL_COLORS = {
    'dephasing': '#1b9e77',
    'relaxation': '#d95f02',
    'depolarizing': '#7570b3',
}
CHANNEL_LABELS = {
    'dephasing': 'Dephasing',
    'relaxation': 'Thermal',
    'depolarizing': 'Depolarizing',
}
DEVICE_LINESTYLES = {'I': '-', 'S': '--', 'T': ':'}
DEVICE_MARKERS = {'I': 'o', 'S': 's', 'T': '^'}


# ----------------------------
# Path discovery
# ----------------------------

def _candidate_paths(script_dir: Path, names: Iterable[str]) -> List[Path]:
    out = []
    for name in names:
        out.extend([
            script_dir / name,
            script_dir / 'paper_data_2' / name,
            Path.cwd() / name,
            Path.cwd() / 'paper_data_2' / name,
        ])
    # de-dup preserve order
    seen = set()
    dedup = []
    for p in out:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp in seen:
            continue
        seen.add(rp)
        dedup.append(p)
    return dedup


def find_quantum_curves_json(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f'quantum_accuracy_curves json not found: {p}')
    script_dir = Path(__file__).resolve().parent
    for p in _candidate_paths(script_dir, ['quantum_accuracy_curves.json']):
        if p.exists():
            return p
    raise FileNotFoundError('Could not locate quantum_accuracy_curves.json')


def _default_shadow_results_paths(script_dir: Path) -> Tuple[Path, Path]:
    cands = [
        (
            Path('/home/ypatel/data1/shadow-qml-ml/data/exp_s1_fixed_alpha_all_ones_fixed_y_all_zeros_q_5_10/scenario1_train_results.json'),
            Path('/home/ypatel/data1/shadow-qml-ml/data/exp_s1_fixed_alpha_all_ones_fixed_y_all_zeros_q_11_12/scenario1_train_results.json'),
        ),
        (
            script_dir.parent.parent / 'shadow-qml-ml/data/exp_s1_fixed_alpha_all_ones_fixed_y_all_zeros_q_5_10/scenario1_train_results.json',
            script_dir.parent.parent / 'shadow-qml-ml/data/exp_s1_fixed_alpha_all_ones_fixed_y_all_zeros_q_11_12/scenario1_train_results.json',
        ),
        (
            Path.cwd() / 'shadow-qml-ml/data/exp_s1_fixed_alpha_all_ones_fixed_y_all_zeros_q_5_10/scenario1_train_results.json',
            Path.cwd() / 'shadow-qml-ml/data/exp_s1_fixed_alpha_all_ones_fixed_y_all_zeros_q_11_12/scenario1_train_results.json',
        ),
    ]
    for p1, p2 in cands:
        if p1.exists() and p2.exists():
            return p1, p2
    return cands[0]


def find_shadow_results_json(explicit_5_10: Optional[str] = None, explicit_11_12: Optional[str] = None) -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    d1, d2 = _default_shadow_results_paths(script_dir)
    p1 = Path(explicit_5_10) if explicit_5_10 else d1
    p2 = Path(explicit_11_12) if explicit_11_12 else d2
    if not p1.exists():
        raise FileNotFoundError(f'shadow-qml-ml results json (q_5_10) not found: {p1}')
    if not p2.exists():
        raise FileNotFoundError(f'shadow-qml-ml results json (q_11_12) not found: {p2}')
    return p1, p2


def default_output_dir(script_dir: Path) -> Path:
    # Prefer a local plots directory next to this script.
    out = script_dir / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------
# Utility helpers
# ----------------------------

def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def make_interp(x: np.ndarray, y: np.ndarray, extrapolate: bool = False) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.unique(x).size < 2:
        c = float(y[0])
        return lambda z: np.full_like(np.asarray(z, dtype=float), c, dtype=float)
    if HAS_PCHIP:
        pchip = PchipInterpolator(x, y, extrapolate=extrapolate)
        return lambda z: np.asarray(pchip(np.asarray(z, dtype=float)), dtype=float)

    def _f(z):
        z_arr = np.asarray(z, dtype=float)
        if extrapolate:
            return np.interp(z_arr, x, y)
        return np.interp(z_arr, x, y, left=np.nan, right=np.nan)

    return _f


def _safe_logit_threshold_transform(X, U: float, L: float = L_BASELINE, eps: float = 1e-6):
    X_arr = np.asarray(X, dtype=float)
    out = np.full_like(X_arr, np.nan, dtype=float)
    valid = np.isfinite(X_arr) & (U > L + 2 * eps) & (X_arr > L + eps) & (X_arr < U - eps)
    if np.any(valid):
        xv = X_arr[valid]
        out[valid] = np.log((xv - L) / (U - xv))
    return float(out) if np.isscalar(X) else out


def step4_design_matrix(n_q: np.ndarray, model: str = STEP4_MODEL) -> np.ndarray:
    n_q = np.asarray(n_q, dtype=float)
    inv_n = 1.0 / n_q
    if model == 'inv_n':
        return np.column_stack([np.ones_like(n_q), inv_n])
    if model == 'inv_n2':
        return np.column_stack([np.ones_like(n_q), inv_n, inv_n ** 2])
    raise ValueError(f'Unsupported step4 model: {model}')


def weighted_linear_fit(n_q: Iterable[float], y: Iterable[float], sigma: Iterable[float], model: str = STEP4_MODEL) -> dict:
    n_q = np.asarray(n_q, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(n_q) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    n_q, y, sigma = n_q[valid], y[valid], sigma[valid]
    min_points = 3 if model == 'inv_n' else 4
    if n_q.size < min_points:
        return {'ok_fit': False, 'reason': 'insufficient_points'}

    X = step4_design_matrix(n_q, model=model)
    w = 1.0 / np.maximum(sigma, MIN_SIGMA_K)
    Xw = X * w[:, None]
    yw = y * w
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return {'ok_fit': False, 'reason': 'lstsq_failed'}
    try:
        xtwx = Xw.T @ Xw
        cov_beta = np.linalg.pinv(xtwx)
    except np.linalg.LinAlgError:
        cov_beta = None

    coef = np.asarray(beta, dtype=float)
    y_hat = X @ coef
    resid = y - y_hat
    cond = float(np.linalg.cond(Xw))
    return {
        'ok_fit': True,
        'coef': coef,
        'coef_names': ['C', 'beta_inv_n'] if model == 'inv_n' else ['C', 'beta_inv_n', 'c_inv_n2'],
        'y_hat': y_hat,
        'resid': resid,
        'cond': cond,
        'n_points': int(n_q.size),
        'n_q': n_q,
        'step4_model': model,
        'cov_beta': np.asarray(cov_beta, dtype=float) if cov_beta is not None else None,
    }


def step4_predict(n_q, coef: np.ndarray, model: str = STEP4_MODEL):
    arr = np.asarray(n_q, dtype=float)
    X = step4_design_matrix(np.ravel(arr), model=model)
    out = (X @ np.asarray(coef, dtype=float)).reshape(arr.shape)
    return float(out) if np.isscalar(n_q) else out


# ----------------------------
# Bootstrap dense threshold data
# ----------------------------

def _parse_val_acc_trials_from_run(run: dict) -> Optional[dict]:
    setup = run.get('setup', {})
    meta = run.get('meta', {})
    try:
        n_q = int(setup['n'])
        channel = str(setup['channel_type'])
        p_val = float(setup['strength'])
    except Exception:
        return None

    per_nps = run.get('per_nps', [])
    if not isinstance(per_nps, list) or not per_nps:
        return None

    nps_list = []
    acc_cols = []
    n_trials_ref = None
    for item in per_nps:
        try:
            nps = int(item['nps'])
        except Exception:
            continue
        ir = item.get('independent_runs', {})
        per_run = ir.get('per_run', []) if isinstance(ir, dict) else []
        vals = []
        for rr in per_run:
            try:
                vals.append(float(rr['logreg']['val']['acc']))
            except Exception:
                continue
        if not vals:
            continue
        if n_trials_ref is None:
            n_trials_ref = len(vals)
        # Require consistent trial count across k for trajectory bootstrap.
        if len(vals) != n_trials_ref:
            return None
        nps_list.append(nps)
        acc_cols.append(vals)

    if not nps_list or not acc_cols:
        return None

    nps_arr = np.asarray(nps_list, dtype=float)
    order = np.argsort(nps_arr)
    nps_arr = nps_arr[order]
    acc_mat = np.asarray(acc_cols, dtype=float)[order, :]
    k_grid = np.log2(nps_arr) / float(n_q)
    return {
        'n_q': n_q,
        'channel': channel,
        'p': float(p_val),
        'nps_grid': nps_arr,
        'k_grid': k_grid,
        'acc_trials': acc_mat,  # shape (n_k, n_trials)
        'n_trials': int(acc_mat.shape[1]),
        'run_name': str(run.get('run_name', '')),
    }


def load_shadowqmlml_curves(results_paths: Iterable[Path]) -> List[dict]:
    curves = []
    seen = set()
    for path in results_paths:
        obj = json.loads(Path(path).read_text())
        for run in obj.get('runs', []):
            parsed = _parse_val_acc_trials_from_run(run)
            if not parsed:
                continue
            key = (parsed['n_q'], parsed['channel'], parsed['p'])
            if key in seen:
                # Prefer first occurrence; these experiment ranges are disjoint in n_q anyway.
                continue
            seen.add(key)
            curves.append(parsed)
    curves.sort(key=lambda r: (r['channel'], r['p'], r['n_q']))
    return curves


def _monotone_curve(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    if HAS_ISOTONIC and y.size >= 2:
        try:
            ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
            x = np.arange(y.size, dtype=float)
            return np.asarray(ir.fit_transform(x, y), dtype=float)
        except Exception:
            pass
    return np.maximum.accumulate(y)


def _invert_threshold_on_curve(k_grid: np.ndarray, y_mon: np.ndarray, threshold: float) -> Tuple[str, float]:
    k_grid = np.asarray(k_grid, dtype=float)
    y_mon = np.asarray(y_mon, dtype=float)
    X = float(threshold)
    valid = np.isfinite(k_grid) & np.isfinite(y_mon)
    k_grid = k_grid[valid]
    y_mon = y_mon[valid]
    if k_grid.size < 2:
        return 'invalid', np.nan
    order = np.argsort(k_grid)
    k_grid = k_grid[order]
    y_mon = y_mon[order]
    y_mon = np.maximum.accumulate(y_mon)

    if y_mon[0] >= X:
        return 'left_censored', float(k_grid[0])
    if y_mon[-1] < X:
        return 'right_censored', float(k_grid[-1])

    j = int(np.argmax(y_mon >= X))
    if j == 0:
        return 'ok', float(k_grid[0])
    y0, y1 = float(y_mon[j - 1]), float(y_mon[j])
    k0, k1 = float(k_grid[j - 1]), float(k_grid[j])
    if y1 <= y0:
        return 'ok', k1
    frac = (X - y0) / (y1 - y0)
    return 'ok', float(k0 + frac * (k1 - k0))


def _postprocess_bootstrap_rows(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        row = dict(r)
        row['channel'] = str(row['channel'])
        row['p'] = float(row['p'])
        row['threshold'] = float(row['threshold'])
        row['n_q'] = int(float(row['n_q']))
        row['status'] = str(row.get('status', ''))
        for k in ['k_x', 'k_x_lo', 'k_x_hi', 'k_x_err_lo', 'k_x_err_hi', 'log2_nps', 'n_valid', 'n_k', 'ci_level', 'n_boot']:
            if k in row:
                row[k] = _to_float(row[k])
        row['is_ok'] = row['status'] == 'ok' and _is_finite_number(row.get('k_x'))
        row['is_censored'] = row['status'] in {'left_censored', 'right_censored'}
        k_ci_width = row.get('k_x_hi', np.nan) - row.get('k_x_lo', np.nan)
        row['k_ci_width'] = k_ci_width
        row['k_ci_halfwidth'] = 0.5 * k_ci_width
        sigma_from_ci = row['k_ci_halfwidth'] / CI_TO_SIGMA_Z if _is_finite_number(row['k_ci_halfwidth']) else np.nan
        sigma_from_err = np.nan
        if _is_finite_number(row.get('k_x_err_lo')) and _is_finite_number(row.get('k_x_err_hi')):
            sigma_from_err = 0.5 * (float(row['k_x_err_lo']) + float(row['k_x_err_hi']))
        sigma = sigma_from_ci if _is_finite_number(sigma_from_ci) else sigma_from_err
        if not _is_finite_number(sigma) or sigma <= 0:
            sigma = MIN_SIGMA_K
        row['sigma_k'] = max(float(sigma), MIN_SIGMA_K)
        out.append(row)
    return out


def bootstrap_dense_threshold_rows(
    curves: List[dict],
    thresholds: np.ndarray = THRESHOLDS_DENSE,
    n_boot: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> List[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    alpha = (1.0 - float(ci_level)) / 2.0
    q_lo = 100.0 * alpha
    q_hi = 100.0 * (1.0 - alpha)

    for curve in curves:
        k_grid = np.asarray(curve['k_grid'], dtype=float)
        acc_trials = np.asarray(curve['acc_trials'], dtype=float)
        n_k, n_trials = acc_trials.shape
        if n_k < 2 or n_trials < 2:
            continue

        # Central curve from observed mean over independent runs.
        mean_curve = np.nanmean(acc_trials, axis=1)
        mean_curve_mon = _monotone_curve(mean_curve)

        # Bootstrap replicate means over runs.
        sample_idx = rng.integers(0, n_trials, size=(n_boot, n_trials))
        bs_curves = np.empty((n_boot, n_k), dtype=float)
        for b in range(n_boot):
            bs_curves[b] = np.nanmean(acc_trials[:, sample_idx[b]], axis=1)
        bs_curves_mon = np.empty_like(bs_curves)
        for b in range(n_boot):
            bs_curves_mon[b] = _monotone_curve(bs_curves[b])

        for X_thr in np.asarray(thresholds, dtype=float):
            status_c, k_center = _invert_threshold_on_curve(k_grid, mean_curve_mon, float(X_thr))
            k_samples = []
            for b in range(n_boot):
                st, kv = _invert_threshold_on_curve(k_grid, bs_curves_mon[b], float(X_thr))
                if st == 'ok' and np.isfinite(kv):
                    k_samples.append(float(kv))
            k_samples = np.asarray(k_samples, dtype=float)

            row = {
                'channel': str(curve['channel']),
                'p': float(curve['p']),
                'n_q': int(curve['n_q']),
                'threshold': float(X_thr),
                'status': status_c,
                'k_x': float(k_center) if (status_c == 'ok' and np.isfinite(k_center)) else np.nan,
                'k_x_lo': np.nan,
                'k_x_hi': np.nan,
                'k_x_err_lo': np.nan,
                'k_x_err_hi': np.nan,
                'log2_nps': float(curve['n_q'] * k_center) if (status_c == 'ok' and np.isfinite(k_center)) else np.nan,
                'n_valid': int(k_samples.size),
                'n_k': int(n_k),
                'n_boot': int(n_boot),
                'ci_level': float(ci_level),
                'n_trials': int(n_trials),
                'run_name': str(curve.get('run_name', '')),
            }
            if k_samples.size > 0:
                lo = float(np.nanpercentile(k_samples, q_lo))
                hi = float(np.nanpercentile(k_samples, q_hi))
                med = float(np.nanmedian(k_samples))
                row['k_x_lo'] = lo
                row['k_x_hi'] = hi
                if status_c == 'ok' and np.isfinite(row['k_x']):
                    row['k_x_err_lo'] = float(max(0.0, row['k_x'] - lo))
                    row['k_x_err_hi'] = float(max(0.0, hi - row['k_x']))
                elif status_c == 'ok':
                    row['k_x'] = med
                    row['log2_nps'] = float(curve['n_q'] * med)
                    row['k_x_err_lo'] = float(max(0.0, med - lo))
                    row['k_x_err_hi'] = float(max(0.0, hi - med))
            rows.append(row)
    return _postprocess_bootstrap_rows(rows)


def group_ok_rows_by_case(rows: List[dict]) -> Dict[Tuple[str, float], List[dict]]:
    out: Dict[Tuple[str, float], List[dict]] = {}
    for r in rows:
        if not r.get('is_ok'):
            continue
        key = (str(r['channel']), float(r['p']))
        out.setdefault(key, []).append(r)
    for key in out:
        out[key].sort(key=lambda rr: (float(rr['threshold']), int(rr['n_q'])))
    return out


# ----------------------------
# Predictor models (opt1, opt2_softslope)
# ----------------------------

@dataclass
class Predictor:
    name: str
    channel: str
    p: float
    threshold_min: float
    threshold_max: float
    predict_k_x: Callable[[np.ndarray, np.ndarray], np.ndarray]
    predict_log2_nps: Callable[[np.ndarray, np.ndarray], np.ndarray]
    predict_nps: Callable[[np.ndarray, np.ndarray], np.ndarray]
    predict_log2_nps_parametric_sigma: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    predict_C: Optional[Callable[[np.ndarray], np.ndarray]] = None
    meta: Optional[dict] = None


def build_option1_predictor(case_rows: List[dict], step4_model: str = STEP4_MODEL) -> Optional[Predictor]:
    if not case_rows:
        return None
    by_thr: Dict[float, List[dict]] = {}
    for r in case_rows:
        by_thr.setdefault(float(r['threshold']), []).append(r)

    fit_rows = []
    for thr in sorted(by_thr):
        g = sorted(by_thr[thr], key=lambda rr: int(rr['n_q']))
        n_q = np.array([int(rr['n_q']) for rr in g], dtype=float)
        kx = np.array([float(rr['k_x']) for rr in g], dtype=float)
        sig = np.array([float(rr['sigma_k']) for rr in g], dtype=float)
        fit = weighted_linear_fit(n_q, kx, sig, model=step4_model)
        if not fit.get('ok_fit', False):
            continue
        row = {
            'threshold': float(thr),
            'C': float(fit['coef'][0]),
            'beta_inv_n': float(fit['coef'][1]),
            'cond': float(fit['cond']),
            'n_points': int(fit['n_points']),
            'cov_beta': np.asarray(fit.get('cov_beta'), dtype=float) if fit.get('cov_beta') is not None else None,
        }
        if step4_model == 'inv_n2' and len(fit['coef']) >= 3:
            row['c_inv_n2'] = float(fit['coef'][2])
        fit_rows.append(row)
    if len(fit_rows) < 2:
        return None

    thresholds = np.array([r['threshold'] for r in fit_rows], dtype=float)
    C_vals = np.array([r['C'] for r in fit_rows], dtype=float)
    B_vals = np.array([r['beta_inv_n'] for r in fit_rows], dtype=float)
    fC = make_interp(thresholds, C_vals, extrapolate=False)
    fB = make_interp(thresholds, B_vals, extrapolate=False)
    fG = None
    if step4_model == 'inv_n2':
        G_vals = np.array([_to_float(r.get('c_inv_n2')) for r in fit_rows], dtype=float)
        fG = make_interp(thresholds, G_vals, extrapolate=False)
    p_dim = 3 if step4_model == 'inv_n2' else 2
    cov_interp = {}
    have_cov = all(isinstance(r.get('cov_beta'), np.ndarray) and r['cov_beta'].shape == (p_dim, p_dim) for r in fit_rows)
    if have_cov:
        for i in range(p_dim):
            for j in range(p_dim):
                vals = np.array([float(r['cov_beta'][i, j]) for r in fit_rows], dtype=float)
                cov_interp[(i, j)] = make_interp(thresholds, vals, extrapolate=False)
    x_min = float(np.min(thresholds))
    x_max = float(np.max(thresholds))
    channel = str(case_rows[0]['channel'])
    p_val = float(case_rows[0]['p'])

    def predict_k_x(X, n_q):
        X_arr, N_arr = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(X_arr, np.nan, dtype=float)
        valid = np.isfinite(X_arr) & np.isfinite(N_arr) & (N_arr > 0) & (X_arr >= x_min) & (X_arr <= x_max)
        if np.any(valid):
            xv = X_arr[valid]
            nv = N_arr[valid]
            y = np.asarray(fC(xv), dtype=float) + np.asarray(fB(xv), dtype=float) / nv
            if step4_model == 'inv_n2' and fG is not None:
                y = y + np.asarray(fG(xv), dtype=float) / (nv ** 2)
            out[valid] = y
        return float(out) if np.isscalar(X) and np.isscalar(n_q) else out

    def predict_C(X):
        return fC(X)

    def predict_log2_nps(X, n_q):
        X_arr, N_arr = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        return N_arr * predict_k_x(X_arr, N_arr)

    def predict_nps(X, n_q):
        return np.power(2.0, predict_log2_nps(X, n_q))

    def predict_log2_nps_parametric_sigma(X, n_q):
        X_arr, N_arr = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(X_arr, np.nan, dtype=float)
        if not have_cov:
            return float(out) if np.isscalar(X) and np.isscalar(n_q) else out
        valid = np.isfinite(X_arr) & np.isfinite(N_arr) & (N_arr > 0) & (X_arr >= x_min) & (X_arr <= x_max)
        if not np.any(valid):
            return float(out) if np.isscalar(X) and np.isscalar(n_q) else out
        idx_flat = np.flatnonzero(valid)
        xv = np.ravel(X_arr[valid]).astype(float)
        nv = np.ravel(N_arr[valid]).astype(float)
        for idx, (x_i, n_i) in enumerate(zip(xv, nv)):
            Sigma = np.zeros((p_dim, p_dim), dtype=float)
            bad = False
            for i in range(p_dim):
                for j in range(p_dim):
                    v = float(np.asarray(cov_interp[(i, j)](x_i), dtype=float))
                    if not np.isfinite(v):
                        bad = True
                        break
                    Sigma[i, j] = v
                if bad:
                    break
            if bad:
                continue
            if step4_model == 'inv_n2':
                g = np.array([n_i, 1.0, 1.0 / n_i], dtype=float)
            else:
                g = np.array([n_i, 1.0], dtype=float)
            var = float(g @ Sigma @ g)
            out.flat[idx_flat[idx]] = np.sqrt(max(var, 0.0))
        return float(out) if np.isscalar(X) and np.isscalar(n_q) else out

    return Predictor(
        name='opt1_dense_inversion',
        channel=channel,
        p=p_val,
        threshold_min=x_min,
        threshold_max=x_max,
        predict_k_x=predict_k_x,
        predict_log2_nps=predict_log2_nps,
        predict_nps=predict_nps,
        predict_log2_nps_parametric_sigma=predict_log2_nps_parametric_sigma,
        predict_C=predict_C,
        meta={'n_threshold_fits': len(fit_rows), 'step4_model': step4_model, 'has_parametric_cov': bool(have_cov)},
    )


def _weighted_line_fit_features_softslope(y: np.ndarray, sigma: np.ndarray, h: np.ndarray, slope_min: float = SOFT_SLOPE_MIN) -> dict:
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    h = np.asarray(h, dtype=float)
    valid = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(h)
    y, sigma, h = y[valid], sigma[valid], h[valid]
    if len(y) < 2:
        return {'ok': False, 'reason': 'insufficient_points'}

    X = np.column_stack([np.ones_like(h), h])
    w = 1.0 / np.maximum(sigma, MIN_SIGMA_K)
    Xw = X * w[:, None]
    yw = y * w
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return {'ok': False, 'reason': 'lstsq_failed'}

    a_raw = float(beta[0])
    b_raw = float(beta[1])
    w2 = w ** 2
    projected = False
    if not (np.isfinite(b_raw) and b_raw > slope_min):
        projected = True
        b_hat = float(slope_min)
        a_hat = float(np.average(y - b_hat * h, weights=w2))
        yhat = a_hat + b_hat * h
    else:
        a_hat = a_raw
        b_hat = b_raw
        yhat = X @ np.array([a_hat, b_hat], dtype=float)

    resid = y - yhat
    ss_res_w = float(np.sum(w2 * resid ** 2))
    ybar = np.average(y, weights=w2)
    ss_tot_w = float(np.sum(w2 * (y - ybar) ** 2))
    r2_w = np.nan if ss_tot_w <= 0 else 1 - ss_res_w / ss_tot_w
    return {
        'ok': True,
        'a': float(a_hat),
        'b': float(b_hat),
        'a_raw': float(a_raw),
        'b_raw': float(b_raw),
        'projected_slope': bool(projected),
        'ss_res_w': ss_res_w,
        'r2_w': float(r2_w),
        'n': int(len(y)),
    }


def _fit_case_threshold_shape_constant_U_softslope(case_rows: List[dict], slope_min: float = SOFT_SLOPE_MIN) -> dict:
    if not case_rows:
        return {'ok': False, 'reason': 'empty_case'}
    by_n: Dict[int, List[dict]] = {}
    for r in case_rows:
        by_n.setdefault(int(r['n_q']), []).append(r)
    if len(by_n) < 3:
        return {'ok': False, 'reason': 'insufficient_nq_groups'}

    for nq in by_n:
        by_n[nq] = sorted(by_n[nq], key=lambda rr: float(rr['threshold']))
    max_thr = max(float(r['threshold']) for r in case_rows)
    u_min = max(max_thr + U_GRID_MARGIN, L_BASELINE + 0.05)
    if not (u_min < U_MAX):
        return {'ok': False, 'reason': 'invalid_U_grid'}
    U_grid = np.linspace(u_min, U_MAX, U_GRID_SIZE)

    best = None
    for U in U_grid:
        rows = []
        total_sse = 0.0
        ok_groups = 0
        n_projected = 0
        for nq, g in by_n.items():
            thr = np.array([float(rr['threshold']) for rr in g], dtype=float)
            h = _safe_logit_threshold_transform(thr, float(U), L=L_BASELINE)
            kx = np.array([float(rr['k_x']) for rr in g], dtype=float)
            sig = np.array([float(rr['sigma_k']) for rr in g], dtype=float)
            fit = _weighted_line_fit_features_softslope(kx, sig, h, slope_min=slope_min)
            if not fit.get('ok', False):
                continue
            ok_groups += 1
            total_sse += float(fit['ss_res_w'])
            n_projected += int(bool(fit.get('projected_slope', False)))
            rows.append({
                'n_q': int(nq),
                'a_n': float(fit['a']),
                'b_n': float(fit['b']),
                'a_n_raw': float(fit.get('a_raw', fit['a'])),
                'b_n_raw': float(fit.get('b_raw', fit['b'])),
                'shape_projected_slope': int(bool(fit.get('projected_slope', False))),
                'shape_r2_w': float(fit['r2_w']),
                'n_thr_used': int(fit['n']),
            })
        if ok_groups < 3:
            continue
        objective = float(total_sse + 1e-9 * n_projected)
        cand = {
            'U': float(U),
            'objective': objective,
            'total_ss_res_w': float(total_sse),
            'n_groups': int(ok_groups),
            'n_projected': int(n_projected),
            'rows': rows,
        }
        if best is None or cand['objective'] < best['objective']:
            best = cand
    if best is None:
        return {'ok': False, 'reason': 'no_valid_U_fit'}
    return {
        'ok': True,
        'U': float(best['U']),
        'param_rows': sorted(best['rows'], key=lambda rr: int(rr['n_q'])),
        'shape_total_ss_res_w': float(best['total_ss_res_w']),
        'shape_n_projected': int(best['n_projected']),
    }


def build_option2_softslope_predictor(case_rows: List[dict], slope_min: float = SOFT_SLOPE_MIN, step4_model: str = STEP4_MODEL) -> Optional[Predictor]:
    if not case_rows:
        return None
    shape_fit = _fit_case_threshold_shape_constant_U_softslope(case_rows, slope_min=slope_min)
    if not shape_fit.get('ok', False):
        return None
    param_rows = shape_fit['param_rows']
    if len(param_rows) < 3:
        return None

    n_q = np.array([int(r['n_q']) for r in param_rows], dtype=float)
    a_n = np.array([float(r['a_n']) for r in param_rows], dtype=float)
    b_n = np.array([float(r['b_n']) for r in param_rows], dtype=float)
    fit_a = weighted_linear_fit(n_q, a_n, np.full_like(a_n, 0.02, dtype=float), model=step4_model)
    fit_b = weighted_linear_fit(n_q, b_n, np.full_like(b_n, 0.02, dtype=float), model=step4_model)
    if not (fit_a.get('ok_fit', False) and fit_b.get('ok_fit', False)):
        return None

    U = float(shape_fit['U'])
    threshold_min = float(min(float(r['threshold']) for r in case_rows))
    threshold_max = float(U - 1e-6)
    channel = str(case_rows[0]['channel'])
    p_val = float(case_rows[0]['p'])

    def _a_of_n(nv):
        return step4_predict(nv, fit_a['coef'], model=step4_model)

    def _b_raw_of_n(nv):
        return step4_predict(nv, fit_b['coef'], model=step4_model)

    def _b_of_n(nv):
        return np.maximum(np.asarray(_b_raw_of_n(nv), dtype=float), float(slope_min))

    def predict_k_x(X, n_q):
        X_arr, N_arr = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(X_arr, np.nan, dtype=float)
        h = _safe_logit_threshold_transform(X_arr, U, L=L_BASELINE)
        valid = np.isfinite(h) & np.isfinite(N_arr) & (N_arr > 0)
        if np.any(valid):
            out[valid] = np.asarray(_a_of_n(N_arr[valid]), dtype=float) + np.asarray(_b_of_n(N_arr[valid]), dtype=float) * h[valid]
        return float(out) if np.isscalar(X) and np.isscalar(n_q) else out

    def predict_C(X):
        X_arr = np.asarray(X, dtype=float)
        out = np.full_like(X_arr, np.nan, dtype=float)
        h = _safe_logit_threshold_transform(X_arr, U, L=L_BASELINE)
        valid = np.isfinite(h)
        if np.any(valid):
            a_inf = float(fit_a['coef'][0])
            b_inf = max(float(fit_b['coef'][0]), float(slope_min))
            out[valid] = a_inf + b_inf * h[valid]
        return float(out) if np.isscalar(X) else out

    def predict_log2_nps(X, n_q):
        X_arr, N_arr = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        return N_arr * predict_k_x(X_arr, N_arr)

    def predict_nps(X, n_q):
        return np.power(2.0, predict_log2_nps(X, n_q))

    def predict_log2_nps_parametric_sigma(X, n_q):
        X_arr, N_arr = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(X_arr, np.nan, dtype=float)
        cov_a = fit_a.get('cov_beta')
        cov_b = fit_b.get('cov_beta')
        if cov_a is None or cov_b is None:
            return float(out) if np.isscalar(X) and np.isscalar(n_q) else out
        cov_a = np.asarray(cov_a, dtype=float)
        cov_b = np.asarray(cov_b, dtype=float)
        h = _safe_logit_threshold_transform(X_arr, U, L=L_BASELINE)
        valid = np.isfinite(h) & np.isfinite(N_arr) & (N_arr > 0)
        if not np.any(valid):
            return float(out) if np.isscalar(X) and np.isscalar(n_q) else out
        idx_flat = np.flatnonzero(valid)
        nv = np.ravel(N_arr[valid]).astype(float)
        hv = np.ravel(h[valid]).astype(float)
        b_raw_v = np.asarray(_b_raw_of_n(nv), dtype=float).reshape(-1)
        for k_idx, (n_i, h_i, b_raw_i) in enumerate(zip(nv, hv, b_raw_v)):
            if step4_model == 'inv_n2':
                d = np.array([1.0, 1.0 / n_i, 1.0 / (n_i ** 2)], dtype=float)
            else:
                d = np.array([1.0, 1.0 / n_i], dtype=float)
            qa = n_i * d
            use_b_deriv = np.isfinite(b_raw_i) and (float(b_raw_i) > float(slope_min))
            qb = (n_i * h_i) * d if use_b_deriv else np.zeros_like(d)
            var = float(qa @ cov_a @ qa + qb @ cov_b @ qb)
            out.flat[idx_flat[k_idx]] = np.sqrt(max(var, 0.0))
        return float(out) if np.isscalar(X) and np.isscalar(n_q) else out

    return Predictor(
        name='opt2_threshold_logistic_proxy_softslope',
        channel=channel,
        p=p_val,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        predict_k_x=predict_k_x,
        predict_log2_nps=predict_log2_nps,
        predict_nps=predict_nps,
        predict_log2_nps_parametric_sigma=predict_log2_nps_parametric_sigma,
        predict_C=predict_C,
        meta={
            'U': U,
            'shape_n_projected': int(shape_fit.get('shape_n_projected', 0)),
            'shape_groups': int(len(param_rows)),
            'slope_min': float(slope_min),
            'step4_model': step4_model,
        },
    )


def build_case_predictors(rows_ok: List[dict], model_kind: str, step4_model: str = STEP4_MODEL) -> Dict[Tuple[str, str], Predictor]:
    grouped = group_ok_rows_by_case(rows_ok)
    out: Dict[Tuple[str, str], Predictor] = {}
    for (channel, p_val), case_rows in grouped.items():
        if model_kind == 'opt1':
            pred = build_option1_predictor(case_rows, step4_model=step4_model)
        elif model_kind == 'opt2_softslope':
            pred = build_option2_softslope_predictor(case_rows, step4_model=step4_model)
        else:
            raise ValueError(f'Unknown model_kind={model_kind}')
        if pred is None:
            continue
        out[(channel, format(p_val, 'g'))] = pred
        out[(channel, f'{p_val:.2f}')] = pred
        out[(channel, f'{p_val:.2f}'.rstrip('0').rstrip('.'))] = pred
    return out


# ----------------------------
# Quantum accuracy curves JSON
# ----------------------------

def load_quantum_accuracy_curves(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    required = ['nq_values', 'devices', 'channels', 'amplitudes', 'readout_errors', 'curves']
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f'Missing keys in quantum_accuracy_curves.json: {missing}')
    return d


# ----------------------------
# Plot data construction
# ----------------------------

def clip_accuracy_to_support(acc: np.ndarray, predictor: Predictor, mode: str = 'clip') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(acc, dtype=float)
    eps = 1e-6
    lo = max(float(predictor.threshold_min), L_BASELINE + eps)
    hi = float(predictor.threshold_max) - eps
    if hi <= lo:
        used = np.full_like(x, np.nan, dtype=float)
        return used, np.zeros_like(x, dtype=bool), np.zeros_like(x, dtype=bool), np.ones_like(x, dtype=bool)
    finite = np.isfinite(x)
    low_mask = finite & (x < lo)
    high_mask = finite & (x > hi)
    if mode == 'strict':
        used = x.copy()
        used[low_mask | high_mask] = np.nan
    else:
        used = np.clip(x, lo, hi)
        used[~finite] = np.nan
    out_mask = ~np.isfinite(used)
    return used, low_mask, high_mask, out_mask


def compute_plot_rows(
    curves_json: dict,
    predictors: Dict[Tuple[str, str], Predictor],
    model_kind: str,
    nq_min: int,
    nq_max: int,
    accuracy_mode: str = 'clip',
    accuracy_eta: float = 0.0,
) -> Tuple[List[dict], dict]:
    eta = float(accuracy_eta)
    if eta < 0:
        raise ValueError(f'accuracy_eta must be >= 0, got {eta}')
    nq_values_full = [int(x) for x in curves_json['nq_values']]
    idx = [i for i, n in enumerate(nq_values_full) if nq_min <= int(n) <= nq_max]
    if not idx:
        raise ValueError(f'No nq values in JSON within [{nq_min}, {nq_max}]')

    rows = []
    diag = {
        'model_kind': model_kind,
        'total_points': 0,
        'pred_points': 0,
        'nan_pred_points': 0,
        'clipped_low_points': 0,
        'clipped_high_points': 0,
        'missing_predictor_series': 0,
        'missing_predictor_keys': [],
        'accuracy_eta': eta,
        'constraint': 'Acc_C >= Acc_Q - eta',
        'eta_shifted_points': 0,
        'eta_target_below_zero_points': 0,
        'nps_ceiled_points': 0,
        'nps_ceiling_total_addition': 0.0,
        'by_series': [],
    }
    missing_keys = set()

    for readout in PLOT_READOUTS:
        for amp in PLOT_AMPLITUDES:
            for device in DEVICE_ORDER:
                for channel in CHANNEL_ORDER:
                    try:
                        acc_full = curves_json['curves'][device][channel][amp][readout]
                    except KeyError:
                        continue
                    acc = np.array([float(acc_full[i]) for i in idx], dtype=float)
                    acc_target = acc - eta if eta != 0.0 else acc.copy()
                    nq_arr = np.array([int(nq_values_full[i]) for i in idx], dtype=float)
                    pred_channel = PREDICTOR_CHANNEL_MAP.get(channel, channel)
                    pred = predictors.get((pred_channel, str(amp)))
                    if pred is None:
                        pred = predictors.get((pred_channel, str(float(amp)))) if amp not in (None, '') else None
                    if pred is None:
                        diag['missing_predictor_series'] += 1
                        missing_keys.add((pred_channel, amp))
                        nps_boundary = np.full_like(nq_arr, np.nan, dtype=float)
                        nps = np.full_like(nq_arr, np.nan, dtype=float)
                        acc_target_used = np.full_like(acc_target, np.nan, dtype=float)
                        acc_used = np.full_like(acc, np.nan, dtype=float)
                        low_mask = np.zeros_like(acc, dtype=bool)
                        high_mask = np.zeros_like(acc, dtype=bool)
                    else:
                        acc_used, low_mask, high_mask, _ = clip_accuracy_to_support(acc_target, pred, mode=accuracy_mode)
                        acc_target_used = acc_target
                        nps_boundary = np.asarray(pred.predict_nps(acc_used, nq_arr), dtype=float)
                        nps = np.where(np.isfinite(nps_boundary) & (nps_boundary > 0), np.ceil(nps_boundary), nps_boundary)

                    for n_q, a_raw, a_tgt, a_used, yb, y, cl, ch in zip(nq_arr, acc, acc_target_used, acc_used, nps_boundary, nps, low_mask, high_mask):
                        rows.append({
                            'model_kind': model_kind,
                            'device': device,
                            'channel': channel,
                            'predictor_channel': pred_channel,
                            'amplitude': str(amp),
                            'readout_error': str(readout),
                            'n_q': int(n_q),
                            'accuracy_raw': float(a_raw),
                            'accuracy_target_eta': float(a_tgt) if _is_finite_number(a_tgt) else np.nan,
                            'accuracy_eta': float(eta),
                            'accuracy_used': float(a_used) if _is_finite_number(a_used) else np.nan,
                            'accuracy_clipped_low': bool(cl),
                            'accuracy_clipped_high': bool(ch),
                            'nps_pred_boundary': float(yb) if _is_finite_number(yb) else np.nan,
                            'log2_nps_pred_boundary': float(np.log2(yb)) if _is_finite_number(yb) and yb > 0 else np.nan,
                            'nps_pred_is_ceiled': bool(_is_finite_number(yb) and _is_finite_number(y) and abs(float(y) - float(yb)) > 1e-12),
                            'nps_pred': float(y) if _is_finite_number(y) else np.nan,
                            'log2_nps_pred': float(np.log2(y)) if _is_finite_number(y) and y > 0 else np.nan,
                        })
                    diag['total_points'] += int(len(nq_arr))
                    diag['pred_points'] += int(np.isfinite(nps).sum())
                    diag['nan_pred_points'] += int((~np.isfinite(nps)).sum())
                    diag['clipped_low_points'] += int(low_mask.sum())
                    diag['clipped_high_points'] += int(high_mask.sum())
                    ceil_mask = np.isfinite(nps_boundary) & np.isfinite(nps) & (np.abs(nps - nps_boundary) > 1e-12)
                    diag['nps_ceiled_points'] += int(ceil_mask.sum())
                    if ceil_mask.any():
                        diag['nps_ceiling_total_addition'] += float(np.nansum((nps - nps_boundary)[ceil_mask]))
                    if eta != 0.0:
                        diag['eta_shifted_points'] += int(np.isfinite(acc_target).sum())
                        diag['eta_target_below_zero_points'] += int((np.isfinite(acc_target) & (acc_target < 0)).sum())
                    diag['by_series'].append({
                        'device': device,
                        'channel': channel,
                        'predictor_channel': pred_channel,
                        'amplitude': str(amp),
                        'readout_error': str(readout),
                        'n_points': int(len(nq_arr)),
                        'n_pred': int(np.isfinite(nps).sum()),
                        'n_nan_pred': int((~np.isfinite(nps)).sum()),
                        'n_clipped_low': int(low_mask.sum()),
                        'n_clipped_high': int(high_mask.sum()),
                        'accuracy_eta': float(eta),
                        'n_ceiled_nps': int(ceil_mask.sum()),
                        'predictor_threshold_min': float(pred.threshold_min) if pred else np.nan,
                        'predictor_threshold_max': float(pred.threshold_max) if pred else np.nan,
                    })

    diag['missing_predictor_keys'] = sorted([{'channel': k[0], 'amplitude': k[1]} for k in missing_keys], key=lambda d: (d['channel'], d['amplitude']))
    return rows, diag


# ----------------------------
# Plotting
# ----------------------------

def _series_label(device: str, channel: str) -> str:
    return f"{device}-{CHANNEL_LABELS.get(channel, channel)}"


def _plot_figure(model_title: str, plot_rows: List[dict], out_path: Path):
    if plt is None:
        raise RuntimeError(f'matplotlib is unavailable in this environment: {MPL_IMPORT_ERROR}')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes = np.asarray(axes)

    # Build lookup for rows by subplot + line
    grouped: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for r in plot_rows:
        key = (r['readout_error'], r['amplitude'], r['device'], r['channel'])
        grouped.setdefault(key, []).append(r)
    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda rr: int(rr['n_q']))

    # Plot lines
    legend_handles = []
    legend_labels = []
    for i, readout in enumerate(PLOT_READOUTS):
        for j, amp in enumerate(PLOT_AMPLITUDES):
            ax = axes[i, j]
            ax.set_title(f"readout={readout}, amp={amp}")
            for device in DEVICE_ORDER:
                for channel in CHANNEL_ORDER:
                    key = (readout, amp, device, channel)
                    rows = grouped.get(key, [])
                    if not rows:
                        continue
                    x = np.array([int(r['n_q']) for r in rows], dtype=float)
                    y = np.array([float(r['nps_pred']) if _is_finite_number(r['nps_pred']) else np.nan for r in rows], dtype=float)
                    color = CHANNEL_COLORS.get(channel, None)
                    ls = DEVICE_LINESTYLES.get(device, '-')
                    marker = DEVICE_MARKERS.get(device, None)
                    (line,) = ax.plot(
                        x,
                        y,
                        color=color,
                        linestyle=ls,
                        linewidth=1.8,
                        marker=marker,
                        markersize=3.5,
                        markevery=max(1, len(x) // 8),
                        alpha=0.95,
                        label=_series_label(device, channel),
                    )
                    if i == 0 and j == 0:
                        legend_handles.append(line)
                        legend_labels.append(_series_label(device, channel))

            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            ax.set_xlim(PLOT_NQ_MIN, PLOT_NQ_MAX)
            if i == 1:
                ax.set_xlabel('Qubits (n_q)')
            if j == 0:
                ax.set_ylabel('Required n_ps (log scale)')

    # Unique legend preserving order
    uniq_h, uniq_l = [], []
    seen = set()
    for h, l in zip(legend_handles, legend_labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)

    if uniq_h:
        fig.legend(
            uniq_h,
            uniq_l,
            loc='center left',
            bbox_to_anchor=(0.995, 0.5),
            frameon=False,
            ncol=1,
            fontsize=9,
        )

    fig.suptitle(model_title, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.88, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='Plot shadow-qml-ml required n_ps from quantum_accuracy_curves.json using opt1/opt2 predictors fit from scenario1_train_results.')
    parser.add_argument('--quantum-json', type=str, default=None, help='Path to quantum_accuracy_curves.json')
    parser.add_argument('--results-q5-10', type=str, default=None, help='Path to scenario1_train_results.json for q_5_10 shadow-qml-ml runs')
    parser.add_argument('--results-q11-12', type=str, default=None, help='Path to scenario1_train_results.json for q_11_12 shadow-qml-ml runs')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for plots and CSV outputs')
    parser.add_argument('--nq-min', type=int, default=PLOT_NQ_MIN)
    parser.add_argument('--nq-max', type=int, default=PLOT_NQ_MAX)
    parser.add_argument('--accuracy-mode', choices=['clip', 'strict'], default='clip', help='How to handle target accuracies outside predictor threshold support')
    parser.add_argument('--eta', type=float, default=0.0, help='Global accuracy slack eta enforcing Acc_C >= Acc_Q - eta (target accuracy is shifted to Acc_Q - eta before inversion)')
    parser.add_argument('--opt2-variant', choices=['softslope'], default='softslope', help='Option 2 variant to use (default: softslope)')
    parser.add_argument('--n-boot', type=int, default=BOOTSTRAP_REPS, help='Bootstrap replicates per (n_q, channel, p) curve for threshold inversion')
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--export-bootstrap-json', action='store_true', help='Also save dense bootstrap threshold rows JSON used for the fits')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    qjson_path = find_quantum_curves_json(args.quantum_json)
    results_q5_10_path, results_q11_12_path = find_shadow_results_json(args.results_q5_10, args.results_q11_12)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(script_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plt is None:
        print('ERROR: matplotlib is unavailable in this shell environment:', MPL_IMPORT_ERROR)
        print('Script was not able to render figures here. Run this script in your notebook/conda environment.')
        return 2

    curves_json = load_quantum_accuracy_curves(qjson_path)
    shadow_curves = load_shadowqmlml_curves([results_q5_10_path, results_q11_12_path])
    bootstrap_rows = bootstrap_dense_threshold_rows(
        shadow_curves,
        thresholds=THRESHOLDS_DENSE,
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed),
        ci_level=BOOTSTRAP_CI_LEVEL,
    )
    rows_ok = [r for r in bootstrap_rows if r.get('is_ok')]

    # Build predictors from all available OK threshold rows (dense support).
    predictors_opt1 = build_case_predictors(rows_ok, model_kind='opt1')
    predictors_opt2 = build_case_predictors(rows_ok, model_kind='opt2_softslope')
    print('Quantum curves JSON:', qjson_path)
    print('Shadow results q5_10:', results_q5_10_path)
    print('Shadow results q11_12:', results_q11_12_path)
    print('Output dir:', out_dir)
    print('Shadow curves loaded:', len(shadow_curves))
    print('Dense thresholds:', THRESHOLDS_DENSE.tolist())
    print('Bootstrap rows:', len(bootstrap_rows), 'ok rows:', len(rows_ok), 'censored rows:', sum(int(r.get('is_censored', False)) for r in bootstrap_rows))
    print('Predictors built (opt1):', sorted({(k[0], getattr(v, 'p', None)) for k,v in predictors_opt1.items() if isinstance(k[1], str) and len(k[1])<=4}))
    print('Predictors built (opt2_softslope):', sorted({(k[0], getattr(v, 'p', None)) for k,v in predictors_opt2.items() if isinstance(k[1], str) and len(k[1])<=4}))

    plot_rows_opt1, diag1 = compute_plot_rows(curves_json, predictors_opt1, 'opt1_dense_inversion', args.nq_min, args.nq_max, accuracy_mode=args.accuracy_mode, accuracy_eta=args.eta)
    plot_rows_opt2, diag2 = compute_plot_rows(curves_json, predictors_opt2, 'opt2_threshold_logistic_proxy_softslope', args.nq_min, args.nq_max, accuracy_mode=args.accuracy_mode, accuracy_eta=args.eta)

    # Save combined plotted values CSV and diagnostics JSON.
    csv_path = out_dir / f'shadowqmlml_required_nps_from_quantum_accuracy_curves_nq{args.nq_min}_{args.nq_max}.csv'
    diag_path = out_dir / f'shadowqmlml_required_nps_from_quantum_accuracy_curves_nq{args.nq_min}_{args.nq_max}_diagnostics.json'
    bootstrap_export_path = out_dir / f'shadowqmlml_bootstrap_k_x_dense_from_scenario1_train_results_nboot{args.n_boot}.json'
    all_rows = plot_rows_opt1 + plot_rows_opt2
    if all_rows:
        cols = [
            'model_kind','device','channel','amplitude','readout_error','n_q',
            'accuracy_raw','accuracy_target_eta','accuracy_eta','accuracy_used','accuracy_clipped_low','accuracy_clipped_high','predictor_channel',
            'nps_pred_boundary','log2_nps_pred_boundary','nps_pred_is_ceiled',
            'nps_pred','log2_nps_pred'
        ]
        with csv_path.open('w') as f:
            f.write(','.join(cols) + '\n')
            for r in all_rows:
                vals = []
                for c in cols:
                    v = r.get(c, '')
                    if isinstance(v, bool):
                        vals.append('true' if v else 'false')
                    elif v is None:
                        vals.append('')
                    else:
                        vals.append(str(v))
                f.write(','.join(vals) + '\n')

    diagnostics = {
        'quantum_json_path': str(qjson_path),
        'shadow_results_q5_10_path': str(results_q5_10_path),
        'shadow_results_q11_12_path': str(results_q11_12_path),
        'nq_plot_range': [int(args.nq_min), int(args.nq_max)],
        'accuracy_mode': args.accuracy_mode,
        'accuracy_eta': float(args.eta),
        'constraint': 'Acc_C >= Acc_Q - eta',
        'reported_nps_semantics': 'nps_pred is ceil(nps_pred_boundary), where nps_pred_boundary solves Acc_C = Acc_Q - eta on the fitted monotone predictor; this makes reported nps conservative for Acc_C >= Acc_Q - eta',
        'opt2_variant': args.opt2_variant,
        'bootstrap_config': {
            'thresholds_dense': THRESHOLDS_DENSE.tolist(),
            'n_boot': int(args.n_boot),
            'seed': int(args.bootstrap_seed),
            'ci_level': BOOTSTRAP_CI_LEVEL,
            'has_isotonic': bool(HAS_ISOTONIC),
            'has_pchip': bool(HAS_PCHIP),
        },
        'shadow_curve_summary': {
            'n_curves': int(len(shadow_curves)),
            'n_q_values': sorted({int(c['n_q']) for c in shadow_curves}),
            'channels': sorted({str(c['channel']) for c in shadow_curves}),
            'p_values': sorted({float(c['p']) for c in shadow_curves}),
            'n_trials_set': sorted({int(c['n_trials']) for c in shadow_curves}),
        },
        'bootstrap_status_counts': {
            'ok': int(sum(r.get('status') == 'ok' for r in bootstrap_rows)),
            'left_censored': int(sum(r.get('status') == 'left_censored' for r in bootstrap_rows)),
            'right_censored': int(sum(r.get('status') == 'right_censored' for r in bootstrap_rows)),
            'invalid': int(sum(r.get('status') == 'invalid' for r in bootstrap_rows)),
        },
        'opt1_diag': diag1,
        'opt2_diag': diag2,
        'opt1_predictor_support': [
            {'channel': p.channel, 'p': p.p, 'threshold_min': p.threshold_min, 'threshold_max': p.threshold_max, **(p.meta or {})}
            for p in sorted({id(v): v for v in predictors_opt1.values()}.values(), key=lambda q: (q.channel, q.p))
        ],
        'opt2_predictor_support': [
            {'channel': p.channel, 'p': p.p, 'threshold_min': p.threshold_min, 'threshold_max': p.threshold_max, **(p.meta or {})}
            for p in sorted({id(v): v for v in predictors_opt2.values()}.values(), key=lambda q: (q.channel, q.p))
        ],
    }
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    if args.export_bootstrap_json:
        bootstrap_export_path.write_text(json.dumps(bootstrap_rows, indent=2))

    opt1_png = out_dir / f'shadowqmlml_required_nps_opt1_from_quantum_accuracy_curves_nq{args.nq_min}_{args.nq_max}.png'
    opt2_png = out_dir / f'shadowqmlml_required_nps_opt2_softslope_from_quantum_accuracy_curves_nq{args.nq_min}_{args.nq_max}.png'
    opt1_pdf = out_dir / f'shadowqmlml_required_nps_opt1_from_quantum_accuracy_curves_nq{args.nq_min}_{args.nq_max}.pdf'
    opt2_pdf = out_dir / f'shadowqmlml_required_nps_opt2_softslope_from_quantum_accuracy_curves_nq{args.nq_min}_{args.nq_max}.pdf'

    title1 = 'Shadow-QML-ML Required n_ps from Quantum Accuracy Curves (Option 1 dense inversion; accuracy targets from JSON)'
    title2 = 'Shadow-QML-ML Required n_ps from Quantum Accuracy Curves (Option 2 soft-slope threshold-logistic proxy; accuracy targets from JSON)'
    _plot_figure(title1, plot_rows_opt1, opt1_png)
    _plot_figure(title2, plot_rows_opt2, opt2_png)
    # Also save PDF copies if possible
    _plot_figure(title1, plot_rows_opt1, opt1_pdf)
    _plot_figure(title2, plot_rows_opt2, opt2_pdf)

    print('\nSaved plots:')
    print(' -', opt1_png)
    print(' -', opt1_pdf)
    print(' -', opt2_png)
    print(' -', opt2_pdf)
    print('Saved data CSV:', csv_path)
    print('Saved diagnostics JSON:', diag_path)
    if args.export_bootstrap_json:
        print('Saved dense bootstrap JSON:', bootstrap_export_path)

    def _diag_line(name: str, d: dict):
        print(f"{name}: total={d['total_points']}, pred={d['pred_points']}, nan={d['nan_pred_points']}, clip_low={d['clipped_low_points']}, clip_high={d['clipped_high_points']}, missing_predictor_series={d['missing_predictor_series']}")

    print('\nDiagnostics summary:')
    _diag_line('opt1', diag1)
    _diag_line('opt2_softslope', diag2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
