#!/usr/bin/env python3
"""
plot_quantum_vs_classical_nps_unified.py
=========================================
Unified quantum vs classical n_ps comparison using the dense-threshold predictor pipeline.

For HG, Eigenshadow, and ML:
  1. Bootstrap a dense grid of accuracy thresholds → k_x(threshold, nq) with CI.
  2. Fit the Step4 model  k_x = C(threshold) + β(threshold)/nq  (WLS).
  3. For each quantum target q_acc(nq) − η, evaluate the predictor at each nq
     to obtain the required classical n_ps.

Advantages over plot_quantum_vs_classical_nps_comparison.py:
  • No discrete first-hit: eliminates quantization and winner's-curse bias.
  • Fits k_x vs 1/nq (Step4 "inv_n") rather than log2(nps) vs nq directly.
    This is the correct scaling law: k_x → C as nq → ∞, with β/nq finite-n correction.
  • Censoring is propagated explicitly: left/right_censored bounds are markers only.
  • Optional random-baseline override: when q_acc−η approaches 0.5+eps(nq),
    n_ps is forced to 1 to make the tail regime explicit.
  • Identical predictor machinery for HG and ML (same bootstrap → invert → Step4 pipeline).
  • HG/Eigenshadow bootstrap supports auto/per_k/cluster/hier_cluster modes.
    In 2D NaN-padded matrices, "cluster" is a column/curve bootstrap that preserves
    cross-k dependence; "per_k" is available as a sensitivity variant.
  • CI bands are calibrated with empirical forward-CV residual quantiles (coverage-matched)
    plus parametric WLS uncertainty in quadrature. This avoids relying on a pure Normal
    residual assumption and corrects the known underconfidence of σ_WLS alone.
    
TODO: refer to ./unified_correlation_issue.md the correlation between boostrapping trials 

Usage:
  python plot_quantum_vs_classical_nps_unified.py [options]

Key options:
  --etas FLOAT ...       Eta values (default: 0.0 0.02 0.05 0.10)
  --n-boot-hg INT        HG bootstrap replicates (default 400)
  --n-boot-ml INT        ML bootstrap replicates (default 400)
  --ci-z FLOAT           z for σ_eff CI bands (default 1.0 → ±1σ_eff)
  --extrap-min-horizons  Minimum CV horizons to trust extrapolation
  --extrap-max-sigma-cv  Maximum sigma_cv to trust extrapolation
  --cv-thr-pct-min/max   Percentile window used for forward-CV threshold slices
  --enforce-threshold-monotone
                          Option E monotonicity guard in threshold dimension
  --target-floor FLOAT   Minimum meaningful q_acc−η (default 0.52)
  --random-baseline-fix / --no-random-baseline-fix
                          Force n_ps=1 when q_acc−η < 0.5+eps(nq)
  --random-baseline-eps0 FLOAT
                          eps0 in eps(nq)=eps0*exp(-decay*(nq-nq_ref))
  --random-baseline-decay FLOAT
                          decay rate in eps(nq)=eps0*exp(-decay*(nq-nq_ref))
  --random-baseline-nq-ref INT
                          nq reference for the eps(nq) decay
  --save-hg-rows         Write HG bootstrap rows to JSON for inspection / caching
  --save-eig-rows        Write Eigenshadow bootstrap rows to JSON for inspection / caching
  --save-ml-rows         Write ML bootstrap rows to JSON for inspection / caching
  --hg-rows-json PATH    Load pre-computed HG rows from JSON (skip re-bootstrap)
  --eig-rows-json PATH   Load pre-computed Eigenshadow rows from JSON (skip re-bootstrap)
  --ml-rows-json PATH    Load pre-computed ML rows from JSON (skip re-bootstrap)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from statistics import NormalDist
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except Exception as _mpl_err:
    plt = None
    _MPL_IMPORT_ERROR = _mpl_err

try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_ISOTONIC = True
except Exception:
    IsotonicRegression = None
    _HAS_ISOTONIC = False

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Constants ─────────────────────────────────────────────────────────────────
L_BASELINE    = 0.5
MIN_SIGMA_K   = 1e-4

THRESHOLDS_DENSE = np.arange(0.51, 0.98, 0.02)
N_BOOT_HG     = 400
N_BOOT_ML     = 400
BOOT_SEED     = 12345
BOOT_CI_LEVEL = 0.90
CI_TO_SIGMA_Z_DEFAULT = NormalDist().inv_cdf(0.5 + 0.5 * BOOT_CI_LEVEL)
Z68_ABS = NormalDist().inv_cdf(0.5 + 0.5 * 0.6827)
Z90_ABS = NormalDist().inv_cdf(0.5 + 0.5 * 0.90)
Z95_ABS = NormalDist().inv_cdf(0.5 + 0.5 * 0.95)
STEP4_MODEL   = "inv_n"              # k_x = C + β/nq

CHANNEL_ORDER   = ["dephasing", "depolarizing", "relaxation"]
PLOT_AMPLITUDES = ["0.01", "0.05", "0.1"]
DEVICE_ORDER    = ["I", "S", "T"]
READOUT_ERRORS  = ["0%", "1%"]
NQ_PRED_DEFAULT = np.arange(5, 51)
DEFAULT_ETAS    = [0.0, 0.02, 0.05, 0.10]
TARGET_FLOOR    = 0.52              # q_acc − η must exceed this to be meaningful
POSTPROCESS_MIN_N_VALID = 20        # minimum bootstrap ok inversions to keep row in Step4 fits
RANDOM_BASELINE_FIX_DEFAULT = True
RANDOM_BASELINE_EPS0_DEFAULT = 0.02
RANDOM_BASELINE_DECAY_DEFAULT = 0.00
RANDOM_BASELINE_NQ_REF_DEFAULT = 5
SENSITIVITY_AMPS_DEFAULT = ["0.01", "0.05", "0.1"]
SENSITIVITY_ETAS_DEFAULT = [0.01, 0.02, 0.05]
SENSITIVITY_DELTAS_DEFAULT = [3, 10, 15, 30]
CACHE_FORMAT_VERSION = 1

# Extrapolation trust gate defaults (can be tuned later from diagnostics).
EXTRAP_TRUST_MIN_HORIZONS = 12
EXTRAP_TRUST_MAX_SIGMA_CV = 1.0     # in log2(nps), ~factor 2 at 1σ
EXTRAP_TRUST_MAX_RMSE_RATIO = 2.5   # rmse_max / rmse_pooled stability cap

# Forward-CV threshold slice percentiles (avoid brittle endpoints by default).
CV_THR_PCT_MIN = 10.0
CV_THR_PCT_MAX = 90.0
CV_MAX_HORIZON = 3                  # multi-step forward-CV max horizon
CV_MIN_SLICE_HORIZONS = 3           # ignore ultra-thin threshold slices for rmse_max
ENFORCE_THRESHOLD_MONOTONE_DEFAULT = False
MONOTONE_THRESHOLD_GRID_SIZE = 256
SIGMA_CV_K_MODE_DEFAULT = "pooled_q"  # pooled_q | pooled_max | median_per_threshold_q
SIGMA_CV_K_COVERAGE_DEFAULT = 0.6827
CV_COVERAGE_ZS_DEFAULT = [1.0, 1.6448536269514722, 1.959963984540054]



# Interval calibration defaults (split-conformal-like on held-out nq).
INTERVAL_METHOD_DEFAULT = "conformal_hybrid"   # cv | conformal | conformal_hybrid
CONFORMAL_HOLDOUT_COUNT_DEFAULT = 2
CONFORMAL_MIN_ABS_ERRORS_DEFAULT = 30
STRICT_SPLIT_CONFORMAL_DEFAULT = False

# HG bootstrap defaults.
HG_BOOTSTRAP_MODE_DEFAULT = "auto"  # auto | per_k | cluster | hier_cluster

# Optional curve-level bootstrap CI on final log2(nps)(nq) trajectories.
CURVE_BOOTSTRAP_CI_DEFAULT = False
CURVE_BOOTSTRAP_SIMULTANEOUS_DEFAULT = False
CURVE_BOOTSTRAP_MIN_VALID_DEFAULT = 100
CURVE_BOOTSTRAP_FIT_DEFAULT = "wls"  # wls | ols

# Observed nq sets for each method
HG_NQ_OBS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
ML_NQ_OBS = list(range(5, 13))      # 5 … 12
EIGENSHADOW_NQ_OBS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]

# ML data uses "thermal" for what HG/quantum-JSON call "relaxation"
ML_CHANNEL_CANONICAL = {"thermal": "relaxation"}

METHOD_ORDER = ["hypergraph", "eigenshadow", "ml"]
METHOD_COLORS = {
    "hypergraph": "#1b7837",
    "eigenshadow": "#c17c00",
    "ml": "#762a83",
}
METHOD_LABELS = {
    "hypergraph": "Hypergraph (HG)",
    "eigenshadow": "Eigenshadow",
    "ml": "ML (LR+feat)",
}
METHOD_MARKERS = {"hypergraph": "o", "eigenshadow": "D", "ml": "s"}
METHOD_LINEWIDTHS = {"hypergraph": 1.8, "eigenshadow": 1.75, "ml": 1.6}
OBS_NQ_BY_METHOD = {
    "hypergraph": HG_NQ_OBS,
    "eigenshadow": EIGENSHADOW_NQ_OBS,
    "ml": ML_NQ_OBS,
}
CHANNEL_TITLES    = {"dephasing": "Dephasing", "depolarizing": "Depolarizing", "relaxation": "Relaxation"}
DEVICE_TITLES     = {"I": "Ideal (I)", "S": "Superconducting (S)", "T": "Trapped-ion (T)"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _thr_key(t: float, ndigits: int = 12) -> float:
    """Canonical threshold key to avoid float-key drift across modules/files."""
    return round(float(t), int(ndigits))


def _safe_nps_from_log2(log2_v: float) -> float:
    """Convert log2(nps) -> nps with physical clamp nps>=1."""
    if not _is_finite(log2_v):
        return float("nan")
    v = max(0.0, float(log2_v))
    return float(np.power(2.0, np.clip(v, -1020.0, 1020.0)))


def _scalarize_interp_value(v) -> float:
    """Robust scalar extraction from interpolation outputs."""
    arr = np.asarray(v, dtype=float)
    if arr.size < 1:
        return float("nan")
    return float(arr.ravel()[0])


def _is_inference_row(row: dict) -> bool:
    """
    Return True only for rows that are valid inference points.

    Guardrail: this prevents any convention-only rows (e.g. pred_status=random_baseline)
    from leaking into fits/CV/calibration if such fields are present.
    """
    if not bool(row.get("is_ok", False)):
        return False
    ps = row.get("pred_status", None)
    if ps is not None and str(ps) != "ok":
        return False
    ip = row.get("is_inference_point", None)
    if ip is not None and not bool(ip):
        return False
    return True


def _central_coverage_from_z(ci_z: float) -> float:
    """Map two-sided z (e.g., 1.0) to central coverage (e.g., 0.6827)."""
    z = abs(_to_float(ci_z))
    if not _is_finite(z):
        z = 1.0
    c = 2.0 * NormalDist().cdf(z) - 1.0
    if not _is_finite(c):
        c = 0.6827
    return float(min(max(c, 1e-6), 0.999999))


def _empirical_abs_quantile_from_anchors(coverage: float, anchors: List[Tuple[float, float]]) -> float:
    """
    Interpolate |error| quantile at a requested central coverage.
    anchors: list of (coverage, q_abs), e.g. (0.6827, q68), (0.90, q90), (0.95, q95).
    """
    c = _to_float(coverage)
    if not _is_finite(c):
        c = 0.6827
    c = float(min(max(c, 1e-6), 0.999999))

    pts: List[Tuple[float, float]] = []
    for cov, qv in anchors:
        cov_f = _to_float(cov)
        qv_f = _to_float(qv)
        if _is_finite(cov_f) and _is_finite(qv_f) and (0.0 < cov_f < 1.0) and qv_f >= 0:
            pts.append((float(cov_f), float(qv_f)))
    if not pts:
        return float("nan")
    pts = sorted(pts, key=lambda t: t[0])

    # Explicit (0,0) anchor keeps small-coverage behavior well-defined.
    if pts[0][0] > 1e-9:
        pts = [(0.0, 0.0), *pts]

    covs = np.array([p[0] for p in pts], dtype=float)
    vals = np.array([p[1] for p in pts], dtype=float)
    if c <= covs[-1]:
        return float(np.interp(c, covs, vals))

    # Mild tail extrapolation: scale by Normal z-ratio beyond highest anchor.
    cov_max = float(covs[-1])
    q_max = float(vals[-1])
    z_c = _to_float(NormalDist().inv_cdf(0.5 + 0.5 * c))
    z_m = _to_float(NormalDist().inv_cdf(0.5 + 0.5 * cov_max))
    if _is_finite(z_c) and _is_finite(z_m) and z_m > 0 and q_max >= 0:
        return float(q_max * (z_c / z_m))
    return q_max


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_hg_merged(start=None) -> Path:
    p = Path(start).resolve() if start else Path.cwd().resolve()
    for d in [p, *p.parents]:
        for c in [
            d / "shadow-qml-analyze" / "paper_data_2" / "hypergraph_merged",
            d / "paper_data_2" / "hypergraph_merged",
            d / "hypergraph_merged",
        ]:
            if (c / "4").exists() or (c / "9").exists():
                return c
    fallback = Path("/home/ypatel/data1/shadow-qml-analyze/paper_data_2/hypergraph_merged")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Cannot locate hypergraph_merged directory")


def _find_shadow_surrogate(start=None) -> Path:
    p = Path(start).resolve() if start else Path.cwd().resolve()
    for d in [p, *p.parents]:
        for c in [
            d / "shadow-qml-analyze" / "paper_data_2" / "shadow_surrogate",
            d / "paper_data_2" / "shadow_surrogate",
            d / "shadow_surrogate",
        ]:
            if (c / "5").exists() and (c / "12").exists():
                return c
    fallback = Path("/home/ypatel/data1/shadow-qml-analyze/paper_data_2/shadow_surrogate")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Cannot locate shadow_surrogate directory")


def _array_sha256(arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _dir_signature(root: Path, *, suffixes: Tuple[str, ...] = (".npy", ".json")) -> dict:
    root = Path(root)
    suffix_set = {str(s).lower() for s in suffixes}
    entries: List[dict] = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if suffix_set and p.suffix.lower() not in suffix_set:
                continue
            st = p.stat()
            entries.append({
                "path": str(p.relative_to(root)),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            })
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": str(root.resolve()),
        "n_files": int(len(entries)),
        "suffixes": [str(s) for s in suffixes],
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _paths_signature(paths: List[Path]) -> dict:
    entries: List[dict] = []
    for p in sorted(Path(x).resolve() for x in paths):
        if not p.exists() or not p.is_file():
            continue
        st = p.stat()
        entries.append({
            "path": str(p),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        })
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "n_files": int(len(entries)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": entries,
    }


def _rows_payload(path: Path) -> Tuple[List[dict], dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    meta: dict = {}
    rows_obj = obj
    if isinstance(obj, dict):
        rows_obj = obj.get("rows", [])
        meta = dict(obj.get("cache_meta", {}) or {})
        if "cache_format_version" in obj:
            meta.setdefault("cache_format_version", obj.get("cache_format_version"))
    if not isinstance(rows_obj, list):
        raise ValueError(f"Expected JSON list of rows in {path}, got {type(obj)}")
    return [dict(r) for r in rows_obj], meta


def _load_rows_json(path: Path) -> List[dict]:
    rows_obj, _meta = _rows_payload(path)
    # Normalize types + recompute sigma_k/is_ok deterministically.
    return _postprocess_rows(rows_obj)


def _load_rows_json_with_meta(path: Path) -> Tuple[List[dict], dict]:
    rows_obj, meta = _rows_payload(path)
    return _postprocess_rows(rows_obj), dict(meta)


def _save_rows_json_with_meta(path: Path, rows: List[dict], cache_meta: dict) -> None:
    payload = {
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "cache_meta": dict(cache_meta),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _cache_meta_matches(expected: dict, got: dict) -> Tuple[bool, str]:
    if not isinstance(got, dict):
        return False, "missing_cache_meta"
    for k, v in expected.items():
        if got.get(k) != v:
            return False, f"mismatch:{k}"
    return True, "ok"


def _cache_key_from_meta(meta: dict) -> str:
    payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _tag_float(x: float) -> str:
    return f"{float(x):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _amp_tag_from_float(x: float) -> str:
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def _corrcoef_pairwise_nan(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for correlation, got shape={arr.shape}")
    n_b, n_t = arr.shape
    R = np.full((n_t, n_t), np.nan, dtype=float)
    N = np.zeros((n_t, n_t), dtype=int)
    if n_b < 2 or n_t < 2:
        return R, N
    for i in range(n_t):
        xi = arr[:, i]
        mxi = np.isfinite(xi)
        for j in range(i, n_t):
            xj = arr[:, j]
            mask = mxi & np.isfinite(xj)
            n = int(np.sum(mask))
            N[i, j] = N[j, i] = n
            if n < 3:
                continue
            a = xi[mask]
            b = xj[mask]
            sa = float(np.std(a))
            sb = float(np.std(b))
            if sa <= 0 or sb <= 0:
                if i == j and sa == 0.0 and sb == 0.0:
                    R[i, j] = 1.0
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            R[i, j] = R[j, i] = c
    return R, N


def _threshold_corr_summary(kx_boot: np.ndarray, thresholds: np.ndarray) -> dict:
    kb = np.asarray(kx_boot, dtype=float)
    thr = np.asarray(thresholds, dtype=float)
    if kb.ndim != 2:
        raise ValueError(f"kx_boot must be 2D (B,T), got shape={kb.shape}")
    if thr.ndim != 1 or kb.shape[1] != thr.size:
        raise ValueError(f"thresholds mismatch: kx_boot={kb.shape}, thresholds={thr.shape}")
    R, N = _corrcoef_pairwise_nan(kb)
    t = int(thr.size)
    iu = np.triu_indices(t, k=1)
    off = R[iu]
    off = off[np.isfinite(off)]
    valid_per_thr = np.sum(np.isfinite(kb), axis=0).astype(int)
    gap_map: Dict[str, List[float]] = {}
    for i in range(t):
        for j in range(i + 1, t):
            cij = _to_float(R[i, j])
            if not _is_finite(cij):
                continue
            gap = abs(float(thr[j]) - float(thr[i]))
            key = f"{gap:.6f}"
            gap_map.setdefault(key, []).append(float(cij))
    rho_by_gap = []
    for gk in sorted(gap_map.keys(), key=lambda x: float(x)):
        vals = np.asarray(gap_map[gk], dtype=float)
        rho_by_gap.append({
            "gap": float(gk),
            "n_pairs": int(vals.size),
            "rho_mean": float(np.nanmean(vals)),
            "rho_median": float(np.nanmedian(vals)),
        })
    return {
        "n_boot": int(kb.shape[0]),
        "n_thresholds": int(t),
        "n_thresholds_with_min3": int(np.sum(valid_per_thr >= 3)),
        "n_pairs_finite": int(off.size),
        "rho_offdiag_mean": float(np.nanmean(off)) if off.size else float("nan"),
        "rho_offdiag_median": float(np.nanmedian(off)) if off.size else float("nan"),
        "rho_offdiag_abs_mean": float(np.nanmean(np.abs(off))) if off.size else float("nan"),
        "rho_offdiag_p10": float(np.nanpercentile(off, 10.0)) if off.size else float("nan"),
        "rho_offdiag_p90": float(np.nanpercentile(off, 90.0)) if off.size else float("nan"),
        "n_valid_per_threshold": [int(v) for v in valid_per_thr.tolist()],
        "rho_by_gap": rho_by_gap,
        "corr_matrix": R.tolist(),
        "pair_counts": N.tolist(),
    }


def _save_kx_boot_case_npz(
    out_path: Path,
    *,
    method: str,
    channel: str,
    amplitude: float,
    thresholds: np.ndarray,
    nq_values: np.ndarray,
    kx_boot_btn: np.ndarray,
    cache_meta: Optional[dict] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        method=np.asarray([str(method)]),
        channel=np.asarray([str(channel)]),
        amplitude=np.asarray([float(amplitude)], dtype=float),
        thresholds=np.asarray(thresholds, dtype=float),
        nq_values=np.asarray(nq_values, dtype=int),
        kx_boot=np.asarray(kx_boot_btn, dtype=np.float32),
    )
    if cache_meta is not None:
        sidecar = out_path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps({"cache_meta": dict(cache_meta)}, indent=2, default=str))


def _interval_method_case_table(
    hg_predictor_meta: dict,
    eig_predictor_meta: dict,
    ml_predictor_meta: dict,
) -> Tuple[List[dict], dict]:
    """
    Flatten predictor interval-method metadata into a per-case table plus compact counts.
    """
    method_maps = [
        ("hypergraph", hg_predictor_meta),
        ("eigenshadow", eig_predictor_meta),
        ("ml", ml_predictor_meta),
    ]
    rows: List[dict] = []
    counts_by_effective: Dict[str, int] = {}
    n_fallback = 0
    n_strict_active = 0

    for method, meta_map in method_maps:
        for key, meta in sorted(meta_map.items(), key=lambda kv: str(kv[0])):
            ch = ""
            amp = ""
            if isinstance(key, tuple) and len(key) >= 2:
                ch = str(key[0])
                amp = str(key[1])
            req = str(meta.get("interval_method_requested", meta.get("interval_method", "cv")))
            eff = str(meta.get("interval_method_effective", meta.get("interval_method", "cv")))
            fb = meta.get("interval_method_fallback", {})
            fb_reason = str(fb.get("reason", "")) if isinstance(fb, dict) else ""
            fb_n_abs = int(_to_float(fb.get("n_abs_errors", 0))) if isinstance(fb, dict) and _is_finite(fb.get("n_abs_errors", 0)) else 0
            conf = meta.get("conformal_calibration", {})
            conf_n_abs = int(_to_float(conf.get("n_abs_errors", 0))) if isinstance(conf, dict) and _is_finite(conf.get("n_abs_errors", 0)) else 0
            holdout = [int(v) for v in conf.get("holdout_nq", [])] if isinstance(conf, dict) and isinstance(conf.get("holdout_nq", []), list) else []
            strict_req = bool(meta.get("strict_split_conformal_requested", False))
            strict_active = bool(meta.get("strict_split_conformal_active", False))
            strict_reason = str(meta.get("strict_split_reason", ""))

            row = {
                "method": method,
                "channel": ch,
                "amplitude": amp,
                "interval_method_requested": req,
                "interval_method_effective": eff,
                "fallback_reason": fb_reason,
                "fallback_n_abs_errors": int(fb_n_abs),
                "conformal_n_abs_errors": int(conf_n_abs),
                "conformal_holdout_nq": [int(v) for v in holdout],
                "strict_split_conformal_requested": strict_req,
                "strict_split_conformal_active": strict_active,
                "strict_split_reason": strict_reason,
            }
            rows.append(row)
            counts_by_effective[eff] = counts_by_effective.get(eff, 0) + 1
            if fb_reason:
                n_fallback += 1
            if strict_active:
                n_strict_active += 1

    summary = {
        "n_cases": int(len(rows)),
        "n_fallback_to_cv": int(n_fallback),
        "n_strict_split_active": int(n_strict_active),
        "counts_by_interval_method_effective": counts_by_effective,
    }
    return rows, summary


def _assemble_case_kx_boot_btn(
    replicate_by_nq: Dict[int, dict],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a case tensor (B,T,N) from per-nq replicate slices.

    Returns (thresholds, nq_values, kx_boot_btn) or None if assembly fails.
    """
    if not replicate_by_nq:
        return None
    rows: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for nq, payload in sorted(replicate_by_nq.items(), key=lambda kv: int(kv[0])):
        kb = np.asarray(payload.get("kx_boot"), dtype=float)
        thr = np.asarray(payload.get("thresholds"), dtype=float)
        if kb.ndim != 2 or thr.ndim != 1:
            continue
        if kb.shape[1] != thr.size or kb.shape[0] < 2 or thr.size < 2:
            continue
        rows.append((int(nq), kb, thr))
    if not rows:
        return None

    # Keep only slices that share the same threshold grid as the first valid slice.
    thr_ref = rows[0][2]
    t_ref = int(thr_ref.size)
    rows_same = [
        (nq, kb) for nq, kb, thr in rows
        if int(thr.size) == t_ref and np.allclose(thr, thr_ref, atol=1e-12, rtol=0.0)
    ]
    if not rows_same:
        return None

    b_common = min(int(kb.shape[0]) for _, kb in rows_same)
    if b_common < 2:
        return None
    nq_vals = np.asarray([int(nq) for nq, _ in rows_same], dtype=int)
    btn = np.full((b_common, t_ref, int(nq_vals.size)), np.nan, dtype=np.float32)
    for idx, (_, kb) in enumerate(rows_same):
        btn[:, :, idx] = np.asarray(kb[:b_common, :t_ref], dtype=np.float32)
    return np.asarray(thr_ref, dtype=float), nq_vals, btn


def _extract_sigma_k_tn_for_case(
    rows: List[dict],
    *,
    channel: str,
    amplitude_tag: str,
    thresholds: np.ndarray,
    nq_values: np.ndarray,
) -> np.ndarray:
    """
    Build sigma_k(threshold, nq) surface aligned to a case tensor grid.
    """
    thr = np.asarray(thresholds, dtype=float).ravel()
    nqs = np.asarray(nq_values, dtype=int).ravel()
    out = np.full((int(thr.size), int(nqs.size)), np.nan, dtype=float)
    if thr.size == 0 or nqs.size == 0:
        return out

    p_target = _to_float(amplitude_tag)
    if not _is_finite(p_target):
        return out

    thr_idx = {_thr_key(float(t)): i for i, t in enumerate(thr.tolist())}
    nq_idx = {int(n): i for i, n in enumerate(nqs.tolist())}
    bucket: Dict[Tuple[int, int], List[float]] = {}

    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("channel", "")) != str(channel):
            continue
        p = _to_float(r.get("p", float("nan")))
        if not _is_finite(p) or abs(float(p) - float(p_target)) > 1e-8:
            continue
        nq = int(_to_float(r.get("n_q", -1)))
        thr_v = _to_float(r.get("threshold", float("nan")))
        sig = _to_float(r.get("sigma_k", float("nan")))
        if nq not in nq_idx:
            continue
        ti = thr_idx.get(_thr_key(float(thr_v)), None) if _is_finite(thr_v) else None
        if ti is None:
            continue
        if not (_is_finite(sig) and float(sig) > 0):
            continue
        bucket.setdefault((int(ti), int(nq_idx[nq])), []).append(float(sig))

    for (ti, ni), vals in bucket.items():
        vv = np.asarray(vals, dtype=float)
        vv = vv[np.isfinite(vv) & (vv > 0)]
        if vv.size > 0:
            out[int(ti), int(ni)] = float(np.nanmedian(vv))
    return out


def _nq_corr_summary_from_btn(kx_boot_btn: np.ndarray, nq_values: np.ndarray, thresholds: np.ndarray) -> dict:
    """
    Correlation across n_q at fixed threshold from kx_boot tensor (B,T,N).
    """
    btn = np.asarray(kx_boot_btn, dtype=float)
    nqs = np.asarray(nq_values, dtype=int)
    thr = np.asarray(thresholds, dtype=float)
    if btn.ndim != 3:
        raise ValueError(f"kx_boot_btn must be 3D (B,T,N), got shape={btn.shape}")
    if btn.shape[1] != thr.size or btn.shape[2] != nqs.size:
        raise ValueError(f"shape mismatch: btn={btn.shape}, thr={thr.shape}, nqs={nqs.shape}")

    gap_map: Dict[str, List[float]] = {}
    off_all: List[float] = []
    per_threshold: List[dict] = []
    n_t = int(thr.size)
    n_nq = int(nqs.size)
    for t_idx in range(n_t):
        series = btn[:, t_idx, :]  # (B,N)
        R, N = _corrcoef_pairwise_nan(series)
        iu = np.triu_indices(n_nq, k=1)
        off = np.asarray(R[iu], dtype=float)
        off = off[np.isfinite(off)]
        off_all.extend(off.tolist())
        for i in range(n_nq):
            for j in range(i + 1, n_nq):
                cij = _to_float(R[i, j])
                if not _is_finite(cij):
                    continue
                gap = abs(int(nqs[j]) - int(nqs[i]))
                key = str(int(gap))
                gap_map.setdefault(key, []).append(float(cij))
        per_threshold.append({
            "threshold": float(thr[t_idx]),
            "n_pairs_finite": int(off.size),
            "rho_offdiag_mean": float(np.nanmean(off)) if off.size else float("nan"),
            "rho_offdiag_median": float(np.nanmedian(off)) if off.size else float("nan"),
        })

    off_arr = np.asarray(off_all, dtype=float)
    rho_by_nq_gap = []
    for gk in sorted(gap_map.keys(), key=lambda x: int(x)):
        vals = np.asarray(gap_map[gk], dtype=float)
        rho_by_nq_gap.append({
            "nq_gap": int(gk),
            "n_pairs": int(vals.size),
            "rho_mean": float(np.nanmean(vals)),
            "rho_median": float(np.nanmedian(vals)),
        })
    return {
        "n_boot": int(btn.shape[0]),
        "n_thresholds": int(n_t),
        "n_nq": int(n_nq),
        "n_pairs_finite": int(off_arr.size),
        "rho_offdiag_mean": float(np.nanmean(off_arr)) if off_arr.size else float("nan"),
        "rho_offdiag_median": float(np.nanmedian(off_arr)) if off_arr.size else float("nan"),
        "rho_offdiag_abs_mean": float(np.nanmean(np.abs(off_arr))) if off_arr.size else float("nan"),
        "rho_offdiag_p10": float(np.nanpercentile(off_arr, 10.0)) if off_arr.size else float("nan"),
        "rho_offdiag_p90": float(np.nanpercentile(off_arr, 90.0)) if off_arr.size else float("nan"),
        "rho_by_nq_gap": rho_by_nq_gap,
        "per_threshold": per_threshold,
    }


def _fit_step4_bootstrap_coefficients(
    kx_boot_btn: np.ndarray,
    nq_values: np.ndarray,
    min_points: int = 3,
    *,
    fit_mode: str = CURVE_BOOTSTRAP_FIT_DEFAULT,
    sigma_k_tn: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Step4 coefficients per bootstrap replicate and threshold:
      k_x(nq) = C + beta / nq

    Inputs:
      kx_boot_btn: (B, T, Nobs)
      nq_values:   (Nobs,)
    Returns:
      (C_boot, B_boot) each shape (B, T), with NaN where fit is invalid.
    """
    btn = np.asarray(kx_boot_btn, dtype=float)
    nqs = np.asarray(nq_values, dtype=float)
    if btn.ndim != 3:
        raise ValueError(f"kx_boot_btn must be 3D (B,T,N), got {btn.shape}")
    if nqs.ndim != 1 or btn.shape[2] != nqs.size:
        raise ValueError(f"nq_values shape mismatch: btn={btn.shape}, nqs={nqs.shape}")
    if not np.all(np.isfinite(nqs) & (nqs > 0)):
        raise ValueError("nq_values must be finite and > 0")

    bsz, t_sz, _ = btn.shape
    C = np.full((bsz, t_sz), np.nan, dtype=float)
    Bc = np.full((bsz, t_sz), np.nan, dtype=float)
    x = 1.0 / nqs
    x2 = x * x

    mode = str(fit_mode).strip().lower()
    if mode not in {"ols", "wls"}:
        mode = "ols"
    sigma_tn = None
    if mode == "wls" and sigma_k_tn is not None:
        sigma_tn = np.asarray(sigma_k_tn, dtype=float)
        if sigma_tn.shape != (t_sz, nqs.size):
            sigma_tn = None

    for t_idx in range(t_sz):
        Y = btn[:, t_idx, :]  # (B,N)
        M = np.isfinite(Y)
        if mode == "wls" and sigma_tn is not None:
            sig_t = sigma_tn[t_idx, :]
            good_sig = np.isfinite(sig_t) & (sig_t > 0)
            w0 = np.zeros_like(sig_t, dtype=float)
            w0[good_sig] = 1.0 / np.maximum(sig_t[good_sig], 1e-12) ** 2
            W = np.where(M & good_sig[None, :], w0[None, :], 0.0)
        else:
            W = M.astype(float)
        S = np.sum(W, axis=1)
        S_pts = np.sum(W > 0, axis=1)
        Y0 = np.where(M, Y, 0.0)
        Sx = np.sum(W * x[None, :], axis=1)
        Sxx = np.sum(W * x2[None, :], axis=1)
        Sy = np.sum(W * Y0, axis=1)
        Sxy = np.sum(W * Y0 * x[None, :], axis=1)
        D = S * Sxx - Sx * Sx
        valid = (S_pts >= int(min_points)) & np.isfinite(D) & (np.abs(D) > 1e-12)
        if not np.any(valid):
            continue
        C[valid, t_idx] = (Sy[valid] * Sxx[valid] - Sx[valid] * Sxy[valid]) / D[valid]
        Bc[valid, t_idx] = (S[valid] * Sxy[valid] - Sx[valid] * Sy[valid]) / D[valid]
    return C, Bc


def _interp_bootstrap_rows_at_threshold(
    thresholds: np.ndarray,
    values_bt: np.ndarray,
    target: float,
) -> np.ndarray:
    """
    Row-wise threshold interpolation for bootstrap arrays.

    thresholds: (T,)
    values_bt:  (B,T)
    target: scalar threshold
    Returns:
      (B,) interpolated values (NaN where unavailable).
    """
    thr = np.asarray(thresholds, dtype=float).ravel()
    vals = np.asarray(values_bt, dtype=float)
    if vals.ndim != 2:
        raise ValueError(f"values_bt must be 2D (B,T), got {vals.shape}")
    if thr.size != vals.shape[1]:
        raise ValueError(f"threshold mismatch: thr={thr.shape}, vals={vals.shape}")
    if thr.size < 2 or not _is_finite(target):
        return np.full(vals.shape[0], np.nan, dtype=float)
    if target <= float(thr[0]):
        out = vals[:, 0]
        return np.where(np.isfinite(out), out, np.nan)
    if target >= float(thr[-1]):
        out = vals[:, -1]
        return np.where(np.isfinite(out), out, np.nan)

    j = int(np.searchsorted(thr, float(target), side="left"))
    j = max(1, min(j, int(thr.size - 1)))
    t0 = float(thr[j - 1])
    t1 = float(thr[j])
    y0 = vals[:, j - 1]
    y1 = vals[:, j]
    out = np.full(vals.shape[0], np.nan, dtype=float)
    good = np.isfinite(y0) & np.isfinite(y1)
    if not np.any(good):
        return out
    if abs(t1 - t0) <= 1e-15:
        out[good] = y1[good]
        return out
    frac = (float(target) - t0) / (t1 - t0)
    out[good] = y0[good] + frac * (y1[good] - y0[good])
    return out


def _curve_bootstrap_log2_bands(
    *,
    thresholds: np.ndarray,
    nq_values: np.ndarray,
    kx_boot_btn: np.ndarray,
    target_by_nq: Dict[int, float],
    nq_pred: np.ndarray,
    coverage: float,
    enforce_threshold_monotone: bool = False,
    simultaneous: bool = False,
    min_valid: int = CURVE_BOOTSTRAP_MIN_VALID_DEFAULT,
    fit_mode: str = CURVE_BOOTSTRAP_FIT_DEFAULT,
    sigma_k_tn: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, dict], dict]:
    """
    Full-pipeline bootstrap bands on final log2(nps)(nq) curve.

    Pipeline per bootstrap replicate:
      kx_boot(thr, nq_obs) -> Step4 fit on nq_obs -> kx_boot(thr, nq_pred)
      -> evaluate at target(q_acc(nq)-eta) -> log2(nps)_boot(nq).
    """
    thr = np.asarray(thresholds, dtype=float).ravel()
    nqs_obs = np.asarray(nq_values, dtype=float).ravel()
    btn = np.asarray(kx_boot_btn, dtype=float)
    nq_arr = np.asarray(nq_pred, dtype=int).ravel()
    if btn.ndim != 3:
        raise ValueError(f"kx_boot_btn must be 3D (B,T,N), got {btn.shape}")
    if thr.size != btn.shape[1] or nqs_obs.size != btn.shape[2]:
        raise ValueError(f"shape mismatch: thr={thr.shape}, btn={btn.shape}, nqs_obs={nqs_obs.shape}")

    C_cov = float(min(max(_to_float(coverage), 1e-6), 0.999999))
    alpha = 0.5 * (1.0 - C_cov)
    q_lo = 100.0 * alpha
    q_hi = 100.0 * (1.0 - alpha)

    fit_req = str(fit_mode).strip().lower()
    if fit_req not in {"ols", "wls"}:
        fit_req = "ols"
    sigma_tn = None
    fit_eff = fit_req
    if fit_req == "wls":
        if sigma_k_tn is not None:
            sigma_tn = np.asarray(sigma_k_tn, dtype=float)
            if sigma_tn.shape != (thr.size, nqs_obs.size):
                sigma_tn = None
        if sigma_tn is None:
            fit_eff = "ols"

    C_boot, B_boot = _fit_step4_bootstrap_coefficients(
        btn,
        nqs_obs,
        min_points=3,
        fit_mode=fit_eff,
        sigma_k_tn=sigma_tn,
    )
    B = int(C_boot.shape[0])
    M = int(nq_arr.size)
    log2_boot = np.full((B, M), np.nan, dtype=float)

    for j, nq in enumerate(nq_arr):
        target = _to_float(target_by_nq.get(int(nq), float("nan")))
        if not _is_finite(target):
            continue
        K = C_boot + (B_boot / float(nq))
        if bool(enforce_threshold_monotone):
            K = np.maximum.accumulate(K, axis=1)
        k_pred_b = _interp_bootstrap_rows_at_threshold(thr, K, target)
        log2_boot[:, j] = float(nq) * k_pred_b

    out: Dict[int, dict] = {}
    counts = np.sum(np.isfinite(log2_boot), axis=0).astype(int)
    center = np.full(M, np.nan, dtype=float)
    lo = np.full(M, np.nan, dtype=float)
    hi = np.full(M, np.nan, dtype=float)

    for j in range(M):
        vals = log2_boot[:, j]
        vals = vals[np.isfinite(vals)]
        if int(vals.size) < int(min_valid):
            continue
        center[j] = float(np.nanmedian(vals))
        lo[j] = float(np.nanpercentile(vals, q_lo))
        hi[j] = float(np.nanpercentile(vals, q_hi))

    simultaneous_applied = False
    if bool(simultaneous):
        dev = np.abs(log2_boot - center[None, :])
        dev[~np.isfinite(log2_boot)] = np.nan
        valid_rep = np.any(np.isfinite(dev), axis=1)
        max_dev = np.full(B, np.nan, dtype=float)
        if np.any(valid_rep):
            max_dev[valid_rep] = np.nanmax(dev[valid_rep], axis=1)
        max_dev = max_dev[np.isfinite(max_dev)]
        if int(max_dev.size) >= int(min_valid):
            q_sup = float(np.nanpercentile(max_dev, 100.0 * C_cov))
            good_center = np.isfinite(center)
            lo[good_center] = center[good_center] - q_sup
            hi[good_center] = center[good_center] + q_sup
            simultaneous_applied = True

    for j, nq in enumerate(nq_arr):
        l2_lo = _to_float(lo[j])
        l2_hi = _to_float(hi[j])
        l2_center = _to_float(center[j])
        if not (_is_finite(l2_lo) and _is_finite(l2_hi)):
            continue
        out[int(nq)] = {
            "log2_center": float(l2_center) if _is_finite(l2_center) else float("nan"),
            "log2_lo": float(l2_lo),
            "log2_hi": float(l2_hi),
            "ci_halfwidth_log2": float(0.5 * (l2_hi - l2_lo)),
            "n_valid": int(counts[j]),
        }

    diag = {
        "n_boot": int(B),
        "n_thresholds": int(thr.size),
        "n_obs_nq": int(nqs_obs.size),
        "n_pred_nq": int(M),
        "coverage": float(C_cov),
        "min_valid": int(min_valid),
        "simultaneous": bool(simultaneous),
        "simultaneous_applied": bool(simultaneous_applied),
        "enforce_threshold_monotone": bool(enforce_threshold_monotone),
        "n_points_with_band": int(len(out)),
        "fit_mode_requested": str(fit_req),
        "fit_mode_effective": str(fit_eff),
    }
    return out, diag


def _apply_curve_bootstrap_intervals_to_rows(
    *,
    rows: List[dict],
    qdata: dict,
    devices: List[str],
    readout_errors: List[str],
    channels: List[str],
    amplitudes: List[str],
    etas: List[float],
    nq_pred: np.ndarray,
    case_surfaces: Dict[str, Dict[Tuple[str, str], dict]],
    target_floor: float,
    ci_z: float,
    enforce_threshold_monotone: bool,
    simultaneous: bool,
    min_valid: int,
    fit_mode: str,
) -> dict:
    """
    Replace pointwise predictor CIs on `ok` rows with full-pipeline bootstrap curve bands.
    """
    idx: Dict[Tuple[str, str, str, float, int, str, str], dict] = {}
    for r in rows:
        key = (
            str(r.get("channel")),
            str(r.get("amplitude")),
            str(r.get("method")),
            float(r.get("eta")),
            int(r.get("nq")),
            str(r.get("device")),
            str(r.get("readout_error")),
        )
        idx[key] = r

    zf = abs(_to_float(ci_z))
    coverage = _central_coverage_from_z(zf if _is_finite(zf) and zf > 0 else 1.0)
    updated = 0
    total_candidates = 0
    case_diags: List[dict] = []
    source_tag = "curve_bootstrap_simultaneous" if bool(simultaneous) else "curve_bootstrap_pointwise"

    for method, case_map in case_surfaces.items():
        for (ch, amp), surf in sorted(case_map.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            thr = np.asarray(surf.get("thresholds"), dtype=float)
            nqs_obs = np.asarray(surf.get("nq_values"), dtype=int)
            btn = np.asarray(surf.get("kx_boot_btn"), dtype=float)
            sigma_tn = np.asarray(surf.get("sigma_k_tn"), dtype=float) if surf.get("sigma_k_tn") is not None else None
            if btn.ndim != 3 or btn.shape[0] < 2:
                continue

            for device in devices:
                for re in readout_errors:
                    for eta in etas:
                        target_by_nq: Dict[int, float] = {}
                        for nq in np.asarray(nq_pred, dtype=int):
                            q_acc = _get_quantum_acc(qdata, device, ch, amp, re, int(nq))
                            target = (q_acc - float(eta)) if _is_finite(q_acc) else float("nan")
                            # Keep full target path for bootstrap interpolation; final row update
                            # still only applies to status==ok points.
                            if _is_finite(target):
                                target_by_nq[int(nq)] = float(target)
                        if not target_by_nq:
                            continue

                        bands, bdiag = _curve_bootstrap_log2_bands(
                            thresholds=thr,
                            nq_values=nqs_obs,
                            kx_boot_btn=btn,
                            target_by_nq=target_by_nq,
                            nq_pred=np.asarray(nq_pred, dtype=int),
                            coverage=coverage,
                            enforce_threshold_monotone=bool(enforce_threshold_monotone),
                            simultaneous=bool(simultaneous),
                            min_valid=int(min_valid),
                            fit_mode=str(fit_mode),
                            sigma_k_tn=sigma_tn,
                        )
                        case_diags.append({
                            "method": str(method),
                            "channel": str(ch),
                            "amplitude": str(amp),
                            "device": str(device),
                            "readout_error": str(re),
                            "eta": float(eta),
                            **bdiag,
                        })

                        for nq in np.asarray(nq_pred, dtype=int):
                            key = (str(ch), str(amp), str(method), float(eta), int(nq), str(device), str(re))
                            row = idx.get(key)
                            if row is None:
                                continue
                            if str(row.get("pred_status", "")) != "ok":
                                continue
                            total_candidates += 1
                            band = bands.get(int(nq))
                            if not isinstance(band, dict):
                                continue
                            l2_lo = _to_float(band.get("log2_lo", float("nan")))
                            l2_hi = _to_float(band.get("log2_hi", float("nan")))
                            hw = _to_float(band.get("ci_halfwidth_log2", float("nan")))
                            nv = int(_to_float(band.get("n_valid", 0))) if _is_finite(band.get("n_valid", 0)) else 0
                            if not (_is_finite(l2_lo) and _is_finite(l2_hi) and _is_finite(hw) and hw >= 0):
                                continue
                            row["log2_nps_lo"] = float(l2_lo)
                            row["log2_nps_hi"] = float(l2_hi)
                            row["nps_lo"] = _safe_nps_from_log2(l2_lo)
                            row["nps_hi"] = _safe_nps_from_log2(l2_hi)
                            row["ci_halfwidth_log2_nps"] = float(hw)
                            if _is_finite(zf) and zf > 0:
                                sig = float(hw / zf)
                                row["sigma_log2_nps"] = sig
                                row["sigma_eff_log2_nps"] = sig
                            fit_eff = str(bdiag.get("fit_mode_effective", fit_mode))
                            row["interval_uncertainty_source"] = f"{source_tag}_{fit_eff}"
                            row["curve_bootstrap_n_valid"] = int(nv)
                            row["curve_bootstrap_fit_mode"] = fit_eff
                            updated += 1

    fit_counts = Counter(str(cd.get("fit_mode_effective", "")) for cd in case_diags)
    return {
        "requested": True,
        "coverage": float(coverage),
        "simultaneous": bool(simultaneous),
        "min_valid": int(min_valid),
        "fit_mode_requested": str(fit_mode),
        "fit_mode_effective_counts": {str(k): int(v) for k, v in fit_counts.items()},
        "total_ok_candidates": int(total_candidates),
        "n_rows_updated": int(updated),
        "n_case_runs": int(len(case_diags)),
        "case_runs": case_diags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Curve monotonization and threshold inversion
# (same logic as shadow module; duplicated here to keep file self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _monotone_curve(y: np.ndarray) -> np.ndarray:
    """NaN-safe isotonic monotonicization with robust fallback."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    x = np.arange(y.size, dtype=float)
    m = np.isfinite(y)
    if int(np.sum(m)) < 2:
        return np.full_like(y, np.nan, dtype=float)
    if _HAS_ISOTONIC:
        try:
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            y_fit = np.full_like(y, np.nan, dtype=float)
            y_fit[m] = np.asarray(ir.fit_transform(x[m], y[m]), dtype=float)
            y_fill = np.interp(x, x[m], y_fit[m])
            return np.maximum.accumulate(y_fill)
        except Exception:
            pass
    # NaN-safe cummax fallback.
    y2 = np.array(y, copy=True, dtype=float)
    y2[~m] = -np.inf
    y2 = np.maximum.accumulate(y2)
    y2[~np.isfinite(y2)] = np.nan
    return y2


def _invert_threshold_on_curve(
    k_grid: np.ndarray, y_mon: np.ndarray, threshold: float
) -> Tuple[str, float]:
    """
    Linear interpolation of threshold crossing on a monotone-increasing curve.

    Returns (status, k_x) where status ∈ {'ok', 'left_censored', 'right_censored', 'invalid'}.
      ok            – threshold crossed between grid points; k_x is linearly interpolated.
      left_censored – curve already ≥ threshold at the smallest k; k_x is the left boundary.
      right_censored– curve never reaches threshold; k_x is the right boundary (lower bound on true k).
    """
    k_grid = np.asarray(k_grid, dtype=float)
    y_mon  = np.asarray(y_mon,  dtype=float)
    X = float(threshold)
    valid = np.isfinite(k_grid) & np.isfinite(y_mon)
    k_grid, y_mon = k_grid[valid], y_mon[valid]
    if k_grid.size < 2:
        return "invalid", float("nan")
    order = np.argsort(k_grid)
    k_grid, y_mon = k_grid[order], y_mon[order]
    y_mon = np.maximum.accumulate(y_mon)   # enforce monotonicity after sort

    if y_mon[0] >= X:
        return "left_censored", float(k_grid[0])
    if y_mon[-1] < X:
        return "right_censored", float(k_grid[-1])

    j = int(np.argmax(y_mon >= X))
    if j == 0:
        return "ok", float(k_grid[0])
    y0, y1 = float(y_mon[j - 1]), float(y_mon[j])
    k0, k1 = float(k_grid[j - 1]), float(k_grid[j])
    if y1 <= y0:
        return "ok", float(k1)
    frac = (X - y0) / (y1 - y0)
    return "ok", float(k0 + frac * (k1 - k0))


# ─────────────────────────────────────────────────────────────────────────────
# Step4 model: k_x = C + β/nq
# ─────────────────────────────────────────────────────────────────────────────

def _step4_design(n_q: np.ndarray) -> np.ndarray:
    n_q = np.asarray(n_q, dtype=float)
    return np.column_stack([np.ones_like(n_q), 1.0 / n_q])


def _weighted_linear_fit(n_q, y, sigma) -> dict:
    """
    WLS for y = [C, β] · [1, 1/nq]^T.

    Returns dict with keys ok_fit, coef, cov_beta, n_points, cond, reason.
    cov_beta is the (2,2) WLS precision-matrix inverse: Cov(β̂) = (X^T W X)^{-1}.
    For propagation: Var(log2_nps) = Var(nq·k_x) = g^T Cov g  where g = [nq, 1]^T.
    """
    n_q   = np.asarray(n_q,   dtype=float)
    y     = np.asarray(y,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = (
        np.isfinite(n_q) & np.isfinite(y) & np.isfinite(sigma)
        & (sigma > 0) & (n_q > 0)
    )
    n_q, y, sigma = n_q[valid], y[valid], sigma[valid]
    if n_q.size < 3:
        return {"ok_fit": False, "reason": "insufficient_points"}
    X  = _step4_design(n_q)
    w  = 1.0 / np.maximum(sigma, MIN_SIGMA_K)
    Xw = X * w[:, None]
    yw = y * w
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return {"ok_fit": False, "reason": "lstsq_failed"}
    try:
        cov_beta = np.linalg.pinv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov_beta = None
    return {
        "ok_fit":   True,
        "coef":     np.asarray(beta, dtype=float),
        "cov_beta": cov_beta,
        "n_points": int(n_q.size),
        "cond":     float(np.linalg.cond(Xw)),
    }


def _step4_predict(n_q, coef: np.ndarray) -> np.ndarray:
    arr = np.asarray(n_q, dtype=float)
    return (_step4_design(arr.ravel()) @ coef).reshape(arr.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_interp(x: np.ndarray, y: np.ndarray, extrapolate: bool = False):
    """Return a callable that interpolates y(x). Uses PCHIP if available."""
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
        return lambda z, _c=c: np.full_like(np.asarray(z, dtype=float), _c)
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(x, y, extrapolate=extrapolate)
        return lambda z: np.asarray(pchip(np.asarray(z, dtype=float)), dtype=float)
    except Exception:
        def _f(z):
            z_arr = np.asarray(z, dtype=float)
            kwargs = {} if extrapolate else {"left": float("nan"), "right": float("nan")}
            return np.interp(z_arr, x, y, **kwargs)
        return _f


# ─────────────────────────────────────────────────────────────────────────────
# Predictor: opt1 dense threshold inversion
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Dense-threshold Step4 predictor.

    predict_log2_nps(threshold, nq)      → log2(n_ps)
    predict_nps(threshold, nq)           → n_ps
    predict_sigma_log2_nps(threshold, nq)→ 1σ parametric WLS uncertainty in log2 space
                                           (from covariance propagation; tends to underestimate)
    predict_sigma_eff(threshold, nq)     → combined 1σ-equivalent in log2 space:
                                           sqrt(σ_WLS² + (nq·σ_CV,k(threshold))²)
    predict_ci_halfwidth_log2(..., ci_z) → CI half-width using:
                                           sqrt((z·σ_WLS)² + (nq·q_CV,|e_k|(threshold,C))²),
                                           where C = 2Φ(z)-1 and q_CV,|e_k| is empirical
                                           CV quantile of |e_k| at coverage C.

    Censoring:
      threshold < threshold_min  → left_censored  (target too easy; nps is an upper bound)
      threshold > threshold_max  → right_censored (target too hard; nps is a lower bound)
      Use _predict_point() to get status-annotated predictions.
    """
    def __init__(self, *, name, channel, p, threshold_min, threshold_max,
             predict_k_x, predict_log2_nps, predict_nps,
             predict_sigma_log2_nps=None,
             sigma_cv_k: float = 0.0,
             sigma_cv_k_obs: Optional[float] = None,
             sigma_cv_k_extrap: Optional[float] = None,
             n_q_obs_max: int = 0,
             cv_obs_quantile_anchors: Optional[List[Tuple[float, float]]] = None,
             predict_sigma_cv_k=None,
             predict_cv_abs_quantile_k=None,
             interval_method: str = "cv",
             conformal_quantile_anchors: Optional[List[Tuple[float, float]]] = None,
             meta=None):
        self.name           = name
        self.channel        = channel
        self.p              = p
        self.threshold_min  = float(threshold_min)
        self.threshold_max  = float(threshold_max)
        self.predict_k_x    = predict_k_x
        self.predict_log2_nps          = predict_log2_nps
        self.predict_nps               = predict_nps
        self.predict_sigma_log2_nps    = predict_sigma_log2_nps
        self.predict_sigma_cv_k        = predict_sigma_cv_k
        self.predict_cv_abs_quantile_k = predict_cv_abs_quantile_k
        sc_k = float(sigma_cv_k)
        self.sigma_cv_k = sc_k if math.isfinite(sc_k) and sc_k >= 0 else 0.0
        sc_obs = _to_float(sigma_cv_k_obs if sigma_cv_k_obs is not None else self.sigma_cv_k)
        sc_ext = _to_float(sigma_cv_k_extrap if sigma_cv_k_extrap is not None else self.sigma_cv_k)
        self.sigma_cv_k_obs = sc_obs if _is_finite(sc_obs) and sc_obs >= 0 else self.sigma_cv_k
        self.sigma_cv_k_extrap = sc_ext if _is_finite(sc_ext) and sc_ext >= 0 else self.sigma_cv_k
        self.n_q_obs_max = int(n_q_obs_max) if int(n_q_obs_max) > 0 else 0
        self.cv_obs_quantile_anchors = list(cv_obs_quantile_anchors or [])
        self.interval_method = str(interval_method or "cv").strip().lower()
        if self.interval_method not in {"cv", "conformal", "conformal_hybrid"}:
            self.interval_method = "cv"
        self.conformal_quantile_anchors = list(conformal_quantile_anchors or [])
        # Back-compat fields filled by builder using a representative nq.
        self.sigma_cv = 0.0
        self.sigma_cv_log2_at_obs_max = 0.0
        self.meta = meta or {}

    def _sigma_cv_k_at(self, X, n_q=None) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        out = np.full_like(Xa, float(self.sigma_cv_k_extrap), dtype=float)
        Na = None
        if n_q is not None:
            Na = np.asarray(np.broadcast_arrays(Xa, np.asarray(n_q, dtype=float))[1], dtype=float)
        if callable(self.predict_sigma_cv_k):
            sig = None
            try:
                if Na is not None:
                    sig = np.asarray(self.predict_sigma_cv_k(Xa, Na), dtype=float)
                else:
                    sig = np.asarray(self.predict_sigma_cv_k(Xa), dtype=float)
            except TypeError:
                sig = np.asarray(self.predict_sigma_cv_k(Xa), dtype=float)
            except Exception:
                sig = None
            if sig is None:
                sig = np.full_like(out, float("nan"), dtype=float)
            if sig.shape != out.shape:
                try:
                    sig = np.broadcast_to(sig, out.shape)
                except Exception:
                    sig = np.full_like(out, float("nan"), dtype=float)
            good = np.isfinite(sig) & (sig >= 0)
            out = np.where(good, sig, out)
        out = np.where(np.isfinite(out) & (out >= 0), out, float(self.sigma_cv_k_extrap))
        if Na is not None and self.n_q_obs_max > 0:
            obs_mask = np.isfinite(Na) & (Na <= float(self.n_q_obs_max))
            out = np.where(obs_mask, float(self.sigma_cv_k_obs), out)
        return out

    def _cv_abs_quantile_k_at(self, X, coverage: float, n_q=None) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        Na = None
        if n_q is not None:
            Na = np.asarray(np.broadcast_arrays(Xa, np.asarray(n_q, dtype=float))[1], dtype=float)
        c = _to_float(coverage)
        if not _is_finite(c):
            c = 0.6827
        c = float(min(max(c, 1e-6), 0.999999))
        if callable(self.predict_cv_abs_quantile_k):
            try:
                try:
                    if Na is not None:
                        q = np.asarray(self.predict_cv_abs_quantile_k(Xa, c, Na), dtype=float)
                    else:
                        q = np.asarray(self.predict_cv_abs_quantile_k(Xa, c), dtype=float)
                except TypeError:
                    q = np.asarray(self.predict_cv_abs_quantile_k(Xa, c), dtype=float)
                if q.shape != Xa.shape:
                    q = np.broadcast_to(q, Xa.shape)
                good = np.isfinite(q) & (q >= 0)
                if np.any(good):
                    fallback = self._sigma_cv_k_at(Xa, Na) * _to_float(NormalDist().inv_cdf(0.5 + 0.5 * c))
                    out = np.where(good, q, fallback)
                    if Na is not None and self.n_q_obs_max > 0:
                        obs_mask = np.isfinite(Na) & (Na <= float(self.n_q_obs_max))
                        q_obs = _empirical_abs_quantile_from_anchors(c, self.cv_obs_quantile_anchors)
                        if not (_is_finite(q_obs) and q_obs >= 0):
                            q_obs = float(self.sigma_cv_k_obs) * _to_float(NormalDist().inv_cdf(0.5 + 0.5 * c))
                        out = np.where(obs_mask, q_obs, out)
                    return out
            except Exception:
                pass
        z = _to_float(NormalDist().inv_cdf(0.5 + 0.5 * c))
        if not _is_finite(z) or z <= 0:
            z = float(Z68_ABS)
        out = self._sigma_cv_k_at(Xa, Na) * float(z)
        if Na is not None and self.n_q_obs_max > 0:
            obs_mask = np.isfinite(Na) & (Na <= float(self.n_q_obs_max))
            q_obs = _empirical_abs_quantile_from_anchors(c, self.cv_obs_quantile_anchors)
            if not (_is_finite(q_obs) and q_obs >= 0):
                q_obs = float(self.sigma_cv_k_obs) * float(z)
            out = np.where(obs_mask, q_obs, out)
        return out

    def _conformal_abs_quantile_k(self, coverage: float) -> float:
        """Return pooled conformal quantile of |e_k| at central coverage."""
        c = _to_float(coverage)
        if not _is_finite(c):
            c = 0.6827
        c = float(min(max(c, 1e-6), 0.999999))
        q = _empirical_abs_quantile_from_anchors(c, self.conformal_quantile_anchors)
        return float(q) if _is_finite(q) and q >= 0 else float("nan")

    def _conformal_abs_quantile_k_at(self, X, coverage: float, n_q=None) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        q = self._conformal_abs_quantile_k(coverage)
        out = np.full_like(Xa, float("nan"), dtype=float)
        if _is_finite(q) and q >= 0:
            out[...] = float(q)
        return out

    def predict_sigma_eff(self, X, n_q) -> np.ndarray:
        """
        Combined 1σ in log2(nps): sqrt(σ_WLS²(X,nq) + σ_CV²).

        σ_WLS: parametric WLS covariance propagation.
        σ_CV,k: empirical model-discrepancy floor in k-space from forward-CV.
                Converted to log2-space at prediction time via nq·σ_CV,k.
        If predict_sigma_log2_nps is unavailable, returns σ_CV alone (as a scalar array).
        """
        Xa, Na = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        sigma_cv_k_arr = self._sigma_cv_k_at(Xa, Na)
        sigma_cv_log2_arr = np.abs(Na) * np.maximum(sigma_cv_k_arr, 0.0)
        if self.predict_sigma_log2_nps is None:
            out = np.where(np.isfinite(sigma_cv_log2_arr), np.maximum(sigma_cv_log2_arr, 0.0), 0.0)
            return np.asarray(out, dtype=float)
        sig_wls = np.asarray(self.predict_sigma_log2_nps(Xa, Na), dtype=float)
        sig_eff = np.where(
            np.isfinite(sig_wls),
            np.sqrt(np.maximum(sig_wls, 0.0) ** 2 + np.maximum(sigma_cv_log2_arr, 0.0) ** 2),
            np.maximum(sigma_cv_log2_arr, 0.0),
        )
        sig_eff = np.where(np.isfinite(sig_eff), sig_eff, 0.0)
        return sig_eff

    def predict_ci_halfwidth_log2(self, X, n_q, ci_z: float = 1.0) -> np.ndarray:
        """
        CI half-width in log2(nps).

        Methods:
          • cv:            sqrt((z·σ_WLS)^2 + (nq·q_CV,|e_k|(C))^2),  C=2Φ(z)-1.
          • conformal:     nq·q_conf,|e_k|(C) from held-out nq calibration (pooled).
          • conformal_hybrid:
                use conformal within observed nq (≤ n_q_obs_max), fall back to cv for extrapolation.
        """
        Xa, Na = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        z = abs(_to_float(ci_z))
        if not _is_finite(z):
            z = 1.0
        c = _central_coverage_from_z(z)

        # ---- CV-style (WLS + forward-CV quantiles) halfwidth
        hw_wls = np.full_like(Xa, float("nan"), dtype=float)
        if self.predict_sigma_log2_nps is not None:
            sig_wls = np.asarray(self.predict_sigma_log2_nps(Xa, Na), dtype=float)
            hw_wls = np.where(np.isfinite(sig_wls) & (sig_wls >= 0), z * sig_wls, float("nan"))

        hw_cv_k = np.asarray(self._cv_abs_quantile_k_at(Xa, c, Na), dtype=float)
        hw_cv_k = np.where(np.isfinite(hw_cv_k) & (hw_cv_k >= 0), hw_cv_k, float("nan"))
        hw_cv = np.abs(Na) * hw_cv_k

        hw_cv_out = np.full_like(Xa, float("nan"), dtype=float)
        both = np.isfinite(hw_wls) & np.isfinite(hw_cv)
        only_wls = np.isfinite(hw_wls) & ~np.isfinite(hw_cv)
        only_cv = ~np.isfinite(hw_wls) & np.isfinite(hw_cv)
        hw_cv_out[both] = np.sqrt(hw_wls[both] ** 2 + hw_cv[both] ** 2)
        hw_cv_out[only_wls] = hw_wls[only_wls]
        hw_cv_out[only_cv] = hw_cv[only_cv]

        # ---- Conformal pooled halfwidth in k-space, mapped to log2 via nq.
        q_conf = self._conformal_abs_quantile_k(c)
        hw_conf = np.full_like(Xa, float("nan"), dtype=float)
        if _is_finite(q_conf) and q_conf >= 0:
            hw_conf = np.abs(Na) * float(q_conf)

        mth = str(getattr(self, "interval_method", "cv") or "cv").strip().lower()
        if mth == "conformal":
            return np.where(np.isfinite(hw_conf), hw_conf, hw_cv_out)

        if mth == "conformal_hybrid" and int(getattr(self, "n_q_obs_max", 0)) > 0:
            obs_mask = np.isfinite(Na) & (Na <= float(self.n_q_obs_max))
            out = np.array(hw_cv_out, copy=True)
            out = np.where(obs_mask & np.isfinite(hw_conf), hw_conf, out)
            return out

        # Default: cv
        return hw_cv_out



# ─────────────────────────────────────────────────────────────────────────────
# Split-conformal-like calibration on held-out nq
# ─────────────────────────────────────────────────────────────────────────────

def _split_train_holdout_nq_from_rows(case_rows: List[dict], holdout_count: int) -> dict:
    """
    Build a right-edge nq split used by held-out calibration.

    Returns:
      {
        ok: bool,
        reason: str,
        nqs: List[int],
        train_nq: List[int],
        holdout_nq: List[int],
      }
    """
    out = {
        "ok": False,
        "reason": "not_run",
        "nqs": [],
        "train_nq": [],
        "holdout_nq": [],
    }
    ok_rows = [r for r in case_rows if _is_inference_row(r) and _is_finite(r.get("k_x"))]
    if not ok_rows:
        out["reason"] = "no_ok_rows"
        return out
    nqs = sorted({int(r["n_q"]) for r in ok_rows if _is_finite(r.get("n_q"))})
    out["nqs"] = [int(v) for v in nqs]
    h = int(holdout_count)
    if h < 1:
        out["reason"] = "holdout_count<1"
        return out
    min_required = max(4, int(h) + 3)
    if len(nqs) < min_required:
        out["reason"] = f"insufficient_nq:{len(nqs)}<{min_required}"
        return out
    holdout_nq = nqs[-h:]
    train_nq = nqs[:-h]
    if len(train_nq) < 3:
        out["reason"] = f"insufficient_train_nq:{len(train_nq)}"
        return out
    out["ok"] = True
    out["reason"] = "ok"
    out["train_nq"] = [int(v) for v in train_nq]
    out["holdout_nq"] = [int(v) for v in holdout_nq]
    return out

def _finite_sample_conformal_quantile(abs_errors: np.ndarray, coverage: float) -> float:
    """Finite-sample conformal quantile for |error| at target central coverage."""
    if abs_errors is None:
        return float("nan")
    e = np.asarray(abs_errors, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    c = _to_float(coverage)
    if not _is_finite(c):
        c = 0.6827
    c = float(min(max(c, 1e-6), 0.999999))
    e_sort = np.sort(e)
    # Standard split-conformal conservative index:
    k = int(math.ceil((e_sort.size + 1) * c)) - 1
    k = max(0, min(k, e_sort.size - 1))
    return float(e_sort[k])


def _split_conformal_calibrate_abs_errors_k(
    case_rows: List[dict],
    holdout_count: int,
    pct_min: float,
    pct_max: float,
    min_abs_errors: int,
    z_list: Optional[List[float]] = None,
) -> dict:
    """
    Fit Step4 on training nq and evaluate on held-out largest nq values to obtain pooled |e_k| residuals.

    Returns dict with:
      ok, reason, holdout_nq, n_abs_errors, thresholds_used,
      anchors (q68/q90/q95), and coverage diagnostics under 'heldout_coverage_abs_ek'.
    """
    out = {
        "ok": False,
        "reason": "not_run",
        "holdout_nq": [],
        "n_abs_errors": 0,
        "thresholds_used": 0,
        "anchors": [],
        "heldout_coverage_abs_ek": {},
    }
    if not case_rows:
        out["reason"] = "empty_case_rows"
        return out

    split = _split_train_holdout_nq_from_rows(case_rows, holdout_count=int(holdout_count))
    if not bool(split.get("ok", False)):
        out["reason"] = str(split.get("reason", "split_failed"))
        out["holdout_nq"] = [int(v) for v in split.get("holdout_nq", [])]
        return out
    train_nq_set = set(int(v) for v in split.get("train_nq", []))
    holdout_nq = [int(v) for v in split.get("holdout_nq", [])]
    out["holdout_nq"] = [int(x) for x in holdout_nq]
    ok_rows = [r for r in case_rows if _is_inference_row(r) and _is_finite(r.get("k_x"))]

    # Threshold selection window
    thrs = sorted({_thr_key(float(r["threshold"])) for r in ok_rows if _is_finite(r.get("threshold"))})
    if len(thrs) < 4:
        out["reason"] = "insufficient_thresholds"
        return out
    lo = float(np.nanpercentile(np.asarray(thrs, dtype=float), float(pct_min)))
    hi = float(np.nanpercentile(np.asarray(thrs, dtype=float), float(pct_max)))
    thr_sel = [t for t in thrs if (t >= lo and t <= hi)]
    if len(thr_sel) < 2:
        out["reason"] = "threshold_window_too_small"
        return out

    by_thr: Dict[float, List[dict]] = {}
    for r in ok_rows:
        thr = _thr_key(float(r["threshold"]))
        if thr in by_thr:
            by_thr[thr].append(r)
        else:
            by_thr[thr] = [r]

    abs_errors: List[float] = []
    n_used_thr = 0

    for thr in thr_sel:
        rows_t = by_thr.get(_thr_key(float(thr)), [])
        if not rows_t:
            continue
        # Separate train/holdout points
        train_pts = [rr for rr in rows_t if int(rr["n_q"]) in train_nq_set and _is_finite(rr.get("sigma_k"))]
        hold_pts = [rr for rr in rows_t if int(rr["n_q"]) in holdout_nq and _is_finite(rr.get("sigma_k"))]
        if len(train_pts) < 3 or len(hold_pts) < 1:
            continue
        nq_tr = np.array([int(rr["n_q"]) for rr in train_pts], dtype=float)
        k_tr  = np.array([float(rr["k_x"]) for rr in train_pts], dtype=float)
        s_tr  = np.array([max(float(rr.get("sigma_k", MIN_SIGMA_K)), MIN_SIGMA_K) for rr in train_pts], dtype=float)
        fit = _weighted_linear_fit(nq_tr, k_tr, s_tr)
        if not fit.get("ok_fit"):
            continue
        C, beta = float(fit["coef"][0]), float(fit["coef"][1])
        n_used_thr += 1
        for rr in hold_pts:
            nq_h = float(rr["n_q"])
            k_obs = float(rr["k_x"])
            if not (_is_finite(nq_h) and nq_h > 0 and _is_finite(k_obs)):
                continue
            k_pred = C + beta / nq_h
            if _is_finite(k_pred):
                abs_errors.append(abs(k_obs - k_pred))

    out["thresholds_used"] = int(n_used_thr)
    out["n_abs_errors"] = int(len(abs_errors))
    if len(abs_errors) < int(min_abs_errors):
        out["reason"] = f"too_few_abs_errors:{len(abs_errors)}<{int(min_abs_errors)}"
        return out

    e = np.asarray(abs_errors, dtype=float)
    out["n_abs_errors"] = int(e.size)

    q68 = _finite_sample_conformal_quantile(e, 0.6827)
    q90 = _finite_sample_conformal_quantile(e, 0.90)
    q95 = _finite_sample_conformal_quantile(e, 0.95)
    anchors = []
    if _is_finite(q68) and q68 >= 0:
        anchors.append((0.6827, float(q68)))
    if _is_finite(q90) and q90 >= 0:
        anchors.append((0.90, float(q90)))
    if _is_finite(q95) and q95 >= 0:
        anchors.append((0.95, float(q95)))
    out["anchors"] = anchors

    # Coverage diagnostics (in k-space; equivalent to log2 after nq scaling).
    if z_list is None:
        z_list = [float(z) for z in CV_COVERAGE_ZS_DEFAULT]
    cov_diag = {}
    for z in z_list:
        zf = abs(_to_float(z))
        if not (_is_finite(zf) and zf > 0):
            continue
        nom = float(_central_coverage_from_z(zf))
        q = _finite_sample_conformal_quantile(e, nom)
        obs = float(np.mean(e <= q)) if _is_finite(q) else float("nan")
        cov_diag[f"{zf:.6g}"] = {"nominal": nom, "observed": obs, "q_abs_k": float(q) if _is_finite(q) else float("nan")}
    out["heldout_coverage_abs_ek"] = cov_diag

    out["ok"] = True
    out["reason"] = "ok"
    return out


def _strict_split_conformal_calibrate_abs_errors_k(
    case_rows: List[dict],
    predictor: Predictor,
    holdout_nq: List[int],
    pct_min: float,
    pct_max: float,
    min_abs_errors: int,
    z_list: Optional[List[float]] = None,
) -> dict:
    """
    Strict split calibration using residuals from the SAME predictor object.
    """
    out = {
        "ok": False,
        "reason": "not_run",
        "holdout_nq": [int(v) for v in holdout_nq],
        "n_abs_errors": 0,
        "thresholds_used": 0,
        "anchors": [],
        "heldout_coverage_abs_ek": {},
        "calibration_model": "predictor_consistent_holdout",
    }
    if predictor is None:
        out["reason"] = "missing_predictor"
        return out
    hold_set = set(int(v) for v in holdout_nq)
    ok_rows = [r for r in case_rows if _is_inference_row(r) and _is_finite(r.get("k_x"))]
    if not ok_rows:
        out["reason"] = "no_ok_rows"
        return out
    hold_rows = [r for r in ok_rows if int(r.get("n_q", -1)) in hold_set]
    if not hold_rows:
        out["reason"] = "no_holdout_rows"
        return out

    thrs_all = sorted({_thr_key(float(r["threshold"])) for r in ok_rows if _is_finite(r.get("threshold"))})
    if len(thrs_all) < 4:
        out["reason"] = "insufficient_thresholds"
        return out
    lo = float(np.nanpercentile(np.asarray(thrs_all, dtype=float), float(pct_min)))
    hi = float(np.nanpercentile(np.asarray(thrs_all, dtype=float), float(pct_max)))
    thr_sel = {t for t in thrs_all if (t >= lo and t <= hi)}
    if len(thr_sel) < 2:
        out["reason"] = "threshold_window_too_small"
        return out

    abs_errors: List[float] = []
    thr_used = set()
    for rr in hold_rows:
        thr = _thr_key(float(rr["threshold"]))
        if thr not in thr_sel:
            continue
        nq_h = int(rr["n_q"])
        k_obs = _to_float(rr.get("k_x", float("nan")))
        if not (_is_finite(k_obs) and nq_h > 0):
            continue
        k_pred_arr = np.asarray(predictor.predict_k_x(float(thr), int(nq_h)), dtype=float)
        k_pred = _scalarize_interp_value(k_pred_arr)
        if not _is_finite(k_pred):
            continue
        abs_errors.append(abs(float(k_obs) - float(k_pred)))
        thr_used.add(float(thr))

    out["thresholds_used"] = int(len(thr_used))
    out["n_abs_errors"] = int(len(abs_errors))
    if len(abs_errors) < int(min_abs_errors):
        out["reason"] = f"too_few_abs_errors:{len(abs_errors)}<{int(min_abs_errors)}"
        return out

    e = np.asarray(abs_errors, dtype=float)
    q68 = _finite_sample_conformal_quantile(e, 0.6827)
    q90 = _finite_sample_conformal_quantile(e, 0.90)
    q95 = _finite_sample_conformal_quantile(e, 0.95)
    anchors = []
    if _is_finite(q68) and q68 >= 0:
        anchors.append((0.6827, float(q68)))
    if _is_finite(q90) and q90 >= 0:
        anchors.append((0.90, float(q90)))
    if _is_finite(q95) and q95 >= 0:
        anchors.append((0.95, float(q95)))
    out["anchors"] = anchors

    if z_list is None:
        z_list = [float(z) for z in CV_COVERAGE_ZS_DEFAULT]
    cov_diag = {}
    for z in z_list:
        zf = abs(_to_float(z))
        if not (_is_finite(zf) and zf > 0):
            continue
        nom = float(_central_coverage_from_z(zf))
        q = _finite_sample_conformal_quantile(e, nom)
        obs = float(np.mean(e <= q)) if _is_finite(q) else float("nan")
        cov_diag[f"{zf:.6g}"] = {"nominal": nom, "observed": obs, "q_abs_k": float(q) if _is_finite(q) else float("nan")}
    out["heldout_coverage_abs_ek"] = cov_diag
    out["ok"] = True
    out["reason"] = "ok"
    return out


def _build_opt1_predictor(
    case_rows: List[dict],
    source_label: str,
    *,
    keep_cv_details: bool = False,
    extrap_min_horizons: int = EXTRAP_TRUST_MIN_HORIZONS,
    extrap_max_sigma_cv: float = EXTRAP_TRUST_MAX_SIGMA_CV,
    extrap_max_rmse_ratio: float = EXTRAP_TRUST_MAX_RMSE_RATIO,
    cv_thr_pct_min: float = CV_THR_PCT_MIN,
    cv_thr_pct_max: float = CV_THR_PCT_MAX,
    enforce_threshold_monotone: bool = ENFORCE_THRESHOLD_MONOTONE_DEFAULT,
    monotone_threshold_grid_size: int = MONOTONE_THRESHOLD_GRID_SIZE,
    sigma_cv_k_mode: str = SIGMA_CV_K_MODE_DEFAULT,
    sigma_cv_k_coverage: float = SIGMA_CV_K_COVERAGE_DEFAULT,
    cv_coverage_zs: Optional[List[float]] = None,
    interval_method: str = INTERVAL_METHOD_DEFAULT,
    conformal_holdout_count: int = CONFORMAL_HOLDOUT_COUNT_DEFAULT,
    conformal_min_abs_errors: int = CONFORMAL_MIN_ABS_ERRORS_DEFAULT,
    strict_split_conformal: bool = STRICT_SPLIT_CONFORMAL_DEFAULT,
) -> Optional[Predictor]:
    """
    Build an opt1 predictor from dense threshold bootstrap rows.

    Algorithm:
      1. Group rows by threshold value t.
      2. At each t, fit WLS: k_x(nq) = C(t) + β(t)/nq using sigma_k weights.
      3. Interpolate C(t) and β(t) across the threshold grid.
      4. Prediction at (threshold, nq): k_x = C(threshold) + β(threshold)/nq,
         log2_nps = nq · k_x.
      5. Parametric σ from propagating the WLS covariance: σ² = g^T Σ g, g=[nq,1].

    Requires ≥ 3 nq values per threshold, and ≥ 2 valid threshold fits.
    """
    if not case_rows:
        return None

    req_m_requested = str(interval_method or "cv").strip().lower()
    if req_m_requested not in {"cv", "conformal", "conformal_hybrid"}:
        req_m_requested = "cv"

    strict_split_requested = bool(strict_split_conformal) and req_m_requested != "cv"
    strict_split_reason = "not_requested" if not strict_split_requested else "fallback_to_default_fit"

    # Optional strict split-conformal:
    # fit the predictor on train nq only and calibrate on held-out nq.
    if strict_split_requested:
        split = _split_train_holdout_nq_from_rows(case_rows, holdout_count=int(conformal_holdout_count))
        if bool(split.get("ok", False)):
            train_set = set(int(v) for v in split.get("train_nq", []))
            train_rows = [r for r in case_rows if int(r.get("n_q", -1)) in train_set]
            strict_pred = _build_opt1_predictor(
                train_rows,
                source_label,
                keep_cv_details=keep_cv_details,
                extrap_min_horizons=extrap_min_horizons,
                extrap_max_sigma_cv=extrap_max_sigma_cv,
                extrap_max_rmse_ratio=extrap_max_rmse_ratio,
                cv_thr_pct_min=cv_thr_pct_min,
                cv_thr_pct_max=cv_thr_pct_max,
                enforce_threshold_monotone=enforce_threshold_monotone,
                monotone_threshold_grid_size=monotone_threshold_grid_size,
                sigma_cv_k_mode=sigma_cv_k_mode,
                sigma_cv_k_coverage=sigma_cv_k_coverage,
                cv_coverage_zs=cv_coverage_zs,
                interval_method="cv",
                conformal_holdout_count=conformal_holdout_count,
                conformal_min_abs_errors=conformal_min_abs_errors,
                strict_split_conformal=False,
            )
            if strict_pred is not None:
                z_list = [abs(_to_float(z)) for z in (cv_coverage_zs or CV_COVERAGE_ZS_DEFAULT)
                          if _is_finite(z) and abs(_to_float(z)) > 0]
                if not z_list:
                    z_list = [float(z) for z in CV_COVERAGE_ZS_DEFAULT]
                conf = _strict_split_conformal_calibrate_abs_errors_k(
                    case_rows,
                    predictor=strict_pred,
                    holdout_nq=[int(v) for v in split.get("holdout_nq", [])],
                    pct_min=float(cv_thr_pct_min),
                    pct_max=float(cv_thr_pct_max),
                    min_abs_errors=int(conformal_min_abs_errors),
                    z_list=z_list,
                )
                strict_pred.meta["strict_split_conformal_requested"] = True
                strict_pred.meta["strict_split_conformal_active"] = True
                strict_pred.meta["strict_split_reason"] = "ok"
                strict_pred.meta["strict_split_train_nq"] = [int(v) for v in split.get("train_nq", [])]
                strict_pred.meta["strict_split_holdout_nq"] = [int(v) for v in split.get("holdout_nq", [])]
                strict_pred.meta["interval_method_requested"] = str(req_m_requested)
                strict_pred.meta["conformal_holdout_count"] = int(conformal_holdout_count)
                strict_pred.meta["conformal_min_abs_errors"] = int(conformal_min_abs_errors)
                strict_pred.meta["conformal_calibration_mode"] = "predictor_consistent_holdout"
                if bool(conf.get("ok", False)) and conf.get("anchors"):
                    strict_pred.interval_method = str(req_m_requested)
                    strict_pred.conformal_quantile_anchors = list(conf["anchors"])
                    strict_pred.meta["conformal_calibration"] = conf
                    strict_pred.meta["interval_method"] = str(req_m_requested)
                else:
                    strict_pred.interval_method = "cv"
                    strict_pred.meta["interval_method"] = "cv"
                    strict_pred.meta["interval_method_fallback"] = {
                        "requested": str(req_m_requested),
                        "effective": "cv",
                        "reason": str(conf.get("reason", "unknown")),
                        "n_abs_errors": int(conf.get("n_abs_errors", 0)),
                    }
                strict_pred.meta["interval_method_effective"] = str(strict_pred.meta.get("interval_method", strict_pred.interval_method))
                return strict_pred
            strict_split_reason = "strict_train_fit_failed"
        else:
            strict_split_reason = str(split.get("reason", "strict_split_failed"))

    by_thr: Dict[float, List[dict]] = {}
    for r in case_rows:
        by_thr.setdefault(_thr_key(float(r["threshold"])), []).append(r)

    fit_rows = []
    for thr in sorted(by_thr):
        g   = sorted(by_thr[thr], key=lambda rr: int(rr["n_q"]))
        nq  = np.array([int(rr["n_q"]) for rr in g], dtype=float)
        kx  = np.array([float(rr["k_x"]) for rr in g], dtype=float)
        sig = np.array([float(rr["sigma_k"]) for rr in g], dtype=float)
        fit = _weighted_linear_fit(nq, kx, sig)
        if not fit["ok_fit"]:
            continue
        fit_rows.append({
            "threshold":   float(thr),
            "C":           float(fit["coef"][0]),
            "beta_inv_n":  float(fit["coef"][1]),
            "cov_beta":    fit.get("cov_beta"),
            "n_points":    int(fit["n_points"]),
            "cond":        float(fit["cond"]),
        })

    if len(fit_rows) < 2:
        return None

    thresholds = np.array([r["threshold"]  for r in fit_rows], dtype=float)
    C_vals     = np.array([r["C"]          for r in fit_rows], dtype=float)
    B_vals     = np.array([r["beta_inv_n"] for r in fit_rows], dtype=float)
    fC = _make_interp(thresholds, C_vals)
    fB = _make_interp(thresholds, B_vals)
    if fC is None or fB is None:
        return None

    # Interpolate covariance elements across thresholds (thresholdwise fallback).
    cov_support_counts: Dict[Tuple[int, int], int] = {}
    cov_interp: Dict[Tuple[int, int], any] = {}
    for i in range(2):
        for j in range(2):
            t_ij: List[float] = []
            v_ij: List[float] = []
            for rr in fit_rows:
                cov = rr.get("cov_beta", None)
                if not isinstance(cov, np.ndarray) or cov.shape != (2, 2):
                    continue
                val = _to_float(cov[i, j])
                if not _is_finite(val):
                    continue
                t_ij.append(float(rr["threshold"]))
                v_ij.append(float(val))
            cov_support_counts[(i, j)] = int(len(v_ij))
            if len(v_ij) >= 2:
                f = _make_interp(np.asarray(t_ij, dtype=float), np.asarray(v_ij, dtype=float))
                if f is not None:
                    cov_interp[(i, j)] = f
            elif len(v_ij) == 1:
                cval = float(v_ij[0])

                def _const_interp(x, _c=cval):
                    xa = np.asarray(x, dtype=float)
                    out = np.full_like(xa, float(_c), dtype=float)
                    return out

                cov_interp[(i, j)] = _const_interp
    have_cov = all((i, j) in cov_interp for i in range(2) for j in range(2))

    x_lo  = float(thresholds.min())
    x_hi  = float(thresholds.max())
    ch    = str(case_rows[0]["channel"])
    p_val = float(case_rows[0]["p"])
    n_q_obs_max = int(max(int(r.get("n_q", 0)) for r in case_rows)) if case_rows else 1
    if n_q_obs_max < 1:
        n_q_obs_max = 1

    def _predict_k_x_raw(X, n_q):
        Xa, Na = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(Xa, float("nan"), dtype=float)
        v = np.isfinite(Xa) & np.isfinite(Na) & (Na > 0) & (Xa >= x_lo) & (Xa <= x_hi)
        if v.any():
            out[v] = np.asarray(fC(Xa[v]), dtype=float) + np.asarray(fB(Xa[v]), dtype=float) / Na[v]
        return out

    # Optional Option-E monotonicity guard:
    # for each n_q, evaluate k_x(threshold, n_q) on a dense threshold grid,
    # enforce non-decreasing behavior via cummax, and interpolate back.
    use_monotone_threshold = bool(enforce_threshold_monotone)
    t_dense = np.linspace(x_lo, x_hi, max(64, int(monotone_threshold_grid_size)))
    mono_cache: Dict[float, tuple] = {}

    def _get_monotone_profile(nq_val: float):
        key = float(nq_val)
        prof = mono_cache.get(key)
        if prof is not None:
            return prof
        k_raw = np.asarray(fC(t_dense), dtype=float) + np.asarray(fB(t_dense), dtype=float) / float(key)
        valid = np.isfinite(t_dense) & np.isfinite(k_raw)
        if int(np.sum(valid)) < 2:
            prof = (None, None)
            mono_cache[key] = prof
            return prof
        t_v = np.asarray(t_dense[valid], dtype=float)
        k_v = np.asarray(k_raw[valid], dtype=float)
        k_mon = np.maximum.accumulate(k_v)
        prof = (t_v, k_mon)
        mono_cache[key] = prof
        return prof

    def predict_k_x(X, n_q):
        Xa, Na = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(Xa, float("nan"), dtype=float)
        v = np.isfinite(Xa) & np.isfinite(Na) & (Na > 0) & (Xa >= x_lo) & (Xa <= x_hi)
        if not v.any():
            return out
        if not use_monotone_threshold:
            out[v] = np.asarray(fC(Xa[v]), dtype=float) + np.asarray(fB(Xa[v]), dtype=float) / Na[v]
            return out

        flat_idx = np.flatnonzero(v)
        xv = Xa[v]
        nv = Na[v]
        out_v = np.full_like(xv, float("nan"), dtype=float)
        uniq_nq, inv = np.unique(nv, return_inverse=True)
        for iu, nq_u in enumerate(uniq_nq):
            mask = inv == iu
            t_prof, k_prof = _get_monotone_profile(float(nq_u))
            if t_prof is None or k_prof is None:
                continue
            out_v[mask] = np.interp(np.asarray(xv[mask], dtype=float), t_prof, k_prof)
        out.flat[flat_idx] = out_v
        return out

    def predict_log2_nps(X, n_q):
        Xa, Na = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        return Na * predict_k_x(Xa, Na)

    def predict_nps(X, n_q):
        l2 = np.asarray(predict_log2_nps(X, n_q), dtype=float)
        out = np.full_like(l2, float("nan"), dtype=float)
        if l2.size == 0:
            return out
        flat = out.ravel()
        lflat = l2.ravel()
        for i, lv in enumerate(lflat):
            flat[i] = _safe_nps_from_log2(_to_float(lv))
        return out

    def predict_sigma_log2_nps(X, n_q):
        """
        Parametric 1σ uncertainty in log2(nps) via delta method:
          σ²_{log2_nps} = g^T Σ(threshold) g,   g = [nq, 1]^T
        where Σ(threshold) = interpolated WLS covariance of (C, β) at the given threshold.
        """
        Xa, Na = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(n_q, dtype=float))
        out = np.full_like(Xa, float("nan"), dtype=float)
        if not have_cov:
            return out
        v = np.isfinite(Xa) & np.isfinite(Na) & (Na > 0) & (Xa >= x_lo) & (Xa <= x_hi)
        if not v.any():
            return out
        xv = Xa[v].ravel()
        nv = Na[v].ravel()
        flat_idx = np.flatnonzero(v)
        for k_idx, (x_i, n_i) in enumerate(zip(xv, nv)):
            Sigma = np.zeros((2, 2), dtype=float)
            bad = False
            for ii in range(2):
                for jj in range(2):
                    f_cov = cov_interp.get((ii, jj), None)
                    if f_cov is None:
                        bad = True
                        break
                    val = _scalarize_interp_value(f_cov(x_i))
                    if not math.isfinite(val):
                        bad = True
                        break
                    Sigma[ii, jj] = val
                if bad:
                    break
            if bad:
                continue
            # g = ∂(log2_nps)/∂(C, β) = ∂(nq·k_x)/∂(C, β) = nq · [1, 1/nq] = [nq, 1]
            g   = np.array([n_i, 1.0], dtype=float)
            var = float(g @ Sigma @ g)
            out.flat[flat_idx[k_idx]] = math.sqrt(max(var, 0.0))
        return out

    pred = Predictor(
        name=f"opt1_dense_{source_label}",
        channel=ch,
        p=p_val,
        threshold_min=x_lo,
        threshold_max=x_hi,
        predict_k_x=predict_k_x,
        predict_log2_nps=predict_log2_nps,
        predict_nps=predict_nps,
        predict_sigma_log2_nps=predict_sigma_log2_nps,
        interval_method=str(interval_method or 'cv'),
        conformal_quantile_anchors=None,
        meta={
            "n_threshold_fits": len(fit_rows),
            "step4_model": STEP4_MODEL,
            "has_parametric_cov": bool(have_cov),
            "cov_support_counts": {f"{i}{j}": int(cov_support_counts.get((i, j), 0)) for i in range(2) for j in range(2)},
            "source": source_label,
            "threshold_monotone_enforced": bool(use_monotone_threshold),
            "threshold_monotone_grid_size": int(max(64, int(monotone_threshold_grid_size))),
        },
    )

    # Always run multi-threshold forward-CV and derive:
    #   1) threshold-dependent empirical CV uncertainty,
    #   2) pooled sigma_cv_k fallback,
    #   3) extrapolation trust metadata for gating dashed tails.
    sigma_mode = str(sigma_cv_k_mode or SIGMA_CV_K_MODE_DEFAULT).strip().lower()
    if sigma_mode not in {"pooled_q", "pooled_max", "median_per_threshold_q"}:
        sigma_mode = str(SIGMA_CV_K_MODE_DEFAULT)
    sigma_cov = _to_float(sigma_cv_k_coverage)
    if not (0.0 < sigma_cov < 1.0):
        sigma_cov = float(SIGMA_CV_K_COVERAGE_DEFAULT)
    if cv_coverage_zs is None:
        z_list = [float(z) for z in CV_COVERAGE_ZS_DEFAULT]
    else:
        z_list = [abs(_to_float(z)) for z in cv_coverage_zs if _is_finite(z) and abs(_to_float(z)) > 0]
    if not z_list:
        z_list = [float(z) for z in CV_COVERAGE_ZS_DEFAULT]

    cv_result = _forward_chain_cv_multi_thr(
        case_rows,
        keep_details=keep_cv_details,
        pct_min=cv_thr_pct_min,
        pct_max=cv_thr_pct_max,
        max_horizon=CV_MAX_HORIZON,
    )
    abs_err_k = np.asarray(cv_result.pop("__abs_errors_k", []), dtype=float)
    abs_err_k_h1 = np.asarray(cv_result.pop("__abs_errors_k_h1", []), dtype=float)
    abs_err_k_h2plus = np.asarray(cv_result.pop("__abs_errors_k_h2plus", []), dtype=float)
    pred.meta["forward_cv"] = cv_result

    def _q_from_cv_stats_k(stats: dict, cov: float) -> float:
        if not isinstance(stats, dict):
            return float("nan")
        q68 = _to_float(stats.get("q68_abs_k", float("nan")))
        q90 = _to_float(stats.get("q90_abs_k", float("nan")))
        q95 = _to_float(stats.get("q95_abs_k", float("nan")))
        return _empirical_abs_quantile_from_anchors(
            cov, [(0.6827, q68), (0.90, q90), (0.95, q95)]
        )

    # Pooled scalar candidates in k-space.
    pooled_q_cov_k = _q_from_cv_stats_k(cv_result, sigma_cov)
    rmse_pooled_k = _to_float(cv_result.get("rmse_pooled_k", float("nan")))
    bias_pooled_k = _to_float(cv_result.get("bias_pooled_k", float("nan")))
    sigma_rmse_debiased_k = (
        math.sqrt(max(rmse_pooled_k ** 2 - bias_pooled_k ** 2, 0.0))
        if (_is_finite(rmse_pooled_k) and _is_finite(bias_pooled_k))
        else float("nan")
    )
    # Horizon-1 (observed-range proxy) scalar candidates in k-space.
    h1_stats = cv_result.get("per_horizon", {}).get("1", {}) if isinstance(cv_result.get("per_horizon", {}), dict) else {}
    h1_q_cov_k = _q_from_cv_stats_k(h1_stats, sigma_cov)
    h1_rmse_k = _to_float(h1_stats.get("rmse_k", float("nan")))
    h1_bias_k = _to_float(h1_stats.get("bias_k", float("nan")))
    h1_sigma_rmse_debiased_k = (
        math.sqrt(max(h1_rmse_k ** 2 - h1_bias_k ** 2, 0.0))
        if (_is_finite(h1_rmse_k) and _is_finite(h1_bias_k))
        else float("nan")
    )
    h1_q68_k = _to_float(h1_stats.get("q68_abs_k", float("nan")))
    h1_q90_k = _to_float(h1_stats.get("q90_abs_k", float("nan")))
    h1_q95_k = _to_float(h1_stats.get("q95_abs_k", float("nan")))
    obs_anchors = []
    if _is_finite(h1_q68_k) and h1_q68_k >= 0:
        obs_anchors.append((0.6827, float(h1_q68_k)))
    if _is_finite(h1_q90_k) and h1_q90_k >= 0:
        obs_anchors.append((0.90, float(h1_q90_k)))
    if _is_finite(h1_q95_k) and h1_q95_k >= 0:
        obs_anchors.append((0.95, float(h1_q95_k)))

    # Threshold-dependent CV sigma_k / quantile interpolation.
    per_thr = cv_result.get("per_threshold", {})
    cv_thr_vals: List[float] = []
    cv_sigma_k_vals: List[float] = []
    cv_q68_vals: List[float] = []
    cv_q90_vals: List[float] = []
    cv_q95_vals: List[float] = []
    per_thr_q_cov_k_vals: List[float] = []
    if isinstance(per_thr, dict):
        for k_thr, st in per_thr.items():
            thr = _to_float(k_thr)
            if not _is_finite(thr) or not isinstance(st, dict):
                continue
            n_h_t_raw = st.get("n_horizons", 0)
            try:
                n_h_t = int(n_h_t_raw)
            except Exception:
                n_h_t_f = _to_float(n_h_t_raw)
                n_h_t = int(n_h_t_f) if _is_finite(n_h_t_f) else 0
            if n_h_t < int(CV_MIN_SLICE_HORIZONS):
                continue
            q68_t = _to_float(st.get("q68_abs_k", float("nan")))
            q90_t = _to_float(st.get("q90_abs_k", float("nan")))
            q95_t = _to_float(st.get("q95_abs_k", float("nan")))
            q_cov_t = _q_from_cv_stats_k(st, sigma_cov)
            rmse_t = _to_float(st.get("rmse_k", float("nan")))
            bias_t = _to_float(st.get("bias_k", float("nan")))
            sigma_rmse_debiased_t = (
                math.sqrt(max(rmse_t ** 2 - bias_t ** 2, 0.0))
                if (_is_finite(rmse_t) and _is_finite(bias_t))
                else float("nan")
            )
            if not (_is_finite(q_cov_t) and q_cov_t >= 0):
                continue
            cv_thr_vals.append(float(thr))
            if sigma_mode == "pooled_max":
                cands_t = [v for v in (q_cov_t, sigma_rmse_debiased_t) if _is_finite(v) and v >= 0]
                sigma_t = float(max(cands_t)) if cands_t else float(q_cov_t)
            else:
                sigma_t = float(q_cov_t)
            cv_sigma_k_vals.append(float(sigma_t))
            per_thr_q_cov_k_vals.append(float(q_cov_t))
            cv_q68_vals.append(float(q68_t) if _is_finite(q68_t) and q68_t >= 0 else float("nan"))
            cv_q90_vals.append(float(q90_t) if _is_finite(q90_t) and q90_t >= 0 else float("nan"))
            cv_q95_vals.append(float(q95_t) if _is_finite(q95_t) and q95_t >= 0 else float("nan"))

    # Select extrapolation scalar sigma_cv_k according to requested rule.
    if sigma_mode == "median_per_threshold_q" and per_thr_q_cov_k_vals:
        sigma_cv_k_extrap = float(np.median(np.asarray(per_thr_q_cov_k_vals, dtype=float)))
    elif sigma_mode == "pooled_max":
        cands = [v for v in (pooled_q_cov_k, sigma_rmse_debiased_k) if _is_finite(v) and v >= 0]
        sigma_cv_k_extrap = float(max(cands)) if cands else float("nan")
    else:
        sigma_cv_k_extrap = float(pooled_q_cov_k)
    if not (_is_finite(sigma_cv_k_extrap) and sigma_cv_k_extrap >= 0):
        fb = [v for v in (
            _to_float(cv_result.get("q68_abs_k", float("nan"))),
            _to_float(cv_result.get("rmse_pooled_k", float("nan"))),
        ) if _is_finite(v) and v >= 0]
        sigma_cv_k_extrap = float(fb[0]) if fb else float(MIN_SIGMA_K)

    # Select observed-range scalar sigma_cv_k from horizon=1 only.
    if sigma_mode == "pooled_max":
        cands_obs = [v for v in (h1_q_cov_k, h1_sigma_rmse_debiased_k) if _is_finite(v) and v >= 0]
        sigma_cv_k_obs = float(max(cands_obs)) if cands_obs else float("nan")
    else:
        sigma_cv_k_obs = float(h1_q_cov_k)
    if not (_is_finite(sigma_cv_k_obs) and sigma_cv_k_obs >= 0):
        cands_obs_fb = [
            v for v in (
                _to_float(h1_stats.get("q68_abs_k", float("nan"))),
                _to_float(h1_stats.get("rmse_k", float("nan"))),
                sigma_cv_k_extrap,
            ) if _is_finite(v) and v >= 0
        ]
        sigma_cv_k_obs = float(cands_obs_fb[0]) if cands_obs_fb else float(MIN_SIGMA_K)

    f_sigma_cv_k = None
    f_q68 = None
    f_q90 = None
    f_q95 = None
    if len(cv_thr_vals) >= 2:
        t_arr = np.array(cv_thr_vals, dtype=float)
        f_sigma_cv_k = _make_interp(t_arr, np.array(cv_sigma_k_vals, dtype=float), extrapolate=True)
        if np.isfinite(cv_q68_vals).sum() >= 2:
            f_q68 = _make_interp(t_arr, np.array(cv_q68_vals, dtype=float), extrapolate=True)
        if np.isfinite(cv_q90_vals).sum() >= 2:
            f_q90 = _make_interp(t_arr, np.array(cv_q90_vals, dtype=float), extrapolate=True)
        if np.isfinite(cv_q95_vals).sum() >= 2:
            f_q95 = _make_interp(t_arr, np.array(cv_q95_vals, dtype=float), extrapolate=True)

    sigma_cv_k_scalar = float(sigma_cv_k_extrap)

    def predict_sigma_cv_k_fn(X, n_q=None):
        Xa = np.asarray(X, dtype=float)
        out = np.full_like(Xa, sigma_cv_k_scalar, dtype=float)
        if f_sigma_cv_k is not None:
            vv = np.asarray(f_sigma_cv_k(Xa), dtype=float)
            good = np.isfinite(vv) & (vv >= 0)
            out = np.where(good, vv, out)
        elif len(cv_sigma_k_vals) == 1:
            out[...] = float(cv_sigma_k_vals[0])
        out = np.where(np.isfinite(out) & (out >= 0), out, sigma_cv_k_scalar)
        if n_q is not None:
            Na = np.asarray(np.broadcast_arrays(Xa, np.asarray(n_q, dtype=float))[1], dtype=float)
            obs_mask = np.isfinite(Na) & (Na <= float(n_q_obs_max))
            out = np.where(obs_mask, float(sigma_cv_k_obs), out)
        return out

    def _q_at(f_interp, Xarr, fallback=np.nan):
        if f_interp is None:
            return np.full_like(Xarr, fallback, dtype=float)
        vals = np.asarray(f_interp(Xarr), dtype=float)
        return np.where(np.isfinite(vals) & (vals >= 0), vals, fallback)

    def predict_cv_abs_quantile_k_fn(X, coverage, n_q=None):
        Xa = np.asarray(X, dtype=float)
        c = _to_float(coverage)
        if not _is_finite(c):
            c = 0.6827
        c = float(min(max(c, 1e-6), 0.999999))

        q68 = _q_at(f_q68, Xa)
        q90 = _q_at(f_q90, Xa)
        q95 = _q_at(f_q95, Xa)
        out = np.full_like(Xa, float("nan"), dtype=float)
        zc = _to_float(NormalDist().inv_cdf(0.5 + 0.5 * c))
        if not _is_finite(zc) or zc <= 0:
            zc = float(Z68_ABS)
        sig_fallback = predict_sigma_cv_k_fn(Xa, n_q)

        flat_idx = np.ndindex(Xa.shape)
        for idx in flat_idx:
            anchors = []
            if _is_finite(q68[idx]):
                anchors.append((0.6827, float(q68[idx])))
            if _is_finite(q90[idx]):
                anchors.append((0.90, float(q90[idx])))
            if _is_finite(q95[idx]):
                anchors.append((0.95, float(q95[idx])))
            qv = _empirical_abs_quantile_from_anchors(c, anchors)
            if _is_finite(qv) and qv >= 0:
                out[idx] = float(qv)
            elif _is_finite(sig_fallback[idx]) and sig_fallback[idx] >= 0:
                out[idx] = float(sig_fallback[idx] * zc)
        if n_q is not None:
            Na = np.asarray(np.broadcast_arrays(Xa, np.asarray(n_q, dtype=float))[1], dtype=float)
            obs_mask = np.isfinite(Na) & (Na <= float(n_q_obs_max))
            q_obs = _empirical_abs_quantile_from_anchors(c, obs_anchors)
            if not (_is_finite(q_obs) and q_obs >= 0):
                q_obs = float(sigma_cv_k_obs) * float(zc)
            out = np.where(obs_mask, q_obs, out)
        return out

    pred.sigma_cv_k = float(sigma_cv_k_extrap)
    pred.sigma_cv_k_obs = float(sigma_cv_k_obs)
    pred.sigma_cv_k_extrap = float(sigma_cv_k_extrap)
    pred.n_q_obs_max = int(n_q_obs_max)
    pred.cv_obs_quantile_anchors = list(obs_anchors)
    pred.predict_sigma_cv_k = predict_sigma_cv_k_fn
    pred.predict_cv_abs_quantile_k = predict_cv_abs_quantile_k_fn
    pred.sigma_cv_log2_obs_at_obs_max = float(n_q_obs_max) * float(pred.sigma_cv_k_obs)
    pred.sigma_cv_log2_extrap_at_obs_max = float(n_q_obs_max) * float(pred.sigma_cv_k_extrap)
    # Backward-compatible alias kept for reporting semantics (observed-range at obs max).
    pred.sigma_cv_log2_at_obs_max = float(pred.sigma_cv_log2_obs_at_obs_max)
    # Gate uses conservative extrapolation-scaled value at obs max.
    pred.sigma_cv_log2_gate_at_obs_max = float(pred.sigma_cv_log2_extrap_at_obs_max)
    pred.sigma_cv = float(pred.sigma_cv_log2_gate_at_obs_max)
    pred.meta["sigma_cv_k_mode"] = str(sigma_mode)
    pred.meta["sigma_cv_k_coverage"] = float(sigma_cov)
    pred.meta["sigma_cv_k_obs"] = float(pred.sigma_cv_k_obs)
    pred.meta["sigma_cv_k_extrap"] = float(pred.sigma_cv_k_extrap)

    # Coverage diagnostics in k-space for chosen sigma policy.
    if abs_err_k.size > 0 or abs_err_k_h1.size > 0 or abs_err_k_h2plus.size > 0:
        cov_diag = {}
        for z in z_list:
            zf = abs(_to_float(z))
            if not (_is_finite(zf) and zf > 0):
                continue
            nom = float(_central_coverage_from_z(zf))
            rec = {"nominal": nom}
            if abs_err_k_h1.size > 0:
                obs_h1 = float(np.mean(np.abs(abs_err_k_h1) <= (zf * float(pred.sigma_cv_k_obs))))
                rec["observed_h1"] = obs_h1
                rec["gap_h1"] = obs_h1 - nom
            if abs_err_k_h2plus.size > 0:
                obs_h2 = float(np.mean(np.abs(abs_err_k_h2plus) <= (zf * float(pred.sigma_cv_k_extrap))))
                rec["observed_h2plus"] = obs_h2
                rec["gap_h2plus"] = obs_h2 - nom
            if abs_err_k.size > 0:
                # Conservative all-horizon reference against extrap sigma.
                obs_all = float(np.mean(np.abs(abs_err_k) <= (zf * float(pred.sigma_cv_k_extrap))))
                rec["observed_all_ref_extrap_sigma"] = obs_all
                rec["gap_all_ref_extrap_sigma"] = obs_all - nom
            cov_diag[f"{zf:.6g}"] = rec
        pred.meta["cv_abs_ek_count"] = int(abs_err_k.size)
        pred.meta["cv_abs_ek_h1_count"] = int(abs_err_k_h1.size)
        pred.meta["cv_abs_ek_h2plus_count"] = int(abs_err_k_h2plus.size)
        pred.meta["cv_coverage_abs_ek"] = cov_diag

    # Extrapolation trust gate:
    #   - CV must be applicable
    #   - sufficient total horizons
    #   - sigma_cv not too large
    #   - slice instability ratio not too large
    cv_app = bool(cv_result.get("applicable", False))
    n_h_raw = cv_result.get("n_total_pooled_horizons", cv_result.get("n_total_horizons", 0))
    try:
        n_h = int(n_h_raw)
    except Exception:
        n_h_f = _to_float(n_h_raw)
        n_h = int(n_h_f) if _is_finite(n_h_f) else 0
    sig_cv = float(pred.sigma_cv_log2_gate_at_obs_max) if _is_finite(pred.sigma_cv_log2_gate_at_obs_max) else float("nan")
    rmse_ratio_k = _to_float(cv_result.get("rmse_max_ratio_k", float("nan")))
    rmse_ratio = _to_float(cv_result.get("rmse_max_ratio", float("nan")))
    rmse_ratio_gate = rmse_ratio_k if _is_finite(rmse_ratio_k) else rmse_ratio

    try:
        gate_min_h = int(extrap_min_horizons)
    except Exception:
        gate_min_h = EXTRAP_TRUST_MIN_HORIZONS
    if gate_min_h < 1:
        gate_min_h = EXTRAP_TRUST_MIN_HORIZONS
    gate_max_sc = (
        float(extrap_max_sigma_cv)
        if _is_finite(extrap_max_sigma_cv) and float(extrap_max_sigma_cv) > 0
        else EXTRAP_TRUST_MAX_SIGMA_CV
    )
    gate_max_ratio = (
        float(extrap_max_rmse_ratio)
        if _is_finite(extrap_max_rmse_ratio) and float(extrap_max_rmse_ratio) > 0
        else EXTRAP_TRUST_MAX_RMSE_RATIO
    )

    extrap_ok = True
    extrap_reason = "ok"
    if not cv_app:
        extrap_ok = False
        extrap_reason = f"cv_not_applicable:{cv_result.get('reason', 'unknown')}"
    elif n_h < gate_min_h:
        extrap_ok = False
        extrap_reason = f"insufficient_total_horizons:{n_h}<{gate_min_h}"
    elif (not _is_finite(sig_cv)) or sig_cv > gate_max_sc:
        extrap_ok = False
        if _is_finite(sig_cv):
            extrap_reason = f"sigma_cv_too_large:{sig_cv:.4g}>{gate_max_sc:.4g}"
        else:
            extrap_reason = "sigma_cv_nan"
    elif (not _is_finite(rmse_ratio_gate)) or rmse_ratio_gate > gate_max_ratio:
        extrap_ok = False
        if _is_finite(rmse_ratio_gate):
            if _is_finite(rmse_ratio_k):
                extrap_reason = f"rmse_ratio_k_too_large:{rmse_ratio_gate:.4g}>{gate_max_ratio:.4g}"
            else:
                extrap_reason = f"rmse_ratio_too_large:{rmse_ratio_gate:.4g}>{gate_max_ratio:.4g}"
        else:
            extrap_reason = "rmse_ratio_nan"

    pred.meta["sigma_cv"] = float(pred.sigma_cv)
    pred.meta["sigma_cv_k"] = float(pred.sigma_cv_k)
    pred.meta["sigma_cv_k_obs"] = float(pred.sigma_cv_k_obs)
    pred.meta["sigma_cv_k_extrap"] = float(pred.sigma_cv_k_extrap)
    pred.meta["sigma_cv_log2_at_obs_max"] = float(pred.sigma_cv_log2_at_obs_max)
    pred.meta["sigma_cv_log2_obs_at_obs_max"] = float(pred.sigma_cv_log2_obs_at_obs_max)
    pred.meta["sigma_cv_log2_extrap_at_obs_max"] = float(pred.sigma_cv_log2_extrap_at_obs_max)
    pred.meta["sigma_cv_log2_gate_at_obs_max"] = float(pred.sigma_cv_log2_gate_at_obs_max)
    pred.meta["sigma_cv_gate_basis"] = "extrap_sigma_k_at_obs_max"
    pred.meta["n_q_obs_max"] = int(n_q_obs_max)
    pred.meta["strict_split_conformal_requested"] = bool(strict_split_requested)
    pred.meta["strict_split_conformal_active"] = False
    pred.meta["strict_split_reason"] = str(strict_split_reason)
    pred.meta["extrapolation_gate"] = {
        "min_total_horizons": int(gate_min_h),
        "horizon_count_type": "pooled_total_forward_cv_horizons",
        "max_sigma_cv": float(gate_max_sc),
        "max_rmse_ratio": float(gate_max_ratio),
    }
    pred.meta["extrapolation_trusted"] = bool(extrap_ok)
    pred.meta["extrapolation_reason"] = str(extrap_reason)

    # Bootstrap quality diagnostics (for warnings/supplement tables).
    floor_thr = max(float(TARGET_FLOOR), float(L_BASELINE + 0.05))
    valid_ratios: List[float] = []
    valid_ratios_floor: List[float] = []
    censor_fracs: List[float] = []
    censor_fracs_floor: List[float] = []
    for rr in case_rows:
        thr = _to_float(rr.get("threshold", float("nan")))
        nb = _to_float(rr.get("n_boot", float("nan")))
        nv = _to_float(rr.get("n_valid", float("nan")))
        if _is_finite(nb) and nb > 0 and _is_finite(nv) and nv >= 0:
            vr = float(nv / nb)
            valid_ratios.append(vr)
            if _is_finite(thr) and float(thr) <= floor_thr:
                valid_ratios_floor.append(vr)
        cf = _to_float(rr.get("censored_boot_frac", float("nan")))
        if _is_finite(cf) and cf >= 0:
            censor_fracs.append(float(cf))
            if _is_finite(thr) and float(thr) <= floor_thr:
                censor_fracs_floor.append(float(cf))
    low_valid_thr = 0.20
    high_censor_thr = 0.20
    bq = {
        "n_rows": int(len(case_rows)),
        "n_rows_with_valid_ratio": int(len(valid_ratios)),
        "n_rows_with_censor_frac": int(len(censor_fracs)),
        "low_valid_threshold": float(low_valid_thr),
        "high_censor_threshold": float(high_censor_thr),
        "low_valid_rate": float(np.mean(np.asarray(valid_ratios, dtype=float) < low_valid_thr)) if valid_ratios else float("nan"),
        "low_valid_rate_near_floor": float(np.mean(np.asarray(valid_ratios_floor, dtype=float) < low_valid_thr)) if valid_ratios_floor else float("nan"),
        "high_censor_rate": float(np.mean(np.asarray(censor_fracs, dtype=float) > high_censor_thr)) if censor_fracs else float("nan"),
        "high_censor_rate_near_floor": float(np.mean(np.asarray(censor_fracs_floor, dtype=float) > high_censor_thr)) if censor_fracs_floor else float("nan"),
        "median_valid_ratio": float(np.nanmedian(np.asarray(valid_ratios, dtype=float))) if valid_ratios else float("nan"),
        "median_censor_frac": float(np.nanmedian(np.asarray(censor_fracs, dtype=float))) if censor_fracs else float("nan"),
    }
    pred.meta["bootstrap_quality"] = bq
    pred.meta["predictor_validation"] = {
        "threshold_order_ok": bool(x_lo < x_hi),
        "n_threshold_fits": int(len(fit_rows)),
        "forward_cv_applicable": bool(cv_app),
        "forward_cv_reason": str(cv_result.get("reason", "ok")),
    }

    # ---- Held-out calibration (split-conformal-like) for interval bands
    req_m = str(req_m_requested)
    pred.interval_method = str(req_m)
    pred.meta["interval_method_requested"] = str(req_m)
    pred.meta["conformal_holdout_count"] = int(conformal_holdout_count)
    pred.meta["conformal_min_abs_errors"] = int(conformal_min_abs_errors)

    if req_m != "cv":
        conf = _split_conformal_calibrate_abs_errors_k(
            case_rows,
            holdout_count=int(conformal_holdout_count),
            pct_min=float(cv_thr_pct_min),
            pct_max=float(cv_thr_pct_max),
            min_abs_errors=int(conformal_min_abs_errors),
            z_list=z_list,
        )
        pred.meta["conformal_calibration_mode"] = "per_threshold_holdout_proxy"
        if bool(conf.get("ok", False)) and conf.get("anchors"):
            pred.conformal_quantile_anchors = list(conf["anchors"])
            pred.meta["conformal_calibration"] = conf
            pred.meta["interval_method"] = str(req_m)
        else:
            pred.interval_method = "cv"
            pred.meta["interval_method"] = "cv"
            pred.meta["interval_method_fallback"] = {
                "requested": str(req_m),
                "effective": "cv",
                "reason": str(conf.get("reason", "unknown")),
                "n_abs_errors": int(conf.get("n_abs_errors", 0)),
            }
    else:
        pred.meta["interval_method"] = "cv"
    pred.meta["interval_method_effective"] = str(pred.meta.get("interval_method", pred.interval_method))

    return pred
# ─────────────────────────────────────────────────────────────────────────────
# Shared postprocessing for bootstrap rows (HG and ML)
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess_rows(rows: List[dict]) -> List[dict]:
    """
    Normalize raw bootstrap rows into the standard format consumed by _build_opt1_predictor.
    Computes sigma_k from the bootstrap CI half-width using per-row ci_level.
    """
    out = []
    for r in rows:
        row = dict(r)
        row["channel"]   = str(row["channel"])
        row["p"]         = float(row["p"])
        row["threshold"] = _thr_key(float(row["threshold"]))
        row["n_q"]       = int(float(row["n_q"]))
        row["status"]    = str(row.get("status", ""))
        n_valid = None
        if "n_valid" in row:
            nv_f = _to_float(row.get("n_valid", float("nan")))
            n_valid = int(nv_f) if _is_finite(nv_f) else 0
            row["n_valid"] = int(n_valid)
        nb_f = _to_float(row.get("n_boot", float("nan")))
        if _is_finite(nb_f):
            row["n_boot"] = int(nb_f)
        if "n_censored_boot" in row and _is_finite(row.get("n_censored_boot", float("nan"))) and _is_finite(row.get("n_boot", float("nan"))):
            ncb = int(_to_float(row.get("n_censored_boot", 0)))
            nb = max(1, int(_to_float(row.get("n_boot", 1))))
            row["censored_boot_frac"] = float(ncb / nb)
        elif "censored_boot_frac" in row:
            row["censored_boot_frac"] = _to_float(row.get("censored_boot_frac", float("nan")))
        for k in ["k_x", "k_x_lo", "k_x_hi", "k_x_err_lo", "k_x_err_hi", "log2_nps"]:
            if k in row:
                row[k] = _to_float(row[k])
        row["is_ok"]       = row["status"] == "ok" and _is_finite(row.get("k_x"))
        row["is_censored"] = row["status"] in {"left_censored", "right_censored"}
        ci_half = 0.5 * (_to_float(row.get("k_x_hi", float("nan"))) -
                         _to_float(row.get("k_x_lo", float("nan"))))
        row["k_ci_halfwidth"] = ci_half
        ci = _to_float(row.get("ci_level", BOOT_CI_LEVEL))
        if not (0.0 < ci < 1.0):
            ci = float(BOOT_CI_LEVEL)
        row["ci_level"] = float(ci)

        # Critical guard: rows without reliable CI support must not receive
        # tiny sigma (which would over-weight them in WLS).
        if row["is_ok"] and (not _is_finite(ci_half)):
            row["sigma_k"] = float("nan")
            row["is_ok"] = False
            row["postprocess_drop_reason"] = "missing_ci"
            out.append(row)
            continue
        if row["is_ok"] and n_valid is not None and n_valid < int(POSTPROCESS_MIN_N_VALID):
            row["sigma_k"] = float("nan")
            row["is_ok"] = False
            row["postprocess_drop_reason"] = f"low_n_valid:{n_valid}<{POSTPROCESS_MIN_N_VALID}"
            out.append(row)
            continue

        z = float(NormalDist().inv_cdf(0.5 + 0.5 * float(ci)))
        if not _is_finite(z) or z <= 0:
            z = float(CI_TO_SIGMA_Z_DEFAULT)
        sigma = ci_half / max(z, 1e-12) if _is_finite(ci_half) else float("nan")
        if not _is_finite(sigma) or sigma <= 0:
            sigma = MIN_SIGMA_K
        row["sigma_k"] = max(float(sigma), MIN_SIGMA_K)
        out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HG dense threshold bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_hg_dense_threshold_rows(
    load_hg_fn,
    hm_dir: Path,
    nq_list: List[int],
    channel: str,
    amplitude: float,
    thresholds: np.ndarray = THRESHOLDS_DENSE,
    n_boot: int = N_BOOT_HG,
    seed: int = BOOT_SEED,
    ci_level: float = BOOT_CI_LEVEL,
    hg_bootstrap_mode: str = HG_BOOTSTRAP_MODE_DEFAULT,
    replicate_sink: Optional[Dict[int, dict]] = None,
) -> List[dict]:
    """
    Bootstrap dense threshold rows for Hypergraph data.

    Input formats supported from load_hg_fn:
      • 2D acc_trials: (n_k, n_trials_max) with NaN-padding.
      • 3D acc_trials: (n_k, n_cluster, n_draw) where clusters correspond to hypergraph instances.

    Bootstrap modes:
      • per_k:        resample trials independently within each k row (legacy; discards cross-k dependence).
      • cluster:      resample *curves* (columns in 2D, clusters in 3D), preserving cross-k dependence.
      • hier_cluster: like cluster, but additionally resamples draws within each cluster (3D only).
      • auto:         choose cluster when the 2D matrix looks like column-aligned NaN-padding; else per_k.

    Returns rows in the same schema as shadow_mod.bootstrap_dense_threshold_rows (postprocessed).
    """
    rng   = np.random.default_rng(seed)
    alpha = (1.0 - float(ci_level)) / 2.0
    q_lo, q_hi = 100.0 * alpha, 100.0 * (1.0 - alpha)
    rows: List[dict] = []
    thresholds_arr = np.asarray(thresholds, dtype=float)

    req_mode = str(hg_bootstrap_mode or HG_BOOTSTRAP_MODE_DEFAULT).strip().lower()
    if req_mode not in {"auto", "per_k", "cluster", "hier_cluster"}:
        req_mode = "auto"

    for nq in sorted(nq_list):
        try:
            acc_trials, k_grid, meta = load_hg_fn(hm_dir, int(nq), channel, float(amplitude))
        except Exception as e:
            print(f"  HG load skip nq={nq} ch={channel} p={amplitude}: {e}")
            continue

        acc_trials = np.asarray(acc_trials, dtype=float)
        k_grid     = np.asarray(k_grid, dtype=float)

        if k_grid.ndim != 1:
            print(f"  HG skip nq={nq} ch={channel} p={amplitude}: invalid k_grid shape {k_grid.shape}")
            continue

        n_k = int(k_grid.size)
        if n_k < 2:
            print(f"  HG skip nq={nq} ch={channel} p={amplitude}: only {n_k} k-values")
            continue

        # ---- Determine effective mode
        mode = req_mode
        if mode == "auto":
            if acc_trials.ndim == 3:
                mode = "cluster"
            elif acc_trials.ndim == 2:
                # Heuristic: many rows have contiguous finite prefix from column 0 (typical NaN-padding).
                contig = 0
                nonempty = 0
                for j in range(n_k):
                    mask = np.isfinite(acc_trials[j])
                    if not np.any(mask):
                        continue
                    nonempty += 1
                    idx = np.flatnonzero(mask)
                    if idx.size > 0 and idx.min() == 0 and np.all(mask[: idx.max() + 1]):
                        contig += 1
                frac = (contig / nonempty) if nonempty > 0 else 0.0
                mode = "cluster" if frac >= 0.8 else "per_k"
            else:
                mode = "per_k"

        # ---- Central mean curve and bootstrap replicate curves
        mean_curve_mon = None
        bs_curves_mon = None
        n_trials_effective = 0
        n_clusters = 0
        n_draws = 0

        if acc_trials.ndim == 3:
            # (n_k, n_cluster, n_draw)
            if acc_trials.shape[0] != n_k:
                print(f"  HG skip nq={nq} ch={channel} p={amplitude}: acc_trials shape {acc_trials.shape} != n_k")
                continue
            n_clusters = int(acc_trials.shape[1])
            n_draws = int(acc_trials.shape[2])
            if n_draws < 1:
                print(f"  HG skip nq={nq} ch={channel} p={amplitude}: n_draws={n_draws} (cannot bootstrap)")
                continue
            n_trials_effective = n_clusters * max(1, n_draws)

            # Reduce within-cluster draws for central curve
            clust_means = np.nanmean(acc_trials, axis=2)  # (n_k, n_cluster)
            mean_curve = np.nanmean(clust_means, axis=1)  # (n_k,)
            mean_curve_mon = _monotone_curve(mean_curve)

            bs_means = np.full((n_boot, n_k), float("nan"), dtype=float)
            if n_clusters < 1:
                continue

            if mode == "hier_cluster":
                # resample clusters; within each selected cluster resample draws
                for b in range(n_boot):
                    cl_idx = rng.integers(0, n_clusters, size=n_clusters)
                    # For each selected cluster, resample draws and compute mean per k, then average clusters.
                    sel = acc_trials[:, cl_idx, :]  # (n_k, n_clusters, n_draw)
                    # Resample draws within each selected cluster
                    dr_idx = rng.integers(0, n_draws, size=(n_clusters, n_draws))
                    # gather: for each cluster c, select draws
                    # We'll do loop over clusters to keep memory bounded.
                    cl_means_b = np.full((n_k, n_clusters), float("nan"), dtype=float)
                    for ci in range(n_clusters):
                        cl_draws = sel[:, ci, :]
                        picks = dr_idx[ci]
                        cl_means_b[:, ci] = np.nanmean(cl_draws[:, picks], axis=1)
                    bs_means[b] = np.nanmean(cl_means_b, axis=1)
            else:
                # cluster bootstrap on clust_means (already averaged over draws)
                for b in range(n_boot):
                    cl_idx = rng.integers(0, n_clusters, size=n_clusters)
                    bs_means[b] = np.nanmean(clust_means[:, cl_idx], axis=1)

            bs_curves_mon = np.empty_like(bs_means)
            for b in range(n_boot):
                bs_curves_mon[b] = _monotone_curve(bs_means[b])

        elif acc_trials.ndim == 2:
            # (n_k, n_trials_max)
            if acc_trials.shape[0] != n_k:
                print(f"  HG skip nq={nq} ch={channel} p={amplitude}: acc_trials shape {acc_trials.shape} != n_k")
                continue
            n_trials_effective = int(acc_trials.shape[1])

            if mode == "cluster":
                # Resample columns (curve-units), preserving cross-k dependence.
                col_mask = np.any(np.isfinite(acc_trials), axis=0)
                valid_cols = np.flatnonzero(col_mask)
                if valid_cols.size < 2:
                    mode = "per_k"  # fallback
                else:
                    mean_curve = np.nanmean(acc_trials[:, valid_cols], axis=1)
                    mean_curve_mon = _monotone_curve(mean_curve)

                    bs_means = np.full((n_boot, n_k), float("nan"), dtype=float)
                    for b in range(n_boot):
                        idx = rng.choice(valid_cols, size=valid_cols.size, replace=True)
                        bs_means[b] = np.nanmean(acc_trials[:, idx], axis=1)

                    bs_curves_mon = np.empty_like(bs_means)
                    for b in range(n_boot):
                        bs_curves_mon[b] = _monotone_curve(bs_means[b])

            if mode == "per_k":
                finite_per_k = [acc_trials[j, np.isfinite(acc_trials[j, :])] for j in range(n_k)]
                n_resample   = [max(1, int(ft.size)) for ft in finite_per_k]
                has_data     = [ft.size > 0 for ft in finite_per_k]

                mean_curve = np.array([
                    float(np.nanmean(acc_trials[j, :])) if has_data[j] else float("nan")
                    for j in range(n_k)
                ])
                mean_curve_mon = _monotone_curve(mean_curve)

                bs_means = np.full((n_boot, n_k), float("nan"), dtype=float)
                for b in range(n_boot):
                    for j in range(n_k):
                        if not has_data[j]:
                            continue
                        samp = rng.choice(finite_per_k[j], size=n_resample[j], replace=True)
                        bs_means[b, j] = float(np.mean(samp))

                bs_curves_mon = np.empty_like(bs_means)
                for b in range(n_boot):
                    bs_curves_mon[b] = _monotone_curve(bs_means[b])

        else:
            print(f"  HG skip nq={nq} ch={channel} p={amplitude}: unsupported acc_trials ndim={acc_trials.ndim}")
            continue

        if mean_curve_mon is None or bs_curves_mon is None or not np.any(np.isfinite(mean_curve_mon)):
            print(f"  HG skip nq={nq} ch={channel} p={amplitude}: mean curve invalid under mode={mode}")
            continue

        # ---- Invert at each threshold
        kx_boot = np.full((int(n_boot), int(thresholds_arr.size)), np.nan, dtype=np.float32) if replicate_sink is not None else None
        for t_idx, X_thr in enumerate(thresholds_arr):
            status_c, k_center = _invert_threshold_on_curve(k_grid, mean_curve_mon, float(X_thr))

            k_samples = []
            n_ok_boot = 0
            n_left_censored_boot = 0
            n_right_censored_boot = 0
            n_invalid_boot = 0
            for b in range(n_boot):
                st, kv = _invert_threshold_on_curve(k_grid, bs_curves_mon[b], float(X_thr))
                if st == "ok" and math.isfinite(kv):
                    k_samples.append(float(kv))
                    n_ok_boot += 1
                    if kx_boot is not None:
                        kx_boot[b, t_idx] = float(kv)
                elif st == "left_censored":
                    n_left_censored_boot += 1
                elif st == "right_censored":
                    n_right_censored_boot += 1
                else:
                    n_invalid_boot += 1
            k_arr = np.asarray(k_samples, dtype=float)

            row: dict = {
                "channel":    channel,
                "p":          float(amplitude),
                "n_q":        int(nq),
                "threshold":  float(X_thr),
                "status":     status_c,
                "k_x":        float(k_center) if (status_c == "ok" and math.isfinite(k_center)) else float("nan"),
                "k_x_lo":     float("nan"),
                "k_x_hi":     float("nan"),
                "k_x_err_lo": float("nan"),
                "k_x_err_hi": float("nan"),
                "log2_nps":   float(nq * k_center) if (status_c == "ok" and math.isfinite(k_center)) else float("nan"),
                "n_valid":    int(k_arr.size),
                "n_k":        int(n_k),
                "n_boot":     int(n_boot),
                "n_ok_boot": int(n_ok_boot),
                "n_left_censored_boot": int(n_left_censored_boot),
                "n_right_censored_boot": int(n_right_censored_boot),
                "n_censored_boot": int(n_left_censored_boot + n_right_censored_boot),
                "censored_boot_frac": float((n_left_censored_boot + n_right_censored_boot) / max(1, int(n_boot))),
                "n_invalid_boot": int(n_invalid_boot),
                "ci_level":   float(ci_level),
                "n_trials":   int(n_trials_effective),
                "hg_bootstrap_mode": str(mode),
                "hg_bootstrap_mode_requested": str(req_mode),
                "n_clusters": int(n_clusters),
                "n_draws":    int(n_draws),
                "source":     "hg",
            }
            if k_arr.size > 0:
                lo  = float(np.nanpercentile(k_arr, q_lo))
                hi  = float(np.nanpercentile(k_arr, q_hi))
                med = float(np.nanmedian(k_arr))
                row["k_x_lo"] = lo
                row["k_x_hi"] = hi
                if status_c == "ok" and math.isfinite(row["k_x"]):
                    row["k_x_err_lo"] = float(max(0.0, row["k_x"] - lo))
                    row["k_x_err_hi"] = float(max(0.0, hi  - row["k_x"]))
                elif status_c == "ok":
                    row["k_x"]      = med
                    row["log2_nps"] = float(nq * med)
                    row["k_x_err_lo"] = float(max(0.0, med - lo))
                    row["k_x_err_hi"] = float(max(0.0, hi  - med))
            rows.append(row)
        if replicate_sink is not None and kx_boot is not None:
            replicate_sink[int(nq)] = {
                "kx_boot": kx_boot,
                "n_boot": int(n_boot),
                "thresholds": np.asarray(thresholds_arr, dtype=float),
                "n_trials": int(n_trials_effective),
                "n_k": int(n_k),
                "hg_bootstrap_mode": str(mode),
                "hg_bootstrap_mode_requested": str(req_mode),
            }

    return _postprocess_rows(rows)


def bootstrap_ml_dense_threshold_rows(
    curves: List[dict],
    thresholds: np.ndarray = THRESHOLDS_DENSE,
    n_boot: int = N_BOOT_ML,
    seed: int = BOOT_SEED,
    ci_level: float = BOOT_CI_LEVEL,
    replicate_sink: Optional[Dict[Tuple[str, float, int], dict]] = None,
) -> List[dict]:
    """
    Bootstrap dense-threshold rows for ML curves.

    Optional replicate_sink captures per-case bootstrap inversion tensors:
      key   : (channel, p, n_q)
      value : {"kx_boot": (B,T), "thresholds": (T,), ...}
    """
    rows: List[dict] = []
    rng = np.random.default_rng(seed)
    thresholds_arr = np.asarray(thresholds, dtype=float)
    alpha = (1.0 - float(ci_level)) / 2.0
    q_lo = 100.0 * alpha
    q_hi = 100.0 * (1.0 - alpha)

    for curve in curves:
        k_grid = np.asarray(curve["k_grid"], dtype=float)
        acc_trials = np.asarray(curve["acc_trials"], dtype=float)
        if acc_trials.ndim != 2:
            continue
        n_k, n_trials = acc_trials.shape
        if n_k < 2 or n_trials < 2:
            continue

        mean_curve = np.nanmean(acc_trials, axis=1)
        mean_curve_mon = _monotone_curve(mean_curve)

        sample_idx = rng.integers(0, n_trials, size=(int(n_boot), n_trials))
        bs_curves = np.empty((int(n_boot), n_k), dtype=float)
        for b in range(int(n_boot)):
            bs_curves[b] = np.nanmean(acc_trials[:, sample_idx[b]], axis=1)
        bs_curves_mon = np.empty_like(bs_curves)
        for b in range(int(n_boot)):
            bs_curves_mon[b] = _monotone_curve(bs_curves[b])

        kx_boot = np.full((int(n_boot), int(thresholds_arr.size)), np.nan, dtype=np.float32) if replicate_sink is not None else None
        for t_idx, X_thr in enumerate(thresholds_arr):
            status_c, k_center = _invert_threshold_on_curve(k_grid, mean_curve_mon, float(X_thr))
            k_samples: List[float] = []
            n_ok_boot = 0
            n_left_censored_boot = 0
            n_right_censored_boot = 0
            n_invalid_boot = 0
            for b in range(int(n_boot)):
                st, kv = _invert_threshold_on_curve(k_grid, bs_curves_mon[b], float(X_thr))
                if st == "ok" and _is_finite(kv):
                    vv = float(kv)
                    k_samples.append(vv)
                    n_ok_boot += 1
                    if kx_boot is not None:
                        kx_boot[b, t_idx] = vv
                elif st == "left_censored":
                    n_left_censored_boot += 1
                elif st == "right_censored":
                    n_right_censored_boot += 1
                else:
                    n_invalid_boot += 1
            k_arr = np.asarray(k_samples, dtype=float)

            row = {
                "channel": str(curve["channel"]),
                "p": float(curve["p"]),
                "n_q": int(curve["n_q"]),
                "threshold": float(X_thr),
                "status": status_c,
                "k_x": float(k_center) if (status_c == "ok" and _is_finite(k_center)) else float("nan"),
                "k_x_lo": float("nan"),
                "k_x_hi": float("nan"),
                "k_x_err_lo": float("nan"),
                "k_x_err_hi": float("nan"),
                "log2_nps": float(curve["n_q"] * k_center) if (status_c == "ok" and _is_finite(k_center)) else float("nan"),
                "n_valid": int(k_arr.size),
                "n_k": int(n_k),
                "n_boot": int(n_boot),
                "n_ok_boot": int(n_ok_boot),
                "n_left_censored_boot": int(n_left_censored_boot),
                "n_right_censored_boot": int(n_right_censored_boot),
                "n_censored_boot": int(n_left_censored_boot + n_right_censored_boot),
                "n_invalid_boot": int(n_invalid_boot),
                "ci_level": float(ci_level),
                "n_trials": int(n_trials),
                "run_name": str(curve.get("run_name", "")),
                "source": "ml",
            }
            if k_arr.size > 0:
                lo = float(np.nanpercentile(k_arr, q_lo))
                hi = float(np.nanpercentile(k_arr, q_hi))
                med = float(np.nanmedian(k_arr))
                row["k_x_lo"] = lo
                row["k_x_hi"] = hi
                if status_c == "ok" and _is_finite(row["k_x"]):
                    row["k_x_err_lo"] = float(max(0.0, row["k_x"] - lo))
                    row["k_x_err_hi"] = float(max(0.0, hi - row["k_x"]))
                elif status_c == "ok":
                    row["k_x"] = med
                    row["log2_nps"] = float(curve["n_q"] * med)
                    row["k_x_err_lo"] = float(max(0.0, med - lo))
                    row["k_x_err_hi"] = float(max(0.0, hi - med))
            rows.append(row)

        if replicate_sink is not None and kx_boot is not None:
            rk = (str(curve["channel"]), float(curve["p"]), int(curve["n_q"]))
            replicate_sink[rk] = {
                "kx_boot": kx_boot,
                "n_boot": int(n_boot),
                "thresholds": np.asarray(thresholds_arr, dtype=float),
                "n_trials": int(n_trials),
                "n_k": int(n_k),
            }

    return _postprocess_rows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Quantum accuracy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_quantum_acc(
    qdata: dict, device: str, channel: str, amplitude: str, readout_error: str, nq: int
) -> float:
    nq_values = [int(v) for v in qdata["nq_values"]]
    try:
        idx = nq_values.index(int(nq))
    except ValueError:
        return float("nan")
    try:
        return float(qdata["curves"][device][channel][amplitude][readout_error][idx])
    except (KeyError, IndexError):
        return float("nan")


def _nq_cutoff(
    qdata: dict, device: str, channel: str, amplitude: str,
    readout_error: str, eta: float, floor: float
) -> Optional[int]:
    """Largest nq where q_acc(nq) − eta ≥ floor.  None if none qualify."""
    cutoff = None
    for nq in qdata["nq_values"]:
        q = _get_quantum_acc(qdata, device, channel, amplitude, readout_error, int(nq))
        if _is_finite(q) and (q - float(eta)) >= float(floor):
            cutoff = int(nq)
    return cutoff


def _random_baseline_eps(
    nq: int,
    eps0: float = RANDOM_BASELINE_EPS0_DEFAULT,
    decay: float = RANDOM_BASELINE_DECAY_DEFAULT,
    nq_ref: int = RANDOM_BASELINE_NQ_REF_DEFAULT,
) -> float:
    """
    Exponential near-random tolerance used for the nps=1 tail override.

    eps(nq) = eps0 * exp(-decay * max(nq - nq_ref, 0))
    """
    e0 = max(0.0, _to_float(eps0))
    lam = max(0.0, _to_float(decay))
    if not _is_finite(e0):
        e0 = float(RANDOM_BASELINE_EPS0_DEFAULT)
    if not _is_finite(lam):
        lam = float(RANDOM_BASELINE_DECAY_DEFAULT)
    dnq = max(0.0, float(int(nq) - int(nq_ref)))
    try:
        return float(e0 * math.exp(-lam * dnq))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Per-point prediction with explicit censoring
# ─────────────────────────────────────────────────────────────────────────────

def _predict_point(
    predictor: Predictor,
    target: float,
    nq: int,
    nq_observed_max: int,
    ci_z: float = 1.0,
) -> dict:
    """
    Predict log2(nps) and nps at (target, nq) with censoring status.

    Censoring semantics:
      left_censored  – target < predictor.threshold_min: task is easier than the
                       easiest calibrated threshold; nps reported is an UPPER BOUND
                       (the true required nps is ≤ this value).
      right_censored – target > predictor.threshold_max: task is harder than the
                       hardest calibrated threshold; nps reported is a LOWER BOUND.
      ok             – target within calibration range; prediction is interpolated.

    Returns dict with keys:
      log2_nps, nps, sigma_log2_nps,
      log2_nps_lo, log2_nps_hi, nps_lo, nps_hi,
      status, is_extrapolated, target_in_range
    """
    base: dict = {
        "log2_nps":       float("nan"),
        "nps":            float("nan"),
        "sigma_log2_nps": float("nan"),
        "sigma_wls_log2_nps": float("nan"),
        "sigma_eff_log2_nps": float("nan"),
        "ci_halfwidth_log2_nps": float("nan"),
        "log2_nps_lo":    float("nan"),
        "log2_nps_hi":    float("nan"),
        "nps_lo":         float("nan"),
        "nps_hi":         float("nan"),
        "status":         "nan",
        "is_extrapolated": int(nq) > int(nq_observed_max),
        "target_in_range": False,
    }
    if not _is_finite(target):
        base["status"] = "target_nan"
        return base

    t_lo = predictor.threshold_min
    t_hi = predictor.threshold_max

    def _eval_log2(thr):
        arr = np.asarray(predictor.predict_log2_nps(float(thr), int(nq)), dtype=float)
        v = float(arr.ravel()[0]) if arr.size > 0 else float("nan")
        return v

    def _safe_nps(log2_v: float) -> float:
        return _safe_nps_from_log2(log2_v)

    if target < t_lo:
        base["status"] = "left_censored"
        v = _eval_log2(t_lo)
        base["log2_nps"] = v
        base["nps"] = _safe_nps(v)
        return base

    if target > t_hi:
        base["status"] = "right_censored"
        v = _eval_log2(t_hi)
        base["log2_nps"] = v
        base["nps"] = _safe_nps(v)
        return base

    base["target_in_range"] = True
    log2_nps = _eval_log2(target)
    if not _is_finite(log2_nps):
        base["status"] = "nan"
        return base

    base["log2_nps"] = log2_nps
    base["nps"]      = _safe_nps(log2_nps)
    base["status"]   = "ok"

    # Optional parametric (WLS-only) sigma for diagnostics.
    if predictor.predict_sigma_log2_nps is not None:
        sig_wls_arr = np.asarray(predictor.predict_sigma_log2_nps(float(target), int(nq)), dtype=float)
        sig_wls = float(sig_wls_arr.ravel()[0]) if sig_wls_arr.size > 0 else float("nan")
        if _is_finite(sig_wls) and sig_wls >= 0:
            base["sigma_wls_log2_nps"] = sig_wls

    # 1σ-equivalent sigma (diagnostic) and calibrated CI halfwidth (for plotting).
    sig_arr = np.asarray(predictor.predict_sigma_eff(float(target), int(nq)), dtype=float)
    sig = float(sig_arr.ravel()[0]) if sig_arr.size > 0 else float("nan")
    if _is_finite(sig) and sig >= 0:
        base["sigma_eff_log2_nps"] = sig
        base["sigma_log2_nps"] = sig

    hw_arr = np.asarray(predictor.predict_ci_halfwidth_log2(float(target), int(nq), ci_z=ci_z), dtype=float)
    hw = float(hw_arr.ravel()[0]) if hw_arr.size > 0 else float("nan")
    if _is_finite(hw) and hw >= 0:
        base["ci_halfwidth_log2_nps"] = hw
        base["log2_nps_lo"]    = log2_nps - hw
        base["log2_nps_hi"]    = log2_nps + hw
        base["nps_lo"]         = _safe_nps(base["log2_nps_lo"])
        base["nps_hi"]         = _safe_nps(base["log2_nps_hi"])
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Forward-chaining CV on Step4 fits (multi-threshold)
# ─────────────────────────────────────────────────────────────────────────────

def _forward_chain_cv_multi_thr(
    case_rows: List[dict],
    n_thr_pcts: int = 5,
    *,
    keep_details: bool = True,
    pct_min: float = CV_THR_PCT_MIN,
    pct_max: float = CV_THR_PCT_MAX,
    max_horizon: int = CV_MAX_HORIZON,
) -> dict:
    """
    Forward-chaining CV on the Step4 fit evaluated at multiple threshold slices.

    Samples thresholds at n_thr_pcts evenly-spaced percentiles over an interior
    range (default p10..p90) of the available threshold grid, snaps each to the
    nearest grid point, then runs grow-by-one forward validation at each threshold
    with horizons h=1..max_horizon from the same training cut t.

    Returns dict with:
      applicable, n_thresholds_evaluated, n_total_horizons,
      rmse_pooled, mae_pooled, bias_pooled  – pooled across all threshold slices,
      rmse_max                               – max per-slice RMSE (conservative floor
                                               used as σ_CV for CI inflation),
      rmse_max_ratio                         – rmse_max / rmse_pooled,
      q50_abs/q68_abs/q90_abs/q95_abs       – pooled |error| quantiles,
      per_threshold                          – per-slice stats keyed by threshold value,
      per_horizon                            – pooled stats for each horizon h.
    """
    ok_rows = [r for r in case_rows if _is_inference_row(r)]
    if not ok_rows:
        return {"applicable": False, "reason": "no_ok_rows"}

    thresholds = sorted({_thr_key(float(r["threshold"])) for r in ok_rows})
    if len(thresholds) < 3:
        return {"applicable": False, "reason": f"only_{len(thresholds)}_thresholds"}

    # Snap interior percentile targets to actual grid thresholds.
    p_lo = float(pct_min) if _is_finite(pct_min) else float(CV_THR_PCT_MIN)
    p_hi = float(pct_max) if _is_finite(pct_max) else float(CV_THR_PCT_MAX)
    if not (0.0 <= p_lo < p_hi <= 100.0):
        p_lo, p_hi = float(CV_THR_PCT_MIN), float(CV_THR_PCT_MAX)
    n_thr_eff = max(3, int(n_thr_pcts))
    pcts = np.linspace(p_lo, p_hi, n_thr_eff)
    selected = sorted(set(
        min(thresholds, key=lambda t: abs(t - float(np.percentile(thresholds, p))))
        for p in pcts
    ))

    max_h = int(max_horizon) if int(max_horizon) > 0 else 1
    all_errors_log2: List[float] = []
    all_errors_k: List[float] = []
    all_errors_log2_by_h: Dict[int, List[float]] = {}
    all_errors_k_by_h: Dict[int, List[float]] = {}
    per_thr: dict = {}
    for rep_thr in selected:
        thr_rows = sorted(
            [r for r in ok_rows if _thr_key(float(r["threshold"])) == _thr_key(float(rep_thr))],
            key=lambda r: int(r["n_q"]),
        )
        if len(thr_rows) < 3:
            continue
        nq_vals  = np.array([int(r["n_q"])       for r in thr_rows], dtype=float)
        kx_vals  = np.array([float(r["k_x"])     for r in thr_rows], dtype=float)
        sig_vals = np.array([float(r["sigma_k"]) for r in thr_rows], dtype=float)

        errors_log2: List[float] = []
        errors_k: List[float] = []
        errors_log2_by_h: Dict[int, List[float]] = {}
        errors_k_by_h: Dict[int, List[float]] = {}
        for t in range(2, len(nq_vals)):
            fit = _weighted_linear_fit(nq_vals[:t], kx_vals[:t], sig_vals[:t])
            if not fit["ok_fit"]:
                continue
            for h in range(1, max_h + 1):
                j = t + (h - 1)
                if j >= len(nq_vals):
                    break
                nq_j   = float(nq_vals[j])
                kx_pred = float(_step4_predict(nq_j, fit["coef"]))
                if not (_is_finite(nq_j) and nq_j > 0):
                    continue
                err_log2 = nq_j * kx_pred - nq_j * float(kx_vals[j])
                err_k = err_log2 / nq_j
                errors_log2.append(err_log2)
                errors_k.append(err_k)
                errors_log2_by_h.setdefault(h, []).append(err_log2)
                errors_k_by_h.setdefault(h, []).append(err_k)
                all_errors_log2_by_h.setdefault(h, []).append(err_log2)
                all_errors_k_by_h.setdefault(h, []).append(err_k)

        if not errors_log2:
            continue
        ea = np.array(errors_log2, dtype=float)
        ek = np.array(errors_k, dtype=float)
        abs_ea = np.abs(ea)
        abs_ek = np.abs(ek)
        thr_out = {
            "rmse":       float(np.sqrt(np.mean(ea ** 2))),
            "mae":        float(np.mean(abs_ea)),
            "bias":       float(np.mean(ea)),
            "rmse_k":     float(np.sqrt(np.mean(ek ** 2))),
            "mae_k":      float(np.mean(abs_ek)),
            "bias_k":     float(np.mean(ek)),
            "n_horizons": int(len(errors_log2)),
            "q50_abs":    float(np.percentile(abs_ea, 50.0)),
            "q68_abs":    float(np.percentile(abs_ea, 68.27)),
            "q90_abs":    float(np.percentile(abs_ea, 90.0)),
            "q95_abs":    float(np.percentile(abs_ea, 95.0)),
            "q50_abs_k":  float(np.percentile(abs_ek, 50.0)),
            "q68_abs_k":  float(np.percentile(abs_ek, 68.27)),
            "q90_abs_k":  float(np.percentile(abs_ek, 90.0)),
            "q95_abs_k":  float(np.percentile(abs_ek, 95.0)),
        }
        if keep_details:
            thr_h = {}
            for h, errs_h in sorted(errors_log2_by_h.items()):
                eh = np.asarray(errs_h, dtype=float)
                ek_h = np.asarray(errors_k_by_h.get(h, []), dtype=float)
                aeh = np.abs(eh)
                aek = np.abs(ek_h)
                thr_h[str(int(h))] = {
                    "n": int(eh.size),
                    "rmse": float(np.sqrt(np.mean(eh ** 2))),
                    "mae": float(np.mean(aeh)),
                    "bias": float(np.mean(eh)),
                    "rmse_k": float(np.sqrt(np.mean(ek_h ** 2))) if ek_h.size else float("nan"),
                    "mae_k": float(np.mean(aek)) if aek.size else float("nan"),
                    "bias_k": float(np.mean(ek_h)) if ek_h.size else float("nan"),
                    "q68_abs": float(np.percentile(aeh, 68.27)),
                    "q90_abs": float(np.percentile(aeh, 90.0)),
                    "q95_abs": float(np.percentile(aeh, 95.0)),
                    "q68_abs_k": float(np.percentile(aek, 68.27)) if aek.size else float("nan"),
                    "q90_abs_k": float(np.percentile(aek, 90.0)) if aek.size else float("nan"),
                    "q95_abs_k": float(np.percentile(aek, 95.0)) if aek.size else float("nan"),
                }
            thr_out["per_horizon"] = thr_h
        per_thr[float(rep_thr)] = thr_out
        all_errors_log2.extend(errors_log2)
        all_errors_k.extend(errors_k)

    if not all_errors_log2:
        return {"applicable": False, "reason": "no_valid_horizons"}

    ea_all    = np.array(all_errors_log2, dtype=float)
    ek_all    = np.array(all_errors_k, dtype=float)
    abs_all   = np.abs(ea_all)
    abs_all_k = np.abs(ek_all)
    rmse_vals = [
        float(v["rmse"]) for v in per_thr.values()
        if int(v.get("n_horizons", 0)) >= int(CV_MIN_SLICE_HORIZONS)
    ]
    if not rmse_vals:
        rmse_vals = [float(v["rmse"]) for v in per_thr.values()]
    rmse_k_vals = [
        float(v["rmse_k"]) for v in per_thr.values()
        if int(v.get("n_horizons", 0)) >= int(CV_MIN_SLICE_HORIZONS) and _is_finite(v.get("rmse_k", float("nan")))
    ]
    if not rmse_k_vals:
        rmse_k_vals = [float(v["rmse_k"]) for v in per_thr.values() if _is_finite(v.get("rmse_k", float("nan")))]
    rmse_pooled = float(np.sqrt(np.mean(ea_all ** 2)))
    rmse_pooled_k = float(np.sqrt(np.mean(ek_all ** 2)))
    rmse_max = float(max(rmse_vals))
    rmse_max_k = float(max(rmse_k_vals)) if rmse_k_vals else float("nan")
    rmse_ratio = float(rmse_max / rmse_pooled) if rmse_pooled > 0 else (1.0 if rmse_max == 0 else float("inf"))
    rmse_ratio_k = (
        float(rmse_max_k / rmse_pooled_k)
        if (_is_finite(rmse_max_k) and rmse_pooled_k > 0)
        else (1.0 if (_is_finite(rmse_max_k) and rmse_max_k == 0) else float("inf"))
    )

    per_h = {}
    for h, errs_h in sorted(all_errors_log2_by_h.items()):
        eh = np.asarray(errs_h, dtype=float)
        ek_h = np.asarray(all_errors_k_by_h.get(h, []), dtype=float)
        aeh = np.abs(eh)
        aek = np.abs(ek_h)
        per_h[str(int(h))] = {
            "n": int(eh.size),
            "rmse": float(np.sqrt(np.mean(eh ** 2))),
            "mae": float(np.mean(aeh)),
            "bias": float(np.mean(eh)),
            "rmse_k": float(np.sqrt(np.mean(ek_h ** 2))) if ek_h.size else float("nan"),
            "mae_k": float(np.mean(aek)) if aek.size else float("nan"),
            "bias_k": float(np.mean(ek_h)) if ek_h.size else float("nan"),
            "q68_abs": float(np.percentile(aeh, 68.27)),
            "q90_abs": float(np.percentile(aeh, 90.0)),
            "q95_abs": float(np.percentile(aeh, 95.0)),
            "q68_abs_k": float(np.percentile(aek, 68.27)) if aek.size else float("nan"),
            "q90_abs_k": float(np.percentile(aek, 90.0)) if aek.size else float("nan"),
            "q95_abs_k": float(np.percentile(aek, 95.0)) if aek.size else float("nan"),
        }

    out = {
        "applicable":             True,
        "max_horizon":            int(max_h),
        "threshold_percentiles":  [float(v) for v in pcts.tolist()],
        "threshold_percentile_window": [float(p_lo), float(p_hi)],
        "n_thresholds_evaluated": int(len(per_thr)),
        "n_total_horizons":       int(len(all_errors_log2)),
        "n_total_pooled_horizons": int(len(all_errors_log2)),
        "n_total_horizons_h1":    int(len(all_errors_log2_by_h.get(1, []))),
        "rmse_pooled":            rmse_pooled,
        "mae_pooled":             float(np.mean(np.abs(ea_all))),
        "bias_pooled":            float(np.mean(ea_all)),
        "rmse_pooled_k":          rmse_pooled_k,
        "mae_pooled_k":           float(np.mean(np.abs(ek_all))),
        "bias_pooled_k":          float(np.mean(ek_all)),
        "rmse_max":               rmse_max,
        "rmse_max_k":             rmse_max_k,
        "rmse_max_ratio":         rmse_ratio,
        "rmse_max_ratio_k":       rmse_ratio_k,
        "q50_abs":                float(np.percentile(abs_all, 50.0)),
        "q68_abs":                float(np.percentile(abs_all, 68.27)),
        "q90_abs":                float(np.percentile(abs_all, 90.0)),
        "q95_abs":                float(np.percentile(abs_all, 95.0)),
        "q50_abs_k":              float(np.percentile(abs_all_k, 50.0)),
        "q68_abs_k":              float(np.percentile(abs_all_k, 68.27)),
        "q90_abs_k":              float(np.percentile(abs_all_k, 90.0)),
        "q95_abs_k":              float(np.percentile(abs_all_k, 95.0)),
        "per_threshold":          {f"{k:.4f}": v for k, v in per_thr.items()},
        "per_horizon":            per_h,
        "__abs_errors_k":         [float(v) for v in abs_all_k.tolist()],
        "__abs_errors_k_h1":      [float(v) for v in np.abs(np.asarray(all_errors_k_by_h.get(1, []), dtype=float)).tolist()],
        "__abs_errors_k_h2plus":  [float(v) for h, vals in all_errors_k_by_h.items() if int(h) >= 2 for v in np.abs(np.asarray(vals, dtype=float)).tolist()],
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_comparison(
    qdata: dict,
    hg_predictors: Dict[Tuple[str, str], Predictor],
    eig_predictors: Dict[Tuple[str, str], Predictor],
    ml_predictors:  Dict[Tuple[str, str], Predictor],
    devices: List[str],
    readout_errors: List[str],
    channels: List[str],
    amplitudes: List[str],
    etas: List[float],
    nq_pred: np.ndarray,
    target_floor: float = TARGET_FLOOR,
    ci_z: float = 1.0,
    random_baseline_fix: bool = RANDOM_BASELINE_FIX_DEFAULT,
    random_baseline_eps0: float = RANDOM_BASELINE_EPS0_DEFAULT,
    random_baseline_decay: float = RANDOM_BASELINE_DECAY_DEFAULT,
    random_baseline_nq_ref: int = RANDOM_BASELINE_NQ_REF_DEFAULT,
) -> Tuple[List[dict], dict]:
    """
    Query classical predictors at q_acc(nq) − η for every (device, re, ch, amp, eta, nq).

    Censoring is preserved: left/right_censored points are included with a bound
    annotation rather than being dropped.

    Returns (rows, diagnostics).
    """
    rows:       List[dict] = []
    diag_cases: List[dict] = []

    obs_max_by_method = {mk: max(v) for mk, v in OBS_NQ_BY_METHOD.items()}
    obs_set_by_method = {mk: set(v) for mk, v in OBS_NQ_BY_METHOD.items()}

    for ch in channels:
        for amp_str in amplitudes:
            for device in devices:
                for re in readout_errors:
                    for eta in etas:
                        nq_cut = _nq_cutoff(qdata, device, ch, amp_str, re, eta, target_floor)

                        for mk, preds in [
                            ("hypergraph", hg_predictors),
                            ("eigenshadow", eig_predictors),
                            ("ml", ml_predictors),
                        ]:
                            obs_max = int(obs_max_by_method[mk])
                            obs_set = obs_set_by_method[mk]
                            pred = preds.get((ch, amp_str))
                            if pred is None:
                                diag_cases.append({
                                    "device": device, "re": re, "ch": ch,
                                    "amp": amp_str, "eta": eta, "method": mk,
                                    "status": "no_predictor",
                                })
                                continue
                            extrap_trusted = bool(pred.meta.get("extrapolation_trusted", True))
                            extrap_reason = str(pred.meta.get("extrapolation_reason", "ok"))
                            interval_req = str(pred.meta.get("interval_method_requested", getattr(pred, "interval_method", "cv")))
                            interval_eff = str(
                                pred.meta.get(
                                    "interval_method_effective",
                                    pred.meta.get("interval_method", getattr(pred, "interval_method", "cv")),
                                )
                            )
                            fb_meta = pred.meta.get("interval_method_fallback", {})
                            fb_reason = str(fb_meta.get("reason", "")) if isinstance(fb_meta, dict) else ""
                            fb_n_abs = (
                                int(_to_float(fb_meta.get("n_abs_errors", 0)))
                                if isinstance(fb_meta, dict) and _is_finite(fb_meta.get("n_abs_errors", 0))
                                else 0
                            )
                            conf_meta = pred.meta.get("conformal_calibration", {})
                            conf_n_abs = (
                                int(_to_float(conf_meta.get("n_abs_errors", 0)))
                                if isinstance(conf_meta, dict) and _is_finite(conf_meta.get("n_abs_errors", 0))
                                else 0
                            )
                            conf_holdout_nq = (
                                [int(v) for v in conf_meta.get("holdout_nq", [])]
                                if isinstance(conf_meta, dict) and isinstance(conf_meta.get("holdout_nq", []), list)
                                else []
                            )
                            strict_req = bool(pred.meta.get("strict_split_conformal_requested", False))
                            strict_active = bool(pred.meta.get("strict_split_conformal_active", False))
                            strict_reason = str(pred.meta.get("strict_split_reason", ""))

                            for nq_i in nq_pred:
                                nq = int(nq_i)
                                q_acc  = _get_quantum_acc(qdata, device, ch, amp_str, re, nq)
                                target = (q_acc - float(eta)) if _is_finite(q_acc) else float("nan")
                                is_meaningful = _is_finite(q_acc) and (q_acc - float(eta)) >= target_floor
                                is_trivial    = (nq_cut is not None) and (nq > nq_cut)
                                is_near_random = not is_meaningful
                                eps_nq = _random_baseline_eps(
                                    nq=nq,
                                    eps0=float(random_baseline_eps0),
                                    decay=float(random_baseline_decay),
                                    nq_ref=int(random_baseline_nq_ref),
                                )
                                random_baseline_thr = float(L_BASELINE + eps_nq) if _is_finite(eps_nq) else float("nan")
                                use_random_baseline = (
                                    bool(random_baseline_fix)
                                    and _is_finite(target)
                                    and _is_finite(random_baseline_thr)
                                    and (target < random_baseline_thr)
                                )

                                if use_random_baseline:
                                    # Manual near-random tail rule (visual convention only):
                                    # if target is effectively random baseline, display nps=1.
                                    # This is NOT an inferred resource estimate, so uncertainty
                                    # fields are left NaN and this status is excluded from fits.
                                    pt = {
                                        "log2_nps": 0.0, "nps": 1.0,
                                        "sigma_log2_nps": float("nan"),
                                        "sigma_wls_log2_nps": float("nan"),
                                        "sigma_eff_log2_nps": float("nan"),
                                        "ci_halfwidth_log2_nps": float("nan"),
                                        "log2_nps_lo": float("nan"), "log2_nps_hi": float("nan"),
                                        "nps_lo": float("nan"), "nps_hi": float("nan"),
                                        "status": "random_baseline",
                                        "is_extrapolated": int(nq) > int(obs_max),
                                        "target_in_range": False,
                                    }
                                elif is_near_random:
                                    # Target is below the meaningful floor (≈ random chance).
                                    # Do not query the predictor; suppress from plots entirely.
                                    pt = {
                                        "log2_nps": float("nan"), "nps": float("nan"),
                                        "sigma_log2_nps": float("nan"),
                                        "sigma_wls_log2_nps": float("nan"),
                                        "sigma_eff_log2_nps": float("nan"),
                                        "ci_halfwidth_log2_nps": float("nan"),
                                        "log2_nps_lo": float("nan"), "log2_nps_hi": float("nan"),
                                        "nps_lo": float("nan"), "nps_hi": float("nan"),
                                        "status": "near_random",
                                        "is_extrapolated": int(nq) > int(obs_max),
                                        "target_in_range": False,
                                    }
                                elif (nq > int(obs_max)) and (not extrap_trusted):
                                    # Extrapolation is explicitly disallowed by trust gate.
                                    pt = {
                                        "log2_nps": float("nan"), "nps": float("nan"),
                                        "sigma_log2_nps": float("nan"),
                                        "sigma_wls_log2_nps": float("nan"),
                                        "sigma_eff_log2_nps": float("nan"),
                                        "ci_halfwidth_log2_nps": float("nan"),
                                        "log2_nps_lo": float("nan"), "log2_nps_hi": float("nan"),
                                        "nps_lo": float("nan"), "nps_hi": float("nan"),
                                        "status": "untrusted_extrapolation",
                                        "is_extrapolated": True,
                                        "target_in_range": False,
                                    }
                                else:
                                    pt = _predict_point(pred, target, nq, obs_max, ci_z=ci_z)

                                rows.append({
                                    "device":        device,
                                    "readout_error": re,
                                    "channel":       ch,
                                    "amplitude":     amp_str,
                                    "eta":           float(eta),
                                    "method":        mk,
                                    "nq":            nq,
                                    "q_acc":         q_acc,
                                    "target_threshold": target,
                                    "is_observed":      nq in obs_set,
                                    "is_extrapolated":  bool(pt["is_extrapolated"]),
                                    "is_trivial_regime": is_trivial,
                                    "is_meaningful":     is_meaningful,
                                    "is_near_random":    is_near_random,
                                    "is_random_baseline_fix": use_random_baseline,
                                    "is_inference_point": bool(pt["status"] == "ok"),
                                    "random_baseline_eps": eps_nq,
                                    "random_baseline_threshold": random_baseline_thr,
                                    "nq_cutoff":         nq_cut,
                                    "pred_status":       pt["status"],
                                    "target_in_pred_range": bool(pt["target_in_range"]),
                                    "extrapolation_trusted": extrap_trusted,
                                    "extrapolation_reason": extrap_reason,
                                    "interval_method_requested": interval_req,
                                    "interval_method_effective": interval_eff,
                                    "interval_method_fallback_reason": fb_reason,
                                    "interval_method_fallback_n_abs_errors": fb_n_abs,
                                    "conformal_n_abs_errors": conf_n_abs,
                                    "conformal_holdout_nq": "|".join(str(v) for v in conf_holdout_nq),
                                    "strict_split_conformal_requested": strict_req,
                                    "strict_split_conformal_active": strict_active,
                                    "strict_split_reason": strict_reason,
                                    "interval_uncertainty_source": "predictor_calibrated",
                                    "curve_bootstrap_n_valid": float("nan"),
                                    "curve_bootstrap_fit_mode": "",
                                    "log2_nps":          pt["log2_nps"],
                                    "nps":               pt["nps"],
                                    "sigma_log2_nps":    pt["sigma_log2_nps"],
                                    "sigma_wls_log2_nps": pt.get("sigma_wls_log2_nps", float("nan")),
                                    "sigma_eff_log2_nps": pt.get("sigma_eff_log2_nps", pt.get("sigma_log2_nps", float("nan"))),
                                    "ci_halfwidth_log2_nps": pt.get("ci_halfwidth_log2_nps", float("nan")),
                                    "log2_nps_lo":       pt["log2_nps_lo"],
                                    "log2_nps_hi":       pt["log2_nps_hi"],
                                    "nps_lo":            pt["nps_lo"],
                                    "nps_hi":            pt["nps_hi"],
                                    "predictor_name":    pred.name,
                                    "predictor_thr_min": pred.threshold_min,
                                    "predictor_thr_max": pred.threshold_max,
                                    "sigma_cv":          pred.sigma_cv,
                                    "sigma_cv_k":        pred.sigma_cv_k,
                                    "sigma_cv_k_obs":    pred.sigma_cv_k_obs,
                                    "sigma_cv_k_extrap": pred.sigma_cv_k_extrap,
                                    "sigma_cv_log2_at_obs_max": pred.sigma_cv_log2_at_obs_max,
                                    "sigma_cv_log2_gate_at_obs_max": pred.sigma_cv_log2_gate_at_obs_max,
                                })

    return rows, {"cases": diag_cases}


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    cols = [
        "device", "readout_error", "channel", "amplitude", "eta", "method",
        "nq", "q_acc", "target_threshold",
        "is_observed", "is_extrapolated", "is_trivial_regime", "is_meaningful", "is_near_random",
        "is_random_baseline_fix", "is_inference_point", "random_baseline_eps", "random_baseline_threshold",
        "nq_cutoff", "pred_status", "target_in_pred_range",
        "extrapolation_trusted", "extrapolation_reason",
        "interval_method_requested", "interval_method_effective",
        "interval_method_fallback_reason", "interval_method_fallback_n_abs_errors",
        "conformal_n_abs_errors", "conformal_holdout_nq",
        "strict_split_conformal_requested", "strict_split_conformal_active", "strict_split_reason",
        "interval_uncertainty_source", "curve_bootstrap_n_valid", "curve_bootstrap_fit_mode",
        "log2_nps", "nps", "sigma_log2_nps", "sigma_wls_log2_nps", "sigma_eff_log2_nps", "ci_halfwidth_log2_nps",
        "sigma_cv", "sigma_cv_k", "sigma_cv_k_obs", "sigma_cv_k_extrap", "sigma_cv_log2_at_obs_max", "sigma_cv_log2_gate_at_obs_max",
        "log2_nps_lo", "log2_nps_hi", "nps_lo", "nps_hi",
        "predictor_name", "predictor_thr_min", "predictor_thr_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, bool):
                    s = "true" if v else "false"
                elif v is None or (isinstance(v, float) and math.isnan(v)):
                    s = ""
                else:
                    s = str(v)
                if any(ch_c in s for ch_c in [",", '"', "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _eta_palette(etas: List[float]) -> dict:
    if plt is None:
        return {e: None for e in etas}
    cmap = plt.get_cmap("plasma")
    n = max(1, len(etas))
    return {e: cmap(i / (n - 1) if n > 1 else 0.5) for i, e in enumerate(sorted(etas))}


def _plot_comparison_figure(
    rows: List[dict],
    etas: List[float],
    device: str,
    readout_error: str,
    channels: List[str],
    amplitudes: List[str],
    nq_pred: np.ndarray,
    out_path: Path,
    title: str,
    ci_z: float = 1.0,
    show_bands: bool = True,
    target_floor: float = TARGET_FLOOR,
) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {_MPL_IMPORT_ERROR}")

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 12), sharex=True, sharey=True)
    axes = np.asarray(axes)
    colors = _eta_palette(etas)

    # Index by (ch, amp, mk, eta, nq) for O(1) lookup
    idx: Dict[tuple, dict] = {}
    for r in rows:
        if r.get("device") != device or r.get("readout_error") != readout_error:
            continue
        key = (r["channel"], r["amplitude"], r["method"], float(r["eta"]), int(r["nq"]))
        idx[key] = r

    obs_max_by_method = {mk: max(v) for mk, v in OBS_NQ_BY_METHOD.items()}
    obs_set_by_method = {mk: set(v) for mk, v in OBS_NQ_BY_METHOD.items()}
    any_gate_shading = False
    any_rb_marker = False

    def _as_bool(v, default: bool = True) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return default

    for row_i, ch in enumerate(channels):
        for col_j, amp in enumerate(amplitudes):
            ax = axes[row_i, col_j]
            ax.set_title(f"{CHANNEL_TITLES.get(ch, ch)} | p={amp}", fontsize=10)

            drawn_obs_ext_line = set()   # avoid duplicate axvlines per subplot
            nq_arr_all = np.asarray(nq_pred, dtype=int)
            nq_max_plot = int(nq_arr_all[-1])

            # Visual cue for intentionally suppressed extrapolation:
            # if a method is marked untrusted, lightly shade nq > obs_max.
            for mk in METHOD_ORDER:
                om = int(obs_max_by_method[mk])
                if om >= nq_max_plot:
                    continue
                trusted_vals = []
                for eta in sorted(etas):
                    rec = idx.get((ch, amp, mk, eta, int(om)))
                    if rec is None:
                        rec = idx.get((ch, amp, mk, eta, int(om + 1)))
                    if rec is None:
                        continue
                    trusted_vals.append(_as_bool(rec.get("extrapolation_trusted", True), default=True))
                if trusted_vals and (not all(trusted_vals)):
                    any_gate_shading = True
                    ax.axvspan(
                        om + 0.5, nq_max_plot + 0.5,
                        color=METHOD_COLORS.get(mk, "gray"),
                        alpha=0.055 if mk == "hypergraph" else 0.040,
                        zorder=0,
                        label="_nolegend_",
                    )
                    ax.axvline(
                        x=om + 0.5,
                        color=METHOD_COLORS.get(mk, "gray"),
                        linestyle="--",
                        linewidth=0.6,
                        alpha=0.35,
                        zorder=2,
                    )

            for eta in sorted(etas):
                col = colors[eta]
                for mk in METHOD_ORDER:
                    nq_arr = nq_arr_all
                    om = obs_max_by_method[mk]

                    def _get(field, default=float("nan")):
                        return np.array([
                            _to_float(idx.get((ch, amp, mk, eta, nq), {}).get(field, default))
                            for nq in nq_arr
                        ])

                    log2_nps = _get("log2_nps")
                    lo_l2    = _get("log2_nps_lo")
                    hi_l2    = _get("log2_nps_hi")
                    # Clip before exponentiation to prevent float64 overflow
                    def _safe_pow2(a):
                        a = np.asarray(a, dtype=float)
                        # n_ps is a count and must be >= 1.
                        aa = np.where(np.isfinite(a), np.maximum(a, 0.0), np.nan)
                        return np.where(np.isfinite(aa), np.power(2.0, np.clip(aa, -1020.0, 1020.0)), float("nan"))
                    nps    = _safe_pow2(log2_nps)
                    nps_lo = _safe_pow2(lo_l2)
                    nps_hi = _safe_pow2(hi_l2)

                    trivial    = np.array([bool(idx.get((ch, amp, mk, eta, nq), {}).get("is_trivial_regime", False)) for nq in nq_arr])
                    meaningful = np.array([bool(idx.get((ch, amp, mk, eta, nq), {}).get("is_meaningful", True))       for nq in nq_arr])
                    rb_fix     = np.array([bool(idx.get((ch, amp, mk, eta, nq), {}).get("is_random_baseline_fix", False)) for nq in nq_arr])
                    status     = np.array([str(idx.get((ch, amp, mk, eta, nq), {}).get("pred_status", ""))            for nq in nq_arr])

                    # ok_plot: finite + not trivial + (meaningful OR manual random-baseline override)
                    ok_plot = np.isfinite(log2_nps) & ~trivial & (meaningful | rb_fix)
                    # line_ok: only inferred predictions — visual convention points are markers only
                    line_ok = ok_plot & (status == "ok")
                    # random-baseline convention (nps=1) shown as markers only
                    rb_show = ok_plot & (status == "random_baseline")
                    # censored_show: meaningful censored bounds shown as markers, line stops here
                    censored_show = ok_plot & ((status == "left_censored") | (status == "right_censored"))

                    nps_plot = np.where(line_ok, nps,    float("nan"))
                    lo_plot  = np.where(line_ok, nps_lo, float("nan"))
                    hi_plot  = np.where(line_ok, nps_hi, float("nan"))

                    lw = METHOD_LINEWIDTHS[mk]

                    # ── CI band (only where line is drawn) ───────────────────
                    if show_bands:
                        vb = np.isfinite(lo_plot) & np.isfinite(hi_plot) & (lo_plot > 0) & (hi_plot > 0)
                        if vb.any():
                            ax.fill_between(
                                nq_arr[vb], lo_plot[vb], hi_plot[vb],
                                color=col, alpha=0.10 if mk == "hypergraph" else 0.07,
                                linewidth=0, zorder=1,
                            )

                    # ── Observed segment (solid) ──────────────────────────────
                    obs_seg = line_ok & (nq_arr <= om)
                    if obs_seg.any():
                        ax.plot(nq_arr[obs_seg], nps_plot[obs_seg],
                                color=col, linestyle="-", linewidth=lw, alpha=0.92, zorder=3)

                    # ── Extrapolated segment (dashed) ─────────────────────────
                    ext_seg = line_ok & (nq_arr > om)
                    if ext_seg.any():
                        ax.plot(nq_arr[ext_seg], nps_plot[ext_seg],
                                color=col, linestyle="--", linewidth=lw, alpha=0.80, zorder=3)

                    # ── Random-baseline convention markers (not inferred points) ──
                    rb_vals = np.where(rb_show, nps, float("nan"))
                    vb_rb = np.isfinite(rb_vals) & (rb_vals > 0)
                    if vb_rb.any():
                        any_rb_marker = True
                        ax.scatter(
                            nq_arr[vb_rb], rb_vals[vb_rb],
                            color=col, marker="x", s=26, alpha=0.85,
                            linewidths=1.0, zorder=5
                        )

                    # ── Censored markers (△ right-censored, ▽ left-censored) ──
                    # Line terminates at last ok point; censored regime shown as markers only.
                    for nq_c in nq_arr[censored_show]:
                        r_c = idx.get((ch, amp, mk, eta, int(nq_c)), {})
                        v_c = _to_float(r_c.get("nps", float("nan")))
                        if not (_is_finite(v_c) and v_c > 0):
                            continue
                        marker = "^" if r_c.get("pred_status") == "right_censored" else "v"
                        ax.scatter(nq_c, v_c, color=col, marker=marker, s=22, alpha=0.75,
                                   edgecolors="k", linewidths=0.4, zorder=5)

                    # ── Raw observed-point scatter (ok predictions at observed nq) ──
                    for nq_c in obs_set_by_method[mk]:
                        r_c = idx.get((ch, amp, mk, eta, nq_c), {})
                        if r_c.get("is_trivial_regime") or not r_c.get("is_meaningful"):
                            continue
                        if r_c.get("pred_status", "") != "ok":
                            continue
                        v_c = _to_float(r_c.get("nps", float("nan")))
                        if not (_is_finite(v_c) and v_c > 0):
                            continue
                        ax.scatter(nq_c, v_c, color=col, marker=METHOD_MARKERS[mk],
                                   s=18 if mk == "hypergraph" else 14,
                                   linewidths=0.6, edgecolors="k", alpha=0.85, zorder=4)

                    # ── Obs→extrap transition line (once per method per subplot) ──
                    vl_key = (ch, amp, mk)
                    if vl_key not in drawn_obs_ext_line:
                        ax.axvline(x=om + 0.5, color="gray", linewidth=0.4,
                                   linestyle=":", alpha=0.35, zorder=2)
                        drawn_obs_ext_line.add(vl_key)

            ax.set_yscale("log")
            ax.grid(alpha=0.22)
            if row_i == 2:
                ax.set_xlabel("Qubits (n_q)", fontsize=9)
            if col_j == 0:
                ax.set_ylabel("Classical n_ps required (log scale)", fontsize=9)

    for ax in axes.flat:
        ax.set_xlim(nq_pred[0] - 0.5, nq_pred[-1] + 0.5)

    # ── Legends ───────────────────────────────────────────────────────────────
    colors_legend = _eta_palette(etas)
    eta_handles = [Line2D([0], [0], color=colors_legend[e], linewidth=2.2) for e in sorted(etas)]
    eta_labels  = [f"η={e:.2f}" for e in sorted(etas)]

    method_handles = [
        Line2D(
            [0], [0], color="gray", linestyle="-",
            linewidth=METHOD_LINEWIDTHS[mk],
            marker=METHOD_MARKERS[mk], markersize=4, label=METHOD_LABELS[mk]
        )
        for mk in METHOD_ORDER
    ]
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.8, label="Observed range"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="Extrapolated (Step4)"),
        Line2D([0], [0], color="gray", marker="^", linestyle="", markersize=6,
               label="Right-censored (lower bound)"),
        Line2D([0], [0], color="gray", marker="v", linestyle="", markersize=6,
               label="Left-censored (upper bound)"),
    ]
    if show_bands:
        style_handles.insert(2, Patch(facecolor="gray", alpha=0.25,
                                      label=f"Calibrated CI (empirical CV + WLS, z={ci_z:.1g})"))
    if any_gate_shading:
        style_handles.append(
            Patch(facecolor="gray", alpha=0.18, label="Extrapolation gated off (untrusted)")
        )
    if any_rb_marker:
        style_handles.append(
            Line2D([0], [0], color="gray", marker="x", linestyle="", markersize=6,
                   label="Chance-level convention (n_ps=1, markers only)")
        )

    fig.legend(eta_handles, eta_labels,
               loc="center left", bbox_to_anchor=(0.86, 0.60),
               frameon=False, title="Eta (η)", title_fontsize=9, fontsize=8)
    fig.legend(method_handles, [METHOD_LABELS[mk] for mk in METHOD_ORDER],
               loc="center left", bbox_to_anchor=(0.86, 0.82),
               frameon=False, title="Method", title_fontsize=9, fontsize=8)
    fig.legend(style_handles, [h.get_label() for h in style_handles],
               loc="center left", bbox_to_anchor=(0.86, 0.22),
               frameon=False, title="Line style", title_fontsize=9, fontsize=8)

    fig.suptitle(title, y=0.985, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.85, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _read_rows_from_csv(path: Path) -> List[dict]:
    import csv

    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    return rows


def _plot_device_row_figure(
    rows: List[dict],
    etas: List[float],
    channel: str,
    readout_error: str,
    amplitude: str,
    nq_pred: np.ndarray,
    out_path: Path,
    title: str,
    ci_z: float = 1.0,
    show_bands: bool = True,
) -> None:
    """
    Render a 1x3 layout for a fixed (channel, readout_error, amplitude),
    with columns corresponding to devices I/S/T.
    """
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {_MPL_IMPORT_ERROR}")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    colors = _eta_palette(etas)

    amp_f = _to_float(amplitude)

    # Index by (device, method, eta, nq) at fixed (channel, readout, amplitude).
    idx: Dict[tuple, dict] = {}
    for r in rows:
        if str(r.get("channel", "")) != str(channel):
            continue
        if str(r.get("readout_error", "")) != str(readout_error):
            continue
        amp_r = _to_float(r.get("amplitude", float("nan")))
        if not (_is_finite(amp_r) and _is_finite(amp_f) and abs(amp_r - amp_f) < 1e-8):
            continue
        eta = _to_float(r.get("eta", float("nan")))
        nq = _to_float(r.get("nq", float("nan")))
        if not (_is_finite(eta) and _is_finite(nq)):
            continue
        key = (str(r.get("device", "")), str(r.get("method", "")), float(eta), int(nq))
        idx[key] = r

    obs_max_by_method = {mk: max(v) for mk, v in OBS_NQ_BY_METHOD.items()}
    obs_set_by_method = {mk: set(v) for mk, v in OBS_NQ_BY_METHOD.items()}
    any_gate_shading = False
    any_rb_marker = False

    def _as_bool(v, default: bool = True) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return default

    for col_j, device in enumerate(DEVICE_ORDER):
        ax = axes[col_j]
        ax.set_title(f"{DEVICE_TITLES.get(device, device)}", fontsize=10)
        nq_arr = np.asarray(nq_pred, dtype=int)
        nq_max_plot = int(nq_arr[-1])

        # Visual cue for intentionally suppressed extrapolation.
        for mk in METHOD_ORDER:
            om = int(obs_max_by_method[mk])
            if om >= nq_max_plot:
                continue
            trusted_vals = []
            for eta in sorted(etas):
                rec = idx.get((device, mk, eta, int(om)))
                if rec is None:
                    rec = idx.get((device, mk, eta, int(om + 1)))
                if rec is None:
                    continue
                trusted_vals.append(_as_bool(rec.get("extrapolation_trusted", True), default=True))
            if trusted_vals and (not all(trusted_vals)):
                any_gate_shading = True
                ax.axvspan(
                    om + 0.5, nq_max_plot + 0.5,
                    color=METHOD_COLORS.get(mk, "gray"),
                    alpha=0.055 if mk == "hypergraph" else 0.040,
                    zorder=0,
                    label="_nolegend_",
                )
                ax.axvline(
                    x=om + 0.5,
                    color=METHOD_COLORS.get(mk, "gray"),
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.35,
                    zorder=2,
                )

        for eta in sorted(etas):
            col = colors[eta]
            for mk in METHOD_ORDER:
                om = obs_max_by_method[mk]

                def _get(field, default=float("nan")):
                    return np.array([
                        _to_float(idx.get((device, mk, eta, int(nq)), {}).get(field, default))
                        for nq in nq_arr
                    ])

                log2_nps = _get("log2_nps")
                lo_l2 = _get("log2_nps_lo")
                hi_l2 = _get("log2_nps_hi")

                def _safe_pow2(a):
                    a = np.asarray(a, dtype=float)
                    aa = np.where(np.isfinite(a), np.maximum(a, 0.0), np.nan)
                    return np.where(np.isfinite(aa), np.power(2.0, np.clip(aa, -1020.0, 1020.0)), float("nan"))

                nps = _safe_pow2(log2_nps)
                nps_lo = _safe_pow2(lo_l2)
                nps_hi = _safe_pow2(hi_l2)

                trivial = np.array([
                    _as_bool(idx.get((device, mk, eta, int(nq)), {}).get("is_trivial_regime", False), default=False)
                    for nq in nq_arr
                ])
                meaningful = np.array([
                    _as_bool(idx.get((device, mk, eta, int(nq)), {}).get("is_meaningful", True), default=True)
                    for nq in nq_arr
                ])
                rb_fix = np.array([
                    _as_bool(idx.get((device, mk, eta, int(nq)), {}).get("is_random_baseline_fix", False), default=False)
                    for nq in nq_arr
                ])
                status = np.array([
                    str(idx.get((device, mk, eta, int(nq)), {}).get("pred_status", ""))
                    for nq in nq_arr
                ])

                ok_plot = np.isfinite(log2_nps) & ~trivial & (meaningful | rb_fix)
                line_ok = ok_plot & (status == "ok")
                rb_show = ok_plot & (status == "random_baseline")
                censored_show = ok_plot & ((status == "left_censored") | (status == "right_censored"))

                nps_plot = np.where(line_ok, nps, float("nan"))
                lo_plot = np.where(line_ok, nps_lo, float("nan"))
                hi_plot = np.where(line_ok, nps_hi, float("nan"))
                lw = METHOD_LINEWIDTHS[mk]

                if show_bands:
                    vb = np.isfinite(lo_plot) & np.isfinite(hi_plot) & (lo_plot > 0) & (hi_plot > 0)
                    if vb.any():
                        ax.fill_between(
                            nq_arr[vb], lo_plot[vb], hi_plot[vb],
                            color=col, alpha=0.10 if mk == "hypergraph" else 0.07,
                            linewidth=0, zorder=1,
                        )

                obs_seg = line_ok & (nq_arr <= om)
                if obs_seg.any():
                    ax.plot(
                        nq_arr[obs_seg], nps_plot[obs_seg],
                        color=col, linestyle="-", linewidth=lw, alpha=0.92, zorder=3,
                    )

                ext_seg = line_ok & (nq_arr > om)
                if ext_seg.any():
                    ax.plot(
                        nq_arr[ext_seg], nps_plot[ext_seg],
                        color=col, linestyle="--", linewidth=lw, alpha=0.80, zorder=3,
                    )

                rb_vals = np.where(rb_show, nps, float("nan"))
                vb_rb = np.isfinite(rb_vals) & (rb_vals > 0)
                if vb_rb.any():
                    any_rb_marker = True
                    ax.scatter(
                        nq_arr[vb_rb], rb_vals[vb_rb],
                        color=col, marker="x", s=26, alpha=0.85,
                        linewidths=1.0, zorder=5,
                    )

                for nq_c in nq_arr[censored_show]:
                    r_c = idx.get((device, mk, eta, int(nq_c)), {})
                    v_c = _to_float(r_c.get("nps", float("nan")))
                    if not (_is_finite(v_c) and v_c > 0):
                        continue
                    marker = "^" if r_c.get("pred_status") == "right_censored" else "v"
                    ax.scatter(
                        nq_c, v_c, color=col, marker=marker, s=22, alpha=0.75,
                        edgecolors="k", linewidths=0.4, zorder=5,
                    )

                for nq_c in obs_set_by_method[mk]:
                    r_c = idx.get((device, mk, eta, int(nq_c)), {})
                    if _as_bool(r_c.get("is_trivial_regime", False), default=False):
                        continue
                    if not _as_bool(r_c.get("is_meaningful", True), default=True):
                        continue
                    if r_c.get("pred_status", "") != "ok":
                        continue
                    v_c = _to_float(r_c.get("nps", float("nan")))
                    if not (_is_finite(v_c) and v_c > 0):
                        continue
                    ax.scatter(
                        nq_c, v_c, color=col, marker=METHOD_MARKERS[mk],
                        s=18 if mk == "hypergraph" else 14,
                        linewidths=0.6, edgecolors="k", alpha=0.85, zorder=4,
                    )

                ax.axvline(
                    x=om + 0.5, color="gray", linewidth=0.4,
                    linestyle=":", alpha=0.35, zorder=2,
                )

        ax.set_yscale("log")
        ax.grid(alpha=0.22)
        ax.set_xlabel("Qubits (n_q)", fontsize=9)
        if col_j == 0:
            ax.set_ylabel("Classical n_ps required (log scale)", fontsize=9)

    for ax in axes.flat:
        ax.set_xlim(nq_pred[0] - 0.5, nq_pred[-1] + 0.5)

    colors_legend = _eta_palette(etas)
    eta_handles = [Line2D([0], [0], color=colors_legend[e], linewidth=2.2) for e in sorted(etas)]
    eta_labels = [f"η={e:.2f}" for e in sorted(etas)]
    method_handles = [
        Line2D(
            [0], [0], color="gray", linestyle="-",
            linewidth=METHOD_LINEWIDTHS[mk], marker=METHOD_MARKERS[mk],
            markersize=4, label=METHOD_LABELS[mk],
        )
        for mk in METHOD_ORDER
    ]
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.8, label="Observed range"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="Extrapolated (inv_n)"),
        Line2D([0], [0], color="gray", marker="^", linestyle="", markersize=6, label="Right-censored (lower bound)"),
        Line2D([0], [0], color="gray", marker="v", linestyle="", markersize=6, label="Left-censored (upper bound)"),
    ]
    if show_bands:
        style_handles.insert(
            2,
            Patch(facecolor="gray", alpha=0.25, label=f"Calibrated CI (empirical CV + WLS, z={ci_z:.1g})"),
        )
    if any_gate_shading:
        style_handles.append(Patch(facecolor="gray", alpha=0.18, label="Extrapolation gated off (untrusted)"))
    if any_rb_marker:
        style_handles.append(
            Line2D([0], [0], color="gray", marker="x", linestyle="", markersize=6, label="Chance-level convention (n_ps=1)")
        )

    fig.legend(
        eta_handles, eta_labels,
        loc="center left", bbox_to_anchor=(0.86, 0.67),
        frameon=False, title="Eta (η)", title_fontsize=9, fontsize=8,
    )
    fig.legend(
        method_handles, [METHOD_LABELS[mk] for mk in METHOD_ORDER],
        loc="center left", bbox_to_anchor=(0.86, 0.83),
        frameon=False, title="Method", title_fontsize=9, fontsize=8,
    )
    fig.legend(
        style_handles, [h.get_label() for h in style_handles],
        loc="center left", bbox_to_anchor=(0.86, 0.27),
        frameon=False, title="Line style", title_fontsize=9, fontsize=8,
    )

    fig.suptitle(title, y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_device_row_figures_from_csv(
    csv_path: Path,
    out_dir: Path,
    channels: Optional[List[str]] = None,
    readout_errors: Optional[List[str]] = None,
    amplitudes: Optional[List[str]] = None,
    ci_z: float = 1.0,
    show_bands: bool = True,
) -> int:
    rows = _read_rows_from_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    ch_avail = sorted({str(r.get("channel", "")) for r in rows if str(r.get("channel", ""))})
    re_avail = sorted({str(r.get("readout_error", "")) for r in rows if str(r.get("readout_error", ""))})
    amp_avail_vals = sorted({
        float(r.get("amplitude")) for r in rows
        if _is_finite(r.get("amplitude", float("nan")))
    })
    amp_avail = [str(a) for a in amp_avail_vals]

    ch_sel = channels if channels else ch_avail
    re_sel = readout_errors if readout_errors else re_avail
    if amplitudes:
        amp_req = [_to_float(a) for a in amplitudes]
        amp_sel_vals = [a for a in amp_avail_vals if any(_is_finite(ar) and abs(a - ar) < 1e-8 for ar in amp_req)]
    else:
        amp_sel_vals = amp_avail_vals

    etas = sorted({
        float(r.get("eta")) for r in rows
        if _is_finite(r.get("eta", float("nan")))
    })
    nq_pred = np.array(sorted({
        int(_to_float(r.get("nq"))) for r in rows
        if _is_finite(r.get("nq", float("nan")))
    }), dtype=int)
    if nq_pred.size < 1:
        raise ValueError(f"No valid nq values found in CSV: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    stem = csv_path.stem
    for ch in ch_sel:
        if ch not in ch_avail:
            continue
        for re in re_sel:
            if re not in re_avail:
                continue
            for amp in amp_sel_vals:
                amp_tag = _amp_tag_from_float(float(amp))
                re_tag = str(re).replace("%", "pct")
                base = f"{stem}_deviceRow_ch_{ch}_p{amp_tag}_{re_tag}"
                title = (
                    f"Unified Comparison (1x3 devices) — channel={CHANNEL_TITLES.get(ch, ch)}, "
                    f"p={float(amp):.2g}, readout={re}"
                )
                for ext in ("png", "pdf"):
                    out_path = out_dir / f"{base}.{ext}"
                    _plot_device_row_figure(
                        rows=rows,
                        etas=etas,
                        channel=str(ch),
                        readout_error=str(re),
                        amplitude=str(amp),
                        nq_pred=nq_pred,
                        out_path=out_path,
                        title=title,
                        ci_z=float(ci_z),
                        show_bands=bool(show_bands),
                    )
                    n_written += 1
    return int(n_written)


def _normalize_cli_list(vals: Optional[List[str]]) -> Optional[List[str]]:
    if not vals:
        return None
    out: List[str] = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            out.extend([p for p in parts if p])
        else:
            out.append(s)
    return out or None


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def _write_sensitivity_report(
    out_dir: Path,
    qdata: dict,
    hg_all_rows: List[dict],
    eig_all_rows: List[dict],
    ml_all_rows: List[dict],
    *,
    device: str,
    readout_error: str,
    target_floor: float,
    ci_z: float,
    baseline_args: dict,
    thresholds: np.ndarray,
    load_hg_fn,
    hm_dir: Path,
    eig_dir: Path,
    shadow_mod,
    results_q5_10: Optional[str],
    results_q11_12: Optional[str],
    n_boot_hg: int,
    n_boot_ml: int,
    boot_ci_level: float,
    boot_seed: int,
    hg_bootstrap_mode: str,
    sensitivity_bootstrap_checks: bool = True,
    sensitivity_boot_counts: Optional[List[int]] = None,
    sensitivity_amps: Optional[List[str]] = None,
    sensitivity_etas: Optional[List[float]] = None,
    sensitivity_deltas: Optional[List[int]] = None,
    curve_case_surfaces: Optional[Dict[str, Dict[Tuple[str, str], dict]]] = None,
    curve_bootstrap_fit: str = CURVE_BOOTSTRAP_FIT_DEFAULT,
    curve_bootstrap_min_valid: int = CURVE_BOOTSTRAP_MIN_VALID_DEFAULT,
    curve_bootstrap_simultaneous: bool = CURVE_BOOTSTRAP_SIMULTANEOUS_DEFAULT,
) -> None:
    """
    Publication-oriented one-at-a-time (OAT) sensitivity report.

    Scope:
      - amps: low/mid/high (default 0.01, 0.05, 0.10)
      - etas: default (0.01, 0.02, 0.05)
      - anchors per method:
          * nq_last_meaningful
          * nq_obs_max
          * nq_obs_max + delta for each requested delta
      - variants covering key degrees-of-freedom knobs
      - two-seed baseline robustness block (full re-bootstrap for seed+1)
      - bootstrap-sampling variants (mode/seed/n_boot) across HG/Eigenshadow/ML
    """
    amps = [str(a) for a in (sensitivity_amps or SENSITIVITY_AMPS_DEFAULT)]
    etas = sorted(set(float(e) for e in (sensitivity_etas or SENSITIVITY_ETAS_DEFAULT)))
    deltas = sorted(set(int(d) for d in (sensitivity_deltas or SENSITIVITY_DELTAS_DEFAULT) if int(d) > 0))
    nq_global_max = int(max(int(v) for v in qdata["nq_values"]))
    methods = list(METHOD_ORDER)
    obs_max_by_method = {mk: max(v) for mk, v in OBS_NQ_BY_METHOD.items()}
    curve_surfaces = curve_case_surfaces if isinstance(curve_case_surfaces, dict) else {}
    curve_fit_default = str(curve_bootstrap_fit).strip().lower()
    if curve_fit_default not in {"wls", "ols"}:
        curve_fit_default = CURVE_BOOTSTRAP_FIT_DEFAULT
    curve_lookup_cache: Dict[Tuple[str, bool], dict] = {}

    def _pick_last_meaningful_nq(channel: str, amp: str, eta_val: float) -> Optional[int]:
        cutoff = None
        for nq in qdata["nq_values"]:
            q = _get_quantum_acc(qdata, device, channel, amp, readout_error, int(nq))
            if _is_finite(q) and (q - float(eta_val)) >= float(target_floor):
                cutoff = int(nq)
        return cutoff

    def _anchors_for(method: str, channel: str, amp: str, eta_val: float) -> List[Tuple[str, Optional[int]]]:
        obs_max = int(obs_max_by_method[method])
        out: List[Tuple[str, Optional[int]]] = []
        out.append(("last_meaningful", _pick_last_meaningful_nq(channel, amp, eta_val)))
        out.append(("obs_max", int(obs_max)))
        for d in deltas:
            out.append((f"obs_max_plus_{int(d)}", int(min(obs_max + int(d), nq_global_max))))
        return out

    def _get_curve_band_lookup(fit_mode: str, enforce_monotone: bool) -> dict:
        key = (str(fit_mode).strip().lower(), bool(enforce_monotone))
        if key in curve_lookup_cache:
            return curve_lookup_cache[key]
        lookup: Dict[Tuple[str, str, str, float, int], dict] = {}
        case_runs = 0
        fit_mode_eff_counts: Dict[str, int] = {}
        for mk in methods:
            mk_cases = curve_surfaces.get(str(mk), {})
            if not isinstance(mk_cases, dict):
                continue
            for ch in CHANNEL_ORDER:
                for amp in amps:
                    surf = mk_cases.get((str(ch), str(amp)))
                    if not isinstance(surf, dict):
                        continue
                    thr = np.asarray(surf.get("thresholds"), dtype=float)
                    nqs_obs = np.asarray(surf.get("nq_values"), dtype=int)
                    btn = np.asarray(surf.get("kx_boot_btn"), dtype=float)
                    sigma_tn = (
                        np.asarray(surf.get("sigma_k_tn"), dtype=float)
                        if surf.get("sigma_k_tn") is not None else None
                    )
                    if btn.ndim != 3 or btn.shape[0] < 2:
                        continue
                    for eta_val in etas:
                        nq_anchor_vals = sorted(set(
                            int(nv) for _, nv in _anchors_for(mk, ch, amp, eta_val) if nv is not None
                        ))
                        if not nq_anchor_vals:
                            continue
                        target_by_nq: Dict[int, float] = {}
                        for nq in nq_anchor_vals:
                            q_acc = _get_quantum_acc(qdata, device, ch, amp, readout_error, int(nq))
                            target = (q_acc - float(eta_val)) if _is_finite(q_acc) else float("nan")
                            if _is_finite(target):
                                target_by_nq[int(nq)] = float(target)
                        if not target_by_nq:
                            continue
                        bands, bdiag = _curve_bootstrap_log2_bands(
                            thresholds=thr,
                            nq_values=nqs_obs,
                            kx_boot_btn=btn,
                            target_by_nq=target_by_nq,
                            nq_pred=np.asarray(nq_anchor_vals, dtype=int),
                            coverage=_central_coverage_from_z(abs(float(ci_z))),
                            enforce_threshold_monotone=bool(enforce_monotone),
                            simultaneous=bool(curve_bootstrap_simultaneous),
                            min_valid=int(curve_bootstrap_min_valid),
                            fit_mode=key[0],
                            sigma_k_tn=sigma_tn,
                        )
                        case_runs += 1
                        fit_eff = str(bdiag.get("fit_mode_effective", key[0]))
                        fit_mode_eff_counts[fit_eff] = fit_mode_eff_counts.get(fit_eff, 0) + 1
                        for nq, band in bands.items():
                            lookup[(str(mk), str(ch), str(amp), float(eta_val), int(nq))] = {
                                "ci_halfwidth_log2": _to_float(band.get("ci_halfwidth_log2", float("nan"))),
                                "n_valid": int(_to_float(band.get("n_valid", 0))) if _is_finite(band.get("n_valid", 0)) else 0,
                                "fit_mode_effective": fit_eff,
                            }
        out = {
            "lookup": lookup,
            "requested_fit_mode": key[0],
            "effective_fit_mode_counts": fit_mode_eff_counts,
            "n_case_runs": int(case_runs),
        }
        curve_lookup_cache[key] = out
        return out

    def _build_predictors_from_rows(
        hg_rows: List[dict],
        eig_rows: List[dict],
        ml_rows: List[dict],
        build_cfg: dict,
    ) -> Tuple[Dict[Tuple[str, str], Predictor], Dict[Tuple[str, str], Predictor], Dict[Tuple[str, str], Predictor]]:
        hg_preds: Dict[Tuple[str,str], Predictor] = {}
        eig_preds: Dict[Tuple[str,str], Predictor] = {}
        ml_preds: Dict[Tuple[str,str], Predictor] = {}
        for amp in amps:
            p_val = float(amp)
            for ch in CHANNEL_ORDER:
                hg_ok_rows = [
                    r for r in hg_rows
                    if _is_inference_row(r) and r["channel"] == ch and abs(r["p"] - p_val) < 1e-8
                ]
                pred_hg = _build_opt1_predictor(
                    hg_ok_rows, source_label="hg",
                    keep_cv_details=False,
                    extrap_min_horizons=int(build_cfg["extrap_min_horizons"]),
                    extrap_max_sigma_cv=float(build_cfg["extrap_max_sigma_cv"]),
                    extrap_max_rmse_ratio=float(build_cfg["extrap_max_rmse_ratio"]),
                    cv_thr_pct_min=float(build_cfg["cv_thr_pct_min"]),
                    cv_thr_pct_max=float(build_cfg["cv_thr_pct_max"]),
                    sigma_cv_k_mode=str(build_cfg["sigma_cv_k_mode"]),
                    sigma_cv_k_coverage=float(build_cfg["sigma_cv_k_coverage"]),
                    cv_coverage_zs=None,
                    enforce_threshold_monotone=bool(build_cfg["enforce_threshold_monotone"]),
                    monotone_threshold_grid_size=int(build_cfg["threshold_monotone_grid_size"]),
                    interval_method=str(build_cfg["interval_method"]),
                    conformal_holdout_count=int(build_cfg["conformal_holdout_count"]),
                    conformal_min_abs_errors=int(build_cfg["conformal_min_abs_errors"]),
                    strict_split_conformal=bool(build_cfg.get("strict_split_conformal", False)),
                )
                if pred_hg is not None:
                    hg_preds[(ch, amp)] = pred_hg

                eig_ok_rows = [
                    r for r in eig_rows
                    if _is_inference_row(r) and r["channel"] == ch and abs(r["p"] - p_val) < 1e-8
                ]
                pred_eig = _build_opt1_predictor(
                    eig_ok_rows, source_label="eigenshadow",
                    keep_cv_details=False,
                    extrap_min_horizons=int(build_cfg["extrap_min_horizons"]),
                    extrap_max_sigma_cv=float(build_cfg["extrap_max_sigma_cv"]),
                    extrap_max_rmse_ratio=float(build_cfg["extrap_max_rmse_ratio"]),
                    cv_thr_pct_min=float(build_cfg["cv_thr_pct_min"]),
                    cv_thr_pct_max=float(build_cfg["cv_thr_pct_max"]),
                    sigma_cv_k_mode=str(build_cfg["sigma_cv_k_mode"]),
                    sigma_cv_k_coverage=float(build_cfg["sigma_cv_k_coverage"]),
                    cv_coverage_zs=None,
                    enforce_threshold_monotone=bool(build_cfg["enforce_threshold_monotone"]),
                    monotone_threshold_grid_size=int(build_cfg["threshold_monotone_grid_size"]),
                    interval_method=str(build_cfg["interval_method"]),
                    conformal_holdout_count=int(build_cfg["conformal_holdout_count"]),
                    conformal_min_abs_errors=int(build_cfg["conformal_min_abs_errors"]),
                    strict_split_conformal=bool(build_cfg.get("strict_split_conformal", False)),
                )
                if pred_eig is not None:
                    eig_preds[(ch, amp)] = pred_eig

                ml_ok_rows = [
                    r for r in ml_rows
                    if _is_inference_row(r) and r["channel"] == ch and abs(r["p"] - p_val) < 1e-8
                ]
                pred_ml = _build_opt1_predictor(
                    ml_ok_rows, source_label="ml",
                    keep_cv_details=False,
                    extrap_min_horizons=int(build_cfg["extrap_min_horizons"]),
                    extrap_max_sigma_cv=float(build_cfg["extrap_max_sigma_cv"]),
                    extrap_max_rmse_ratio=float(build_cfg["extrap_max_rmse_ratio"]),
                    cv_thr_pct_min=float(build_cfg["cv_thr_pct_min"]),
                    cv_thr_pct_max=float(build_cfg["cv_thr_pct_max"]),
                    sigma_cv_k_mode=str(build_cfg["sigma_cv_k_mode"]),
                    sigma_cv_k_coverage=float(build_cfg["sigma_cv_k_coverage"]),
                    cv_coverage_zs=None,
                    enforce_threshold_monotone=bool(build_cfg["enforce_threshold_monotone"]),
                    monotone_threshold_grid_size=int(build_cfg["threshold_monotone_grid_size"]),
                    interval_method=str(build_cfg["interval_method"]),
                    conformal_holdout_count=int(build_cfg["conformal_holdout_count"]),
                    conformal_min_abs_errors=int(build_cfg["conformal_min_abs_errors"]),
                    strict_split_conformal=bool(build_cfg.get("strict_split_conformal", False)),
                )
                if pred_ml is not None:
                    ml_preds[(ch, amp)] = pred_ml
        return hg_preds, eig_preds, ml_preds

    def _eval_point(
        pred: Optional[Predictor],
        method: str,
        channel: str,
        amp: str,
        eta_val: float,
        nq_eval: Optional[int],
        anchor_name: str,
        eval_cfg: dict,
    ) -> dict:
        obs_max = int(obs_max_by_method[method])
        rec = {
            "method": method,
            "channel": channel,
            "amplitude": amp,
            "eta": float(eta_val),
            "anchor": str(anchor_name),
            "nq": int(nq_eval) if nq_eval is not None else None,
            "obs_max": obs_max,
            "is_extrapolated": (int(nq_eval) > obs_max) if nq_eval is not None else False,
            "q_acc": float("nan"),
            "target": float("nan"),
            "status": "missing_anchor",
            "log2_nps": float("nan"),
            "ci_hw_log2": float("nan"),
            "curve_bootstrap_fit_mode": "",
            "curve_bootstrap_n_valid": float("nan"),
            "random_baseline_engaged": False,
            "extrapolation_trusted": bool(pred.meta.get("extrapolation_trusted", True)) if pred is not None else False,
        }
        if nq_eval is None:
            return rec
        if pred is None:
            rec["status"] = "no_predictor"
            return rec

        nq = int(nq_eval)
        q_acc = _get_quantum_acc(qdata, device, channel, amp, readout_error, nq)
        target = (q_acc - float(eta_val)) if _is_finite(q_acc) else float("nan")
        rec["q_acc"] = q_acc
        rec["target"] = target

        eps_nq = _random_baseline_eps(
            nq=nq,
            eps0=float(eval_cfg["random_baseline_eps0"]),
            decay=float(eval_cfg["random_baseline_decay"]),
            nq_ref=int(eval_cfg["random_baseline_nq_ref"]),
        )
        rb_thr = float(L_BASELINE + eps_nq) if _is_finite(eps_nq) else float("nan")
        is_meaningful = _is_finite(target) and (target >= float(target_floor))
        use_rb = (
            bool(eval_cfg["random_baseline_fix"])
            and _is_finite(target)
            and _is_finite(rb_thr)
            and (target < rb_thr)
        )

        if use_rb:
            rec["status"] = "random_baseline"
            rec["log2_nps"] = 0.0
            rec["ci_hw_log2"] = float("nan")
            rec["random_baseline_engaged"] = True
            return rec
        if not is_meaningful:
            rec["status"] = "near_random"
            return rec
        if nq > obs_max and not bool(pred.meta.get("extrapolation_trusted", True)):
            rec["status"] = "untrusted_extrapolation"
            return rec

        pt = _predict_point(pred, float(target), nq, obs_max, ci_z=float(ci_z))
        rec["status"] = str(pt.get("status", "nan"))
        rec["log2_nps"] = _to_float(pt.get("log2_nps", float("nan")))
        rec["ci_hw_log2"] = _to_float(pt.get("ci_halfwidth_log2_nps", float("nan")))
        rec["is_extrapolated"] = bool(pt.get("is_extrapolated", nq > obs_max))
        if rec["status"] == "ok":
            fit_mode_req = str(eval_cfg.get("curve_bootstrap_fit_mode", "")).strip().lower()
            band_lookup = eval_cfg.get("curve_band_lookup", {})
            if fit_mode_req and isinstance(band_lookup, dict):
                bk = (str(method), str(channel), str(amp), float(eta_val), int(nq))
                b = band_lookup.get(bk)
                if isinstance(b, dict):
                    hw_b = _to_float(b.get("ci_halfwidth_log2", float("nan")))
                    if _is_finite(hw_b) and hw_b >= 0:
                        rec["ci_hw_log2"] = float(hw_b)
                    rec["curve_bootstrap_fit_mode"] = str(b.get("fit_mode_effective", fit_mode_req))
                    nv = _to_float(b.get("n_valid", float("nan")))
                    if _is_finite(nv):
                        rec["curve_bootstrap_n_valid"] = int(nv)
        return rec

    def _evaluate_variant(
        variant_name: str,
        hg_preds: Dict[Tuple[str, str], Predictor],
        eig_preds: Dict[Tuple[str, str], Predictor],
        ml_preds: Dict[Tuple[str, str], Predictor],
        eval_cfg: dict,
    ) -> List[dict]:
        recs: List[dict] = []
        for amp in amps:
            for eta_val in etas:
                for ch in CHANNEL_ORDER:
                    for mk in methods:
                        if mk == "hypergraph":
                            pred = hg_preds.get((ch, amp))
                        elif mk == "eigenshadow":
                            pred = eig_preds.get((ch, amp))
                        else:
                            pred = ml_preds.get((ch, amp))
                        for anchor_name, nq_anchor in _anchors_for(mk, ch, amp, eta_val):
                            rr = _eval_point(pred, mk, ch, amp, eta_val, nq_anchor, anchor_name, eval_cfg)
                            rr["variant"] = variant_name
                            recs.append(rr)
        return recs

    def _summarize_records(records: List[dict]) -> dict:
        n_total = int(len(records))
        if n_total == 0:
            return {"n_total": 0}
        status_counts: Dict[str, int] = {}
        n_extrap = 0
        n_gated = 0
        n_rb = 0
        ci_ok_vals: List[float] = []
        for r in records:
            st = str(r.get("status", ""))
            status_counts[st] = status_counts.get(st, 0) + 1
            if bool(r.get("is_extrapolated", False)):
                n_extrap += 1
            if st == "untrusted_extrapolation":
                n_gated += 1
            if bool(r.get("random_baseline_engaged", False)):
                n_rb += 1
            if st == "ok":
                hw = _to_float(r.get("ci_hw_log2", float("nan")))
                if _is_finite(hw) and hw >= 0:
                    ci_ok_vals.append(float(hw))
        ci_arr = np.asarray(ci_ok_vals, dtype=float)
        return {
            "n_total": n_total,
            "status_counts": status_counts,
            "n_extrapolated": int(n_extrap),
            "n_untrusted_extrapolation": int(n_gated),
            "n_random_baseline": int(n_rb),
            "extrapolation_gated_rate": float(n_gated / n_extrap) if n_extrap > 0 else float("nan"),
            "random_baseline_rate": float(n_rb / n_total),
            "n_ok_with_ci": int(ci_arr.size),
            "ci_hw_log2_median": float(np.nanmedian(ci_arr)) if ci_arr.size else float("nan"),
            "ci_hw_log2_p90": float(np.nanpercentile(ci_arr, 90.0)) if ci_arr.size else float("nan"),
            "ci_hw_log2_max": float(np.nanmax(ci_arr)) if ci_arr.size else float("nan"),
        }

    def _beats_map(records: List[dict]) -> Dict[Tuple[str, str, float, str, Optional[int]], bool]:
        def _is_ok_point(r: Optional[dict]) -> bool:
            if r is None:
                return False
            if str(r.get("status", "")) != "ok":
                return False
            return _is_finite(r.get("log2_nps", float("nan")))

        by_key: Dict[Tuple[str, str, float, str, Optional[int]], Dict[str, dict]] = {}
        for r in records:
            key = (
                str(r.get("channel")),
                str(r.get("amplitude")),
                float(r.get("eta")),
                str(r.get("anchor")),
                int(r["nq"]) if r.get("nq") is not None else None,
            )
            by_key.setdefault(key, {})[str(r.get("method"))] = r
        out: Dict[Tuple[str, str, float, str, Optional[int]], bool] = {}
        for key, g in by_key.items():
            hg = g.get("hypergraph")
            ml = g.get("ml")
            if not (_is_ok_point(hg) and _is_ok_point(ml)):
                continue
            hg_l2 = _to_float(hg.get("log2_nps", float("nan")))
            ml_l2 = _to_float(ml.get("log2_nps", float("nan")))
            if not (_is_finite(hg_l2) and _is_finite(ml_l2)):
                continue
            out[key] = bool(hg_l2 < ml_l2)
        return out

    def _flip_summary(base_records: List[dict], var_records: List[dict]) -> dict:
        bmap = _beats_map(base_records)
        vmap = _beats_map(var_records)
        keys = sorted(set(bmap.keys()) & set(vmap.keys()))
        if not keys:
            return {"n_comparable": 0, "n_flips": 0, "flip_rate": float("nan")}
        n_flip = sum(1 for k in keys if bool(bmap[k]) != bool(vmap[k]))
        return {
            "n_comparable": int(len(keys)),
            "n_flips": int(n_flip),
            "flip_rate": float(n_flip / len(keys)),
        }

    def _log2_abs_delta_stats(base_records: List[dict], var_records: List[dict]) -> dict:
        def _is_ok_point(r: Optional[dict]) -> bool:
            if r is None:
                return False
            if str(r.get("status", "")) != "ok":
                return False
            return _is_finite(r.get("log2_nps", float("nan")))

        def _method_key(r: dict) -> tuple:
            return (
                str(r.get("method")),
                str(r.get("channel")),
                str(r.get("amplitude")),
                float(r.get("eta")),
                str(r.get("anchor")),
                int(r["nq"]) if r.get("nq") is not None else None,
            )
        b0 = {_method_key(r): r for r in base_records}
        b1 = {_method_key(r): r for r in var_records}
        common = sorted(set(b0.keys()) & set(b1.keys()))
        abs_deltas = []
        for k in common:
            if not (_is_ok_point(b0[k]) and _is_ok_point(b1[k])):
                continue
            v0 = _to_float(b0[k].get("log2_nps", float("nan")))
            v1 = _to_float(b1[k].get("log2_nps", float("nan")))
            if _is_finite(v0) and _is_finite(v1):
                abs_deltas.append(abs(v1 - v0))
        arr = np.asarray(abs_deltas, dtype=float)
        return {
            "n_common_finite": int(arr.size),
            "median": float(np.nanmedian(arr)) if arr.size else float("nan"),
            "p90": float(np.nanpercentile(arr, 90.0)) if arr.size else float("nan"),
            "max": float(np.nanmax(arr)) if arr.size else float("nan"),
        }

    def _ci_hw_abs_delta_stats(base_records: List[dict], var_records: List[dict]) -> dict:
        def _method_key(r: dict) -> tuple:
            return (
                str(r.get("method")),
                str(r.get("channel")),
                str(r.get("amplitude")),
                float(r.get("eta")),
                str(r.get("anchor")),
                int(r["nq"]) if r.get("nq") is not None else None,
            )

        b0 = {_method_key(r): r for r in base_records}
        b1 = {_method_key(r): r for r in var_records}
        common = sorted(set(b0.keys()) & set(b1.keys()))
        abs_deltas = []
        for k in common:
            if str(b0[k].get("status", "")) != "ok" or str(b1[k].get("status", "")) != "ok":
                continue
            v0 = _to_float(b0[k].get("ci_hw_log2", float("nan")))
            v1 = _to_float(b1[k].get("ci_hw_log2", float("nan")))
            if _is_finite(v0) and _is_finite(v1):
                abs_deltas.append(abs(v1 - v0))
        arr = np.asarray(abs_deltas, dtype=float)
        return {
            "n_common_finite": int(arr.size),
            "median": float(np.nanmedian(arr)) if arr.size else float("nan"),
            "p90": float(np.nanpercentile(arr, 90.0)) if arr.size else float("nan"),
            "max": float(np.nanmax(arr)) if arr.size else float("nan"),
        }

    def _build_key(cfg: dict) -> tuple:
        return (
            int(cfg["extrap_min_horizons"]),
            float(cfg["extrap_max_sigma_cv"]),
            float(cfg["extrap_max_rmse_ratio"]),
            float(cfg["cv_thr_pct_min"]),
            float(cfg["cv_thr_pct_max"]),
            str(cfg["sigma_cv_k_mode"]),
            float(cfg["sigma_cv_k_coverage"]),
            bool(cfg["enforce_threshold_monotone"]),
            int(cfg["threshold_monotone_grid_size"]),
            str(cfg["interval_method"]),
            int(cfg["conformal_holdout_count"]),
            int(cfg["conformal_min_abs_errors"]),
            bool(cfg.get("strict_split_conformal", False)),
        )

    def _load_ml_curves_obs() -> List[dict]:
        r5, r11 = shadow_mod.find_shadow_results_json(results_q5_10, results_q11_12)
        ml_curves = shadow_mod.load_shadowqmlml_curves([r5, r11])
        return [c for c in ml_curves if int(c["n_q"]) in set(ML_NQ_OBS)]

    def _bootstrap_rows_for_seed(
        seed: int,
        *,
        hg_mode: Optional[str] = None,
        n_boot_hg_override: Optional[int] = None,
        n_boot_ml_override: Optional[int] = None,
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        mode = str(hg_mode if hg_mode is not None else hg_bootstrap_mode)
        nbh = int(n_boot_hg_override) if n_boot_hg_override is not None else int(n_boot_hg)
        nbm = int(n_boot_ml_override) if n_boot_ml_override is not None else int(n_boot_ml)
        hg_rows_seed: List[dict] = []
        for ch in CHANNEL_ORDER:
            for amp in amps:
                rows = bootstrap_hg_dense_threshold_rows(
                    load_hg_fn=load_hg_fn,
                    hm_dir=hm_dir,
                    nq_list=HG_NQ_OBS,
                    channel=ch,
                    amplitude=float(amp),
                    thresholds=thresholds,
                    n_boot=int(nbh),
                    seed=int(seed),
                    ci_level=float(boot_ci_level),
                    hg_bootstrap_mode=str(mode),
                )
                hg_rows_seed.extend(rows)

        eig_rows_seed: List[dict] = []
        for ch in CHANNEL_ORDER:
            for amp in amps:
                rows = bootstrap_hg_dense_threshold_rows(
                    load_hg_fn=load_hg_fn,
                    hm_dir=eig_dir,
                    nq_list=EIGENSHADOW_NQ_OBS,
                    channel=ch,
                    amplitude=float(amp),
                    thresholds=thresholds,
                    n_boot=int(nbh),
                    seed=int(seed),
                    ci_level=float(boot_ci_level),
                    hg_bootstrap_mode=str(mode),
                )
                eig_rows_seed.extend(rows)

        ml_curves_obs = _load_ml_curves_obs()
        ml_rows_seed = shadow_mod.bootstrap_dense_threshold_rows(
            ml_curves_obs,
            thresholds=thresholds,
            n_boot=int(nbm),
            seed=int(seed),
            ci_level=float(boot_ci_level),
        )
        for r in ml_rows_seed:
            r["channel"] = ML_CHANNEL_CANONICAL.get(r["channel"], r["channel"])
        ml_rows_seed = _postprocess_rows(ml_rows_seed)
        return hg_rows_seed, eig_rows_seed, ml_rows_seed

    base_build = {
        "extrap_min_horizons": int(baseline_args["extrap_min_horizons"]),
        "extrap_max_sigma_cv": float(baseline_args["extrap_max_sigma_cv"]),
        "extrap_max_rmse_ratio": float(baseline_args["extrap_max_rmse_ratio"]),
        "cv_thr_pct_min": float(baseline_args["cv_thr_pct_min"]),
        "cv_thr_pct_max": float(baseline_args["cv_thr_pct_max"]),
        "sigma_cv_k_mode": str(baseline_args["sigma_cv_k_mode"]),
        "sigma_cv_k_coverage": float(baseline_args["sigma_cv_k_coverage"]),
        "enforce_threshold_monotone": bool(baseline_args["enforce_threshold_monotone"]),
        "threshold_monotone_grid_size": int(baseline_args["threshold_monotone_grid_size"]),
        "interval_method": str(baseline_args["interval_method"]),
        "conformal_holdout_count": int(baseline_args["conformal_holdout_count"]),
        "conformal_min_abs_errors": int(baseline_args["conformal_min_abs_errors"]),
        "strict_split_conformal": bool(baseline_args.get("strict_split_conformal", False)),
    }
    base_eval = {
        "random_baseline_fix": bool(baseline_args["random_baseline_fix"]),
        "random_baseline_eps0": float(baseline_args["random_baseline_eps0"]),
        "random_baseline_decay": float(baseline_args["random_baseline_decay"]),
        "random_baseline_nq_ref": int(baseline_args["random_baseline_nq_ref"]),
    }

    variant_specs = [
        ("baseline", {}, {}),
        ("interval=cv", {"interval_method": "cv"}, {}),
        ("interval=conformal_hybrid", {"interval_method": "conformal_hybrid"}, {}),
        ("cv_thr=5-95", {"cv_thr_pct_min": 5.0, "cv_thr_pct_max": 95.0}, {}),
        ("cv_thr=20-80", {"cv_thr_pct_min": 20.0, "cv_thr_pct_max": 80.0}, {}),
        ("monotone=on", {"enforce_threshold_monotone": True}, {}),
        ("monotone=off", {"enforce_threshold_monotone": False}, {}),
        ("random_baseline=off", {}, {"random_baseline_fix": False}),
        ("random_eps0=0.01", {}, {"random_baseline_fix": True, "random_baseline_eps0": 0.01}),
        ("random_eps0=0.02", {}, {"random_baseline_fix": True, "random_baseline_eps0": 0.02}),
        ("extrap_min_h=2", {"extrap_min_horizons": 2}, {}),
        ("extrap_min_h=3", {"extrap_min_horizons": 3}, {}),
        ("extrap_min_h=4", {"extrap_min_horizons": 4}, {}),
        ("extrap_rmse_ratio=2.0", {"extrap_max_rmse_ratio": 2.0}, {}),
        ("extrap_rmse_ratio=3.0", {"extrap_max_rmse_ratio": 3.0}, {}),
        ("conformal_holdout=1", {"conformal_holdout_count": 1}, {}),
        ("conformal_holdout=2", {"conformal_holdout_count": 2}, {}),
        ("conformal_holdout=3", {"conformal_holdout_count": 3}, {}),
        ("conformal_min_abs_errors=20", {"conformal_min_abs_errors": 20}, {}),
        ("conformal_min_abs_errors=50", {"conformal_min_abs_errors": 50}, {}),
    ]
    if bool(curve_surfaces):
        variant_specs.extend([
            ("curve_fit=wls", {}, {"curve_bootstrap_fit_mode": "wls"}),
            ("curve_fit=ols", {}, {"curve_bootstrap_fit_mode": "ols"}),
        ])

    pred_cache: Dict[
        tuple,
        Tuple[
            Dict[Tuple[str, str], Predictor],
            Dict[Tuple[str, str], Predictor],
            Dict[Tuple[str, str], Predictor],
        ],
    ] = {}
    results = {
        "config": {
            "device": device,
            "readout_error": readout_error,
            "target_floor": float(target_floor),
            "ci_z": float(ci_z),
            "amps": amps,
            "etas": [float(v) for v in etas],
            "deltas": [int(v) for v in deltas],
            "nq_max_global": int(nq_global_max),
            "obs_max_by_method": obs_max_by_method,
            "delta_choice_note": "delta=3 is the primary validated extrapolation anchor because forward-CV max_horizon=3; larger deltas are stress anchors.",
            "sensitivity_boot_counts": [int(v) for v in (sensitivity_boot_counts or [])],
            "strict_split_conformal": bool(base_build.get("strict_split_conformal", False)),
            "curve_bootstrap_fit_default": str(curve_fit_default),
            "curve_bootstrap_min_valid": int(curve_bootstrap_min_valid),
            "curve_bootstrap_simultaneous": bool(curve_bootstrap_simultaneous),
            "curve_bootstrap_surfaces_available": bool(curve_surfaces),
        },
        "variants": {},
    }

    baseline_records: List[dict] = []
    for name, build_over, eval_over in variant_specs:
        build_cfg = dict(base_build)
        build_cfg.update(build_over)
        eval_cfg = dict(base_eval)
        eval_cfg.update(eval_over)
        runtime_eval_cfg = dict(eval_cfg)
        req_fit_mode = str(eval_cfg.get("curve_bootstrap_fit_mode", "")).strip().lower()
        if req_fit_mode:
            lookup_payload = _get_curve_band_lookup(
                req_fit_mode,
                bool(build_cfg["enforce_threshold_monotone"]),
            )
            runtime_eval_cfg["curve_band_lookup"] = lookup_payload.get("lookup", {})
            eval_cfg["curve_bootstrap_fit_mode_effective_counts"] = lookup_payload.get("effective_fit_mode_counts", {})
            eval_cfg["curve_bootstrap_lookup_case_runs"] = int(lookup_payload.get("n_case_runs", 0))
        key = _build_key(build_cfg)
        if key not in pred_cache:
            pred_cache[key] = _build_predictors_from_rows(hg_all_rows, eig_all_rows, ml_all_rows, build_cfg)
        hg_preds, eig_preds, ml_preds = pred_cache[key]
        recs = _evaluate_variant(name, hg_preds, eig_preds, ml_preds, runtime_eval_cfg)
        results["variants"][name] = {
            "build_config": build_cfg,
            "eval_config": eval_cfg,
            "summary": _summarize_records(recs),
            "records": recs,
        }
        if name == "baseline":
            baseline_records = recs

    for name in results["variants"]:
        if name == "baseline":
            continue
        results["variants"][name]["flip_vs_baseline"] = _flip_summary(
            baseline_records, results["variants"][name]["records"]
        )
        results["variants"][name]["ci_hw_delta_vs_baseline"] = _ci_hw_abs_delta_stats(
            baseline_records, results["variants"][name]["records"]
        )

    robustness = {
        "requested": True,
        "seed_primary": int(boot_seed),
        "seed_secondary": int(boot_seed) + 1,
        "available": False,
    }
    try:
        sec_seed = int(boot_seed) + 1
        hg_rows_2, eig_rows_2, ml_rows_2 = _bootstrap_rows_for_seed(sec_seed)
        hg_preds_2, eig_preds_2, ml_preds_2 = _build_predictors_from_rows(hg_rows_2, eig_rows_2, ml_rows_2, base_build)
        recs_2 = _evaluate_variant("baseline_seed_secondary", hg_preds_2, eig_preds_2, ml_preds_2, base_eval)
        flip = _flip_summary(baseline_records, recs_2)
        robustness.update({
            "available": True,
            "baseline_seed_primary_summary": _summarize_records(baseline_records),
            "baseline_seed_secondary_summary": _summarize_records(recs_2),
            "flip_vs_primary_baseline": flip,
            "log2_delta_abs_stats": _log2_abs_delta_stats(baseline_records, recs_2),
            "ci_hw_delta_abs_stats": _ci_hw_abs_delta_stats(baseline_records, recs_2),
        })
    except Exception as err:  # noqa: BLE001
        robustness["reason"] = f"secondary_seed_failed: {err}"
    results["baseline_seed_robustness"] = robustness

    bootstrap_sampling: dict = {
        "requested": bool(sensitivity_bootstrap_checks),
        "available": False,
        "variants": {},
    }
    if bool(sensitivity_bootstrap_checks):
        min_boot = 20
        nbh_half = int(max(min_boot, int(round(0.5 * int(n_boot_hg)))))
        nbm_half = int(max(min_boot, int(round(0.5 * int(n_boot_ml)))))
        boot_counts = sorted(set(int(v) for v in (sensitivity_boot_counts or []) if int(v) > 0))
        has_hier_data = any(int(_to_float(r.get("n_draws", 0))) > 1 for r in hg_all_rows) or any(
            int(_to_float(r.get("n_draws", 0))) > 1 for r in eig_all_rows
        )
        boot_variant_specs = [
            ("seed_plus_1", {"seed": int(boot_seed) + 1}),
            ("hg_eig_mode_per_k", {"seed": int(boot_seed), "hg_mode": "per_k"}),
            ("hg_eig_mode_cluster", {"seed": int(boot_seed), "hg_mode": "cluster"}),
        ]
        if boot_counts:
            for nb in boot_counts:
                if nb == int(n_boot_hg) and nb == int(n_boot_ml):
                    continue
                boot_variant_specs.append(
                    (f"nboot={int(nb)}", {"seed": int(boot_seed), "n_boot_hg": int(nb), "n_boot_ml": int(nb)})
                )
        else:
            boot_variant_specs.append(
                ("nboot_half", {"seed": int(boot_seed), "n_boot_hg": int(nbh_half), "n_boot_ml": int(nbm_half)})
            )
        if has_hier_data:
            boot_variant_specs.append(("hg_eig_mode_hier_cluster", {"seed": int(boot_seed), "hg_mode": "hier_cluster"}))

        def _mode_counts(rows: List[dict]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for r in rows:
                m = r.get("hg_bootstrap_mode", None)
                if m is None:
                    continue
                mk = str(m)
                out[mk] = out.get(mk, 0) + 1
            return out

        try:
            for vname, spec in boot_variant_specs:
                seed_v = int(spec.get("seed", int(boot_seed)))
                mode_v = str(spec.get("hg_mode", str(hg_bootstrap_mode)))
                nbh_v = int(spec.get("n_boot_hg", int(n_boot_hg)))
                nbm_v = int(spec.get("n_boot_ml", int(n_boot_ml)))
                hg_rows_v, eig_rows_v, ml_rows_v = _bootstrap_rows_for_seed(
                    seed_v,
                    hg_mode=mode_v,
                    n_boot_hg_override=nbh_v,
                    n_boot_ml_override=nbm_v,
                )
                hg_preds_v, eig_preds_v, ml_preds_v = _build_predictors_from_rows(
                    hg_rows_v, eig_rows_v, ml_rows_v, base_build
                )
                recs_v = _evaluate_variant(vname, hg_preds_v, eig_preds_v, ml_preds_v, base_eval)
                bootstrap_sampling["variants"][vname] = {
                    "bootstrap_config": {
                        "seed": int(seed_v),
                        "hg_mode_requested": str(mode_v),
                        "n_boot_hg": int(nbh_v),
                        "n_boot_ml": int(nbm_v),
                    },
                    "realized_hg_mode_counts": _mode_counts(hg_rows_v),
                    "realized_eig_mode_counts": _mode_counts(eig_rows_v),
                    "summary": _summarize_records(recs_v),
                    "flip_vs_baseline": _flip_summary(baseline_records, recs_v),
                    "log2_delta_abs_stats_vs_baseline": _log2_abs_delta_stats(baseline_records, recs_v),
                    "ci_hw_delta_abs_stats_vs_baseline": _ci_hw_abs_delta_stats(baseline_records, recs_v),
                }
            bootstrap_sampling["available"] = True
        except Exception as err:  # noqa: BLE001
            bootstrap_sampling["reason"] = f"bootstrap_sampling_failed: {err}"
    results["bootstrap_sampling_variants"] = bootstrap_sampling

    json_path = out_dir / "sensitivity_report.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    md_lines = []
    md_lines.append("# Sensitivity Report\n")
    md_lines.append(f"- device: {device}, readout_error: {readout_error}\n")
    md_lines.append(f"- amps: {amps}\n")
    md_lines.append(f"- etas: {[float(v) for v in etas]}\n")
    md_lines.append(f"- deltas: {[int(v) for v in deltas]}\n")
    md_lines.append(f"- curve-bootstrap fit default: {curve_fit_default}\n")
    md_lines.append(f"- curve-bootstrap surfaces available: {bool(curve_surfaces)}\n")
    md_lines.append("- Anchor policy: `last_meaningful`, `obs_max`, and `obs_max + delta`.\n")
    md_lines.append("- Primary extrapolation anchor is `delta=3` (aligned with forward-CV max horizon=3). Larger deltas are stress tests.\n\n")

    md_lines.append("## Variant Summary\n")
    md_lines.append("| variant | n_total | ok | untrusted_extrap | random_baseline | gated_extrap_rate | rb_rate | CI hw med/p90 | HG<ML flip vs baseline | CI-hw Δ vs baseline med/p90/max |\n")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, payload in results["variants"].items():
        s = payload.get("summary", {})
        sc = s.get("status_counts", {})
        flip = payload.get("flip_vs_baseline", {})
        ci_d = payload.get("ci_hw_delta_vs_baseline", {})
        md_lines.append(
            f"| {name} | {int(s.get('n_total', 0))} | {int(sc.get('ok', 0))} | "
            f"{int(sc.get('untrusted_extrapolation', 0))} | {int(sc.get('random_baseline', 0))} | "
            f"{_to_float(s.get('extrapolation_gated_rate', float('nan'))):.3g} | "
            f"{_to_float(s.get('random_baseline_rate', float('nan'))):.3g} | "
            f"{_to_float(s.get('ci_hw_log2_median', float('nan'))):.3g} / {_to_float(s.get('ci_hw_log2_p90', float('nan'))):.3g} | "
            f"{_to_float(flip.get('flip_rate', float('nan'))):.3g} | "
            f"{_to_float(ci_d.get('median', float('nan'))):.3g} / {_to_float(ci_d.get('p90', float('nan'))):.3g} / {_to_float(ci_d.get('max', float('nan'))):.3g} |\n"
        )

    md_lines.append("\n## Two-Seed Baseline Robustness\n")
    if robustness.get("available"):
        ld = robustness.get("log2_delta_abs_stats", {})
        cd = robustness.get("ci_hw_delta_abs_stats", {})
        fl = robustness.get("flip_vs_primary_baseline", {})
        md_lines.append(
            f"- Seeds compared: {robustness['seed_primary']} vs {robustness['seed_secondary']}\n"
            f"- Common finite points: {int(ld.get('n_common_finite', 0))}\n"
            f"- |Δlog2(nps)| median/p90/max: {ld.get('median', float('nan')):.3g} / "
            f"{ld.get('p90', float('nan')):.3g} / {ld.get('max', float('nan')):.3g}\n"
            f"- |ΔCI_hw(log2)| median/p90/max: {cd.get('median', float('nan')):.3g} / "
            f"{cd.get('p90', float('nan')):.3g} / {cd.get('max', float('nan')):.3g}\n"
            f"- HG<ML flip rate vs primary baseline: {fl.get('flip_rate', float('nan')):.3g} "
            f"({int(fl.get('n_flips', 0))}/{int(fl.get('n_comparable', 0))})\n"
        )
    else:
        md_lines.append(f"- Secondary-seed robustness unavailable: {robustness.get('reason', 'unknown')}\n")

    md_lines.append("\n## Bootstrap Sampling Variants\n")
    bsv = results.get("bootstrap_sampling_variants", {})
    if bsv.get("requested") and bsv.get("available"):
        md_lines.append("| variant | n_total | ok | HG<ML flip vs baseline | |Δlog2(nps)| median/p90/max | |ΔCI_hw(log2)| med/p90/max |\n")
        md_lines.append("|---|---:|---:|---:|---:|---:|\n")
        for name, payload in bsv.get("variants", {}).items():
            ss = payload.get("summary", {})
            sc = ss.get("status_counts", {})
            ff = payload.get("flip_vs_baseline", {})
            dd = payload.get("log2_delta_abs_stats_vs_baseline", {})
            cd = payload.get("ci_hw_delta_abs_stats_vs_baseline", {})
            md_lines.append(
                f"| {name} | {int(ss.get('n_total', 0))} | {int(sc.get('ok', 0))} | "
                f"{_to_float(ff.get('flip_rate', float('nan'))):.3g} | "
                f"{_to_float(dd.get('median', float('nan'))):.3g} / "
                f"{_to_float(dd.get('p90', float('nan'))):.3g} / "
                f"{_to_float(dd.get('max', float('nan'))):.3g} | "
                f"{_to_float(cd.get('median', float('nan'))):.3g} / "
                f"{_to_float(cd.get('p90', float('nan'))):.3g} / "
                f"{_to_float(cd.get('max', float('nan'))):.3g} |\n"
            )
    elif bsv.get("requested"):
        md_lines.append(f"- Bootstrap-sampling checks unavailable: {bsv.get('reason', 'unknown')}\n")
    else:
        md_lines.append("- Bootstrap-sampling checks not requested.\n")

    md_path = out_dir / "sensitivity_report.md"
    md_path.write_text("".join(md_lines))

def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Unified quantum vs classical n_ps comparison via dense-threshold Step4 predictor. "
            "Bootstraps k_x(threshold, nq) for both HG and ML, fits k_x = C + β/nq, "
            "then queries the predictor along the quantum accuracy target curve q_acc(nq) − η."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--etas", type=float, nargs="*", default=None,
                        help=f"Eta values. Default: {DEFAULT_ETAS}")
    parser.add_argument("--thresholds-lo",   type=float, default=0.51)
    parser.add_argument("--thresholds-hi",   type=float, default=0.98,
                        help="Upper limit (exclusive) for threshold grid")
    parser.add_argument("--thresholds-step", type=float, default=0.02)
    parser.add_argument("--n-boot-hg",  type=int,   default=N_BOOT_HG)
    parser.add_argument("--n-boot-ml",  type=int,   default=N_BOOT_ML)
    parser.add_argument("--boot-seed",  type=int,   default=BOOT_SEED)
    parser.add_argument("--boot-ci-level", type=float, default=BOOT_CI_LEVEL,
                        help="CI level for bootstrap threshold CIs (default 0.90)")
    parser.add_argument("--ci-z", type=float, default=1.0,
                        help="z-score used for WLS term and mapping to empirical CV coverage (default 1.0)")


    parser.add_argument("--interval-method", type=str, default=INTERVAL_METHOD_DEFAULT,
                    choices=["cv", "conformal", "conformal_hybrid"],
                    help="Interval construction: cv (default legacy), conformal (held-out), or conformal_hybrid.")
    parser.add_argument("--conformal-holdout-count", type=int, default=CONFORMAL_HOLDOUT_COUNT_DEFAULT,
                    help="Number of largest observed nq values held out for conformal calibration (default 2).")
    parser.add_argument("--conformal-min-abs-errors", type=int, default=CONFORMAL_MIN_ABS_ERRORS_DEFAULT,
                    help="Minimum pooled held-out |e_k| residuals required to enable conformal (default 30).")
    parser.add_argument("--strict-split-conformal", dest="strict_split_conformal", action="store_true",
                    default=STRICT_SPLIT_CONFORMAL_DEFAULT,
                    help="Train Step4 on train-nq only and calibrate conformal on held-out nq (strict split).")
    parser.add_argument("--no-strict-split-conformal", dest="strict_split_conformal", action="store_false",
                    help="Disable strict split-conformal training/calibration split.")
    parser.add_argument("--hg-bootstrap-mode", type=str, default=HG_BOOTSTRAP_MODE_DEFAULT,
                    choices=["auto", "per_k", "cluster", "hier_cluster"],
                    help="HG bootstrap mode: per_k (legacy) or cluster/hier_cluster when alignment exists.")
    parser.add_argument("--run-sensitivity", action="store_true",
                    help="Write a short sensitivity report (markdown + json) to the output directory.")
    parser.add_argument("--target-floor", type=float, default=TARGET_FLOOR,
                        help=f"Min meaningful q_acc−η (default {TARGET_FLOOR})")
    parser.add_argument("--random-baseline-fix", dest="random_baseline_fix", action="store_true",
                        default=RANDOM_BASELINE_FIX_DEFAULT,
                        help="Force n_ps=1 when q_acc−η < 0.5 + eps(nq).")
    parser.add_argument("--no-random-baseline-fix", dest="random_baseline_fix", action="store_false",
                        help="Disable the manual random-baseline n_ps=1 override.")
    parser.add_argument("--random-baseline-eps0", type=float, default=RANDOM_BASELINE_EPS0_DEFAULT,
                        help="eps0 in eps(nq)=eps0*exp(-decay*(nq-nq_ref)) (default 0.02).")
    parser.add_argument("--random-baseline-decay", type=float, default=RANDOM_BASELINE_DECAY_DEFAULT,
                        help=f"Decay in eps(nq)=eps0*exp(-decay*(nq-nq_ref)) (default {RANDOM_BASELINE_DECAY_DEFAULT:.2f}).")
    parser.add_argument("--random-baseline-nq-ref", type=int, default=RANDOM_BASELINE_NQ_REF_DEFAULT,
                        help="nq_ref in eps(nq)=eps0*exp(-decay*(nq-nq_ref)) (default 5).")
    parser.add_argument("--devices",       nargs="*", default=None)
    parser.add_argument("--readout-errors", nargs="*", default=None)
    parser.add_argument("--nq-pred-min",   type=int,  default=5)
    parser.add_argument("--nq-pred-max",   type=int,  default=50)
    parser.add_argument("--hypergraph-merged-dir", type=str, default=None)
    parser.add_argument("--shadow-surrogate-dir", type=str, default=None)
    parser.add_argument("--quantum-json",   type=str, default=None)
    parser.add_argument("--results-q5-10",  type=str, default=None)
    parser.add_argument("--results-q11-12", type=str, default=None)
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--device-row-from-csv", type=str, default=None,
                        help="Render 1x3 device-column figures from an existing unified CSV, then exit.")
    parser.add_argument("--device-row-output-dir", type=str, default=None,
                        help="Output directory for --device-row-from-csv mode (default: <csv_parent>/device_row_views).")
    parser.add_argument("--device-row-channels", nargs="*", default=None,
                        help="Optional channel filter(s) for --device-row-from-csv (default: all in CSV).")
    parser.add_argument("--device-row-readout-errors", nargs="*", default=None,
                        help="Optional readout-error filter(s), e.g. '0%' '1%'.")
    parser.add_argument("--device-row-amplitudes", nargs="*", default=None,
                        help="Optional amplitude filter(s), e.g. 0.01 0.05 0.1.")
    parser.add_argument("--bootstrap-cache-dir", type=str, default=None,
                        help="Directory for bootstrap row caches (HG/Eigenshadow/ML).")
    parser.add_argument("--bootstrap-cache-mode", type=str, default="off",
                        choices=["off", "read", "write", "readwrite"],
                        help="Bootstrap cache policy: off, read, write, readwrite.")
    parser.add_argument("--save-bootstrap-replicates", action="store_true",
                        help="Save per-case bootstrap k_x replicate tensors (B,T,N) for threshold-correlation diagnostics.")
    parser.add_argument("--bootstrap-replicates-dir", type=str, default=None,
                        help="Directory for saved bootstrap replicate tensors and correlation diagnostics.")
    parser.add_argument("--bootstrap-corr-diagnostics", action="store_true",
                        help="Compute threshold-correlation diagnostics from bootstrap k_x replicates.")
    parser.add_argument("--curve-bootstrap-ci", action="store_true",
                        default=CURVE_BOOTSTRAP_CI_DEFAULT,
                        help="Use full-pipeline bootstrap replicate surfaces to set final curve-level CI bands.")
    parser.add_argument("--curve-bootstrap-simultaneous", action="store_true",
                        default=CURVE_BOOTSTRAP_SIMULTANEOUS_DEFAULT,
                        help="When --curve-bootstrap-ci is on, use simultaneous (sup-norm) bands instead of pointwise bands.")
    parser.add_argument("--curve-bootstrap-min-valid", type=int, default=CURVE_BOOTSTRAP_MIN_VALID_DEFAULT,
                        help="Minimum finite bootstrap replicates required per nq point for curve-level CI update.")
    parser.add_argument("--curve-bootstrap-fit", type=str, default=CURVE_BOOTSTRAP_FIT_DEFAULT,
                        choices=["wls", "ols"],
                        help="Step4 fit used inside full-pipeline curve bootstrap (default: wls).")
    parser.add_argument("--no-bands", action="store_true",
                        help="Suppress parametric CI fill bands")
    parser.add_argument("--save-hg-rows", action="store_true",
                        help="Write HG bootstrap rows to JSON for inspection / caching")
    parser.add_argument("--save-eig-rows", action="store_true",
                        help="Write Eigenshadow bootstrap rows to JSON for inspection / caching")
    parser.add_argument("--save-ml-rows", action="store_true",
                        help="Write ML bootstrap rows to JSON for inspection / caching")
    parser.add_argument("--hg-rows-json", type=str, default=None,
                        help="Load pre-computed HG bootstrap threshold rows JSON (skip HG re-bootstrap)")
    parser.add_argument("--eig-rows-json", type=str, default=None,
                        help="Load pre-computed Eigenshadow bootstrap threshold rows JSON (skip Eigenshadow re-bootstrap)")
    parser.add_argument("--ml-rows-json", type=str, default=None,
                        help="Load pre-computed ML bootstrap threshold rows JSON (skip ML re-bootstrap)")
    parser.add_argument("--sensitivity-bootstrap-checks", dest="sensitivity_bootstrap_checks",
                        action="store_true", default=True,
                        help="Include bootstrap-sampling variants (mode/seed/n_boot) in sensitivity report.")
    parser.add_argument("--no-sensitivity-bootstrap-checks", dest="sensitivity_bootstrap_checks",
                        action="store_false",
                        help="Disable bootstrap-sampling variants in sensitivity report.")
    parser.add_argument("--sensitivity-boot-counts", type=int, nargs="*", default=None,
                        help="Optional n_boot values for sensitivity bootstrap variants (e.g., 400 1600 4000).")
    parser.add_argument("--forward-cv", action="store_true",
                        help="Include detailed per-threshold CV diagnostics in JSON (CV always runs for σ_CV)")
    parser.add_argument("--extrap-min-horizons", type=int, default=EXTRAP_TRUST_MIN_HORIZONS,
                        help="Minimum forward-CV horizons required to trust extrapolation (default 12).")
    parser.add_argument("--extrap-max-sigma-cv", type=float, default=EXTRAP_TRUST_MAX_SIGMA_CV,
                        help="Maximum allowed sigma_cv in log2-space at observed n_q max (default 1.0).")
    parser.add_argument("--extrap-max-rmse-ratio", type=float, default=EXTRAP_TRUST_MAX_RMSE_RATIO,
                        help="Maximum allowed rmse_max/rmse_pooled to trust extrapolation (default 2.5).")
    parser.add_argument("--cv-thr-pct-min", type=float, default=CV_THR_PCT_MIN,
                        help="Lower percentile for threshold slices in forward-CV (default 10).")
    parser.add_argument("--cv-thr-pct-max", type=float, default=CV_THR_PCT_MAX,
                        help="Upper percentile for threshold slices in forward-CV (default 90).")
    parser.add_argument("--sigma-cv-k-mode", type=str, default=SIGMA_CV_K_MODE_DEFAULT,
                        choices=["pooled_q", "pooled_max", "median_per_threshold_q"],
                        help="How to derive scalar sigma_cv_k from forward-CV k-errors.")
    parser.add_argument("--sigma-cv-k-coverage", type=float, default=SIGMA_CV_K_COVERAGE_DEFAULT,
                        help="Target central coverage used for sigma_cv_k quantile rule (default 0.6827).")
    parser.add_argument("--cv-coverage-zs", type=float, nargs="*", default=None,
                        help="z values for reported CV coverage diagnostics (default: 1.0 1.64485 1.95996).")
    parser.add_argument("--enforce-threshold-monotone", action="store_true",
                        help="Option E: enforce non-decreasing k_x vs threshold at each n_q.")
    parser.add_argument("--threshold-monotone-grid-size", type=int, default=MONOTONE_THRESHOLD_GRID_SIZE,
                        help="Dense threshold grid size used by Option E (default 256).")
    args = parser.parse_args()

    if plt is None:
        print(f"ERROR: matplotlib unavailable: {_MPL_IMPORT_ERROR}")
        return 2

    if args.device_row_from_csv:
        csv_path = Path(args.device_row_from_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"--device-row-from-csv not found: {csv_path}")
        out_dir_row = (
            Path(args.device_row_output_dir).expanduser().resolve()
            if args.device_row_output_dir else (csv_path.parent / "device_row_views")
        )
        ch_sel = _normalize_cli_list(args.device_row_channels)
        re_sel = _normalize_cli_list(args.device_row_readout_errors)
        amp_sel = _normalize_cli_list(args.device_row_amplitudes)
        n_written = _render_device_row_figures_from_csv(
            csv_path=csv_path,
            out_dir=out_dir_row,
            channels=ch_sel,
            readout_errors=re_sel,
            amplitudes=amp_sel,
            ci_z=float(args.ci_z),
            show_bands=not bool(args.no_bands),
        )
        print(f"Rendered device-row figures from CSV: {csv_path}")
        print(f"Files written: {n_written}")
        print(f"Output dir: {out_dir_row}")
        return 0

    etas          = sorted(set(float(e) for e in args.etas)) if args.etas else DEFAULT_ETAS
    devices       = args.devices       or DEVICE_ORDER
    readout_errors = args.readout_errors or READOUT_ERRORS
    if int(args.extrap_min_horizons) < 1:
        raise ValueError("--extrap-min-horizons must be >= 1")
    if not (_is_finite(args.extrap_max_sigma_cv) and float(args.extrap_max_sigma_cv) > 0):
        raise ValueError("--extrap-max-sigma-cv must be > 0")
    if not (_is_finite(args.extrap_max_rmse_ratio) and float(args.extrap_max_rmse_ratio) > 0):
        raise ValueError("--extrap-max-rmse-ratio must be > 0")
    if not (0.0 <= float(args.cv_thr_pct_min) < float(args.cv_thr_pct_max) <= 100.0):
        raise ValueError("--cv-thr-pct-min/max must satisfy 0 <= min < max <= 100")
    if not (0.0 < float(args.sigma_cv_k_coverage) < 1.0):
        raise ValueError("--sigma-cv-k-coverage must satisfy 0 < value < 1")
    if int(args.threshold_monotone_grid_size) < 64:
        raise ValueError("--threshold-monotone-grid-size must be >= 64")
    if not (_is_finite(args.random_baseline_eps0) and float(args.random_baseline_eps0) >= 0):
        raise ValueError("--random-baseline-eps0 must be >= 0")
    if not (_is_finite(args.random_baseline_decay) and float(args.random_baseline_decay) >= 0):
        raise ValueError("--random-baseline-decay must be >= 0")
    if int(args.curve_bootstrap_min_valid) < 2:
        raise ValueError("--curve-bootstrap-min-valid must be >= 2")
    if str(args.curve_bootstrap_fit).strip().lower() not in {"wls", "ols"}:
        raise ValueError("--curve-bootstrap-fit must be one of: wls, ols")
    thresholds    = np.arange(float(args.thresholds_lo), float(args.thresholds_hi),
                              float(args.thresholds_step))
    nq_pred       = np.arange(int(args.nq_pred_min), int(args.nq_pred_max) + 1)
    out_dir       = Path(args.output_dir) if args.output_dir else (SCRIPT_DIR / "plots_unified")
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_boot_repl = bool(
        args.save_bootstrap_replicates
        or args.bootstrap_corr_diagnostics
        or args.curve_bootstrap_ci
    )
    boot_repl_dir = (
        Path(args.bootstrap_replicates_dir)
        if args.bootstrap_replicates_dir else (out_dir / "bootstrap_replicates")
    )
    if collect_boot_repl:
        boot_repl_dir.mkdir(parents=True, exist_ok=True)
    cache_mode = str(args.bootstrap_cache_mode).strip().lower()
    cache_read = cache_mode in {"read", "readwrite"}
    cache_write = cache_mode in {"write", "readwrite"}
    cache_dir = Path(args.bootstrap_cache_dir) if args.bootstrap_cache_dir else (out_dir / "bootstrap_cache")
    if cache_write:
        cache_dir.mkdir(parents=True, exist_ok=True)
    thresholds_sha256 = _array_sha256(np.asarray(thresholds, dtype=float))
    script_sha256 = _file_sha256(Path(__file__))

    # ── Load dependent modules ─────────────────────────────────────────────────
    hg_mod = _load_module(
        "hg_raw_mod_unified",
        SCRIPT_DIR / "plot_hypergraph_vs_ml_nps_multiaccuracy_from_raw_discrete.py",
    )
    shadow_mod = _load_module(
        "shadow_mod_unified",
        SCRIPT_DIR / "plot_shadowqmlml_nps_from_quantum_accuracy_curves.py",
    )
    load_hg_fn = hg_mod.load_hypergraph_acc_trials
    hm_dir = (
        Path(args.hypergraph_merged_dir) if args.hypergraph_merged_dir
        else _find_hg_merged()
    )
    eig_dir = (
        Path(args.shadow_surrogate_dir) if args.shadow_surrogate_dir
        else _find_shadow_surrogate()
    )

    q_json_path = (
        Path(args.quantum_json) if args.quantum_json
        else SCRIPT_DIR / "quantum_accuracy_curves.json"
    )
    qdata = shadow_mod.load_quantum_accuracy_curves(q_json_path)
    print(f"Quantum JSON: nq={qdata['nq_values'][0]}..{qdata['nq_values'][-1]}, "
          f"devices={qdata['devices']}, channels={qdata['channels']}")
    print(f"Threshold grid: {thresholds[0]:.2f}..{thresholds[-1]:.2f}  "
          f"({len(thresholds)} thresholds)")

    # Optional replicate capture for threshold-correlation diagnostics.
    hg_repl_cases: Dict[Tuple[str, float], Dict[int, dict]] = {}
    eig_repl_cases: Dict[Tuple[str, float], Dict[int, dict]] = {}
    ml_repl_cases: Dict[Tuple[str, float], Dict[int, dict]] = {}

    # ── HG: bootstrap dense threshold rows + build predictors ─────────────────
    hg_source_sig = _dir_signature(hm_dir, suffixes=(".npy",))
    hg_cache_meta = {
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "method": "hypergraph",
        "script_sha256": str(script_sha256),
        "source_root": str(hm_dir.resolve()),
        "source_signature_sha256": str(hg_source_sig["sha256"]),
        "nq_list": [int(v) for v in HG_NQ_OBS],
        "channels": [str(v) for v in CHANNEL_ORDER],
        "amplitudes": [str(v) for v in PLOT_AMPLITUDES],
        "thresholds_sha256": str(thresholds_sha256),
        "n_boot": int(args.n_boot_hg),
        "seed": int(args.boot_seed),
        "ci_level": float(args.boot_ci_level),
        "bootstrap_mode": str(args.hg_bootstrap_mode),
    }
    hg_cache_path = cache_dir / f"hg_bootstrap_rows_{_cache_key_from_meta(hg_cache_meta)}.json"
    print("\nBootstrapping HG dense threshold rows …")
    hg_all_rows: List[dict] = []
    hg_loaded = False
    if args.hg_rows_json:
        hg_all_rows, hg_meta_got = _load_rows_json_with_meta(Path(args.hg_rows_json))
        print(f"  Loaded HG rows from {args.hg_rows_json}: {len(hg_all_rows)}")
        if hg_meta_got:
            ok, reason = _cache_meta_matches(hg_cache_meta, hg_meta_got)
            if not ok:
                print(f"  WARN HG explicit rows metadata mismatch ({reason}); using file as requested.")
        hg_loaded = True
    elif cache_read and hg_cache_path.exists():
        hg_rows_cached, hg_meta_got = _load_rows_json_with_meta(hg_cache_path)
        ok, reason = _cache_meta_matches(hg_cache_meta, hg_meta_got)
        if ok:
            hg_all_rows = hg_rows_cached
            hg_loaded = True
            print(f"  Loaded HG rows from cache: {hg_cache_path}")
        else:
            print(f"  HG cache metadata mismatch ({reason}); recomputing.")
    elif cache_read:
        print(f"  HG cache miss: {hg_cache_path.name}")

    if not hg_loaded:
        hg_all_rows: List[dict] = []
        for ch in CHANNEL_ORDER:
            for amp_str in PLOT_AMPLITUDES:
                p_val = float(amp_str)
                print(f"  HG ch={ch} p={amp_str} …", flush=True)
                case_repl_sink = {} if collect_boot_repl else None
                case_rows = bootstrap_hg_dense_threshold_rows(
                    load_hg_fn=load_hg_fn,
                    hm_dir=hm_dir,
                    nq_list=HG_NQ_OBS,
                    channel=ch,
                    amplitude=p_val,
                    thresholds=thresholds,
                    n_boot=args.n_boot_hg,
                    seed=args.boot_seed,
                    ci_level=args.boot_ci_level,
                    hg_bootstrap_mode=str(args.hg_bootstrap_mode),
                    replicate_sink=case_repl_sink,
                )
                hg_all_rows.extend(case_rows)
                if collect_boot_repl and case_repl_sink:
                    hg_repl_cases[(str(ch), float(p_val))] = case_repl_sink
                n_ok = sum(1 for r in case_rows if _is_inference_row(r))
                print(f"    → {n_ok}/{len(case_rows)} ok rows")
        if cache_write:
            _save_rows_json_with_meta(hg_cache_path, hg_all_rows, hg_cache_meta)
            print(f"  HG cache saved → {hg_cache_path}")
    elif collect_boot_repl:
        print("  HG rows loaded; recomputing HG bootstrap replicates for diagnostics …")
        for ch in CHANNEL_ORDER:
            for amp_str in PLOT_AMPLITUDES:
                p_val = float(amp_str)
                case_repl_sink = {}
                _ = bootstrap_hg_dense_threshold_rows(
                    load_hg_fn=load_hg_fn,
                    hm_dir=hm_dir,
                    nq_list=HG_NQ_OBS,
                    channel=ch,
                    amplitude=p_val,
                    thresholds=thresholds,
                    n_boot=args.n_boot_hg,
                    seed=args.boot_seed,
                    ci_level=args.boot_ci_level,
                    hg_bootstrap_mode=str(args.hg_bootstrap_mode),
                    replicate_sink=case_repl_sink,
                )
                if case_repl_sink:
                    hg_repl_cases[(str(ch), float(p_val))] = case_repl_sink

    if args.save_hg_rows:
        hg_path = out_dir / "hg_bootstrap_threshold_rows.json"
        _save_rows_json_with_meta(hg_path, hg_all_rows, hg_cache_meta)
        print(f"  HG rows saved → {hg_path}")

    def _warn_predictor_quality(method_label: str, ch: str, amp_str: str, pred_obj: Predictor) -> None:
        val = pred_obj.meta.get("predictor_validation", {}) if isinstance(pred_obj.meta, dict) else {}
        bq = pred_obj.meta.get("bootstrap_quality", {}) if isinstance(pred_obj.meta, dict) else {}
        if isinstance(val, dict):
            if not bool(val.get("threshold_order_ok", True)):
                print(f"  WARN {method_label} [{ch},{amp_str}] invalid threshold order in predictor.")
            if int(_to_float(val.get("n_threshold_fits", 0))) < 2:
                print(f"  WARN {method_label} [{ch},{amp_str}] n_threshold_fits < 2.")
            if not bool(val.get("forward_cv_applicable", True)):
                print(f"  WARN {method_label} [{ch},{amp_str}] forward-CV not applicable: {val.get('forward_cv_reason', 'unknown')}")
        if isinstance(bq, dict):
            lvf = _to_float(bq.get("low_valid_rate_near_floor", float("nan")))
            hcf = _to_float(bq.get("high_censor_rate_near_floor", float("nan")))
            if _is_finite(lvf) and lvf > 0.30:
                print(f"  WARN {method_label} [{ch},{amp_str}] high low-valid ratio near floor: {lvf:.3f}")
            if _is_finite(hcf) and hcf > 0.30:
                print(f"  WARN {method_label} [{ch},{amp_str}] high censoring near floor: {hcf:.3f}")

    print("\nBuilding HG opt1 predictors …")
    hg_predictors: Dict[Tuple[str, str], Predictor] = {}
    hg_predictor_meta: dict = {}
    for ch in CHANNEL_ORDER:
        for amp_str in PLOT_AMPLITUDES:
            p_val = float(amp_str)
            ok_rows = [r for r in hg_all_rows
                       if _is_inference_row(r) and r["channel"] == ch and abs(r["p"] - p_val) < 1e-8]
            pred = _build_opt1_predictor(ok_rows, source_label="hg",
                                         keep_cv_details=bool(args.forward_cv),
                                         extrap_min_horizons=args.extrap_min_horizons,
                                         extrap_max_sigma_cv=args.extrap_max_sigma_cv,
                                         extrap_max_rmse_ratio=args.extrap_max_rmse_ratio,
                                         cv_thr_pct_min=args.cv_thr_pct_min,
                                         cv_thr_pct_max=args.cv_thr_pct_max,
                                         sigma_cv_k_mode=args.sigma_cv_k_mode,
                                         sigma_cv_k_coverage=args.sigma_cv_k_coverage,
                                         cv_coverage_zs=args.cv_coverage_zs,
                                         enforce_threshold_monotone=bool(args.enforce_threshold_monotone),
                                         monotone_threshold_grid_size=int(args.threshold_monotone_grid_size),
                                         interval_method=str(args.interval_method),
                                         conformal_holdout_count=int(args.conformal_holdout_count),
                                         conformal_min_abs_errors=int(args.conformal_min_abs_errors),
                                         strict_split_conformal=bool(args.strict_split_conformal))
            if pred is None:
                print(f"  WARN HG predictor FAILED: ch={ch} p={amp_str} "
                      f"({len(ok_rows)} ok rows available)")
                continue
            hg_predictors[(ch, amp_str)] = pred
            meta_entry = {
                **pred.meta,
                "n_ok_rows": len(ok_rows),
                "threshold_min": pred.threshold_min,
                "threshold_max": pred.threshold_max,
                "sigma_cv": float(pred.sigma_cv),
                "sigma_cv_k": float(pred.sigma_cv_k),
                "sigma_cv_k_obs": float(pred.sigma_cv_k_obs),
                "sigma_cv_k_extrap": float(pred.sigma_cv_k_extrap),
                "sigma_cv_log2_at_obs_max": float(pred.sigma_cv_log2_at_obs_max),
                "sigma_cv_log2_gate_at_obs_max": float(pred.sigma_cv_log2_gate_at_obs_max),
            }
            hg_predictor_meta[(ch, amp_str)] = meta_entry
            print(f"  HG [{ch},{amp_str}]: thr=[{pred.threshold_min:.3f},{pred.threshold_max:.3f}]"
                  f"  n_fits={pred.meta['n_threshold_fits']}  has_cov={pred.meta['has_parametric_cov']}")
            _warn_predictor_quality("HG", ch, amp_str, pred)

    # ── Eigenshadow: bootstrap dense threshold rows + build predictors ───────
    eig_source_sig = _dir_signature(eig_dir, suffixes=(".npy",))
    eig_cache_meta = {
        "cache_format_version": int(CACHE_FORMAT_VERSION),
        "method": "eigenshadow",
        "script_sha256": str(script_sha256),
        "source_root": str(eig_dir.resolve()),
        "source_signature_sha256": str(eig_source_sig["sha256"]),
        "nq_list": [int(v) for v in EIGENSHADOW_NQ_OBS],
        "channels": [str(v) for v in CHANNEL_ORDER],
        "amplitudes": [str(v) for v in PLOT_AMPLITUDES],
        "thresholds_sha256": str(thresholds_sha256),
        "n_boot": int(args.n_boot_hg),
        "seed": int(args.boot_seed),
        "ci_level": float(args.boot_ci_level),
        "bootstrap_mode": str(args.hg_bootstrap_mode),
    }
    eig_cache_path = cache_dir / f"eigenshadow_bootstrap_rows_{_cache_key_from_meta(eig_cache_meta)}.json"
    print("\nBootstrapping Eigenshadow dense threshold rows …")
    eig_all_rows: List[dict] = []
    eig_loaded = False
    if args.eig_rows_json:
        eig_all_rows, eig_meta_got = _load_rows_json_with_meta(Path(args.eig_rows_json))
        print(f"  Loaded Eigenshadow rows from {args.eig_rows_json}: {len(eig_all_rows)}")
        if eig_meta_got:
            ok, reason = _cache_meta_matches(eig_cache_meta, eig_meta_got)
            if not ok:
                print(f"  WARN Eigenshadow explicit rows metadata mismatch ({reason}); using file as requested.")
        eig_loaded = True
    elif cache_read and eig_cache_path.exists():
        eig_rows_cached, eig_meta_got = _load_rows_json_with_meta(eig_cache_path)
        ok, reason = _cache_meta_matches(eig_cache_meta, eig_meta_got)
        if ok:
            eig_all_rows = eig_rows_cached
            eig_loaded = True
            print(f"  Loaded Eigenshadow rows from cache: {eig_cache_path}")
        else:
            print(f"  Eigenshadow cache metadata mismatch ({reason}); recomputing.")
    elif cache_read:
        print(f"  Eigenshadow cache miss: {eig_cache_path.name}")

    if not eig_loaded:
        eig_all_rows = []
        for ch in CHANNEL_ORDER:
            for amp_str in PLOT_AMPLITUDES:
                p_val = float(amp_str)
                print(f"  Eigenshadow ch={ch} p={amp_str} …", flush=True)
                case_repl_sink = {} if collect_boot_repl else None
                case_rows = bootstrap_hg_dense_threshold_rows(
                    load_hg_fn=load_hg_fn,
                    hm_dir=eig_dir,
                    nq_list=EIGENSHADOW_NQ_OBS,
                    channel=ch,
                    amplitude=p_val,
                    thresholds=thresholds,
                    n_boot=args.n_boot_hg,
                    seed=args.boot_seed,
                    ci_level=args.boot_ci_level,
                    hg_bootstrap_mode=str(args.hg_bootstrap_mode),
                    replicate_sink=case_repl_sink,
                )
                eig_all_rows.extend(case_rows)
                if collect_boot_repl and case_repl_sink:
                    eig_repl_cases[(str(ch), float(p_val))] = case_repl_sink
                n_ok = sum(1 for r in case_rows if _is_inference_row(r))
                print(f"    → {n_ok}/{len(case_rows)} ok rows")
        if cache_write:
            _save_rows_json_with_meta(eig_cache_path, eig_all_rows, eig_cache_meta)
            print(f"  Eigenshadow cache saved → {eig_cache_path}")
    elif collect_boot_repl:
        print("  Eigenshadow rows loaded; recomputing bootstrap replicates for diagnostics …")
        for ch in CHANNEL_ORDER:
            for amp_str in PLOT_AMPLITUDES:
                p_val = float(amp_str)
                case_repl_sink = {}
                _ = bootstrap_hg_dense_threshold_rows(
                    load_hg_fn=load_hg_fn,
                    hm_dir=eig_dir,
                    nq_list=EIGENSHADOW_NQ_OBS,
                    channel=ch,
                    amplitude=p_val,
                    thresholds=thresholds,
                    n_boot=args.n_boot_hg,
                    seed=args.boot_seed,
                    ci_level=args.boot_ci_level,
                    hg_bootstrap_mode=str(args.hg_bootstrap_mode),
                    replicate_sink=case_repl_sink,
                )
                if case_repl_sink:
                    eig_repl_cases[(str(ch), float(p_val))] = case_repl_sink

    if args.save_eig_rows:
        eig_path = out_dir / "eigenshadow_bootstrap_threshold_rows.json"
        _save_rows_json_with_meta(eig_path, eig_all_rows, eig_cache_meta)
        print(f"  Eigenshadow rows saved → {eig_path}")

    n_eig_ok = sum(1 for r in eig_all_rows if _is_inference_row(r))
    print(f"  Eigenshadow bootstrap: {n_eig_ok}/{len(eig_all_rows)} ok rows")

    print("\nBuilding Eigenshadow opt1 predictors …")
    eig_predictors: Dict[Tuple[str, str], Predictor] = {}
    eig_predictor_meta: dict = {}
    for ch in CHANNEL_ORDER:
        for amp_str in PLOT_AMPLITUDES:
            p_val = float(amp_str)
            ok_rows = [r for r in eig_all_rows
                       if _is_inference_row(r) and r["channel"] == ch and abs(r["p"] - p_val) < 1e-8]
            pred = _build_opt1_predictor(ok_rows, source_label="eigenshadow",
                                         keep_cv_details=bool(args.forward_cv),
                                         extrap_min_horizons=args.extrap_min_horizons,
                                         extrap_max_sigma_cv=args.extrap_max_sigma_cv,
                                         extrap_max_rmse_ratio=args.extrap_max_rmse_ratio,
                                         cv_thr_pct_min=args.cv_thr_pct_min,
                                         cv_thr_pct_max=args.cv_thr_pct_max,
                                         sigma_cv_k_mode=args.sigma_cv_k_mode,
                                         sigma_cv_k_coverage=args.sigma_cv_k_coverage,
                                         cv_coverage_zs=args.cv_coverage_zs,
                                         enforce_threshold_monotone=bool(args.enforce_threshold_monotone),
                                         monotone_threshold_grid_size=int(args.threshold_monotone_grid_size),
                                         interval_method=str(args.interval_method),
                                         conformal_holdout_count=int(args.conformal_holdout_count),
                                         conformal_min_abs_errors=int(args.conformal_min_abs_errors),
                                         strict_split_conformal=bool(args.strict_split_conformal))
            if pred is None:
                print(f"  WARN Eigenshadow predictor FAILED: ch={ch} p={amp_str} "
                      f"({len(ok_rows)} ok rows available)")
                continue
            eig_predictors[(ch, amp_str)] = pred
            meta_entry = {
                **pred.meta,
                "n_ok_rows": len(ok_rows),
                "threshold_min": pred.threshold_min,
                "threshold_max": pred.threshold_max,
                "sigma_cv": float(pred.sigma_cv),
                "sigma_cv_k": float(pred.sigma_cv_k),
                "sigma_cv_k_obs": float(pred.sigma_cv_k_obs),
                "sigma_cv_k_extrap": float(pred.sigma_cv_k_extrap),
                "sigma_cv_log2_at_obs_max": float(pred.sigma_cv_log2_at_obs_max),
                "sigma_cv_log2_gate_at_obs_max": float(pred.sigma_cv_log2_gate_at_obs_max),
            }
            eig_predictor_meta[(ch, amp_str)] = meta_entry
            print(f"  Eigenshadow [{ch},{amp_str}]: thr=[{pred.threshold_min:.3f},{pred.threshold_max:.3f}]"
                  f"  n_fits={pred.meta['n_threshold_fits']}  has_cov={pred.meta['has_parametric_cov']}")
            _warn_predictor_quality("Eigenshadow", ch, amp_str, pred)

    # ── ML: load shadow curves, bootstrap, normalise, build predictors ────────
    ml_cache_meta: dict = {}
    ml_cache_path: Optional[Path] = None
    r5 = r11 = None
    need_ml_source = (not bool(args.ml_rows_json)) or cache_read or cache_write
    if need_ml_source:
        r5, r11 = shadow_mod.find_shadow_results_json(args.results_q5_10, args.results_q11_12)
        ml_source_sig = _paths_signature([Path(r5), Path(r11)])
        ml_cache_meta = {
            "cache_format_version": int(CACHE_FORMAT_VERSION),
            "method": "ml",
            "script_sha256": str(script_sha256),
            "source_signature_sha256": str(ml_source_sig["sha256"]),
            "nq_list": [int(v) for v in ML_NQ_OBS],
            "channels": [str(v) for v in CHANNEL_ORDER],
            "amplitudes": [str(v) for v in PLOT_AMPLITUDES],
            "thresholds_sha256": str(thresholds_sha256),
            "n_boot": int(args.n_boot_ml),
            "seed": int(args.boot_seed),
            "ci_level": float(args.boot_ci_level),
            "source_paths": [str(Path(r5).resolve()), str(Path(r11).resolve())],
        }
        ml_cache_path = cache_dir / f"ml_bootstrap_rows_{_cache_key_from_meta(ml_cache_meta)}.json"

    print("\nBootstrapping ML dense threshold rows …")
    ml_loaded = False
    if args.ml_rows_json:
        ml_all_rows_raw, ml_meta_got = _load_rows_json_with_meta(Path(args.ml_rows_json))
        for r in ml_all_rows_raw:
            r["channel"] = ML_CHANNEL_CANONICAL.get(r.get("channel"), r.get("channel"))
        # Re-normalize for defensive consistency with older cached JSON files.
        ml_all_rows_raw = _postprocess_rows(ml_all_rows_raw)
        print(f"  Loaded ML rows from {args.ml_rows_json}: {len(ml_all_rows_raw)}")
        if ml_meta_got and ml_cache_meta:
            ok, reason = _cache_meta_matches(ml_cache_meta, ml_meta_got)
            if not ok:
                print(f"  WARN ML explicit rows metadata mismatch ({reason}); using file as requested.")
        ml_loaded = True
    elif cache_read and ml_cache_path is not None and ml_cache_path.exists():
        ml_rows_cached, ml_meta_got = _load_rows_json_with_meta(ml_cache_path)
        ok, reason = _cache_meta_matches(ml_cache_meta, ml_meta_got)
        if ok:
            ml_all_rows_raw = ml_rows_cached
            for r in ml_all_rows_raw:
                r["channel"] = ML_CHANNEL_CANONICAL.get(r.get("channel"), r.get("channel"))
            ml_all_rows_raw = _postprocess_rows(ml_all_rows_raw)
            ml_loaded = True
            print(f"  Loaded ML rows from cache: {ml_cache_path}")
        else:
            print(f"  ML cache metadata mismatch ({reason}); recomputing.")
    elif cache_read and ml_cache_path is not None:
        print(f"  ML cache miss: {ml_cache_path.name}")

    if not ml_loaded:
        if r5 is None or r11 is None:
            r5, r11 = shadow_mod.find_shadow_results_json(args.results_q5_10, args.results_q11_12)
        ml_curves = shadow_mod.load_shadowqmlml_curves([r5, r11])
        ml_curves_obs = [c for c in ml_curves if int(c["n_q"]) in set(ML_NQ_OBS)]
        print(f"  ML curves: {len(ml_curves_obs)} in observed nq range {ML_NQ_OBS}")
        ml_repl_sink_raw = {} if collect_boot_repl else None
        ml_all_rows_raw = bootstrap_ml_dense_threshold_rows(
            ml_curves_obs,
            thresholds=thresholds,
            n_boot=args.n_boot_ml,
            seed=args.boot_seed,
            ci_level=args.boot_ci_level,
            replicate_sink=ml_repl_sink_raw,
        )
        # Normalise ML channel names ('thermal' → 'relaxation').
        for r in ml_all_rows_raw:
            r["channel"] = ML_CHANNEL_CANONICAL.get(r["channel"], r["channel"])
        # Normalize types/sigma_k consistently (also handles JSON-like strings).
        ml_all_rows_raw = _postprocess_rows(ml_all_rows_raw)
        if collect_boot_repl and ml_repl_sink_raw:
            for (ch_raw, p_val, nq), payload in ml_repl_sink_raw.items():
                ch = ML_CHANNEL_CANONICAL.get(str(ch_raw), str(ch_raw))
                key = (str(ch), float(p_val))
                ml_repl_cases.setdefault(key, {})[int(nq)] = payload
        if cache_write and ml_cache_path is not None and ml_cache_meta:
            _save_rows_json_with_meta(ml_cache_path, ml_all_rows_raw, ml_cache_meta)
            print(f"  ML cache saved → {ml_cache_path}")
    elif collect_boot_repl:
        print("  ML rows loaded; recomputing ML bootstrap replicates for diagnostics …")
        if r5 is None or r11 is None:
            r5, r11 = shadow_mod.find_shadow_results_json(args.results_q5_10, args.results_q11_12)
        ml_curves = shadow_mod.load_shadowqmlml_curves([r5, r11])
        ml_curves_obs = [c for c in ml_curves if int(c["n_q"]) in set(ML_NQ_OBS)]
        ml_repl_sink_raw = {}
        _ = bootstrap_ml_dense_threshold_rows(
            ml_curves_obs,
            thresholds=thresholds,
            n_boot=args.n_boot_ml,
            seed=args.boot_seed,
            ci_level=args.boot_ci_level,
            replicate_sink=ml_repl_sink_raw,
        )
        for (ch_raw, p_val, nq), payload in ml_repl_sink_raw.items():
            ch = ML_CHANNEL_CANONICAL.get(str(ch_raw), str(ch_raw))
            key = (str(ch), float(p_val))
            ml_repl_cases.setdefault(key, {})[int(nq)] = payload

    if args.save_ml_rows:
        ml_path = out_dir / "ml_bootstrap_threshold_rows.json"
        _save_rows_json_with_meta(ml_path, ml_all_rows_raw, ml_cache_meta if ml_cache_meta else {"method": "ml"})
        print(f"  ML rows saved → {ml_path}")

    n_ml_ok = sum(1 for r in ml_all_rows_raw if _is_inference_row(r))
    print(f"  ML bootstrap: {n_ml_ok}/{len(ml_all_rows_raw)} ok rows")

    print("\nBuilding ML opt1 predictors …")
    ml_predictors: Dict[Tuple[str, str], Predictor] = {}
    ml_predictor_meta: dict = {}
    for ch in CHANNEL_ORDER:
        for amp_str in PLOT_AMPLITUDES:
            p_val = float(amp_str)
            ok_rows = [r for r in ml_all_rows_raw
                       if _is_inference_row(r) and r["channel"] == ch and abs(r["p"] - p_val) < 1e-8]
            pred = _build_opt1_predictor(ok_rows, source_label="ml",
                                         keep_cv_details=bool(args.forward_cv),
                                         extrap_min_horizons=args.extrap_min_horizons,
                                         extrap_max_sigma_cv=args.extrap_max_sigma_cv,
                                         extrap_max_rmse_ratio=args.extrap_max_rmse_ratio,
                                         cv_thr_pct_min=args.cv_thr_pct_min,
                                         cv_thr_pct_max=args.cv_thr_pct_max,
                                         sigma_cv_k_mode=args.sigma_cv_k_mode,
                                         sigma_cv_k_coverage=args.sigma_cv_k_coverage,
                                         cv_coverage_zs=args.cv_coverage_zs,
                                         enforce_threshold_monotone=bool(args.enforce_threshold_monotone),
                                         monotone_threshold_grid_size=int(args.threshold_monotone_grid_size),
                                         interval_method=str(args.interval_method),
                                         conformal_holdout_count=int(args.conformal_holdout_count),
                                         conformal_min_abs_errors=int(args.conformal_min_abs_errors),
                                         strict_split_conformal=bool(args.strict_split_conformal))
            if pred is None:
                print(f"  WARN ML predictor FAILED: ch={ch} p={amp_str} "
                      f"({len(ok_rows)} ok rows available)")
                continue
            ml_predictors[(ch, amp_str)] = pred
            meta_entry = {
                **pred.meta,
                "n_ok_rows": len(ok_rows),
                "threshold_min": pred.threshold_min,
                "threshold_max": pred.threshold_max,
                "sigma_cv": float(pred.sigma_cv),
                "sigma_cv_k": float(pred.sigma_cv_k),
                "sigma_cv_k_obs": float(pred.sigma_cv_k_obs),
                "sigma_cv_k_extrap": float(pred.sigma_cv_k_extrap),
                "sigma_cv_log2_at_obs_max": float(pred.sigma_cv_log2_at_obs_max),
                "sigma_cv_log2_gate_at_obs_max": float(pred.sigma_cv_log2_gate_at_obs_max),
            }
            ml_predictor_meta[(ch, amp_str)] = meta_entry
            print(f"  ML [{ch},{amp_str}]: thr=[{pred.threshold_min:.3f},{pred.threshold_max:.3f}]"
                  f"  n_fits={pred.meta['n_threshold_fits']}  has_cov={pred.meta['has_parametric_cov']}")
            _warn_predictor_quality("ML", ch, amp_str, pred)

    # ── Compute comparison ─────────────────────────────────────────────────────
    print(f"\nComputing comparison for {len(devices)} device(s), "
          f"{len(readout_errors)} readout_error(s), {len(etas)} eta(s), "
          f"{len(nq_pred)} nq values …")
    curve_bootstrap_ci_summary = {
        "requested": bool(args.curve_bootstrap_ci),
        "applied": False,
        "fit_mode_requested": str(args.curve_bootstrap_fit),
    }
    comp_rows, comp_diag = compute_comparison(
        qdata=qdata,
        hg_predictors=hg_predictors,
        eig_predictors=eig_predictors,
        ml_predictors=ml_predictors,
        devices=devices,
        readout_errors=readout_errors,
        channels=CHANNEL_ORDER,
        amplitudes=PLOT_AMPLITUDES,
        etas=etas,
        nq_pred=nq_pred,
        target_floor=args.target_floor,
        ci_z=args.ci_z,
        random_baseline_fix=bool(args.random_baseline_fix),
        random_baseline_eps0=float(args.random_baseline_eps0),
        random_baseline_decay=float(args.random_baseline_decay),
        random_baseline_nq_ref=int(args.random_baseline_nq_ref),
    )
    n_ok_rows   = sum(1 for r in comp_rows if r.get("pred_status") == "ok")
    n_cens_rows = sum(1 for r in comp_rows if r.get("pred_status") in ("left_censored", "right_censored"))
    n_untrusted_rows = sum(1 for r in comp_rows if r.get("pred_status") == "untrusted_extrapolation")
    n_rb_rows = sum(1 for r in comp_rows if r.get("pred_status") == "random_baseline")
    print(f"  {len(comp_rows)} total rows: {n_ok_rows} ok, {n_cens_rows} censored, "
          f"{n_untrusted_rows} untrusted_extrapolation, {n_rb_rows} random_baseline, "
          f"{len(comp_rows) - n_ok_rows - n_cens_rows - n_untrusted_rows - n_rb_rows} other")

    # ── Write CSV + diagnostics ────────────────────────────────────────────────
    etas_tag    = "_".join(f"{e:.2f}".replace(".", "p") for e in etas)
    floor_tag   = f"_floor{args.target_floor:.2f}".replace(".", "p")
    boot_tag    = f"_nhg{args.n_boot_hg}_nml{args.n_boot_ml}"
    base_tag    = f"qvc_unified_eta_{etas_tag}{floor_tag}{boot_tag}"

    curve_bootstrap_surfaces: Dict[str, Dict[Tuple[str, str], dict]] = {
        "hypergraph": {},
        "eigenshadow": {},
        "ml": {},
    }
    method_rows_for_sigma: Dict[str, List[dict]] = {
        "hypergraph": hg_all_rows,
        "eigenshadow": eig_all_rows,
        "ml": ml_all_rows_raw,
    }
    bootstrap_repl_summary = {
        "requested": bool(collect_boot_repl),
        "save_replicates": bool(args.save_bootstrap_replicates),
        "compute_corr_diagnostics": bool(args.bootstrap_corr_diagnostics),
        "curve_bootstrap_ci_requested": bool(args.curve_bootstrap_ci),
        "curve_bootstrap_fit_requested": str(args.curve_bootstrap_fit),
        "replicates_dir": str(boot_repl_dir.resolve()) if collect_boot_repl else None,
        "cases_with_tensors": 0,
        "cases_saved_npz": 0,
        "corr_cases": 0,
        "corr_diagnostics_path": None,
        "saved_npz_paths": [],
        "methods": {},
    }
    if collect_boot_repl:
        method_case_map: Dict[str, Dict[Tuple[str, float], Dict[int, dict]]] = {
            "hypergraph": hg_repl_cases,
            "eigenshadow": eig_repl_cases,
            "ml": ml_repl_cases,
        }
        corr_cases: List[dict] = []
        saved_npz_paths: List[str] = []
        for method, case_map in method_case_map.items():
            method_meta = {"n_cases_with_replicates": 0, "n_cases_with_tensors": 0}
            for (ch, p_val), replicate_by_nq in sorted(case_map.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                method_meta["n_cases_with_replicates"] += 1
                assembled = _assemble_case_kx_boot_btn(replicate_by_nq)
                if assembled is None:
                    continue
                thr, nqs, kx_btn = assembled
                method_meta["n_cases_with_tensors"] += 1
                bootstrap_repl_summary["cases_with_tensors"] += 1
                amp_tag = _amp_tag_from_float(float(p_val))
                sigma_tn = _extract_sigma_k_tn_for_case(
                    method_rows_for_sigma.get(str(method), []),
                    channel=str(ch),
                    amplitude_tag=str(amp_tag),
                    thresholds=np.asarray(thr, dtype=float),
                    nq_values=np.asarray(nqs, dtype=int),
                )
                curve_bootstrap_surfaces.setdefault(str(method), {})[(str(ch), str(amp_tag))] = {
                    "thresholds": np.asarray(thr, dtype=float),
                    "nq_values": np.asarray(nqs, dtype=int),
                    "kx_boot_btn": np.asarray(kx_btn, dtype=np.float32),
                    "sigma_k_tn": np.asarray(sigma_tn, dtype=np.float32),
                }
                case_tag = f"{method}_{ch}_p{_tag_float(float(p_val))}"
                if args.save_bootstrap_replicates:
                    npz_path = boot_repl_dir / f"{base_tag}_{case_tag}_kx_boot_btn.npz"
                    _save_kx_boot_case_npz(
                        npz_path,
                        method=method,
                        channel=str(ch),
                        amplitude=float(p_val),
                        thresholds=thr,
                        nq_values=nqs,
                        kx_boot_btn=kx_btn,
                        cache_meta={
                            "script_sha256": str(script_sha256),
                            "method": str(method),
                            "channel": str(ch),
                            "amplitude": float(p_val),
                            "n_boot": int(kx_btn.shape[0]),
                            "thresholds_sha256": str(_array_sha256(thr)),
                        },
                    )
                    saved_npz_paths.append(str(npz_path))
                if args.bootstrap_corr_diagnostics:
                    per_nq = {}
                    for n_idx, nq in enumerate(nqs):
                        per_nq[str(int(nq))] = _threshold_corr_summary(kx_btn[:, :, n_idx], thr)
                    nq_corr = _nq_corr_summary_from_btn(kx_btn, nqs, thr)
                    offdiag_means = [
                        _to_float(v.get("rho_offdiag_mean", float("nan")))
                        for v in per_nq.values()
                        if _is_finite(v.get("rho_offdiag_mean", float("nan")))
                    ]
                    corr_cases.append({
                        "method": str(method),
                        "channel": str(ch),
                        "amplitude": float(p_val),
                        "amplitude_tag": str(amp_tag),
                        "n_boot": int(kx_btn.shape[0]),
                        "n_thresholds": int(kx_btn.shape[1]),
                        "n_nq": int(kx_btn.shape[2]),
                        "nq_values": [int(v) for v in nqs.tolist()],
                        "thresholds": [float(v) for v in thr.tolist()],
                        "rho_offdiag_mean_across_nq_mean": float(np.nanmean(offdiag_means)) if offdiag_means else float("nan"),
                        "rho_offdiag_mean_across_nq_median": float(np.nanmedian(offdiag_means)) if offdiag_means else float("nan"),
                        "threshold_corr_by_nq": per_nq,
                        "nq_corr_at_fixed_threshold": nq_corr,
                    })
            bootstrap_repl_summary["methods"][method] = method_meta

        if args.save_bootstrap_replicates:
            bootstrap_repl_summary["cases_saved_npz"] = int(len(saved_npz_paths))
            bootstrap_repl_summary["saved_npz_paths"] = saved_npz_paths
        if args.bootstrap_corr_diagnostics:
            bootstrap_repl_summary["corr_cases"] = int(len(corr_cases))
            corr_diag_path = boot_repl_dir / f"{base_tag}_bootstrap_threshold_corr_diagnostics.json"
            corr_payload = {
                "base_tag": str(base_tag),
                "n_boot_hg": int(args.n_boot_hg),
                "n_boot_ml": int(args.n_boot_ml),
                "boot_seed": int(args.boot_seed),
                "boot_ci_level": float(args.boot_ci_level),
                "thresholds": [float(v) for v in np.asarray(thresholds, dtype=float).tolist()],
                "script_sha256": str(script_sha256),
                "methods": bootstrap_repl_summary["methods"],
                "cases": corr_cases,
            }
            corr_diag_path.write_text(json.dumps(corr_payload, indent=2, default=str))
            bootstrap_repl_summary["corr_diagnostics_path"] = str(corr_diag_path)
            print(f"Bootstrap threshold-correlation diagnostics: {corr_diag_path}")

    if bool(args.curve_bootstrap_ci):
        n_surface_cases = sum(len(v) for v in curve_bootstrap_surfaces.values())
        if n_surface_cases > 0:
            cb_sum = _apply_curve_bootstrap_intervals_to_rows(
                rows=comp_rows,
                qdata=qdata,
                devices=devices,
                readout_errors=readout_errors,
                channels=CHANNEL_ORDER,
                amplitudes=PLOT_AMPLITUDES,
                etas=etas,
                nq_pred=nq_pred,
                case_surfaces=curve_bootstrap_surfaces,
                target_floor=float(args.target_floor),
                ci_z=float(args.ci_z),
                enforce_threshold_monotone=bool(args.enforce_threshold_monotone),
                simultaneous=bool(args.curve_bootstrap_simultaneous),
                min_valid=int(args.curve_bootstrap_min_valid),
                fit_mode=str(args.curve_bootstrap_fit),
            )
            curve_bootstrap_ci_summary = {"requested": True, "applied": True, **cb_sum}
        else:
            curve_bootstrap_ci_summary = {
                "requested": True,
                "applied": False,
                "reason": "no_bootstrap_surfaces_available",
                "fit_mode_requested": str(args.curve_bootstrap_fit),
            }

    csv_path    = out_dir / f"{base_tag}.csv"
    diag_path   = out_dir / f"{base_tag}_diagnostics.json"

    _write_csv(csv_path, comp_rows)
    interval_case_table, interval_case_summary = _interval_method_case_table(
        hg_predictor_meta=hg_predictor_meta,
        eig_predictor_meta=eig_predictor_meta,
        ml_predictor_meta=ml_predictor_meta,
    )

    comp_diag.update({
        "etas":              etas,
        "target_floor":      args.target_floor,
        "n_boot_hg":         args.n_boot_hg,
        "n_boot_eigenshadow": args.n_boot_hg,
        "n_boot_ml":         args.n_boot_ml,
        "boot_ci_level":     args.boot_ci_level,
        "ci_z":              args.ci_z,
        "thresholds_range":  [float(thresholds[0]), float(thresholds[-1])],
        "step4_model":       STEP4_MODEL,
        "hg_predictors":     {str(k): v for k, v in hg_predictor_meta.items()},
        "eigenshadow_predictors": {str(k): v for k, v in eig_predictor_meta.items()},
        "ml_predictors":     {str(k): v for k, v in ml_predictor_meta.items()},
        "hg_predictor_keys": [str(k) for k in hg_predictors],
        "eigenshadow_predictor_keys": [str(k) for k in eig_predictors],
        "ml_predictor_keys": [str(k) for k in ml_predictors],
        "n_comp_rows":       len(comp_rows),
        "n_ok_rows":         n_ok_rows,
        "n_censored_rows":   n_cens_rows,
        "n_untrusted_extrapolation_rows": n_untrusted_rows,
        "n_random_baseline_rows": n_rb_rows,
        "interval_method_case_table": interval_case_table,
        "interval_method_case_summary": interval_case_summary,
        "extrapolation_gate": {
            "min_total_horizons": int(args.extrap_min_horizons),
            "max_sigma_cv": float(args.extrap_max_sigma_cv),
            "max_rmse_ratio": float(args.extrap_max_rmse_ratio),
        },
        "interval_config": {
            "interval_method_requested": str(args.interval_method),
            "conformal_holdout_count": int(args.conformal_holdout_count),
            "conformal_min_abs_errors": int(args.conformal_min_abs_errors),
            "strict_split_conformal": bool(args.strict_split_conformal),
            "curve_bootstrap_fit": str(args.curve_bootstrap_fit),
        },
        "curve_bootstrap_ci": curve_bootstrap_ci_summary,
        "random_baseline_override": {
            "enabled": bool(args.random_baseline_fix),
            "baseline": float(L_BASELINE),
            "eps0": float(args.random_baseline_eps0),
            "decay": float(args.random_baseline_decay),
            "nq_ref": int(args.random_baseline_nq_ref),
            "formula": "eps(nq)=eps0*exp(-decay*max(nq-nq_ref,0)); override when q_acc-eta < 0.5+eps(nq)",
        },
        "cv_threshold_percentiles": [float(args.cv_thr_pct_min), float(args.cv_thr_pct_max)],
        "cv_max_horizon": int(CV_MAX_HORIZON),
        "sigma_cv_k_mode": str(args.sigma_cv_k_mode),
        "sigma_cv_k_coverage": float(args.sigma_cv_k_coverage),
        "cv_coverage_zs": [float(z) for z in (args.cv_coverage_zs if args.cv_coverage_zs else CV_COVERAGE_ZS_DEFAULT)],
        "enforce_threshold_monotone": bool(args.enforce_threshold_monotone),
        "threshold_monotone_grid_size": int(args.threshold_monotone_grid_size),
        "bootstrap_cache": {
            "mode": str(cache_mode),
            "dir": str(cache_dir.resolve()) if cache_mode != "off" else None,
            "hg_cache_path": str(hg_cache_path) if 'hg_cache_path' in locals() else None,
            "eigenshadow_cache_path": str(eig_cache_path) if 'eig_cache_path' in locals() else None,
            "ml_cache_path": str(ml_cache_path) if ml_cache_path is not None else None,
        },
        "bootstrap_replicates": bootstrap_repl_summary,
        "source_signatures": {
            "hypergraph": hg_source_sig,
            "eigenshadow": eig_source_sig,
            "ml": {
                "sha256": ml_cache_meta.get("source_signature_sha256", None),
                "source_paths": ml_cache_meta.get("source_paths", None),
            },
        },
    })
    diag_path.write_text(json.dumps(comp_diag, indent=2, default=str))



    if bool(getattr(args, "run_sensitivity", False)):
        try:
            # Representative configuration for sensitivity: first device/readout.
            dev0 = devices[0] if devices else "I"
            re0 = readout_errors[0] if readout_errors else "0%"
            baseline_args = {
                "extrap_min_horizons": int(args.extrap_min_horizons),
                "extrap_max_sigma_cv": float(args.extrap_max_sigma_cv),
                "extrap_max_rmse_ratio": float(args.extrap_max_rmse_ratio),
                "cv_thr_pct_min": float(args.cv_thr_pct_min),
                "cv_thr_pct_max": float(args.cv_thr_pct_max),
                "sigma_cv_k_mode": str(args.sigma_cv_k_mode),
                "sigma_cv_k_coverage": float(args.sigma_cv_k_coverage),
                "enforce_threshold_monotone": bool(args.enforce_threshold_monotone),
                "threshold_monotone_grid_size": int(args.threshold_monotone_grid_size),
                "interval_method": str(args.interval_method),
                "conformal_holdout_count": int(args.conformal_holdout_count),
                "conformal_min_abs_errors": int(args.conformal_min_abs_errors),
                "strict_split_conformal": bool(args.strict_split_conformal),
                "random_baseline_fix": bool(args.random_baseline_fix),
                "random_baseline_eps0": float(args.random_baseline_eps0),
                "random_baseline_decay": float(args.random_baseline_decay),
                "random_baseline_nq_ref": int(args.random_baseline_nq_ref),
            }
            _write_sensitivity_report(
                out_dir=out_dir,
                qdata=qdata,
                hg_all_rows=hg_all_rows,
                eig_all_rows=eig_all_rows,
                ml_all_rows=ml_all_rows_raw,
                device=dev0,
                readout_error=re0,
                target_floor=float(args.target_floor),
                ci_z=float(args.ci_z),
                baseline_args=baseline_args,
                thresholds=thresholds,
                load_hg_fn=load_hg_fn,
                hm_dir=hm_dir,
                eig_dir=eig_dir,
                shadow_mod=shadow_mod,
                results_q5_10=args.results_q5_10,
                results_q11_12=args.results_q11_12,
                n_boot_hg=int(args.n_boot_hg),
                n_boot_ml=int(args.n_boot_ml),
                boot_ci_level=float(args.boot_ci_level),
                boot_seed=int(args.boot_seed),
                hg_bootstrap_mode=str(args.hg_bootstrap_mode),
                sensitivity_bootstrap_checks=bool(args.sensitivity_bootstrap_checks),
                sensitivity_boot_counts=list(args.sensitivity_boot_counts) if args.sensitivity_boot_counts else None,
                sensitivity_amps=list(SENSITIVITY_AMPS_DEFAULT),
                sensitivity_etas=list(SENSITIVITY_ETAS_DEFAULT),
                sensitivity_deltas=list(SENSITIVITY_DELTAS_DEFAULT),
                curve_case_surfaces=curve_bootstrap_surfaces if bool(curve_bootstrap_surfaces) else None,
                curve_bootstrap_fit=str(args.curve_bootstrap_fit),
                curve_bootstrap_min_valid=int(args.curve_bootstrap_min_valid),
                curve_bootstrap_simultaneous=bool(args.curve_bootstrap_simultaneous),
            )
            print(f"  Sensitivity report written → {out_dir / 'sensitivity_report.md'}")
        except Exception as _sens_err:
            print(f"  WARN sensitivity report failed: {_sens_err}")

    # ── Render figures ─────────────────────────────────────────────────────────
    print("\nRendering figures …")
    ci_label = f"empirical-CV + WLS (z={args.ci_z:.2g})"
    for device in devices:
        for re in readout_errors:
            re_tag   = re.replace("%", "pct")
            fig_base = out_dir / f"{base_tag}_{device}_{re_tag}"
            dev_rows = [r for r in comp_rows
                        if r["device"] == device and r["readout_error"] == re]
            if not dev_rows:
                print(f"  No rows for device={device} re={re}; skipping figure.")
                continue
            title = (
                f"Quantum vs Classical (Unified Step4) — "
                f"device={DEVICE_TITLES.get(device, device)}, readout_error={re}\n"
                f"k_x = C + β/nq  |  target = q_acc(nq)−η  |  "
                f"floor={args.target_floor:.2f}  |  CI={ci_label}"
            )
            for ext in ("png", "pdf"):
                _plot_comparison_figure(
                    rows=dev_rows,
                    etas=etas,
                    device=device,
                    readout_error=re,
                    channels=CHANNEL_ORDER,
                    amplitudes=PLOT_AMPLITUDES,
                    nq_pred=nq_pred,
                    out_path=fig_base.with_suffix(f".{ext}"),
                    title=title,
                    ci_z=args.ci_z,
                    show_bands=not args.no_bands,
                    target_floor=args.target_floor,
                )
            print(f"  Saved: {fig_base}.png/.pdf")

    print("\nDone.")
    print(f"CSV:         {csv_path}")
    print(f"Diagnostics: {diag_path}")
    print(f"Outputs:     {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
