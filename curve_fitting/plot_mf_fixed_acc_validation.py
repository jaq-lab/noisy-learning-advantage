#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except Exception as e:  # pragma: no cover
    plt = None
    MPL_IMPORT_ERROR = e


SCRIPT_DIR = Path(__file__).resolve().parent
UNIFIED_PATH = SCRIPT_DIR / "plot_quantum_vs_classical_nps_unified.py"
HG_RAW_PATH = SCRIPT_DIR / "plot_hypergraph_vs_ml_nps_multiaccuracy_from_raw_discrete.py"
SHADOW_PATH = SCRIPT_DIR / "plot_shadowqmlml_nps_from_quantum_accuracy_curves.py"

METHOD_ORDER = ["hypergraph", "eigenshadow", "ml"]
METHOD_LABELS = {
    "hypergraph": "Hypergraph",
    "eigenshadow": "Eigenshadow",
    "ml": "Shadow-based ML",
}
METHOD_COLORS = {
    "hypergraph": "#1f77b4",
    "eigenshadow": "#d17b00",
    "ml": "#2f7d3d",
}
TARGET_COLORS = {0.6: "#1f77b4", 0.8: "#d62728"}
P_INTENSITY = {0.01: 0.55, 0.05: 0.72, 0.1: 1.0}


def _to_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _safe_nps_from_log2(log2_v: float) -> float:
    if not _is_finite(log2_v):
        return float("nan")
    return float(np.power(2.0, np.clip(max(0.0, float(log2_v)), -1020.0, 1020.0)))


def _epsilon_color_map_fig3(amplitudes: List[float]) -> Dict[float, str]:
    """
    Match fig3_mf_protocols.ipynb sampling: curves use ``cmap(np.linspace(0.4, 0.9, n))``
    per channel; legends use ``plt.cm.Blues/Reds/Greens(0.7)``. Here the weakest→strongest
    ε_p maps to Blues / Greens / Reds with the same linspace(0.4, 0.9, n) levels (softer
    than high-saturation colormap tails).
    """
    ps = sorted({float(x) for x in amplitudes})
    if plt is None:
        return {p: "#555555" for p in ps}
    n = len(ps)
    if n == 1:
        return {ps[0]: mcolors.to_hex(plt.cm.Blues(0.7))}
    levels = np.linspace(0.4, 0.9, n)
    if n == 2:
        cmaps = (plt.cm.Blues, plt.cm.Reds)
    elif n == 3:
        cmaps = (plt.cm.Blues, plt.cm.Greens, plt.cm.Reds)
    else:
        cmap = plt.get_cmap("tab10")
        return {p: mcolors.to_hex(cmap(i / max(n - 1, 1))) for i, p in enumerate(ps)}
    return {p: mcolors.to_hex(cm(float(lev))) for p, cm, lev in zip(ps, cmaps, levels)}


def _target_linestyle_map(targets: List[float]) -> Dict[float, str]:
    st = sorted({float(t) for t in targets})
    cycle = ["-", "--", "-.", (0, (3, 1, 1, 1))]
    return {t: cycle[i % len(cycle)] for i, t in enumerate(st)}


def _shade_color(base_color: str, intensity: float) -> tuple:
    s = float(np.clip(intensity, 0.0, 1.0))
    rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
    out = 1.0 - s * (1.0 - rgb)
    return tuple(np.clip(out, 0.0, 1.0))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _obs_nq_by_method(unified, method: str, rows: List[dict], nq_min: int = 5) -> List[int]:
    nq_set = {
        int(_to_float(r.get("n_q", r.get("nq", np.nan))))
        for r in rows
        if unified._is_inference_row(r) and _is_finite(r.get("n_q", r.get("nq", np.nan)))
    }
    obs_from_rows = sorted(v for v in nq_set if int(v) >= int(nq_min))
    if obs_from_rows:
        return obs_from_rows
    obs_default = list(unified.OBS_NQ_BY_METHOD.get(method, []))
    return sorted(int(v) for v in obs_default if int(v) >= int(nq_min))


def _build_method_rows(
    unified,
    hg_mod,
    shadow_mod,
    *,
    method: str,
    channel: str,
    amplitude: float,
    thresholds: np.ndarray,
    n_boot_hg: int,
    n_boot_ml: int,
    boot_seed: int,
    boot_ci_level: float,
    hg_bootstrap_mode: str,
    hypergraph_merged_dir: Path | None,
    shadow_surrogate_dir: Path | None,
    results_q5_10: str | None,
    results_q11_12: str | None,
    rows_json: str | None,
) -> List[dict]:
    if rows_json:
        rows = unified._load_rows_json(Path(rows_json))
        if method == "ml":
            for r in rows:
                r["channel"] = unified.ML_CHANNEL_CANONICAL.get(r.get("channel"), r.get("channel"))
            rows = unified._postprocess_rows(rows)
    else:
        if method in {"hypergraph", "eigenshadow"}:
            load_hg_fn = hg_mod.load_hypergraph_acc_trials
            hm_dir = (
                Path(hypergraph_merged_dir)
                if (method == "hypergraph" and hypergraph_merged_dir)
                else (Path(shadow_surrogate_dir) if (method == "eigenshadow" and shadow_surrogate_dir) else None)
            )
            if hm_dir is None:
                hm_dir = unified._find_hg_merged() if method == "hypergraph" else unified._find_shadow_surrogate()
            nq_list = unified.HG_NQ_OBS if method == "hypergraph" else unified.EIGENSHADOW_NQ_OBS
            rows = unified.bootstrap_hg_dense_threshold_rows(
                load_hg_fn=load_hg_fn,
                hm_dir=hm_dir,
                nq_list=nq_list,
                channel=channel,
                amplitude=float(amplitude),
                thresholds=np.asarray(thresholds, dtype=float),
                n_boot=int(n_boot_hg),
                seed=int(boot_seed),
                ci_level=float(boot_ci_level),
                hg_bootstrap_mode=str(hg_bootstrap_mode),
            )
        elif method == "ml":
            r5, r11 = shadow_mod.find_shadow_results_json(results_q5_10, results_q11_12)
            ml_curves = shadow_mod.load_shadowqmlml_curves([r5, r11])
            ml_curves_obs = [c for c in ml_curves if int(c["n_q"]) in set(unified.ML_NQ_OBS)]
            rows = unified.bootstrap_ml_dense_threshold_rows(
                ml_curves_obs,
                thresholds=np.asarray(thresholds, dtype=float),
                n_boot=int(n_boot_ml),
                seed=int(boot_seed),
                ci_level=float(boot_ci_level),
            )
            for r in rows:
                r["channel"] = unified.ML_CHANNEL_CANONICAL.get(r["channel"], r["channel"])
            rows = unified._postprocess_rows(rows)
        else:
            raise ValueError(f"Unsupported method: {method}")

    return [
        r
        for r in rows
        if str(r.get("channel", "")) == str(channel)
        and abs(_to_float(r.get("p")) - float(amplitude)) < 1e-12
    ]


def _filter_rows_case(rows: List[dict], *, channel: str, amplitude: float) -> List[dict]:
    return [
        r
        for r in rows
        if str(r.get("channel", "")) == str(channel)
        and abs(_to_float(r.get("p")) - float(amplitude)) < 1e-12
    ]


def _build_predictor_for_method(unified, method_rows: List[dict], method: str):
    ok_rows = [r for r in method_rows if unified._is_inference_row(r)]
    if not ok_rows:
        return None
    return unified._build_opt1_predictor(ok_rows, source_label=str(method))


def _interp_field_at_target(rows_nq: List[dict], target: float, field: str = "k_x") -> float:
    """
    Interpolate one field (typically k_x) over threshold for a fixed n_q.
    Uses only status='ok' rows and linear interpolation in threshold.
    """
    pts: Dict[float, List[float]] = {}
    for r in rows_nq:
        if str(r.get("status", "")) != "ok":
            continue
        t = _to_float(r.get("threshold", np.nan))
        v = _to_float(r.get(field, np.nan))
        if _is_finite(t) and _is_finite(v):
            pts.setdefault(float(t), []).append(float(v))
    if len(pts) < 2:
        return float("nan")
    t_arr = np.array(sorted(pts.keys()), dtype=float)
    v_arr = np.array([float(np.mean(pts[t])) for t in t_arr], dtype=float)
    target = float(target)
    if target < float(t_arr.min()) or target > float(t_arr.max()):
        return float("nan")
    return float(np.interp(target, t_arr, v_arr))


def _observed_k_at_nq_target(rows: List[dict], nq: int, target: float) -> float:
    rows_nq = [r for r in rows if int(_to_float(r.get("n_q", r.get("nq", np.nan)))) == int(nq)]
    return _interp_field_at_target(rows_nq, float(target), field="k_x")


def _extract_observed_curve(rows: List[dict], obs_nq: List[int], target: float) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for nq in sorted(int(v) for v in obs_nq):
        k_obs = _observed_k_at_nq_target(rows, int(nq), float(target))
        if not _is_finite(k_obs):
            continue
        l2_obs = float(nq) * float(k_obs)
        nps_obs = _safe_nps_from_log2(l2_obs)
        if _is_finite(nps_obs) and nps_obs > 0:
            xs.append(float(nq))
            ys.append(float(nps_obs))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _predict_extrapolated_curve(unified, predictor, obs_max: int, nq_grid: np.ndarray, target: float, ci_z: float):
    x_ext: List[float] = []
    y_ext: List[float] = []
    y_lo: List[float] = []
    y_hi: List[float] = []
    x_untrusted: List[float] = []

    extrap_trusted = bool(predictor.meta.get("extrapolation_trusted", True))
    for nq in nq_grid:
        nq_i = int(nq)
        if nq_i <= int(obs_max):
            continue
        rec = unified._predict_point(predictor, float(target), nq_i, int(obs_max), ci_z=float(ci_z))
        if not extrap_trusted:
            rec["status"] = "untrusted_extrapolation"
        st = str(rec.get("status", ""))
        if st == "ok":
            nps = _to_float(rec.get("nps", np.nan))
            lo = _to_float(rec.get("nps_lo", np.nan))
            hi = _to_float(rec.get("nps_hi", np.nan))
            if _is_finite(nps) and nps > 0:
                x_ext.append(float(nq_i))
                y_ext.append(float(nps))
                y_lo.append(float(lo) if _is_finite(lo) and lo > 0 else float("nan"))
                y_hi.append(float(hi) if _is_finite(hi) and hi > 0 else float("nan"))
        elif st == "untrusted_extrapolation":
            x_untrusted.append(float(nq_i))
    return (
        np.asarray(x_ext, dtype=float),
        np.asarray(y_ext, dtype=float),
        np.asarray(y_lo, dtype=float),
        np.asarray(y_hi, dtype=float),
        np.asarray(x_untrusted, dtype=float),
    )


def _predict_model_curve_full_grid(
    unified, predictor, obs_max: int, nq_grid: np.ndarray, target: float, ci_z: float
):
    """
    Predicted n_c vs n_q on every n_q in nq_grid (status ok), plus untrusted n_q > obs_max
    when applicable. Includes n_q ≤ obs_max so the fit is visible in the low-n_q regime.
    """
    x_fit: List[float] = []
    y_fit: List[float] = []
    y_lo: List[float] = []
    y_hi: List[float] = []
    x_untrusted: List[float] = []

    extrap_trusted = bool(predictor.meta.get("extrapolation_trusted", True))
    for nq in nq_grid:
        nq_i = int(nq)
        rec = unified._predict_point(predictor, float(target), nq_i, int(obs_max), ci_z=float(ci_z))
        if not extrap_trusted and nq_i > int(obs_max):
            rec["status"] = "untrusted_extrapolation"
        st = str(rec.get("status", ""))
        if st == "ok":
            nps = _to_float(rec.get("nps", np.nan))
            lo = _to_float(rec.get("nps_lo", np.nan))
            hi = _to_float(rec.get("nps_hi", np.nan))
            if _is_finite(nps) and nps > 0:
                x_fit.append(float(nq_i))
                y_fit.append(float(nps))
                y_lo.append(float(lo) if _is_finite(lo) and lo > 0 else float("nan"))
                y_hi.append(float(hi) if _is_finite(hi) and hi > 0 else float("nan"))
        elif st == "untrusted_extrapolation" and nq_i > int(obs_max):
            x_untrusted.append(float(nq_i))
    return (
        np.asarray(x_fit, dtype=float),
        np.asarray(y_fit, dtype=float),
        np.asarray(y_lo, dtype=float),
        np.asarray(y_hi, dtype=float),
        np.asarray(x_untrusted, dtype=float),
    )


def _run_holdout_backtest(
    unified,
    *,
    method: str,
    channel: str,
    amplitude: float,
    rows: List[dict],
    targets: List[float],
    ci_z: float,
    holdout_counts: List[int],
    nq_min: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Walk-forward holdout on observed nq:
      hold out last h nq values (h in holdout_counts),
      fit on remaining nq, evaluate at exact target(s) on held-out nq.
    Returns (point_records, summary_records).
    """
    point_rows: List[dict] = []
    summary_rows: List[dict] = []
    obs_nq_all = _obs_nq_by_method(unified, method, rows, nq_min=nq_min)
    nominal_cov = float(2.0 * NormalDist().cdf(abs(float(ci_z))) - 1.0)

    for h in sorted(set(int(v) for v in holdout_counts if int(v) >= 1)):
        if len(obs_nq_all) < (h + 4):
            summary_rows.append(
                {
                    "method": method,
                    "channel": str(channel),
                    "amplitude": float(amplitude),
                    "holdout_count": int(h),
                    "target": "all",
                    "n_points": 0,
                    "nominal_coverage": nominal_cov,
                    "observed_coverage": float("nan"),
                    "mae_k": float("nan"),
                    "mae_log2": float("nan"),
                    "rmse_k": float("nan"),
                    "rmse_log2": float("nan"),
                    "reason": "insufficient_observed_nq",
                }
            )
            continue

        holdout_nq = obs_nq_all[-h:]
        train_nq = set(obs_nq_all[:-h])
        train_rows = [
            r
            for r in rows
            if unified._is_inference_row(r)
            and int(_to_float(r.get("n_q", r.get("nq", np.nan)))) in train_nq
        ]
        pred = unified._build_opt1_predictor(train_rows, source_label=f"{method}_holdout_h{h}")
        if pred is None:
            summary_rows.append(
                {
                    "method": method,
                    "channel": str(channel),
                    "amplitude": float(amplitude),
                    "holdout_count": int(h),
                    "target": "all",
                    "n_points": 0,
                    "nominal_coverage": nominal_cov,
                    "observed_coverage": float("nan"),
                    "mae_k": float("nan"),
                    "mae_log2": float("nan"),
                    "rmse_k": float("nan"),
                    "rmse_log2": float("nan"),
                    "reason": "predictor_fit_failed",
                }
            )
            continue

        obs_max_train = int(max(train_nq))
        for nq in holdout_nq:
            for target in targets:
                k_obs = _observed_k_at_nq_target(rows, int(nq), float(target))
                if not _is_finite(k_obs):
                    continue
                obs_log2 = float(nq) * float(k_obs)
                rec = unified._predict_point(pred, float(target), int(nq), int(obs_max_train), ci_z=float(ci_z))
                pred_log2 = _to_float(rec.get("log2_nps", np.nan))
                st = str(rec.get("status", ""))
                lo = _to_float(rec.get("log2_nps_lo", np.nan))
                hi = _to_float(rec.get("log2_nps_hi", np.nan))
                covered = (
                    _is_finite(lo)
                    and _is_finite(hi)
                    and _is_finite(obs_log2)
                    and (float(lo) <= float(obs_log2) <= float(hi))
                )
                err_log2 = float(pred_log2 - obs_log2) if _is_finite(pred_log2) else float("nan")
                err_k = float(err_log2 / float(nq)) if _is_finite(err_log2) and int(nq) > 0 else float("nan")
                point_rows.append(
                    {
                        "method": method,
                        "channel": str(channel),
                        "amplitude": float(amplitude),
                        "holdout_count": int(h),
                        "nq_holdout": int(nq),
                        "target": float(target),
                        "obs_k": float(k_obs),
                        "obs_log2_nps": float(obs_log2),
                        "pred_status": st,
                        "pred_log2_nps": float(pred_log2) if _is_finite(pred_log2) else float("nan"),
                        "pred_log2_nps_lo": float(lo) if _is_finite(lo) else float("nan"),
                        "pred_log2_nps_hi": float(hi) if _is_finite(hi) else float("nan"),
                        "err_log2": float(err_log2) if _is_finite(err_log2) else float("nan"),
                        "err_k": float(err_k) if _is_finite(err_k) else float("nan"),
                        "covered": int(bool(covered)),
                        "nominal_coverage": nominal_cov,
                        "train_nq_max": int(obs_max_train),
                    }
                )

        # summarize by target and pooled
        for t_key in [*targets, "all"]:
            rr = [
                r
                for r in point_rows
                if r["method"] == method
                and int(r["holdout_count"]) == int(h)
                and (r["target"] == t_key if t_key != "all" else True)
                and _is_finite(r.get("err_log2", np.nan))
            ]
            if not rr:
                summary_rows.append(
                    {
                        "method": method,
                        "channel": str(channel),
                        "amplitude": float(amplitude),
                        "holdout_count": int(h),
                        "target": t_key,
                        "n_points": 0,
                        "nominal_coverage": nominal_cov,
                        "observed_coverage": float("nan"),
                        "mae_k": float("nan"),
                        "mae_log2": float("nan"),
                        "rmse_k": float("nan"),
                        "rmse_log2": float("nan"),
                        "reason": "no_valid_points",
                    }
                )
                continue
            e_l2 = np.array([float(r["err_log2"]) for r in rr], dtype=float)
            e_k = np.array([float(r["err_k"]) for r in rr], dtype=float)
            cov = float(np.mean(np.array([int(r["covered"]) for r in rr], dtype=float)))
            summary_rows.append(
                {
                    "method": method,
                    "channel": str(channel),
                    "amplitude": float(amplitude),
                    "holdout_count": int(h),
                    "target": t_key,
                    "n_points": int(len(rr)),
                    "nominal_coverage": nominal_cov,
                    "observed_coverage": cov,
                    "mae_k": float(np.mean(np.abs(e_k))),
                    "mae_log2": float(np.mean(np.abs(e_l2))),
                    "rmse_k": float(np.sqrt(np.mean(e_k ** 2))),
                    "rmse_log2": float(np.sqrt(np.mean(e_l2 ** 2))),
                    "reason": "ok",
                }
            )

    return point_rows, summary_rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                cols.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_validation(
    unified,
    predictors_by_case: Dict[Tuple[str, str, float], object],
    rows_by_case: Dict[Tuple[str, str, float], List[dict]],
    *,
    channels: List[str],
    amplitudes: List[float],
    targets: List[float],
    nq_min: int,
    nq_max: int,
    ci_z: float,
    output_png: Path,
    output_pdf: Path,
    mplstyle_path: Path | None = None,
    figsize_inches: Tuple[float, float] | None = None,
    normalize_y_by_2pow_nq: bool = False,
    extrapolated_line_only: bool = False,
    plot_style: str = "default",
) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_IMPORT_ERROR}")

    if mplstyle_path is not None:
        plt.style.use(str(mplstyle_path))

    use_fig3 = str(plot_style) == "fig3_mf"
    eps_color_fig3 = _epsilon_color_map_fig3(amplitudes) if use_fig3 else {}
    tls_map = _target_linestyle_map(targets) if use_fig3 else {}
    line_only = bool(extrapolated_line_only) or use_fig3

    _fs_title = float(plt.rcParams.get("axes.titlesize", 9))
    _fs_label = float(plt.rcParams.get("axes.labelsize", 9))
    _fs_leg = float(plt.rcParams.get("legend.fontsize", 7))
    nq_grid = np.arange(int(nq_min), int(nq_max) + 1)
    nr = len(channels)
    nc = len(METHOD_ORDER)
    if figsize_inches is None:
        figsize_inches = (10.0, 6.0)
    fig, axes = plt.subplots(nr, nc, figsize=figsize_inches, sharex=True, sharey=True)
    if nr == 1:
        axes = np.array([axes])
    if nc == 1:
        axes = axes.reshape(nr, 1)

    all_y: List[float] = []
    for i, ch in enumerate(channels):
        for j, method in enumerate(METHOD_ORDER):
            ax = axes[i, j]
            obs_max_vals: List[int] = []
            any_drawn = False

            for p in amplitudes:
                pred = predictors_by_case.get((method, ch, float(p)))
                rows = rows_by_case.get((method, ch, float(p)), [])
                if pred is None or not rows:
                    continue
                obs_nq_list = _obs_nq_by_method(unified, method, rows, nq_min=nq_min)
                if not obs_nq_list:
                    continue
                obs_max = int(max(int(v) for v in obs_nq_list))
                obs_max_vals.append(obs_max)

                for target in targets:
                    if use_fig3:
                        col = eps_color_fig3[float(p)]
                        ls_model = tls_map[float(target)]
                    else:
                        base_col = TARGET_COLORS.get(float(target), "#333333")
                        col = _shade_color(base_col, P_INTENSITY.get(float(p), 0.88))
                        ls_model = "--"
                    x_obs, y_obs = _extract_observed_curve(rows, obs_nq_list, float(target))
                    if use_fig3:
                        x_ext, y_ext, y_lo, y_hi, x_untrusted = _predict_model_curve_full_grid(
                            unified,
                            pred,
                            obs_max,
                            nq_grid,
                            float(target),
                            float(ci_z),
                        )
                    else:
                        x_ext, y_ext, y_lo, y_hi, x_untrusted = _predict_extrapolated_curve(
                            unified,
                            pred,
                            obs_max,
                            nq_grid,
                            float(target),
                            float(ci_z),
                        )
                    if normalize_y_by_2pow_nq:
                        if x_obs.size > 0:
                            y_obs = y_obs / np.power(2.0, x_obs)
                        if x_ext.size > 0:
                            s_ext = np.power(2.0, x_ext)
                            y_ext = y_ext / s_ext
                            y_lo = np.where(np.isfinite(y_lo), y_lo / s_ext, y_lo)
                            y_hi = np.where(np.isfinite(y_hi), y_hi / s_ext, y_hi)

                    if x_obs.size > 0:
                        any_drawn = True
                        all_y.extend(list(y_obs))
                        _ms_obs = float(plt.rcParams.get("lines.markersize", 3.0)) + 1.2
                        if use_fig3:
                            ax.plot(
                                x_obs,
                                y_obs,
                                linestyle="none",
                                marker="o",
                                markersize=_ms_obs,
                                color=col,
                                markerfacecolor=col,
                                markeredgecolor="black",
                                markeredgewidth=0.42,
                                zorder=6,
                            )
                        else:
                            ax.plot(
                                x_obs,
                                y_obs,
                                color=col,
                                linestyle="-",
                                linewidth=float(plt.rcParams.get("lines.linewidth", 2.0)),
                                marker="o",
                                markersize=_ms_obs,
                                markerfacecolor=col,
                                markeredgecolor="black",
                                markeredgewidth=0.42,
                                zorder=6,
                            )

                    if x_ext.size > 0:
                        any_drawn = True
                        all_y.extend(list(y_ext))
                        _lw_ext = float(plt.rcParams.get("lines.linewidth", 2.0)) * 0.95
                        v = np.isfinite(y_lo) & np.isfinite(y_hi) & (y_lo > 0) & (y_hi > 0)
                        if np.any(v):
                            ax.fill_between(
                                x_ext[v],
                                y_lo[v],
                                y_hi[v],
                                color=col,
                                alpha=0.26,
                                linewidth=0.0,
                                zorder=2,
                            )
                            if not line_only:
                                ax.plot(x_ext[v], y_lo[v], color=col, linewidth=0.75, alpha=0.28, zorder=3)
                                ax.plot(x_ext[v], y_hi[v], color=col, linewidth=0.75, alpha=0.28, zorder=3)
                        if line_only:
                            ax.plot(
                                x_ext,
                                y_ext,
                                color=col,
                                linestyle=ls_model,
                                linewidth=_lw_ext,
                                zorder=5,
                            )
                        else:
                            ax.plot(
                                x_ext,
                                y_ext,
                                color=col,
                                linestyle="--",
                                linewidth=_lw_ext,
                                marker="^",
                                markersize=float(plt.rcParams.get("lines.markersize", 3.0)) + 1.4,
                                markerfacecolor="white",
                                markeredgecolor=col,
                                markeredgewidth=0.95,
                                zorder=5,
                            )

                    if x_untrusted.size > 0 and x_obs.size >= 2:
                        any_drawn = True
                        lx = np.asarray(x_obs, dtype=float)
                        ly = np.log2(np.asarray(y_obs, dtype=float))
                        m, b = np.polyfit(lx, ly, 1)
                        y_un = np.power(2.0, m * x_untrusted + b)
                        all_y.extend(list(y_un))
                        ax.plot(x_untrusted, y_un, color=col, linestyle=":", linewidth=1.6, alpha=0.45, zorder=1)

            if obs_max_vals:
                obs_split = int(max(obs_max_vals))
                ax.axvline(float(obs_split) + 0.5, color="gray", linestyle=":", linewidth=0.95, alpha=0.7)
                ax.text(
                    float(obs_split) + 0.65,
                    0.95,
                    "obs→extrap",
                    transform=ax.get_xaxis_transform(),
                    fontsize=_fs_label,
                    color="gray",
                )

            if i == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=_fs_title)
            if j == 0:
                ylabel = (
                    f"{ch.title()}\n"
                    + r"Sample Complexity $n_c(A_{\mathrm{target}}) / 2^{n_q}$"
                    if normalize_y_by_2pow_nq
                    else f"{ch.title()}\n" + r"Sample Complexity $n_c(A_{\mathrm{target}})$"
                )
                ax.set_ylabel(ylabel, fontsize=_fs_label)
            if i == (nr - 1):
                ax.set_xlabel(r"Number of Qubits $n_q$", fontsize=_fs_label)

            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(float(nq_min) - 0.4, float(nq_max) + 0.4)
            if not any_drawn:
                ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes, ha="center", va="center", fontsize=10)

    if all_y:
        ymin = min(max(min(all_y) / 2.0, 0.8), 1.0)
        ymax = max(all_y) * 1.9
        for ax in axes.ravel():
            ax.set_ylim(ymin, ymax)

    _lw_leg = float(plt.rcParams.get("lines.linewidth", 2.0)) * 1.2
    _ms_leg = float(plt.rcParams.get("lines.markersize", 3.0)) + 1.5
    _lw_leg_ext = float(plt.rcParams.get("lines.linewidth", 2.0)) * 0.95
    if use_fig3:
        p_handles = [
            Line2D(
                [0],
                [0],
                color=eps_color_fig3[float(p)],
                linestyle="-",
                linewidth=_lw_leg,
                label=rf"$\epsilon_p = {float(p):g}$",
            )
            for p in sorted(set(float(x) for x in amplitudes))
        ]
        acc_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=tls_map[float(t)],
                linewidth=_lw_leg_ext * 1.2,
                label=rf"$A_{{\mathrm{{target}}}} = {t:g}$",
            )
            for t in sorted(set(float(x) for x in targets))
        ]
        sem_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="none",
                marker="o",
                markersize=_ms_leg,
                markeredgecolor="black",
                markeredgewidth=0.42,
                label="observed (empirical)",
            ),
            Patch(
                facecolor="gray",
                alpha=0.18,
                label=f"CI band on model curve ({ci_z:.3g}-sigma)",
            ),
            Line2D([0], [0], color="gray", linestyle=":", linewidth=1.7, label="untrusted extrapolation guide"),
        ]
    else:
        acc_handles = [
            Line2D(
                [0],
                [0],
                color=TARGET_COLORS[t],
                linewidth=_lw_leg,
                label=rf"$A_{{\mathrm{{target}}}} = {t:g}$",
            )
            for t in targets
        ]
        p_handles = [
            Line2D(
                [0],
                [0],
                color=_shade_color("#555555", P_INTENSITY.get(float(p), 0.88)),
                linewidth=_lw_leg,
                label=rf"$\epsilon_p = {float(p):g}$",
            )
            for p in amplitudes
        ]
        if extrapolated_line_only:
            extrap_leg = Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=_lw_leg_ext * 1.2,
                label="extrapolated (fit, exact target)",
            )
        else:
            extrap_leg = Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                marker="^",
                markerfacecolor="white",
                markersize=_ms_leg,
                label="extrapolated (fit, exact target)",
            )
        sem_handles = [
            Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=_ms_leg, label="observed (empirical)"),
            extrap_leg,
            Patch(facecolor="gray", alpha=0.18, label=f"CI band on extrapolated segment ({ci_z:.3g}-sigma)"),
            Line2D([0], [0], color="gray", linestyle=":", linewidth=1.7, label="untrusted extrapolation guide"),
        ]
    # In-panel legends: second row (relaxation when nr>=2); col0 noise, col1 semantics, col2 A_target
    _leg_row = 1 if nr > 1 else 0
    _leg_fs = max(_fs_leg, 8.0)
    _leg_kw = {"frameon": True, "fontsize": _leg_fs}
    _leg_fs_sem = max(_leg_fs - 2.0, 6.0)
    _leg_loc = {"loc": "upper left", "bbox_to_anchor": (0.02, 0.88)}  # ~10% below previous top anchor
    axes[_leg_row, 0].legend(handles=p_handles, **_leg_loc, **_leg_kw)
    if nc > 1:
        if nc > 2:
            axes[_leg_row, 1].legend(
                handles=sem_handles,
                **_leg_loc,
                frameon=True,
                fontsize=_leg_fs_sem,
            )
            axes[_leg_row, 2].legend(handles=acc_handles, **_leg_loc, **_leg_kw)
        else:
            axes[_leg_row, 1].legend(handles=acc_handles, **_leg_loc, **_leg_kw)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate fixed-accuracy (0.6/0.8) n_c vs n_q curves with observed points and extrapolated CI bands."
    )
    ap.add_argument("--channel", type=str, default="relaxation", choices=["dephasing", "depolarizing", "relaxation"])
    ap.add_argument("--channels", nargs="*", default=None,
                    help="Optional multiple channels; overrides --channel. Example: --channels dephasing relaxation")
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--ps", nargs="*", type=float, default=None,
                    help="Optional multiple noise strengths p; overrides --p. Example: --ps 0.05 0.1")
    ap.add_argument("--targets", nargs="*", type=float, default=[0.6, 0.8])
    ap.add_argument("--nq-min", type=int, default=5)
    ap.add_argument("--nq-max", type=int, default=50)
    ap.add_argument("--ci-z", type=float, default=1.0)
    ap.add_argument("--holdout-counts", nargs="*", type=int, default=[1, 2],
                    help="Walk-forward holdout counts on observed nq for validation (default: 1 2).")

    ap.add_argument("--n-boot-hg", type=int, default=400)
    ap.add_argument("--n-boot-ml", type=int, default=400)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--boot-ci-level", type=float, default=0.90)
    ap.add_argument("--hg-bootstrap-mode", type=str, default="auto", choices=["auto", "per_k", "cluster", "hier_cluster"])

    ap.add_argument("--hypergraph-merged-dir", type=str, default=None)
    ap.add_argument("--shadow-surrogate-dir", type=str, default=None)
    ap.add_argument("--results-q5-10", type=str, default=None)
    ap.add_argument("--results-q11-12", type=str, default=None)

    ap.add_argument("--hg-rows-json", type=str, default=None)
    ap.add_argument("--eig-rows-json", type=str, default=None)
    ap.add_argument("--ml-rows-json", type=str, default=None)

    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument(
        "--mplstyle",
        type=str,
        default=None,
        help="Optional matplotlib style file (e.g. figures_notebooks/single_column.mplstyle).",
    )
    ap.add_argument(
        "--normalize-2pow-nq",
        action="store_true",
        help="Plot n_c / 2^{n_q} on the y-axis (baseline vs Hilbert-space dimension).",
    )
    ap.add_argument(
        "--extrap-line-only",
        action="store_true",
        help="Draw extrapolation as dashed line + CI band only (no triangle markers).",
    )
    ap.add_argument(
        "--plot-style",
        type=str,
        default="default",
        choices=["default", "fig3_mf"],
        help="fig3_mf: Blues/Greens/Reds ε_p colors, solid/dashed by A_target, markers-only data, model on full n_q grid.",
    )
    args = ap.parse_args()

    if not UNIFIED_PATH.exists():
        raise FileNotFoundError(UNIFIED_PATH)
    unified = _load_module("unified_fixed_acc_validation_mod", UNIFIED_PATH)
    hg_mod = _load_module("hg_raw_fixed_acc_validation_mod", HG_RAW_PATH)
    shadow_mod = _load_module("shadow_fixed_acc_validation_mod", SHADOW_PATH)

    targets = sorted(set(float(t) for t in args.targets))
    thresholds_fit = np.asarray(unified.THRESHOLDS_DENSE, dtype=float)
    if len(targets) < 1:
        raise ValueError("Need at least one target accuracy in --targets.")

    channels = [str(c) for c in (args.channels if args.channels else [args.channel])]
    channels = [c for c in channels if c in {"dephasing", "depolarizing", "relaxation"}]
    if not channels:
        raise ValueError("No valid channels selected.")
    amplitudes = sorted(set(float(v) for v in (args.ps if args.ps else [args.p])))

    rows_by_case: Dict[Tuple[str, str, float], List[dict]] = {}
    predictors: Dict[Tuple[str, str, float], object] = {}
    rows_json_map = {
        "hypergraph": args.hg_rows_json,
        "eigenshadow": args.eig_rows_json,
        "ml": args.ml_rows_json,
    }
    for method in METHOD_ORDER:
        preloaded_rows = None
        if rows_json_map[method]:
            preloaded_rows = unified._load_rows_json(Path(rows_json_map[method]))
            if method == "ml":
                for r in preloaded_rows:
                    r["channel"] = unified.ML_CHANNEL_CANONICAL.get(r.get("channel"), r.get("channel"))
                preloaded_rows = unified._postprocess_rows(preloaded_rows)

        for ch in channels:
            for amp in amplitudes:
                if preloaded_rows is not None:
                    rows = _filter_rows_case(preloaded_rows, channel=str(ch), amplitude=float(amp))
                else:
                    rows = _build_method_rows(
                        unified,
                        hg_mod,
                        shadow_mod,
                        method=method,
                        channel=str(ch),
                        amplitude=float(amp),
                        thresholds=thresholds_fit,
                        n_boot_hg=int(args.n_boot_hg),
                        n_boot_ml=int(args.n_boot_ml),
                        boot_seed=int(args.boot_seed),
                        boot_ci_level=float(args.boot_ci_level),
                        hg_bootstrap_mode=str(args.hg_bootstrap_mode),
                        hypergraph_merged_dir=Path(args.hypergraph_merged_dir).resolve() if args.hypergraph_merged_dir else None,
                        shadow_surrogate_dir=Path(args.shadow_surrogate_dir).resolve() if args.shadow_surrogate_dir else None,
                        results_q5_10=args.results_q5_10,
                        results_q11_12=args.results_q11_12,
                        rows_json=None,
                    )

                if not rows:
                    print(f"[warn] no rows for method={method}, channel={ch}, p={amp}")
                rows_by_case[(method, str(ch), float(amp))] = rows
                pred = _build_predictor_for_method(unified, rows, method)
                if pred is None:
                    print(f"[warn] predictor unavailable for method={method}, channel={ch}, p={amp}")
                predictors[(method, str(ch), float(amp))] = pred
                if pred is not None:
                    t_lo = float(pred.threshold_min)
                    t_hi = float(pred.threshold_max)
                    for t in targets:
                        if float(t) < t_lo or float(t) > t_hi:
                            print(
                                f"[warn] method={method}, channel={ch}, p={amp}: "
                                f"target={t:.3f} out of fitted threshold range [{t_lo:.3f}, {t_hi:.3f}]"
                            )

    if not any(v is not None for v in predictors.values()):
        raise RuntimeError("No predictors built. Check inputs.")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (SCRIPT_DIR / "plots_fixed_acc_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_tag = "_".join(str(t).replace(".", "p") for t in targets)
    p_tag = "_".join(str(p).replace(".", "p") for p in amplitudes)
    ch_tag = "_".join(channels)
    base = f"mf_fixedacc_validation_ch_{ch_tag}_p_{p_tag}_targets_{t_tag}_ci{args.ci_z:.3g}".replace(".", "p")
    out_png = out_dir / f"{base}.png"
    out_pdf = out_dir / f"{base}.pdf"

    _mpl = Path(args.mplstyle).resolve() if args.mplstyle else None
    if _mpl is not None and not _mpl.exists():
        raise FileNotFoundError(f"--mplstyle not found: {_mpl}")

    plot_validation(
        unified,
        predictors,
        rows_by_case,
        channels=channels,
        amplitudes=amplitudes,
        targets=targets,
        nq_min=int(args.nq_min),
        nq_max=int(args.nq_max),
        ci_z=float(args.ci_z),
        output_png=out_png,
        output_pdf=out_pdf,
        mplstyle_path=_mpl,
        normalize_y_by_2pow_nq=bool(args.normalize_2pow_nq),
        extrapolated_line_only=bool(args.extrap_line_only),
        plot_style=str(args.plot_style),
    )

    # Holdout validation + coverage diagnostics.
    holdout_counts = sorted(set(int(v) for v in args.holdout_counts if int(v) >= 1))
    backtest_points: List[dict] = []
    backtest_summary: List[dict] = []
    for method in METHOD_ORDER:
        for ch in channels:
            for amp in amplitudes:
                rows = rows_by_case.get((method, str(ch), float(amp)), [])
                if not rows:
                    continue
                pts, summ = _run_holdout_backtest(
                    unified,
                    method=method,
                    channel=str(ch),
                    amplitude=float(amp),
                    rows=rows,
                    targets=targets,
                    ci_z=float(args.ci_z),
                    holdout_counts=holdout_counts,
                    nq_min=int(args.nq_min),
                )
                backtest_points.extend(pts)
                backtest_summary.extend(summ)

    bt_points_csv = out_dir / f"{base}_holdout_points.csv"
    bt_summary_csv = out_dir / f"{base}_holdout_summary.csv"
    bt_summary_json = out_dir / f"{base}_holdout_summary.json"
    _write_csv(bt_points_csv, backtest_points)
    _write_csv(bt_summary_csv, backtest_summary)
    bt_summary_json.write_text(
        json.dumps(
            {
                "config": {
                    "channels": [str(c) for c in channels],
                    "p_values": [float(v) for v in amplitudes],
                    "targets": [float(t) for t in targets],
                    "ci_z": float(args.ci_z),
                    "holdout_counts": [int(v) for v in holdout_counts],
                },
                "rows_summary": backtest_summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[ok] saved: {out_png}")
    print(f"[ok] saved: {out_pdf}")
    print(f"[ok] saved: {bt_points_csv}")
    print(f"[ok] saved: {bt_summary_csv}")
    print(f"[ok] saved: {bt_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
