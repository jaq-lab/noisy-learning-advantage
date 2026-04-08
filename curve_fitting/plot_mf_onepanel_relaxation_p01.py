#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
except Exception as e:  # pragma: no cover
    plt = None
    MPL_IMPORT_ERROR = e


METHOD_ORDER = ["hypergraph", "eigenshadow", "ml"]
METHOD_LABELS = {"hypergraph": "Hypergraph", "eigenshadow": "Shadow-based Eigenshadow", "ml": "Shadow-based ML"}
METHOD_INTENSITY = {"hypergraph": 0.50, "eigenshadow": 0.72, "ml": 0.96}
METHOD_MARKERS = {"hypergraph": "o", "eigenshadow": "s", "ml": "^"}

ETA_STYLES = {0.01: "-", 0.05: "--"}
ETA_LABELS = {0.01: r"$\eta=1\%$", 0.05: r"$\eta=5\%$"}

DEVICE_MAP = {"I": "A", "T": "B", "A": "A", "B": "B"}
DEVICE_BASE_COLORS = {"A": "#2a6fba", "B": "#c93b3b"}
DEVICE_ALPHAS = {"A": 0.95, "B": 0.98}
TIME_REF_LINES = [
    (8.64e10, "1 d"),
    (3.154e13, "1 y"),
    (3.154e16, "1000 y"),
]

METHOD_LS_FIG1_MS = {"hypergraph": "-.", "eigenshadow": "-", "ml": "--"}
METHOD_LABELS_FIG1_MS = {"hypergraph": "Hypergraph", "eigenshadow": "Eigenshadow", "ml": "ML"}
ETA_STYLES_FIG1_MS = {0.05: "-", 0.01: "--"}

FIG1_MS_DEVICE_DEFAULT = {"A": "#C62828", "B": "#1565C0"}
FIG1_MS_LW_MAIN = 1.8
FIG1_MS_LEG_LW = 1.0
FIG1_MS_LEG_LW_ALPHA = 2.0
FIG1_MS_XLIM = (10.0, 52.0)
FIG1_MS_YMIN = 1e3
FIG1_MS_ALPHA_HUE_BY_ETA_KEY = {0.01: "#2CA02C", 0.05: "#D62728"}
FIG1_MS_ALPHA_MODE_COLORS = {
    "nq": FIG1_MS_ALPHA_HUE_BY_ETA_KEY[0.05],
    "nq2": FIG1_MS_ALPHA_HUE_BY_ETA_KEY[0.01],
}
FIG1_MS_ETA_GREY_DARK = "#4a4a4a"
FIG1_MS_ETA_GREY_LIGHT_BLEND = "#f5f5f5"
FIG1_MS_ETA_GREY_LIGHT_LEGEND = "#c0c0c0"
FIG1_MS_ALPHA_ETA_GREY_BLEND = 0.22
FIG1_MS_ETA01_LINE_TOWARD_WHITE = 0.52
FIG1_MS_ALPHA_MODE_LABELS = {
    "nq": r"$|\alpha| = n_q$",
    "nq2": r"$|\alpha| = n_q/2$",
}
FIG1_MS_DAY_NC = 9e10
FIG1_MS_YEAR_NC = FIG1_MS_DAY_NC * 365.25
FIG1_MS_HREF = [
    (FIG1_MS_YEAR_NC * 1000.0, "1000 y"),
    (FIG1_MS_YEAR_NC, "1 y"),
    (FIG1_MS_DAY_NC, "1 d"),
]
FIG1_MS_YTICKS_ZOOM = [1e5, 1e10, 1e15, 1e20, 1e25]
FIG1_MS_LEGEND_FONTSIZE = 7
FIG1_MS_LEGEND_TITLE_FONTSIZE = 7


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


def _load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _shade_color(base_color: str, intensity: float) -> tuple:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
    # intensity=1 -> original color, intensity=0 -> white
    out = 1.0 - intensity * (1.0 - rgb)
    return tuple(np.clip(out, 0.0, 1.0))


def _fig1_device_eta_color(hex_color: str, eta: float) -> str:
    eta = float(eta)
    if abs(eta - 0.05) < 1e-5:
        g_hex = FIG1_MS_ETA_GREY_DARK
    elif abs(eta - 0.01) < 1e-5:
        g_hex = FIG1_MS_ETA_GREY_LIGHT_BLEND
    else:
        g_hex = "#888888"
    g_rgb = np.array(mcolors.to_rgb(g_hex), dtype=float)
    b_rgb = np.array(mcolors.to_rgb(hex_color), dtype=float)
    w = float(np.clip(FIG1_MS_ALPHA_ETA_GREY_BLEND, 0.0, 1.0))
    out = w * g_rgb + (1.0 - w) * b_rgb
    return mcolors.to_hex(tuple(np.clip(out, 0.0, 1.0)))


def _fig1_lighten_line_color(hex_color: str, toward_white: float) -> str:
    t = float(np.clip(toward_white, 0.0, 1.0))
    rgb = np.array(mcolors.to_rgb(hex_color), dtype=float)
    out = (1.0 - t) * rgb + t
    return mcolors.to_hex(tuple(np.clip(out, 0.0, 1.0)))


def _auto_ci_band_label(ci_z: float) -> str:
    z = _to_float(ci_z, default=np.nan)
    if not _is_finite(z) or z <= 0:
        return "CI band"
    if abs(z - 1.0) < 1e-9:
        return r"1-$\sigma$ CI band"
    if abs(z - 2.0) < 1e-9 or abs(z - 1.95996398454) < 0.03:
        return r"2-$\sigma$ CI band"
    return f"{z:.3g}-$\\sigma$ CI band"


def _parse_readout_by_device(items: List[str] | None) -> Dict[str, str]:
    """
    Parse CLI tokens of form DEVICE=READOUT (e.g., A=0.1% B=1%).
    Device aliases I/T/A/B are normalized through DEVICE_MAP.
    """
    out: Dict[str, str] = {}
    if not items:
        return out
    for tok in items:
        s = str(tok).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Invalid --readout-error-by-device token '{s}'. Expected DEVICE=READOUT.")
        k, v = s.split("=", 1)
        dev_in = str(k).strip()
        dev = DEVICE_MAP.get(dev_in, dev_in)
        ro = str(v).strip()
        if not dev or not ro:
            raise ValueError(f"Invalid --readout-error-by-device token '{s}'.")
        out[dev] = ro
    return out


def _filter_rows(
    rows: List[dict],
    *,
    channel: str,
    amplitude: float,
    etas: List[float],
    devices: List[str],
    readout_error: str,
    readout_by_device: Dict[str, str],
) -> Dict[Tuple[str, str, float], List[dict]]:
    out: Dict[Tuple[str, str, float], List[dict]] = {}
    for r in rows:
        method = str(r.get("method", ""))
        if method not in METHOD_ORDER:
            continue
        ch = str(r.get("channel", ""))
        if ch != channel:
            continue
        amp = _to_float(r.get("amplitude"))
        if abs(amp - float(amplitude)) > 1e-12:
            continue
        eta = _to_float(r.get("eta"))
        if min(abs(eta - e) for e in etas) > 1e-12:
            continue
        dev_in = str(r.get("device", ""))
        dev = DEVICE_MAP.get(dev_in, dev_in)
        if dev not in devices:
            continue
        ro_expected = str(readout_by_device.get(dev, readout_error))
        if str(r.get("readout_error", "")) != ro_expected:
            continue
        nq = _to_float(r.get("nq", r.get("n_q")))
        if not _is_finite(nq):
            continue
        key = (method, dev, float(eta))
        out.setdefault(key, []).append(r)
    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda rr: _to_float(rr.get("nq", rr.get("n_q"))))
    return out


def plot_onepanel(
    rows: List[dict],
    *,
    channel: str,
    amplitude: float,
    etas: List[float],
    devices: List[str],
    readout_error: str,
    readout_by_device: Dict[str, str],
    out_png: Optional[Path] = None,
    out_pdf: Optional[Path] = None,
    ci_band_label: str = r"1-$\sigma$ CI band",
    manuscript_fig1: bool = False,
    manuscript_dual_alpha_rows: Optional[Dict[str, List[dict]]] = None,
    figsize: Tuple[float, float] = (14.2, 7.3),
    close_fig: bool = True,
    device_base_colors: Optional[Dict[str, str]] = None,
):
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_IMPORT_ERROR}")
    if manuscript_dual_alpha_rows is not None and not manuscript_fig1:
        raise ValueError("manuscript_dual_alpha_rows requires manuscript_fig1=True")

    if manuscript_fig1 and manuscript_dual_alpha_rows:
        parts: List[Tuple[Optional[str], Dict[Tuple[str, str, float], List[dict]]]] = []
        for k in ("nq2", "nq"):
            if k not in manuscript_dual_alpha_rows:
                raise KeyError(f"manuscript_dual_alpha_rows missing key {k!r}")
            g = _filter_rows(
                manuscript_dual_alpha_rows[k],
                channel=channel,
                amplitude=amplitude,
                etas=etas,
                devices=devices,
                readout_error=readout_error,
                readout_by_device=readout_by_device,
            )
            if not g:
                raise RuntimeError(f"No rows matched filters for dual-|α| key {k!r}.")
            parts.append((k, g))
    else:
        grouped = _filter_rows(
            rows,
            channel=channel,
            amplitude=amplitude,
            etas=etas,
            devices=devices,
            readout_error=readout_error,
            readout_by_device=readout_by_device,
        )
        if not grouped:
            raise RuntimeError("No rows matched the requested filter.")
        parts = [(None, grouped)]

    if manuscript_fig1:
        figsize = figsize if figsize != (14.2, 7.3) else (3.5, 3.0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    dev_colors = {**FIG1_MS_DEVICE_DEFAULT, **(device_base_colors or {})} if manuscript_fig1 else DEVICE_BASE_COLORS
    eta_plot_order = sorted(etas)
    eta_ls_map = ETA_STYLES_FIG1_MS if manuscript_fig1 else ETA_STYLES

    all_x: List[float] = []
    all_y: List[float] = []
    all_nq_min = np.inf
    all_nq_max = -np.inf

    for alpha_key, grouped_cur in parts:
        for method in METHOD_ORDER:
            for dev in devices:
                for eta in eta_plot_order:
                    key = (method, dev, float(eta))
                    rs = grouped_cur.get(key, [])
                    if not rs:
                        continue
                    x_all = np.array([_to_float(r.get("nq", r.get("n_q"))) for r in rs], dtype=float)
                    y_all = np.array([_to_float(r.get("nps")) for r in rs], dtype=float)
                    ylo_all = np.array([_to_float(r.get("nps_lo")) for r in rs], dtype=float)
                    yhi_all = np.array([_to_float(r.get("nps_hi")) for r in rs], dtype=float)
                    st_all = np.array([str(r.get("pred_status", "")) for r in rs], dtype=object)

                    m_finite = np.isfinite(x_all)
                    if not m_finite.any():
                        continue
                    all_nq_min = min(all_nq_min, float(np.nanmin(x_all[m_finite])))
                    all_nq_max = max(all_nq_max, float(np.nanmax(x_all[m_finite])))

                    base = dev_colors.get(dev, "#444444")
                    if manuscript_fig1:
                        if alpha_key is not None:
                            base = FIG1_MS_ALPHA_MODE_COLORS.get(alpha_key, base)
                        col = _fig1_device_eta_color(base, float(eta))
                    else:
                        col = _shade_color(base, METHOD_INTENSITY.get(method, 0.85))

                    _is_fig1_eta01 = manuscript_fig1 and abs(float(eta) - 0.01) < 1e-5
                    col_line = (
                        _fig1_lighten_line_color(col, FIG1_MS_ETA01_LINE_TOWARD_WHITE)
                        if _is_fig1_eta01
                        else col
                    )
                    _dev_a = DEVICE_ALPHAS.get(dev, 0.9)
                    _curve_alpha = _dev_a
                    _fill_alpha = 0.14
                    _tail_alpha = 0.38

                    m_ok = np.isfinite(x_all) & np.isfinite(y_all) & (y_all > 0) & (st_all == "ok")
                    if not m_ok.any():
                        x_ok = np.array([], dtype=float)
                        y_ok = np.array([], dtype=float)
                        ylo_ok = np.array([], dtype=float)
                        yhi_ok = np.array([], dtype=float)
                    else:
                        x_ok = x_all[m_ok]
                        y_ok = y_all[m_ok]
                        ylo_ok = ylo_all[m_ok]
                        yhi_ok = yhi_all[m_ok]
                        all_x.extend(list(x_ok))
                        all_y.extend(list(y_ok))

                    vb = np.isfinite(ylo_ok) & np.isfinite(yhi_ok) & (ylo_ok > 0) & (yhi_ok > 0)
                    if np.any(vb):
                        fill_c = mcolors.to_hex(col) if isinstance(col, tuple) else col
                        ax.fill_between(
                            x_ok[vb],
                            ylo_ok[vb],
                            yhi_ok[vb],
                            color=fill_c,
                            alpha=_fill_alpha,
                            linewidth=0.0,
                            zorder=1,
                        )
                    if x_ok.size:
                        if manuscript_fig1:
                            _z_main = 5 if abs(float(eta) - 0.05) < 1e-5 else 4
                            ax.plot(
                                x_ok,
                                y_ok,
                                color=col_line,
                                linestyle=METHOD_LS_FIG1_MS.get(method, "-"),
                                linewidth=FIG1_MS_LW_MAIN,
                                alpha=_curve_alpha,
                                zorder=_z_main,
                            )
                        else:
                            ax.plot(
                                x_ok,
                                y_ok,
                                color=col,
                                linestyle=eta_ls_map.get(float(eta), "-"),
                                linewidth=2.4,
                                marker=METHOD_MARKERS.get(method, "o"),
                                markersize=5.4,
                                markerfacecolor=col,
                                markeredgecolor="black",
                                markeredgewidth=0.45,
                                alpha=_curve_alpha,
                                markevery=4 if method != "ml" else 2,
                            )

                    if method == "ml" and np.isfinite(all_nq_max):
                        x_untrusted = x_all[np.isfinite(x_all) & (st_all == "untrusted_extrapolation")]
                        x_untrusted = np.sort(np.unique(x_untrusted))
                        if x_untrusted.size >= 1 and x_ok.size >= 2:
                            lx = np.asarray(x_ok, dtype=float)
                            ly = np.log2(np.asarray(y_ok, dtype=float))
                            m, b = np.polyfit(lx, ly, 1)
                            y_tail = np.power(2.0, m * x_untrusted + b)
                            ax.plot(
                                x_untrusted,
                                y_tail,
                                color=col_line if manuscript_fig1 else col,
                                linestyle=METHOD_LS_FIG1_MS.get(method, "-") if manuscript_fig1 else ":",
                                linewidth=FIG1_MS_LW_MAIN if manuscript_fig1 else 1.8,
                                alpha=_tail_alpha,
                                zorder=2,
                            )

                    rb = np.isfinite(x_all) & (st_all == "random_baseline")
                    if np.any(rb):
                        y_rb = y_all[rb].copy()
                        y_rb[~(np.isfinite(y_rb) & (y_rb > 0))] = 1.0
                        x_rb = x_all[rb].copy()
                        o_rb = np.argsort(x_rb)
                        x_rb = x_rb[o_rb]
                        y_rb = y_rb[o_rb]
                        all_y.extend(list(y_rb[np.isfinite(y_rb) & (y_rb > 0)]))

                        if method != "ml":
                            _rb_ls = (
                                METHOD_LS_FIG1_MS.get(method, "-")
                                if manuscript_fig1
                                else eta_ls_map.get(float(eta), "-")
                            )
                            _rb_lw = FIG1_MS_LW_MAIN if manuscript_fig1 else 2.4
                            _rb_alpha = _curve_alpha if manuscript_fig1 else DEVICE_ALPHAS.get(dev, 0.9)
                            ax.plot(
                                x_rb,
                                y_rb,
                                color=col_line if manuscript_fig1 else col,
                                linestyle=_rb_ls,
                                linewidth=_rb_lw,
                                alpha=_rb_alpha,
                                zorder=4,
                            )
                            if x_ok.size and x_rb.size:
                                j_last = int(np.argmax(x_ok))
                                x_last = float(x_ok[j_last])
                                y_last = float(y_ok[j_last])
                                if x_rb[0] > x_last and y_last > 0:
                                    ax.plot(
                                        [x_last, float(x_rb[0])],
                                        [y_last, float(y_rb[0])],
                                        color=col_line if manuscript_fig1 else col,
                                        linestyle=_rb_ls,
                                        linewidth=_rb_lw,
                                        alpha=_rb_alpha,
                                        zorder=3,
                                    )
                        ax.scatter(
                            x_rb,
                            y_rb,
                            color=col_line if manuscript_fig1 else col,
                            marker="x",
                            s=22 if manuscript_fig1 else 34,
                            linewidths=1.0 if manuscript_fig1 else 1.2,
                            alpha=_curve_alpha if manuscript_fig1 else 0.92,
                            zorder=5,
                        )

    ax.set_yscale("log")
    if manuscript_fig1:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*FIG1_MS_XLIM)
        if all_y:
            ymax = max(max(all_y) * 1.35, FIG1_MS_YMIN * 10.0)
            ymax = max(ymax, max(y for y, _ in FIG1_MS_HREF) * 1.05)
            ax.set_ylim(FIG1_MS_YMIN, ymax)
        ax.set_yticks(FIG1_MS_YTICKS_ZOOM)
        ax.set_xlabel("number of qubits $n_q$")
        ax.set_ylabel(r"Copies to match FQ $n_c(A_Q-\eta)$")
        x_href = FIG1_MS_XLIM[1] * 0.96
        for y_val, label in FIG1_MS_HREF:
            ax.axhline(y=y_val, color="black", linestyle=":", linewidth=0.8, alpha=0.85, zorder=0)
            ax.text(
                x_href,
                y_val/4,
                " " + label,
                transform=ax.transData,
                fontsize=7,
                color="black",
                alpha=0.9,
                va="center",
                ha="center",
            )
        leg_methods = ax.legend(
            [
                Line2D(
                    [0],
                    [0],
                    color="k",
                    linestyle=METHOD_LS_FIG1_MS[m],
                    linewidth=FIG1_MS_LEG_LW,
                )
                for m in METHOD_ORDER
            ],
            [METHOD_LABELS_FIG1_MS[m] for m in METHOD_ORDER],
            title="MF Method",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=1,
            frameon=True,
            fontsize=FIG1_MS_LEGEND_FONTSIZE,
            title_fontsize=FIG1_MS_LEGEND_TITLE_FONTSIZE,
        )
        leg_eta = ax.legend(
            [
                Line2D(
                    [0],
                    [0],
                    color=FIG1_MS_ETA_GREY_DARK,
                    linestyle="-",
                    linewidth=FIG1_MS_LEG_LW,
                ),
                Line2D(
                    [0],
                    [0],
                    color=FIG1_MS_ETA_GREY_LIGHT_LEGEND,
                    linestyle="-",
                    linewidth=FIG1_MS_LEG_LW,
                ),
            ],
            ["η = 5%", "η = 1%"],
            loc="center left",
            bbox_to_anchor=(0.02, 0.5),
            frameon=True,
            fontsize=FIG1_MS_LEGEND_FONTSIZE,
        )
        ax.add_artist(leg_methods)
        ax.add_artist(leg_eta)
        if manuscript_dual_alpha_rows:
            leg_alpha = ax.legend(
                [
                    Line2D(
                        [0],
                        [0],
                        color=FIG1_MS_ALPHA_MODE_COLORS["nq"],
                        linestyle="-",
                        linewidth=FIG1_MS_LEG_LW_ALPHA,
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=FIG1_MS_ALPHA_MODE_COLORS["nq2"],
                        linestyle="-",
                        linewidth=FIG1_MS_LEG_LW_ALPHA,
                    ),
                ],
                [
                    FIG1_MS_ALPHA_MODE_LABELS["nq"],
                    FIG1_MS_ALPHA_MODE_LABELS["nq2"],
                ],
                loc="lower right",
                bbox_to_anchor=(0.98, 0.02),
                ncol=1,
                frameon=True,
                fontsize=FIG1_MS_LEGEND_FONTSIZE,
            )
            ax.add_artist(leg_alpha)
        elif len(devices) > 1:
            leg_devices = ax.legend(
                [
                    Line2D([0], [0], color=dev_colors.get(d, "#444444"), linestyle="-", linewidth=2.5)
                    for d in devices
                ],
                list(devices),
                title="Device",
                loc="lower right",
                bbox_to_anchor=(0.98, 0.02),
                ncol=1,
                frameon=True,
                fontsize=FIG1_MS_LEGEND_FONTSIZE,
                title_fontsize=FIG1_MS_LEGEND_TITLE_FONTSIZE,
            )
            ax.add_artist(leg_devices)
        fig.tight_layout()
    else:
        ax.grid(alpha=0.24)
        ax.set_xlabel(r"Number of qubits $n_q$", fontsize=14)
        ax.set_ylabel(r"Sample complexity $n_c$ to match accuracy $A_Q - \eta$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.axvline(15, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(15.25, 0.94, "obs→extrap", transform=ax.get_xaxis_transform(), fontsize=10, color="gray")

        if np.isfinite(all_nq_min) and np.isfinite(all_nq_max):
            ax.set_xlim(all_nq_min - 0.4, all_nq_max + 0.4)
        if all_y:
            ymin = min(max(min(all_y) / 1.8, 0.8), 1.0)
            ymax = max(all_y) * 1.9
            ymax = max(ymax, max(v for v, _ in TIME_REF_LINES) * 1.12)
            ax.set_ylim(ymin, ymax)

        for yref, label in TIME_REF_LINES:
            ax.axhline(y=yref, color="gray", linestyle=":", linewidth=1.0, alpha=0.62, zorder=0)
            ax.text(
                0.995,
                yref,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=9.5,
                color="gray",
            )

        dev_handles = [
            Line2D([0], [0], color=DEVICE_BASE_COLORS[d], linewidth=3.0, label=f"Device {d}")
            for d in devices
        ]
        eta_handles = [
            Line2D([0], [0], color="black", linewidth=2.4, linestyle=ETA_STYLES[e], label=ETA_LABELS[e])
            for e in etas
        ]
        method_handles = [
            Line2D(
                [0],
                [0],
                color=_shade_color("#555555", METHOD_INTENSITY[m]),
                linewidth=2.8,
                marker=METHOD_MARKERS[m],
                markersize=6,
                label=METHOD_LABELS[m],
            )
            for m in METHOD_ORDER
        ]

        leg1 = ax.legend(
            handles=dev_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            frameon=False,
            title="Device Color",
            fontsize=11,
            title_fontsize=11,
        )
        ax.add_artist(leg1)
        leg2 = ax.legend(
            handles=eta_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.73),
            frameon=False,
            title="Eta Line",
            fontsize=11,
            title_fontsize=11,
        )
        ax.add_artist(leg2)
        sem_handles = [Line2D([0], [0], color="gray", linewidth=6, alpha=0.14, label=str(ci_band_label))]
        sem_handles.append(
            Line2D([0], [0], color="gray", linestyle=":", linewidth=1.8, alpha=0.45, label="ML untrusted tail")
        )
        sem_handles.append(
            Line2D([0], [0], color="gray", marker="x", linestyle="", markersize=7, label="random baseline")
        )
        leg3 = ax.legend(
            handles=method_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.47),
            frameon=False,
            title="MF Protocol",
            fontsize=11,
            title_fontsize=11,
        )
        ax.add_artist(leg3)
        ax.legend(
            handles=sem_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.23),
            frameon=False,
            title="Semantics",
            fontsize=11,
            title_fontsize=11,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

    if out_png is not None or out_pdf is not None:
        parent = (out_png or out_pdf).parent
        parent.mkdir(parents=True, exist_ok=True)
    if out_png is not None:
        fig.savefig(out_png, dpi=260, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, dpi=260, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig, ax


def main() -> int:
    ap = argparse.ArgumentParser(description="One-panel 12-line plot for thermal p=0.1 (MF protocols, eta, devices).")
    ap.add_argument("--csv", type=str, required=True, help="Unified CSV path (output from plot_quantum_vs_classical_nps_unified.py).")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--channel", type=str, default="relaxation")
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--etas", nargs="*", type=float, default=[0.01, 0.05])
    ap.add_argument("--devices", nargs="*", default=["A", "B"])
    ap.add_argument("--readout-error", type=str, default="0%",
                    help="Fallback readout filter when --readout-error-by-device is not provided for a device.")
    ap.add_argument("--readout-error-by-device", nargs="*", default=None,
                    help="Per-device readout filters, e.g.: A=0.1% B=1%")
    ap.add_argument("--ci-z", type=float, default=1.0,
                    help="z used for CI semantics label (e.g., 1.0->1-sigma, 1.96->~2-sigma).")
    ap.add_argument(
        "--ci-band-label",
        type=str,
        default=None,
        help="Optional legend label override. If omitted, label is derived from --ci-z.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = _load_rows(csv_path)
    readout_by_device = _parse_readout_by_device(args.readout_error_by_device)
    ci_band_label = str(args.ci_band_label) if args.ci_band_label else _auto_ci_band_label(float(args.ci_z))

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (csv_path.parent / "onepanel")
    out_dir.mkdir(parents=True, exist_ok=True)
    if readout_by_device:
        ro_tag = "_".join(
            f"{d}{str(readout_by_device[d]).replace('%', 'pct').replace('.', 'p')}"
            for d in sorted(readout_by_device)
        )
    else:
        ro_tag = str(args.readout_error).replace('%', 'pct')
    base = (
        f"mf_onepanel_ch_{args.channel}_p_{str(args.p).replace('.', 'p')}"
        f"_eta_{'_'.join(str(e).replace('.', 'p') for e in args.etas)}"
        f"_devices_{''.join(args.devices)}_re_{ro_tag}"
    )
    out_png = out_dir / f"{base}.png"
    out_pdf = out_dir / f"{base}.pdf"

    plot_onepanel(
        rows,
        channel=str(args.channel),
        amplitude=float(args.p),
        etas=[float(x) for x in args.etas],
        devices=[str(d) for d in args.devices],
        readout_error=str(args.readout_error),
        readout_by_device=readout_by_device,
        out_png=out_png,
        out_pdf=out_pdf,
        ci_band_label=ci_band_label,
    )
    print(f"[ok] saved: {out_png}")
    print(f"[ok] saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
