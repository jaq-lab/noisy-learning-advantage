#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
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


def _filter_rows(
    rows: List[dict],
    *,
    channel: str,
    amplitude: float,
    etas: List[float],
    devices: List[str],
    readout_error: str,
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
        if str(r.get("readout_error", "")) != str(readout_error):
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
    out_png: Path,
    out_pdf: Path,
    ci_band_label: str = r"1-$\sigma$ CI band",
) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_IMPORT_ERROR}")

    grouped = _filter_rows(
        rows,
        channel=channel,
        amplitude=amplitude,
        etas=etas,
        devices=devices,
        readout_error=readout_error,
    )
    if not grouped:
        raise RuntimeError("No rows matched the requested filter.")

    fig, ax = plt.subplots(1, 1, figsize=(14.2, 7.3))

    all_x, all_y = [], []
    all_nq_min = np.inf
    all_nq_max = -np.inf
    for method in METHOD_ORDER:
        for dev in devices:
            for eta in etas:
                key = (method, dev, float(eta))
                rs = grouped.get(key, [])
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

                base = DEVICE_BASE_COLORS.get(dev, "#444444")
                col = _shade_color(base, METHOD_INTENSITY.get(method, 0.85))
                m_ok = np.isfinite(x_all) & np.isfinite(y_all) & (y_all > 0) & (st_all == "ok")
                if not m_ok.any():
                    # still allow plotting random-baseline markers below
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
                    ax.fill_between(
                        x_ok[vb],
                        ylo_ok[vb],
                        yhi_ok[vb],
                        color=col,
                        alpha=0.14,
                        linewidth=0.0,
                        zorder=1,
                    )
                if x_ok.size:
                    ax.plot(
                        x_ok,
                        y_ok,
                        color=col,
                        linestyle=ETA_STYLES.get(float(eta), "-"),
                        linewidth=2.4,
                        marker=METHOD_MARKERS.get(method, "o"),
                        markersize=5.4,
                        markerfacecolor=col,
                        markeredgecolor="black",
                        markeredgewidth=0.45,
                        alpha=DEVICE_ALPHAS.get(dev, 0.9),
                        markevery=4 if method != "ml" else 2,
                    )

                # Draw a faint dotted continuation for ML when extrapolated region is gated as untrusted.
                if method == "ml" and np.isfinite(all_nq_max):
                    x_untrusted = x_all[np.isfinite(x_all) & (st_all == "untrusted_extrapolation")]
                    x_untrusted = np.sort(np.unique(x_untrusted))
                    if x_untrusted.size >= 1 and x_ok.size >= 2:
                        lx = np.asarray(x_ok, dtype=float)
                        ly = np.log2(np.asarray(y_ok, dtype=float))
                        # Robust short extrapolation baseline from trusted segment.
                        m, b = np.polyfit(lx, ly, 1)
                        y_tail = np.power(2.0, m * x_untrusted + b)
                        ax.plot(
                            x_untrusted,
                            y_tail,
                            color=col,
                            linestyle=":",
                            linewidth=1.8,
                            alpha=0.38,
                            zorder=2,
                        )

                # Explicit random-baseline markers (nps=1 convention).
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
                        # Connect baseline points for trusted-series context (skip ML: untrusted tail already shown).
                        ax.plot(
                            x_rb,
                            y_rb,
                            color=col,
                            linestyle=ETA_STYLES.get(float(eta), "-"),
                            linewidth=1.45,
                            alpha=0.55,
                            zorder=4,
                        )
                        # Bridge from last trusted point to first baseline point when separated by an untrusted gap.
                        if x_ok.size and x_rb.size:
                            j_last = int(np.argmax(x_ok))
                            x_last = float(x_ok[j_last])
                            y_last = float(y_ok[j_last])
                            if x_rb[0] > x_last and y_last > 0:
                                ax.plot(
                                    [x_last, float(x_rb[0])],
                                    [y_last, float(y_rb[0])],
                                    color=col,
                                    linestyle=":",
                                    linewidth=1.2,
                                    alpha=0.42,
                                    zorder=3,
                                )
                    ax.scatter(
                        x_rb,
                        y_rb,
                        color=col,
                        marker="x",
                        s=34,
                        linewidths=1.2,
                        alpha=0.92,
                        zorder=5,
                    )

    ax.set_yscale("log")
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

    # Time reference lines in sample-complexity units.
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
            [0], [0],
            color=_shade_color("#555555", METHOD_INTENSITY[m]),
            linewidth=2.8,
            marker=METHOD_MARKERS[m],
            markersize=6,
            label=METHOD_LABELS[m],
        )
        for m in METHOD_ORDER
    ]

    leg1 = ax.legend(handles=dev_handles, loc="upper left", bbox_to_anchor=(1.01, 1.00), frameon=False, title="Device Color", fontsize=11, title_fontsize=11)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=eta_handles, loc="upper left", bbox_to_anchor=(1.01, 0.73), frameon=False, title="Eta Line", fontsize=11, title_fontsize=11)
    ax.add_artist(leg2)
    sem_handles = [Line2D([0], [0], color="gray", linewidth=6, alpha=0.14, label=str(ci_band_label))]
    sem_handles.append(Line2D([0], [0], color="gray", linestyle=":", linewidth=1.8, alpha=0.45, label="ML untrusted tail"))
    sem_handles.append(Line2D([0], [0], color="gray", marker="x", linestyle="", markersize=7, label="random baseline"))
    leg3 = ax.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(1.01, 0.47), frameon=False, title="MF Protocol", fontsize=11, title_fontsize=11)
    ax.add_artist(leg3)
    ax.legend(handles=sem_handles, loc="upper left", bbox_to_anchor=(1.01, 0.23), frameon=False, title="Semantics", fontsize=11, title_fontsize=11)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="One-panel 12-line plot for thermal p=0.1 (MF protocols, eta, devices).")
    ap.add_argument("--csv", type=str, required=True, help="Unified CSV path (output from plot_quantum_vs_classical_nps_unified.py).")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--channel", type=str, default="relaxation")
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--etas", nargs="*", type=float, default=[0.01, 0.05])
    ap.add_argument("--devices", nargs="*", default=["A", "B"])
    ap.add_argument("--readout-error", type=str, default="0%")
    ap.add_argument(
        "--ci-band-label",
        type=str,
        default=r"1-$\sigma$ CI band",
        help="Legend label for uncertainty band (e.g., '1-$\\sigma$ CI band', '95% CI band (~2$\\sigma$)').",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = _load_rows(csv_path)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (csv_path.parent / "onepanel")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"mf_onepanel_ch_{args.channel}_p_{str(args.p).replace('.', 'p')}"
        f"_eta_{'_'.join(str(e).replace('.', 'p') for e in args.etas)}"
        f"_devices_{''.join(args.devices)}_re_{str(args.readout_error).replace('%', 'pct')}"
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
        out_png=out_png,
        out_pdf=out_pdf,
        ci_band_label=str(args.ci_band_label),
    )
    print(f"[ok] saved: {out_png}")
    print(f"[ok] saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
