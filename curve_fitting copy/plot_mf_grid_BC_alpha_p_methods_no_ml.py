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
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as e:  # pragma: no cover
    plt = None
    MPL_IMPORT_ERROR = e


CHANNEL_ORDER = ["dephasing", "depolarizing", "relaxation"]
CHANNEL_TITLES = {"dephasing": "Dephasing", "depolarizing": "Depolarizing", "relaxation": "Relaxation"}
DEVICE_ORDER = ["B", "C"]
DEVICE_MAP = {"I": "A", "T": "B", "S": "C", "A": "A", "B": "B", "C": "C"}
METHOD_ORDER = ["hypergraph", "eigenshadow"]
METHOD_LABELS = {"hypergraph": "Hypergraph", "eigenshadow": "Shadow-based Eigenshadow"}
METHOD_LINESTYLES = {"hypergraph": "-", "eigenshadow": "--"}

ALPHA_ORDER = ["nq", "nq2"]
ALPHA_LABELS = {"nq": r"$|\alpha|=n_q$", "nq2": r"$|\alpha|=n_q/2$"}
ALPHA_BASE_COLORS = {"nq": "#1f77b4", "nq2": "#d62728"}
P_DEFAULT = [0.01, 0.1]
P_INTENSITY = {0.01: 0.62, 0.1: 1.00}
P_LABELS = {0.01: r"$\epsilon_p=0.01$", 0.1: r"$\epsilon_p=0.1$"}
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


def _shade_color(base: str, intensity: float) -> tuple:
    rgb = np.array(mcolors.to_rgb(base), dtype=float)
    s = float(np.clip(intensity, 0.0, 1.0))
    out = 1.0 - s * (1.0 - rgb)
    return tuple(np.clip(out, 0.0, 1.0))


def _auto_ci_band_label(ci_z: float) -> str:
    z = _to_float(ci_z, default=np.nan)
    if not _is_finite(z) or z <= 0:
        return "CI band"
    if abs(z - 1.0) < 1e-9:
        return r"1-$\sigma$ CI band"
    if abs(z - 2.0) < 1e-9 or abs(z - 1.95996398454) < 0.03:
        return r"2-$\sigma$ CI band"
    return f"{z:.3g}-$\\sigma$ CI band"


def _anchor_y_at_onset(xq: float, x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if not np.any(m):
        return float("nan")
    xf = np.asarray(x[m], dtype=float)
    yf = np.asarray(y[m], dtype=float)
    # If exact x exists, use it directly.
    exact = np.where(np.isclose(xf, float(xq), rtol=0.0, atol=1e-12))[0]
    if exact.size:
        return float(yf[int(exact[0])])
    # Otherwise anchor from the closest previous point (or next if needed).
    left = np.where(xf < float(xq))[0]
    if left.size:
        return float(yf[int(left[-1])])
    right = np.where(xf > float(xq))[0]
    if right.size:
        return float(yf[int(right[0])])
    return float(yf[0])


def _read_rows(csv_path: Path, alpha_mode: str) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            rr["alpha_mode"] = alpha_mode
            rows.append(rr)
    return rows


def _parse_readout_by_device(items: List[str] | None) -> Dict[str, str]:
    """
    Parse CLI tokens of form DEVICE=READOUT (e.g., B=0.1% C=1%).
    Device aliases I/T/S/A/B/C are normalized through DEVICE_MAP.
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


def _collect_series(
    rows: List[dict],
    *,
    eta: float,
    readout_error: str,
    readout_by_device: Dict[str, str],
    channels: List[str],
    devices: List[str],
    p_values: List[float],
) -> Dict[Tuple[str, str, str, float, str], List[dict]]:
    out: Dict[Tuple[str, str, str, float, str], List[dict]] = {}
    for r in rows:
        alpha_mode = str(r.get("alpha_mode", ""))
        if alpha_mode not in ALPHA_ORDER:
            continue
        ch = str(r.get("channel", "")).strip().lower()
        if ch not in channels:
            continue
        dev = DEVICE_MAP.get(str(r.get("device", "")).strip(), str(r.get("device", "")).strip())
        if dev not in devices:
            continue
        ro_expected = str(readout_by_device.get(dev, readout_error))
        if str(r.get("readout_error", "")) != ro_expected:
            continue
        if abs(_to_float(r.get("eta")) - float(eta)) > 1e-12:
            continue
        method = str(r.get("method", ""))
        if method not in METHOD_ORDER:
            continue
        p = _to_float(r.get("amplitude"))
        if min(abs(p - float(pp)) for pp in p_values) > 1e-12:
            continue
        nq = _to_float(r.get("nq", r.get("n_q")))
        if not _is_finite(nq):
            continue
        key = (dev, ch, alpha_mode, float(p), method)
        out.setdefault(key, []).append(r)

    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda rr: _to_float(rr.get("nq", rr.get("n_q"))))
    return out


def plot_grid(
    series: Dict[Tuple[str, str, str, float, str], List[dict]],
    *,
    channels: List[str],
    devices: List[str],
    p_values: List[float],
    eta: float,
    out_png: Path,
    out_pdf: Path,
    ci_band_label: str = r"1-$\sigma$ CI band",
) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_IMPORT_ERROR}")

    nr = len(channels)
    nc = len(devices)
    fig, axes = plt.subplots(nr, nc, figsize=(15.4, 11.8), sharex=True, sharey=True)
    if nr == 1:
        axes = np.array([axes])
    if nc == 1:
        axes = axes.reshape(nr, 1)

    all_x: List[float] = []
    all_y: List[float] = []

    for i, ch in enumerate(channels):
        for j, dev in enumerate(devices):
            ax = axes[i, j]
            for alpha_mode in ALPHA_ORDER:
                for p in p_values:
                    for method in METHOD_ORDER:
                        rs = series.get((dev, ch, alpha_mode, float(p), method), [])
                        if not rs:
                            continue
                        x_all = np.array([_to_float(r.get("nq", r.get("n_q"))) for r in rs], dtype=float)
                        y_all = np.array([_to_float(r.get("nps")) for r in rs], dtype=float)
                        ylo_all = np.array([_to_float(r.get("nps_lo")) for r in rs], dtype=float)
                        yhi_all = np.array([_to_float(r.get("nps_hi")) for r in rs], dtype=float)
                        st_all = np.array([str(r.get("pred_status", "")) for r in rs], dtype=object)
                        mf = np.isfinite(x_all)
                        if np.any(mf):
                            all_x.extend(list(x_all[mf]))

                        color = _shade_color(ALPHA_BASE_COLORS[alpha_mode], P_INTENSITY.get(float(p), 0.85))

                        m_ok = np.isfinite(x_all) & np.isfinite(y_all) & (y_all > 0) & (st_all == "ok")
                        if np.any(m_ok):
                            x_ok = x_all[m_ok]
                            y_ok = y_all[m_ok]
                            ylo_ok = ylo_all[m_ok]
                            yhi_ok = yhi_all[m_ok]
                            all_x.extend(list(x_ok))
                            all_y.extend(list(y_ok))

                            vb = np.isfinite(ylo_ok) & np.isfinite(yhi_ok) & (ylo_ok > 0) & (yhi_ok > 0)
                            if np.any(vb):
                                ax.fill_between(x_ok[vb], ylo_ok[vb], yhi_ok[vb], color=color, alpha=0.08, linewidth=0.0, zorder=1)

                            ax.plot(
                                x_ok,
                                y_ok,
                                color=color,
                                linestyle=METHOD_LINESTYLES[method],
                                linewidth=2.0,
                                alpha=0.93,
                                zorder=3,
                            )
                        else:
                            x_ok = np.array([], dtype=float)
                            y_ok = np.array([], dtype=float)

                        # Random baseline markers.
                        rb = np.isfinite(x_all) & (st_all == "random_baseline")
                        if np.any(rb):
                            y_rb = y_all[rb].copy()
                            y_rb[~(np.isfinite(y_rb) & (y_rb > 0))] = 1.0
                            x_rb = x_all[rb].copy()
                            o_rb = np.argsort(x_rb)
                            x_rb = x_rb[o_rb]
                            y_rb = y_rb[o_rb]
                            all_y.extend(list(y_rb[np.isfinite(y_rb) & (y_rb > 0)]))

                            ax.plot(
                                x_rb,
                                y_rb,
                                color=color,
                                linestyle=METHOD_LINESTYLES[method],
                                linewidth=1.45,
                                alpha=0.55,
                                zorder=4,
                            )
                            # Bridge from last trusted point to first baseline point when there is a gap.
                            if x_ok.size and x_rb.size:
                                j_last = int(np.argmax(x_ok))
                                x_last = float(x_ok[j_last])
                                y_last = float(y_ok[j_last])
                                if x_rb[0] > x_last and y_last > 0:
                                    ax.plot(
                                        [x_last, float(x_rb[0])],
                                        [y_last, float(y_rb[0])],
                                        color=color,
                                        linestyle=":",
                                        linewidth=1.2,
                                        alpha=0.42,
                                        zorder=3,
                                    )

                            ax.scatter(
                                x_rb,
                                y_rb,
                                color=color,
                                marker="x",
                                s=26,
                                linewidths=1.05,
                                alpha=0.95,
                                zorder=7,
                            )

            ax.set_yscale("log")
            ax.grid(alpha=0.22)
            for yref, lab in TIME_REF_LINES:
                ax.axhline(y=yref, color="gray", linestyle=":", linewidth=0.8, alpha=0.50, zorder=0)
            if i == 0:
                ax.set_title(f"Device {dev}", fontsize=13)
            if j == 0:
                ax.set_ylabel(r"Sample Complexity $n_c$", fontsize=12)
            if j == (nc - 1):
                ax.text(
                    1.03,
                    0.5,
                    CHANNEL_TITLES.get(ch, ch.title()),
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                    fontsize=11.5,
                    color="black",
                    alpha=0.92,
                )
            if i == nr - 1:
                ax.set_xlabel(r"Number of qubits $n_q$", fontsize=12)

    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        for ax in axes.ravel():
            ax.set_xlim(xmin - 0.4, xmax + 0.4)
    if all_y:
        ymin = min(max(min(all_y) / 1.7, 0.8), 1.0)
        ymax = max(max(all_y) * 1.8, max(y for y, _ in TIME_REF_LINES) * 1.1)
        for ax in axes.ravel():
            ax.set_ylim(ymin, ymax)
            # place y-ref labels at right edge on each subplot
            for yref, lab in TIME_REF_LINES:
                ax.text(0.99, yref, lab, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, color="gray")

    alpha_handles = [Line2D([0], [0], color=ALPHA_BASE_COLORS[a], linewidth=3.0, label=ALPHA_LABELS[a]) for a in ALPHA_ORDER]
    p_handles = [Line2D([0], [0], color=_shade_color("#444444", P_INTENSITY[p]), linewidth=3.0, label=P_LABELS.get(float(p), f"p={p:g}")) for p in p_values]
    method_handles = [Line2D([0], [0], color="black", linestyle=METHOD_LINESTYLES[m], linewidth=2.2, label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    sem_handles = [
        Line2D([0], [0], color="gray", linewidth=6, alpha=0.08, label=str(ci_band_label)),
        Line2D([0], [0], color="gray", marker="x", linestyle="", markersize=6, label="random baseline"),
    ]

    leg1 = fig.legend(handles=alpha_handles, loc="upper left", bbox_to_anchor=(0.82, 0.97), frameon=False, title="Concept class", fontsize=10, title_fontsize=10)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=p_handles, loc="upper left", bbox_to_anchor=(0.82, 0.80), frameon=False, title=r"Noise $\epsilon_p$", fontsize=10, title_fontsize=10)
    fig.add_artist(leg2)
    leg3 = fig.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(0.82, 0.62), frameon=False, title="MF protocol", fontsize=10, title_fontsize=10)
    fig.add_artist(leg3)
    fig.legend(handles=sem_handles, loc="upper left", bbox_to_anchor=(0.82, 0.43), frameon=False, title="Semantics", fontsize=10, title_fontsize=10)

    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="2-column (B,C) x 3-row (channels) MF comparison with alpha/p styling (HG + Eigenshadow only).")
    ap.add_argument("--csv-alpha-nq", type=str, nargs="+", required=True,
                    help="One or more unified CSV files for |alpha|=n_q.")
    ap.add_argument("--csv-alpha-nq2", type=str, nargs="+", required=True,
                    help="One or more unified CSV files for |alpha|=n_q/2.")
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--readout-error", type=str, default="0%",
                    help="Fallback readout filter when --readout-error-by-device is not provided for a device.")
    ap.add_argument("--readout-error-by-device", nargs="*", default=None,
                    help="Per-device readout filters, e.g.: B=0.1% C=1%")
    ap.add_argument("--devices", nargs="*", default=DEVICE_ORDER)
    ap.add_argument("--channels", nargs="*", default=CHANNEL_ORDER)
    ap.add_argument("--ps", nargs="*", type=float, default=P_DEFAULT)
    ap.add_argument("--ci-z", type=float, default=1.0,
                    help="z used for CI semantics label (e.g., 1.0->1-sigma, 1.96->~2-sigma).")
    ap.add_argument(
        "--ci-band-label",
        type=str,
        default=None,
        help="Optional legend label override. If omitted, label is derived from --ci-z.",
    )
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    p1_list = [Path(p).resolve() for p in args.csv_alpha_nq]
    p2_list = [Path(p).resolve() for p in args.csv_alpha_nq2]
    for p in p1_list + p2_list:
        if not p.exists():
            raise FileNotFoundError(p)

    rows: List[dict] = []
    for p1 in p1_list:
        rows.extend(_read_rows(p1, "nq"))
    for p2 in p2_list:
        rows.extend(_read_rows(p2, "nq2"))
    channels = [c for c in args.channels if c in CHANNEL_ORDER]
    devices = [d for d in args.devices if d in DEVICE_ORDER]
    p_values = sorted(set(float(x) for x in args.ps))
    readout_by_device = _parse_readout_by_device(args.readout_error_by_device)
    ci_band_label = str(args.ci_band_label) if args.ci_band_label else _auto_ci_band_label(float(args.ci_z))

    series = _collect_series(
        rows,
        eta=float(args.eta),
        readout_error=str(args.readout_error),
        readout_by_device=readout_by_device,
        channels=channels,
        devices=devices,
        p_values=p_values,
    )
    if not series:
        raise RuntimeError("No rows matched filter; check eta/readout/readout-by-device/devices/channels/ps and CSV inputs.")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else p1_list[0].parent / "grid_BC"
    out_dir.mkdir(parents=True, exist_ok=True)
    if readout_by_device:
        ro_tag = "_".join(f"{d}{str(readout_by_device[d]).replace('%','pct').replace('.','p')}" for d in sorted(readout_by_device))
    else:
        ro_tag = str(args.readout_error).replace('%', 'pct')
    base = (
        f"mf_grid_BC_channels_no_ml_eta_{str(args.eta).replace('.', 'p')}"
        f"_re_{ro_tag}"
        f"_p_{'_'.join(str(p).replace('.','p') for p in p_values)}"
    )
    out_png = out_dir / f"{base}.png"
    out_pdf = out_dir / f"{base}.pdf"

    plot_grid(
        series,
        channels=channels,
        devices=devices,
        p_values=p_values,
        eta=float(args.eta),
        out_png=out_png,
        out_pdf=out_pdf,
        ci_band_label=ci_band_label,
    )
    print(f"[ok] saved: {out_png}")
    print(f"[ok] saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
