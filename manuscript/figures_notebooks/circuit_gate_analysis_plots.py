"""Load ``circuit_gate_analysis.csv`` and plot device A gate / delay summaries.

Kept next to the CSV in ``code/quantum_simulation/``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CIRCUIT_GATE_ANALYSIS_CSV = Path(__file__).resolve().parent / "circuit_gate_analysis.csv"

# Two-qubit coherence time used to normalize delays and total wait (nanoseconds).
T_2Q_NS = 100.0


def load_circuit_gate_dataframe(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Read the analysis CSV and restore list/dict columns saved as strings."""
    path = Path(csv_path) if csv_path is not None else CIRCUIT_GATE_ANALYSIS_CSV
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)

    if "delay_durations_ns" in df.columns:

        def _parse_list(v):
            if isinstance(v, str):
                return ast.literal_eval(v)
            if isinstance(v, (list, np.ndarray)):
                return list(v)
            return v

        df["delay_durations_ns"] = df["delay_durations_ns"].map(_parse_list)

    if "gate_types" in df.columns:

        def _parse_gt(v):
            if isinstance(v, str) and v.strip():
                try:
                    return ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    return v
            return v

        df["gate_types"] = df["gate_types"].map(_parse_gt)

    return df


def fit_power_law_nq(
    nq_arr: np.ndarray, y_arr: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    """Return (c, alpha) for y ≈ c * nq**alpha, or (None, None) if fit not possible."""
    mask = (y_arr > 0) & (nq_arr > 0)
    if mask.sum() < 2:
        return None, None
    alpha, log_c = np.polyfit(np.log(nq_arr[mask]), np.log(y_arr[mask]), 1)
    return float(np.exp(log_c)), float(alpha)


def plot_powerlaw_extrap(
    ax,
    nq_arr: np.ndarray,
    y_arr: np.ndarray,
    color: str,
    label: str,
    nq_max_extrap: float,
) -> None:
    """Scatter data + c*nq^alpha; dashed beyond last data n_q."""
    c, alpha = fit_power_law_nq(nq_arr, y_arr)
    if c is None:
        return
    n_lo = float(np.min(nq_arr))
    n_hi = float(nq_max_extrap)
    nq_line = np.linspace(n_lo, n_hi, 400)
    y_line = c * nq_line**alpha
    n_data_max = float(np.max(nq_arr))
    in_sample = nq_line <= n_data_max
    extrap = nq_line > n_data_max
    ax.plot(
        nq_line[in_sample],
        y_line[in_sample],
        "-",
        color=color,
        alpha=0.95,
        label=label,
    )
    if np.any(extrap):
        ax.plot(nq_line[extrap], y_line[extrap], "--", color=color, alpha=0.95)


def plot_device_a_panel(
    df: pd.DataFrame,
    device: str = "A1",
    nq_extrap: float = 50,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Left: 1Q/2Q counts + power-law fits (extrapolated to ``nq_extrap``); total wait on twin axis
    in units of total delay / ``T_2Q_NS``.
    Right: pcolormesh of delay-duration histograms vs n_q; delay axis is τ / ``T_2Q_NS``.
    """
    if "delay_durations_ns" not in df.columns:
        raise ValueError(
            "df has no 'delay_durations_ns'. Re-run the analysis with delay extraction, "
            "or regenerate the CSV from that pipeline."
        )

    dA = df[df["device"] == device].sort_values("nq")
    if dA.empty:
        raise ValueError(f"No rows for device {device}.")

    nqs = dA["nq"].to_numpy()
    delay_lists = dA["delay_durations_ns"].to_numpy()

    total_wait_ns = np.array([float(np.sum(x)) if len(x) else 0.0 for x in delay_lists])
    total_wait_over_T2q = total_wait_ns / T_2Q_NS

    all_delays = np.concatenate([np.asarray(x, dtype=float) for x in delay_lists if len(x)])
    if all_delays.size == 0:
        raise ValueError("No delay gates recorded for this device; check transpiled circuits.")

    tmax_ns = float(np.max(all_delays))
    n_bins = 48
    bin_edges_ns = np.linspace(0.0, tmax_ns * 1.01, n_bins + 1)
    bin_edges_over_T2q = bin_edges_ns / T_2Q_NS
    Z = np.zeros((len(nqs), n_bins))
    for i, arr in enumerate(delay_lists):
        arr = np.asarray(arr, dtype=float)
        if arr.size:
            Z[i, :], _ = np.histogram(arr, bins=bin_edges_ns)

    if len(nqs) == 1:
        y_edges = np.array([nqs[0] - 0.5, nqs[0] + 0.5])
    else:
        y_edges = np.empty(len(nqs) + 1)
        y_edges[0] = nqs[0] - 0.5 * (nqs[1] - nqs[0])
        y_edges[-1] = nqs[-1] + 0.5 * (nqs[-1] - nqs[-2])
        y_edges[1:-1] = 0.5 * (nqs[:-1] + nqs[1:])

    fs = figsize if figsize is not None else (14, 5.2)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=fs, constrained_layout=True)

    twq = dA["two_qubit_gates"].to_numpy(dtype=float)
    soq = dA["single_qubit_gates"].to_numpy(dtype=float)

    ax_l.scatter(nqs, twq, color="C0", s=30, zorder=5, label="Two-qubit gates (data)")
    c2, a2 = fit_power_law_nq(nqs, twq)
    if c2 is not None:
        plot_powerlaw_extrap(
            ax_l,
            nqs,
            twq,
            "C0",
            rf"2Q fit: ${c2:.3g}\,n_q^{{{a2:.2f}}}$",
            nq_extrap,
        )

    ax_l.scatter(nqs, soq, color="C1", s=30, marker="s", zorder=5, label="Single-qubit gates (data)")
    c1, a1 = fit_power_law_nq(nqs, soq)
    if c1 is not None:
        plot_powerlaw_extrap(
            ax_l,
            nqs,
            soq,
            "C1",
            rf"1Q fit: ${c1:.3g}\,n_q^{{{a1:.2f}}}$",
            nq_extrap,
        )

    ax_l.set_xlabel(r"Number of qubits $n_q$")
    ax_l.set_ylabel("Gate count")
    ax_l.set_xlim(left=nqs.min() - 0.5, right=nq_extrap + 0.5)
    ax_l.set_axisbelow(True)
    ax_l.grid(True, alpha=0.35)

    ax_m = ax_l.twinx()
    ax_m.spines["right"].set_visible(True)
    ax_m.scatter(
        nqs, total_wait_over_T2q, color="C3", marker="^", s=36, zorder=5, label="Total wait (data)"
    )
    cw, aw = fit_power_law_nq(nqs, total_wait_over_T2q)
    if cw is not None:
        plot_powerlaw_extrap(
            ax_m,
            nqs,
            total_wait_over_T2q,
            "C3",
            rf"Wait fit: ${cw:.3g}\,n_q^{{{aw:.2f}}}$",
            nq_extrap,
        )
    ax_m.set_ylabel(r"Total waiting time / $T_{2\mathrm{q}}$", color="C3")
    ax_m.tick_params(axis="y", labelcolor="C3")
    ax_m.grid(False)

    lines_l, lab_l = ax_l.get_legend_handles_labels()
    lines_m, lab_m = ax_m.get_legend_handles_labels()
    leg = ax_l.legend(
        lines_l + lines_m,
        lab_l + lab_m,
        loc="upper left",
        fontsize=7,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="0.85",
    )
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(1000)

    ax_l.set_title("Gates and wait time")
    pcm = ax_r.pcolormesh(bin_edges_over_T2q, y_edges, Z, shading="auto", cmap="viridis")
    ax_r.set_xlabel(r"Delay $\tau / T_{2\mathrm{q}}$")
    ax_r.set_ylabel(r"Number of qubits $n_q$")
    ax_r.set_title("Delay histograms")
    fig.colorbar(pcm, ax=ax_r, label="Counts per bin")

    if show:
        plt.show()
    return fig
