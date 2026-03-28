#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


def acc_dephasing(p: float, d: float, nq: int) -> float:
    vp = (1.0 - 2.0 * float(p)) ** float(d)
    return float(0.5 * (1.0 + vp))


def acc_depolarizing(p: float, d: float, nq: int) -> float:
    vp = ((1.0 - 4.0 * float(p) / 3.0) ** float(d)) * ((1.0 - 2.0 * float(p) / 3.0) ** float(int(nq) - float(d)))
    return float(0.5 * (1.0 + vp))


def acc_relaxation(p: float, d: float, nq: int) -> float:
    gamma_active = (np.sqrt(1.0 - float(p)) ** float(d))
    gamma_passive_avg = ((1.0 - float(p) / 2.0) ** (float(int(nq) - float(d))))
    vp = gamma_active * gamma_passive_avg
    return float(0.5 * (1.0 + vp))


CHANNEL_TO_ACC_FN: Dict[str, Callable[[float, float, int], float]] = {
    "dephasing": acc_dephasing,
    "depolarizing": acc_depolarizing,
    "relaxation": acc_relaxation,
}

_NQ_KEY_TO_DMODE = {
    # V_mr_nq_grid may appear for both alpha=nq and alpha=nq/2 sources.
    "V_mr_nq_grid": None,
    "V_m_nq_grid": "nq",
    "V_m_alpha_equals_nq_grid": "nq",
    "V_m_nq_over_2_grid": "nq_over_2",
    "V_m_alpha_equals_nq_over_2_grid": "nq_over_2",
}


def _format_readout_label(eps_r: float) -> str:
    pct = 100.0 * float(eps_r)
    if abs(pct - round(pct)) < 1e-12:
        return f"{int(round(pct))}%"
    s = f"{pct:.6f}".rstrip("0").rstrip(".")
    return f"{s}%"


def _dict_series_on_grid(grid: List[int], raw: dict, *, name: str) -> List[float]:
    vm_map = {int(k): float(v) for k, v in raw.items()}
    out = [vm_map[int(n)] for n in grid if int(n) in vm_map]
    if len(out) != len(grid):
        missing = [int(n) for n in grid if int(n) not in vm_map]
        raise ValueError(
            f"Missing entries in {name}: nq={missing[:8]}" + (" ..." if len(missing) > 8 else "")
        )
    return out


def _series_on_grid(grid: List[int], raw, *, name: str) -> List[float]:
    if isinstance(raw, dict):
        return _dict_series_on_grid(grid, raw, name=name)
    if isinstance(raw, list):
        vals = [float(x) for x in raw]
        if len(vals) != len(grid):
            raise ValueError(
                f"Length mismatch in {name}: nq_grid={len(grid)} vs values={len(vals)}"
            )
        return vals
    raise TypeError(f"Unsupported type in {name}: {type(raw)}")


def _select_visibility_key(payload: dict, d_mode: str) -> str:
    preferred = (
        ["V_mr_nq_grid", "V_m_nq_grid", "V_m_alpha_equals_nq_grid"]
        if d_mode == "nq"
        else ["V_mr_nq_grid", "V_m_nq_over_2_grid", "V_m_alpha_equals_nq_over_2_grid"]
    )
    for k in preferred:
        if k in payload:
            return k
    keys = [k for k in payload.keys() if k in _NQ_KEY_TO_DMODE]
    raise KeyError(f"No supported visibility key found for d_mode={d_mode}. Present keys={keys}")


def _extract_readout_eps(src: dict, payload: dict, dev_ab: str) -> float:
    if "eps_r" in payload:
        return float(payload["eps_r"])
    ro = src.get("readout_eps_by_device", {})
    if isinstance(ro, dict) and dev_ab in ro:
        return float(ro[dev_ab])
    return 0.0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _compute_vq_from_base_visibility(
    vis_values: List[float],
    nq_values: List[int],
    *,
    p: float,
    channel: str,
    d_mode: str,
) -> List[float]:
    fn = CHANNEL_TO_ACC_FN[channel]
    out: List[float] = []
    for nq, vis in zip(nq_values, vis_values):
        d = float(nq) if d_mode == "nq" else float(nq) / 2.0
        aq_noise_only = fn(float(p), d, int(nq))
        vp = 2.0 * float(aq_noise_only) - 1.0
        vq = float(vis) * float(vp)
        out.append(vq)
    return out


def build_curves(
    vm_grid_path: Path,
    *,
    d_mode: str,
    amplitudes: List[str],
    nq_min: int,
    nq_max: int,
) -> dict:
    src = _load_json(vm_grid_path)
    exports = dict(src.get("exports", {}))
    if not exports:
        raise ValueError(f"No 'exports' section in {vm_grid_path}")

    # Source maps I->A, T->B, S->C. Reverse map to match existing unified scripts.
    dev_map = dict(src.get("device_mapping", {}))
    rev_map = {str(v): str(k) for k, v in dev_map.items()}  # A->I, B->T, C->S

    devices_out: List[str] = []
    device_readout_eps: Dict[str, float] = {}
    device_visibility_key: Dict[str, str] = {}
    curves: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    nq_ref: List[int] = []

    for dev_ab, payload in exports.items():
        dev_ab = str(dev_ab)
        dev_out = rev_map.get(dev_ab, dev_ab)
        devices_out.append(dev_out)
        nq_raw = payload.get("nq_grid", src.get("nq_grid", []))
        nq_grid = [int(x) for x in nq_raw]
        if not nq_grid:
            raise ValueError(f"Missing nq_grid for device={dev_ab} in {vm_grid_path}")

        vis_key = _select_visibility_key(payload, d_mode=d_mode)
        key_mode = _NQ_KEY_TO_DMODE.get(vis_key)
        if key_mode not in {d_mode, None}:
            raise ValueError(
                f"Visibility key {vis_key} incompatible with d_mode={d_mode} for device={dev_ab}"
            )
        vis_grid = _series_on_grid(nq_grid, payload[vis_key], name=f"{vis_key} device={dev_ab}")

        idx = [i for i, n in enumerate(nq_grid) if int(nq_min) <= int(n) <= int(nq_max)]
        if not idx:
            raise ValueError(f"No nq in requested range [{nq_min},{nq_max}] for device={dev_ab}")
        nq_use = [nq_grid[i] for i in idx]
        vis_use = [vis_grid[i] for i in idx]
        if not nq_ref:
            nq_ref = list(nq_use)
        elif nq_ref != nq_use:
            raise ValueError("nq grid mismatch across devices in source V_mr file")

        eps_r = _extract_readout_eps(src, payload, dev_ab)
        ro_label = _format_readout_label(eps_r)
        device_readout_eps[dev_out] = float(eps_r)
        device_visibility_key[dev_out] = vis_key

        curves.setdefault(dev_out, {})
        for channel in ["dephasing", "relaxation", "depolarizing"]:
            curves[dev_out].setdefault(channel, {})
            for amp in amplitudes:
                p = float(amp)
                vq = _compute_vq_from_base_visibility(vis_use, nq_use, p=p, channel=channel, d_mode=d_mode)
                aq = [float(np.clip(0.5 * (1.0 + x), 0.0, 1.0)) for x in vq]
                curves[dev_out][channel][str(amp)] = {ro_label: aq}

    readout_union = sorted({_format_readout_label(v) for v in device_readout_eps.values()})

    out = {
        "description": (
            "Quantum accuracy curves derived from clean visibility grids and channel attenuation V_p; "
            "A_Q = (1 + V * V_p)/2, where V is V_mr when present, otherwise V_m."
        ),
        "source_vm_grid_file": str(vm_grid_path.resolve()),
        "alpha_d_mode": d_mode,
        "nq_values": nq_ref,
        "devices": sorted(set(devices_out), key=lambda x: ["I", "S", "T"].index(x) if x in ["I", "S", "T"] else 99),
        "channels": ["dephasing", "relaxation", "depolarizing"],
        "amplitudes": [str(a) for a in amplitudes],
        "readout_errors": readout_union,
        "curves": curves,
        "device_mapping_source": dev_map,
        "device_readout_eps": device_readout_eps,
        "device_readout_labels": {k: _format_readout_label(v) for k, v in device_readout_eps.items()},
        "device_visibility_key": device_visibility_key,
        "note": "Devices map as I->A, T->B, S->C in the source visibility JSON.",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build quantum_accuracy_curves-style JSON from clean visibility grids (nq and nq/2)."
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="/home/ypatel/data1/noisy-learning-advantage/data",
    )
    ap.add_argument(
        "--amplitudes",
        nargs="*",
        default=["0.01", "0.05", "0.1"],
    )
    ap.add_argument("--nq-min", type=int, default=5)
    ap.add_argument("--nq-max", type=int, default=50)
    ap.add_argument(
        "--src-nq",
        type=str,
        default="fig4_clean_m_vm_grids_nq_readout.json",
        help="Source JSON for |alpha|=nq (supports V_mr_* or V_m_* schemas).",
    )
    ap.add_argument(
        "--src-nq2",
        type=str,
        default="fig4_clean_m_vm_grids_nq2_readout.json",
        help="Source JSON for |alpha|=nq/2 (supports V_mr_* or V_m_* schemas).",
    )
    ap.add_argument(
        "--out-nq",
        type=str,
        default="quantum_accuracy_curves_from_vm_alpha_nq.json",
        help="Output JSON filename for |alpha|=nq curves (written under --data-dir).",
    )
    ap.add_argument(
        "--out-nq2",
        type=str,
        default="quantum_accuracy_curves_from_vm_alpha_nq2.json",
        help="Output JSON filename for |alpha|=nq/2 curves (written under --data-dir).",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    src_nq = data_dir / str(args.src_nq)
    src_nq2 = data_dir / str(args.src_nq2)
    if not src_nq.exists():
        raise FileNotFoundError(src_nq)
    if not src_nq2.exists():
        raise FileNotFoundError(src_nq2)

    out_nq = data_dir / str(args.out_nq)
    out_nq2 = data_dir / str(args.out_nq2)

    obj_nq = build_curves(
        src_nq,
        d_mode="nq",
        amplitudes=[str(x) for x in args.amplitudes],
        nq_min=int(args.nq_min),
        nq_max=int(args.nq_max),
    )
    obj_nq2 = build_curves(
        src_nq2,
        d_mode="nq_over_2",
        amplitudes=[str(x) for x in args.amplitudes],
        nq_min=int(args.nq_min),
        nq_max=int(args.nq_max),
    )
    _save_json(out_nq, obj_nq)
    _save_json(out_nq2, obj_nq2)

    print(f"[ok] wrote: {out_nq}")
    print(f"[ok] wrote: {out_nq2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
