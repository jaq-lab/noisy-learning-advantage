#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _compute_vq_from_vm(
    vmr_values: List[float],
    nq_values: List[int],
    *,
    p: float,
    channel: str,
    d_mode: str,
) -> List[float]:
    fn = CHANNEL_TO_ACC_FN[channel]
    out: List[float] = []
    for nq, vmr in zip(nq_values, vmr_values):
        d = float(nq) if d_mode == "nq" else float(nq) / 2.0
        aq_noise_only = fn(float(p), d, int(nq))
        vp = 2.0 * float(aq_noise_only) - 1.0
        vq = float(vmr) * float(vp)
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
    curves: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    nq_ref: List[int] = []

    for dev_ab, payload in exports.items():
        dev_ab = str(dev_ab)
        dev_out = rev_map.get(dev_ab, dev_ab)
        devices_out.append(dev_out)
        nq_grid = [int(x) for x in payload.get("nq_grid", [])]
        vmr_raw = payload.get("V_mr_nq_grid", {})
        if isinstance(vmr_raw, dict):
            vmr_map = {int(k): float(v) for k, v in vmr_raw.items()}
            vmr_grid = [vmr_map[int(n)] for n in nq_grid if int(n) in vmr_map]
            if len(vmr_grid) != len(nq_grid):
                missing = [int(n) for n in nq_grid if int(n) not in vmr_map]
                raise ValueError(
                    f"Missing V_mr entries for device={dev_ab} at nq={missing[:8]}"
                    + (" ..." if len(missing) > 8 else "")
                )
        elif isinstance(vmr_raw, list):
            vmr_grid = [float(x) for x in vmr_raw]
        else:
            raise TypeError(f"Unsupported V_mr_nq_grid type for device={dev_ab}: {type(vmr_raw)}")
        if len(nq_grid) != len(vmr_grid):
            raise ValueError(
                f"Length mismatch for device={dev_ab}: nq_grid={len(nq_grid)} vs V_mr_nq_grid={len(vmr_grid)}"
            )

        idx = [i for i, n in enumerate(nq_grid) if int(nq_min) <= int(n) <= int(nq_max)]
        if not idx:
            raise ValueError(f"No nq in requested range [{nq_min},{nq_max}] for device={dev_ab}")
        nq_use = [nq_grid[i] for i in idx]
        vmr_use = [vmr_grid[i] for i in idx]
        if not nq_ref:
            nq_ref = list(nq_use)
        elif nq_ref != nq_use:
            raise ValueError("nq grid mismatch across devices in source V_mr file")

        curves.setdefault(dev_out, {})
        for channel in ["dephasing", "relaxation", "depolarizing"]:
            curves[dev_out].setdefault(channel, {})
            for amp in amplitudes:
                p = float(amp)
                vq = _compute_vq_from_vm(vmr_use, nq_use, p=p, channel=channel, d_mode=d_mode)
                aq = [float(np.clip(0.5 * (1.0 + x), 0.0, 1.0)) for x in vq]
                curves[dev_out][channel][str(amp)] = {"0%": aq}

    out = {
        "description": (
            "Quantum accuracy curves derived from clean V_mr(nq) grids and channel attenuation V_p; "
            "A_Q = (1 + V_mr * V_p)/2."
        ),
        "source_vm_grid_file": str(vm_grid_path.resolve()),
        "alpha_d_mode": d_mode,
        "nq_values": nq_ref,
        "devices": sorted(set(devices_out), key=lambda x: ["I", "S", "T"].index(x) if x in ["I", "S", "T"] else 99),
        "channels": ["dephasing", "relaxation", "depolarizing"],
        "amplitudes": [str(a) for a in amplitudes],
        "readout_errors": ["0%"],
        "curves": curves,
        "device_mapping_source": dev_map,
        "note": "Devices map as I->A, T->B, S->C in the source V_mr JSON.",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build quantum_accuracy_curves-style JSON from fig4_clean_vm_grids_{nq,nq2}.json."
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
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    src_nq = data_dir / "fig4_clean_vm_grids_nq.json"
    src_nq2 = data_dir / "fig4_clean_vm_grids_nq2.json"
    if not src_nq.exists():
        raise FileNotFoundError(src_nq)
    if not src_nq2.exists():
        raise FileNotFoundError(src_nq2)

    out_nq = data_dir / "quantum_accuracy_curves_from_vm_alpha_nq.json"
    out_nq2 = data_dir / "quantum_accuracy_curves_from_vm_alpha_nq2.json"

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
