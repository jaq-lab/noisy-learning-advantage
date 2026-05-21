"""Utilities for loading precomputed validation shadow feature caches."""

import json
import os
import pickle
from typing import Dict, Sequence
from typing import Optional

import jax.numpy as jnp
import numpy as np


def _build_alpha_pattern(name: str, nq: int):
    if name == "minimal":
        return tuple([0] * (nq - 1) + [1])
    if name == "max":
        return tuple([1] * nq)
    if name == "three":
        ones = min(3, nq)
        return tuple([0] * (nq - ones) + [1] * ones)
    return None


def _build_y_pattern(name: str, nq: int):
    if name == "minimal":
        return "0" * nq
    if name == "max":
        if nq == 1:
            return "0"
        return "1" * (nq - 1) + "0"
    if name == "alternating":
        base = "10" if nq % 2 == 0 else "01"
        return (base * (nq // 2 + 1))[:nq]
    return None


def _infer_alpha_pattern(alpha_bits: Sequence[int]) -> str:
    nq = len(alpha_bits)
    for name in ("minimal", "three", "max"):
        candidate = _build_alpha_pattern(name, nq)
        if candidate is not None and tuple(alpha_bits) == candidate:
            return name
    return "custom_" + "".join(map(str, alpha_bits))


def _infer_y_pattern(y_bits: str) -> str:
    nq = len(y_bits)
    for name in ("minimal", "max", "alternating"):
        candidate = _build_y_pattern(name, nq)
        if candidate is not None and y_bits == candidate:
            return name
    return "custom_" + y_bits


def _format_strength(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "_")


def _features_array_to_dict(indices, feature_array, nps_list):
    features_dict = {}
    for row_idx, sample_idx in enumerate(indices):
        per_sample = {}
        for col_idx, nps in enumerate(nps_list):
            per_sample[int(nps)] = jnp.array(feature_array[row_idx, col_idx])
        features_dict[int(sample_idx)] = per_sample
    return features_dict


def _hamming_weight(n: int) -> int:
    return bin(n).count("1")


def _compute_feature_vector(dm: np.ndarray, alpha_bits: Sequence[int], y_bits: str, nq: int) -> np.ndarray:
    alpha_int = int("".join(map(str, alpha_bits)), 2)
    y_int = int(y_bits, 2)
    idx0 = y_int ^ alpha_int
    idx1 = y_int

    scale = 2 ** nq
    features = []

    # Base diagonals
    features.append(dm[idx0, idx0].real * scale)
    features.append(dm[idx1, idx1].real * scale)

    # Central element
    features.append(dm[idx0, idx1].real * scale)

    # Precompute hamming groups
    groups = {}
    for n in range(2**nq):
        groups.setdefault(_hamming_weight(n), []).append(n)

    # Averaged products
    for k in range(1, nq):
        candidates = [n for n in groups.get(k, []) if n != idx0 and n != idx1]
        if candidates:
            products = []
            for n_val in candidates:
                val0 = dm[idx0, n_val].real * scale
                val1 = dm[n_val, idx1].real * scale
                products.append(val0 * val1)
            features.append(float(np.mean(products)))
        else:
            features.append(0.0)

    # Averaged diagonals
    for k in range(1, nq):
        candidates = [n for n in groups.get(k, []) if n != idx0 and n != idx1]
        if candidates:
            diagonals = [dm[n_val, n_val].real * scale for n_val in candidates]
            features.append(float(np.mean(diagonals)))
        else:
            features.append(0.0)

    return np.asarray(features, dtype=np.float32)


def _features_dict_from_raw(densities: np.ndarray, indices: np.ndarray, nps_list: Sequence[int], alpha_bits, y_bits, nq: int):
    features_dict = {}
    for row, sample_idx in enumerate(indices):
        per_sample = {}
        for col, nps in enumerate(nps_list):
            dm = densities[row, col]
            feat = _compute_feature_vector(dm, alpha_bits, y_bits, nq)
            per_sample[int(nps)] = jnp.array(feat)
        features_dict[int(sample_idx)] = per_sample
    return features_dict


def try_load_precomputed_feature_cache(*args, **kwargs):
    """Backward-compatible stub that always misses (training caches no longer stored)."""
    return None


def try_load_validation_cache(config: Dict, nps_list: Sequence[int], cache_root: Optional[str] = None):
    if cache_root is None:
        cache_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../precomputed_validation"))
    else:
        cache_root = os.path.abspath(cache_root)

    channel_cfg = config.get("channel_config", {})
    channel = channel_cfg.get("type") or config.get("white_noise_decoherence_type")
    if channel is None:
        return None

    strength = channel_cfg.get("strength")
    if strength is None:
        strength = config.get("white_noise_thermal_strength", 0.0)

    alpha_bits = tuple(config.get("alpha_targ"))
    y_bits = config.get("y_targ")

    alpha_pattern = _infer_alpha_pattern(alpha_bits)
    y_pattern = _infer_y_pattern(y_bits)

    filename_root = (
        f"validation_nq{config['n']}_channel-{channel}_strength-{_format_strength(strength)}"
        f"_alpha-{alpha_pattern}_y-{y_pattern}"
    )
    pkl_path = os.path.join(cache_root, f"{filename_root}.pkl")
    json_path = os.path.join(cache_root, f"{filename_root}.json")

    if not os.path.exists(pkl_path) or not os.path.exists(json_path):
        raw = load_validation_density_matrices(config, cache_root)
        if raw is None:
            return None
        features_dict = _features_dict_from_raw(
            raw["density_matrices"], raw["indices"], nps_list, alpha_bits, y_bits, config["n"]
        )
        return {
            "features_dict": features_dict,
            "indices": raw["indices"].tolist(),
            "labels": raw["labels"].tolist(),
            "metadata": raw["metadata"],
            "density_path": os.path.join(cache_root, f"validation_raw_nq{config['n']}_channel-{channel}_strength-{_format_strength(strength)}.pkl"),
        }

    with open(json_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    cache_nps = [int(n) for n in metadata.get("nps_list", [])]
    if not cache_nps:
        return None

    try:
        column_indices = [cache_nps.index(int(nps)) for nps in nps_list]
    except ValueError:
        print(f"[CACHE] Requested nps {nps_list} not covered by cache {pkl_path}")
        return None

    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)

    features_array = np.asarray(payload["features"], dtype=np.float32)
    labels_array = np.asarray(payload["labels"], dtype=np.int8)
    indices = np.asarray(payload.get("indices", np.arange(features_array.shape[0])), dtype=int)

    if features_array.ndim != 3:
        print(f"[CACHE] Unexpected feature array shape in {pkl_path}: {features_array.shape}")
        return None

    features_subset = features_array[:, column_indices, :]
    features_dict = _features_array_to_dict(indices, features_subset, nps_list)

    return {
        "features_dict": features_dict,
        "indices": indices.tolist(),
        "labels": labels_array.tolist(),
        "metadata": metadata,
    }


def load_validation_density_matrices(config: Dict, cache_root: Optional[str] = None):
    if cache_root is None:
        cache_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../precomputed_validation"))
    else:
        cache_root = os.path.abspath(cache_root)

    channel_cfg = config.get("channel_config", {})
    channel = channel_cfg.get("type") or config.get("white_noise_decoherence_type")
    if channel is None:
        return None

    strength = channel_cfg.get("strength")
    if strength is None:
        strength = config.get("white_noise_thermal_strength", 0.0)

    filename_root = (
        f"validation_raw_nq{config['n']}_channel-{channel}_strength-{_format_strength(strength)}"
    )
    raw_pkl = os.path.join(cache_root, f"{filename_root}.pkl")
    raw_json = os.path.join(cache_root, f"{filename_root}.json")

    if not os.path.exists(raw_pkl) or not os.path.exists(raw_json):
        return None

    with open(raw_json, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    with open(raw_pkl, "rb") as fh:
        payload = pickle.load(fh)

    densities = np.asarray(payload["density_matrices"], dtype=np.complex64)
    indices = np.asarray(payload["indices"], dtype=int)
    labels = np.asarray(payload["labels"], dtype=np.int8)

    return {
        "density_matrices": densities,
        "indices": indices,
        "labels": labels,
        "metadata": metadata,
    }
