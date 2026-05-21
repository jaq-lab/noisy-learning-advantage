from __future__ import annotations
from typing import Dict, Tuple, Any, List, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import jax.random as jr
try:
    from shadow_funcs_dm import jit_vectorized_shadow_computation, get_shadow_rho_noiseless_pure
except ImportError:
    # If shadow_funcs_dm is not on sys.path, try adding the sibling
    # `shadows_simulation` directory (expected layout: code/quantum_sim2 and code/shadows_simulation)
    import sys
    from pathlib import Path
    this = Path(__file__).resolve()
    candidate = this.parent.parent / "shadows_simulation"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
    else:
        # fall back to original parent-of-parent (older layouts)
        sys.path.insert(0, str(this.parent.parent))
    from shadow_funcs_dm import jit_vectorized_shadow_computation, get_shadow_rho_noiseless_pure

Array = jnp.ndarray

# -----------------------------------------------------------------------------
# Optional: enable float64/complex128 to match NumPy complex128 behavior
# from jax import config
# config.update("jax_enable_x64", True)

EPS = 1e-12

# =============================================================================
# SECTION 1: NOISE CHANNEL DEFINITIONS (JAX arrays)
# =============================================================================

def get_kraus_operators(channel_config: Dict[str, Any]) -> Array:
    """Return a stack of single-qubit Kraus operators as a JAX array.

    Args:
        channel_config: dict with keys:
            - 'type': {'dephasing','bit_flip','thermal','depolarizing'}
            - 'strength': float in [0,1]
            - 'thermal_p_exc': (for thermal/relaxation) float in [0,1]

    Returns:
        K: jnp.ndarray with shape (K, 2, 2), complex dtype.
    """
    ctype = channel_config.get("type")
    p = channel_config.get("strength", 0.0)

    if ctype == "dephasing":  # NOTE: updated to match Z flip.
        K0 = jnp.sqrt(1.0 - p) * jnp.eye(2, dtype=jnp.complex128)
        K1 = jnp.sqrt(p) * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
        K = jnp.stack([K0, K1], axis=0)
    elif ctype == "bit_flip":        
        s = jnp.sqrt(1.0 - p)
        r = jnp.sqrt(p)
        K0 = jnp.array([[s, 0.0], [0.0, s]], dtype=jnp.complex128)
        K1 = jnp.array([[0.0, r], [r, 0.0]], dtype=jnp.complex128)
        K = jnp.stack([K0, K1], axis=0)
    elif ctype == "thermal":
        gamma = p
        p_exc = channel_config.get("thermal_p_exc", 0.0)
        s0 = jnp.sqrt(1.0 - p_exc)
        s1 = jnp.sqrt(p_exc)
        K0 = s0 * jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]], dtype=jnp.complex128)
        K1 = s0 * jnp.array([[0.0, jnp.sqrt(gamma)], [0.0, 0.0]], dtype=jnp.complex128)
        K2 = s1 * jnp.array([[jnp.sqrt(1.0 - gamma), 0.0], [0.0, 1.0]], dtype=jnp.complex128)
        K3 = s1 * jnp.array([[0.0, 0.0], [jnp.sqrt(gamma), 0.0]], dtype=jnp.complex128)
        K = jnp.stack([K0, K1, K2, K3], axis=0)

    elif ctype == "depolarizing":
        # p is the total probability of an error occurring.
        # The error is equally likely to be X, Y, or Z (p/3 each).
        s = jnp.sqrt(1.0 - p)
        r = jnp.sqrt(p / 3.0)

        K0 = jnp.array([[s, 0.0], [0.0, s]], dtype=jnp.complex128)              # sqrt(1-p) * I
        K1 = jnp.array([[0.0, r], [r, 0.0]], dtype=jnp.complex128)              # sqrt(p/3) * X
        K2 = jnp.array([[0.0, -1j * r], [1j * r, 0.0]], dtype=jnp.complex128)   # sqrt(p/3) * Y
        K3 = jnp.array([[r, 0.0], [0.0, -r]], dtype=jnp.complex128)             # sqrt(p/3) * Z
        K = jnp.stack([K0, K1, K2, K3], axis=0)
    else:
        raise ValueError(f"Unsupported channel type: {ctype}")

    return K

# =============================================================================
# SECTION 2: STATE PREPARATION (JAX)
# =============================================================================

def generate_psi_F_vector(F_list_vals: Array, N: int) -> Array:
    """Generate normalized state |psi_F> with entries (-1)^F_k.

    Args:
        F_list_vals: array-like of shape (2**N,), values 0/1 (ints)
        N: number of qubits

    Returns:
        psi: complex JAX array shape (2**N,)
    """
    F = jnp.asarray(F_list_vals)
    dim = 1 << N
    if F.shape[0] != dim:
        raise ValueError("F_list_vals must have length 2**N")
    psi = (-1.0) ** F.astype(jnp.int32)
    psi = psi.astype(jnp.complex128)
    norm = jnp.linalg.norm(psi)
    psi = psi / jnp.where(norm > 1e-9, norm, 1.0)
    return psi

# =============================================================================
# SECTION 3: LOCAL SINGLE-QUBIT APPLY USING lax.switch (JIT-safe)
# =============================================================================

def _apply_local_op_switch(psi: Array, op2x2: Array, N: int, qubit: Array) -> Array:
    """Apply a 2x2 op to the given qubit index using fixed-shape branches.

    Works with jit because the reshape plan per qubit is static per branch.
    """
    dim = psi.shape[0]

    def make_branch(i: int):
        A = 1 << (N - i - 1)
        B = 1 << i
        def branch(v: Array) -> Array:
            v3 = v.reshape((A, 2, B))  # (a, q, b)
            v3p = jnp.einsum('Qq,aqb->aQb', op2x2, v3)
            return v3p.reshape((dim,))
        return branch

    branches = tuple(make_branch(i) for i in range(N))
    return lax.switch(qubit, branches, psi)

# =============================================================================
# SECTION 4: SINGLE-TRAJECTORY SAMPLER (PURE JAX, JIT + VMAP)
# =============================================================================

def _kraus_is_identity_like(Ks: Array, tol: float = 1e-10) -> Array:
    """Return boolean mask (K,) where True if K is proportional to identity (or zero).
    Condition: off-diagonals ~ 0 and diagonals equal within tol.
    """
    off = jnp.abs(Ks[:, 0, 1]) + jnp.abs(Ks[:, 1, 0])
    diff = jnp.abs(Ks[:, 0, 0] - Ks[:, 1, 1])
    return (off <= tol) & (diff <= tol)

def _trajectory_from_key(key: Array, psi0: Array, N: int, Ks: Array) -> Tuple[Array, Array, Array]:
    """Run one quantum trajectory across N qubits with Kraus ops Ks.

    Returns (psi_final, num_errors, code) where:
      - num_errors: counts Kraus choices that are **not** proportional to identity
      - code: base-B integer encoding of the jump record (B = Ks.shape[0])
    """
    # Use a Python int for K (static from shape) to avoid tracer-based shapes
    K = Ks.shape[0]
    # Precompute which Kraus are identity-like (no-error) and store as int32 mask
    is_identity_like = _kraus_is_identity_like(Ks)
    # For channels where K0 is not strictly identity-like but represents "no error",
    # explicitly mark K0 as identity. This handles dephasing channels with non-standard representation.
    is_identity_like = is_identity_like.at[0].set(True)  # K0 always counts as "no error"
    is_error_i32 = (~is_identity_like).astype(jnp.int32)

    def body(carry, i):
        key, psi, err, code, factor = carry
        key, sub = jr.split(key)

        apply_K = lambda Kmat: _apply_local_op_switch(psi, Kmat, N, i)
        outcomes = jax.vmap(apply_K, in_axes=(0,))(Ks)  # (K, 2**N)
        norms = jax.vmap(lambda v: jnp.real(jnp.vdot(v, v)))(outcomes)  # (K,)
        sum_p = jnp.sum(norms)
        # Avoid constructing shapes from tracers; use full_like for uniform fallback
        uniform = jnp.full_like(norms, 1.0 / norms.shape[0])
        probs = jnp.where(sum_p > 1e-9, norms / sum_p, uniform)

        chosen = jr.categorical(sub, jnp.log(probs + EPS))  # int index
        chosen = lax.convert_element_type(chosen, jnp.int32)
        chosen_v = outcomes[chosen]
        chosen_norm = jnp.sqrt(jnp.maximum(norms[chosen], EPS))
        psi_new = chosen_v / chosen_norm
        err_new = err + is_error_i32[chosen]
        code_new = code + chosen * factor
        factor_new = factor * jnp.int32(K)
        return (key, psi_new, err_new, code_new, factor_new), None

    init = (key, psi0, jnp.int32(0), jnp.int32(0), jnp.int32(1))
    (keyf, psif, errf, codef, _), _ = lax.scan(body, init, xs=jnp.arange(N), length=N)
    return psif, errf, codef

# JIT-compiled entry (N is static for best performance)
trajectory_from_key_jit = jax.jit(_trajectory_from_key, static_argnames=("N",))

# =============================================================================
# SECTION 5: PUBLIC SAMPLING APIS (JIT + VMAP friendly)
# =============================================================================

def sample_trajectories(key: Array, fs: Array, nq: int, channel_config: Dict[str, Any], num_samples: int) -> Tuple[Array, Array, Array]:
    """Sample `num_samples` trajectories.

    Args:
        key: PRNGKey
        fs: (2**nq,) int array used to build psi_F
        nq: number of qubits (static for jit)
        channel_config: noise channel dict
        num_samples: number of independent trajectories

    Returns:
        states: (S, 2**nq) complex JAX array
        errors: (S,) int32 JAX array (number of non-identity Kraus picks)
        codes:  (S,) int32 JAX array (base-B encoded jump record)
    """
    Ks = get_kraus_operators(channel_config)
    psi0 = generate_psi_F_vector(fs, nq)
    keys = jr.split(key, num_samples)
    one = lambda k: trajectory_from_key_jit(k, psi0, nq, Ks)
    states, errs, codes = jax.vmap(one)(keys)
    return states, errs, codes


def sample_trajectories_batched(keys: Array, fs: Array, nq: int, channel_config: Dict[str, Any]) -> Tuple[Array, Array, Array]:
    """VMAP over a batch of keys (one trajectory per key).

    Args:
        keys: (B, 2) PRNGKey array
        fs: (2**nq,) ints
        nq: int
        channel_config: dict

    Returns:
        states: (B, 2**nq), errors: (B,), codes: (B,)
    """
    Ks = get_kraus_operators(channel_config)
    psi0 = generate_psi_F_vector(fs, nq)
    one = lambda k: trajectory_from_key_jit(k, psi0, nq, Ks)
    return jax.vmap(one)(keys)

# =============================================================================
# SECTION 6: JAX-NATIVE GROUPING BY JUMP CODE (STATIC-SHAPE OUTPUT)
# =============================================================================

def group_trajectories_jax(states: Array, errs: Array, codes: Array) -> Dict[str, Array]:
    """JAX-native grouping by discrete jump codes.

    This returns **static-shape** arrays (size = number of samples S):
      - data is sorted by code
      - `is_group_start` marks the first row of each group
      - `counts_at_starts` holds multiplicities at group starts, 0 elsewhere

    You can pass this through `jit` / `vmap` safely.
    A non-jitted convenience wrapper can compress groups to variable-length outputs.
    """
    S = codes.shape[0]
    order = jnp.argsort(codes, stable=True)
    codes_sorted = codes[order]
    errs_sorted = errs[order]
    states_sorted = states[order]

    # group start when code changes
    first = jnp.array([True])
    changes = jnp.concatenate([first, codes_sorted[1:] != codes_sorted[:-1]])
    is_group_start = changes

    # Group ids via cumsum over starts
    gid = jnp.cumsum(is_group_start.astype(jnp.int32)) - 1  # (S,)

    # Bincount with static length S (counts beyond actual G will be 0)
    counts_by_gid = jnp.bincount(gid, length=S)  # (S,)

    # Number of groups = last gid + 1
    G = gid[-1] + 1

    # Assign counts to start positions without dynamic slicing
    counts_at_starts = jnp.where(is_group_start, counts_by_gid[gid], 0)

    return {
        "order": order,
        "states_sorted": states_sorted,
        "errs_sorted": errs_sorted,
        "codes_sorted": codes_sorted,
        "is_group_start": is_group_start,
        "counts_at_starts": counts_at_starts,
        "num_groups": G,
    }


# =============================================================================
# SECTION 7: CONVENIENCE WRAPPERS
# =============================================================================
from typing import NamedTuple

class GroupedSamples(NamedTuple):
    """JAX-jittable grouped output (static shapes).

    Shapes:
      - ideal_state: (D,)
      - ideal_multiplicity: () int32
      - nonideal_states: (S, D)  (states_sorted padded; use nonideal_mask)
      - nonideal_multiplicities: (S,) int32  (counts at group starts; 0 elsewhere)
      - nonideal_errors: (S,) int32  (errors aligned with states_sorted)
      - nonideal_mask: (S,) bool  (True at non-ideal group starts)
    """
    ideal_state: Array
    ideal_multiplicity: Array
    nonideal_states: Array
    nonideal_multiplicities: Array
    nonideal_errors: Array
    nonideal_mask: Array


def generate_efficient_noisy_samples_arrays(key: Array, fs: Array, nq: int, channel_config: Dict[str, Any], num_samples: int) -> Dict[str, Array]:
    """JAX-native arrays API: sample then group by jump codes.

    Returns dict of arrays with static shapes (see `group_trajectories_jax`).
    """
    states, errs, codes = sample_trajectories(key, fs, nq, channel_config, num_samples)
    return group_trajectories_jax(states, errs, codes)


def generate_efficient_noisy_samples_named(
    key: Array,
    fs: Array,
    nq: int,
    channel_config: Dict[str, Any],
    num_samples: int,
) -> GroupedSamples:
    """Fully JAX-jittable wrapper returning a NamedTuple with static shapes.

    It exposes ideal and non-ideal groups without host compression. Non-ideal
    outputs are size-S arrays with a boolean mask indicating which entries are
    real group starts. This structure is a pytree and jittable/vmappable.
    """
    grouped = generate_efficient_noisy_samples_arrays(key, fs, nq, channel_config, num_samples)
    states_sorted = grouped["states_sorted"]
    errs_sorted = grouped["errs_sorted"]
    counts_at_starts = grouped["counts_at_starts"]
    is_start = grouped["is_group_start"]

    # Identify ideal and non-ideal group-starts
    start_is_ideal = is_start & (errs_sorted == 0)
    start_is_nonideal = is_start & (errs_sorted > 0)

    # Get ideal multiplicity and state; if no ideal present, multiplicity=0 and fall back to psi_F
    ideal_idx = jnp.argmax(start_is_ideal.astype(jnp.int32))
    has_ideal = jnp.any(start_is_ideal)
    ideal_mult = jnp.where(has_ideal, counts_at_starts[ideal_idx], jnp.int32(0))
    psi0 = generate_psi_F_vector(fs, nq)
    ideal_state = jnp.where(has_ideal, states_sorted[ideal_idx], psi0)

    # Non-ideal outputs are aligned to S with a mask
    nonideal_states = states_sorted
    nonideal_multiplicities = counts_at_starts
    nonideal_errors = errs_sorted
    nonideal_mask = start_is_nonideal

    return GroupedSamples(
        ideal_state=ideal_state,
        ideal_multiplicity=ideal_mult,
        nonideal_states=nonideal_states,
        nonideal_multiplicities=nonideal_multiplicities,
        nonideal_errors=nonideal_errors,
        nonideal_mask=nonideal_mask,
    )


def expand_grouped_to_megabatch_from_arrays(grouped: Dict[str, Array]) -> Array:
    """JAX-jittable expansion to (S, D) mega-batch by run-length decoding.

    Uses cumulative counts + searchsorted so output shape is static (S, D) and
    no dynamic slicing is needed. This replicates *group representatives* (at
    group starts) by their multiplicities.

    Args:
        grouped: dict returned by generate_efficient_noisy_samples_arrays

    Returns:
        mega: (S, D) complex array, where S is the original number of samples.
    """
    states_sorted = grouped["states_sorted"]        # (S, D)
    counts_at_starts = grouped["counts_at_starts"]  # (S,)
    S = counts_at_starts.shape[0]
    prefix = jnp.cumsum(counts_at_starts)
    pos = jnp.arange(S, dtype=jnp.int32)
    idx = jnp.searchsorted(prefix, pos, side="right")
    mega = states_sorted[idx]
    return mega


def expand_grouped_to_megabatch_named(gs: GroupedSamples) -> Array:
    """Same expansion as above but takes the NamedTuple output.

    Returns:
        mega: (S, D) with ideal and non-ideal states replicated by multiplicity.
    """
    states_sorted = gs.nonideal_states              # (S, D)
    counts_at_starts = gs.nonideal_multiplicities   # (S,)
    S = counts_at_starts.shape[0]
    prefix = jnp.cumsum(counts_at_starts)
    pos = jnp.arange(S, dtype=jnp.int32)
    idx = jnp.searchsorted(prefix, pos, side="right")
    mega = states_sorted[idx]
    return mega

# =============================================================================
# SECTION 7.1: FULLY-JIT SHADOWS OVER ALL TRAJECTORIES (NO MEGA MATERIALIZATION)
# =============================================================================

def _sample_trajectories_with_Ks(key: Array, fs: Array, nq: int, Ks: Array, num_samples: int) -> Tuple[Array, Array, Array]:
    """JAX-friendly sampler that takes pre-built Kraus array Ks.

    Uses fold_in instead of split so `num_samples` needn't be a Python static, though
    shapes are still determined by it.
    """
    psi0 = generate_psi_F_vector(fs, nq)
    # Build per-trajectory subkeys via fold_in (works under jit without static num)
    idxs = jnp.arange(num_samples, dtype=jnp.uint32)
    keys = jax.vmap(lambda i: jr.fold_in(key, i))(idxs)
    one = lambda k: trajectory_from_key_jit(k, psi0, nq, Ks)
    states, errs, codes = jax.vmap(one)(keys)
    return states, errs, codes


def generate_efficient_noisy_samples_arrays_with_Ks(key: Array, fs: Array, nq: int, Ks: Array, num_samples: int) -> Dict[str, Array]:
    states, errs, codes = _sample_trajectories_with_Ks(key, fs, nq, Ks, num_samples)
    return group_trajectories_jax(states, errs, codes)


@partial(jax.jit, static_argnames=("nq", "num_samples", "batch_size", "r", "shots"))
def mcs_shadows_all_jit(
    main_key: Array,
    fs: Array,
    nq: int,
    Ks: Array,
    num_samples: int,
    batch_size: int,
    r: int,
    shots: int,
):
    """Compute mean classical shadow over *all* trajectories under one jit.

    Avoids materializing the (S,D) mega-batch by using a run-length decoding index map
    and a fixed-size batched loop (lax.fori_loop). Requires that jit_vectorized_shadow_computation
    is available in scope and accepts (states: (B,D), keys: (B,2), n, r, shots) -> (B,D,D).
    """
    grouped = generate_efficient_noisy_samples_arrays_with_Ks(main_key, fs, nq, Ks, num_samples)
    states_sorted = grouped["states_sorted"]           # (S, D)
    counts_at_starts = grouped["counts_at_starts"]     # (S,)
    S = states_sorted.shape[0]
    D = states_sorted.shape[1]

    # Build index map pos->rep-start via cumulative counts
    pos = jnp.arange(S, dtype=jnp.int32)
    prefix = jnp.cumsum(counts_at_starts)
    idx = jnp.searchsorted(prefix, pos, side="right")     # (S,)

    # Generate one PRNG key per trajectory via fold_in
    traj_keys = jax.vmap(lambda i: jr.fold_in(main_key, i))(jnp.arange(S, dtype=jnp.uint32))  # (S,2)

    # Pad to a multiple of batch_size for static slicing inside the loop
    num_batches = (S + batch_size - 1) // batch_size
    P = num_batches * batch_size
    pad = P - S

    def pad1d(x, val):
        return jnp.pad(x, (0, pad), constant_values=val)

    def pad2d(x, val_row):
        return jnp.pad(x, ((0, pad), (0, 0)), constant_values=0).at[S:].set(val_row)

    # For indices, repeat the last valid index; for keys, repeat the last key; for mask, 1 then 0s
    last_idx = jnp.where(S > 0, idx[-1], jnp.int32(0))
    idx_padded = jnp.pad(idx, (0, pad), constant_values=0)
    idx_padded = idx_padded.at[S:].set(last_idx)

    last_key = jnp.where(S > 0, traj_keys[-1], jr.PRNGKey(0))
    keys_padded = jnp.pad(traj_keys, ((0, pad), (0, 0)), constant_values=0)
    keys_padded = keys_padded.at[S:].set(last_key)

    mask = jnp.concatenate([jnp.ones((S,), dtype=jnp.float32), jnp.zeros((pad,), dtype=jnp.float32)])  # (P,)

    def body(i, acc):
        start = i * batch_size
        idx_slice = lax.dynamic_slice(idx_padded, (start,), (batch_size,))
        keys_slice = lax.dynamic_slice(keys_padded, (start, 0), (batch_size, 2))
        mask_slice = lax.dynamic_slice(mask, (start,), (batch_size,))
        states_batch = states_sorted[idx_slice]  # (B, D)
        # Call user's shadow kernel
        shadows = jit_vectorized_shadow_computation(states_batch, keys_slice, nq, r, shots)  # (B, D, D)
        # Mask padded rows and accumulate
        acc = acc + (shadows * mask_slice[:, None, None]).sum(axis=0)
        return acc

    init = jnp.zeros((D, D), dtype=jnp.complex128)
    total = lax.fori_loop(0, num_batches, body, init)
    return total / jnp.maximum(S, 1)


@partial(jax.jit, static_argnames=("nq","num_samples","batch_size","r","shots","weights_kind", "use_complex64"))
def mcs_shadows_streaming_jit(
    main_key,
    fs,
    nq: int,
    Ks,
    num_samples: int,
    batch_size: int,
    r: int,
    shots: int,
    *,
    weights_kind: int = 0,
    proposal_p: float = 0.0,
    target_p: float = 0.0,
    m_susceptible: int = 0,
    use_complex64: bool = False,
):
    """Memory-efficient fully-jitted pipeline with static dtype choice.

    Fixes dtype promotion by ensuring the shadow kernel output is cast to the
    accumulator dtype, and weights use a matching real dtype to avoid upcasts.
    """
    psi0 = generate_psi_F_vector(fs, nq)
    Ks_local = Ks
    if use_complex64:
        psi0 = psi0.astype(jnp.complex64)
        Ks_local = Ks.astype(jnp.complex64)
    # Choose dtypes for accumulation
    dtype_c = psi0.dtype
    dtype_r = jnp.float32 if dtype_c == jnp.complex64 else jnp.float64

    D = psi0.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    def batch_keys(i):
        start = i * batch_size
        idx = start + jnp.arange(batch_size, dtype=jnp.uint32)
        idx = jnp.minimum(idx, jnp.uint32(max(num_samples - 1, 0)))
        return jax.vmap(lambda j: jr.fold_in(main_key, j))(idx)

    def compute_logw(errs_batch):
        # compute in float64 for stability then cast down to dtype_r after exp
        k = errs_batch.astype(jnp.float64)
        def _zeros(x): return jnp.zeros_like(x)
        def _binom_nq(x):
            eps = 1e-12
            ph = jnp.clip(jnp.float64(proposal_p), eps, 1 - eps)
            pl = jnp.clip(jnp.float64(target_p),   eps, 1 - eps)
            nqf = jnp.float64(nq)
            return x*(jnp.log(pl)-jnp.log(ph)) + (nqf-x)*(jnp.log1p(-pl)-jnp.log1p(-ph))
        def _ad_const_m(x):
            eps = 1e-12
            gh = jnp.clip(jnp.float64(proposal_p), eps, 1 - eps)
            gl = jnp.clip(jnp.float64(target_p),   eps, 1 - eps)
            m  = jnp.float64(m_susceptible)
            return x*(jnp.log(gl)-jnp.log(gh)) + (m-x)*(jnp.log1p(-gl)-jnp.log1p(-gh))
        logw64 = lax.switch(weights_kind, (_zeros,_binom_nq,_ad_const_m), k)
        return jnp.exp(logw64).astype(dtype_r)

    def body(i, carry):
        acc_mat, acc_w = carry
        keys = batch_keys(i)
        def one(k):
            psif, err, _ = trajectory_from_key_jit(k, psi0, nq, Ks_local)
            return psif, err
        psis, errs = jax.vmap(one)(keys)
        start = i * batch_size
        valid = jnp.minimum(batch_size, num_samples - start)
        mask = (jnp.arange(batch_size) < valid).astype(dtype_r)
        shadows = jit_vectorized_shadow_computation(psis, keys, nq, r, shots)
        # Cast shadows to match accumulator complex dtype to avoid c128 upcast
        shadows = shadows.astype(dtype_c)
        w = compute_logw(errs) * mask
        acc_mat = acc_mat + (shadows * w[:,None,None]).sum(axis=0)
        acc_w = acc_w + w.sum()
        return (acc_mat, acc_w)

    init = (jnp.zeros((D,D), dtype=dtype_c), dtype_r(0.0))
    total_mat, total_w = lax.fori_loop(0, num_batches, body, init)
    total_w = jnp.maximum(total_w, dtype_r(1.0))
    return total_mat / total_w




@partial(jax.jit, static_argnames=("nq","num_samples","batch_size","r","shots","weights_kind", "use_complex64"))
def mcs_shadows_streaming_jit_try1(
    main_key,
    fs,
    nq: int,
    Ks,
    num_samples: int,
    batch_size: int,
    r: int,
    shots: int,
    *,
    weights_kind: int = 0,      # 0: equal; 1: binomial(nq); 2: AD/thermal with constant m
    proposal_p: float = 0.0,
    target_p: float = 0.0,
    m_susceptible: int = 0,
    use_complex64: bool = False,
):
    """Memory-efficient fully-jitted pipeline with per-sample accumulation.

    Never materializes (B, D, D). Accumulates the weighted shadow matrix
    one sample at a time inside the batch fori_loop.
    """
    psi0 = generate_psi_F_vector(fs, nq)
    Ks_local = Ks
    if use_complex64:
        psi0 = psi0.astype(jnp.complex64)
        Ks_local = Ks.astype(jnp.complex64)

    # Accumulator dtypes
    dtype_c = psi0.dtype
    dtype_r = jnp.float32 if dtype_c == jnp.complex64 else jnp.float64

    D = psi0.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    def batch_keys(i):
        start = i * batch_size
        idx = start + jnp.arange(batch_size, dtype=jnp.uint32)
        idx = jnp.minimum(idx, jnp.uint32(jnp.maximum(num_samples - 1, 0)))
        return jax.vmap(lambda j: jr.fold_in(main_key, j))(idx)  # (B,2)

    def compute_logw(errs_batch):
        # compute log-weights in float64 for stability
        k = errs_batch.astype(jnp.float64)
        def _zeros(x): return jnp.zeros_like(x)
        def _binom_nq(x):
            eps = 1e-12
            ph = jnp.clip(jnp.float64(proposal_p), eps, 1 - eps)
            pl = jnp.clip(jnp.float64(target_p),   eps, 1 - eps)
            nqf = jnp.float64(nq)
            return x*(jnp.log(pl)-jnp.log(ph)) + (nqf-x)*(jnp.log1p(-pl)-jnp.log1p(-ph))
        def _ad_const_m(x):
            eps = 1e-12
            gh = jnp.clip(jnp.float64(proposal_p), eps, 1 - eps)
            gl = jnp.clip(jnp.float64(target_p),   eps, 1 - eps)
            m  = jnp.float64(m_susceptible)  # TODO: per-rep m
            return x*(jnp.log(gl)-jnp.log(gh)) + (m-x)*(jnp.log1p(-gl)-jnp.log1p(-gh))
        logw64 = lax.switch(weights_kind, (_zeros, _binom_nq, _ad_const_m), k)
        return jnp.exp(logw64).astype(dtype_r)

    def body(i, carry):
        acc_mat, acc_w = carry
        keys = batch_keys(i)  # (B,2)

        # Simulate a batch of trajectories
        def one(k):
            psif, err, _ = trajectory_from_key_jit(k, psi0, nq, Ks_local)
            return psif, err
        psis, errs = jax.vmap(one)(keys)  # (B,D), (B,)

        # Mask for last (possibly partial) batch
        start = i * batch_size
        valid = jnp.minimum(batch_size, num_samples - start)
        mask  = (jnp.arange(batch_size) < valid).astype(dtype_r)  # (B,)

        # Importance weights
        w = compute_logw(errs) * mask  # (B,)

        # Accumulate per sample (no (B,D,D) materialization)
        def add_one(j, A):
            rho_j = get_shadow_rho_noiseless_pure(psis[j], keys[j], nq, r, shots)  # (D,D), complex128 by TC default
            rho_j = rho_j.astype(dtype_c)  # match accumulator dtype
            return A + rho_j * w[j]

        acc_mat = lax.fori_loop(0, batch_size, add_one, acc_mat)
        acc_w   = acc_w + w.sum()
        return (acc_mat, acc_w)

    init = (jnp.zeros((D, D), dtype=dtype_c), dtype_r(0.0))
    total_mat, total_w = lax.fori_loop(0, num_batches, body, init)
    total_w = jnp.maximum(total_w, dtype_r(1.0))
    print(nq)
    return total_mat / total_w


@partial(jax.jit, static_argnames=("nq","num_samples","batch_size","mini_batch_size","r","shots","weights_kind","use_complex64"))
def mcs_shadows_streaming_2stage_jit(
    main_key,
    fs,
    nq: int,
    Ks,
    num_samples: int,
    batch_size: int,
    mini_batch_size: int,
    r: int,
    shots: int,
    *,
    weights_kind: int = 0,
    proposal_p: float = 0.0,
    target_p: float = 0.0,
    m_susceptible: int = 0,
    use_complex64: bool = False,
):
    """Two-stage streaming MCS+shadows.

    - Outer loop (size `batch_size`): simulate a block of trajectories -> (B,D) psis, (B,) errs, (B,2) keys.
    - Inner loop (size `mini_batch_size`): compute shadows for a small vectorized chunk
      to leverage GPU parallelism without materializing huge (B,D,D).

    Memory model (approx, complex64):
      psi buffer ~ B * D * 8 bytes
      shadow chunk ~ M * D * D * 8 bytes
    Choose B, M so the sum stays well below VRAM.
    """
    psi0 = generate_psi_F_vector(fs, nq)
    Ks_local = Ks
    if use_complex64:
        psi0 = psi0.astype(jnp.complex64)
        Ks_local = Ks.astype(jnp.complex64)
    dtype_c = psi0.dtype
    dtype_r = jnp.float32 if dtype_c == jnp.complex64 else jnp.float64

    D = psi0.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    inner_steps = (batch_size + mini_batch_size - 1) // mini_batch_size

    def batch_keys(i):
        start = i * batch_size
        idx = start + jnp.arange(batch_size, dtype=jnp.uint32)
        idx = jnp.minimum(idx, jnp.uint32(jnp.maximum(num_samples - 1, 0)))
        return jax.vmap(lambda j: jr.fold_in(main_key, j))(idx)  # (B,2)

    def compute_logw(errs_batch):
        k = errs_batch.astype(jnp.float64)
        def _zeros(x): return jnp.zeros_like(x)
        def _binom_nq(x):
            eps = 1e-12
            ph = jnp.clip(jnp.float64(proposal_p), eps, 1 - eps)
            pl = jnp.clip(jnp.float64(target_p),   eps, 1 - eps)
            nqf = jnp.float64(nq)
            return x*(jnp.log(pl)-jnp.log(ph)) + (nqf-x)*(jnp.log1p(-pl)-jnp.log1p(-ph))
        def _ad_const_m(x):
            eps = 1e-12
            gh = jnp.clip(jnp.float64(proposal_p), eps, 1 - eps)
            gl = jnp.clip(jnp.float64(target_p),   eps, 1 - eps)
            m  = jnp.float64(m_susceptible)
            return x*(jnp.log(gl)-jnp.log(gh)) + (m-x)*(jnp.log1p(-gl)-jnp.log1p(-gh))
        logw64 = lax.switch(weights_kind, (_zeros,_binom_nq,_ad_const_m), k)
        return jnp.exp(logw64).astype(dtype_r)

    def outer_body(i, carry):
        acc_mat, acc_w = carry
        keysB = batch_keys(i)  # (B,2)

        # Simulate a batch of trajectories
        def one(k):
            psif, err, _ = trajectory_from_key_jit(k, psi0, nq, Ks_local)
            return psif, err
        psisB, errsB = jax.vmap(one)(keysB)  # (B,D), (B,)

        start = i * batch_size
        validB = jnp.minimum(batch_size, num_samples - start)
        maskB  = (jnp.arange(batch_size) < validB).astype(dtype_r)
        wB = compute_logw(errsB) * maskB

        def inner_body(j, A):
            acc_mat_inner, acc_w_inner = A
            startM = j * mini_batch_size
            keysM = lax.dynamic_slice(keysB, (startM, 0), (mini_batch_size, 2))
            psisM = lax.dynamic_slice(psisB, (startM, 0), (mini_batch_size, D))
            wM    = lax.dynamic_slice(wB,   (startM,),   (mini_batch_size,))

            # Vectorized shadow computation on the mini batch
            rhosM = jit_vectorized_shadow_computation(psisM, keysM, nq, r, shots)  # (M,D,D) c128 by TC default
            rhosM = rhosM.astype(dtype_c)
            acc_mat_inner = acc_mat_inner + (rhosM * wM[:, None, None]).sum(axis=0)
            acc_w_inner   = acc_w_inner + wM.sum()
            return (acc_mat_inner, acc_w_inner)

        acc_mat, acc_w = lax.fori_loop(0, inner_steps, inner_body, (acc_mat, acc_w))
        return (acc_mat, acc_w)

    init = (jnp.zeros((D, D), dtype=dtype_c), dtype_r(0.0))
    total_mat, total_w = lax.fori_loop(0, num_batches, outer_body, init)
    total_w = jnp.maximum(total_w, dtype_r(1.0))
    return total_mat / total_w


def generate_efficient_noisy_samples(
    key: Array,
    fs: Array,
    nq: int,
    channel_config: Dict[str, Any],
    num_samples: int,
) -> Dict[str, Any]:
    """Non-jitted convenience wrapper that formats to the original dict schema.

    Heavy compute remains device-side; we only compress groups on host.
    Schema:
      {
        "ideal_state": {"state_vector": jnp.ndarray, "multiplicity": int, "num_errors": 0},
        "noisy_realizations": [ {"state_vector": jnp.ndarray, "multiplicity": int, "num_errors": int}, ... ]
      }
    """
    grouped = generate_efficient_noisy_samples_arrays(key, fs, nq, channel_config, num_samples)

    states_sorted = grouped["states_sorted"]
    errs_sorted = grouped["errs_sorted"]
    codes_sorted = grouped["codes_sorted"]
    is_start = grouped["is_group_start"]
    counts_at_starts = grouped["counts_at_starts"]

    # Move to host for variable-length compression
    states_h = jax.device_get(states_sorted)
    errs_h = jax.device_get(errs_sorted)
    starts_h = jax.device_get(is_start)
    counts_h = jax.device_get(counts_at_starts)

    out_noisy: List[Dict[str, Any]] = []
    ideal_state = None

    idx = 0
    while idx < len(states_h):
        if starts_h[idx]:
            count = int(counts_h[idx])
            if count == 0:
                idx += 1
                continue
            state_vec = states_h[idx]
            num_errors = int(errs_h[idx])
            if num_errors == 0:
                ideal_state = {"state_vector": jnp.asarray(state_vec), "multiplicity": count, "num_errors": 0}
            else:
                out_noisy.append({
                    "state_vector": jnp.asarray(state_vec),
                    "multiplicity": count,
                    "num_errors": num_errors,
                })
            idx += count
        else:
            idx += 1

    if ideal_state is None:
        ideal_state = {
            "state_vector": generate_psi_F_vector(fs, nq),
            "multiplicity": 0,
            "num_errors": 0,
        }

    return {
        "ideal_state": ideal_state,
        "noisy_realizations": out_noisy,
    }



# if __name__ == "__main__":
#     key = jr.PRNGKey(7)
#     N = 7
#     dim = 1 << N
#     F = (jnp.arange(dim) & 1).astype(jnp.int32)

    # # --- Test 1: bit-flip p=0 should yield zero errors and identical states ---
    # cfg0 = {"type": "bit_flip", "strength": 0.0}
    # S = 16
    # states0, errs0, codes0 = sample_trajectories(key, F, N, cfg0, num_samples=S)
    # assert states0.shape == (S, dim)
    # assert (errs0 == 0).all(), "Errors should be zero when p=0"
    # psi_ref = generate_psi_F_vector(F, N)
    # assert jnp.allclose(states0, jnp.tile(psi_ref, (S,1)))
    # grouped0 = group_trajectories_jax(states0, errs0, codes0)
    # # Only one group (all codes identical)
    # assert int(grouped0["num_groups"]) == 1
    # assert int(jnp.sum(grouped0["counts_at_starts"])) == S

    # # --- Test 2: dephasing p>0 yields multiple groups and some errors ---
    # cfg1 = {"type": "dephasing", "strength": 0.3}
    # S = 256
    # states1, errs1, codes1 = sample_trajectories(key, F, N, cfg1, num_samples=S)
    # grouped1 = group_trajectories_jax(states1, errs1, codes1)
    # assert states1.shape == (S, dim)
    # assert int(jnp.sum(errs1 > 0)) > 0
    # # counts must sum back to S
    # assert int(jnp.sum(grouped1["counts_at_starts"])) == S

    # # --- Test 3: VMAP over keys ---
    # keys = jr.split(key, 32)
    # s, e, c = sample_trajectories_batched(keys, F, N, cfg1)
    # assert s.shape == (32, dim)
    # assert e.shape == (32,)
    # assert c.shape == (32,)

    # # --- Test 4: JIT end-to-end grouping arrays API ---
    # def arrays_api(k, ns):
    #     return generate_efficient_noisy_samples_arrays(k, F, N, cfg1, num_samples=ns)
    # arrays_api_jit = jax.jit(arrays_api, static_argnames=("ns",))
    # grouped_j = arrays_api_jit(key, 128)
    # assert int(jnp.sum(grouped_j["counts_at_starts"])) == 128

    # # --- Test 5: Dict wrapper sanity and accounting ---
    # out = generate_efficient_noisy_samples(key, F, N, cfg1, num_samples=128)
    # total = out["ideal_state"]["multiplicity"] + sum(x["multiplicity"] for x in out["noisy_realizations"])
    # assert total == 128

    # # --- Test 6: relaxation with gamma=0 behaves like identity (no errors) ---
    # cfg_rel0 = {"type": "relaxation", "strength": 0.0, "p_exc": 0.25}
    # states_rel0, errs_rel0, codes_rel0 = sample_trajectories(key, F, N, cfg_rel0, num_samples=16)
    # assert int(jnp.sum(errs_rel0)) == 0

    # --- Test 7: MCS vs DMS convergence (requires tensorcircuit) ---
    # try:
    #     import tensorcircuit as tc
    #     from tensorcircuit import backend as tc_backend
    #     tc.set_backend("jax")

    #     def operator_2_norm(R):
    #         """
    #         Calculate the operator 2-norm.
        
    #         Args:
    #             R (array): The operator whose norm we want to calculate.
        
    #         Returns:
    #             Scalar corresponding to the norm.
    #         """
    #         return jnp.sqrt(jnp.trace(R.conjugate().transpose() @ R))

    #     def dms_after_channel(F_vals: Array, nq: int, cfg: Dict[str, Any]) -> Array:
    #         Ks = get_kraus_operators(cfg)
    #         psi0 = generate_psi_F_vector(F_vals, nq)
    #         rho0 = jnp.outer(psi0, jnp.conj(psi0))
    #         rho_tc = tc.array_to_tensor(rho0)
    #         Ks_tc = [tc.array_to_tensor(Ks[k]) for k in range(Ks.shape[0])]
    #         dmc = tc.DMCircuit(nq, dminputs=rho_tc)
    #         for i in range(nq):
    #             dmc.general_kraus(Ks_tc, i)
    #         rho_final_tc = dmc.state()
    #         return jnp.asarray(tc_backend.numpy(rho_final_tc)), rho_tc

    #     def mcs_density(states: Array) -> Array:
    #         # Sum_s |psi_s><psi_s| / S  ==  (states^H @ states) / S
    #         S = states.shape[0]
    #         return (states.conj().T @ states) / S

    #     # Compare over increasing S: error should decrease
    #     for cfg in (
    #         {"type": "dephasing", "strength": 0.1},
    #         {"type": "bit_flip", "strength": 0.2},
    #         {"type": "relaxation", "strength": 0.1, "p_exc": 0.00},
    #     ):
    #         rho_dms, rho = dms_after_channel(F, N, cfg)
    #         denom = jnp.linalg.norm(rho_dms)
    #         denom_rho = jnp.linalg.norm(rho)
    #         prev_err = None
    #         for S in (1, 64, 256, 1024, 4096):
    #             statesS, _, _ = sample_trajectories(key, F, N, cfg, num_samples=S)
    #             rho_mcs = mcs_density(statesS)
    #             # print(rho_mcs)
    #             # err = jnp.linalg.norm(rho_mcs - rho_dms) / (denom + 1e-12)
    #             # err_rho = jnp.linalg.norm(rho - rho_dms) / (denom_rho + 1e-12)
    #             err_mcs = operator_2_norm(rho_dms - rho_mcs)
    #             err_dms = operator_2_norm(rho - rho_dms)
    #             print()
    #             print(err_mcs)
    #             print(err_dms)
    #             print()
    #             if prev_err is not None:
    #                 assert err <= prev_err + 5e-3, f"MCS error did not decrease: {err} > {prev_err}"
    #             prev_err = err
    # except Exception as _e:
    #     # If tensorcircuit isn't available, we skip this test silently.
    #     pass

    # print("All tests passed.")


# import numpy as np
# import time
# # Prepare channel & Kraus (outside jit)
# channel_config = {"type": "dephasing", "strength": 0.2}
# Ks = get_kraus_operators(channel_config)

# # Compile the end-to-end JIT


# # Inputs
# key = jax.random.PRNGKey(0)
# N_range = [7,8,9,10,11,12,13]
# Nf = 10
# # dim = 1 << nq
# # F = (jnp.arange(dim) & 1).astype(jnp.int32)

# # Params
# num_samples = 4**N    # S
# batch_size  = 1000     # B (tune for memory)
# r = 1
# shots = 1
# s_time = time.time()

# for nq in N_range:
#     F_matrix = np.random.randint(2, size=(Nf, 2**nq))  # RANDOM TODO: pseudorandom
#     for F_vec in F_matrix:
#         # psi_f = generate_psi_F_vector(F_vec, nq)
#         # rho_f = jnp.outer(psi_f, jnp.conjugate(psi_f))
#         shadow = mcs_shadows_all_jit(key, F_vec, nq, Ks, num_samples, batch_size, r, shots)
#         print(f"{time.time() - s_time} N ={nq}")
#         s_time = time.time()


# key = jr.PRNGKey(123)

# # Helper: density matrix from batch of pure states
# def density_from_states(states):
#     S = states.shape[0]
#     return (states.conj().T @ states) / jnp.maximum(S, 1)

# # Settings
# N_range = [7,8,9,10]
# Nf = 10

# for N in N_range:
#     dim = 1 << N
#     F_mat = np.random.randint(2, size=(Nf, 2**N))
#     S = int(3**N)  # number of trajectories

#     for F in F_mat:
#         # --- Test A: Dephasing with p ~ 1 -> I/2^N ---
#         cfg_dep_hi = {"type": "dephasing", "strength": 0.999999}
#         Ks_dep = get_kraus_operators(cfg_dep_hi)
        
#         # Sample trajectories (states only)
#         psi0 = generate_psi_F_vector(F, N)
#         keys = jr.split(key, S)
#         one = lambda k: trajectory_from_key_jit(k, psi0, N, Ks_dep)
#         states_dep, _, _ = jax.vmap(one)(keys)
#         rho_mcs_dep = density_from_states(states_dep)
        
#         rho_target_mm = jnp.eye(dim, dtype=jnp.complex128) / dim
#         err_dep = jnp.linalg.norm(rho_mcs_dep - rho_target_mm) / jnp.linalg.norm(rho_target_mm)
#         # With S=4096 and p~1, this should be tiny c_dep * sqrt(2^N / S) where c_dep is a constant, we use c_dep = 1.5.
#         assert float(err_dep) < 3/2 * jnp.sqrt(2**N / S), f"Dephasing high-noise not maximally mixed enough: relerr={float(err_dep):.3e}"
        
#         # --- Test B: Relaxation (gamma -> 1, p_exc=0) -> |0...0><0...0| ---
#         cfg_rel_hi = {"type": "relaxation", "strength": 0.999999, "p_exc": 0.0}
#         Ks_rel = get_kraus_operators(cfg_rel_hi)
        
#         one_rel = lambda k: trajectory_from_key_jit(k, psi0, N, Ks_rel)
#         states_rel, _, _ = jax.vmap(one_rel)(keys)
#         rho_mcs_rel = density_from_states(states_rel)
        
#         # |0...0> projector
#         e0 = jnp.zeros((dim,), dtype=jnp.complex128).at[0].set(1.0)
#         rho_target_rel = jnp.outer(e0, jnp.conj(e0))
#         err_rel = jnp.linalg.norm(rho_mcs_rel - rho_target_rel) / jnp.linalg.norm(rho_target_rel)
#         assert float(err_rel) < 3 * jnp.sqrt(2**N / S), f"Relaxation high-noise not ground state enough: relerr={float(err_rel):.3e}" # tiny c_dep * sqrt(2^N / S) where c_dep is a constant, we use c_dep = 3.
        
#         print(f"Trajectory pipeline tests passed: dephasing->I/2^N and relaxation->|0...0><0...0| N={N}.")


def generate_all_nps_checkpoints(main_key, fs, nq, Ks, max_nps, batch_size, r, shots, 
                                  checkpoint_nps_list, use_complex64=False):
    """Generate shadow DMs at multiple nps checkpoints efficiently.
    
    This generates shadows ONCE at max_nps and checkpoints during accumulation.
    This is much more efficient than calling the function multiple times.
    
    Args:
        main_key: JAX random key
        fs: F function values  
        nq: number of qubits
        Ks: Kraus operators
        max_nps: maximum nps to use
        batch_size: batch size for MC
        r: shadow parameter
        shots: shots per snapshot
        checkpoint_nps_list: sorted list of nps checkpoints (must be <= max_nps)
        use_complex64: use float32/complex64
        
    Returns:
        dict mapping nps -> DM matrix
    """
    # Sort checkpoints and filter to <= max_nps
    checkpoint_nps_list = sorted([nps for nps in checkpoint_nps_list if nps <= max_nps])
    
    if len(checkpoint_nps_list) == 0:
        return {}
    
    # Use the checkpointing version that generates once and checkpoints during accumulation
    return mcs_shadows_with_checkpoints_python(
        main_key, fs, nq, Ks, max_nps, batch_size, r, shots,
        checkpoint_nps_list,
        use_complex64=use_complex64
    )


def mcs_shadows_with_checkpoints_python(
    main_key,
    fs,
    nq: int,
    Ks,
    max_samples: int,
    batch_size: int,
    r: int,
    shots: int,
    checkpoint_nps_list,
    *,
    use_complex64: bool = False,
):
    """
    Generate shadow with checkpoints by generating ONCE at max_samples
    and checkpointing during accumulation.
    
    This Python wrapper manually tracks progress and checkpoints the accumulated DM.
    
    Returns:
        dict mapping checkpoint nps (as int) -> accumulated DM
    """
    psi0 = generate_psi_F_vector(fs, nq)
    Ks_local = Ks
    if use_complex64:
        psi0 = psi0.astype(jnp.complex64)
        Ks_local = Ks.astype(jnp.complex64)
    
    dtype_c = psi0.dtype
    dtype_r = jnp.float32 if dtype_c == jnp.complex64 else jnp.float64
    D = psi0.shape[0]
    
    num_batches = (max_samples + batch_size - 1) // batch_size
    
    def batch_keys(i):
        start = i * batch_size
        idx = start + jnp.arange(batch_size, dtype=jnp.uint32)
        idx = jnp.minimum(idx, jnp.uint32(jnp.maximum(max_samples - 1, 0)))
        return jax.vmap(lambda j: jr.fold_in(main_key, j))(idx)
    
    def one(k):
        psif, err, _ = trajectory_from_key_jit(k, psi0, nq, Ks_local)
        return psif, err
    
    # Initialize accumulation
    acc_mat = jnp.zeros((D, D), dtype=dtype_c)
    acc_w = dtype_r(0.0)
    samples_done = 0
    
    # Track which checkpoints we've passed
    checkpoint_results = {}
    checkpoint_idx = 0
    sorted_checkpoints = sorted(checkpoint_nps_list)
    
    # Process batches and checkpoint as we go
    for i in range(num_batches):
        keys = batch_keys(i)
        psis, errs = jax.vmap(one)(keys)
        
        start = i * batch_size
        valid = int(jnp.minimum(batch_size, max_samples - start))
        mask = (jnp.arange(batch_size) < valid).astype(dtype_r)
        
        shadows = jit_vectorized_shadow_computation(psis, keys, nq, r, shots)
        shadows = shadows.astype(dtype_c)
        w = mask  # Equal weights
        
        # Process this batch incrementally, potentially splitting at checkpoint boundaries
        batch_start_idx = 0
        while batch_start_idx < valid:
            # Find the next checkpoint that falls within the remaining part of this batch
            next_checkpoint = None
            samples_to_process = valid - batch_start_idx
            
            if checkpoint_idx < len(sorted_checkpoints):
                checkpoint_nps = sorted_checkpoints[checkpoint_idx]
                samples_needed = checkpoint_nps - samples_done
                
                if samples_needed > 0 and samples_needed <= samples_to_process:
                    # This checkpoint falls within this batch - process up to it
                    next_checkpoint = checkpoint_nps
                    samples_to_process = samples_needed
            
            # Process a sub-batch (either up to checkpoint or rest of batch)
            sub_batch_mask = (jnp.arange(batch_size) >= batch_start_idx) & (jnp.arange(batch_size) < batch_start_idx + samples_to_process)
            sub_batch_mask = sub_batch_mask.astype(dtype_r)
            sub_batch_w = w * sub_batch_mask
            
            # Accumulate this sub-batch
            acc_mat = acc_mat + (shadows * sub_batch_w[:, None, None]).sum(axis=0)
            acc_w = acc_w + sub_batch_w.sum()
            samples_done += samples_to_process
            batch_start_idx += samples_to_process
            
            # Checkpoint if we've reached it
            if next_checkpoint is not None:
                # Normalize by exactly checkpoint_nps to get the mean
                # acc_mat and acc_w now contain sum over exactly checkpoint_nps samples
                if acc_w > 0:
                    dm = acc_mat / jnp.maximum(acc_w, dtype_r(1.0))
                    checkpoint_results[int(next_checkpoint)] = dm
                checkpoint_idx += 1
    
    return checkpoint_results


def generate_all_nps_checkpoints_batched(main_key, fs_batch, nq, Ks, max_nps, batch_size_mcs, r, shots,
                                          checkpoint_nps_list, use_complex64=False):
    """Generate shadow DMs at multiple nps checkpoints for multiple samples in parallel.
    
    This uses vmap to process multiple samples simultaneously, which is much faster than
    processing them one at a time.
    
    Args:
        main_key: JAX random key (will be split for each sample)
        fs_batch: (batch_size, 2**nq) array of F function values for multiple samples
        nq: number of qubits
        Ks: Kraus operators
        max_nps: maximum nps to use
        batch_size_mcs: batch size for MC sampling (per sample)
        r: shadow parameter
        shots: shots per snapshot
        checkpoint_nps_list: sorted list of nps checkpoints (must be <= max_nps)
        use_complex64: use float32/complex64
        
    Returns:
        List of dicts, each mapping nps -> DM matrix for one sample
    """
    # Generate different keys for each sample
    num_samples = fs_batch.shape[0]
    keys = jax.random.split(main_key, num_samples)
    
    # Define function to process a single sample
    def process_single_sample(key, fs):
        return generate_all_nps_checkpoints(
            key, fs, nq, Ks, max_nps, batch_size_mcs, r, shots,
            checkpoint_nps_list, use_complex64=use_complex64
        )
    
    # Note: vmap over dictionaries is tricky, so we'll use a regular loop for now
    # but this still allows JAX to parallelize the underlying operations
    results = []
    for i in range(num_samples):
        result = process_single_sample(keys[i], fs_batch[i])
        results.append(result)
    
    return results