"""
Classical shadows functions
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
import pynvml
from typing import Any, Union, Optional, Sequence, Tuple, List
from string import ascii_letters as ABC
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

import tensorcircuit as tc
from tensorcircuit.cons import backend, dtypestr, rdtypestr
from tensorcircuit.circuit import Circuit
from tensorcircuit import DMCircuit
tc.set_backend("jax")
tc.set_dtype("complex128")

Tensor = Any

def get_free_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(jax.local_devices()[0].id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return meminfo.free  # in bytes

def estimate_max_batch_size(n_qubits, dtype='complex64', safety_factor=0.8):
    free_mem = get_free_gpu_memory()
    dim = 2 ** n_qubits
    bytes_per_entry = 8 if dtype == 'complex64' else 16
    size_rho = dim * dim * bytes_per_entry  # bytes per rho
    max_B = int((free_mem * safety_factor) // size_rho)
    return max(1, max_B)

def shadow_snapshots_dm(
    rho: Tensor,
    pauli_strings: Tensor,
    status: Optional[Tensor] = None,
    sub: Optional[Sequence[int]] = None,
    measurement_only: bool = False,
) -> Tensor:
    r"""To generate the shadow snapshots from given pauli string observables on psi

    :param rho: shape = (2 ** nq,2 ** nq), where nq is the number of qubits
    :type: Tensor
    :param pauli_strings: shape = (ns, nq), where ns is the number of pauli strings
    :type: Tensor
    :param status: shape = None or (ns, repeat), where repeat is the times to measure on one pauli string
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]
    :param measurement_only: return snapshots (True) or snapshot states (False), default=False
    :type: bool

    :return snapshots: shape = (ns, repeat, nq) if measurement_only=True otherwise (ns, repeat, nq, 2, 2)
    :rtype: Tensor
    """
    pauli_strings = tc.backend.cast(pauli_strings, dtype="int32") - 1
    ns, nq = pauli_strings.shape
    if 2**nq != rho.shape[0]:
        raise ValueError(
            f"The number of qubits of psi and pauli_strings should be the same, "
            f"but got {nq} and {int(np.log2( rho.shape[0] ))}."
        )
    if status is None:
        status = tc.backend.convert_to_tensor(np.random.rand(ns, 1))
    elif status.shape[0] != ns:
        raise ValueError(f"status.shape[0] should be {ns}, but got {status.shape[0]}.")
    status = tc.backend.cast(status, dtype=rdtypestr)
    repeat = status.shape[1]

    angles = tc.backend.cast(
        tc.backend.convert_to_tensor(
            [
                [-np.pi / 2, np.pi / 4, 0],
                [np.pi / 4, np.pi / 2, 0],
                [0, 0, 0],
            ]
        ),
        dtype=rdtypestr,
    )  # (3, 3)

    def proj_measure(pauli_string: Tensor, st: Tensor) -> Tensor:
        c_ = DMCircuit(nqubits=nq, dminputs=rho)
        for i in range(nq):
            c_.r(  # type: ignore
                i,
                theta=tc.backend.gather1d(
                    tc.backend.gather1d(angles, tc.backend.gather1d(pauli_string, i)), 0
                ),
                alpha=tc.backend.gather1d(
                    tc.backend.gather1d(angles, tc.backend.gather1d(pauli_string, i)), 1
                ),
                phi=tc.backend.gather1d(
                    tc.backend.gather1d(angles, tc.backend.gather1d(pauli_string, i)), 2
                ),
            )
        return c_.sample(batch=repeat, format="sample_bin", allow_state=True, status=st)

    vpm = tc.backend.vmap(proj_measure, vectorized_argnums=(0, 1))
    snapshots = vpm(pauli_strings, status)  # (ns, repeat, nq)
    if measurement_only:
        return snapshots if sub is None else slice_sub(snapshots, sub)
    else:
        return local_snapshot_states(snapshots, pauli_strings + 1, sub)


def shadow_snapshots(
    psi: Tensor,
    pauli_strings: Tensor,
    status: Optional[Tensor] = None,
    sub: Optional[Sequence[int]] = None,
    measurement_only: bool = False,
) -> Tensor:
    r"""To generate the shadow snapshots from given pauli string observables on psi

    :param psi: shape = (2 ** nq,), where nq is the number of qubits
    :type: Tensor
    :param pauli_strings: shape = (ns, nq), where ns is the number of pauli strings
    :type: Tensor
    :param status: shape = None or (ns, repeat), where repeat is the times to measure on one pauli string
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]
    :param measurement_only: return snapshots (True) or snapshot states (False), default=False
    :type: bool

    :return snapshots: shape = (ns, repeat, nq) if measurement_only=True otherwise (ns, repeat, nq, 2, 2)
    :rtype: Tensor
    """
    pauli_strings = tc.backend.cast(pauli_strings, dtype="int32") - 1
    ns, nq = pauli_strings.shape
    if 2**nq != len(psi):
        raise ValueError(
            f"The number of qubits of psi and pauli_strings should be the same, "
            f"but got {nq} and {int(np.log2(len(psi)))}."
        )
    if status is None:
        status = tc.backend.convert_to_tensor(np.random.rand(ns, 1))
    elif status.shape[0] != ns:
        raise ValueError(f"status.shape[0] should be {ns}, but got {status.shape[0]}.")
    status = tc.backend.cast(status, dtype=rdtypestr)
    repeat = status.shape[1]

    angles = tc.backend.cast(
        tc.backend.convert_to_tensor(
            [
                [-np.pi / 2, np.pi / 4, 0],
                [np.pi / 4, np.pi / 2, 0],
                [0, 0, 0],
            ]
        ),
        dtype=rdtypestr,
    )  # (3, 3)

    def proj_measure(pauli_string: Tensor, st: Tensor) -> Tensor:
        c_ = Circuit(nq, inputs=psi)
        for i in range(nq):
            c_.r(  # type: ignore
                i,
                theta=tc.backend.gather1d(
                    tc.backend.gather1d(angles, tc.backend.gather1d(pauli_string, i)), 0
                ),
                alpha=tc.backend.gather1d(
                    tc.backend.gather1d(angles, tc.backend.gather1d(pauli_string, i)), 1
                ),
                phi=tc.backend.gather1d(
                    tc.backend.gather1d(angles, tc.backend.gather1d(pauli_string, i)), 2
                ),
            )
        return c_.sample(batch=repeat, format="sample_bin", allow_state=True, status=st)

    vpm = tc.backend.vmap(proj_measure, vectorized_argnums=(0, 1))
    snapshots = vpm(pauli_strings, status)  # (ns, repeat, nq)
    if measurement_only:
        return snapshots if sub is None else slice_sub(snapshots, sub)
    else:
        return local_snapshot_states(snapshots, pauli_strings + 1, sub)


def local_snapshot_states(
    snapshots: Tensor, pauli_strings: Tensor, sub: Optional[Sequence[int]] = None
) -> Tensor:
    r"""To generate the local snapshots states from snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = (ns, nq) or (ns, repeat, nq)
    :type: Tensor
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return lss_states: shape = (ns, repeat, nq, 2, 2)
    :rtype: Tensor
    """
    pauli_strings = tc.backend.cast(pauli_strings, dtype="int32") - 1
    if len(pauli_strings.shape) < len(snapshots.shape):
        pauli_strings = tc.backend.tile(
            pauli_strings[:, None, :], (1, snapshots.shape[1], 1)
        )  # (ns, repeat, nq)

    X_dm = tc.backend.cast(
        tc.backend.convert_to_tensor([[[1, 1], [1, 1]], [[1, -1], [-1, 1]]]) / 2,
        dtype=dtypestr,
    )
    Y_dm = tc.backend.cast(
        tc.backend.convert_to_tensor(
            np.array([[[1, -1j], [1j, 1]], [[1, 1j], [-1j, 1]]]) / 2
        ),
        dtype=dtypestr,
    )
    Z_dm = tc.backend.cast(
        tc.backend.convert_to_tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]), dtype=dtypestr
    )
    pauli_dm = tc.backend.stack((X_dm, Y_dm, Z_dm), axis=0)  # (3, 2, 2, 2)

    def dm(p: Tensor, s: Tensor) -> Tensor:
        return tc.backend.gather1d(tc.backend.gather1d(pauli_dm, p), s)

    v = tc.backend.vmap(dm, vectorized_argnums=(0, 1))
    vv = tc.backend.vmap(v, vectorized_argnums=(0, 1))
    vvv = tc.backend.vmap(vv, vectorized_argnums=(0, 1))

    lss_states = vvv(pauli_strings, snapshots)
    if sub is not None:
        lss_states = slice_sub(lss_states, sub)
    return 3 * lss_states - tc.backend.eye(2)[None, None, None, :, :]



def global_shadow_state(
    snapshots: Tensor,
    pauli_strings: Optional[Tensor] = None,
    sub: Optional[Sequence[int]] = None,
) -> Tensor:
    r"""To generate the global shadow state from local snapshot states or snapshots and pauli strings

    :param snapshots: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param pauli_strings: shape = None or (ns, nq) or (ns, repeat, nq)
    :type: Optional[Tensor]
    :param sub: qubit indices of subsystem
    :type: Optional[Sequence[int]]

    :return gsdw_state: shape = (2 ** nq, 2 ** nq)
    :rtype: Tensor
    """
    if pauli_strings is not None:
        if len(snapshots.shape) != 3:
            raise ValueError(
                f"snapshots should be 3-d if pauli_strings is not None, got {len(snapshots.shape)}-d instead."
            )
        lss_states = local_snapshot_states(
            snapshots, pauli_strings, sub
        )  # (ns, repeat, nq_sub, 2, 2)
    else:
        if sub is not None:
            lss_states = slice_sub(snapshots, sub)
        else:
            lss_states = snapshots  # (ns, repeat, nq, 2, 2)

    nq = lss_states.shape[2]

    def tensor_prod(dms: Tensor) -> Tensor:
        res = tc.backend.gather1d(dms, 0)
        for i in range(1, nq):
            res = tc.backend.kron(res, tc.backend.gather1d(dms, i))
        return res

    v = tc.backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = tc.backend.vmap(v, vectorized_argnums=0)
    gss_states = vv(lss_states)
    return tc.backend.mean(gss_states, axis=(0, 1))


def slice_sub(entirety: Tensor, sub: Sequence[int]) -> Tensor:
    r"""To slice off the subsystem

    :param entirety: shape = (ns, repeat, nq, 2, 2) or (ns, repeat, nq)
    :type: Tensor
    :param sub: qubit indices of subsystem
    :type: Sequence[int]

    :return subsystem: shape = (ns, repeat, nq_sub, 2, 2)
    :rtype: Tensor
    """
    if len(entirety.shape) < 3:
        entirety = entirety[:, None, :]

    def slc(x: Tensor, idx: Tensor) -> Tensor:
        return tc.backend.gather1d(x, idx)

    v = tc.backend.vmap(slc, vectorized_argnums=(1,))
    vv = tc.backend.vmap(v, vectorized_argnums=(0,))
    vvv = tc.backend.vmap(vv, vectorized_argnums=(0,))
    return vvv(entirety, tc.backend.convert_to_tensor(sub))


@partial(tc.backend.jit, static_argnums=(3,))
def shadow_ss_dm(rho, pauli_strings, status, measurement_only=False):
    return shadow_snapshots_dm(
        rho, pauli_strings, status, measurement_only=measurement_only
    )


@partial(tc.backend.jit, static_argnums=(3,))
def shadow_ss(psi, pauli_strings, status, measurement_only=False):
    return shadow_snapshots(
        psi, pauli_strings, status, measurement_only=measurement_only
    )

def reconstructed_shadow_state(snapshots_states):
    return global_shadow_state(snapshots=snapshots_states)

gssjit = tc.backend.jit(reconstructed_shadow_state)


def get_shadow_rho_noiseless(psi, n, r, shots):
    nps = shots

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(nps, n)))

    status = tc.backend.convert_to_tensor(np.random.rand(nps, r))

    ss_states = shadow_ss(psi, pauli_strings, status)

    return gssjit(ss_states)



def get_shadow_rho_noisy(rho, n, r, shots):
    nps = shots

    pauli_strings = tc.backend.convert_to_tensor(np.random.randint(1, 4, size=(nps, n)))

    status = tc.backend.convert_to_tensor(np.random.rand(nps, r))

    ss_states = shadow_ss_dm(rho, pauli_strings, status)

    return gssjit(ss_states)


def get_shadow_rho_adaptive_noisy(rho, n, r, shots, safety_factor):
    max_batch = estimate_max_batch_size(n, dtype='complex128', safety_factor=safety_factor)
    n_batches = (shots + max_batch - 1) // max_batch

    rho_final = 0
    for i in range(n_batches):
        batch_shots = min(max_batch, shots - i * max_batch)

        rho_intermediate = get_shadow_rho_noisy(rho, n, r, batch_shots)

        jax.block_until_ready(rho_intermediate)  # ensure memory is flushed

        rho_final += rho_intermediate

        del rho_intermediate

    return rho_final / n_batches


def global_shadow_state1(
    snapshots: Tensor,
) -> Tensor:
    nq = snapshots.shape[2]

    @jax.jit
    def tensor_prod(dms: Tensor) -> Tensor:
        res = dms[0]
        for i in range(1, nq):
            res = jnp.kron(res, dms[i])
        return res
        
    v = tc.backend.vmap(tensor_prod, vectorized_argnums=0)
    vv = tc.backend.vmap(v, vectorized_argnums=0)
    gss_states = vv(snapshots)
    return tc.backend.mean(gss_states, axis=(0, 1))

gssjit1 = tc.backend.jit(global_shadow_state1)

def get_shadow_rho_noiseless_pure(psi: Tensor, key: Tensor, n: int, r: int, shots: int) -> Tensor:
    pauli_key, status_key = jax.random.split(key)
    pauli_strings = jax.random.randint(pauli_key, shape=(shots, n), minval=1, maxval=4)
    status = jax.random.uniform(status_key, shape=(shots, r))
    lss_states = shadow_ss(psi, pauli_strings, status)
    return gssjit1(lss_states)

# def get_shadow_XZ_rho_noiseless_pure(psi: Tensor, pauli_strings: Tensor, n: int, r: int, shots: int) -> Tensor:
#     status = jax.random.uniform(status_key, shape=(shots, r))
#     lss_states = shadow_ss(psi, pauli_strings, status)
#     return gssjit1(lss_states)
# This JIT-ed function is our core engine. It's efficient for a batch of `nf`.
# We will call it repeatedly on small batches.
jit_vectorized_shadow_computation = tc.backend.jit(
    tc.backend.vmap(get_shadow_rho_noiseless_pure, vectorized_argnums=(0, 1)),
    static_argnums=(2, 3, 4))

@partial(tc.backend.jit, static_argnums=(3,))
def _shadow_bitstrings(psi: Tensor, pauli_strings: Tensor, status: Tensor, measurement_only: bool =True):
    """To generate the shadow snapshots from given pauli string observables on psi

    :param psi: shape = (2 ** nq,), where nq is the number of qubits
    :param pauli_strings: shape = (ns, nq), where ns is the number of pauli strings
    :param status: shape = None or (ns, repeat), where repeat is the times to measure on one pauli string
    :param measurement_only: return snapshots (True) or snapshot states (False), default=False

    :return snapshots: shape = (ns, repeat, nq) if measurement_only=True otherwise (ns, repeat, nq, 2, 2)
    """
    return shadows.shadow_snapshots(
        psi, pauli_strings, status, measurement_only=measurement_only
    ) # (ns, repeat, nq)

jit_vmap_vectorized_bitstrings_XZ_computation = tc.backend.jit(
    tc.backend.vmap(_shadow_bitstrings, vectorized_argnums=(0, 0)),
    static_argnums=(2, 3))