"""The SCF checkpoint file, read by a reader that shares no code with its writer.

Numpy reads the frozen v1 file with pickle off, and this file recomputes the
checksum and the array values from their documented definitions. Rust holds the
current writer to this file's structure
(`a_fresh_file_has_the_frozen_files_member_structure` in
crates/ferric-scf/src/checkpoint/tests.rs). No ferric import: the point is that
none is needed.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parents[3] / "testdata/checkpoint/synthetic_rohf_v1.npz"
NAO, NMO = 7, 6

# name -> (tag, shape, Fortran order?), as built by `orbitals`/`final_state`.
MATRICES = {
    "density_total": (1.0, (NAO, NAO), False),
    "density_alpha": (2.0, (NAO, NAO), True),
    "density_beta": (3.0, (NAO, NAO), False),
    "mos_alpha": (4.0, (NAO, NMO), True),
    "mos_beta": (5.0, (NAO, NMO), True),
    "fock_alpha": (8.0, (NAO, NAO), False),
    "fock_beta": (9.0, (NAO, NAO), True),
    "rohf_spin_fock_alpha": (10.0, (NAO, NAO), False),
    "rohf_spin_fock_beta": (11.0, (NAO, NAO), True),
}
VECTORS = {"eps_alpha": 6.0, "eps_beta": 7.0}


def fnv1a64(state, data):
    for byte in data:
        state = ((state ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return state


def le64(value):
    return struct.pack("<Q", value)


def documented_checksum(npz):
    state = 0xCBF29CE484222325
    meta = npz["meta"].tobytes()
    state = fnv1a64(state, b"ferric-scf-checkpoint\0")
    state = fnv1a64(state, le64(int(npz["format_version"][0])))
    state = fnv1a64(state, le64(len(meta)))
    state = fnv1a64(state, meta)
    names = sorted(
        (n for n in npz.files if n not in ("meta", "format_version", "checksum")),
        key=lambda n: n.encode(),
    )
    for name in names:
        array = npz[name]
        state = fnv1a64(state, le64(len(name)))
        state = fnv1a64(state, name.encode())
        state = fnv1a64(state, le64(array.ndim))
        for dim in array.shape:
            state = fnv1a64(state, le64(dim))
        fortran = array.ndim > 1 and np.isfortran(array)
        state = fnv1a64(state, bytes([1 if fortran else 0]))
        order = "F" if fortran else "C"
        state = fnv1a64(state, array.astype("<f8").tobytes(order=order))
    return state


@pytest.fixture(scope="module")
def npz():
    with np.load(GOLDEN, allow_pickle=False) as loaded:
        yield {name: loaded[name] for name in loaded.files} | {"files": loaded.files}


class Npz(dict):
    @property
    def files(self):
        return self["files"]


def test_members_and_dtypes(npz):
    names = set(npz["files"])
    assert names == {"meta", "format_version", "checksum"} | set(MATRICES) | set(
        VECTORS
    )
    assert npz["format_version"].dtype == np.uint64
    assert npz["format_version"].tolist() == [1]
    assert npz["checksum"].dtype == np.uint64
    assert npz["meta"].dtype == np.uint8
    for name in list(MATRICES) + list(VECTORS):
        assert npz[name].dtype == np.float64, name


def test_arrays_match_their_definition_bit_for_bit(npz):
    for name, (tag, shape, fortran) in MATRICES.items():
        array = npz[name]
        assert array.shape == shape, name
        assert bool(np.isfortran(array)) is fortran, name
        for i in range(shape[0]):
            for j in range(shape[1]):
                want = tag + math.sqrt(i + 1) / (j + 2)
                assert array[i, j].tobytes() == np.float64(want).tobytes(), (name, i, j)
    for name, tag in VECTORS.items():
        want = [tag - (k + 1) / 3 for k in range(NMO)]
        assert npz[name].tobytes() == np.array(want).tobytes(), name


def test_checksum_is_the_documented_one(npz):
    assert int(npz["checksum"][0]) == documented_checksum(Npz(npz))


def test_meta_is_plain_json_with_the_route_and_exit(npz):
    meta = json.loads(npz["meta"].tobytes().decode())
    assert meta["spin"] == "RestrictedOpen"
    state = meta["final_state"]
    assert state["exit"] == "Converged"
    assert state["df_jk"]["j_aux"] == "def2-universal-jkfit"
    assert state["df_jk"]["op"]["kind"] == "ErfcCoulomb"
    names = {f["name"] for f in meta["identity"]["fields"]}
    assert {"atoms", "basis[Z=1]", "basis[Z=8]", "xc"} <= names
    assert meta["identity"]["nao"] == NAO


def test_flipping_a_value_changes_the_documented_checksum(npz):
    """The reimplementation above has to be able to fail."""
    tampered = Npz(npz)
    array = tampered["density_total"].copy()
    array[0, 0] = np.nextafter(array[0, 0], 2 * array[0, 0])
    tampered["density_total"] = array
    assert documented_checksum(tampered) != int(npz["checksum"][0])
