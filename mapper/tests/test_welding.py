"""Invariants for welding patches into the persistent mesh.

Run directly (`python mapper/tests/test_welding.py`) or under pytest. CPU only:
`GlobalMesh` is deliberately device-agnostic so the geometry can be tested
without a GPU. `TriangleModel` is not covered here -- it hardcodes cuda.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAPPER = os.path.dirname(_HERE)
sys.path.insert(0, _MAPPER)

# load scene.patch / scene.global_mesh without scene/__init__, which drags in
# the whole 3DGS dataset stack
_pkg = types.ModuleType("scene")
_pkg.__path__ = [os.path.join(_MAPPER, "scene")]
sys.modules.setdefault("scene", _pkg)
for _n in ("patch", "global_mesh"):
    if "scene." + _n not in sys.modules:
        _s = importlib.util.spec_from_file_location(
            "scene." + _n, os.path.join(_MAPPER, "scene", _n + ".py"))
        _m = importlib.util.module_from_spec(_s)
        sys.modules["scene." + _n] = _m
        _s.loader.exec_module(_m)
patch = sys.modules["scene.patch"]
gmesh = sys.modules["scene.global_mesh"]


class FakePatch:
    def __init__(self, verts, faces):
        self.vertices = torch.tensor(verts, dtype=torch.float32)
        self.faces = torch.tensor(faces, dtype=torch.int64)
        self.colors = torch.zeros(len(verts), 3)
        self.age = 0


def _square(x0):
    """Unit square at x0..x0+1, split into two triangles."""
    return FakePatch([[x0, 0, 0], [x0 + 1, 0, 0], [x0 + 1, 1, 0], [x0, 1, 0]],
                     [[0, 1, 2], [0, 2, 3]])


def test_pinch_vertex_does_not_fuse_rings():
    """Two triangles meeting at one vertex are two rings, not a figure-eight.

    The old walk picked an arbitrary unused outgoing boundary half-edge, which
    fuses them. A fused ring projects to a self-touching polygon and makes the
    seam boolean subtract the wrong region.
    """
    bowtie = np.array([[0, 1, 2], [2, 3, 4]])
    loops = patch.boundary_loops(bowtie, 5)
    assert len(loops) == 2, f"expected 2 rings, got {len(loops)}: {loops}"
    assert sorted(len(r) for r in loops) == [3, 3]


def test_closed_surface_has_no_boundary():
    tet = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    assert patch.boundary_loops(tet, 4) == []


def test_weld_reuses_vertices_and_closes_the_seam():
    m = gmesh.GlobalMesh(device="cpu")
    m.weld(_square(0), np.full(4, -1, np.int64), kf_index=0, seed_version=0)
    b_alone = m.boundary_edge_count()
    assert b_alone == 4

    # second square sharing the right-hand edge: its verts 0 and 3 ARE globals 1, 2
    m.weld(_square(1), np.array([1, -1, -1, 2], np.int64),
           kf_index=1, seed_version=m.version)

    assert len(m.vertices) == 6, "shared vertices were duplicated instead of welded"
    assert m.boundary_edge_count() == 6, "the seam did not close"
    assert len(m.boundary_loops()) == 1, "welded squares should have one outline"
    assert m.verify(full=True) == []


def test_weld_rejects_a_stale_seed_version():
    """ids name vertices in the mesh as it was at seed time."""
    m = gmesh.GlobalMesh(device="cpu")
    m.weld(_square(0), np.full(4, -1, np.int64), kf_index=0, seed_version=0)
    stale = m.version - 1
    try:
        m.weld(_square(1), np.full(4, -1, np.int64), kf_index=1, seed_version=stale)
    except RuntimeError as e:
        assert "version" in str(e)
        return
    raise AssertionError("a stale seed version was accepted")


def test_weld_rejects_out_of_range_ids():
    m = gmesh.GlobalMesh(device="cpu")
    try:
        m.weld(_square(0), np.array([99, -1, -1, -1], np.int64),
               kf_index=0, seed_version=0)
    except RuntimeError as e:
        assert "out of range" in str(e)
        return
    raise AssertionError("an id past the end of the mesh was accepted")


def test_edge_split_removes_the_t_junction():
    """A seam vertex inside a map edge must split that edge's face.

    Otherwise the map keeps one long edge with a vertex sitting in its middle,
    and the crack opens as soon as anything moves.
    """
    m = gmesh.GlobalMesh(device="cpu")
    m.weld(FakePatch([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
                     [[0, 1, 2], [0, 2, 3]]),
           np.full(4, -1, np.int64), kf_index=0, seed_version=0)
    n_faces = len(m.faces)

    # new patch whose vertex 0 is halfway along the map's boundary edge (1,2)
    p = FakePatch([[2, 1, 0], [3, 1, 0], [3, 0, 0]], [[0, 1, 2]])
    m.weld(p, np.full(3, -1, np.int64), kf_index=1, seed_version=m.version,
           splits=[(0, 1, 2, 0.5)])

    assert len(m.faces) == n_faces + 2, "the split should have added one face"
    assert m.n_splits_applied == 1 and m.n_splits_skipped == 0
    mid = m.vertices[4].tolist()
    assert mid == [2.0, 1.0, 0.0], f"split vertex at {mid}, wanted the midpoint"
    # the split vertex lies on the older surface, so it belongs to that patch
    assert m.vertex_patch[4] == 0
    assert m.verify(full=True) == []
    assert _t_junctions(m) == 0


def test_split_children_inherit_the_parent_patch():
    m = gmesh.GlobalMesh(device="cpu")
    m.weld(FakePatch([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
                     [[0, 1, 2], [0, 2, 3]]),
           np.full(4, -1, np.int64), kf_index=0, seed_version=0)
    p = FakePatch([[2, 0.5, 0], [2, 1.5, 0], [3, 1, 0]], [[0, 1, 2]])
    m.weld(p, np.full(3, -1, np.int64), kf_index=1, seed_version=m.version,
           splits=[(0, 1, 2, 0.25), (1, 1, 2, 0.75)])   # two splits, one edge

    assert m.n_splits_applied == 2, "multiple splits on one edge must all apply"
    old = m.faces_of(0)
    assert len(old) == 4, f"2 faces split once each way -> 4, got {len(old)}"
    assert set(m.face_patch[old].tolist()) == {0}
    assert m.verify(full=True) == []


def test_duplicate_and_degenerate_faces_are_refused():
    m = gmesh.GlobalMesh(device="cpu")
    m.weld(_square(0), np.full(4, -1, np.int64), kf_index=0, seed_version=0)
    n = len(m.faces)
    # a patch that welds all three vertices of an existing face, plus a face
    # collapsed to a line
    dup = FakePatch([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 1, 2], [0, 1, 1]])
    m.weld(dup, np.array([0, 1, 2], np.int64), kf_index=1, seed_version=m.version)
    assert len(m.faces) == n, "a duplicate or degenerate face got in"
    assert m.verify(full=True) == []


def test_per_patch_masks():
    m = gmesh.GlobalMesh(device="cpu")
    m.weld(_square(0), np.full(4, -1, np.int64), kf_index=0, seed_version=0)
    m.weld(_square(1), np.array([1, -1, -1, 2], np.int64),
           kf_index=5, seed_version=m.version)

    assert list(m.faces_of(0)) == [0, 1] and list(m.faces_of(1)) == [2, 3]
    assert int(m.vertex_mask(0).sum()) == 4
    assert int(m.vertex_mask(1).sum()) == 4
    both = m.vertex_mask(0) & m.vertex_mask(1)
    assert int(both.sum()) == 2, "the shared seam should appear in both masks"
    assert list(m.active_patches(5, window=1)) == [1]
    assert sorted(m.active_patches(5, window=10)) == [0, 1]
    # creator wins: the shared vertices stay with patch 0
    assert list(m.vertex_patch) == [0, 0, 0, 0, 1, 1]


def _t_junctions(m, tol=1e-6):
    v = m.vertices.numpy()
    he, twin, _ = patch.halfedge_maps(m.faces)
    bnd = he[twin < 0]
    if not len(bnd):
        return 0
    A, B = v[bnd[:, 0]], v[bnd[:, 1]]
    D = B - A
    L2 = np.maximum((D * D).sum(1), 1e-20)
    bad = 0
    for vi in np.unique(bnd):
        t = ((v[vi] - A) * D).sum(1) / L2
        d = np.linalg.norm(v[vi] - (A + t[:, None] * D), axis=1)
        bad += int(((d < tol) & (t > tol) & (t < 1 - tol) &
                    (bnd[:, 0] != vi) & (bnd[:, 1] != vi)).sum())
    return bad


def test_densify_leaves_model_edges_whole():
    """A ring edge that is a real map edge must not gain a mid-point.

    That point would carry no id, land inside the map's edge, and be
    back-projected onto the measured surface -- a crack. Being merely
    id-tagged at both ends is not enough: chords across the seam join two map
    vertices that share no edge, and splitting those is free.
    """
    xy = np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]])
    ids = np.array([7, 8, 9, -1], np.int64)
    real = np.unique(patch.edge_key(np.array([7]), np.array([8])))  # only 7-8

    out, oid = patch._densify_ring(xy, ids, max_len=10.0, model_edges=real)
    # the 7-8 edge survives as a single span; the 8-9 chord is densified
    kept = [i for i, o in enumerate(oid) if o == 7]
    assert len(kept) == 1
    nxt = oid[(kept[0] + 1) % len(oid)]
    assert nxt == 8, f"the real map edge 7-8 was split (next id {nxt})"
    assert (oid == 8).sum() == 1 and (oid == -1).sum() > 1, "chord was not densified"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
