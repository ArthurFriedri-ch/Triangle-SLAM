"""Invariants for the CDT patch seeder's geometry helpers.

Run directly (`python mapper/tests/test_patch_geometry.py`) or under pytest.
Deliberately imports patch.py by path: `scene/__init__.py` drags in the whole
3DGS dataset stack, which these tests do not need.
"""
import importlib.util
import os
import sys

import numpy as np
import triangle as tr

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAPPER = os.path.dirname(_HERE)
sys.path.insert(0, _MAPPER)

_spec = importlib.util.spec_from_file_location(
    "patch_under_test", os.path.join(_MAPPER, "scene", "patch.py"))
patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch)


def _ring(cx, cy, rx, ry, n, phase=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def test_no_sliver_blowup():
    """Nearly coincident model outlines must not explode the triangulation.

    This is the failure that killed the old seeder: two patches viewing the
    same surface project to curves a fraction of a pixel apart, and -q20 fills
    the hairline sliver between them with quality triangles. At a 0.001 px gap
    the old path emitted 2.8M triangles for two rings alone.
    """
    cov = [_ring(320, 240, 300, 225, 256)]
    counts = []
    for gap in (10.0, 1.0, 0.01, 0.001, 1e-5):
        model = [_ring(320, 240, 180 + gap * j, 140 + gap * j, 150)
                 for j in range(2)]
        ids = [np.arange(150) + 150 * j for j in range(2)]
        rings, ring_ids, holes, _ = patch.seam_boolean(cov, [False], model, ids)
        assert rings, f"gap={gap}: boolean produced nothing"

        P, S, off = [], [], 0
        for xy in rings:
            k = len(xy)
            idx = np.arange(off, off + k)
            P.append(xy)
            S.append(np.stack([idx, np.roll(idx, -1)], 1))
            off += k
        tri_in = {"vertices": np.concatenate(P), "segments": np.concatenate(S).astype(np.int32)}
        if len(holes):
            tri_in["holes"] = holes
        out = tr.triangulate(tri_in, "pq20a1152Y")
        counts.append(len(out["triangles"]))

    assert max(counts) < 3 * min(counts), \
        f"triangle count depends on sliver width: {counts}"
    assert max(counts) < 20000, f"triangulation blew up: {counts}"


def test_densify_preserves_vertices_exactly():
    """Densification must not move an original vertex by even one ulp.

    Original vertices carry weld ids; if densify perturbs them the id lookup
    silently starts missing.
    """
    xy = _ring(100, 100, 60, 40, 17) + 0.123456789
    ids = np.arange(17)
    out, out_ids = patch._densify_ring(xy, ids, max_len=5.0)
    keep = out_ids >= 0
    assert (out[keep] == xy[out_ids[keep]]).all(), "densify perturbed an original vertex"
    assert (out_ids[keep] == ids[out_ids[keep]]).all()
    assert len(out) > len(xy), "nothing was densified"


def test_id_lookup_recovers_exact_and_rejects_far():
    key = _ring(200, 200, 90, 70, 64)
    kid = np.arange(64)
    query = np.vstack([key, key + 5.0])          # half exact, half 5 px away
    ids, n_hit, _ = patch._id_lookup(query, key, kid, tol=patch.ID_TOL)
    assert (ids[:64] == kid).all(), "exact matches not recovered"
    assert (ids[64:] == -1).all(), "far points matched anyway"
    assert n_hit == 64


def test_boundary_loops_orders_a_ring():
    """A triangulated annulus has two boundary rings, each properly ordered."""
    outer = _ring(0, 0, 10, 10, 24)
    inner = _ring(0, 0, 4, 4, 12)
    P = np.vstack([outer, inner])
    S = np.vstack([
        np.stack([np.arange(24), np.roll(np.arange(24), -1)], 1),
        np.stack([np.arange(12), np.roll(np.arange(12), -1)], 1) + 24,
    ]).astype(np.int32)
    out = tr.triangulate({"vertices": P, "segments": S,
                          "holes": np.array([[0.0, 0.0]])}, "pa2")
    faces = out["triangles"].astype(np.int64)
    loops = patch.boundary_loops(faces, len(out["vertices"]))

    assert len(loops) == 2, f"expected 2 boundary rings, got {len(loops)}"
    V = out["vertices"]
    for ring in loops:
        step = np.linalg.norm(np.diff(V[ring], axis=0, append=V[ring][:1]), axis=1)
        # a correctly ordered ring only ever steps between neighbours; a fused
        # or shuffled ring shows a long jump across the annulus
        assert step.max() < 5.0, f"ring is not ordered, max step {step.max():.2f}"


def test_mask_from_rings_punches_holes():
    """Even-odd fill: an interior ring must come out as a hole, not a fill."""
    rings = [_ring(50, 50, 40, 40, 64), _ring(50, 50, 15, 15, 32)]
    m = patch._mask_from_rings(rings, (100, 100))
    assert m[50, 50] == 0, "interior ring was filled instead of punched out"
    assert m[50, 20] == 1, "annulus body is not filled"
    assert m[2, 2] == 0, "fill leaked outside the outer ring"


def test_seam_boolean_tags_surviving_model_vertices():
    """Model vertices that survive onto the seam must keep their ids."""
    cov = [_ring(320, 240, 280, 200, 128)]
    model = [_ring(200, 240, 120, 100, 96)]
    ids = [np.arange(96) + 7]
    rings, ring_ids, holes, stats = patch.seam_boolean(cov, [False], model, ids)
    assert rings
    all_ids = np.concatenate(ring_ids)
    assert (all_ids >= 0).sum() > 0, "no weld ids survived the boolean"
    # every recovered id must be a real model id, and sit where that model
    # vertex projected to
    q = np.concatenate(rings)
    m = all_ids >= 0
    assert set(np.unique(all_ids[m])).issubset(set(ids[0].tolist()))
    want = model[0][all_ids[m] - 7]
    assert np.abs(q[m] - want).max() < patch.ID_TOL, "tagged point is not at the model vertex"


def test_occlusion_reopens_what_the_camera_sees_past():
    """Model area the camera sees a nearer surface through must be re-seeded.

    The old coverage test was pure 2-D: anything inside a projected model
    triangle counted as covered regardless of z, so a surface in front of the
    map was never seeded.
    """
    import torch
    H, W = 480, 640
    rendered = np.full((H, W), 3.0, np.float32)      # the map, 3 m away
    measured = np.full((H, W), 3.0, np.float32)
    measured[150:330, 200:420] = 1.5                 # new surface, 1.5 m away
    revealed = patch._revealed_polygons(torch.from_numpy(rendered),
                                        torch.from_numpy(measured), (H, W))
    assert revealed is not None and revealed.area > 0.9 * 180 * 220

    cov = [_ring(320, 240, 300, 225, 256)]
    model = [_ring(320, 240, 280, 210, 160)]
    ids = [np.arange(160)]

    r_off, _, _, _ = patch.seam_boolean(cov, [False], model, ids, revealed=None)
    r_on, _, _, _ = patch.seam_boolean(cov, [False], model, ids, revealed=revealed)
    m_off = patch._mask_from_rings(r_off, (H, W))
    m_on = patch._mask_from_rings(r_on, (H, W))

    assert m_on.sum() > m_off.sum(), "occlusion did not re-open anything"
    box = m_on[150:330, 200:420]
    assert box.mean() > 0.9, f"only {box.mean():.0%} of the revealed box is kept"
    assert m_off[150:330, 200:420].mean() < 0.1, "box was already kept without occlusion"


def test_thin_respects_separation():
    rng = np.random.default_rng(0)
    pts = rng.random((2000, 2)) * 200
    keep = patch._thin(pts, 8.0)
    k = pts[keep]
    d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 8.0, f"kept points only {d.min():.2f} apart, wanted > 8"


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
