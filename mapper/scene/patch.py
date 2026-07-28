import cv2
import torch
import numpy as np
import shapely
import triangle as tr   # pip install triangle
import torch.nn.functional as F
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.timing import tic, count

# Snap-rounding grid for the seam boolean, in pixels. Two jobs: it makes the
# overlay deterministic, and it guarantees no sliver thinner than GRID can reach
# Triangle. A hairline sliver is what makes the quality mesher explode -- at a
# 0.001 px gap `pq20a1152` emits 2.8M triangles for two rings.
GRID = 1e-3
# Match radius when recovering weld ids after the boolean. Must exceed GRID,
# since set_precision moves surviving vertices by up to half a grid cell.
ID_TOL = 5e-3
# Cap on output triangles, as a multiple of the expected count, with an absolute
# floor. The tripwire for the sliver blowup above.
TRI_BLOWUP_FACTOR = 50
TRI_BLOWUP_FLOOR = 5000

# Occlusion: how much further away the map has to be, as a fraction of the
# measured depth, before its area is re-opened for seeding. Relative because
# triangulation error grows with depth. Raise it to make the check coarser --
# every misfire seeds a second surface layer over ground the map already has.
OCCLUSION_REL = 0.10
# Absolute backstop in record units, so the budget does not vanish near zero.
OCCLUSION_FLOOR = 0.02
# Smallest re-opened blob worth believing, in pixels. Speckle at this scale is
# depth noise along discontinuities, not a genuinely occluded surface.
OCCLUSION_MIN_AREA = 200


class Patch:
    """One keyframe's seeded geometry. Geometry only today."""
    def __init__(self, vertices, colors, faces, age):
        self.vertices = vertices    # (V,3) world xyz
        self.colors   = colors      # (V,3) 0..1
        self.faces    = faces       # (F,3) int index triplets
        self.age      = age         # keyframe index at seeding -> for age coloring
        self._bloops  = None        # cached oriented boundary rings, see boundary_loops

    def boundary_loops(self):
        """Oriented closed rings of this patch's boundary, as vertex indices."""
        if self._bloops is None:
            self._bloops = boundary_loops(self.faces.detach().cpu().numpy(),
                                          len(self.vertices))
        return self._bloops


def seed_patch(kf, model, mode="cdt", **kw):
    if mode == "grid":
        return seed_patch_grid(kf, model, **kw)
    elif mode == "cdt":
        return seed_patch_cdt(kf, model, **kw)
    raise ValueError(mode)

def seed_patch_grid(kf, mask, step=64, max_depth_jump=0.1, max_cov=5000):
    """
    Triangulate a coarse regular grid over the region of `kf` selected by
    `mask` (novelty mask: valid depth AND not-yet-covered), back-project
    to world space using kf.depth + kf.K + kf.c2w, and sample per-vertex
    color from kf.rgb. Appends the resulting Patch to self.patches.

    Returns the Patch, or None if nothing survives filtering.
    """
    device = kf.depth.device
    H, W = kf.depth.shape

    if mask.dtype != torch.bool:
        mask = mask.bool()

    fx, fy = kf.K[0, 0], kf.K[1, 1]
    cx, cy = kf.K[0, 2], kf.K[1, 2]

    ys = torch.arange(0, H, step, device=device)
    xs = torch.arange(0, W, step, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # (rows, cols)
    rows, cols = gy.shape

    z = kf.depth[gy, gx]
    valid = mask[gy, gx] & (z > 0)

    if max_cov is not None:
        valid &= kf.cov[gy, gx] < max_cov

    # back-project to camera space, then to world space
    X = (gx - cx) * z / fx
    Y = (gy - cy) * z / fy
    cam_pts = torch.stack([X, Y, z, torch.ones_like(z)], dim=-1)  # (rows, cols, 4)
    world_pts = torch.einsum("ij,rcj->rci", kf.c2w, cam_pts)[..., :3]

    vertices = world_pts.reshape(-1, 3)
    colors = kf.rgb[gy, gx].reshape(-1, 3)

    idx = torch.arange(rows * cols, device=device).reshape(rows, cols)

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if not (valid[r, c] and valid[r, c + 1] and valid[r + 1, c] and valid[r + 1, c + 1]):
                continue
            if max_depth_jump is not None:
                quad_z = torch.stack([z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]])
                if (quad_z.max() - quad_z.min()) > max_depth_jump:
                    continue

            v00, v01 = idx[r, c], idx[r, c + 1]
            v10, v11 = idx[r + 1, c], idx[r + 1, c + 1]

            d1 = abs(z[r, c] - z[r + 1, c + 1])
            d2 = abs(z[r, c + 1] - z[r + 1, c])
            if d1 <= d2:
                faces.append((v00, v10, v11)); faces.append((v00, v11, v01))
            else:
                faces.append((v00, v10, v01)); faces.append((v01, v10, v11))

    if not faces:
        return None

    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)

    # drop unreferenced grid vertices (masked-out / filtered by covariance or depth jump)
    used = torch.unique(faces_t)
    remap = torch.full((rows * cols,), -1, dtype=torch.int64, device=device)
    remap[used] = torch.arange(len(used), device=device)

    patch = Patch(
        vertices=vertices[used],
        faces=remap[faces_t],
        colors=colors[used],
        age=kf.index
    )
    return patch


def seed_patch_cdt(kf, model=None, model_loops=None, rendered_depth=None,
                   invalid_mask=None, occlusion_rel=OCCLUSION_REL,
                   occlusion_min_area=OCCLUSION_MIN_AREA,
                   max_corners=400, corner_quality=0.01,
                   nms_radius=3, k_harris=0.04, smooth_ksize=5,
                   min_dist=10, cov_max=1e5, debug=True):
    """
    Triangulate the region that is (inside the valid-data contour) AND (not
    already covered by `model` as seen from this camera).

    `invalid_mask` decides what "valid data" means and defaults to
    `kf.cov > cov_max`. With ground-truth depth the SLAM covariance describes a
    depth map that is no longer being used, so the caller passes the depth
    validity mask instead and coverage becomes the only other constraint.

    The seam comes from exact polygon arithmetic -- one boolean,
    `cov_region - union(projected model outlines)` -- so the new patch's edge
    matches the existing geometry vertex-for-vertex. Every inside/outside
    *classification* goes through a mask rasterised from those same polygons,
    which is where the old code spent its time doing exact geometry it did not
    need.

    `model_loops` are the model's cached oriented boundary rings (vertex index
    arrays); derived from the face array if not supplied. `rendered_depth` is
    the model's depth as seen from this camera; when given, model regions that
    sit *behind* what this keyframe measures are treated as not-yet-covered.

    Returns (patch, tile, ids); `tile` is None unless `debug`, and all three are
    None if nothing could be seeded. `ids[k] >= 0` means vertex k is model
    vertex ids[k] -- that is the welding table.
    """
    device = kf.depth.device
    H, W = kf.depth.shape
    depth = kf.depth

    # --- 1. region loops (vector) ----------------------------------------
    # `invalid_mask` lets the caller choose what bounds the region. With
    # ground-truth depth the covariance is meaningless, so the caller passes
    # the depth-validity mask instead and coverage becomes the only other
    # constraint.
    if invalid_mask is None:
        invalid_mask = kf.cov > cov_max
    with tic("region_loops"):
        res = _region_loops(invalid_mask)
    if res is None:
        return None, None, None
    loops, is_hole, _markers, region = res      # markers unused now

    # --- 2. model boundary rings, projected -------------------------------
    with tic("project_loops"):
        rings_uv, rings_id = [], []
        if model is not None and len(model.vertices):
            if model_loops is None:
                model_loops = boundary_loops(
                    model._triangle_indices.detach().cpu().numpy(),
                    len(model.vertices))
            rings_uv, rings_id = project_loops(model_loops, model.vertices, kf,
                                               shape=(H, W))
    count("n_model_rings", len(rings_uv))

    # --- 3. one boolean: cov region minus what the model already covers ----
    with tic("seam_boolean"):
        occluders = None
        if rendered_depth is not None and len(rings_uv):
            occluders = _revealed_polygons(rendered_depth, depth, (H, W),
                                           rel=occlusion_rel,
                                           min_area=occlusion_min_area)
        rings, ring_ids, holes_xy, stats = seam_boolean(
            loops, is_hole, rings_uv, rings_id, revealed=occluders)
    for k, v in stats.items():
        count(k, v)
    if not rings:
        return None, None, None

    # --- 4. PSLG: densify each ring; no crossings are possible ------------
    with tic("pslg"):
        # the map's real boundary edges, so densification can leave them intact
        model_edges = None
        if model_loops:
            model_edges = np.unique(np.concatenate(
                [edge_key(r, np.roll(r, -1)) for r in model_loops]))

        P, S, ids, off = [], [], [], 0
        for xy, vid in zip(rings, ring_ids):
            xy2, id2 = _densify_ring(xy, vid, min_dist * 1.5, model_edges)
            k = len(xy2)
            idx = np.arange(off, off + k)
            P.append(xy2)
            ids.append(id2)
            S.append(np.stack([idx, np.roll(idx, -1)], 1))
            off += k
        P = np.concatenate(P).astype(np.float64)
        S = np.concatenate(S).astype(np.int64)
        ids = np.concatenate(ids).astype(np.int64)
        P, S, ids = _dedupe(P, S, ids)
    if len(S) < 3:
        return None, None, None

    # --- 5. masks: every classification below is an O(1) lookup ------------
    with tic("masks"):
        keep_mask = _mask_from_rings(rings, (H, W))
        # distance to the nearest non-kept pixel == clearance from the seam
        clearance = cv2.distanceTransform(keep_mask, cv2.DIST_L2, 3)

    def _ok(pts):
        x = np.round(pts[:, 0]).astype(np.int64).clip(0, W - 1)
        y = np.round(pts[:, 1]).astype(np.int64).clip(0, H - 1)
        return (keep_mask[y, x] > 0) & (clearance[y, x] > min_dist)

    # --- 6. interior seeds -----------------------------------------------
    with tic("seeds"):
        step = max(int(min_dist), 1)
        gy, gx = np.mgrid[step // 2:H:step, step // 2:W:step]
        cand = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
        cand = cand[_ok(cand)]

        corners = _harris_seeds(depth, region, _ok, max_corners, corner_quality,
                                nms_radius, k_harris, smooth_ksize, device)

        seeds = [p for p in (corners, cand) if len(p)]
        seeds = np.concatenate(seeds, 0) if seeds else np.empty((0, 2))
        if len(seeds):
            seeds = seeds[_thin(seeds, min_dist * 0.6)]
    count("n_seeds", len(seeds))

    # --- 7. triangulate the kept region only ------------------------------
    nP = len(P)
    vertices = np.vstack([P, seeds]) if len(seeds) else P
    tri_in = {"vertices": np.ascontiguousarray(vertices, np.float64),
              "segments": np.ascontiguousarray(S, np.int32)}
    if len(holes_xy):
        tri_in["holes"] = np.ascontiguousarray(holes_xy, np.float64)

    assert S.max() < len(vertices)

    max_area = min_dist ** 2 * 2
    with tic("triangulate"):
        out = tr.triangulate(tri_in, "pq20a%dY" % max_area)
    V = out["vertices"]
    F_all = out["triangles"].astype(np.int64)
    count("n_tri_out", len(F_all))
    # Triangle appends Steiner points, so the input prefix survives in order.
    # This is what keeps `ids` aligned; it is not documented, so assert it.
    assert np.allclose(V[:nP], P), "Triangle reordered the input vertices; ids misaligned"

    # The ratio alone false-positives on a tiny kept region, where `expected`
    # rounds to 1 and any legitimate patch trips it; the floor is what makes it
    # a blowup detector rather than a small-patch filter. Observed real blowups
    # were 51k and 2.8M triangles, so this sits well below them.
    expected = max(int(keep_mask.sum() / max_area), 1)
    if len(F_all) > max(TRI_BLOWUP_FACTOR * expected, TRI_BLOWUP_FLOOR):
        # A sliver survived the boolean. Bail loudly rather than march into the
        # rest of the pipeline with a multi-million-triangle mesh.
        print(f"[seed_patch] triangulation blew up: {len(F_all)} tris vs "
              f"~{expected} expected; {len(P)} PSLG pts, {len(rings)} rings. "
              f"Skipping keyframe {kf.index}.")
        return None, None, None

    ids_full = np.concatenate([ids, -np.ones(len(V) - nP, np.int64)])
    if len(F_all) == 0:
        return None, None, None

    # --- 8. compact ------------------------------------------------------
    used = np.unique(F_all)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    faces_np = remap[F_all]
    V = V[used]
    ids_full = ids_full[used]
    count("n_ids_ge0", int((ids_full >= 0).sum()))

    # --- 8b. seam vertices that land inside an existing model edge --------
    with tic("edge_splits"):
        splits, snaps = seam_edge_splits(V, ids_full, faces_np,
                                         rings_uv, rings_id, model_edges)
        for k, gid in snaps.items():          # close enough to weld outright
            ids_full[k] = gid

    # --- 9. back-project -------------------------------------------------
    with tic("backproject"):
        verts2d = torch.from_numpy(V).float().to(device)
        u = verts2d[:, 0].round().long().clamp(0, W - 1)
        v = verts2d[:, 1].round().long().clamp(0, H - 1)
        z = depth[v, u]
        fx, fy, cx, cy = kf.K[0, 0], kf.K[1, 1], kf.K[0, 2], kf.K[1, 2]
        cam = torch.stack([(verts2d[:, 0] - cx) * z / fx,
                           (verts2d[:, 1] - cy) * z / fy,
                           z, torch.ones_like(z)], -1)
        world = torch.einsum("ij,nj->ni", kf.c2w, cam)[:, :3]

        # A weld vertex IS the model vertex. Re-measuring it from this
        # keyframe's depth puts it somewhere slightly different, which opens a
        # 3-D crack and -- because both copies then live in the model and
        # project a fraction of a pixel apart -- manufactures exactly the
        # hairline slivers that make the mesher explode next keyframe.
        m = ids_full >= 0
        if m.any() and model is not None:
            src = torch.from_numpy(ids_full[m]).to(device)
            world[m] = model.vertices.detach()[src].to(world.dtype)

        # A vertex that splits a model edge must sit *on* that edge, or the
        # split would drag the existing surface off to wherever this keyframe's
        # depth happened to land. Same reasoning as the exact-match override.
        if splits and model is not None:
            mv = model.vertices.detach()
            k = torch.tensor([s[0] for s in splits], dtype=torch.long, device=device)
            a = torch.tensor([s[1] for s in splits], dtype=torch.long, device=device)
            b = torch.tensor([s[2] for s in splits], dtype=torch.long, device=device)
            t = torch.tensor([s[3] for s in splits], dtype=world.dtype,
                             device=device)[:, None]
            world[k] = mv[a] * (1 - t) + mv[b] * t

    if not (z > 0).all():
        print(f"[seed_patch] {(z <= 0).sum().item()} vertices with zero depth")

    patch = Patch(vertices=world,
                  faces=torch.from_numpy(faces_np).to(device),
                  colors=kf.rgb[v, u], age=kf.index)
    # (patch vertex, model vertex a, model vertex b, t) -- the map's own edge
    # (a,b) has to be split at t when this patch is welded
    patch.edge_splits = splits

    if not debug:
        return patch, None, ids_full
    with tic("debug_render"):
        tile = debug_render_weld(invalid_mask, loops, is_hole, rings,
                                 V, faces_np, ids_full, seeds)
    return patch, tile, ids_full


def _thin(pts, r):
    """Keep a point only if it is > r from every previously kept one.

    Same semantics as the old O(n^2) version, but bucketed: with cells of side
    r, a cell holds at most a handful of mutually-separated points, so the
    neighbourhood scan is O(1) per point.
    """
    keep = np.zeros(len(pts), bool)
    cells = {}
    r2 = r * r
    inv = 1.0 / r
    for i, (x, y) in enumerate(pts):
        cx, cy = int(np.floor(x * inv)), int(np.floor(y * inv))
        hit = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in cells.get((cx + dx, cy + dy), ()):
                    if (px - x) ** 2 + (py - y) ** 2 <= r2:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            keep[i] = True
            cells.setdefault((cx, cy), []).append((x, y))
    return keep


def _rasterize(loops, is_hole, shape):
    """Region enclosed by the outer loop minus the hole loops."""
    reg = np.zeros(shape, np.uint8)
    for xy, h in zip(loops, is_hole):
        if not h:
            cv2.fillPoly(reg, [np.round(xy).astype(np.int32)], 1)
    for xy, h in zip(loops, is_hole):
        if h:
            cv2.fillPoly(reg, [np.round(xy).astype(np.int32)], 0)
    return reg


def _mask_from_rings(rings, shape):
    """Rasterise boolean-output rings. One call, even-odd fill handles holes."""
    reg = np.zeros(shape, np.uint8)
    if rings:
        cv2.fillPoly(reg, [np.round(r).astype(np.int32) for r in rings], 1)
    return reg


def _region_loops(invalid, margin=2, smooth=5, close_r=3,
                  n_outer=256, n_hole=32, max_iter=5):
    """Contour the largest region containing no `invalid` pixel.

    `invalid` is a bool mask -- covariance above threshold, or depth that is
    not valid. Returns (loops, is_hole, markers, region); `region` is
    guaranteed to contain no invalid pixel. Loops are the ONLY constraints.
    """
    inv0 = (invalid.detach().cpu().numpy() if torch.is_tensor(invalid)
            else np.asarray(invalid)).astype(np.uint8)
    shape = inv0.shape
    pad = margin + smooth

    for _ in range(max_iter):
        inv = inv0
        if close_r:
            inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE,
                                   np.ones((2 * close_r + 1,) * 2, np.uint8))
        inv = cv2.dilate(inv, np.ones((2 * pad + 1,) * 2, np.uint8))

        n, lab, st, _ = cv2.connectedComponentsWithStats(1 - inv, 8)
        if n < 2:
            return None
        core = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

        cnts, hier = cv2.findContours(core, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_NONE)
        if hier is None:
            pad += smooth
            continue
        hier = hier[0]

        loops, is_hole, markers, ok = [], [], [], True
        for i, cnt in enumerate(cnts):
            hole = hier[i][3] != -1
            xy = _resample_smooth(cnt, n_hole if hole else n_outer, smooth)
            if xy is None:                 # loop too small to represent
                ok = hole is False         # dropping a hole is not allowed
                if not ok:
                    break
                continue
            loops.append(xy)
            is_hole.append(hole)
        if not ok or not loops or not any(not h for h in is_hole):
            pad += smooth
            continue

        region = _rasterize(loops, is_hole, shape)
        if (region & inv0).any():          # smoothing leaked over bad pixels
            pad += smooth
            continue

        for xy, h in zip(loops, is_hole):  # markers only after verification
            if not h:
                continue
            tmp = np.zeros(shape, np.uint8)
            cv2.fillPoly(tmp, [np.round(xy).astype(np.int32)], 1)
            dt = cv2.distanceTransform(tmp, cv2.DIST_L2, 3)
            y, x = np.unravel_index(np.argmax(dt), dt.shape)
            markers.append((float(x), float(y)))

        return loops, is_hole, (np.array(markers, np.float64)
                                if markers else None), region
    return None


def _resample_smooth(cnt, n_pts, smooth):
    cnt = cnt.reshape(-1, 2).astype(np.float64)
    if len(cnt) < 8:
        return None
    seg = np.linalg.norm(np.diff(cnt, axis=0, append=cnt[:1]), axis=1)
    d = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0, d[-1], n_pts, endpoint=False)
    xy = np.stack([np.interp(t, d[:-1], cnt[:, 0]),
                   np.interp(t, d[:-1], cnt[:, 1])], 1)
    if smooth > 1:
        w = np.ones(smooth) / smooth
        p = np.r_[xy[-smooth:], xy, xy[:smooth]]
        xy = np.stack([np.convolve(p[:, i], w, "same")[smooth:-smooth]
                       for i in range(2)], 1)
    return xy


# --------------------------------------------------------------------------
# mesh boundary -> oriented rings
# --------------------------------------------------------------------------

def boundary_halfedges(faces):
    """Directed half-edges used by exactly one face, i.e. the mesh boundary."""
    f = np.asarray(faces)
    if len(f) == 0:
        return np.empty((0, 2), np.int64)
    de = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    _, inv, cnt = np.unique(np.sort(de, 1), axis=0,
                            return_inverse=True, return_counts=True)
    return de[cnt[inv.ravel()] == 1].astype(np.int64)


def edge_key(a, b):
    """Order-independent packed key for an undirected edge. Vectorised."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    return (np.minimum(a, b) << 32) | np.maximum(a, b)


def halfedge_maps(faces):
    """(he, twin, nxt) for a triangle soup.

    Half-edge `3f+i` runs from face f's vertex i to vertex (i+1)%3, so `nxt` is
    arithmetic. `twin[h]` is the opposing half-edge, or -1 on the boundary. An
    edge carrying three or more half-edges is non-manifold; those are left with
    twin -1 so they read as boundary rather than silently pairing off two of
    the three arbitrarily.
    """
    f = np.asarray(faces, np.int64)
    n = len(f)
    he = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1).reshape(-1, 2)
    idx = np.arange(3 * n)
    nxt = 3 * (idx // 3) + (idx + 1) % 3

    order = np.argsort(edge_key(he[:, 0], he[:, 1]), kind="stable")
    ks = edge_key(he[:, 0], he[:, 1])[order]
    twin = np.full(3 * n, -1, np.int64)
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    run = np.diff(np.r_[start, len(ks)])
    pair = start[run == 2]
    twin[order[pair]] = order[pair + 1]
    twin[order[pair + 1]] = order[pair]
    if (run > 2).any():
        count("n_nonmanifold_edges", int((run > 2).sum()))
    return he, twin, nxt


def boundary_loops(faces, n_vertices=None):
    """Chain the boundary half-edges into oriented closed rings.

    Returns a list of vertex-index arrays, each a ring with no repeated last
    point.

    The successor of boundary half-edge (u,v) is found by rotating around v
    through the face fan -- `nxt`, then `twin`, until a half-edge has no twin --
    not by picking any unused outgoing boundary half-edge. At a pinch vertex,
    one the boundary passes through twice, the arbitrary choice fuses two rings
    into a figure-eight, which then projects to a self-touching polygon and
    makes the seam boolean subtract the wrong region. Welding creates pinch
    vertices routinely: any new patch touching an old one at a single corner
    makes one.
    """
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return []
    he, twin, nxt = halfedge_maps(f)
    bnd = np.flatnonzero(twin < 0)
    if len(bnd) == 0:
        return []

    nv = n_vertices if n_vertices else int(he.max()) + 1
    count("n_pinch_vertices",
          int((np.bincount(he[bnd, 0], minlength=nv) > 1).sum()))

    seen = np.zeros(len(he), bool)
    loops = []
    for h0 in bnd:
        if seen[h0]:
            continue
        ring, h = [], int(h0)
        while not seen[h]:
            seen[h] = True
            ring.append(int(he[h, 0]))
            g = int(nxt[h])
            while twin[g] >= 0:              # rotate around the shared vertex
                g = int(nxt[twin[g]])
            h = g
        if len(ring) >= 3:
            loops.append(np.asarray(ring, np.int64))
    return loops


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def _clip_plane(C, ids, value, axis, keep_greater):
    """Sutherland-Hodgman against one axis-aligned half-space, vectorised.

    C is (L,k) of whatever is being carried (3-D camera points, or 2-D uv).
    Vertices introduced at a crossing get id -1: they are not model vertices.
    """
    v = C[:, axis]
    keep = (v > value) if keep_greater else (v < value)
    if keep.all():
        return C, ids
    if not keep.any():
        return C[:0], ids[:0]
    nxt = np.roll(np.arange(len(C)), -1)
    denom = C[nxt, axis] - v
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    t = (value - v) / denom
    X = C + t[:, None] * (C[nxt] - C)

    n = len(C)
    out = np.empty((2 * n, C.shape[1]), np.float64)
    oid = np.empty(2 * n, np.int64)
    sel = np.empty(2 * n, bool)
    out[0::2], out[1::2] = C, X
    oid[0::2], oid[1::2] = ids, -1
    sel[0::2], sel[1::2] = keep, keep != keep[nxt]
    return out[sel], oid[sel]


def project_loops(loops, verts_world, kf, near=1e-3, shape=None, snap=GRID):
    """Project 3-D boundary rings into kf's image.

    Returns (uv_rings, id_rings). One matmul for the whole model, a vectorised
    near-plane clip per ring, whole-ring frustum culling, and a clip to a
    generous box -- as z approaches the near plane uv blows up to +-1e6 px,
    which wrecks the relative epsilons of every downstream predicate.
    """
    w2c = torch.inverse(kf.c2w)
    cam = (w2c[:3, :3] @ verts_world.detach().T).T + w2c[:3, 3]
    cam = cam.detach().cpu().numpy().astype(np.float64)
    K = kf.K.detach().cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    H, W = shape if shape is not None else (int(2 * cy), int(2 * cx))
    box = (-W, -H, 2 * W, 2 * H)

    uv_rings, id_rings = [], []
    for ring in loops:
        C = cam[ring]
        if C[:, 2].max() <= near:                     # entirely behind
            continue
        ids = np.asarray(ring, np.int64)
        C, ids = _clip_plane(C, ids, near, 2, True)
        if len(C) < 3:
            continue
        uv = np.stack([C[:, 0] * fx / C[:, 2] + cx,
                       C[:, 1] * fy / C[:, 2] + cy], 1)
        if uv[:, 0].max() < box[0] or uv[:, 0].min() > box[2] or \
           uv[:, 1].max() < box[1] or uv[:, 1].min() > box[3]:
            continue                                   # misses the image
        for axis, val, greater in ((0, box[0], True), (0, box[2], False),
                                   (1, box[1], True), (1, box[3], False)):
            if len(uv) < 3:
                break
            uv, ids = _clip_plane(uv, ids, val, axis, greater)
        if len(uv) < 3:
            continue
        uv = np.round(uv / snap) * snap                # onto the boolean's grid
        keep = np.r_[True, (np.diff(uv, axis=0) != 0).any(1)]
        uv, ids = uv[keep], ids[keep]
        if len(uv) >= 3 and (uv[0] == uv[-1]).all():
            uv, ids = uv[:-1], ids[:-1]
        if len(uv) < 3:
            continue
        uv_rings.append(uv)
        id_rings.append(ids)
    return uv_rings, id_rings


# --------------------------------------------------------------------------
# the seam boolean
# --------------------------------------------------------------------------

def _as_polygons(g):
    if g is None or g.is_empty:
        return []
    t = g.geom_type
    if t == "Polygon":
        return [g]
    if t in ("MultiPolygon", "GeometryCollection"):
        return [p for gg in g.geoms for p in _as_polygons(gg)]
    return []


def _poly(xy):
    """Polygon from a ring, repairing self-intersections from silhouette folds."""
    if len(xy) < 3:
        return None
    p = Polygon(xy)
    if p.is_valid:
        return p
    ps = _as_polygons(shapely.make_valid(p))
    if not ps:
        return None
    return ps[0] if len(ps) == 1 else unary_union(ps)


def _id_lookup(query, key_xy, key_ids, tol=ID_TOL):
    """Nearest model vertex within `tol`, vectorised via a bucketed hash.

    Returns (ids, n_hit, n_collision). A collision is two *distinct* model
    vertices landing in the same bucket -- which is the norm, not an edge case,
    because two patches that already share a seam have two ids at one point.
    """
    if len(query) == 0 or len(key_xy) == 0:
        return np.full(len(query), -1, np.int64), 0, 0
    cell = tol
    kq = np.floor(key_xy / cell).astype(np.int64)
    key = (kq[:, 0] << 32) ^ (kq[:, 1] & 0xffffffff)
    o = np.argsort(key, kind="stable")
    key_s, k_s, id_s = key[o], key_xy[o], key_ids[o]

    dup = int((np.diff(key_s) == 0).sum())

    out = np.full(len(query), -1, np.int64)
    best = np.full(len(query), np.inf)
    qq = np.floor(query / cell).astype(np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            kk = ((qq[:, 0] + dx) << 32) ^ ((qq[:, 1] + dy) & 0xffffffff)
            pos = np.searchsorted(key_s, kk).clip(0, len(key_s) - 1)
            hit = key_s[pos] == kk
            d = np.linalg.norm(query - k_s[pos], axis=1)
            take = hit & (d < tol) & (d < best)
            out[take] = id_s[pos][take]
            best[take] = d[take]
    return out, int((out >= 0).sum()), dup


def _revealed_polygons(rendered_depth, kf_depth, shape, rel=OCCLUSION_REL,
                       min_area=OCCLUSION_MIN_AREA, open_r=2,
                       floor=OCCLUSION_FLOOR):
    """Where the model sits *behind* what this keyframe measures.

    Those pixels show a surface in front of the model that has never been
    seeded, so they must not count as covered. This is the only occlusion case
    that is unambiguous: the reverse (model in front of the measurement) means
    one of the two is wrong, and guessing there would cause more harm.

    The test is *relative*: triangulation depth error grows roughly with z^2, so
    a single absolute threshold is simultaneously too tight far away and too
    loose up close. It fires when the map is further than `kd * (1 + rel)`, with
    `floor` as an absolute backstop so the budget does not vanish as depth goes
    to zero. Both depths are in the record's own units -- the SLAM is monocular,
    so those are not metres; see utils/gt_depth.py.

    Loosening this is what stops the map doubling: every misfire re-opens
    already-covered area and seeds a second surface layer over it.

    Pixel-accurate by construction -- but the seam it produces separates new
    geometry from new geometry, with no model vertex to weld to, so exactness
    is not required here the way it is on the mesh boundary.
    """
    rd = rendered_depth.detach().cpu().numpy() if torch.is_tensor(rendered_depth) \
        else np.asarray(rendered_depth)
    kd = kf_depth.detach().cpu().numpy() if torch.is_tensor(kf_depth) \
        else np.asarray(kf_depth)
    if rd.shape != shape:
        return None
    gap = np.maximum(kd * rel, floor)
    mask = ((rd > 0) & (kd > 0) & (rd > kd + gap)).astype(np.uint8)
    count("n_revealed_px", int(mask.sum()))
    if not mask.any():
        return None
    if open_r:
        k = np.ones((2 * open_r + 1,) * 2, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    polys = []
    for c, h in zip(cnts, hier[0]):
        if h[3] != -1 or len(c) < 3:
            continue
        p = _poly(c.reshape(-1, 2).astype(np.float64))
        if p is not None and p.area >= min_area:
            polys.append(p)
    return unary_union(polys) if polys else None


def seam_boolean(cov_loops, cov_is_hole, model_uv, model_ids,
                 revealed=None, grid=GRID, tol=ID_TOL, min_ring_area=1e-3):
    """cov_region - union(projected model rings), as simple tagged rings.

    Returns (rings, ring_ids, holes_xy, stats). The rings are simple, disjoint
    and correctly nested by construction, so the PSLG built from them has no
    crossings and no T-junctions -- which is what lets the whole segment-soup
    cleanup stage disappear.
    """
    stats = {}
    outer = [p for p in (_poly(xy) for xy, h in zip(cov_loops, cov_is_hole)
                         if not h) if p is not None]
    if not outer:
        return [], [], np.empty((0, 2)), stats
    cov = unary_union(outer)
    holes = [p for p in (_poly(xy) for xy, h in zip(cov_loops, cov_is_hole)
                         if h) if p is not None]
    if holes:
        cov = cov.difference(unary_union(holes))

    keep = cov
    key_xy = np.empty((0, 2))
    key_ids = np.empty(0, np.int64)
    if model_uv:
        mdl_polys = [p for p in (_poly(xy) for xy in model_uv) if p is not None]
        if mdl_polys:
            mdl = unary_union(mdl_polys)
            if revealed is not None and not revealed.is_empty:
                # a model region the camera sees past is not covered
                mdl = mdl.difference(revealed)
            keep = shapely.set_precision(cov, grid).difference(
                shapely.set_precision(mdl, grid))
        key_xy = np.concatenate(model_uv) if model_uv else key_xy
        key_ids = np.concatenate(model_ids) if model_ids else key_ids

    keep = shapely.set_precision(keep, grid)
    if not keep.is_valid:
        keep = shapely.make_valid(keep)

    rings, ring_ids, holes_xy = [], [], []
    n_pts = 0
    for g in _as_polygons(keep):
        if g.area < min_ring_area:
            continue
        for i, r in enumerate([g.exterior, *g.interiors]):
            xy = np.asarray(r.coords)[:-1]          # drop the repeated close
            if len(xy) < 3 or abs(Polygon(xy).area) < min_ring_area:
                continue
            rings.append(xy)
            n_pts += len(xy)
            if i > 0:
                holes_xy.append(list(Polygon(xy).representative_point().coords)[0])

    if rings:
        q = np.concatenate(rings)
        ids, n_hit, n_dup = _id_lookup(q, key_xy, key_ids, tol)
        cuts = np.cumsum([len(r) for r in rings])[:-1]
        ring_ids = np.split(ids, cuts) if len(cuts) else [ids]
        stats.update(n_seam_pts=n_pts, n_ids_recovered=n_hit,
                     n_id_bucket_collisions=n_dup, n_model_pts=len(key_xy))
    stats["n_rings"] = len(rings)
    return rings, ring_ids, np.array(holes_xy, np.float64).reshape(-1, 2), stats


def seam_edge_splits(V2d, ids_full, faces_np, rings_uv, rings_id, model_edges,
                     tol=ID_TOL, t_eps=0.05):
    """Seam vertices that land inside an existing model boundary edge.

    Where the validity contour crosses the map's outline, the boolean puts a
    brand-new vertex (id -1) partway along a model edge. That edge belongs to a
    face nobody is splitting, so welding as-is leaves a T-junction that opens
    into a crack the moment the vertices move. These are the ones that have to
    be resolved by splitting the model's own face.

    Returns (splits, snaps): `splits` is a list of (patch_vertex, a, b, t) and
    `snaps` maps patch_vertex -> model vertex for crossings that landed close
    enough to an endpoint to be an ordinary weld instead.
    """
    splits, snaps = [], {}
    if not rings_uv or model_edges is None or not len(model_edges):
        return splits, snaps

    # candidate seam vertices: on this patch's own boundary, carrying no id
    cand = [k for r in boundary_loops(faces_np, len(V2d)) for k in r
            if ids_full[k] < 0]
    if not cand:
        return splits, snaps
    cand = np.asarray(sorted(set(cand)), np.int64)
    P = V2d[cand]

    # every genuine model boundary edge, as a projected 2-D segment
    A, B, IA, IB = [], [], [], []
    for uv, vid in zip(rings_uv, rings_id):
        a, b = vid, np.roll(vid, -1)
        ok = (a >= 0) & (b >= 0)
        if not ok.any():
            continue
        k = edge_key(np.where(ok, a, 0), np.where(ok, b, 0))
        pos = np.clip(np.searchsorted(model_edges, k), 0, len(model_edges) - 1)
        ok &= model_edges[pos] == k
        if not ok.any():
            continue
        A.append(uv[ok]); B.append(np.roll(uv, -1, 0)[ok])
        IA.append(a[ok]); IB.append(b[ok])
    if not A:
        return splits, snaps
    A = np.concatenate(A); B = np.concatenate(B)
    IA = np.concatenate(IA); IB = np.concatenate(IB)

    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    for i, p in zip(cand, P):
        t = (((p - A) * AB).sum(1) / L2).clip(0.0, 1.0)
        d = np.linalg.norm(p - (A + t[:, None] * AB), axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            continue
        L = float(np.sqrt(L2[j]))
        if t[j] * L < t_eps:                       # effectively at an endpoint
            snaps[int(i)] = int(IA[j])
        elif (1.0 - t[j]) * L < t_eps:
            snaps[int(i)] = int(IB[j])
        else:
            splits.append((int(i), int(IA[j]), int(IB[j]), float(t[j])))
    count("n_seam_edge_splits", len(splits))
    count("n_seam_endpoint_snaps", len(snaps))
    return splits, snaps


def _densify_ring(xy, ids, max_len, model_edges=None):
    """Split ring edges longer than max_len. Original vertices survive exactly.

    A ring edge that *is* an existing model boundary edge is never split. The
    inserted point would carry id -1, land in the interior of that edge, and be
    back-projected onto the measured surface rather than onto the edge itself --
    a T-junction against a model face that nobody splits, i.e. a crack. Nothing
    is lost by leaving it: `-Y` stops Triangle splitting the segment anyway.

    `model_edges` must be the sorted packed keys of the real boundary edges.
    Testing "both endpoints carry an id" is not enough: `project_loops` drops
    consecutive duplicates after grid-snapping, and the boolean can emit a chord
    joining two model vertices that share no edge -- suppressing those costs a
    long unsplittable triangle for no topological gain.
    """
    d = np.linalg.norm(np.diff(xy, axis=0, append=xy[:1]), axis=1)
    n = np.maximum(1, np.ceil(d / max_len).astype(np.int64))
    if model_edges is not None and len(model_edges) and len(ids):
        a, b = ids, np.roll(ids, -1)
        both = (a >= 0) & (b >= 0)
        k = edge_key(np.where(both, a, 0), np.where(both, b, 0))
        pos = np.clip(np.searchsorted(model_edges, k), 0, len(model_edges) - 1)
        on_model = both & (model_edges[pos] == k)
        count("n_model_edges_kept_whole", int(on_model.sum()))
        n = np.where(on_model, 1, n)
    idx = np.repeat(np.arange(len(xy)), n)
    k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = (k / np.repeat(n, n))[:, None]
    nxt = np.roll(xy, -1, 0)
    # t == 0 reproduces xy[idx] bit-exactly, so ids and positions both survive
    return xy[idx] * (1.0 - t) + nxt[idx] * t, np.where(k == 0, ids[idx], -1)


def _dedupe(P, S, ids, grid=1e-4):
    """Collapse coincident vertices, preferring model-tagged ones."""
    order = np.argsort(-(ids >= 0).astype(np.int8), kind="stable")
    Pp, idsp = P[order], ids[order]
    back = np.empty(len(P), np.int64)
    back[order] = np.arange(len(P))

    q = np.round(Pp / grid).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True,
                              return_inverse=True)
    P2, ids2 = Pp[first], idsp[first]

    S2 = inv.ravel()[back[S]]
    S2 = S2[S2[:, 0] != S2[:, 1]]
    S2 = np.unique(np.sort(S2, 1), axis=0)
    return P2, S2, ids2


def _harris_seeds(depth, region, ok_fn, max_corners, corner_quality,
                  nms_radius, k_harris, smooth_ksize, device):
    reg_t = torch.from_numpy(region).bool().to(device)
    d = depth.float().clone()
    d[~reg_t] = 0.0
    Ix, Iy = torch.zeros_like(d), torch.zeros_like(d)
    Ix[:, 1:-1] = (d[:, 2:] - d[:, :-2]) * 0.5
    Iy[1:-1, :] = (d[2:, :] - d[:-2, :]) * 0.5

    def bb(x, k):
        return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2)[0, 0]

    Sxx, Syy, Sxy = bb(Ix * Ix, smooth_ksize), bb(Iy * Iy, smooth_ksize), \
                    bb(Ix * Iy, smooth_ksize)
    R = (Sxx * Syy - Sxy * Sxy) - k_harris * (Sxx + Syy) ** 2
    R[~reg_t] = R.min()
    pooled = F.max_pool2d(R[None, None], 2 * nms_radius + 1,
                          stride=1, padding=nms_radius)[0, 0]
    peak = (R == pooled) & (R > corner_quality * R.max())
    ys, xs = torch.nonzero(peak, as_tuple=True)
    if ys.numel() > max_corners:
        top = torch.topk(R[ys, xs], max_corners).indices
        ys, xs = ys[top], xs[top]
    if ys.numel() == 0:
        return np.empty((0, 2))
    pts = torch.stack([xs.float(), ys.float()], 1).cpu().numpy().astype(np.float64)
    return pts[ok_fn(pts)]


def debug_render_weld(invalid, loops, is_hole, rings, V, faces, ids, seeds):
    inv = (invalid.detach().cpu().numpy() if torch.is_tensor(invalid)
           else np.asarray(invalid)).astype(np.uint8)
    H, W = inv.shape
    c = np.zeros((H, W, 3), np.uint8)
    c[inv > 0] = (70, 0, 60)

    if rings:                                          # kept region, blue
        cv2.fillPoly(c, [np.round(r).astype(np.int32) for r in rings],
                     (25, 35, 70))
    if len(faces):                                     # new mesh, grey edges
        cv2.polylines(c, list(np.round(V[faces]).astype(np.int32)), True,
                      (150, 150, 150), 1, cv2.LINE_AA)
    if rings:                                          # seam, white
        cv2.polylines(c, [np.round(r).astype(np.int32) for r in rings], True,
                      (255, 255, 255), 1, cv2.LINE_AA)
    for xy, h in zip(loops, is_hole):                  # covariance constraint
        cv2.polylines(c, [np.round(xy).astype(np.int32)], True,
                      (255, 165, 0) if h else (0, 255, 255), 2, cv2.LINE_AA)
    for p in V[ids >= 0]:                              # weldable, magenta
        cv2.circle(c, tuple(np.round(p).astype(int)), 3, (255, 0, 255), -1)
    for p in seeds:
        cv2.circle(c, tuple(np.round(p).astype(int)), 1, (0, 255, 0), -1)
    return torch.from_numpy(c).float().div(255.).permute(2, 0, 1)
