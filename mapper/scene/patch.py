import cv2
import torch
import numpy as np
import triangle as tr   # pip install triangle

class Patch:
    """One keyframe's seeded geometry. Geometry only today."""
    def __init__(self, vertices, colors, faces, age):
        self.vertices = vertices    # (V,3) world xyz
        self.colors   = colors      # (V,3) 0..1
        self.faces    = faces       # (F,3) int index triplets
        self.age      = age         # keyframe index at seeding -> for age coloring

def seed_patch(kf, mask, mode="cdt", **kw):
    if mode == "grid":
        return seed_patch_grid(kf, mask, **kw)
    elif mode == "cdt":
        return seed_patch_cdt(kf, mask, **kw)
    raise ValueError(mode)

def seed_patch_grid(kf, mask, step=16, max_depth_jump=0.1, max_cov=5000):
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

def seed_patch_cdt(kf, mask, step=16, max_cov=None, edge_grad_thresh=0.1,
                   contour_stride=3):
    device = kf.depth.device
    H, W = kf.depth.shape
    if mask.dtype != torch.bool:
        mask = mask.bool()

    depth = kf.depth
    valid = mask & (depth > 0)
    if max_cov is not None:
        thresh = torch.quantile(kf.cov[valid], max_cov) if max_cov < 1.0 else max_cov
        valid = valid & (kf.cov < thresh)

    # --- 1. detect depth discontinuities (relative gradient) ---
    dzdx = torch.zeros_like(depth); dzdy = torch.zeros_like(depth)
    dzdx[:, :-1] = (depth[:, 1:] - depth[:, :-1]).abs()
    dzdy[:-1, :] = (depth[1:, :] - depth[:-1, :]).abs()
    rel_grad = (dzdx + dzdy) / depth.clamp_min(1e-6)
    edges = (rel_grad > edge_grad_thresh) & valid

    # --- 2. build the PSLG: grid points (unconstrained) + contour segments ---
    ys = torch.arange(0, H, step, device=device)
    xs = torch.arange(0, W, step, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid_ok = valid[gy, gx] & ~edges[gy, gx]

    points, segments = build_pslg(depth, valid, edges, gy, gx, grid_ok,
                                   contour_stride=contour_stride)
    if points.shape[0] < 3:
        return None

    # --- 3. constrained Delaunay via triangle ---
    tri_in = {"vertices": points, "segments": segments}
    out = tr.triangulate(tri_in, "p")

    verts2d = torch.from_numpy(out["vertices"]).float().to(device)
    faces   = torch.from_numpy(out["triangles"].astype(np.int64)).to(device)


    # --- 4. back-project the 2D vertices to world space ---
    u = verts2d[:, 0].round().long().clamp(0, W - 1)
    v = verts2d[:, 1].round().long().clamp(0, H - 1)
    z = depth[v, u]
    fx, fy = kf.K[0, 0], kf.K[1, 1]
    cx, cy = kf.K[0, 2], kf.K[1, 2]
    X = (verts2d[:, 0] - cx) * z / fx
    Y = (verts2d[:, 1] - cy) * z / fy
    cam = torch.stack([X, Y, z, torch.ones_like(z)], -1)
    world = torch.einsum("ij,nj->ni", kf.c2w, cam)[:, :3]
    colors = kf.rgb[v, u]

    # --- 5. drop faces whose vertices span a discontinuity in 3D (safety) ---
    tri_z = z[faces]                                     # (F,3)
    span = tri_z.max(1).values - tri_z.min(1).values
    keep = span < (edge_grad_thresh * tri_z.mean(1))     # relative
    # also drop faces touching invalid depth
    keep &= (z[faces] > 0).all(1)
    faces = faces[keep]

    used = torch.unique(faces)
    remap = torch.full((world.shape[0],), -1, dtype=torch.long, device=device)
    remap[used] = torch.arange(used.numel(), device=device)

    return Patch(vertices=world[used], faces=remap[faces],
                 colors=colors[used], age=kf.index)


def build_pslg(depth, valid, edges, grid_gy, grid_gx, grid_ok, contour_stride=3):
    """
    Assemble the CDT input: grid points (unconstrained) + contour points
    (constrained). Returns (points Nx2, segments Mx2 int) with segments
    indexing into points.
    """
    pts = []
    segs = []

    # 1. grid points on flat, non-edge regions — no constraints
    gx = grid_gx[grid_ok].float().cpu().numpy()
    gy = grid_gy[grid_ok].float().cpu().numpy()
    grid_xy = np.stack([gx, gy], axis=1)
    pts.append(grid_xy)
    n = len(grid_xy)

    # 2. trace discontinuity contours on the edge MASK (not a point list)
    edge_mask = edges.cpu().numpy().astype(np.uint8)          # HxW binary
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        poly = contour[:, 0, :]            # (K,2) as (x,y), already ORDERED
        poly = poly[::contour_stride]       # subsample along the contour
        if len(poly) < 2:
            continue
        pts.append(poly.astype(np.float32))
        # segments chain THIS contour's points, offset by current vertex count
        idx = np.arange(n, n + len(poly))
        seg = np.stack([idx[:-1], idx[1:]], axis=1)
        segs.append(seg)
        n += len(poly)

    points = np.concatenate(pts, axis=0)
    segments = np.concatenate(segs, axis=0) if segs else np.zeros((0,2), np.int64)
    return points, segments
