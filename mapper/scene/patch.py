import cv2
import torch
import numpy as np
import triangle as tr   # pip install triangle
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


class Patch:
    """One keyframe's seeded geometry. Geometry only today."""
    def __init__(self, vertices, colors, faces, age):
        self.vertices = vertices    # (V,3) world xyz
        self.colors   = colors      # (V,3) 0..1
        self.faces    = faces       # (F,3) int index triplets
        self.age      = age         # keyframe index at seeding -> for age coloring

def seed_patch(kf, mode="cdt", **kw):
    if mode == "grid":
        return seed_patch_grid(kf, **kw)
    elif mode == "cdt":
        return seed_patch_cdt(kf, **kw)
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

def seed_patch_cdt(kf, max_corners=400, corner_quality=0.01, nms_radius=4,
                   k_harris=0.04, smooth_ksize=5, min_dist=6,
                   cov_max=1e4, debug=True):
    device = kf.depth.device
    H, W = kf.depth.shape
    depth = kf.depth

    res = _cov_loops(kf.cov, cov_max=cov_max)
    if res is None:
        return (None, None) if debug else None
    loops, is_hole, markers, region = res

    reg_t = torch.from_numpy(region).bool().to(device)
    dt_t = torch.from_numpy(
        cv2.distanceTransform(region, cv2.DIST_L2, 5)).to(device)

    # --- Harris corners, gated on clearance from every loop --------------
    d = depth.float().clone()
    d[~reg_t] = 0.0
    Ix, Iy = torch.zeros_like(d), torch.zeros_like(d)
    Ix[:, 1:-1] = (d[:, 2:] - d[:, :-2]) * 0.5
    Iy[1:-1, :] = (d[2:, :] - d[:-2, :]) * 0.5

    def box_blur(x, k):
        return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2)[0, 0]

    Sxx, Syy = box_blur(Ix * Ix, smooth_ksize), box_blur(Iy * Iy, smooth_ksize)
    Sxy = box_blur(Ix * Iy, smooth_ksize)
    R = (Sxx * Syy - Sxy * Sxy) - k_harris * (Sxx + Syy) ** 2
    R[~reg_t] = R.min()

    pooled = F.max_pool2d(R[None, None], 2 * nms_radius + 1,
                          stride=1, padding=nms_radius)[0, 0]
    is_peak = (R == pooled) & (R > corner_quality * R.max()) & (dt_t > min_dist)
    ys, xs = torch.nonzero(is_peak, as_tuple=True)
    if ys.numel() > max_corners:
        top = torch.topk(R[ys, xs], max_corners).indices
        ys, xs = ys[top], xs[top]
    corners = torch.stack([xs.float(), ys.float()], 1).cpu().numpy()

    # --- lattice fill (region already excludes holes) --------------------
    step = max(int(min_dist), 1)
    gy, gx = np.mgrid[step // 2:H:step, step // 2:W:step]
    gy, gx = gy.ravel(), gx.ravel()
    keep = cv2.distanceTransform(region, cv2.DIST_L2, 5)[gy, gx] > min_dist
    lattice = np.stack([gx[keep], gy[keep]], 1).astype(np.float64)

    seeds = [p for p in (corners, lattice) if len(p)]
    seeds = np.concatenate(seeds, 0) if seeds else np.empty((0, 2))
    if len(seeds):
        seeds = seeds[_thin(seeds, min_dist * 0.6)]

    # --- PSLG: loops only ------------------------------------------------
    verts, segs, off = [], [], 0
    for xy in loops:
        k = len(xy)
        idx = np.arange(off, off + k)
        verts.append(xy)
        segs.append(np.stack([idx, np.roll(idx, -1)], 1))
        off += k
    loop_v = np.concatenate(verts)
    loop_s = np.concatenate(segs)
    vertices = np.vstack([loop_v, seeds]) if len(seeds) else loop_v

    assert loop_s.max() < len(vertices)
    assert len(np.unique(vertices.round(6), axis=0)) == len(vertices)
    if markers is not None:
        assert not region[markers[:, 1].astype(int),
                          markers[:, 0].astype(int)].any()

    tri_in = {"vertices": np.ascontiguousarray(vertices, np.float64),
              "segments": np.ascontiguousarray(loop_s, np.int32)}
    if markers is not None:
        tri_in["holes"] = np.ascontiguousarray(markers, np.float64)
    out = tr.triangulate(tri_in, "pq20a%d" % (min_dist * min_dist * 2))

    verts2d = torch.from_numpy(out["vertices"]).float().to(device)
    faces = torch.from_numpy(out["triangles"].astype(np.int64)).to(device)

    # --- back-project; NO face filtering ---------------------------------
    u = verts2d[:, 0].round().long().clamp(0, W - 1)
    v = verts2d[:, 1].round().long().clamp(0, H - 1)
    z = depth[v, u]
    fx, fy, cx, cy = kf.K[0, 0], kf.K[1, 1], kf.K[0, 2], kf.K[1, 2]
    cam = torch.stack([(verts2d[:, 0] - cx) * z / fx,
                       (verts2d[:, 1] - cy) * z / fy,
                       z, torch.ones_like(z)], -1)
    world = torch.einsum("ij,nj->ni", kf.c2w, cam)[:, :3]

    patch = Patch(vertices=world, faces=faces,
                  colors=kf.rgb[v, u], age=kf.index)
    if not debug:
        return patch
    return patch, debug_render_blob(kf.cov > cov_max, region, loops, is_hole,
                                    markers, seeds,
                                    verts2d.cpu().numpy(), faces.cpu().numpy())

def debug_render_blob(invalid, region, loops, is_hole, markers, seeds,
                      verts2d=None, faces=None):
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    inv = to_np(invalid).astype(np.uint8)
    reg = to_np(region).astype(np.uint8)
    H, W = inv.shape

    c = np.zeros((H, W, 3), np.uint8)
    c[reg > 0] = (95, 95, 95)                    # meshed
    c[(inv > 0) & (reg == 0)] = (110, 0, 90)     # excluded, above threshold
    c[(inv > 0) & (reg > 0)] = (255, 0, 0)       # VIOLATION — must be empty

    if faces is not None and len(faces):
        for tri in to_np(verts2d)[to_np(faces)]:
            cv2.polylines(c, [np.round(tri).astype(np.int32)], True,
                          (160, 160, 160), 1, cv2.LINE_AA)
    for xy, h in zip(loops, is_hole):
        cv2.polylines(c, [np.round(xy).astype(np.int32)], True,
                      (255, 165, 0) if h else (0, 255, 255), 2, cv2.LINE_AA)
    if markers is not None:
        for x, y in markers:
            cv2.drawMarker(c, (int(x), int(y)), (255, 0, 0),
                           cv2.MARKER_TILTED_CROSS, 10, 2)
    for x, y in to_np(seeds):
        cv2.circle(c, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
    return torch.from_numpy(c).float().div(255.).permute(2, 0, 1)

def _thin(pts, r):
    """Greedy: keep a point only if it's > r from all previously kept ones."""
    keep = np.zeros(len(pts), bool)
    kept = []
    for i, p in enumerate(pts):
        if not kept or np.min(np.linalg.norm(np.array(kept) - p, axis=1)) > r:
            keep[i] = True
            kept.append(p)
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


def _cov_loops(cov, cov_max=1e5, margin=2, smooth=5, close_r=3,
               n_outer=256, n_hole=32, max_iter=5):
    """
    Returns (loops, is_hole, markers, region) where `region` is guaranteed to
    contain no pixel with cov > cov_max. Loops are the ONLY constraints.
    """
    inv0 = (cov > cov_max).detach().cpu().numpy().astype(np.uint8)
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